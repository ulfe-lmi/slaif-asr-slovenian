#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.batched_streaming import NvidiaSmiMonitor, file_sha256, parse_monitor_csv
from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, atomic_write_text, load_jsonl, sha256_file
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.acoustic_quality import read_audio_stats
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread
from slaif_asr.scale200_corpus import (
    TRAINING_VIEWS,
    build_exposure_schedule,
    build_prompt,
    fixed_text_path,
    generated_all_path,
    load_augmentation_config,
    load_existing_holdout,
    load_experiment_config,
    load_generation_config,
    multiplier_table,
    parse_generated_outputs,
    planned_prompts,
    protected_indexes,
    raw_generation_dir,
    rejected_path,
    repo_path,
    run_dir,
    select_fixed_rows,
    stable_sha256,
    text_certificate_path,
    validate_and_select_text,
    verify_directional_reference,
    whole_file_command_path,
    write_rejections,
    write_review_capsule,
    write_text_public_reports,
    expand_whole_file_decision,
    filter_records,
)
from slaif_asr.transcript_preserving_augmentation import (
    assignment_for,
    parameters_for_profile,
    render_augmented_file,
)


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs/experiments/gams1600_nine_voice_augmented_v1.json"
SUPERTONIC_SCALE200_CONFIG = REPO_ROOT / "configs/tts/supertonic3_sl_scale200_training_v1.json"
FAST_DIRECTIONAL_CONFIG = REPO_ROOT / "configs/experiments/fast_batched_directional_replay_v1.json"
ARM_NAME = "gams1600_nine_voice_augmented_joint_adapter_dim32"
AUGMENTATION_RENDER_VERSION = "scale200-transcript-preserving-render-v2"

_JOINT_PATH = Path(__file__).with_name("run_corpus_v2_joint_adapter_diagnostic.py")
_JOINT_SPEC = importlib.util.spec_from_file_location("_slaif_joint_runner_scale200", _JOINT_PATH)
if _JOINT_SPEC is None or _JOINT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import joint-adapter runner")
_JOINT = importlib.util.module_from_spec(_JOINT_SPEC)
_JOINT_SPEC.loader.exec_module(_JOINT)
finite_grad_norm = _JOINT.finite_grad_norm
mean_loss = _JOINT.mean_loss


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment = load_experiment_config(path)
    text = load_generation_config(repo_path(experiment["text_config"]))
    augmentation = load_augmentation_config(repo_path(experiment["augmentation_config"]))
    return experiment, text, augmentation


def stage_verify(config_path: Path) -> dict[str, Any]:
    experiment, text_config, augmentation = load_all_configs(config_path)
    directional = experiment["directional_reference_report"]
    reference = verify_directional_reference(directional["path"], directional["sha256"])
    holdout = load_existing_holdout(text_config)
    protected = [path for path in text_config["protected_indexes"] if repo_path(path).exists()]
    payload = {
        "status": "PASSED",
        "work_order_id": experiment["work_order_id"],
        "text_config": repo_path(experiment["text_config"]).as_posix(),
        "augmentation_config": repo_path(experiment["augmentation_config"]).as_posix(),
        "prompt_cells": len(text_config["prompt_cells"]),
        "requested_rows": text_config["target_requested_rows"],
        "final_rows": text_config["final_rows"],
        "existing_holdout_rows": len(holdout),
        "protected_indexes_present": len(protected),
        "directional_reference": reference,
        "multiplier_table": multiplier_table(),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _generate_with_scale200_fallback(
    *,
    tokenizer: Any,
    model: Any,
    prompts: list[dict[str, Any]],
    text_config: dict[str, Any],
    requested_batch_size: int,
) -> tuple[list[tuple[dict[str, Any], str]], int, list[str]]:
    from scripts.generate_gams_corpus_v2 import generate_batch

    batch_size = requested_batch_size
    fallback = int(text_config["generation"]["oom_fallback_batch_size"])
    results: list[tuple[dict[str, Any], str]] = []
    notes: list[str] = []
    index = 0
    while index < len(prompts):
        batch = prompts[index : index + batch_size]
        seed = sum(int(item["seed"]) for item in batch) % (2**31 - 1)
        try:
            outputs = generate_batch(
                tokenizer=tokenizer,
                model=model,
                prompts=[str(item["prompt"]) for item in batch],
                seed=seed,
                max_new_tokens=int(text_config["generation"]["max_new_tokens"]),
                temperature=float(text_config["generation"]["temperature"]),
                top_p=float(text_config["generation"]["top_p"]),
            )
        except RuntimeError as exc:
            text = str(exc).lower()
            if "out of memory" not in text and "cuda" not in text:
                raise
            if batch_size != int(text_config["generation"]["prompt_batch_size"]):
                raise
            notes.append(f"batch {batch_size} failed at prompt {index}; retrying with {fallback}: {type(exc).__name__}")
            batch_size = fallback
            continue
        if len(outputs) != len(batch):
            raise RuntimeError(f"output-count mismatch: {len(outputs)} outputs for {len(batch)} prompts")
        batch_results = list(zip(batch, outputs, strict=True))
        for prompt_meta, raw in batch_results:
            atomic_write_text(raw_output_path(text_config, prompt_meta), raw)
        results.extend(batch_results)
        index += len(batch)
        print(
            json.dumps(
                {
                    "event": "progress",
                    "stage": "generate-text",
                    "processed_prompts": index,
                    "total_prompts": len(prompts),
                    "batch_size": len(batch),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    return results, batch_size, notes


def raw_output_path(text_config: dict[str, Any], prompt_meta: dict[str, Any]) -> Path:
    return raw_generation_dir(text_config) / f"{prompt_meta['cell_id']}-attempt-{prompt_meta['attempt_index']:02d}.txt"


def raw_output_key(prompt_meta: dict[str, Any]) -> tuple[str, int]:
    return str(prompt_meta["cell_id"]), int(prompt_meta["attempt_index"])


def load_existing_raw_outputs(text_config: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    outputs: list[tuple[dict[str, Any], str]] = []
    for path in sorted(raw_generation_dir(text_config).glob("cell??-attempt-??.txt")):
        stem = path.stem
        cell_id, attempt = stem.split("-attempt-", 1)
        outputs.append(
            (
                {"cell_id": cell_id, "attempt_index": int(attempt)},
                path.read_text(encoding="utf-8"),
            )
        )
    return outputs


def retained_counts(
    records: list[dict[str, Any]],
    rejections: list[Any],
    *,
    text_config: dict[str, Any],
    protected: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
) -> dict[str, int]:
    retained, _rejected, _summary = filter_records(
        records,
        config=text_config,
        existing_rejections=rejections,
        protected=protected,
        holdout_rows=holdout,
    )
    counts: dict[str, int] = defaultdict(int)
    for row in retained:
        counts[str(row["generation"]["prompt_cell"])] += 1
    return counts


def stage_generate_text(config_path: Path) -> dict[str, Any]:
    from scripts.generate_gams_corpus_v2 import load_model
    from slaif_asr.corpus_v2_generation import GpuMonitor

    _experiment, text_config, _augmentation = load_all_configs(config_path)
    info = require_single_visible_cuda(allowed_name_fragments=("A100",))
    run_dir(text_config).mkdir(parents=True, exist_ok=True)
    raw_generation_dir(text_config).mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    all_rejections: list[Any] = []
    used_batch = int(text_config["generation"]["prompt_batch_size"])
    fallback_notes: list[str] = []
    protected = protected_indexes(text_config)
    holdout = load_existing_holdout(text_config)
    start = time.perf_counter()
    existing_outputs = load_existing_raw_outputs(text_config)
    existing_keys = {raw_output_key(meta) for meta, _raw in existing_outputs}
    if existing_outputs:
        parsed, parser_rejections = parse_generated_outputs(config=text_config, outputs=existing_outputs)
        all_records.extend(parsed)
        all_rejections.extend(parser_rejections)
    retained_by_cell = retained_counts(
        all_records,
        all_rejections,
        text_config=text_config,
        protected=protected,
        holdout=holdout,
    )
    tokenizer = None
    model = None
    with GpuMonitor(
        physical_selector=info.physical_selector,
        output_path=run_dir(text_config) / "gpu-monitor.local.csv",
        interval_seconds=float(text_config["generation"]["monitor_interval_seconds"]),
    ):
        diversity_max = int(text_config.get("diversity_retry", {}).get("maximum_retry_attempt", -1))
        maximum_attempts = max(
            max(int(cell["maximum_retries"]) for cell in text_config["prompt_cells"]),
            diversity_max,
        ) + 1
        for attempt_index in range(maximum_attempts):
            prompts = planned_prompts(text_config, retained_by_cell, attempt_index)
            prompts = [prompt for prompt in prompts if raw_output_key(prompt) not in existing_keys]
            if not prompts:
                continue
            if tokenizer is None or model is None:
                tokenizer, model = load_model(text_config)
            outputs, used_batch, notes = _generate_with_scale200_fallback(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                text_config=text_config,
                requested_batch_size=used_batch,
            )
            fallback_notes.extend(notes)
            for prompt_meta, _raw in outputs:
                existing_keys.add(raw_output_key(prompt_meta))
            parsed, parser_rejections = parse_generated_outputs(config=text_config, outputs=outputs)
            all_records.extend(parsed)
            all_rejections.extend(parser_rejections)
            retained_by_cell = retained_counts(
                all_records,
                all_rejections,
                text_config=text_config,
                protected=protected,
                holdout=holdout,
            )
            retained_count = sum(retained_by_cell.values())
            base_requirements_met = retained_count >= int(text_config["minimum_admissible_rows"]) and all(
                retained_by_cell.get(str(cell["cell_id"]), 0) >= int(text_config["final_rows_per_cell"])
                for cell in text_config["prompt_cells"]
            )
            diversity = text_config.get("diversity_retry", {})
            diversity_cells = {str(item) for item in diversity.get("cell_ids", [])}
            diversity_max = int(diversity.get("maximum_retry_attempt", -1))
            diversity_budget_complete = not diversity_cells or attempt_index >= diversity_max
            if base_requirements_met and diversity_budget_complete:
                break
    all_records = sorted(all_records, key=lambda row: str(row["candidate_id"]))
    atomic_write_jsonl(generated_all_path(text_config), all_records)
    write_rejections(rejected_path(text_config), all_rejections)
    retained, rejected, filter_summary = filter_records(
        all_records,
        config=text_config,
        existing_rejections=all_rejections,
        protected=protected,
        holdout_rows=holdout,
    )
    write_rejections(rejected_path(text_config), rejected)
    payload = {
        "status": "PASSED",
        "stage": "generate-text",
        "generated_rows": len(all_records),
        "admissible_rows": len(retained),
        "used_prompt_batch_size": used_batch,
        "fallback_notes": fallback_notes,
        "wall_time_seconds": round(time.perf_counter() - start, 6),
        "generated_all_sha256": sha256_file(generated_all_path(text_config)),
        "filter_summary": filter_summary,
    }
    atomic_write_json(run_dir(text_config) / "generation-summary.local.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_validate_text(config_path: Path) -> dict[str, Any]:
    _experiment, text_config, _augmentation = load_all_configs(config_path)
    payload = validate_and_select_text(text_config)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_prepare_human_decision(config_path: Path) -> dict[str, Any]:
    _experiment, text_config, _augmentation = load_all_configs(config_path)
    if not fixed_text_path(text_config).exists():
        raise FileNotFoundError(fixed_text_path(text_config))
    payload = write_review_capsule(text_config)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_admit_text(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    _experiment, text_config, _augmentation = load_all_configs(config_path)
    payload = expand_whole_file_decision(
        text_config,
        outcome=args.whole_file_outcome,
        review_revision=args.review_revision,
        decision_id=args.decision_id,
        expected_corpus_sha256=args.expected_corpus_sha256,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _piper_audio_paths(text_config: dict[str, Any]) -> Any:
    from slaif_asr.acoustic_quality import AudioPaths

    root = run_dir(text_config) / "piper"
    return AudioPaths(
        run_root=root,
        native_dir=root / "native-22050",
        final_dir=root / "final-16000",
        log_dir=root / "logs",
        benchmark_dir=root / "benchmark",
        audio_manifest=root / "audio-manifest.local.jsonl",
        validation_report=root / "audio-validation.local.json",
        synthesis_summary=root / "audio-synthesis-summary.local.json",
        benchmark_summary=root / "benchmark" / "benchmark-summary.local.json",
        gpu_monitor=root / "gpu-monitor.local.csv",
    )


def _scale200_tts_items(text_config: dict[str, Any]) -> list[Any]:
    from slaif_asr.acoustic_quality import CorpusV2TtsItem

    rows = load_jsonl(fixed_text_path(text_config))
    items = []
    for row in rows:
        items.append(
            CorpusV2TtsItem(
                candidate_id=str(row["candidate_id"]),
                spoken_text=str(row["spoken_text"]),
                target_text=str(row["target_text"]),
                language="sl-SI",
                partition_role="selected_training",
                source_id=str(row["source_id"]),
                source_family_id=str(row["source_family_id"]),
                utterance_family_id=str(row["utterance_family_id"]),
                domain=str(row.get("domain", "")),
                phenomena=tuple(str(item) for item in row.get("phenomena", [])),
            )
        )
    return sorted(items, key=lambda item: item.candidate_id)


def stage_synthesize_piper(config_path: Path) -> dict[str, Any]:
    from slaif_asr.acoustic_quality import GpuMonitor, monitor_summary, render_one_item
    from slaif_asr.tts import load_tts_config

    _experiment, text_config, _augmentation = load_all_configs(config_path)
    require_text_accepted(config_path)
    gpu = require_single_visible_cuda(allowed_name_fragments=("A100",))
    paths = _piper_audio_paths(text_config)
    if paths.audio_manifest.exists():
        rows = load_jsonl(paths.audio_manifest)
        if len(rows) == 1600:
            summary = read_json(paths.synthesis_summary) if paths.synthesis_summary.exists() else {}
            summary["status"] = "PASSED"
            summary["reused_existing_manifest"] = True
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
    items = _scale200_tts_items(text_config)
    if len(items) != 1600:
        raise RuntimeError(f"expected 1600 Piper items, found {len(items)}")
    started = time.perf_counter()
    tts_config = load_tts_config()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    last_emit = started
    with GpuMonitor(paths.gpu_monitor, physical_selector=gpu.physical_selector, interval_seconds=0.2):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(render_one_item, item=item, tts_config=tts_config, paths=paths, output_root=None): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:  # noqa: BLE001 - local failure evidence
                    failures.append({"candidate_id": item.candidate_id, "reason": type(exc).__name__, "detail": str(exc)})
                processed = len(rows) + len(failures)
                now = time.perf_counter()
                if processed % 25 == 0 or now - last_emit >= 10.0:
                    elapsed = now - started
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "stage": "synthesize-piper",
                                "processed_rows": processed,
                                "total_rows": len(items),
                                "successful": len(rows),
                                "failed": len(failures),
                                "elapsed_seconds": round(elapsed, 6),
                                "rows_per_second": round(processed / elapsed, 6) if elapsed else None,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    last_emit = now
    rows = sorted(rows, key=lambda row: str(row["candidate_id"]))
    total_duration = sum(float(row["duration_seconds"]) for row in rows)
    atomic_write_jsonl(paths.audio_manifest, rows)
    wall = time.perf_counter() - started
    summary = {
        "worker_count": 8,
        "requested": len(items),
        "successful": len(rows),
        "failed": len(failures),
        "failures": failures,
        "wall_time_seconds": round(wall, 6),
        "utterances_per_minute": round((len(rows) / wall) * 60.0, 6) if wall else None,
        "audio_seconds_per_wall_second": round(total_duration / wall, 6) if wall else None,
        "total_audio_duration_seconds": round(total_duration, 6),
        "monitor": monitor_summary(paths.gpu_monitor),
        "schema_version": "1.0",
        "status": "PASSED" if not failures and len(rows) == 1600 else "FAILED",
        "synthesis_version": "scale200-piper-synthesis-v1",
        "selected_worker_count": 8,
        "audio_manifest_sha256": sha256_file(paths.audio_manifest),
        "fixed_text_sha256": sha256_file(fixed_text_path(text_config)),
        "voice": "piper-sl_SI-artur-medium",
        "gpu": gpu.to_dict(),
    }
    atomic_write_json(paths.synthesis_summary, summary)
    if summary["status"] != "PASSED":
        raise RuntimeError(f"Piper synthesis failures: {len(failures)}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def stage_synthesize_supertonic(config_path: Path, interval: float) -> dict[str, Any]:
    from slaif_asr.supertonic3_tts import load_supertonic_config, synthesize_batched_supertonic_audio

    require_text_accepted(config_path)
    tts_config = load_supertonic_config(SUPERTONIC_SCALE200_CONFIG)
    summary = synthesize_batched_supertonic_audio(tts_config, progress_interval_seconds=interval)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def _audio_data_root(text_config: dict[str, Any]) -> Path:
    return run_dir(text_config)


def _augmentation_paths(text_config: dict[str, Any]) -> dict[str, Path]:
    root = _audio_data_root(text_config) / "augmentation"
    return {
        "root": root,
        "audio": root / "final-16000",
        "manifest": root / "augmentation-manifest.local.jsonl",
        "summary": root / "augmentation-summary.local.json",
        "validation": root / "augmentation-validation.local.json",
        "all_views": _audio_data_root(text_config) / "all-views.local.jsonl",
        "exposure_schedule": _audio_data_root(text_config) / "exposure-schedule.local.jsonl",
        "audio_certificate_local": _audio_data_root(text_config) / "scale200-audio-certificate.local.json",
    }


def _supertonic_manifest_path() -> Path:
    return repo_path("runs/data-quality/sl-corpus-v3-gams-1600-training-v1/supertonic/training-audio-manifest.local.jsonl")


def _load_clean_audio_indexes(text_config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    piper_rows = load_jsonl(_piper_audio_paths(text_config).audio_manifest)
    supertonic_rows = load_jsonl(_supertonic_manifest_path())
    if len(piper_rows) != 1600:
        raise RuntimeError(f"expected 1600 Piper clean rows, found {len(piper_rows)}")
    if len(supertonic_rows) != 12800:
        raise RuntimeError(f"expected 12800 Supertonic clean rows, found {len(supertonic_rows)}")
    clean_by_key: dict[str, dict[str, Any]] = {}
    clean_rows: list[dict[str, Any]] = []
    for row in piper_rows:
        semantic_key = str(row["candidate_id"])
        view_id = "piper-sl_SI-artur-medium"
        normalized = {
            **row,
            "semantic_key": semantic_key,
            "view_id": view_id,
            "engine": "piper",
            "voice_style_id": view_id,
            "profile_id": "clean",
            "view_type": "clean",
            "source_audio_filepath": row["audio_filepath"],
            "source_audio_sha256": row["audio_sha256"],
        }
        clean_by_key[f"{semantic_key}:{view_id}"] = normalized
        clean_rows.append(normalized)
    for row in supertonic_rows:
        semantic_key = str(row["source_key"])
        view_id = f"supertonic-{row['voice_style_id']}"
        normalized = {
            **row,
            "semantic_key": semantic_key,
            "view_id": view_id,
            "engine": "supertonic-3",
            "profile_id": "clean",
            "view_type": "clean",
            "source_audio_filepath": row["audio_filepath"],
            "source_audio_sha256": row["audio_sha256"],
        }
        clean_by_key[f"{semantic_key}:{view_id}"] = normalized
        clean_rows.append(normalized)
    return clean_by_key, clean_rows


def _build_augmentation_tasks(text_config: dict[str, Any], augmentation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(load_jsonl(fixed_text_path(text_config)), key=lambda row: stable_sha256(str(row["candidate_id"])))
    paths = _augmentation_paths(text_config)
    tasks: list[dict[str, Any]] = []
    for semantic_position, row in enumerate(rows):
        semantic_key = str(row["candidate_id"])
        for profile_index, profile in enumerate(augmentation["augmentation_profiles"]):
            profile_id = str(profile["profile_id"])
            assignment = assignment_for(
                semantic_key=semantic_key,
                semantic_position=semantic_position,
                profile_id=profile_id,
                profile_index=profile_index,
                clean_voices=TRAINING_VIEWS,
            )
            output = paths["audio"] / profile_id / assignment.source_voice / f"{semantic_key}.{profile_id}.{assignment.source_voice}.wav"
            tasks.append(
                {
                    "semantic_key": semantic_key,
                    "semantic_position": semantic_position,
                    "profile_id": profile_id,
                    "source_voice": assignment.source_voice,
                    "parameter_seed": assignment.parameter_seed,
                    "parameters": parameters_for_profile(profile, semantic_key=semantic_key),
                    "output_path": output,
                    "row": row,
                }
            )
    if len(tasks) != 17600:
        raise RuntimeError(f"expected 17600 augmentation tasks, got {len(tasks)}")
    return tasks


def _augment_one(task: dict[str, Any], source_row: dict[str, Any], augmentation_sha: str) -> dict[str, Any]:
    output_path = Path(task["output_path"])
    details = render_augmented_file(
        source_audio_path=Path(str(source_row["source_audio_filepath"])),
        output_audio_path=output_path,
        profile_id=str(task["profile_id"]),
        parameters=dict(task["parameters"]),
        seed_text=str(task["parameter_seed"]),
    )
    details["resumed_existing_file"] = False
    row = task["row"]
    return {
        "schema_version": "1.0",
        "view_type": "augmented",
        "partition_role": "selected_training",
        "semantic_key": task["semantic_key"],
        "semantic_position": task["semantic_position"],
        "profile_id": task["profile_id"],
        "source_voice": task["source_voice"],
        "source_audio_filepath": source_row["source_audio_filepath"],
        "source_audio_sha256": source_row["source_audio_sha256"],
        "audio_filepath": str(output_path.resolve()),
        "audio_sha256": details["output_audio_sha256"],
        "target_text_sha256": row.get("target_text_sha256") or source_row.get("target_text_sha256"),
        "source_family_id": row["source_family_id"],
        "utterance_family_id": row["utterance_family_id"],
        "source_id": row["source_id"],
        "domain": row.get("domain"),
        "phenomena": row.get("phenomena", []),
        "parameters": task["parameters"],
        "transform_details": details,
        "augmentation_policy_sha256": augmentation_sha,
        "augmentation_render_version": AUGMENTATION_RENDER_VERSION,
    }


def stage_augment(config_path: Path) -> dict[str, Any]:
    _experiment, text_config, augmentation = load_all_configs(config_path)
    require_text_accepted(config_path)
    paths = _augmentation_paths(text_config)
    if paths["manifest"].exists():
        rows = load_jsonl(paths["manifest"])
        if len(rows) == 17600 and all(row.get("augmentation_render_version") == AUGMENTATION_RENDER_VERSION for row in rows):
            summary = read_json(paths["summary"]) if paths["summary"].exists() else {}
            summary["status"] = "PASSED"
            summary["reused_existing_manifest"] = True
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
    clean_by_key, _clean_rows = _load_clean_audio_indexes(text_config)
    tasks = _build_augmentation_tasks(text_config, augmentation)
    augmentation_sha = sha256_file(repo_path("configs/augmentation/scale200_transcript_preserving_v1.json"))
    workers = min(32, os.cpu_count() or 1)
    started = time.perf_counter()
    last_emit = started
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for task in tasks:
            source_key = f"{task['semantic_key']}:{task['source_voice']}"
            source = clean_by_key.get(source_key)
            if source is None:
                failures.append({"reason": "missing_source_voice", "semantic_hash": stable_sha256(str(task["semantic_key"])), "voice": task["source_voice"]})
                continue
            futures[pool.submit(_augment_one, task, source, augmentation_sha)] = task
        for future in as_completed(futures):
            task = futures[future]
            try:
                output_rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - local failure evidence
                failures.append({"reason": type(exc).__name__, "semantic_hash": stable_sha256(str(task["semantic_key"])), "profile_id": task["profile_id"]})
            processed = len(output_rows) + len(failures)
            now = time.perf_counter()
            if processed % 100 == 0 or now - last_emit >= 10.0:
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "stage": "augment",
                            "processed_rows": processed,
                            "total_rows": len(tasks),
                            "successful": len(output_rows),
                            "failed": len(failures),
                            "elapsed_seconds": round(elapsed, 6),
                            "rows_per_second": round(processed / elapsed, 6) if elapsed else None,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                last_emit = now
    output_rows.sort(key=lambda row: (str(row["semantic_key"]), str(row["profile_id"]), str(row["source_voice"])))
    atomic_write_jsonl(paths["manifest"], output_rows)
    wall = time.perf_counter() - started
    profile_counts = Counter(str(row["profile_id"]) for row in output_rows)
    source_voice_counts = Counter(str(row["source_voice"]) for row in output_rows)
    summary = {
        "schema_version": "1.0",
        "augmentation_render_version": AUGMENTATION_RENDER_VERSION,
        "status": "PASSED" if not failures and len(output_rows) == 17600 else "FAILED",
        "requested": len(tasks),
        "generated": len(output_rows),
        "failed": len(failures),
        "failures": failures[:200],
        "workers": workers,
        "wall_time_seconds": round(wall, 6),
        "items_per_second": round(len(output_rows) / wall, 6) if wall else None,
        "manifest_sha256": sha256_file(paths["manifest"]),
        "profile_counts": dict(sorted(profile_counts.items())),
        "source_voice_counts": dict(sorted(source_voice_counts.items())),
    }
    atomic_write_json(paths["summary"], summary)
    if summary["status"] != "PASSED":
        raise RuntimeError(f"augmentation failed: {summary['failed']} failures")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def _normalize_view_row(row: dict[str, Any], *, view_type: str) -> dict[str, Any]:
    semantic_key = str(row.get("semantic_key") or row.get("candidate_id") or row.get("source_key"))
    voice = str(row.get("view_id") or row.get("voice_style_id") or row.get("source_voice") or "piper-sl_SI-artur-medium")
    if voice in {"M1", "M2", "M3", "M4", "F1", "F2", "F3", "F4"}:
        voice = f"supertonic-{voice}"
    return {
        "view_type": view_type,
        "semantic_key": semantic_key,
        "voice": voice,
        "profile_id": str(row.get("profile_id", "clean")),
        "audio_filepath": row["audio_filepath"],
        "audio_sha256": row["audio_sha256"],
        "target_text_sha256": row.get("target_text_sha256") or row.get("source_text_sha256"),
        "source_family_id": row.get("source_family_id"),
        "utterance_family_id": row.get("utterance_family_id"),
    }


def _stat_view(row: dict[str, Any]) -> dict[str, Any]:
    stats = read_audio_stats(Path(str(row["audio_filepath"])))
    return {
        **row,
        "observed_audio_sha256": stats.sha256,
        "sample_rate": stats.sample_rate,
        "channels": stats.channels,
        "sample_width": stats.sample_width,
        "frames": stats.frames,
        "duration_seconds": stats.duration_seconds,
        "peak_ratio": stats.peak_ratio,
        "rms_ratio": stats.rms_ratio,
        "active_frame_fraction": stats.active_frame_fraction,
        "clipping_fraction": stats.clipping_fraction,
    }


def stage_validate_audio(config_path: Path) -> dict[str, Any]:
    _experiment, text_config, augmentation = load_all_configs(config_path)
    require_text_accepted(config_path)
    paths = _augmentation_paths(text_config)
    clean_by_key, clean_rows = _load_clean_audio_indexes(text_config)
    augmented_rows = load_jsonl(paths["manifest"])
    if len(clean_rows) != 14400:
        raise RuntimeError(f"expected 14400 clean views, found {len(clean_rows)}")
    if len(augmented_rows) != 17600:
        raise RuntimeError(f"expected 17600 augmented views, found {len(augmented_rows)}")
    views = [_normalize_view_row(row, view_type="clean") for row in clean_rows] + [
        _normalize_view_row(row, view_type="augmented") for row in augmented_rows
    ]
    started = time.perf_counter()
    last_emit = started
    workers = min(32, os.cpu_count() or 1)
    stats_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_stat_view, row): row for row in views}
        for future in as_completed(futures):
            try:
                stats_rows.append(future.result())
            except Exception as exc:  # noqa: BLE001 - local validation evidence
                row = futures[future]
                issues.append({"reason": type(exc).__name__, "semantic_hash": stable_sha256(str(row["semantic_key"]))})
            processed = len(stats_rows) + len(issues)
            now = time.perf_counter()
            if processed % 100 == 0 or now - last_emit >= 10.0:
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "stage": "validate-audio",
                            "processed_rows": processed,
                            "total_rows": len(views),
                            "issues": len(issues),
                            "elapsed_seconds": round(elapsed, 6),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                last_emit = now
    paths_seen = Counter(str(row["audio_filepath"]) for row in stats_rows)
    hashes_seen = Counter(str(row["observed_audio_sha256"]) for row in stats_rows)
    semantic_seen = Counter(str(row["semantic_key"]) for row in stats_rows)
    for row in stats_rows:
        if row["observed_audio_sha256"] != row["audio_sha256"]:
            issues.append({"reason": "audio_sha256_mismatch", "semantic_hash": stable_sha256(str(row["semantic_key"]))})
        if row["sample_rate"] != 16000 or row["channels"] != 1 or row["sample_width"] != 2:
            issues.append({"reason": "audio_format", "semantic_hash": stable_sha256(str(row["semantic_key"]))})
        if not (0.2 <= float(row["duration_seconds"]) <= 30.0):
            issues.append({"reason": "duration_bounds", "semantic_hash": stable_sha256(str(row["semantic_key"]))})
        if float(row["peak_ratio"]) <= 0.001 or float(row["rms_ratio"]) <= 0.0001:
            issues.append({"reason": "silence", "semantic_hash": stable_sha256(str(row["semantic_key"]))})
        if float(row["clipping_fraction"]) > 0.01:
            issues.append({"reason": "clipping", "semantic_hash": stable_sha256(str(row["semantic_key"]))})
    duplicate_paths = sum(count - 1 for count in paths_seen.values() if count > 1)
    duplicate_hashes = sum(count - 1 for count in hashes_seen.values() if count > 1)
    if duplicate_paths:
        issues.append({"reason": "duplicate_audio_path", "count": duplicate_paths})
    if duplicate_hashes:
        issues.append({"reason": "duplicate_audio_sha256", "count": duplicate_hashes})
    if len(semantic_seen) != 1600 or any(count != 20 for count in semantic_seen.values()):
        issues.append({"reason": "semantic_view_count"})
    schedule, schedule_summary = build_exposure_schedule(load_jsonl(fixed_text_path(text_config)), augmentation)
    atomic_write_jsonl(paths["exposure_schedule"], schedule)
    atomic_write_jsonl(paths["all_views"], sorted(stats_rows, key=lambda row: (str(row["semantic_key"]), str(row["view_type"]), str(row["voice"]), str(row["profile_id"]))))
    profile_counts = Counter(str(row["profile_id"]) for row in stats_rows if row["view_type"] == "augmented")
    voice_counts = Counter(str(row["voice"]) for row in stats_rows)
    duration_by_view = {
        "clean": round(sum(float(row["duration_seconds"]) for row in stats_rows if row["view_type"] == "clean"), 6),
        "augmented": round(sum(float(row["duration_seconds"]) for row in stats_rows if row["view_type"] == "augmented"), 6),
    }
    validation = {
        "schema_version": "1.0",
        "validator": "scale200-audio-validator-v1",
        "status": "AUDIO_ACCEPTED" if not issues else "AUDIO_REJECTED",
        "view_records": len(stats_rows),
        "clean_files": sum(1 for row in stats_rows if row["view_type"] == "clean"),
        "augmented_files": sum(1 for row in stats_rows if row["view_type"] == "augmented"),
        "semantic_rows": len(semantic_seen),
        "profile_counts": dict(sorted(profile_counts.items())),
        "voice_counts": dict(sorted(voice_counts.items())),
        "duration_seconds": duration_by_view,
        "duplicate_paths": duplicate_paths,
        "duplicate_hashes": duplicate_hashes,
        "issues_by_reason": dict(sorted(Counter(str(issue["reason"]) for issue in issues).items())),
        "issues": issues[:500],
        "schedule": {**schedule_summary, "schedule_sha256": sha256_file(paths["exposure_schedule"])},
        "hashes": {
            "piper_audio_manifest_sha256": sha256_file(_piper_audio_paths(text_config).audio_manifest),
            "supertonic_audio_manifest_sha256": sha256_file(_supertonic_manifest_path()),
            "augmentation_manifest_sha256": sha256_file(paths["manifest"]),
            "all_views_sha256": sha256_file(paths["all_views"]),
        },
        "limitations": [
            "All audio is synthetic.",
            "The 200x multiplier is exposure count, not independent linguistic information.",
            "No TRAINING_ELIGIBLE status is issued.",
        ],
    }
    atomic_write_json(paths["validation"], validation)
    local_certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v3-nine-voice-augmented-audio-v1-local",
        "status": validation["status"],
        "corpus_id": text_config["corpus_id"],
        "text_sha256": sha256_file(fixed_text_path(text_config)),
        "view_records": len(stats_rows),
        "clean_files": validation["clean_files"],
        "augmented_files": validation["augmented_files"],
        "schedule_sha256": validation["schedule"]["schedule_sha256"],
        "all_views_sha256": validation["hashes"]["all_views_sha256"],
    }
    atomic_write_json(paths["audio_certificate_local"], local_certificate)
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    if validation["status"] != "AUDIO_ACCEPTED":
        raise RuntimeError(f"audio validation failed: {validation['issues_by_reason']}")
    return validation


def _experiment_run_dir(experiment: dict[str, Any]) -> Path:
    return repo_path(experiment["local_outputs"]["run_root"])


def _audio_certificate_path(experiment: dict[str, Any]) -> Path:
    return repo_path(experiment["public_outputs"]["audio_certificate"])


def _experiment_certificate_path(experiment: dict[str, Any]) -> Path:
    return repo_path(experiment["public_outputs"]["experiment_certificate"])


def _audio_report_json_path(experiment: dict[str, Any]) -> Path:
    return repo_path(experiment["public_outputs"]["audio_report_json"])


def _audio_report_markdown_path(experiment: dict[str, Any]) -> Path:
    return repo_path(experiment["public_outputs"]["audio_report_markdown"])


def _experiment_report_json_path(experiment: dict[str, Any]) -> Path:
    return repo_path(experiment["public_outputs"]["experiment_report_json"])


def _experiment_report_markdown_path(experiment: dict[str, Any]) -> Path:
    return repo_path(experiment["public_outputs"]["experiment_report_markdown"])


def _public_payload_safe(payload: Any) -> None:
    from slaif_asr.corpus_v2_training import assert_public_report_safe

    assert_public_report_safe(payload)


def _public_model_identity(experiment: dict[str, Any]) -> dict[str, Any]:
    model = experiment["model"]
    return {
        "repository": model["repository"],
        "revision": model["revision"],
        "checkpoint_sha256": model["checkpoint_sha256"],
        "nemo_revision": model["nemo_revision"],
    }


def _write_audio_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Data Report 0012: Nine-voice Scale-200 Acoustic Admission",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This report records aggregate-only evidence for the synthetic scale-200 audio bank. It does not contain generated text, candidate IDs, local paths, or audio.",
        "",
        "## Counts",
        "",
        f"- Semantic texts: {payload['counts']['semantic_rows']}",
        f"- Clean files: {payload['counts']['clean_files']}",
        f"- Augmented files: {payload['counts']['augmented_files']}",
        f"- Total view records: {payload['counts']['view_records']}",
        f"- Exposure records: {payload['schedule']['exposures']}",
        "",
        "## Hashes",
        "",
        f"- Fixed text SHA256: `{payload['hashes']['fixed_text_sha256']}`",
        f"- All views SHA256: `{payload['hashes']['all_views_sha256']}`",
        f"- Exposure schedule SHA256: `{payload['schedule']['schedule_sha256']}`",
        "",
        "## Safety",
        "",
        f"- Duplicate paths: {payload['validation']['duplicate_paths']}",
        f"- Duplicate hashes: {payload['validation']['duplicate_hashes']}",
        f"- Issue reasons: `{payload['validation']['issues_by_reason']}`",
        "- `TRAINING_ELIGIBLE` was not issued.",
        "- The 200x multiplier refers to exposure count, not independent linguistic information.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(lines))


def write_audio_and_training_authorization(config_path: Path) -> dict[str, Any]:
    experiment, text_config, augmentation = load_all_configs(config_path)
    require_text_accepted(config_path)
    paths = _augmentation_paths(text_config)
    validation = read_json(paths["validation"])
    if validation.get("status") != "AUDIO_ACCEPTED":
        raise RuntimeError("audio validation must be AUDIO_ACCEPTED before training authorization")
    piper_summary = read_json(_piper_audio_paths(text_config).synthesis_summary)
    super_summary = read_json(_audio_data_root(text_config) / "supertonic" / "batched-synthesis-summary.local.json")
    augmentation_summary = read_json(paths["summary"])
    fixed_text_sha = sha256_file(fixed_text_path(text_config))
    config_sha = sha256_file(repo_path(config_path))
    augmentation_sha = sha256_file(repo_path(experiment["augmentation_config"]))
    audio_certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v3-nine-voice-augmented-audio-v1",
        "status": "AUDIO_ACCEPTED",
        "decision_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "corpus_id": text_config["corpus_id"],
        "text_certificate_sha256": sha256_file(text_certificate_path()),
        "fixed_text_sha256": fixed_text_sha,
        "counts": {
            "semantic_rows": int(validation["semantic_rows"]),
            "clean_files": int(validation["clean_files"]),
            "augmented_files": int(validation["augmented_files"]),
            "view_records": int(validation["view_records"]),
            "piper_clean_files": 1600,
            "supertonic_clean_files": 12800,
        },
        "hashes": validation["hashes"],
        "schedule": {
            "exposures": int(validation["schedule"]["exposures"]),
            "optimizer_steps": int(validation["schedule"]["optimizer_steps"]),
            "schedule_sha256": validation["schedule"]["schedule_sha256"],
            "heldout_voice_exposures": validation["schedule"]["heldout_voice_exposures"],
        },
        "validation": {
            "duplicate_paths": int(validation["duplicate_paths"]),
            "duplicate_hashes": int(validation["duplicate_hashes"]),
            "issues_by_reason": validation["issues_by_reason"],
        },
        "voices": {
            "clean_training_views": list(TRAINING_VIEWS),
            "voice_counts": validation["voice_counts"],
        },
        "augmentation": {
            "policy_sha256": augmentation_sha,
            "render_version": AUGMENTATION_RENDER_VERSION,
            "profile_counts": validation["profile_counts"],
        },
        "synthesis": {
            "piper": {
                "audio_manifest_sha256": piper_summary["audio_manifest_sha256"],
                "wall_time_seconds": piper_summary["wall_time_seconds"],
                "utterances_per_minute": piper_summary["utterances_per_minute"],
            },
            "supertonic": {
                "audio_manifest_sha256": super_summary["manifests"]["audio_manifest_sha256"],
                "native_rows": super_summary["native_rows"],
                "converted_rows": super_summary["converted_rows"],
                "resumed_from_complete_manifests": bool(super_summary.get("resumed_from_complete_manifests", False)),
            },
            "augmentation": {
                "manifest_sha256": augmentation_summary["manifest_sha256"],
                "wall_time_seconds": augmentation_summary["wall_time_seconds"],
                "items_per_second": augmentation_summary["items_per_second"],
            },
        },
        "limitations": validation["limitations"],
        "prohibited_statuses": ["TRAINING_ELIGIBLE"],
    }
    _public_payload_safe(audio_certificate)
    atomic_write_json(_audio_certificate_path(experiment), audio_certificate)

    audio_report = {
        "schema_version": "1.0",
        "report_id": "0012-nine-voice-scale200-acoustic-admission",
        "status": "AUDIO_ACCEPTED",
        "certificate_sha256": sha256_file(_audio_certificate_path(experiment)),
        "counts": audio_certificate["counts"],
        "hashes": {"fixed_text_sha256": fixed_text_sha, **validation["hashes"]},
        "schedule": audio_certificate["schedule"],
        "validation": audio_certificate["validation"],
        "voice_counts": validation["voice_counts"],
        "profile_counts": validation["profile_counts"],
        "duration_seconds": validation["duration_seconds"],
        "multiplier_table": multiplier_table(),
        "limitations": audio_certificate["limitations"],
    }
    _public_payload_safe(audio_report)
    atomic_write_json(_audio_report_json_path(experiment), audio_report)
    _write_audio_markdown(_audio_report_markdown_path(experiment), audio_report)

    experiment_certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v3-scale200-joint-adapter-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "decision_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "work_order_id": experiment["work_order_id"],
        "named_exception": "scale-200 synthetic joint-adapter diagnostic",
        "text_certificate_sha256": sha256_file(text_certificate_path()),
        "audio_certificate_sha256": sha256_file(_audio_certificate_path(experiment)),
        "fixed_text_sha256": fixed_text_sha,
        "all_views_sha256": validation["hashes"]["all_views_sha256"],
        "exposure_schedule_sha256": validation["schedule"]["schedule_sha256"],
        "experiment_config_sha256": config_sha,
        "augmentation_policy_sha256": augmentation_sha,
        "model": _public_model_identity(experiment),
        "adapter": experiment["adapter"],
        "training": experiment["training"],
        "evaluation": experiment["evaluation"],
        "authorized_actions": [
            "internally train one frozen-base Slovenian joint-adapter arm on the declared 32,000-exposure synthetic schedule",
            "evaluate the trained adapter using the declared batch-32 directional suite",
            "write privacy-safe aggregate evidence",
        ],
        "prohibited_actions": [
            "issuing TRAINING_ELIGIBLE",
            "accepting a checkpoint or adapter as a parent",
            "running batch-1 canonical evaluation",
            "using M5/F5 or holdout rows in training",
            "publishing generated text, audio, adapter, or checkpoint artifacts",
            "changing adapter rank, optimizer, learning rate, batch size, or text membership",
        ],
        "limitations": [
            "DIAGNOSTIC_ONLY synthetic experiment.",
            "The 200x multiplier is an exposure-count multiplier, not independent linguistic information.",
            "Fast batch-32 evaluation is directional and noncanonical.",
        ],
    }
    _public_payload_safe(experiment_certificate)
    atomic_write_json(_experiment_certificate_path(experiment), experiment_certificate)
    result = {
        "status": "PASSED",
        "audio_certificate_sha256": sha256_file(_audio_certificate_path(experiment)),
        "audio_report_sha256": sha256_file(_audio_report_json_path(experiment)),
        "experiment_certificate_sha256": sha256_file(_experiment_certificate_path(experiment)),
    }
    return result


def require_training_authorization(config_path: Path) -> dict[str, Any]:
    from slaif_asr.corpus_v2_training import git_tracked_and_clean_at_head

    experiment, text_config, _augmentation = load_all_configs(config_path)
    certificate_path = _experiment_certificate_path(experiment)
    if not certificate_path.exists():
        raise RuntimeError("scale-200 DIAGNOSTIC_ONLY experiment certificate is missing")
    tracked = git_tracked_and_clean_at_head(certificate_path)
    certificate = read_json(certificate_path)
    if certificate.get("status") != "DIAGNOSTIC_ONLY":
        raise RuntimeError("scale-200 experiment certificate must be DIAGNOSTIC_ONLY")
    if certificate.get("work_order_id") != "0025":
        raise RuntimeError("scale-200 experiment certificate work-order mismatch")
    if certificate.get("experiment_config_sha256") != sha256_file(repo_path(config_path)):
        raise RuntimeError("scale-200 experiment certificate config SHA mismatch")
    if certificate.get("audio_certificate_sha256") != sha256_file(_audio_certificate_path(experiment)):
        raise RuntimeError("scale-200 experiment certificate audio certificate SHA mismatch")
    if certificate.get("fixed_text_sha256") != sha256_file(fixed_text_path(text_config)):
        raise RuntimeError("scale-200 experiment certificate text SHA mismatch")
    return {"certificate": certificate, "tracked": tracked}


def require_nemotron_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Nemotron stages must run with CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("Nemotron stages must run with NVIDIA_TF32_OVERRIDE=0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def configure_torch() -> Any:
    return _JOINT.configure_torch()


def restore_base_model(experiment: dict[str, Any], *, reporter: LiveProgressReporter | None = None) -> Any:
    return _JOINT.restore_base_model(experiment, reporter=reporter)


def prepare_adapter_model(model: Any, experiment: dict[str, Any], *, enable: bool) -> dict[str, Any]:
    return _JOINT.prepare_adapter_model(model, experiment, enable=enable)


def _load_scale_text_by_id(text_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(fixed_text_path(text_config))
    if len(rows) != 1600:
        raise RuntimeError(f"expected 1600 fixed text rows, found {len(rows)}")
    return {str(row["candidate_id"]): row for row in rows}


def _view_lookup(text_config: dict[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    paths = _augmentation_paths(text_config)
    rows = load_jsonl(paths["all_views"])
    if len(rows) != 32000:
        raise RuntimeError(f"expected 32000 all-view rows, found {len(rows)}")
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["semantic_key"]), str(row["view_type"]), str(row["voice"]), str(row["profile_id"]))
        if key in lookup:
            raise RuntimeError(f"duplicate view key: {key}")
        lookup[key] = row
    return lookup


def _training_record_from_view(text_row: dict[str, Any], view_row: dict[str, Any]) -> Any:
    from slaif_asr.corpus_v2_training import TrainingRecord

    path = Path(str(view_row["audio_filepath"]))
    if not path.exists():
        raise FileNotFoundError(path)
    if str(view_row["target_text_sha256"]) != stable_sha256(str(text_row["target_text"])):
        raise RuntimeError("scale-200 text/audio text-hash mismatch")
    return TrainingRecord(
        selected_training_id=str(text_row["candidate_id"]),
        audio_filepath=str(path),
        duration=float(view_row["duration_seconds"]),
        text=str(text_row["target_text"]),
        text_sha256=str(view_row["target_text_sha256"]),
        audio_sha256=str(view_row["audio_sha256"]),
        selection_reason="scale200",
        selection_rank=int(text_row["generation"]["prompt_cell"].removeprefix("cell")) if str(text_row["generation"]["prompt_cell"]).startswith("cell") else 0,
    )


def _clean_probe_records(text_config: dict[str, Any]) -> list[Any]:
    from slaif_asr.corpus_v2_training import select_probe_records

    text_by_id = _load_scale_text_by_id(text_config)
    views = _view_lookup(text_config)
    records = []
    for semantic_key in sorted(text_by_id):
        view = views[(semantic_key, "clean", "piper-sl_SI-artur-medium", "clean")]
        records.append(_training_record_from_view(text_by_id[semantic_key], view))
    return select_probe_records(records, 32), records


def _load_scheduled_round_records(text_config: dict[str, Any]) -> tuple[dict[int, list[Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    paths = _augmentation_paths(text_config)
    text_by_id = _load_scale_text_by_id(text_config)
    views = _view_lookup(text_config)
    schedule = load_jsonl(paths["exposure_schedule"])
    if len(schedule) != 32000:
        raise RuntimeError(f"expected 32000 exposure schedule rows, found {len(schedule)}")
    rounds: dict[int, list[Any]] = defaultdict(list)
    meta_by_audio: dict[str, dict[str, Any]] = {}
    for item in schedule:
        semantic_key = str(item["semantic_key"])
        key = (semantic_key, str(item["view_type"]), str(item["voice"]), str(item["profile_id"]))
        view = views[key]
        record = _training_record_from_view(text_by_id[semantic_key], view)
        rounds[int(item["round"])].append(record)
        meta_by_audio[record.audio_filepath] = {
            "voice": item["voice"],
            "profile_id": item["profile_id"],
            "view_type": item["view_type"],
            "spec_augment": bool(item.get("spec_augment", False)),
        }
    for round_index in range(1, 21):
        rows = rounds.get(round_index, [])
        if len(rows) != 1600:
            raise RuntimeError(f"round {round_index} has {len(rows)} rows, expected 1600")
    return rounds, meta_by_audio, {"schedule_sha256": sha256_file(paths["exposure_schedule"])}


def verify_scale_artifact(experiment: dict[str, Any]) -> dict[str, Any]:
    from slaif_asr.slovenian_joint_adapter import compare_base_state, load_adapter_artifact, load_adapter_spec, state_dict_cpu

    configure_torch()
    spec = load_adapter_spec(experiment["adapter"]["config"])
    model = restore_base_model(experiment)
    base_state = state_dict_cpu(model)
    artifact = _experiment_run_dir(experiment) / ARM_NAME / "artifacts" / "sl-si-joint-adapter-v1.pt"
    payload = load_adapter_artifact(artifact, model=model, spec=spec)
    restored_state = state_dict_cpu(model)
    base_integrity = compare_base_state(base_state, restored_state)
    if not base_integrity["base_tensors_identical"]:
        raise RuntimeError("adapter restore changed base tensors")
    report = {
        "status": "PASSED",
        "artifact_name": payload["adapter_name"],
        "base_integrity": base_integrity,
        "disabled_after_restore": True,
    }
    atomic_write_json(_experiment_run_dir(experiment) / ARM_NAME / "restore-integrity.local.json", report)
    return report


def stage_train(config_path: Path, interval: float) -> dict[str, Any]:
    from slaif_asr.corpus_v2_scoring import CHECKPOINT_SHA256, NEMO_REVISION, verify_runtime_identities
    from slaif_asr.corpus_v2_training import assert_epoch_covers_once, deterministic_epoch_batches, make_training_batch, rnnt_loss, runtime_summary
    from slaif_asr.prompt_column import derive_prompt_column_selection
    from slaif_asr.slovenian_joint_adapter import (
        ADAPTER_NAME,
        adapter_parameters,
        compare_adapter_state,
        compare_base_state,
        expected_trainable_count,
        load_adapter_spec,
        save_adapter_artifact,
        state_dict_cpu,
        verify_optimizer_scope,
    )

    experiment, text_config, _augmentation = load_all_configs(config_path)
    require_training_authorization(config_path)
    require_nemotron_env()
    runtime = verify_runtime_identities(check_gpu=True)
    arm_dir = _experiment_run_dir(experiment) / ARM_NAME
    summary_path = arm_dir / "training-summary.local.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("status") == "PASSED" and int(summary.get("optimizer_steps", -1)) == 4000:
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
    torch = configure_torch()
    probe_records, full_clean_probe = _clean_probe_records(text_config)
    rounds, meta_by_audio, schedule_summary = _load_scheduled_round_records(text_config)
    reporter = LiveProgressReporter(stage="train", arm=ARM_NAME, ndjson_path=arm_dir / "progress" / "train.local.ndjson")
    reporter.start("training scale-200 joint adapter")
    model = restore_base_model(experiment, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=arm_dir / "progress" / "restore.local.ndjson"))
    model.eval()
    adapter_summary = prepare_adapter_model(model, experiment, enable=True)
    initial_state = state_dict_cpu(model)
    prompt_selection = derive_prompt_column_selection(model, "sl-SI")
    if adapter_summary["trainable_parameters"] != expected_trainable_count(adapter_summary["joint_hidden"], 32):
        raise RuntimeError("scale-200 joint-adapter trainable parameter count mismatch")
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=float(experiment["training"]["learning_rate"]), weight_decay=0.0)
    verify_optimizer_scope(optimizer, model)
    initial_probe = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
    initial_full = mean_loss(model, prompt_selection.prompt_index, full_clean_probe, device="cuda")
    probe_curve = [{"round": 0, "mean_loss": round(initial_probe, 6)}]
    adapter_norm_curve: list[dict[str, Any]] = []
    grad_norms: list[float] = []
    optimizer_steps = 0
    sample_exposures = 0
    audio_seconds = 0.0
    padded_audio_seconds = 0.0
    rolling_losses: list[float] = []
    voice_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    view_type_counts: Counter[str] = Counter()
    monitor_path = arm_dir / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index="1", output_csv=monitor_path, interval_seconds=0.2)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    monitor.start()
    try:
        total_steps = int(experiment["training"]["optimizer_steps"])
        for round_index in range(1, 21):
            records = rounds[round_index]
            layout = deterministic_epoch_batches(
                records,
                batch_size=int(experiment["training"]["batch_size"]),
                epoch=round_index,
                seed=int(experiment["training"]["seed"]),
                bucketed=True,
            )
            assert_epoch_covers_once(layout, len(records))
            for batch_indices in layout.batches:
                batch_records = [records[index] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                loss = rnnt_loss(model, make_training_batch(model, batch_records, device="cuda"), prompt_selection.prompt_index)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite scale-200 training loss")
                loss.backward()
                grad_norm, grads_ok = finite_grad_norm(adapter_parameters(model))
                if not grads_ok:
                    raise RuntimeError("non-finite scale-200 training gradient")
                for name, parameter in model.named_parameters():
                    if not name.startswith(f"joint.adapter_layer.{ADAPTER_NAME}.") and parameter.grad is not None:
                        raise RuntimeError(f"pretrained parameter received gradient: {name}")
                optimizer.step()
                optimizer_steps += 1
                sample_exposures += len(batch_records)
                audio_seconds += sum(row.duration for row in batch_records)
                padded_audio_seconds += max(row.duration for row in batch_records) * len(batch_records)
                for row in batch_records:
                    meta = meta_by_audio[row.audio_filepath]
                    voice_counts[str(meta["voice"])] += 1
                    profile_counts[str(meta["profile_id"])] += 1
                    view_type_counts[str(meta["view_type"])] += 1
                loss_value = float(loss.detach().cpu())
                rolling_losses.append(loss_value)
                rolling_losses = rolling_losses[-10:]
                grad_norms.append(grad_norm)
                if optimizer_steps % 25 == 0:
                    elapsed = time.perf_counter() - started
                    reporter.progress(
                        epoch=round_index,
                        total_epochs=20,
                        step=optimizer_steps,
                        total_steps=total_steps,
                        current_loss=round(loss_value, 6),
                        rolling_mean_loss=round(sum(rolling_losses) / len(rolling_losses), 6),
                        examples_per_second=round(sample_exposures / elapsed, 6) if elapsed else None,
                        audio_seconds_per_wall_second=round(audio_seconds / elapsed, 6) if elapsed else None,
                        cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    )
            probe_loss = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
            probe_curve.append({"round": round_index, "mean_loss": round(probe_loss, 6)})
            adapter_norm = sum(float(torch.linalg.vector_norm(parameter.detach()).cpu()) for parameter in adapter_parameters(model))
            adapter_norm_curve.append({"round": round_index, "adapter_parameter_norm": round(adapter_norm, 6)})
    except Exception as exc:
        reporter.failed(message="training failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()
    reporter.heartbeat(message="post-training validation starting", step=optimizer_steps, total_steps=int(experiment["training"]["optimizer_steps"]))
    with heartbeat_thread(reporter, interval_seconds=interval, message="post-training validation in progress"):
        wall = time.perf_counter() - started
        final_probe = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
        final_full = mean_loss(model, prompt_selection.prompt_index, full_clean_probe, device="cuda")
        trained_state = state_dict_cpu(model)
        base_integrity = compare_base_state(initial_state, trained_state)
        adapter_integrity = compare_adapter_state(initial_state, trained_state)
        if not base_integrity["base_tensors_identical"]:
            raise RuntimeError("pretrained tensor changed during scale-200 joint-adapter training")
        artifact_path = arm_dir / "artifacts" / "sl-si-joint-adapter-v1.pt"
        artifact_sha = save_adapter_artifact(
            artifact_path,
            model=model,
            spec=load_adapter_spec(experiment["adapter"]["config"]),
            metadata={
                "base_checkpoint_sha256": CHECKPOINT_SHA256,
                "nemo_revision": NEMO_REVISION,
                "fixed_text_sha256": sha256_file(fixed_text_path(text_config)),
                "all_views_sha256": sha256_file(_augmentation_paths(text_config)["all_views"]),
                "exposure_schedule_sha256": schedule_summary["schedule_sha256"],
                "experiment_config_sha256": sha256_file(repo_path(config_path)),
                "adapter_config_sha256": sha256_file(repo_path(experiment["adapter"]["config"])),
            },
        )
        verify_command = [sys.executable, "-u", __file__, "--config", str(config_path), "--stage", "verify-artifact"]
        completed = subprocess.run(verify_command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
        (arm_dir / "verify-artifact.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError("scale-200 adapter artifact restore verification failed")
        _JOINT.enable_for_target_language(model, "sl-SI")
        checkpoint_out = arm_dir / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
        model.save_to(str(checkpoint_out))
    payload = {
        "arm": ARM_NAME,
        "status": "PASSED",
        "batch_size": int(experiment["training"]["batch_size"]),
        "duration_bucketing": True,
        "rounds": 20,
        "sample_exposures": sample_exposures,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(experiment["training"]["learning_rate"]),
        "initial_probe_loss": round(initial_probe, 6),
        "final_probe_loss": round(final_probe, 6),
        "initial_full_clean_loss": round(initial_full, 6),
        "final_full_clean_loss": round(final_full, 6),
        "full_loss_reduction_percent": round((initial_full - final_full) / initial_full * 100.0, 6) if initial_full else None,
        "probe_curve": probe_curve,
        "gradient_norm": {"min": round(min(grad_norms), 6), "max": round(max(grad_norms), 6), "final": round(grad_norms[-1], 6)},
        "adapter_norm_curve": adapter_norm_curve,
        "wall_time_seconds": round(wall, 6),
        "examples_per_second": round(sample_exposures / wall, 6) if wall else None,
        "audio_seconds_per_wall_second": round(audio_seconds / wall, 6) if wall else None,
        "padding_ratio": round(padded_audio_seconds / audio_seconds, 6) if audio_seconds else None,
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "adapter": adapter_summary,
        "trainable_parameter_count": adapter_summary["trainable_parameters"],
        "base_integrity": base_integrity,
        "adapter_integrity": adapter_integrity,
        "artifact_sha256": artifact_sha,
        "restore_integrity": read_json(arm_dir / "restore-integrity.local.json"),
        "evaluation_checkpoint_sha256": sha256_file(checkpoint_out),
        "exposure_counts_by_voice": dict(sorted(voice_counts.items())),
        "exposure_counts_by_profile": dict(sorted(profile_counts.items())),
        "exposure_counts_by_view_type": dict(sorted(view_type_counts.items())),
        "schedule_sha256": schedule_summary["schedule_sha256"],
        "runtime": runtime,
        "runtime_summary": runtime_summary(),
    }
    atomic_write_json(summary_path, payload)
    reporter.complete("training complete", step=optimizer_steps, total_steps=int(experiment["training"]["optimizer_steps"]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "PASSED", "arm": ARM_NAME, "optimizer_steps": optimizer_steps}, ensure_ascii=False, sort_keys=True))
    return payload


def _metric_row(split: dict[str, Any]) -> dict[str, Any]:
    normalized = split["metrics"]["normalized"]
    raw = split["metrics"]["raw"]
    return {
        "wer": round(float(normalized["corpus_wer"]), 3),
        "cer": round(float(normalized["corpus_cer"]), 3),
        "raw_wer": round(float(raw["corpus_wer"]), 3),
        "raw_cer": round(float(raw["corpus_cer"]), 3),
        "empty": int(raw["empty_hypothesis_count"]),
    }


def _scale200_burden(metrics: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]]) -> float:
    burden = 0.0
    for split in ("fleurs_v2", "artur_j"):
        burden += max(0.0, metrics[split]["wer"] - base[split]["wer"])
        burden += max(0.0, metrics[split]["cer"] - base[split]["cer"])
    return round(burden, 6)


def _scale200_synthetic_improves(metrics: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]], split: str) -> bool:
    return metrics[split]["wer"] < base[split]["wer"] or metrics[split]["cer"] < base[split]["cer"]


def _classify_scale200(metric_table: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    base = metric_table["base"]
    replay = metric_table["batched_replay_joint_adapter"]
    scale = metric_table["scale200_joint_adapter"]
    piper_gain = _scale200_synthetic_improves(scale, base, "piper_synthetic_holdout")
    super_gain = _scale200_synthetic_improves(scale, base, "supertonic_heldout_voice_holdout")
    real_non_regression = True
    real_improvement = False
    for split in ("fleurs_v2", "artur_j"):
        if scale[split]["wer"] - base[split]["wer"] > 1.0:
            real_non_regression = False
        if scale[split]["cer"] - base[split]["cer"] > 1.5:
            real_non_regression = False
        if scale[split]["empty"] > base[split]["empty"]:
            real_non_regression = False
        if scale[split]["wer"] - base[split]["wer"] < 0.0 or scale[split]["cer"] - base[split]["cer"] < 0.0:
            real_improvement = True
    burden = _scale200_burden(scale, base)
    reference_burden = 9.536
    burden_reduction = (reference_burden - burden) / reference_burden * 100.0 if reference_burden else 0.0
    no_worse_than_replay = True
    for split in ("fleurs_v2", "artur_j"):
        if scale[split]["wer"] - replay[split]["wer"] > 0.5 or scale[split]["cer"] - replay[split]["cer"] > 0.5:
            no_worse_than_replay = False
    if piper_gain and super_gain and real_non_regression and real_improvement:
        classification = "SCALE200_SYNTHETIC_REAL_GAIN_DIRECTIONAL"
    elif piper_gain and super_gain and burden_reduction >= 30.0 and no_worse_than_replay:
        classification = "SCALE200_SYNTHETIC_MITIGATES_REPLAY_REGRESSION"
    elif piper_gain or super_gain:
        classification = "SCALE200_SYNTHETIC_CONFIRMS_SYNTHETIC_ONLY"
    else:
        classification = "SCALE200_SYNTHETIC_NOT_SUPPORTED"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "piper_holdout_improves": piper_gain,
        "supertonic_heldout_improves": super_gain,
        "real_non_regression": real_non_regression,
        "real_improvement": real_improvement,
        "reference_replay_burden": reference_burden,
        "scale200_burden": burden,
        "burden_reduction_percent": round(burden_reduction, 6),
        "no_metric_more_than_half_point_worse_than_replay": no_worse_than_replay,
    }


def stage_evaluate_directional(config_path: Path) -> dict[str, Any]:
    from slaif_asr.corpus_v2_scoring import verify_runtime_identities
    from slaif_asr.directional_evaluation import (
        DirectionalModel,
        load_directional_suite,
        run_directional_model,
        suite_plan_hash,
        write_privacy_safe_suite_manifest,
    )

    experiment, _text_config, _augmentation = load_all_configs(config_path)
    require_training_authorization(config_path)
    require_nemotron_env()
    verify_runtime_identities(check_gpu=True)
    checkpoint = _experiment_run_dir(experiment) / ARM_NAME / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
    if not checkpoint.exists():
        raise RuntimeError("scale-200 adapter checkpoint is missing")
    fast_config = read_json(FAST_DIRECTIONAL_CONFIG)
    reference_report_info = experiment["directional_reference_report"]
    reference_report_path = repo_path(reference_report_info["path"])
    if sha256_file(reference_report_path) != reference_report_info["sha256"]:
        raise RuntimeError("Experiment 0012 directional reference report SHA mismatch")
    reference_report = read_json(reference_report_path)
    suite_records, split_records = load_directional_suite(fast_config)
    output_dir = _experiment_run_dir(experiment) / "directional-evaluation"
    suite_manifest_sha = write_privacy_safe_suite_manifest(output_dir / "suite-plan.local.jsonl", suite_records)
    model = DirectionalModel("scale200_joint_adapter", checkpoint, sha256_file(checkpoint), "work_order_0025")
    summary = run_directional_model(
        config=fast_config,
        model=model,
        suite_records=suite_records,
        split_records=split_records,
        run_dir=output_dir,
        python_executable=Path(sys.executable),
    )
    scale_metrics = {
        split_name: _metric_row(split)
        for split_name, split in summary["splits"].items()
    }
    metric_table = dict(reference_report["directional_evaluation"]["metric_table"])
    metric_table["scale200_joint_adapter"] = scale_metrics
    decision = _classify_scale200(metric_table)
    payload = {
        "status": "PASSED",
        "suite_rows": len(suite_records),
        "suite_plan_sha256": suite_plan_hash(suite_records),
        "suite_manifest_sha256": suite_manifest_sha,
        "evaluation_policy": {
            "batch_size": 32,
            "duration_bucketing": True,
            "att_context_size": fast_config["directional_evaluation"]["att_context_size"],
            "target_lang": fast_config["directional_evaluation"]["target_lang"],
            "canonical": False,
            "promotion_eligible": False,
        },
        "reference_report_sha256": reference_report_info["sha256"],
        "models": {"scale200_joint_adapter": summary},
        "metric_table": metric_table,
        "decision": decision,
    }
    atomic_write_json(output_dir / "summary.local.json", payload)
    print(json.dumps({"status": "PASSED", "classification": decision["classification"]}, ensure_ascii=False, sort_keys=True))
    return payload


def _write_experiment_markdown(path: Path, payload: dict[str, Any]) -> None:
    table = payload["directional_evaluation"]["metric_table"]
    decision = payload["directional_evaluation"]["decision"]
    lines = [
        "# Experiment 0013: GaMS 1600 Nine-voice Scale-200 Directional Diagnostic",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This diagnostic uses 1,600 accepted synthetic texts, nine clean synthetic voice sources, and eleven transcript-preserving augmentation views. Evaluation is fast batch-32 directional evidence only; no canonical batch-1 evaluation was run.",
        "",
        "## Scale",
        "",
        f"- Semantic rows: {payload['data']['semantic_rows']}",
        f"- Clean files: {payload['data']['clean_files']}",
        f"- Augmented files: {payload['data']['augmented_files']}",
        f"- Exposure records: {payload['training']['sample_exposures']}",
        "- The 200x multiplier refers to exposure count, not independent linguistic information.",
        "",
        "## Directional Metrics",
        "",
        "| Model | Piper holdout WER/CER | Supertonic holdout WER/CER | FLEURS-v2 WER/CER | ARTUR-J WER/CER |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in ("base", "piper_joint_adapter", "supertonic3_joint_adapter", "batched_replay_joint_adapter", "scale200_joint_adapter"):
        row = table[model_id]
        lines.append(
            f"| {model_id} | {row['piper_synthetic_holdout']['wer']}/{row['piper_synthetic_holdout']['cer']} | "
            f"{row['supertonic_heldout_voice_holdout']['wer']}/{row['supertonic_heldout_voice_holdout']['cer']} | "
            f"{row['fleurs_v2']['wer']}/{row['fleurs_v2']['cer']} | {row['artur_j']['wer']}/{row['artur_j']['cer']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Regression burden: {decision['scale200_burden']}",
            f"- Burden reduction versus replay reference: {decision['burden_reduction_percent']}%",
            f"- Classification: `{decision['classification']}`",
            "- Accepted parent: `none`",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(lines))


def stage_final_summarize(config_path: Path) -> dict[str, Any]:
    experiment, text_config, _augmentation = load_all_configs(config_path)
    auth = require_training_authorization(config_path)
    audio_report = read_json(_audio_report_json_path(experiment))
    training = read_json(_experiment_run_dir(experiment) / ARM_NAME / "training-summary.local.json")
    evaluation = read_json(_experiment_run_dir(experiment) / "directional-evaluation" / "summary.local.json")

    def public_suite_summary(suite: dict[str, Any]) -> dict[str, Any]:
        layout = suite["layout"]
        return {
            "rows": suite["rows"],
            "prediction_count": suite["prediction_count"],
            "audio_duration_seconds": suite["audio_duration_seconds"],
            "wall_time_seconds": suite["wall_time_seconds"],
            "real_time_factor": suite["real_time_factor"],
            "rows_per_second": suite["rows_per_second"],
            "audio_seconds_per_wall_second": suite["audio_seconds_per_wall_second"],
            "gpu_monitor": suite["gpu_monitor"],
            "layout": {
                "batch_size": layout["batch_size"],
                "bucketed": layout["bucketed"],
                "batch_count": layout["batch_count"],
                "full_batch_count": layout["full_batch_count"],
                "final_partial_batch_size": layout["final_partial_batch_size"],
                "actual_audio_seconds": layout["actual_audio_seconds"],
                "padded_audio_seconds": layout["padded_audio_seconds"],
                "padding_ratio": layout["padding_ratio"],
                "max_padded_batch_duration": layout["max_padded_batch_duration"],
            },
        }

    public = {
        "schema_version": "1.0",
        "experiment_id": experiment["experiment_id"],
        "work_order_id": experiment["work_order_id"],
        "status": "completed in PR; pending strategic review",
        "accepted_parent": "none",
        "canonical": False,
        "promotion_eligible": False,
        "authorization": {
            "status": auth["certificate"]["status"],
            "certificate_sha256": sha256_file(_experiment_certificate_path(experiment)),
            "audio_certificate_sha256": sha256_file(_audio_certificate_path(experiment)),
            "text_certificate_sha256": sha256_file(text_certificate_path()),
            "experiment_config_sha256": sha256_file(repo_path(config_path)),
        },
        "data": {
            "corpus_id": text_config["corpus_id"],
            "semantic_rows": audio_report["counts"]["semantic_rows"],
            "clean_files": audio_report["counts"]["clean_files"],
            "augmented_files": audio_report["counts"]["augmented_files"],
            "view_records": audio_report["counts"]["view_records"],
            "fixed_text_sha256": sha256_file(fixed_text_path(text_config)),
            "all_views_sha256": audio_report["hashes"]["all_views_sha256"],
            "schedule_sha256": audio_report["schedule"]["schedule_sha256"],
            "voice_counts": audio_report["voice_counts"],
            "profile_counts": audio_report["profile_counts"],
            "multiplier_table": multiplier_table(),
        },
        "model": _public_model_identity(experiment),
        "adapter": {
            "name": experiment["adapter"]["name"],
            "module": experiment["adapter"]["module"],
            "bottleneck_dimension": experiment["adapter"]["bottleneck_dimension"],
            "trainable_parameters": training["trainable_parameter_count"],
        },
        "training": {
            key: training[key]
            for key in (
                "status",
                "batch_size",
                "rounds",
                "sample_exposures",
                "optimizer_steps",
                "learning_rate",
                "initial_probe_loss",
                "final_probe_loss",
                "initial_full_clean_loss",
                "final_full_clean_loss",
                "full_loss_reduction_percent",
                "wall_time_seconds",
                "examples_per_second",
                "audio_seconds_per_wall_second",
                "padding_ratio",
                "gpu_monitor",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "base_integrity",
                "adapter_integrity",
                "restore_integrity",
                "exposure_counts_by_voice",
                "exposure_counts_by_profile",
            )
            if key in training
        },
        "directional_evaluation": {
            "suite_rows": evaluation["suite_rows"],
            "policy": evaluation["evaluation_policy"],
            "metric_table": evaluation["metric_table"],
            "decision": evaluation["decision"],
            "scale200_model_summary": {
                "checkpoint_sha256": evaluation["models"]["scale200_joint_adapter"]["checkpoint_sha256"],
                "suite": public_suite_summary(evaluation["models"]["scale200_joint_adapter"]["suite"]),
            },
        },
        "limitations": [
            "Directional batch-32 metrics are not canonical acceptance evidence.",
            "No batch-1 evaluation was run.",
            "All training remains synthetic.",
            "The 200x multiplier refers to exposure count, not independent linguistic information.",
            "No checkpoint or adapter is accepted as a parent.",
            "TRAINING_ELIGIBLE was not issued.",
        ],
    }
    _public_payload_safe(public)
    atomic_write_json(_experiment_report_json_path(experiment), public)
    _write_experiment_markdown(_experiment_report_markdown_path(experiment), public)
    result = {
        "status": "PASSED",
        "json_sha256": sha256_file(_experiment_report_json_path(experiment)),
        "markdown_sha256": sha256_file(_experiment_report_markdown_path(experiment)),
        "classification": public["directional_evaluation"]["decision"]["classification"],
        "accepted_parent": "none",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def require_text_accepted(config_path: Path) -> dict[str, Any]:
    _experiment, text_config, _augmentation = load_all_configs(config_path)
    certificate = text_certificate_path()
    if not certificate.exists():
        raise RuntimeError(
            "TEXT_ACCEPTED certificate is missing. Stop at the human checkpoint and run admit-text after an exact whole-file decision."
        )
    payload = read_json(certificate)
    if payload.get("status") != "TEXT_ACCEPTED":
        raise RuntimeError(f"text certificate status is not TEXT_ACCEPTED: {payload.get('status')}")
    if payload.get("fixed_text_sha256") != sha256_file(fixed_text_path(text_config)):
        raise RuntimeError("text certificate hash does not match fixed text")
    return payload


def stage_not_yet_safe(config_path: Path, stage: str) -> dict[str, Any]:
    require_text_accepted(config_path)
    raise RuntimeError(
        f"stage {stage} is intentionally fail-closed in this implementation pass. "
        "Implement and verify the audio/training/evaluation stage after the exact human text decision."
    )


def stage_summarize(config_path: Path) -> dict[str, Any]:
    experiment, text_config, _augmentation = load_all_configs(config_path)
    if (_experiment_run_dir(experiment) / "directional-evaluation" / "summary.local.json").exists():
        return stage_final_summarize(config_path)
    if _augmentation_paths(text_config)["validation"].exists():
        text_public = {}
        if text_certificate_path().exists():
            validator_report = read_json(run_dir(text_config) / "text-selection-summary.local.json") if (run_dir(text_config) / "text-selection-summary.local.json").exists() else {}
            text_public = write_text_public_reports(text_config, read_json(text_certificate_path()), validator_report=validator_report)
        audio_public = write_audio_and_training_authorization(config_path)
        result = {"status": "PASSED", "text": text_public, "audio": audio_public}
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    if text_certificate_path().exists():
        certificate = read_json(text_certificate_path())
    else:
        certificate = {
            "status": "DRAFT",
            "whole_file_decision": None,
            "limitations": [
                "Awaiting exact whole-file human decision before TTS.",
                "No audio, training, or evaluation stage has run.",
                "No TRAINING_ELIGIBLE status is issued.",
            ],
        }
    validator_report = read_json(run_dir(text_config) / "text-selection-summary.local.json") if (run_dir(text_config) / "text-selection-summary.local.json").exists() else {}
    public = write_text_public_reports(text_config, certificate, validator_report=validator_report)
    if fixed_text_path(text_config).exists():
        rows = load_jsonl(fixed_text_path(text_config))
        schedule, schedule_summary = build_exposure_schedule(rows, load_augmentation_config(repo_path("configs/augmentation/scale200_transcript_preserving_v1.json")))
        schedule_path = run_dir(text_config) / "exposure-schedule.preview.local.jsonl"
        atomic_write_jsonl(schedule_path, schedule)
        public["exposure_schedule_preview"] = {
            **schedule_summary,
            "schedule_sha256": sha256_file(schedule_path),
        }
    print(json.dumps(public, ensure_ascii=False, sort_keys=True))
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the scale-200 synthetic diagnostic stages.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "verify",
            "generate-text",
            "validate-text",
            "prepare-human-decision",
            "admit-text",
            "synthesize-piper",
            "synthesize-supertonic",
            "augment",
            "validate-audio",
            "train",
            "evaluate-directional",
            "verify-artifact",
            "summarize",
        ],
    )
    parser.add_argument("--whole-file-outcome", choices=sorted({"ACCEPT", "REJECT_GRAMMAR", "REJECT_SEMANTICS", "REJECT_UNNATURAL", "REJECT_TEMPLATE", "REJECT_METADATA_LEAK", "REJECT_DUPLICATE", "REJECT_DOMAIN", "REJECT_TRANSCRIPTION", "REVISE_AND_REREVIEW"}))
    parser.add_argument("--review-revision")
    parser.add_argument("--decision-id")
    parser.add_argument("--expected-corpus-sha256")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    try:
        if args.stage == "verify":
            stage_verify(args.config)
        elif args.stage == "generate-text":
            stage_generate_text(args.config)
        elif args.stage == "validate-text":
            stage_validate_text(args.config)
        elif args.stage == "prepare-human-decision":
            stage_prepare_human_decision(args.config)
        elif args.stage == "admit-text":
            missing = [
                name
                for name in ("whole_file_outcome", "review_revision", "decision_id", "expected_corpus_sha256", "expected_rows")
                if getattr(args, name) in {None, ""}
            ]
            if missing:
                raise ValueError(f"admit-text missing required arguments: {missing}")
            stage_admit_text(args.config, args)
        elif args.stage == "synthesize-piper":
            stage_synthesize_piper(args.config)
        elif args.stage == "synthesize-supertonic":
            stage_synthesize_supertonic(args.config, args.progress_interval_seconds)
        elif args.stage == "augment":
            stage_augment(args.config)
        elif args.stage == "validate-audio":
            stage_validate_audio(args.config)
        elif args.stage == "train":
            stage_train(args.config, args.progress_interval_seconds)
        elif args.stage == "evaluate-directional":
            stage_evaluate_directional(args.config)
        elif args.stage == "verify-artifact":
            experiment, _text_config, _augmentation = load_all_configs(args.config)
            result = verify_scale_artifact(experiment)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        elif args.stage == "summarize":
            stage_summarize(args.config)
        else:
            raise AssertionError(args.stage)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "stage": args.stage, "error": str(exc), "error_type": type(exc).__name__}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
