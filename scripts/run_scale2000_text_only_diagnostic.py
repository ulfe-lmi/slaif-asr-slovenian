#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, atomic_write_text, canonical_json_sha256, sha256_file
from slaif_asr.gams_retry_controller import AttemptRecord, AttemptTask, RetryState, initial_tasks, load_state, plan_refill_tasks, save_state
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.scale2000_corpus import (
    build_combined_rows,
    build_new_record,
    build_task_prompt,
    expand_whole_file_decision,
    fixed_combined_text_path,
    build_scale2000_exposure_schedule_from_views,
    classify_scale2000,
    generation_state_path,
    load_scale2000_experiment_config,
    load_scale2000_generation_config,
    new_addition_path,
    protected_config_fingerprints,
    retry_history_path,
    retry_limits_from_config,
    run_dir,
    scale2000_multiplier_table,
    SelectionShortfall,
    verify_inherited_rows,
    verify_prompt_cells_match_anchor,
    verify_scale200_report,
)
from slaif_asr.scale200_corpus import TRAINING_VIEWS, extract_utterance_lines, filter_records, load_augmentation_config, load_existing_holdout, protected_indexes, write_rejections
from slaif_asr.scale200_corpus import stable_sha256
from slaif_asr.transcript_preserving_augmentation import assignment_for, parameters_for_profile, render_augmented_file
from slaif_asr.corpus_v2_generation import GpuMonitor
from slaif_asr.batched_streaming import NvidiaSmiMonitor, parse_monitor_csv
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs/experiments/gams16000_scale2000_text_only_v1.json"
SUPERTONIC_SCALE2000_CONFIG = REPO_ROOT / "configs/tts/supertonic3_sl_scale2000_training_v1.json"
AUGMENTATION_RENDER_VERSION = "scale200-transcript-preserving-render-v2"
INHERITED_SCALE200_ROOT = REPO_ROOT / "runs/data-quality/sl-corpus-v3-gams-1600-training-v1"
EXPECTED_INHERITED_ALL_VIEWS_SHA256 = "c5232cd020dc1926e0732ff6bafab515385956dd6c9410cc144a37655330e775"
EXPECTED_INHERITED_AUDIO_CERTIFICATE_SHA256 = "eccdad5a200c46a6d1426ae0b7ea90e42695d2a41ff7e0c4ff670979d1293585"
ARM_NAME = "gams16000_nine_voice_augmented_joint_adapter_dim32"
FAST_DIRECTIONAL_CONFIG = REPO_ROOT / "configs/experiments/fast_batched_directional_replay_v1.json"

_JOINT_PATH = Path(__file__).with_name("run_corpus_v2_joint_adapter_diagnostic.py")
_JOINT: Any | None = None


def _joint_runner() -> Any:
    global _JOINT
    if _JOINT is None:
        spec = importlib.util.spec_from_file_location("_slaif_joint_runner_scale2000", _JOINT_PATH)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RuntimeError("cannot import joint-adapter runner")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _JOINT = module
    return _JOINT


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment = load_scale2000_experiment_config(path)
    text = load_scale2000_generation_config(REPO_ROOT / experiment["text_config"])
    augmentation = load_augmentation_config(REPO_ROOT / experiment["augmentation_config"])
    return experiment, text, augmentation


def stage_verify(config_path: Path) -> dict[str, Any]:
    experiment, text, augmentation = load_configs(config_path)
    verify_prompt_cells_match_anchor(text)
    inherited = verify_inherited_rows(text)
    reference = verify_scale200_report(
        experiment["directional_reference_report"]["path"],
        experiment["directional_reference_report"]["sha256"],
    )
    fingerprints = protected_config_fingerprints(experiment["protected_unchanged_configs"])
    limits = retry_limits_from_config(text)
    task_count = len(initial_tasks(sorted({cell["cell_id"] for cell in text["prompt_cells"]}), limits=limits))
    payload = {
        "status": "PASSED",
        "work_order_id": experiment["work_order_id"],
        "combined_corpus_id": text["corpus_id"],
        "new_addition_corpus_id": text["new_addition_corpus_id"],
        "inherited_rows": len(inherited),
        "prompt_cells": len(text["prompt_cells"]),
        "initial_shard_count": task_count,
        "initial_requested_rows": text["initial_requested_rows"],
        "maximum_requested_rows": text["maximum_requested_rows"],
        "multiplier_table": scale2000_multiplier_table(),
        "directional_reference": reference,
        "protected_config_fingerprints": fingerprints,
        "augmentation_config_sha256": canonical_json_sha256(augmentation),
        "generation_config_sha256": canonical_json_sha256(text),
        "experiment_config_sha256": canonical_json_sha256(experiment),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_prepare_human_decision(config_path: Path) -> dict[str, Any]:
    _experiment, text, _augmentation = load_configs(config_path)
    path = fixed_combined_text_path(text)
    if not path.exists():
        raise FileNotFoundError(path)
    rows_data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    corpus_hash = sha256_file(path)
    rows = len(rows_data)
    command = " ".join(
        [
            ".venv/bin/python",
            "scripts/run_scale2000_text_only_diagnostic.py",
            "--stage admit-text",
            "--whole-file-outcome ACCEPT",
            "--review-revision human-scale2000-review-v1",
            "--decision-id human-scale2000-decision-v1",
            f"--expected-corpus-sha256 {corpus_hash}",
            "--expected-rows 16000",
        ]
    )
    run_dir(text).mkdir(parents=True, exist_ok=True)
    tsv_buffer = io.StringIO()
    writer = csv.DictWriter(
        tsv_buffer,
        delimiter="\t",
        fieldnames=[
            "prompt_cell",
            "row_origin",
            "domain",
            "spoken_text",
            "target_text",
            "outcome",
            "review_revision",
            "reason_codes",
        ],
    )
    writer.writeheader()
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows_data:
        by_cell.setdefault(str(row.get("generation", {}).get("prompt_cell", "unknown")), []).append(row)
    for cell_id in sorted(by_cell):
        for row in sorted(by_cell[cell_id], key=lambda item: stable_sha256(str(item.get("candidate_id", "")))):
            candidate_id = str(row.get("candidate_id", ""))
            writer.writerow(
                {
                    "prompt_cell": cell_id,
                    "row_origin": "new" if candidate_id.startswith("gamsv4-") else "inherited_scale200",
                    "domain": row.get("domain", ""),
                    "spoken_text": row.get("spoken_text", ""),
                    "target_text": row.get("target_text", ""),
                    "outcome": "",
                    "review_revision": "",
                    "reason_codes": "",
                }
            )
    atomic_write_text(run_dir(text) / "review-capsule.local.tsv", tsv_buffer.getvalue())
    md_lines = [
        "# Corpus v4 16000 Whole-File Review Capsule",
        "",
        f"Corpus: `{text['corpus_id']}`",
        f"SHA256: `{corpus_hash}`",
        f"Rows: `{rows}`",
        "",
        "This capsule is local review evidence and is not committed.",
        "It distinguishes inherited scale-200 rows from newly generated scale-2000 rows.",
        "Do not approve if quality is mixed.",
        "",
    ]
    for cell_id in sorted(by_cell):
        cell_rows = sorted(by_cell[cell_id], key=lambda item: stable_sha256(str(item.get("candidate_id", ""))))
        inherited_count = sum(not str(row.get("candidate_id", "")).startswith("gamsv4-") for row in cell_rows)
        new_count = len(cell_rows) - inherited_count
        md_lines.extend([f"## {cell_id}", "", f"Inherited rows: {inherited_count}; new rows: {new_count}.", ""])
        for row in cell_rows:
            origin = "new" if str(row.get("candidate_id", "")).startswith("gamsv4-") else "inherited"
            md_lines.append(f"- [{origin}] {row.get('spoken_text', '')}")
        md_lines.append("")
    atomic_write_text(run_dir(text) / "review-capsule.local.md", "\n".join(md_lines))
    atomic_write_text(run_dir(text) / "whole-file-decision-command.local.txt", command + "\n")
    payload = {
        "status": "READY_FOR_HUMAN_DECISION",
        "decision": f"ACCEPT or REJECT {text['corpus_id']} {corpus_hash} {rows}",
        "combined_sha256": corpus_hash,
        "rows": rows,
        "review_capsule_markdown": str(run_dir(text) / "review-capsule.local.md"),
        "review_capsule_tsv": str(run_dir(text) / "review-capsule.local.tsv"),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def raw_generation_dir(config: dict[str, Any]) -> Path:
    return run_dir(config) / "raw-generation"


def prompt_path(config: dict[str, Any], task: AttemptTask) -> Path:
    return raw_generation_dir(config) / f"{task.attempt_id}.prompt.txt"


def raw_output_path(config: dict[str, Any], task: AttemptTask) -> Path:
    return raw_generation_dir(config) / f"{task.attempt_id}.txt"


def rejected_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "rejected.local.jsonl"


def generated_all_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "generated-all.local.jsonl"


def parse_task_output(config: dict[str, Any], task: AttemptTask, raw: str) -> tuple[list[dict[str, Any]], list[Any]]:
    lines, rejected = extract_utterance_lines(raw, cell_id=task.cell_id, attempt_id=task.attempt_id)
    cells = {str(cell["cell_id"]): cell for cell in config["prompt_cells"]}
    rows = [
        build_new_record(config=config, cell=cells[task.cell_id], task=task, output_ordinal=line.output_ordinal, text=line.text)
        for line in lines
    ]
    return rows, rejected


def load_completed_records(config: dict[str, Any], state: RetryState) -> tuple[list[dict[str, Any]], list[Any]]:
    records: list[dict[str, Any]] = []
    rejected: list[Any] = []
    for attempt_id, record in sorted(state.records.items()):
        if record.status != "completed":
            continue
        raw_path = raw_generation_dir(config) / f"{attempt_id}.txt"
        if not raw_path.exists():
            raise RuntimeError(f"completed attempt is missing raw output: {attempt_id}")
        rows, attempt_rejected = parse_task_output(config, record.task, raw_path.read_text(encoding="utf-8"))
        records.extend(rows)
        rejected.extend(attempt_rejected)
    return records, rejected


def filter_and_select(config: dict[str, Any], state: RetryState) -> tuple[dict[str, Any], dict[str, int]]:
    inherited = verify_inherited_rows(config)
    generated, parser_rejections = load_completed_records(config, state)
    atomic_write_jsonl(generated_all_path(config), generated)
    retained, rejected, filter_summary = filter_records(
        [*inherited, *generated],
        config=config,
        existing_rejections=parser_rejections,
        protected=protected_indexes(config),
        holdout_rows=load_existing_holdout(config),
    )
    new_retained = [row for row in retained if str(row.get("candidate_id", "")).startswith("gamsv4-")]
    write_rejections(rejected_path(config), rejected)
    per_cell = Counter(str(row.get("generation", {}).get("prompt_cell", "unknown")) for row in new_retained)
    target = int(config["new_rows_per_cell"]) + int(config.get("new_surplus_per_cell", 40))
    shortfalls = {
        str(cell["cell_id"]): target - per_cell[str(cell["cell_id"])]
        for cell in config["prompt_cells"]
        if per_cell[str(cell["cell_id"])] < target
    }
    summary: dict[str, Any] = {
        "generated_rows": len(generated),
        "new_admissible_rows": len(new_retained),
        "new_admissible_per_cell": dict(sorted(per_cell.items())),
        "filter_summary": filter_summary,
        "shortfalls": shortfalls,
        "retry_budget": state.budget_summary(retry_limits_from_config(config)),
    }
    if shortfalls:
        summary["status"] = "REFILL_REQUIRED"
    else:
        try:
            combined, selection = build_combined_rows(inherited, new_retained, config=config)
        except SelectionShortfall as exc:
            summary["status"] = "REFILL_REQUIRED"
            summary["selection_constraint_shortfalls"] = exc.shortfalls
            atomic_write_json(run_dir(config) / "text-generation-summary.local.json", summary)
            return summary, exc.shortfalls
        selected_new_ids = {str(row["candidate_id"]) for row in combined if str(row["candidate_id"]).startswith("gamsv4-")}
        selected_new = [row for row in new_retained if str(row["candidate_id"]) in selected_new_ids]
        atomic_write_jsonl(fixed_combined_text_path(config), combined)
        atomic_write_jsonl(new_addition_path(config), selected_new)
        summary.update(
            {
                "status": "STRUCTURALLY_READY_FOR_HUMAN_REVIEW",
                "combined_rows": len(combined),
                "new_selected_rows": len(selected_new),
                "combined_sha256": sha256_file(fixed_combined_text_path(config)),
                "new_addition_sha256": sha256_file(new_addition_path(config)),
                "selection": selection,
            }
        )
    atomic_write_json(run_dir(config) / "text-generation-summary.local.json", summary)
    return summary, shortfalls


def generate_task_batch(tokenizer: Any, model: Any, config: dict[str, Any], tasks: list[AttemptTask]) -> list[tuple[AttemptTask, str]]:
    from scripts.generate_gams_corpus_v2 import generate_batch

    outputs = generate_batch(
        tokenizer=tokenizer,
        model=model,
        prompts=[build_task_prompt(config, task) for task in tasks],
        seed=sum(task.seed for task in tasks) % (2**31 - 1),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
    )
    if len(outputs) != len(tasks):
        raise RuntimeError(f"output-count mismatch: {len(outputs)} outputs for {len(tasks)} tasks")
    return list(zip(tasks, outputs, strict=True))


def run_generation_tasks(config: dict[str, Any], tasks: list[AttemptTask], state: RetryState) -> None:
    from scripts.generate_gams_corpus_v2 import load_model

    if not tasks:
        return
    info = require_single_visible_cuda(allowed_name_fragments=("A100",))
    raw_generation_dir(config).mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model(config)
    batch_size = int(config["generation"]["prompt_batch_size"])
    fallback = int(config["generation"]["oom_fallback_batch_size"])
    index = 0
    with GpuMonitor(
        physical_selector=info.physical_selector,
        output_path=run_dir(config) / "gpu-monitor.local.csv",
        interval_seconds=float(config["generation"].get("monitor_interval_seconds", 0.2)),
    ):
        while index < len(tasks):
            batch = tasks[index : index + batch_size]
            try:
                outputs = generate_task_batch(tokenizer, model, config, batch)
            except RuntimeError as exc:
                lowered = str(exc).lower()
                if batch_size == int(config["generation"]["prompt_batch_size"]) and ("out of memory" in lowered or "cuda" in lowered):
                    batch_size = fallback
                    continue
                for task in batch:
                    state.record(AttemptRecord(task=task, status="failed", error=exc.__class__.__name__))
                save_state(generation_state_path(config), state)
                raise
            for task, raw in outputs:
                atomic_write_text(prompt_path(config, task), build_task_prompt(config, task))
                atomic_write_text(raw_output_path(config, task), raw)
                rows, rejected = parse_task_output(config, task, raw)
                state.record(
                    AttemptRecord(
                        task=task,
                        status="completed",
                        parsed_rows=len(rows),
                        rejection_counts=dict(Counter(item.reason for item in rejected)),
                    )
                )
            save_state(generation_state_path(config), state)
            index += len(batch)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "stage": "generate-text-until-valid",
                        "processed_attempts": state.total_attempts(),
                        "batch_size": len(batch),
                        "total_attempt_budget": retry_limits_from_config(config).max_total_attempts,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )


def stage_generate_text_until_valid(config_path: Path, *, restart_from_scratch: bool = False) -> dict[str, Any]:
    _experiment, text, _augmentation = load_configs(config_path)
    verify_prompt_cells_match_anchor(text)
    verify_inherited_rows(text)
    run_dir(text).mkdir(parents=True, exist_ok=True)
    raw_generation_dir(text).mkdir(parents=True, exist_ok=True)
    if restart_from_scratch and generation_state_path(text).exists():
        raise RuntimeError("restart-from-scratch is destructive and must be handled manually")
    state = load_state(generation_state_path(text))
    limits = retry_limits_from_config(text)
    cell_ids = sorted(str(cell["cell_id"]) for cell in text["prompt_cells"])
    initial = [task for task in initial_tasks(cell_ids, limits=limits) if task.attempt_id not in state.completed_attempt_ids]
    started = time.perf_counter()
    run_generation_tasks(text, initial, state)
    summary, shortfalls = filter_and_select(text, state)
    round_index = 1
    while shortfalls and (limits.max_verification_rounds is None or round_index <= limits.max_verification_rounds):
        tasks = plan_refill_tasks(
            shortfalls,
            verification_round=round_index,
            state=state,
            limits=limits,
            diversity_guidance=text.get("diversity_retry_guidance", ()),
        )
        if not tasks:
            break
        run_generation_tasks(text, tasks, state)
        summary, shortfalls = filter_and_select(text, state)
        with retry_history_path(text).open("a", encoding="utf-8") as fp:
            fp.write(
                json.dumps(
                    {
                        "verification_round": round_index,
                        "shortfalls": shortfalls,
                        "summary_status": summary["status"],
                        "retry_budget": state.budget_summary(limits),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        round_index += 1
    if shortfalls:
        raise RuntimeError(f"text generation ended before sufficient text: {shortfalls}")
    summary["wall_time_seconds"] = round(time.perf_counter() - started, 6)
    summary["retry_budget"] = state.budget_summary(limits)
    atomic_write_json(run_dir(text) / "text-generation-summary.local.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def stage_not_yet_safe(config_path: Path, stage: str) -> dict[str, Any]:
    experiment, text, augmentation = load_configs(config_path)
    run_dir(text).mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "NOT_RUN",
        "stage": stage,
        "reason": "This stage is implemented in later commit phases or requires prior whole-file text admission.",
        "work_order_id": experiment["work_order_id"],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_admit_text(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    _experiment, text, _augmentation = load_configs(config_path)
    payload = expand_whole_file_decision(
        text,
        outcome=args.whole_file_outcome,
        review_revision=args.review_revision,
        decision_id=args.decision_id,
        expected_corpus_sha256=args.expected_corpus_sha256,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def require_text_accepted(config_path: Path) -> dict[str, Any]:
    _experiment, text, _augmentation = load_configs(config_path)
    certificate_path = REPO_ROOT / text["public_certificates"]["text"]
    if not certificate_path.exists():
        raise RuntimeError("TEXT_ACCEPTED certificate is missing")
    certificate = read_json(certificate_path)
    if certificate.get("status") != "TEXT_ACCEPTED":
        raise RuntimeError(f"text certificate is not TEXT_ACCEPTED: {certificate.get('status')}")
    if certificate.get("fixed_text_sha256") != sha256_file(fixed_combined_text_path(text)):
        raise RuntimeError("text certificate hash does not match fixed combined text")
    return certificate


def _piper_new_audio_paths(text: dict[str, Any]) -> Any:
    from slaif_asr.acoustic_quality import AudioPaths

    root = run_dir(text) / "piper-new"
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


def _new_addition_tts_items(text: dict[str, Any]) -> list[Any]:
    from slaif_asr.acoustic_quality import CorpusV2TtsItem

    rows = [json.loads(line) for line in new_addition_path(text).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(text["new_rows"]):
        raise RuntimeError(f"expected {text['new_rows']} new rows, found {len(rows)}")
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


def _require_piper_gpu_selectors(selectors: tuple[str, ...]) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    available: dict[str, dict[str, Any]] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        available[parts[0]] = {"index": parts[0], "name": parts[1], "memory_total_mib": parts[2]}
    missing = [selector for selector in selectors if selector not in available]
    if missing:
        raise RuntimeError(f"requested Piper GPU selector(s) not present: {missing}")
    for selector in selectors:
        if "A100" not in available[selector]["name"]:
            raise RuntimeError(f"Piper selector {selector} is not an A100: {available[selector]['name']}")
    return {"selectors": list(selectors), "devices": {selector: available[selector] for selector in selectors}}


def _completed_piper_row(item: Any, tts_config: dict[str, Any], paths: Any) -> dict[str, Any] | None:
    from slaif_asr.data_quality import sha256_text
    from slaif_asr.tts import sox_version, validate_wav

    native_path = paths.native_dir / f"{item.candidate_id}.native.wav"
    final_path = paths.final_dir / f"{item.candidate_id}.wav"
    log_path = paths.log_dir / f"{item.candidate_id}.piper.log"
    if not final_path.exists() or not native_path.exists():
        return None
    native_info = validate_wav(native_path, sample_rate=int(tts_config["voice"]["native_sample_rate"]))
    final_info = validate_wav(final_path, sample_rate=int(tts_config["voice"]["final_asr_sample_rate"]))
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Failed to create CUDAExecutionProvider" in log_text or "CPUExecutionProvider" in log_text:
            raise RuntimeError(f"{item.candidate_id}: existing Piper log shows CPU fallback")
        if "Using CUDA" not in log_text and "CUDAExecutionProvider" not in log_text:
            raise RuntimeError(f"{item.candidate_id}: existing Piper log does not confirm CUDA execution")
    return {
        "schema_version": "1.0",
        "candidate_id": item.candidate_id,
        "audio_filepath": str(final_path.resolve()),
        "duration_seconds": round(final_info.duration_seconds, 6),
        "sample_rate": final_info.sample_rate,
        "channels": final_info.channels,
        "sample_width": final_info.sample_width,
        "text": item.target_text,
        "target_text_sha256": sha256_text(item.target_text),
        "language": item.language,
        "target_lang": item.language,
        "partition_role": item.partition_role,
        "source_type": "synthetic_tts",
        "source_id": item.source_id,
        "source_family_id": item.source_family_id,
        "utterance_family_id": item.utterance_family_id,
        "domain": item.domain,
        "phenomena": list(item.phenomena),
        "audio_sha256": final_info.sha256,
        "native_audio": {
            "path": str(native_path.resolve()),
            "sample_rate": native_info.sample_rate,
            "channels": native_info.channels,
            "sample_width": native_info.sample_width,
            "sha256": native_info.sha256,
        },
        "audio_validation": {
            "final_peak_ratio": round(final_info.peak_ratio, 6),
            "native_peak_ratio": round(native_info.peak_ratio, 6),
            "conversion": {
                "tool": "sox",
                "version": sox_version(),
                "parameters": ["-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer"],
            },
        },
        "tts": {
            "engine": tts_config["engine"]["repository_name"],
            "engine_revision": tts_config["engine"]["revision"],
            "engine_license": tts_config["engine"]["license"],
            "voice": tts_config["voice"]["name"],
            "voice_repository": tts_config["voice"]["repository"],
            "voice_revision": tts_config["voice"]["revision"],
            "execution_provider": tts_config["runtime"]["required_execution_provider"],
            "physical_gpu_selector": "resumed_existing",
        },
        "runtime": {
            "piper_wall_time_seconds": None,
            "log_sha256": sha256_file(log_path) if log_path.exists() else None,
            "resumed_existing_file": True,
        },
    }


def stage_synthesize_piper_new(config_path: Path) -> dict[str, Any]:
    from slaif_asr.acoustic_quality import GpuMonitor as PiperGpuMonitor
    from slaif_asr.acoustic_quality import monitor_summary, render_one_item
    from slaif_asr.tts import load_tts_config

    _experiment, text, _augmentation = load_configs(config_path)
    require_text_accepted(config_path)
    selector_text = os.environ.get("SCALE2000_PIPER_GPU_SELECTORS", "1")
    piper_gpu_selectors = tuple(selector.strip() for selector in selector_text.split(",") if selector.strip())
    if not piper_gpu_selectors:
        raise RuntimeError("SCALE2000_PIPER_GPU_SELECTORS must name at least one physical GPU selector")
    gpu = _require_piper_gpu_selectors(piper_gpu_selectors)
    paths = _piper_new_audio_paths(text)
    if paths.audio_manifest.exists():
        rows = [json.loads(line) for line in paths.audio_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing = [row.get("audio_filepath") for row in rows if not Path(str(row.get("audio_filepath", ""))).exists()]
        if len(rows) == int(text["new_rows"]) and not missing:
            summary = read_json(paths.synthesis_summary) if paths.synthesis_summary.exists() else {}
            summary["status"] = "PASSED"
            summary["reused_existing_manifest"] = True
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
        if missing:
            raise RuntimeError("existing Piper scale-2000 manifest references missing files")
    items = _new_addition_tts_items(text)
    started = time.perf_counter()
    tts_config = load_tts_config()
    rows_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    pending: list[Any] = []
    for item in items:
        existing = _completed_piper_row(item, tts_config, paths)
        if existing is None:
            pending.append(item)
        else:
            rows_by_id[item.candidate_id] = existing
    last_emit = started
    workers_per_gpu = int(os.environ.get("SCALE2000_PIPER_WORKERS_PER_GPU", "32"))
    worker_count = workers_per_gpu * len(piper_gpu_selectors)
    monitor_paths = {
        selector: paths.gpu_monitor.with_name(f"{paths.gpu_monitor.stem}.gpu{selector}{paths.gpu_monitor.suffix}")
        for selector in piper_gpu_selectors
    }
    with ExitStack() as stack:
        for selector in piper_gpu_selectors:
            stack.enter_context(PiperGpuMonitor(monitor_paths[selector], physical_selector=selector, interval_seconds=0.2))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {}
            for index, item in enumerate(pending):
                selector = piper_gpu_selectors[index % len(piper_gpu_selectors)]
                futures[
                    pool.submit(
                        render_one_item,
                        item=item,
                        tts_config=tts_config,
                        paths=paths,
                        output_root=None,
                        cuda_visible_devices=selector,
                    )
                ] = (item, selector)
            for future in as_completed(futures):
                item, selector = futures[future]
                try:
                    rows_by_id[item.candidate_id] = future.result()
                except Exception as exc:  # noqa: BLE001 - local failure evidence
                    failures.append({"candidate_id": item.candidate_id, "gpu_selector": selector, "reason": type(exc).__name__, "detail": str(exc)})
                processed = len(rows_by_id) + len(failures)
                now = time.perf_counter()
                if processed % 100 == 0 or now - last_emit >= 10.0:
                    elapsed = now - started
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "stage": "synthesize-piper-new",
                                "processed_rows": processed,
                                "total_rows": len(items),
                                "successful": len(rows_by_id),
                                "failed": len(failures),
                                "resumed_existing": len(items) - len(pending),
                                "worker_count": worker_count,
                                "gpu_selectors": list(piper_gpu_selectors),
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
    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    paths.audio_manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(paths.audio_manifest, rows)
    total_duration = sum(float(row["duration_seconds"]) for row in rows)
    wall = time.perf_counter() - started
    summary = {
        "schema_version": "1.0",
        "status": "PASSED" if not failures and len(rows) == int(text["new_rows"]) else "FAILED",
        "synthesis_version": "scale2000-piper-new-synthesis-v1",
        "manual_runtime_override": "user authorized parallel TTS scheduling; Piper was pinned to configured selectors while Supertonic used GPU0",
        "gpu_selectors": list(piper_gpu_selectors),
        "workers_per_gpu": workers_per_gpu,
        "selected_worker_count": worker_count,
        "requested": len(items),
        "resumed_existing": len(items) - len(pending),
        "successful": len(rows),
        "failed": len(failures),
        "failures": failures[:200],
        "wall_time_seconds": round(wall, 6),
        "utterances_per_minute": round((len(rows) / wall) * 60.0, 6) if wall else None,
        "audio_seconds_per_wall_second": round(total_duration / wall, 6) if wall else None,
        "total_audio_duration_seconds": round(total_duration, 6),
        "audio_manifest_sha256": sha256_file(paths.audio_manifest),
        "new_addition_sha256": sha256_file(new_addition_path(text)),
        "monitor": {selector: monitor_summary(path) for selector, path in monitor_paths.items()},
        "gpu": gpu,
    }
    atomic_write_json(paths.synthesis_summary, summary)
    if summary["status"] != "PASSED":
        raise RuntimeError(f"Piper new synthesis failed: {len(failures)} failures")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def stage_synthesize_supertonic_new(config_path: Path, progress_interval_seconds: float = 5.0) -> dict[str, Any]:
    from slaif_asr.supertonic3_tts import load_supertonic_config, synthesize_batched_supertonic_audio

    require_text_accepted(config_path)
    config = load_supertonic_config(SUPERTONIC_SCALE2000_CONFIG)
    summary = synthesize_batched_supertonic_audio(config, progress_interval_seconds=progress_interval_seconds)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def _data_paths(text: dict[str, Any]) -> dict[str, Path]:
    root = run_dir(text)
    augmentation_root = root / "augmentation-new"
    return {
        "root": root,
        "augmentation_root": augmentation_root,
        "augmentation_audio": augmentation_root / "final-16000",
        "augmentation_manifest": augmentation_root / "augmentation-manifest.local.jsonl",
        "augmentation_summary": augmentation_root / "augmentation-summary.local.json",
        "validation": root / "audio-validation.local.json",
        "all_views": root / "all-views.local.jsonl",
        "exposure_schedule": root / "exposure-schedule.local.jsonl",
        "audio_certificate_local": root / "scale2000-audio-certificate.local.json",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_inherited_audio_artifacts() -> list[dict[str, Any]]:
    all_views = INHERITED_SCALE200_ROOT / "all-views.local.jsonl"
    public_certificate = REPO_ROOT / "docs/data-certificates/sl-corpus-v3-nine-voice-augmented-audio-v1.json"
    if sha256_file(all_views) != EXPECTED_INHERITED_ALL_VIEWS_SHA256:
        raise RuntimeError("inherited scale-200 all-view artifact SHA mismatch")
    if sha256_file(public_certificate) != EXPECTED_INHERITED_AUDIO_CERTIFICATE_SHA256:
        raise RuntimeError("inherited scale-200 public audio certificate SHA mismatch")
    rows = _load_jsonl(all_views)
    if len(rows) != 32000:
        raise RuntimeError(f"expected 32000 inherited view rows, found {len(rows)}")
    return rows


def _supertonic_new_manifest_path() -> Path:
    return REPO_ROOT / "runs/data-quality/sl-corpus-v4-gams-16000-training-v1/supertonic-new/training-audio-manifest.local.jsonl"


def _load_new_clean_audio_indexes(text: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    piper_rows = _load_jsonl(_piper_new_audio_paths(text).audio_manifest)
    supertonic_rows = _load_jsonl(_supertonic_new_manifest_path())
    if len(piper_rows) != int(text["new_rows"]):
        raise RuntimeError(f"expected {text['new_rows']} new Piper rows, found {len(piper_rows)}")
    if len(supertonic_rows) != int(text["new_rows"]) * 8:
        raise RuntimeError(f"expected {int(text['new_rows']) * 8} new Supertonic rows, found {len(supertonic_rows)}")
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
        voice = f"supertonic-{row['voice_style_id']}"
        normalized = {
            **row,
            "semantic_key": semantic_key,
            "view_id": voice,
            "engine": "supertonic-3",
            "voice_style_id": voice,
            "profile_id": "clean",
            "view_type": "clean",
            "source_audio_filepath": row["audio_filepath"],
            "source_audio_sha256": row["audio_sha256"],
        }
        clean_by_key[f"{semantic_key}:{voice}"] = normalized
        clean_rows.append(normalized)
    return clean_by_key, clean_rows


def _build_new_augmentation_tasks(text: dict[str, Any], augmentation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(_load_jsonl(new_addition_path(text)), key=lambda row: stable_sha256(str(row["candidate_id"])))
    paths = _data_paths(text)
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
            output = paths["augmentation_audio"] / profile_id / assignment.source_voice / f"{semantic_key}.{profile_id}.{assignment.source_voice}.wav"
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
    expected = int(text["new_rows"]) * 11
    if len(tasks) != expected:
        raise RuntimeError(f"expected {expected} augmentation tasks, got {len(tasks)}")
    return tasks


def _completed_augmented_row(task: dict[str, Any], source_row: dict[str, Any], augmentation_sha: str) -> dict[str, Any] | None:
    output = Path(task["output_path"])
    if not output.exists():
        return None
    from slaif_asr.acoustic_quality import read_audio_stats

    stats = read_audio_stats(output)
    row = task["row"]
    return {
        "schema_version": "1.0",
        "view_type": "augmented",
        "partition_role": "selected_training",
        "semantic_key": task["semantic_key"],
        "semantic_position": task["semantic_position"],
        "profile_id": task["profile_id"],
        "source_voice": task["source_voice"],
        "voice": task["source_voice"],
        "source_audio_filepath": source_row["source_audio_filepath"],
        "source_audio_sha256": source_row["source_audio_sha256"],
        "audio_filepath": str(output.resolve()),
        "audio_sha256": stats.sha256,
        "target_text_sha256": row.get("target_text_sha256") or source_row.get("target_text_sha256"),
        "source_family_id": row["source_family_id"],
        "utterance_family_id": row["utterance_family_id"],
        "source_id": row["source_id"],
        "domain": row.get("domain"),
        "phenomena": row.get("phenomena", []),
        "parameters": task["parameters"],
        "transform_details": {"resumed_existing_file": True, "output_audio_sha256": stats.sha256},
        "augmentation_policy_sha256": augmentation_sha,
        "augmentation_render_version": AUGMENTATION_RENDER_VERSION,
    }


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
        "voice": task["source_voice"],
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


def stage_augment_new(config_path: Path) -> dict[str, Any]:
    _experiment, text, augmentation = load_configs(config_path)
    require_text_accepted(config_path)
    paths = _data_paths(text)
    expected = int(text["new_rows"]) * 11
    if paths["augmentation_manifest"].exists():
        rows = _load_jsonl(paths["augmentation_manifest"])
        if len(rows) == expected and all(row.get("augmentation_render_version") == AUGMENTATION_RENDER_VERSION for row in rows):
            summary = read_json(paths["augmentation_summary"]) if paths["augmentation_summary"].exists() else {}
            summary["status"] = "PASSED"
            summary["reused_existing_manifest"] = True
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
    clean_by_key, _clean_rows = _load_new_clean_audio_indexes(text)
    tasks = _build_new_augmentation_tasks(text, augmentation)
    augmentation_sha = sha256_file(REPO_ROOT / "configs/augmentation/scale200_transcript_preserving_v1.json")
    workers = min(32, os.cpu_count() or 1)
    started = time.perf_counter()
    last_emit = started
    output_rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task in tasks:
        source = clean_by_key.get(f"{task['semantic_key']}:{task['source_voice']}")
        if source is None:
            failures.append({"reason": "missing_source_voice", "semantic_hash": stable_sha256(str(task["semantic_key"])), "voice": task["source_voice"]})
            continue
        existing = _completed_augmented_row(task, source, augmentation_sha)
        key = (str(task["semantic_key"]), str(task["profile_id"]), str(task["source_voice"]))
        if existing is None:
            pending.append((task, source))
        else:
            output_rows_by_key[key] = existing
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_augment_one, task, source, augmentation_sha): task for task, source in pending}
        for future in as_completed(futures):
            task = futures[future]
            key = (str(task["semantic_key"]), str(task["profile_id"]), str(task["source_voice"]))
            try:
                output_rows_by_key[key] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"reason": type(exc).__name__, "semantic_hash": stable_sha256(str(task["semantic_key"])), "profile_id": task["profile_id"]})
            processed = len(output_rows_by_key) + len(failures)
            now = time.perf_counter()
            if processed % 500 == 0 or now - last_emit >= 10.0:
                elapsed = now - started
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "stage": "augment-new",
                            "processed_rows": processed,
                            "total_rows": len(tasks),
                            "successful": len(output_rows_by_key),
                            "failed": len(failures),
                            "resumed_existing": len(tasks) - len(pending),
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
    output_rows = [output_rows_by_key[key] for key in sorted(output_rows_by_key)]
    atomic_write_jsonl(paths["augmentation_manifest"], output_rows)
    wall = time.perf_counter() - started
    summary = {
        "schema_version": "1.0",
        "augmentation_render_version": AUGMENTATION_RENDER_VERSION,
        "status": "PASSED" if not failures and len(output_rows) == expected else "FAILED",
        "requested": len(tasks),
        "generated": len(output_rows),
        "resumed_existing": len(tasks) - len(pending),
        "failed": len(failures),
        "failures": failures[:500],
        "workers": workers,
        "wall_time_seconds": round(wall, 6),
        "items_per_second": round(len(output_rows) / wall, 6) if wall else None,
        "manifest_sha256": sha256_file(paths["augmentation_manifest"]),
        "profile_counts": dict(sorted(Counter(str(row["profile_id"]) for row in output_rows).items())),
        "source_voice_counts": dict(sorted(Counter(str(row["source_voice"]) for row in output_rows).items())),
    }
    atomic_write_json(paths["augmentation_summary"], summary)
    if summary["status"] != "PASSED":
        raise RuntimeError(f"augmentation failed: {summary['failed']} failures")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def _normalize_view_row(row: dict[str, Any], *, view_type: str, inherited: bool) -> dict[str, Any]:
    semantic_key = str(row.get("semantic_key") or row.get("candidate_id") or row.get("source_key"))
    voice = str(row.get("voice") or row.get("view_id") or row.get("voice_style_id") or row.get("source_voice") or "piper-sl_SI-artur-medium")
    if voice in {"M1", "M2", "M3", "M4", "F1", "F2", "F3", "F4"}:
        voice = f"supertonic-{voice}"
    return {
        "view_type": view_type,
        "origin": "inherited_scale200" if inherited else "new_scale2000",
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
    from slaif_asr.acoustic_quality import read_audio_stats

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


def stage_validate_combined_audio(config_path: Path) -> dict[str, Any]:
    _experiment, text, augmentation = load_configs(config_path)
    require_text_accepted(config_path)
    paths = _data_paths(text)
    inherited_rows = _verify_inherited_audio_artifacts()
    _clean_by_key, new_clean_rows = _load_new_clean_audio_indexes(text)
    new_augmented_rows = _load_jsonl(paths["augmentation_manifest"])
    if len(new_clean_rows) != 129600:
        raise RuntimeError(f"expected 129600 new clean rows, found {len(new_clean_rows)}")
    if len(new_augmented_rows) != 158400:
        raise RuntimeError(f"expected 158400 new augmented rows, found {len(new_augmented_rows)}")
    views = [
        _normalize_view_row(row, view_type=str(row.get("view_type", "clean")), inherited=True)
        for row in inherited_rows
    ]
    views.extend(_normalize_view_row(row, view_type="clean", inherited=False) for row in new_clean_rows)
    views.extend(_normalize_view_row(row, view_type="augmented", inherited=False) for row in new_augmented_rows)
    if len(views) != 320000:
        raise RuntimeError(f"combined view count mismatch: {len(views)}")
    started = time.perf_counter()
    last_emit = started
    workers = min(32, os.cpu_count() or 1)
    stats_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_stat_view, row): row for row in views}
        for future in as_completed(futures):
            row = futures[future]
            try:
                stats_rows.append(future.result())
            except Exception as exc:  # noqa: BLE001
                issues.append({"reason": type(exc).__name__, "semantic_hash": stable_sha256(str(row["semantic_key"]))})
            processed = len(stats_rows) + len(issues)
            now = time.perf_counter()
            if processed % 500 == 0 or now - last_emit >= 10.0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "stage": "validate-combined-audio",
                            "processed_rows": processed,
                            "total_rows": len(views),
                            "issues": len(issues),
                            "elapsed_seconds": round(now - started, 6),
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
    if len(semantic_seen) != 16000 or any(count != 20 for count in semantic_seen.values()):
        issues.append({"reason": "semantic_view_count", "semantic_rows": len(semantic_seen)})
    schedule_summary_payload = _write_view_aware_schedule(text, augmentation, stats_rows)
    sorted_stats = sorted(stats_rows, key=lambda row: (str(row["semantic_key"]), str(row["view_type"]), str(row["voice"]), str(row["profile_id"])))
    atomic_write_jsonl(paths["all_views"], sorted_stats)
    validation = {
        "schema_version": "1.0",
        "validator": "scale2000-audio-validator-v1",
        "status": "AUDIO_ACCEPTED" if not issues else "AUDIO_REJECTED",
        "semantic_rows": len(semantic_seen),
        "view_records": len(stats_rows),
        "clean_files": sum(1 for row in stats_rows if row["view_type"] == "clean"),
        "augmented_files": sum(1 for row in stats_rows if row["view_type"] == "augmented"),
        "inherited_view_records": sum(1 for row in stats_rows if row["origin"] == "inherited_scale200"),
        "new_view_records": sum(1 for row in stats_rows if row["origin"] == "new_scale2000"),
        "profile_counts": dict(sorted(Counter(str(row["profile_id"]) for row in stats_rows if row["view_type"] == "augmented").items())),
        "voice_counts": dict(sorted(Counter(str(row["voice"]) for row in stats_rows).items())),
        "duration_seconds": {
            "clean": round(sum(float(row["duration_seconds"]) for row in stats_rows if row["view_type"] == "clean"), 6),
            "augmented": round(sum(float(row["duration_seconds"]) for row in stats_rows if row["view_type"] == "augmented"), 6),
        },
        "duplicate_paths": duplicate_paths,
        "duplicate_hashes": duplicate_hashes,
        "issues_by_reason": dict(sorted(Counter(str(issue["reason"]) for issue in issues).items())),
        "issues": issues[:500],
        "schedule": schedule_summary_payload,
        "hashes": {
            "inherited_all_views_sha256": sha256_file(INHERITED_SCALE200_ROOT / "all-views.local.jsonl"),
            "piper_new_audio_manifest_sha256": sha256_file(_piper_new_audio_paths(text).audio_manifest),
            "supertonic_new_audio_manifest_sha256": sha256_file(_supertonic_new_manifest_path()),
            "augmentation_new_manifest_sha256": sha256_file(paths["augmentation_manifest"]),
            "all_views_sha256": sha256_file(paths["all_views"]),
        },
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "limitations": [
            "All audio is synthetic.",
            "The 2000x multiplier is exposure count, not independent linguistic information.",
            "No TRAINING_ELIGIBLE status is issued.",
        ],
    }
    atomic_write_json(paths["validation"], validation)
    atomic_write_json(
        paths["audio_certificate_local"],
        {
            "schema_version": "1.0",
            "certificate_id": "sl-corpus-v4-scale2000-audio-v1-local",
            "status": validation["status"],
            "corpus_id": text["corpus_id"],
            "text_sha256": sha256_file(fixed_combined_text_path(text)),
            "view_records": len(stats_rows),
            "clean_files": validation["clean_files"],
            "augmented_files": validation["augmented_files"],
            "schedule_sha256": validation["schedule"]["schedule_sha256"],
            "all_views_sha256": validation["hashes"]["all_views_sha256"],
        },
    )
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    if validation["status"] != "AUDIO_ACCEPTED":
        raise RuntimeError(f"audio validation failed: {validation['issues_by_reason']}")
    return validation


def _experiment_run_dir(experiment: dict[str, Any]) -> Path:
    return REPO_ROOT / experiment["local_outputs"]["run_root"]


def _public_output_path(experiment: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / experiment["public_outputs"][key]


def require_audio_accepted(config_path: Path) -> dict[str, Any]:
    _experiment, text, _augmentation = load_configs(config_path)
    validation_path = _data_paths(text)["validation"]
    if not validation_path.exists():
        raise RuntimeError("scale-2000 audio validation has not run")
    validation = read_json(validation_path)
    if validation.get("status") != "AUDIO_ACCEPTED":
        raise RuntimeError(f"scale-2000 audio is not AUDIO_ACCEPTED: {validation.get('status')}")
    if int(validation.get("view_records", 0)) != 320000:
        raise RuntimeError("scale-2000 audio validation view count mismatch")
    return validation


def require_nemotron_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Nemotron stages must run with CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("Nemotron stages must run with NVIDIA_TF32_OVERRIDE=0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def configure_torch() -> Any:
    return _joint_runner().configure_torch()


def restore_base_model(experiment: dict[str, Any], *, reporter: LiveProgressReporter | None = None) -> Any:
    return _joint_runner().restore_base_model(experiment, reporter=reporter)


def prepare_adapter_model(model: Any, experiment: dict[str, Any], *, enable: bool) -> dict[str, Any]:
    return _joint_runner().prepare_adapter_model(model, experiment, enable=enable)


def _load_text_by_id(text: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(fixed_combined_text_path(text))
    if len(rows) != 16000:
        raise RuntimeError(f"expected 16000 fixed text rows, found {len(rows)}")
    return {str(row["candidate_id"]): row for row in rows}


def _view_lookup(text: dict[str, Any]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    rows = _load_jsonl(_data_paths(text)["all_views"])
    if len(rows) != 320000:
        raise RuntimeError(f"expected 320000 all-view rows, found {len(rows)}")
    lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["semantic_key"]), str(row["view_type"]), str(row["voice"]), str(row["profile_id"]))
        if key in lookup:
            raise RuntimeError(f"duplicate view key: {key}")
        lookup[key] = row
    return lookup


def _write_view_aware_schedule(text: dict[str, Any], augmentation: dict[str, Any], view_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    paths = _data_paths(text)
    if view_rows is None:
        view_rows = _load_jsonl(paths["all_views"])
    schedule, schedule_summary = build_scale2000_exposure_schedule_from_views(
        _load_jsonl(fixed_combined_text_path(text)),
        augmentation,
        view_rows,
    )
    atomic_write_jsonl(paths["exposure_schedule"], schedule)
    schedule_payload = {**schedule_summary, "schedule_sha256": sha256_file(paths["exposure_schedule"])}
    if paths["validation"].exists():
        validation = read_json(paths["validation"])
        validation["schedule"] = schedule_payload
        validation.setdefault("hashes", {})["all_views_sha256"] = sha256_file(paths["all_views"])
        atomic_write_json(paths["validation"], validation)
    if paths["audio_certificate_local"].exists():
        certificate = read_json(paths["audio_certificate_local"])
        certificate["schedule_sha256"] = schedule_payload["schedule_sha256"]
        certificate["all_views_sha256"] = sha256_file(paths["all_views"])
        atomic_write_json(paths["audio_certificate_local"], certificate)
    return schedule_payload


def _training_record_from_view(text_row: dict[str, Any], view_row: dict[str, Any], *, selection_reason: str) -> Any:
    from slaif_asr.corpus_v2_training import TrainingRecord

    path = Path(str(view_row["audio_filepath"]))
    if not path.exists():
        raise FileNotFoundError(path)
    if str(view_row["target_text_sha256"]) != stable_sha256(str(text_row["target_text"])):
        raise RuntimeError("scale-2000 text/audio text-hash mismatch")
    return TrainingRecord(
        selected_training_id=str(text_row["candidate_id"]),
        audio_filepath=str(path),
        duration=float(view_row["duration_seconds"]),
        text=str(text_row["target_text"]),
        text_sha256=str(view_row["target_text_sha256"]),
        audio_sha256=str(view_row["audio_sha256"]),
        selection_reason=selection_reason,
        selection_rank=int(text_row["generation"]["prompt_cell"].removeprefix("cell")) if str(text_row["generation"]["prompt_cell"]).startswith("cell") else 0,
    )


def _probe_records(text: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    from slaif_asr.corpus_v2_training import select_probe_records

    text_by_id = _load_text_by_id(text)
    views = _view_lookup(text)
    clean_piper_records = []
    inherited_clean_piper_records = []
    for semantic_key in sorted(text_by_id):
        view = views[(semantic_key, "clean", "piper-sl_SI-artur-medium", "clean")]
        record = _training_record_from_view(text_by_id[semantic_key], view, selection_reason="scale2000_clean_probe")
        clean_piper_records.append(record)
        if semantic_key.startswith("gamsv3-"):
            inherited_clean_piper_records.append(record)
    if len(inherited_clean_piper_records) != 1600:
        raise RuntimeError("anchor probe could not find exactly 1600 inherited clean Piper rows")
    return (
        select_probe_records(inherited_clean_piper_records, 32),
        select_probe_records(clean_piper_records, 320),
        inherited_clean_piper_records,
        clean_piper_records,
    )


def _load_scheduled_round_records(text: dict[str, Any], augmentation: dict[str, Any]) -> tuple[dict[int, list[Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    paths = _data_paths(text)
    text_by_id = _load_text_by_id(text)
    views = _view_lookup(text)
    schedule = _load_jsonl(paths["exposure_schedule"])
    if len(schedule) != 320000:
        raise RuntimeError(f"expected 320000 exposure schedule rows, found {len(schedule)}")
    missing_keys = []
    for item in schedule:
        key = (str(item["semantic_key"]), str(item["view_type"]), str(item["voice"]), str(item["profile_id"]))
        if key not in views:
            missing_keys.append(key)
            if len(missing_keys) >= 5:
                break
    if missing_keys:
        _write_view_aware_schedule(text, augmentation)
        schedule = _load_jsonl(paths["exposure_schedule"])
    rounds: dict[int, list[Any]] = defaultdict(list)
    meta_by_audio: dict[str, dict[str, Any]] = {}
    for item in schedule:
        semantic_key = str(item["semantic_key"])
        key = (semantic_key, str(item["view_type"]), str(item["voice"]), str(item["profile_id"]))
        view = views[key]
        record = _training_record_from_view(text_by_id[semantic_key], view, selection_reason="scale2000")
        rounds[int(item["round"])].append(record)
        meta_by_audio[record.audio_filepath] = {
            "voice": item["voice"],
            "profile_id": item["profile_id"],
            "view_type": item["view_type"],
            "spec_augment": bool(item.get("spec_augment", False)),
        }
    for round_index in range(1, 21):
        rows = rounds.get(round_index, [])
        if len(rows) != 16000:
            raise RuntimeError(f"round {round_index} has {len(rows)} rows, expected 16000")
    return rounds, meta_by_audio, {"schedule_sha256": sha256_file(paths["exposure_schedule"])}


def verify_scale2000_artifact(experiment: dict[str, Any]) -> dict[str, Any]:
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

    experiment, text, augmentation = load_configs(config_path)
    require_audio_accepted(config_path)
    require_nemotron_env()
    runtime = verify_runtime_identities(check_gpu=True)
    arm_dir = _experiment_run_dir(experiment) / ARM_NAME
    summary_path = arm_dir / "training-summary.local.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary.get("status") == "PASSED" and int(summary.get("optimizer_steps", -1)) == 40000:
            print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
            return summary
    torch = configure_torch()
    anchor_probe, scale_probe, anchor_full, scale_full = _probe_records(text)
    rounds, meta_by_audio, schedule_summary = _load_scheduled_round_records(text, augmentation)
    reporter = LiveProgressReporter(stage="train", arm=ARM_NAME, ndjson_path=arm_dir / "progress" / "train.local.ndjson")
    reporter.start("training scale-2000 joint adapter")
    model = restore_base_model(experiment, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=arm_dir / "progress" / "restore.local.ndjson"))
    model.eval()
    adapter_summary = prepare_adapter_model(model, experiment, enable=True)
    initial_state = state_dict_cpu(model)
    prompt_selection = derive_prompt_column_selection(model, "sl-SI")
    if adapter_summary["trainable_parameters"] != expected_trainable_count(adapter_summary["joint_hidden"], 32):
        raise RuntimeError("scale-2000 joint-adapter trainable parameter count mismatch")
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=float(experiment["training"]["learning_rate"]), weight_decay=0.0)
    verify_optimizer_scope(optimizer, model)
    joint_runner = _joint_runner()
    initial_anchor = joint_runner.mean_loss(model, prompt_selection.prompt_index, anchor_probe, device="cuda")
    initial_scale = joint_runner.mean_loss(model, prompt_selection.prompt_index, scale_probe, device="cuda")
    initial_anchor_full = joint_runner.mean_loss(model, prompt_selection.prompt_index, anchor_full, device="cuda")
    initial_scale_full = joint_runner.mean_loss(model, prompt_selection.prompt_index, scale_full, device="cuda")
    anchor_curve = [{"round": 0, "mean_loss": round(initial_anchor, 6)}]
    scale_curve = [{"round": 0, "mean_loss": round(initial_scale, 6)}]
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
                    raise RuntimeError("non-finite scale-2000 training loss")
                loss.backward()
                grad_norm, grads_ok = joint_runner.finite_grad_norm(adapter_parameters(model))
                if not grads_ok:
                    raise RuntimeError("non-finite scale-2000 training gradient")
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
                rolling_losses = rolling_losses[-20:]
                grad_norms.append(grad_norm)
                if optimizer_steps % 100 == 0:
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
            anchor_loss = joint_runner.mean_loss(model, prompt_selection.prompt_index, anchor_probe, device="cuda")
            scale_loss = joint_runner.mean_loss(model, prompt_selection.prompt_index, scale_probe, device="cuda")
            anchor_curve.append({"round": round_index, "mean_loss": round(anchor_loss, 6)})
            scale_curve.append({"round": round_index, "mean_loss": round(scale_loss, 6)})
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
        final_anchor = joint_runner.mean_loss(model, prompt_selection.prompt_index, anchor_probe, device="cuda")
        final_scale = joint_runner.mean_loss(model, prompt_selection.prompt_index, scale_probe, device="cuda")
        final_anchor_full = joint_runner.mean_loss(model, prompt_selection.prompt_index, anchor_full, device="cuda")
        final_scale_full = joint_runner.mean_loss(model, prompt_selection.prompt_index, scale_full, device="cuda")
        trained_state = state_dict_cpu(model)
        base_integrity = compare_base_state(initial_state, trained_state)
        adapter_integrity = compare_adapter_state(initial_state, trained_state)
        if not base_integrity["base_tensors_identical"]:
            raise RuntimeError("pretrained tensor changed during scale-2000 joint-adapter training")
        artifact_path = arm_dir / "artifacts" / "sl-si-joint-adapter-v1.pt"
        artifact_sha = save_adapter_artifact(
            artifact_path,
            model=model,
            spec=load_adapter_spec(experiment["adapter"]["config"]),
            metadata={
                "base_checkpoint_sha256": CHECKPOINT_SHA256,
                "nemo_revision": NEMO_REVISION,
                "fixed_text_sha256": sha256_file(fixed_combined_text_path(text)),
                "all_views_sha256": sha256_file(_data_paths(text)["all_views"]),
                "exposure_schedule_sha256": schedule_summary["schedule_sha256"],
                "experiment_config_sha256": sha256_file(REPO_ROOT / config_path),
                "adapter_config_sha256": sha256_file(REPO_ROOT / experiment["adapter"]["config"]),
            },
        )
        verify_command = [sys.executable, "-u", __file__, "--config", str(config_path), "--stage", "verify-artifact"]
        completed = subprocess.run(verify_command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
        (arm_dir / "verify-artifact.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError("scale-2000 adapter artifact restore verification failed")
        joint_runner.enable_for_target_language(model, "sl-SI")
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
        "initial_anchor_probe_loss": round(initial_anchor, 6),
        "final_anchor_probe_loss": round(final_anchor, 6),
        "initial_scale_probe_loss": round(initial_scale, 6),
        "final_scale_probe_loss": round(final_scale, 6),
        "initial_anchor_full_loss": round(initial_anchor_full, 6),
        "final_anchor_full_loss": round(final_anchor_full, 6),
        "initial_scale_full_loss": round(initial_scale_full, 6),
        "final_scale_full_loss": round(final_scale_full, 6),
        "anchor_probe_curve": anchor_curve,
        "scale_probe_curve": scale_curve,
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


def stage_evaluate_directional(config_path: Path) -> dict[str, Any]:
    from slaif_asr.corpus_v2_scoring import verify_runtime_identities
    from slaif_asr.directional_evaluation import (
        DirectionalModel,
        load_directional_suite,
        run_directional_model,
        suite_plan_hash,
        write_privacy_safe_suite_manifest,
    )

    experiment, _text, _augmentation = load_configs(config_path)
    require_audio_accepted(config_path)
    require_nemotron_env()
    verify_runtime_identities(check_gpu=True)
    checkpoint = _experiment_run_dir(experiment) / ARM_NAME / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
    if not checkpoint.exists():
        raise RuntimeError("scale-2000 adapter checkpoint is missing")
    reference_report_info = experiment["directional_reference_report"]
    reference_report_path = REPO_ROOT / reference_report_info["path"]
    if sha256_file(reference_report_path) != reference_report_info["sha256"]:
        raise RuntimeError("Experiment 0013 directional reference report SHA mismatch")
    reference_report = read_json(reference_report_path)
    fast_config = read_json(FAST_DIRECTIONAL_CONFIG)
    suite_records, split_records = load_directional_suite(fast_config)
    output_dir = _experiment_run_dir(experiment) / "directional-evaluation"
    suite_manifest_sha = write_privacy_safe_suite_manifest(output_dir / "suite-plan.local.jsonl", suite_records)
    model = DirectionalModel("scale2000_joint_adapter", checkpoint, sha256_file(checkpoint), "work_order_0026")
    summary = run_directional_model(
        config=fast_config,
        model=model,
        suite_records=suite_records,
        split_records=split_records,
        run_dir=output_dir,
        python_executable=Path(sys.executable),
    )
    scale2000_metrics = {split_name: _metric_row(split) for split_name, split in summary["splits"].items()}
    metric_table = dict(reference_report["directional_evaluation"]["metric_table"])
    metric_table["scale2000_joint_adapter"] = scale2000_metrics
    decision = classify_scale2000(
        base_metrics=metric_table["base"],
        scale200_metrics=metric_table["scale200_joint_adapter"],
        scale2000_metrics=scale2000_metrics,
    )
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
        "models": {"scale2000_joint_adapter": summary},
        "metric_table": metric_table,
        "decision": decision,
    }
    atomic_write_json(output_dir / "summary.local.json", payload)
    print(json.dumps({"status": "PASSED", "classification": decision["classification"]}, ensure_ascii=False, sort_keys=True))
    return payload


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
        "# Data Report 0014: Scale-2000 Acoustic Admission",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This report records aggregate-only evidence for the nested scale-2000 synthetic audio bank. It contains no generated text, row IDs, local paths, or audio.",
        "",
        "## Counts",
        "",
        f"- Semantic rows: {payload['counts']['semantic_rows']}",
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
        "- The 2000x multiplier refers to exposure count, not independent linguistic information.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, "\n".join(lines))


def _write_experiment_markdown(path: Path, payload: dict[str, Any]) -> None:
    table = payload["directional_evaluation"]["metric_table"]
    decision = payload["directional_evaluation"]["decision"]
    lines = [
        "# Experiment 0014: GaMS 16000 Scale-2000 Text-Only Directional Diagnostic",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This diagnostic changes semantic text count from 1,600 to 16,000 while preserving the scale-200 voices, augmentation policy, joint-adapter surface, training protocol, and batch-32 directional evaluation policy. No canonical batch-1 evaluation was run.",
        "",
        "## Scale",
        "",
        f"- Semantic rows: {payload['data']['semantic_rows']}",
        f"- Clean files: {payload['data']['clean_files']}",
        f"- Augmented files: {payload['data']['augmented_files']}",
        f"- Exposure records: {payload['training']['sample_exposures']}",
        "- The 2000x multiplier refers to exposure count, not independent linguistic information.",
        "",
        "## Directional Metrics",
        "",
        "| Model | Piper holdout WER/CER | Supertonic holdout WER/CER | FLEURS-v2 WER/CER | ARTUR-J WER/CER |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_id in ("base", "scale200_joint_adapter", "scale2000_joint_adapter"):
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
            f"- Scale-200 burden: {decision['scale200_burden']}",
            f"- Scale-2000 burden: {decision['scale2000_burden']}",
            f"- Burden change: {decision['burden_change']}",
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


def stage_summarize(config_path: Path) -> dict[str, Any]:
    experiment, text, _augmentation = load_configs(config_path)
    validation = require_audio_accepted(config_path)
    training_path = _experiment_run_dir(experiment) / ARM_NAME / "training-summary.local.json"
    evaluation_path = _experiment_run_dir(experiment) / "directional-evaluation" / "summary.local.json"
    if not training_path.exists() or not evaluation_path.exists():
        raise RuntimeError("training and directional evaluation must complete before final summarize")
    training = read_json(training_path)
    evaluation = read_json(evaluation_path)
    fixed_text_sha = sha256_file(fixed_combined_text_path(text))
    audio_certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v4-scale2000-audio-v1",
        "status": "AUDIO_ACCEPTED",
        "decision_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "corpus_id": text["corpus_id"],
        "fixed_text_sha256": fixed_text_sha,
        "inherited_all_views_sha256": validation["hashes"]["inherited_all_views_sha256"],
        "counts": {
            "semantic_rows": int(validation["semantic_rows"]),
            "clean_files": int(validation["clean_files"]),
            "augmented_files": int(validation["augmented_files"]),
            "view_records": int(validation["view_records"]),
            "inherited_view_records": int(validation["inherited_view_records"]),
            "new_view_records": int(validation["new_view_records"]),
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
        "voices": {"clean_training_views": list(TRAINING_VIEWS), "voice_counts": validation["voice_counts"]},
        "limitations": validation["limitations"],
        "prohibited_statuses": ["TRAINING_ELIGIBLE"],
    }
    _public_payload_safe(audio_certificate)
    atomic_write_json(_public_output_path(experiment, "audio_certificate"), audio_certificate)
    audio_report = {
        "schema_version": "1.0",
        "report_id": "0014-scale2000-acoustic-admission",
        "status": "AUDIO_ACCEPTED",
        "certificate_sha256": sha256_file(_public_output_path(experiment, "audio_certificate")),
        "counts": audio_certificate["counts"],
        "hashes": {"fixed_text_sha256": fixed_text_sha, **validation["hashes"]},
        "schedule": audio_certificate["schedule"],
        "validation": audio_certificate["validation"],
        "voice_counts": validation["voice_counts"],
        "profile_counts": validation["profile_counts"],
        "duration_seconds": validation["duration_seconds"],
        "multiplier_table": scale2000_multiplier_table(),
        "limitations": audio_certificate["limitations"],
    }
    _public_payload_safe(audio_report)
    atomic_write_json(_public_output_path(experiment, "audio_report_json"), audio_report)
    _write_audio_markdown(_public_output_path(experiment, "audio_report_markdown"), audio_report)
    experiment_certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v4-scale2000-joint-adapter-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "decision_date": time.strftime("%Y-%m-%d", time.gmtime()),
        "work_order_id": experiment["work_order_id"],
        "text_certificate_sha256": sha256_file(_public_output_path(experiment, "text_certificate")),
        "audio_certificate_sha256": sha256_file(_public_output_path(experiment, "audio_certificate")),
        "fixed_text_sha256": fixed_text_sha,
        "all_views_sha256": validation["hashes"]["all_views_sha256"],
        "exposure_schedule_sha256": validation["schedule"]["schedule_sha256"],
        "experiment_config_sha256": sha256_file(REPO_ROOT / config_path),
        "augmentation_policy_sha256": sha256_file(REPO_ROOT / experiment["augmentation_config"]),
        "model": _public_model_identity(experiment),
        "adapter": experiment["adapter"],
        "training": experiment["training"],
        "evaluation": experiment["evaluation"],
        "authorized_actions": [
            "internally train one frozen-base Slovenian joint-adapter arm on the declared 320,000-exposure synthetic schedule",
            "evaluate the trained adapter using the declared batch-32 directional suite",
            "write privacy-safe aggregate evidence",
        ],
        "prohibited_actions": [
            "issuing TRAINING_ELIGIBLE",
            "accepting a checkpoint or adapter as a parent",
            "running batch-1 canonical evaluation",
            "using M5/F5 or holdout rows in training",
            "publishing generated text, audio, adapter, or checkpoint artifacts",
            "changing adapter rank, optimizer, learning rate, batch size, or augmentation policy",
        ],
        "limitations": [
            "DIAGNOSTIC_ONLY synthetic experiment.",
            "The 2000x multiplier is an exposure-count multiplier, not independent linguistic information.",
            "Fast batch-32 evaluation is directional and noncanonical.",
        ],
    }
    _public_payload_safe(experiment_certificate)
    atomic_write_json(_public_output_path(experiment, "experiment_certificate"), experiment_certificate)

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
                "padding_ratio": layout["padding_ratio"],
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
            "status": experiment_certificate["status"],
            "certificate_sha256": sha256_file(_public_output_path(experiment, "experiment_certificate")),
            "audio_certificate_sha256": sha256_file(_public_output_path(experiment, "audio_certificate")),
            "text_certificate_sha256": sha256_file(_public_output_path(experiment, "text_certificate")),
            "experiment_config_sha256": sha256_file(REPO_ROOT / config_path),
        },
        "data": {
            "corpus_id": text["corpus_id"],
            "semantic_rows": audio_report["counts"]["semantic_rows"],
            "clean_files": audio_report["counts"]["clean_files"],
            "augmented_files": audio_report["counts"]["augmented_files"],
            "view_records": audio_report["counts"]["view_records"],
            "fixed_text_sha256": fixed_text_sha,
            "all_views_sha256": audio_report["hashes"]["all_views_sha256"],
            "schedule_sha256": audio_report["schedule"]["schedule_sha256"],
            "voice_counts": audio_report["voice_counts"],
            "profile_counts": audio_report["profile_counts"],
            "multiplier_table": scale2000_multiplier_table(),
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
                "initial_anchor_probe_loss",
                "final_anchor_probe_loss",
                "initial_scale_probe_loss",
                "final_scale_probe_loss",
                "initial_anchor_full_loss",
                "final_anchor_full_loss",
                "initial_scale_full_loss",
                "final_scale_full_loss",
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
            "scale2000_model_summary": {
                "checkpoint_sha256": evaluation["models"]["scale2000_joint_adapter"]["checkpoint_sha256"],
                "suite": public_suite_summary(evaluation["models"]["scale2000_joint_adapter"]["suite"]),
            },
        },
        "limitations": [
            "Directional batch-32 metrics are not canonical acceptance evidence.",
            "No batch-1 evaluation was run.",
            "All training remains synthetic.",
            "The 2000x multiplier refers to exposure count, not independent linguistic information.",
            "No checkpoint or adapter is accepted as a parent.",
            "TRAINING_ELIGIBLE was not issued.",
        ],
    }
    _public_payload_safe(public)
    atomic_write_json(_public_output_path(experiment, "experiment_report_json"), public)
    _write_experiment_markdown(_public_output_path(experiment, "experiment_report_markdown"), public)
    result = {
        "status": "PASSED",
        "json_sha256": sha256_file(_public_output_path(experiment, "experiment_report_json")),
        "markdown_sha256": sha256_file(_public_output_path(experiment, "experiment_report_markdown")),
        "classification": public["directional_evaluation"]["decision"]["classification"],
        "accepted_parent": "none",
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--restart-from-scratch", action="store_true")
    parser.add_argument("--whole-file-outcome")
    parser.add_argument("--review-revision")
    parser.add_argument("--decision-id")
    parser.add_argument("--expected-corpus-sha256")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.stage == "verify":
        stage_verify(args.config)
    elif args.stage == "generate-text-until-valid":
        stage_generate_text_until_valid(args.config, restart_from_scratch=args.restart_from_scratch)
    elif args.stage == "prepare-human-decision":
        stage_prepare_human_decision(args.config)
    elif args.stage == "admit-text":
        stage_admit_text(args.config, args)
    elif args.stage == "synthesize-piper-new":
        stage_synthesize_piper_new(args.config)
    elif args.stage == "synthesize-supertonic-new":
        stage_synthesize_supertonic_new(args.config, args.progress_interval_seconds)
    elif args.stage == "augment-new":
        stage_augment_new(args.config)
    elif args.stage == "validate-combined-audio":
        stage_validate_combined_audio(args.config)
    elif args.stage == "verify-artifact":
        experiment, _text, _augmentation = load_configs(args.config)
        print(json.dumps(verify_scale2000_artifact(experiment), ensure_ascii=False, sort_keys=True))
    elif args.stage == "train":
        stage_train(args.config, args.progress_interval_seconds)
    elif args.stage == "evaluate-directional":
        stage_evaluate_directional(args.config)
    elif args.stage == "summarize":
        stage_summarize(args.config)
    else:
        raise SystemExit(f"unknown stage: {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
