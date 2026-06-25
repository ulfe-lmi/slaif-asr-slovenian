#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, atomic_write_text, load_jsonl, sha256_file
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.scale200_corpus import (
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


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs/experiments/gams1600_nine_voice_augmented_v1.json"


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
    _experiment, text_config, _augmentation = load_all_configs(config_path)
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
            "summarize",
        ],
    )
    parser.add_argument("--whole-file-outcome", choices=sorted({"ACCEPT", "REJECT_GRAMMAR", "REJECT_SEMANTICS", "REJECT_UNNATURAL", "REJECT_TEMPLATE", "REJECT_METADATA_LEAK", "REJECT_DUPLICATE", "REJECT_DOMAIN", "REJECT_TRANSCRIPTION", "REVISE_AND_REREVIEW"}))
    parser.add_argument("--review-revision")
    parser.add_argument("--decision-id")
    parser.add_argument("--expected-corpus-sha256")
    parser.add_argument("--expected-rows", type=int)
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
        elif args.stage in {"synthesize-piper", "synthesize-supertonic", "augment", "validate-audio", "train", "evaluate-directional"}:
            stage_not_yet_safe(args.config, args.stage)
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
