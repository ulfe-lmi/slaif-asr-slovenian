#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.corpus_v2_generation import (
    GpuMonitor,
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    write_rejections,
)
from slaif_asr.corpus_v2_holdout import (
    config_sha256,
    ensure_protected_indexes,
    expected_holdout_draft,
    filter_and_select_fixed_holdout,
    fixed_holdout_path,
    generated_all_path,
    gpu_monitor_path,
    load_candidate_source,
    load_config,
    parse_generated_outputs,
    planned_prompts_for_attempt,
    raw_generation_dir,
    rejected_path,
    run_dir,
    summarize_generation_metadata,
    validate_fixed_holdout,
    whole_file_command_path,
    write_public_reports,
    write_review_capsule,
)
from slaif_asr.data_quality import load_protected_index, sha256_file
from slaif_asr.gpu_policy import require_single_visible_cuda

from scripts.generate_gams_corpus_v2 import generate_with_fallback, load_model


REPO_ROOT = Path(__file__).resolve().parents[1]


def stage_verify(config: dict[str, Any]) -> int:
    info = require_single_visible_cuda(allowed_name_fragments=("A100",))
    tokenizer, model = load_model(config)
    metadata = {
        "stage": "verify",
        "gpu": info.to_dict(),
        "model": config["model"]["repository"],
        "revision": config["model"]["revision"],
        "quantization": config["quantization"]["policy"],
        "device_map": getattr(model, "hf_device_map", {}),
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "configuration_sha256": config_sha256(config),
    }
    if any(str(device) in {"cpu", "disk"} for device in metadata["device_map"].values()):
        raise RuntimeError(f"CPU or disk offload detected: {metadata['device_map']}")
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return 0


def stage_generate(config: dict[str, Any]) -> int:
    info = require_single_visible_cuda(allowed_name_fragments=("A100",))
    load_candidate_source(config)
    tokenizer, model = load_model(config)
    run_dir(config).mkdir(parents=True, exist_ok=True)
    raw_generation_dir(config).mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    all_rejections = []
    used_batch_size = int(config["generation"]["prompt_batch_size"])
    fallback_notes: list[str] = []
    start = time.perf_counter()
    with GpuMonitor(
        physical_selector=info.physical_selector,
        output_path=gpu_monitor_path(config),
        interval_seconds=float(config["generation"]["monitor_interval_seconds"]),
    ):
        max_attempt = max(int(cell["maximum_retries"]) for cell in config["prompt_cells"])
        for attempt_index in range(max_attempt + 1):
            prompts = planned_prompts_for_attempt(config, defaultdict(int), attempt_index)
            if not prompts:
                continue
            outputs, used_batch_size, notes = generate_with_fallback(
                tokenizer=tokenizer,
                model=model,
                prompts=prompts,
                config=config,
                requested_batch_size=used_batch_size,
            )
            fallback_notes.extend(notes)
            parsed_records, parser_rejections = parse_generated_outputs(config=config, outputs=outputs)
            all_rejections.extend(parser_rejections)
            all_records.extend(parsed_records)
    atomic_write_jsonl(generated_all_path(config), sorted(all_records, key=lambda item: str(item["candidate_id"])))
    write_rejections(rejected_path(config), all_rejections)
    metadata = summarize_generation_metadata(
        config,
        wall_time_seconds=time.perf_counter() - start,
        used_batch_size=used_batch_size,
        fallback_notes=fallback_notes,
    )
    atomic_write_json(run_dir(config) / "generation-summary.local.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    # The final stage performs protected-index and candidate-overlap validation.
    return 0


def stage_validate(config: dict[str, Any]) -> int:
    if not generated_all_path(config).exists():
        raise FileNotFoundError(f"missing generated holdout rows: {generated_all_path(config)}")
    protected_index_paths, protected_identities = ensure_protected_indexes(config)
    generated = load_jsonl(generated_all_path(config))
    previous_rejections = [
        row for row in (load_jsonl(rejected_path(config)) if rejected_path(config).exists() else [])
        if str(row.get("reason", "")).startswith("parser_")
    ]
    _admissible, fixed, rejected, selection_summary = filter_and_select_fixed_holdout(
        generated,
        config=config,
        existing_rejections=previous_rejections,
        protected_indexes=[],
    )
    # Use the verified protected indexes for the final fixed set and the full
    # text validator; the generator-time filter never sees raw protected text.
    _protected_admissible, protected_fixed, protected_rejected, protected_summary = filter_and_select_fixed_holdout(
        fixed,
        config=config,
        existing_rejections=[],
        protected_indexes=[load_protected_index(path) for path in protected_index_paths],
    )
    fixed = protected_fixed
    rejected = [*rejected, *protected_rejected]
    selection_summary = {
        **selection_summary,
        "protected_fixed_by_cell": protected_summary.get("fixed_by_cell", {}),
    }
    atomic_write_jsonl(fixed_holdout_path(config), fixed)
    write_rejections(rejected_path(config), rejected)
    atomic_write_json(run_dir(config) / "selection-summary.local.json", {**selection_summary, "protected_indexes": protected_identities})
    report = validate_fixed_holdout(config)
    summary = {
        "corpus_id": config["corpus_id"],
        "decision_reasons": report.get("decision_reasons", []),
        "fixed_holdout_rows": len(fixed),
        "final_text_status": report.get("final_text_status"),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if expected_holdout_draft(report) and len(fixed) == int(config["fixed_holdout_rows"]) else 1


def stage_prepare_review(config: dict[str, Any]) -> int:
    if not fixed_holdout_path(config).exists():
        raise FileNotFoundError(f"missing fixed holdout: {fixed_holdout_path(config)}")
    write_review_capsule(config)
    summary = {
        "corpus_id": config["corpus_id"],
        "fixed_holdout_sha256": sha256_file(fixed_holdout_path(config)),
        "review_capsule_rows": len(load_jsonl(fixed_holdout_path(config))),
        "whole_file_decision_command": str(whole_file_command_path(config).relative_to(REPO_ROOT)),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def stage_summarize(config: dict[str, Any]) -> int:
    payload = write_public_reports(config)
    summary = {
        "corpus_id": payload["corpus_id"],
        "fixed_holdout_rows": payload["fixed_holdout_rows"],
        "fixed_holdout_sha256": payload["holdout_file_hashes"]["fixed_holdout_sha256"],
        "status": payload["validator"]["status"],
        "decision_reasons": payload["validator"]["decision_reasons"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate the independent corpus-v2 synthetic holdout.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/generation/slovenian_corpus_v2_holdout_v1.json",
    )
    parser.add_argument(
        "--stage",
        choices=("verify", "generate", "validate", "prepare-review", "summarize", "all"),
        required=True,
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "verify":
        return stage_verify(config)
    if args.stage == "generate":
        return stage_generate(config)
    if args.stage == "validate":
        return stage_validate(config)
    if args.stage == "prepare-review":
        return stage_prepare_review(config)
    if args.stage == "summarize":
        return stage_summarize(config)
    if args.stage == "all":
        for stage in ("generate", "validate", "prepare-review", "summarize"):
            code = {
                "generate": stage_generate,
                "validate": stage_validate,
                "prepare-review": stage_prepare_review,
                "summarize": stage_summarize,
            }[stage](config)
            if code != 0:
                return code
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
