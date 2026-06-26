#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, atomic_write_text, canonical_json_sha256, sha256_file
from slaif_asr.gams_retry_controller import AttemptRecord, AttemptTask, RetryState, initial_tasks, load_state, plan_refill_tasks, save_state
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.scale2000_corpus import (
    build_combined_rows,
    build_new_record,
    build_task_prompt,
    fixed_combined_text_path,
    generation_state_path,
    load_scale2000_experiment_config,
    load_scale2000_generation_config,
    new_addition_path,
    protected_config_fingerprints,
    retry_history_path,
    retry_limits_from_config,
    run_dir,
    scale2000_multiplier_table,
    verify_inherited_rows,
    verify_prompt_cells_match_anchor,
    verify_scale200_report,
)
from slaif_asr.scale200_corpus import extract_utterance_lines, filter_records, load_augmentation_config, load_existing_holdout, protected_indexes, write_rejections
from slaif_asr.corpus_v2_generation import GpuMonitor


DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs/experiments/gams16000_scale2000_text_only_v1.json"


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
    corpus_hash = sha256_file(path)
    rows = sum(1 for _ in path.open("r", encoding="utf-8"))
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
    atomic_write_text(run_dir(text) / "whole-file-decision-command.local.txt", command + "\n")
    payload = {
        "status": "READY_FOR_HUMAN_DECISION",
        "decision": f"ACCEPT or REJECT {text['corpus_id']} {corpus_hash} {rows}",
        "combined_sha256": corpus_hash,
        "rows": rows,
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
        combined, selection = build_combined_rows(inherited, new_retained, config=config)
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
    while shortfalls and round_index <= limits.max_verification_rounds:
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
        raise RuntimeError(f"retry budget ended before sufficient text: {shortfalls}")
    summary["wall_time_seconds"] = round(time.perf_counter() - started, 6)
    summary["retry_budget"] = state.budget_summary(limits)
    atomic_write_json(run_dir(text) / "text-generation-summary.local.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return summary


def stage_not_yet_safe(config_path: Path, stage: str) -> dict[str, Any]:
    experiment, text, _augmentation = load_configs(config_path)
    run_dir(text).mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "NOT_RUN",
        "stage": stage,
        "reason": "This stage is implemented in later commit phases or requires prior whole-file text admission.",
        "work_order_id": experiment["work_order_id"],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


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
    args = parser.parse_args()

    if args.stage == "verify":
        stage_verify(args.config)
    elif args.stage == "generate-text-until-valid":
        stage_generate_text_until_valid(args.config, restart_from_scratch=args.restart_from_scratch)
    elif args.stage == "prepare-human-decision":
        stage_prepare_human_decision(args.config)
    elif args.stage in {
        "admit-text",
        "synthesize-piper-new",
        "synthesize-supertonic-new",
        "augment-new",
        "validate-combined-audio",
        "train",
        "evaluate-directional",
        "summarize",
    }:
        stage_not_yet_safe(args.config, args.stage)
    else:
        raise SystemExit(f"unknown stage: {args.stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
