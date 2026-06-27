#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from slaif_asr.gams_retry_controller import AttemptRecord, AttemptTask, load_state, save_state
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.corpus_v2_generation import GpuMonitor
from slaif_asr.scale200_corpus import extract_utterance_lines
from slaif_asr.scale8000_corpus import (
    build_scale8000_combined_rows,
    build_new_record,
    build_task_prompt,
    build_dual_gpu_generation_plan,
    fixed_combined_text_path,
    generated_all_path,
    load_worker_generated_rows,
    load_scale8000_generation_config,
    new_addition_path,
    prompt_cells,
    prompt_path,
    raw_generation_dir,
    raw_output_path,
    rejected_path,
    refill_plan_path,
    refill_summary_path,
    run_directory,
    safe_public_status_report,
    scale8000_multiplier_table,
    storage_preflight,
    text_generation_summary_path,
    verify_inherited_scale2000_rows,
    worker_initial_tasks,
    worker_log_path,
    worker_monitor_path,
    worker_pid_path,
    worker_state_path,
    worker_summary_path,
)
from slaif_asr.scale200_corpus import filter_records, load_existing_holdout, protected_indexes, write_rejections


DEFAULT_CONFIG = REPO_ROOT / "configs/generation/gams_corpus_v5_scale8000_v1.json"


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": " ".join(command), "exit_code": completed.returncode, "output_tail": completed.stdout[-4000:]}


def _cell_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(cell["cell_id"]): cell for cell in prompt_cells(config)}


def _parse_task_output(config: dict[str, Any], task: Any, raw: str) -> tuple[list[dict[str, Any]], list[Any]]:
    lines, rejected = extract_utterance_lines(raw, cell_id=task.cell_id, attempt_id=task.attempt_id)
    cells = _cell_lookup(config)
    rows = [
        build_new_record(config, cells[task.cell_id], task, line.output_ordinal, line.text)
        for line in lines
    ]
    return rows, rejected


def _generate_task_batch(tokenizer: Any, model: Any, config: dict[str, Any], tasks: list[Any]) -> list[tuple[Any, str]]:
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


def _run_generation_task_list(config: dict[str, Any], worker: str, tasks: list[AttemptTask], *, stage_name: str) -> dict[str, Any]:
    from scripts.generate_gams_corpus_v2 import load_model

    if worker not in {"gpu0", "gpu1"}:
        raise ValueError("--worker must be gpu0 or gpu1")
    verify_inherited_scale2000_rows(config)
    info = require_single_visible_cuda(allowed_name_fragments=("A100",))
    run_directory(config).mkdir(parents=True, exist_ok=True)
    raw_generation_dir(config, worker).mkdir(parents=True, exist_ok=True)
    state_path = worker_state_path(config, worker)
    state = load_state(state_path)
    tasks = [task for task in tasks if task.attempt_id not in state.completed_attempt_ids]
    started = time.perf_counter()
    completed_at_start = len(state.completed_attempt_ids)
    tokenizer, model = load_model(config)
    batch_size = int(config["generation"]["prompt_batch_size"])
    fallback = int(config["generation"]["oom_fallback_batch_size"])
    processed_this_run = 0
    with GpuMonitor(
        physical_selector=info.physical_selector,
        output_path=worker_monitor_path(config, worker),
        interval_seconds=float(config["generation"].get("monitor_interval_seconds", 0.2)),
    ):
        index = 0
        while index < len(tasks):
            batch = tasks[index : index + batch_size]
            try:
                outputs = _generate_task_batch(tokenizer, model, config, batch)
            except RuntimeError as exc:
                lowered = str(exc).lower()
                if batch_size == int(config["generation"]["prompt_batch_size"]) and ("out of memory" in lowered or "cuda" in lowered):
                    batch_size = fallback
                    print(
                        json.dumps(
                            {
                                "event": "oom_fallback",
                                "stage": stage_name,
                                "worker": worker,
                                "fallback_batch_size": batch_size,
                                "error": exc.__class__.__name__,
                            },
                            sort_keys=True,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                for task in batch:
                    state.record(AttemptRecord(task=task, status="failed", error=exc.__class__.__name__))
                save_state(state_path, state)
                raise
            for task, raw in outputs:
                atomic_write_text(prompt_path(config, task, worker), build_task_prompt(config, task))
                atomic_write_text(raw_output_path(config, task, worker), raw)
                rows, rejected = _parse_task_output(config, task, raw)
                state.record(
                    AttemptRecord(
                        task=task,
                        status="completed",
                        parsed_rows=len(rows),
                        rejection_counts=dict(Counter(item.reason for item in rejected)),
                    )
                )
            save_state(state_path, state)
            processed_this_run += len(batch)
            index += len(batch)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "stage": stage_name,
                        "worker": worker,
                        "processed_attempts_this_run": processed_this_run,
                        "completed_attempts_total": len(state.completed_attempt_ids),
                        "worker_total_attempts": completed_at_start + len(tasks),
                        "batch_size": len(batch),
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    payload = {
        "status": "COMPLETED",
        "worker": worker,
        "gpu": info.to_dict(),
        "completed_at_start": completed_at_start,
        "processed_this_run": processed_this_run,
        "completed_attempts_total": len(state.completed_attempt_ids),
        "worker_total_attempts": completed_at_start + len(tasks),
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "state_path_token": state_path.name,
        "raw_generation_dir_token": f"raw-generation/{worker}",
        "stage": stage_name,
    }
    atomic_write_json(worker_summary_path(config, worker), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_generate_text_worker(config_path: Path, worker: str) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    return _run_generation_task_list(config, worker, worker_initial_tasks(config, worker), stage_name="generate-text-worker")


def stage_launch_dual_text_generation(config_path: Path, runs_root: Path | None) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    preflight = storage_preflight(config)
    if not preflight["sufficient"]:
        raise RuntimeError("storage preflight is insufficient for scale-8000 generation")
    run_directory(config).mkdir(parents=True, exist_ok=True)
    (run_directory(config) / "logs").mkdir(parents=True, exist_ok=True)
    launched: dict[str, Any] = {}
    for worker, selector in (("gpu0", "0"), ("gpu1", "1")):
        log_path = worker_log_path(config, worker)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = selector
        env["PYTHONUNBUFFERED"] = "1"
        if runs_root is not None:
            env["SLAIF_ASR_RUNS_ROOT"] = str(runs_root)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            "generate-text-worker",
            "--worker",
            worker,
            "--config",
            str(config_path),
        ]
        if runs_root is not None:
            command.extend(["--runs-root", str(runs_root)])
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched[worker] = {
            "pid": process.pid,
            "physical_gpu": selector,
            "cuda_visible_devices": selector,
            "logical_device": "cuda:0",
            "log_token": log_path.name,
            "command": " ".join(command),
        }
    payload = {
        "status": "LAUNCHED",
        "stage": "generate-text-dual",
        "workers": launched,
        "runs_root_token": "SLAIF_ASR_RUNS_ROOT" if runs_root is not None else "repository_runs",
    }
    atomic_write_json(worker_pid_path(config), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_generation_status(config_path: Path) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    workers: dict[str, Any] = {}
    for worker in ("gpu0", "gpu1"):
        state = load_state(worker_state_path(config, worker))
        total = len(worker_initial_tasks(config, worker))
        log_path = worker_log_path(config, worker)
        workers[worker] = {
            "completed_attempts": len(state.completed_attempt_ids),
            "total_attempts": total,
            "remaining_attempts": max(0, total - len(state.completed_attempt_ids)),
            "state_exists": worker_state_path(config, worker).exists(),
            "log_exists": log_path.exists(),
            "log_tail": log_path.read_text(encoding="utf-8", errors="replace")[-2000:] if log_path.exists() else "",
        }
    payload = {"status": "RUNNING" if any(item["remaining_attempts"] for item in workers.values()) else "COMPLETED", "workers": workers}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_merge_text(config_path: Path) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    inherited = verify_inherited_scale2000_rows(config)
    generated, parser_rejections, worker_status = load_worker_generated_rows(config)
    incomplete = {
        worker: status
        for worker, status in worker_status.items()
        if int(status["completed_attempts"]) < int(status["expected_attempts"])
    }
    if incomplete:
        raise RuntimeError(f"generation workers are incomplete: {incomplete}")
    atomic_write_jsonl(generated_all_path(config), generated)
    retained, rejected, filter_summary = filter_records(
        [*inherited, *generated],
        config=config,
        existing_rejections=parser_rejections,
        protected=protected_indexes(config),
        holdout_rows=load_existing_holdout(config),
    )
    new_retained = [row for row in retained if str(row.get("candidate_id", "")).startswith("gamsv5-")]
    combined, selection = build_scale8000_combined_rows(inherited, new_retained, config=config)
    selected_new_ids = {str(row["candidate_id"]) for row in combined if str(row.get("candidate_id", "")).startswith("gamsv5-")}
    selected_new = [row for row in new_retained if str(row["candidate_id"]) in selected_new_ids]
    atomic_write_jsonl(fixed_combined_text_path(config), combined)
    atomic_write_jsonl(new_addition_path(config), selected_new)
    write_rejections(rejected_path(config), rejected)
    payload = {
        "status": "STRUCTURALLY_READY_FOR_HUMAN_REVIEW",
        "corpus_id": config["corpus_id"],
        "generated_rows": len(generated),
        "new_admissible_rows": len(new_retained),
        "combined_rows": len(combined),
        "new_selected_rows": len(selected_new),
        "combined_sha256": sha256_file(fixed_combined_text_path(config)),
        "new_addition_sha256": sha256_file(new_addition_path(config)),
        "worker_status": worker_status,
        "filter_summary": filter_summary,
        "selection": selection,
    }
    atomic_write_json(text_generation_summary_path(config), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _completed_attempts_by_cell(config: dict[str, Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for worker in ("gpu0", "gpu1"):
        state = load_state(worker_state_path(config, worker))
        for record in state.records.values():
            if record.status == "completed":
                counts[record.task.cell_id] += 1
    return dict(counts)


def stage_plan_refill(config_path: Path) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    inherited = verify_inherited_scale2000_rows(config)
    generated, parser_rejections, worker_status = load_worker_generated_rows(config)
    retained, rejected, filter_summary = filter_records(
        [*inherited, *generated],
        config=config,
        existing_rejections=parser_rejections,
        protected=protected_indexes(config),
        holdout_rows=load_existing_holdout(config),
    )
    new_retained = [row for row in retained if str(row.get("candidate_id", "")).startswith("gamsv5-")]
    retained_by_cell = Counter(str(row.get("generation", {}).get("prompt_cell", "unknown")) for row in new_retained)
    attempts_by_cell = _completed_attempts_by_cell(config)
    target = int(config["new_rows_per_cell"]) + int(config["new_surplus_per_cell"])
    shortfalls = {
        str(cell["cell_id"]): target - retained_by_cell[str(cell["cell_id"])]
        for cell in prompt_cells(config)
        if retained_by_cell[str(cell["cell_id"])] < target
    }
    plans: dict[str, list[AttemptTask]] = {"gpu0": [], "gpu1": []}
    refill_counts_by_cell: dict[str, int] = {}
    if shortfalls:
        all_refill_tasks: list[AttemptTask] = []
        existing_refills: Counter[str] = Counter()
        for worker in ("gpu0", "gpu1"):
            state = load_state(worker_state_path(config, worker))
            for record in state.records.values():
                if record.task.reason == "targeted_refill":
                    existing_refills[record.task.cell_id] += 1
        for cell_id, deficit in sorted(shortfalls.items()):
            attempts_done = max(1, int(attempts_by_cell.get(cell_id, 0)))
            observed_yield = retained_by_cell[cell_id] / attempts_done
            conservative_yield = max(8.0, observed_yield * 0.75)
            attempt_count = max(1, math.ceil(deficit / conservative_yield) + 5)
            refill_counts_by_cell[cell_id] = attempt_count
            for offset in range(attempt_count):
                sequence = existing_refills[cell_id] + offset + 1
                shard_id = f"refill{sequence:04d}"
                seed = int(
                    __import__("hashlib")
                    .sha256(f"scale8000-refill-v1:{cell_id}:{shard_id}".encode("utf-8"))
                    .hexdigest()[:12],
                    16,
                ) % (2**31 - 1)
                all_refill_tasks.append(
                    AttemptTask(
                        cell_id=cell_id,
                        shard_id=shard_id,
                        attempt_index=0,
                        verification_round=1 + max(existing_refills.values(), default=0),
                        requested_rows=int(config["requested_rows_per_shard"]),
                        seed=seed,
                        reason="targeted_refill",
                        diversity_guidance=tuple(config.get("diversity_retry_guidance", ())),
                    )
                )
        for index, task in enumerate(all_refill_tasks):
            plans["gpu0" if index % 2 == 0 else "gpu1"].append(task)
    for worker, tasks in plans.items():
        atomic_write_json(refill_plan_path(config, worker), {"tasks": [task.to_json() for task in tasks]})
    write_rejections(rejected_path(config), rejected)
    payload = {
        "status": "REFILL_REQUIRED" if shortfalls else "NO_REFILL_NEEDED",
        "new_admissible_rows": len(new_retained),
        "new_admissible_per_cell": dict(sorted(retained_by_cell.items())),
        "shortfalls": shortfalls,
        "refill_attempts_by_cell": refill_counts_by_cell,
        "refill_attempts_by_worker": {worker: len(tasks) for worker, tasks in plans.items()},
        "worker_status": worker_status,
        "filter_summary": filter_summary,
    }
    atomic_write_json(refill_summary_path(config), payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _load_refill_plan(config: dict[str, Any], worker: str) -> list[AttemptTask]:
    path = refill_plan_path(config, worker)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [AttemptTask.from_json(item) for item in payload.get("tasks", [])]


def stage_generate_refill_worker(config_path: Path, worker: str) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    return _run_generation_task_list(config, worker, _load_refill_plan(config, worker), stage_name="generate-refill-worker")


def stage_launch_dual_refill(config_path: Path, runs_root: Path | None) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    run_directory(config).mkdir(parents=True, exist_ok=True)
    (run_directory(config) / "logs").mkdir(parents=True, exist_ok=True)
    launched: dict[str, Any] = {}
    for worker, selector in (("gpu0", "0"), ("gpu1", "1")):
        plan_tasks = _load_refill_plan(config, worker)
        if not plan_tasks:
            launched[worker] = {"status": "SKIPPED", "reason": "empty refill plan"}
            continue
        log_path = run_directory(config) / "logs" / f"{worker}.generate-refill.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = selector
        env["PYTHONUNBUFFERED"] = "1"
        if runs_root is not None:
            env["SLAIF_ASR_RUNS_ROOT"] = str(runs_root)
        command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--stage",
            "generate-refill-worker",
            "--worker",
            worker,
            "--config",
            str(config_path),
        ]
        if runs_root is not None:
            command.extend(["--runs-root", str(runs_root)])
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launched[worker] = {
            "status": "LAUNCHED",
            "pid": process.pid,
            "physical_gpu": selector,
            "cuda_visible_devices": selector,
            "planned_attempts": len(plan_tasks),
            "log_token": log_path.name,
        }
    payload = {"status": "LAUNCHED", "stage": "generate-refill-dual", "workers": launched}
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_verify(config_path: Path) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    inherited = verify_inherited_scale2000_rows(config)
    payload = {
        "status": "PASSED",
        "corpus_id": config["corpus_id"],
        "inherited_rows": len(inherited),
        "multiplier_table": scale8000_multiplier_table(),
        "dual_gpu_plan": build_dual_gpu_generation_plan(config),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_preflight(config_path: Path) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    payload = storage_preflight(config)
    payload["status"] = "PASSED" if payload["sufficient"] else "ENVIRONMENT_BLOCKED"
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _markdown_report(payload: dict[str, Any]) -> str:
    preflight = payload.get("storage_preflight", {})
    plan = payload["scale8000_plan"]
    lines = [
        "# Scale-8000 Dual-GPU Generation",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This report is privacy-safe planning and preflight evidence. It contains no raw generated text, audio paths, hypotheses, model artifacts, or monitor CSV data.",
        "",
        "## Corpus",
        "",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Parent scale-2000 corpus: `{payload['parent_scale2000']['corpus_id']}`",
        f"- Parent scale-2000 SHA256: `{payload['parent_scale2000']['sha256']}`",
        f"- Semantic rows planned: `{plan['semantic_rows']}`",
        f"- Clean files/views planned: `{plan['clean_files']}`",
        f"- Augmented files/views planned: `{plan['augmented_files']}`",
        f"- Total views/exposures planned: `{plan['total_views']}`",
        "",
        "## Inclusion",
        "",
        f"- Policy: `{payload['inclusion_policy']['type']}`",
        f"- Evidence: {payload['inclusion_policy']['description']}",
        "",
        "## Dual-GPU Plan",
        "",
    ]
    for worker, worker_payload in payload["dual_gpu_plan"]["workers"].items():
        lines.append(
            f"- `{worker}`: physical GPU `{worker_payload['physical_gpu']}`, "
            f"`CUDA_VISIBLE_DEVICES={worker_payload['cuda_visible_devices']}`, "
            f"tasks `{worker_payload['task_count']}`, requested rows `{worker_payload['requested_rows']}`"
        )
    lines.extend(
        [
            "",
            "## Storage Preflight",
            "",
            f"- Available bytes: `{preflight.get('available_bytes', 'unknown')}`",
            f"- Projected new bytes: `{preflight.get('projected_new_bytes', 'unknown')}`",
            f"- Required free bytes with safety margin: `{preflight.get('required_free_bytes', 'unknown')}`",
            f"- Sufficient: `{preflight.get('sufficient', 'unknown')}`",
            "",
            "## Decision",
            "",
            "Generation must not begin while storage preflight is insufficient. This is an environment blocker, not a corpus acceptance decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def stage_write_report(config_path: Path, canonical_results_path: Path | None = None) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    canonical = {}
    if canonical_results_path and canonical_results_path.exists():
        canonical = json.loads(canonical_results_path.read_text(encoding="utf-8"))
    preflight = storage_preflight(config)
    payload = safe_public_status_report(config, canonical_results=canonical, preflight=preflight)
    report_paths = config["public_reports"]
    json_path = REPO_ROOT / report_paths["planning_report_json"]
    md_path = REPO_ROOT / report_paths["planning_report_markdown"]
    atomic_write_json(json_path, payload)
    atomic_write_text(md_path, _markdown_report(payload))
    print(json.dumps({"status": payload["status"], "json": str(json_path), "markdown": str(md_path)}, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        choices=[
            "verify",
            "preflight",
            "write-report",
            "generate-text-worker",
            "generate-text-dual",
            "generation-status",
            "merge-text",
            "plan-refill",
            "generate-refill-worker",
            "generate-refill-dual",
        ],
        required=True,
    )
    parser.add_argument("--canonical-results", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--worker", choices=["gpu0", "gpu1"])
    args = parser.parse_args()
    if args.runs_root is not None:
        os.environ["SLAIF_ASR_RUNS_ROOT"] = str(args.runs_root.expanduser().resolve())
    if args.stage == "verify":
        stage_verify(args.config)
    elif args.stage == "preflight":
        stage_preflight(args.config)
    elif args.stage == "write-report":
        stage_write_report(args.config, args.canonical_results)
    elif args.stage == "generate-text-worker":
        if args.worker is None:
            raise SystemExit("--worker is required for generate-text-worker")
        stage_generate_text_worker(args.config, args.worker)
    elif args.stage == "generate-text-dual":
        stage_launch_dual_text_generation(args.config, args.runs_root)
    elif args.stage == "generation-status":
        stage_generation_status(args.config)
    elif args.stage == "merge-text":
        stage_merge_text(args.config)
    elif args.stage == "plan-refill":
        stage_plan_refill(args.config)
    elif args.stage == "generate-refill-worker":
        if args.worker is None:
            raise SystemExit("--worker is required for generate-refill-worker")
        stage_generate_refill_worker(args.config, args.worker)
    elif args.stage == "generate-refill-dual":
        stage_launch_dual_refill(args.config, args.runs_root)
    else:  # pragma: no cover
        raise AssertionError(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
