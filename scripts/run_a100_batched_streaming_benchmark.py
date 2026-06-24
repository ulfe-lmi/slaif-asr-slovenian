#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.batched_streaming import (
    arm_is_parity_eligible,
    compare_predictions,
    file_sha256,
    load_gate_records,
    load_local_predictions,
    metrics_for,
    parse_monitor_csv,
    privacy_safe_arm_summary,
    query_physical_gpu,
    read_json,
    round_float,
    run_batched_arm,
    run_old_ordered_arm,
    scientific_classification,
    select_batch_policy,
    select_hash_subset,
    should_run_batch_256,
)
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json


DEFAULT_CONFIG = Path("configs/experiments/a100_batched_streaming_v1.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def rel(path: Path) -> Path:
    return path if path.is_absolute() else repo_root() / path


def config_sha256(path: Path) -> str:
    return file_sha256(path)


def git_commit() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def load_config(path: Path) -> dict[str, Any]:
    return read_json(path)


def run_dir(config: dict[str, Any]) -> Path:
    return rel(Path(config["run_directory"]))


def stage_dir(config: dict[str, Any], stage: str) -> Path:
    path = run_dir(config) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env.setdefault("CUDA_VISIBLE_DEVICES", config["hardware"]["required_cuda_visible_devices"])
    env.setdefault("NEMO_ROOT", str(rel(Path(config["nemo"]["source_tree"]))))
    return env


def python_executable() -> Path:
    return Path(sys.executable)


def nemo_script(config: dict[str, Any]) -> Path:
    return rel(Path(config["nemo"]["streaming_script"])).resolve()


def checkpoint_path(config: dict[str, Any]) -> Path:
    return rel(Path(config["base_model"]["checkpoint_path"])).resolve()


def verify_static_identities(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = checkpoint_path(config)
    checkpoint_sha = file_sha256(checkpoint)
    expected_checkpoint_sha = config["base_model"]["checkpoint_sha256"]
    if checkpoint_sha != expected_checkpoint_sha:
        raise RuntimeError(f"checkpoint SHA256 mismatch: {checkpoint_sha} != {expected_checkpoint_sha}")
    nemo_root = rel(Path(config["nemo"]["source_tree"]))
    completed = subprocess.run(["git", "-C", str(nemo_root), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, check=True)
    nemo_revision = completed.stdout.strip()
    if nemo_revision != config["nemo"]["revision"]:
        raise RuntimeError(f"NeMo revision mismatch: {nemo_revision} != {config['nemo']['revision']}")
    gate_results = {}
    for name, gate in config["gates"].items():
        records = load_gate_records(
            rel(Path(gate["manifest"])),
            expected_sha256=gate["manifest_sha256"],
            expected_rows=int(gate["expected_rows"]),
            gate_id=gate["gate_id"],
        )
        gate_results[name] = {
            "gate_id": gate["gate_id"],
            "manifest_sha256": gate["manifest_sha256"],
            "rows": len(records),
            "audio_duration_seconds": round(sum(item.duration for item in records), 6),
        }
    return {
        "checkpoint_sha256": checkpoint_sha,
        "nemo_revision": nemo_revision,
        "gates": gate_results,
    }


def verify_gpu(config: dict[str, Any]) -> dict[str, Any]:
    required_selector = str(config["hardware"]["required_cuda_visible_devices"])
    actual_selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if actual_selector != required_selector:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be {required_selector}, got {actual_selector!r}")
    info = require_single_visible_cuda(allowed_name_fragments=(config["hardware"]["allowed_gpu_name_fragment"],))
    physical = query_physical_gpu(required_selector)
    if physical["memory_used_mib"] > float(config["hardware"]["idle_memory_mib_max"]):
        raise RuntimeError(f"physical GPU {required_selector} memory is not idle: {physical['memory_used_mib']} MiB")
    if physical["utilization_percent"] > float(config["hardware"]["idle_utilization_percent_max"]):
        raise RuntimeError(f"physical GPU {required_selector} utilization is not idle: {physical['utilization_percent']}%")
    return {"logical": info.to_dict(), "physical": physical}


def gate_records(config: dict[str, Any], key: str):
    gate = config["gates"][key]
    return load_gate_records(
        rel(Path(gate["manifest"])),
        expected_sha256=gate["manifest_sha256"],
        expected_rows=int(gate["expected_rows"]),
        gate_id=gate["gate_id"],
    )


def stage_verify(config: dict[str, Any], config_path: Path) -> int:
    payload = {
        "stage": "verify",
        "repository_commit": git_commit(),
        "configuration_sha256": config_sha256(config_path),
        "host": socket.gethostname(),
        "static_identities": verify_static_identities(config),
        "gpu": verify_gpu(config),
        "python": sys.version.split()[0],
    }
    atomic_write_json(stage_dir(config, "verify") / "summary.local.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def load_arm_predictions(path: Path) -> dict[str, str]:
    return load_local_predictions(path)


def compare_and_enrich(
    *,
    records,
    baseline_arm: dict[str, Any],
    candidate_arm: dict[str, Any],
    baseline_predictions: dict[str, str],
    candidate_predictions: dict[str, str],
    max_peak_memory_mib: float,
) -> dict[str, Any]:
    comparison = compare_predictions(
        records,
        baseline_predictions,
        candidate_predictions,
        baseline_metrics=baseline_arm["metrics"],
        candidate_metrics=candidate_arm["metrics"],
    )
    candidate_arm["exact_mismatch_count"] = comparison.exact_mismatch_count
    candidate_arm["normalized_mismatch_count"] = comparison.normalized_mismatch_count
    candidate_arm["missing_ids"] = comparison.missing_ids
    candidate_arm["duplicate_ids"] = comparison.duplicate_ids
    candidate_arm["unexpected_ids"] = comparison.unexpected_ids
    candidate_arm["metric_differences"] = comparison.metric_differences
    candidate_arm["empty_hypothesis_difference"] = comparison.empty_hypothesis_difference
    candidate_arm["parity_eligible"] = arm_is_parity_eligible(
        candidate_arm,
        comparison,
        max_peak_memory_mib=max_peak_memory_mib,
    )
    return candidate_arm


def stage_official_parity(config: dict[str, Any]) -> int:
    records = select_hash_subset(gate_records(config, "fleurs"), int(config["official_parity"]["subset_size"]))
    root = stage_dir(config, "official-parity") / time.strftime("%Y%m%d-%H%M%S")
    env = runtime_env(config)
    old = run_old_ordered_arm(
        records=records,
        run_dir=root / "old-path",
        python_executable=python_executable(),
        repo_script=repo_root() / "scripts" / "run_streaming_inference.py",
        checkpoint=checkpoint_path(config),
        context=config["inference"]["att_context_size"],
        env=env,
    )
    new = run_batched_arm(
        records=records,
        batch_size=1,
        bucketed=False,
        run_dir=root / "new-path",
        python_executable=python_executable(),
        nemo_script=nemo_script(config),
        checkpoint=checkpoint_path(config),
        context=config["inference"]["att_context_size"],
        env=env,
        physical_gpu_index=config["monitor"]["physical_gpu_index"],
        monitor_interval_seconds=float(config["monitor"]["sample_interval_seconds"]),
    )
    old_predictions = load_arm_predictions(root / "old-path" / "old-predictions.local.jsonl")
    new_predictions = load_arm_predictions(root / "new-path" / "predictions.local.jsonl")
    old_metrics = metrics_for(records, old_predictions)
    comparison = compare_predictions(
        records,
        old_predictions,
        new_predictions,
        baseline_metrics=old_metrics,
        candidate_metrics=new["metrics"],
    )
    payload = {
        "stage": "official-parity",
        "rows": len(records),
        "old_path": old,
        "new_path": privacy_safe_arm_summary(new),
        "missing": len(comparison.missing_ids),
        "duplicates": len(comparison.duplicate_ids),
        "exact_mismatches": comparison.exact_mismatch_count,
        "normalized_mismatches": comparison.normalized_mismatch_count,
        "metric_differences": comparison.metric_differences,
        "result": "PASSED" if comparison.exact_parity else "FAILED",
    }
    atomic_write_json(stage_dir(config, "official-parity") / "summary.local.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if comparison.exact_parity else 2


def stage_sweep(config: dict[str, Any]) -> int:
    records = gate_records(config, "fleurs")
    root = stage_dir(config, "sweep") / time.strftime("%Y%m%d-%H%M%S")
    env = runtime_env(config)
    arms: list[dict[str, Any]] = []
    baseline_predictions: dict[str, str] | None = None
    baseline_arm: dict[str, Any] | None = None
    for batch_size in config["sweep"]["batch_sizes"]:
        verify_gpu(config)
        arm_root = root / f"batch-{batch_size:03d}-bucketed"
        arm = run_batched_arm(
            records=records,
            batch_size=int(batch_size),
            bucketed=True,
            run_dir=arm_root,
            python_executable=python_executable(),
            nemo_script=nemo_script(config),
            checkpoint=checkpoint_path(config),
            context=config["inference"]["att_context_size"],
            env=env,
            physical_gpu_index=config["monitor"]["physical_gpu_index"],
            monitor_interval_seconds=float(config["monitor"]["sample_interval_seconds"]),
        )
        if int(batch_size) == 1:
            baseline_arm = arm
            baseline_predictions = load_arm_predictions(arm_root / "predictions.local.jsonl")
            arm["exact_mismatch_count"] = 0
            arm["normalized_mismatch_count"] = 0
            arm["metric_differences"] = {}
            arm["empty_hypothesis_difference"] = 0
            arm["parity_eligible"] = arm.get("status") == "PASSED"
        elif arm.get("status") == "PASSED" and baseline_arm is not None and baseline_predictions is not None:
            predictions = load_arm_predictions(arm_root / "predictions.local.jsonl")
            arm = compare_and_enrich(
                records=records,
                baseline_arm=baseline_arm,
                candidate_arm=arm,
                baseline_predictions=baseline_predictions,
                candidate_predictions=predictions,
                max_peak_memory_mib=float(config["hardware"]["max_peak_memory_mib"]),
            )
        else:
            arm["parity_eligible"] = False
        atomic_write_json(arm_root / "summary.local.json", arm)
        arms.append(arm)
        if arm.get("status") == "ENVIRONMENT_BLOCKED":
            break
    by_batch = {int(arm["batch_size"]): arm for arm in arms}
    if config["sweep"].get("conditional_batch_256") and 64 in by_batch and 128 in by_batch:
        if should_run_batch_256(
            by_batch[128],
            by_batch[64],
            max_memory_mib=float(config["sweep"]["conditional_batch_256_max_memory_mib"]),
            min_gain=float(config["sweep"]["conditional_batch_256_min_gain_over_64"]),
        ):
            verify_gpu(config)
            arm_root = root / "batch-256-bucketed"
            arm = run_batched_arm(
                records=records,
                batch_size=256,
                bucketed=True,
                run_dir=arm_root,
                python_executable=python_executable(),
                nemo_script=nemo_script(config),
                checkpoint=checkpoint_path(config),
                context=config["inference"]["att_context_size"],
                env=env,
                physical_gpu_index=config["monitor"]["physical_gpu_index"],
                monitor_interval_seconds=float(config["monitor"]["sample_interval_seconds"]),
            )
            if arm.get("status") == "PASSED" and baseline_arm is not None and baseline_predictions is not None:
                predictions = load_arm_predictions(arm_root / "predictions.local.jsonl")
                arm = compare_and_enrich(
                    records=records,
                    baseline_arm=baseline_arm,
                    candidate_arm=arm,
                    baseline_predictions=baseline_predictions,
                    candidate_predictions=predictions,
                    max_peak_memory_mib=float(config["hardware"]["max_peak_memory_mib"]),
                )
            arms.append(arm)
    selected = select_batch_policy(arms, within_best_fraction=float(config["selection"]["within_best_fraction"]))
    payload = {
        "stage": "sweep",
        "gate_id": config["gates"]["fleurs"]["gate_id"],
        "manifest_sha256": config["gates"]["fleurs"]["manifest_sha256"],
        "rows": len(records),
        "arms": [privacy_safe_arm_summary(arm) for arm in arms],
        "selected_from_fleurs": selected,
        "run_root": str(root),
    }
    atomic_write_json(stage_dir(config, "sweep") / "summary.local.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def load_sweep_summary(config: dict[str, Any]) -> dict[str, Any]:
    path = stage_dir(config, "sweep") / "summary.local.json"
    if not path.exists():
        raise FileNotFoundError(f"run sweep first: {path}")
    return read_json(path)


def selected_fleurs_batch(config: dict[str, Any]) -> int:
    return int(load_sweep_summary(config)["selected_from_fleurs"]["batch_size"])


def stage_validate_selected(config: dict[str, Any]) -> int:
    sweep = load_sweep_summary(config)
    candidate_batch = int(sweep["selected_from_fleurs"]["batch_size"])
    env = runtime_env(config)
    root = stage_dir(config, "validate-selected") / time.strftime("%Y%m%d-%H%M%S")
    records = gate_records(config, "artur_j")
    candidates = [candidate_batch] + [
        int(arm["batch_size"])
        for arm in sorted(sweep["arms"], key=lambda item: int(item["batch_size"]), reverse=True)
        if int(arm["batch_size"]) < candidate_batch and arm.get("parity_eligible")
    ]
    candidates = list(dict.fromkeys(candidates))
    batch1 = run_batched_arm(
        records=records,
        batch_size=1,
        bucketed=True,
        run_dir=root / "artur-batch-001-bucketed",
        python_executable=python_executable(),
        nemo_script=nemo_script(config),
        checkpoint=checkpoint_path(config),
        context=config["inference"]["att_context_size"],
        env=env,
        physical_gpu_index=config["monitor"]["physical_gpu_index"],
        monitor_interval_seconds=float(config["monitor"]["sample_interval_seconds"]),
    )
    baseline_predictions = load_arm_predictions(root / "artur-batch-001-bucketed" / "predictions.local.jsonl")
    selected_artur = None
    artur_arms = [batch1]
    for batch_size in candidates:
        if batch_size == 1:
            selected_artur = batch1
            selected_artur["parity_eligible"] = True
            break
        verify_gpu(config)
        arm_root = root / f"artur-batch-{batch_size:03d}-bucketed"
        arm = run_batched_arm(
            records=records,
            batch_size=batch_size,
            bucketed=True,
            run_dir=arm_root,
            python_executable=python_executable(),
            nemo_script=nemo_script(config),
            checkpoint=checkpoint_path(config),
            context=config["inference"]["att_context_size"],
            env=env,
            physical_gpu_index=config["monitor"]["physical_gpu_index"],
            monitor_interval_seconds=float(config["monitor"]["sample_interval_seconds"]),
        )
        if arm.get("status") == "PASSED":
            predictions = load_arm_predictions(arm_root / "predictions.local.jsonl")
            arm = compare_and_enrich(
                records=records,
                baseline_arm=batch1,
                candidate_arm=arm,
                baseline_predictions=baseline_predictions,
                candidate_predictions=predictions,
                max_peak_memory_mib=float(config["hardware"]["max_peak_memory_mib"]),
            )
        artur_arms.append(arm)
        if arm.get("parity_eligible"):
            selected_artur = arm
            break
    if selected_artur is None:
        selected_artur = batch1
        selected_artur["parity_eligible"] = True
    final_batch = int(selected_artur["batch_size"])
    fleurs_records = gate_records(config, "fleurs")
    unbucketed = run_batched_arm(
        records=fleurs_records,
        batch_size=final_batch,
        bucketed=False,
        run_dir=root / f"fleurs-batch-{final_batch:03d}-unbucketed",
        python_executable=python_executable(),
        nemo_script=nemo_script(config),
        checkpoint=checkpoint_path(config),
        context=config["inference"]["att_context_size"],
        env=env,
        physical_gpu_index=config["monitor"]["physical_gpu_index"],
        monitor_interval_seconds=float(config["monitor"]["sample_interval_seconds"]),
    )
    # Reuse the local bucketed batch-1 baseline predictions from the sweep for the diagnostic unbucketed comparison.
    sweep_root = Path(sweep["run_root"])
    fleurs_baseline_predictions = load_arm_predictions(sweep_root / "batch-001-bucketed" / "predictions.local.jsonl")
    fleurs_baseline_summary = read_json(sweep_root / "batch-001-bucketed" / "summary.local.json")
    if unbucketed.get("status") == "PASSED":
        predictions = load_arm_predictions(root / f"fleurs-batch-{final_batch:03d}-unbucketed" / "predictions.local.jsonl")
        unbucketed = compare_and_enrich(
            records=fleurs_records,
            baseline_arm=fleurs_baseline_summary,
            candidate_arm=unbucketed,
            baseline_predictions=fleurs_baseline_predictions,
            candidate_predictions=predictions,
            max_peak_memory_mib=float(config["hardware"]["max_peak_memory_mib"]),
        )
    batch1_throughput = float(sweep["arms"][0]["end_to_end_audio_seconds_per_wall_second"])
    selected_fleurs_arm = next(
        arm for arm in sweep["arms"] if int(arm["batch_size"]) == final_batch and arm.get("bucketed")
    )
    selected_throughput = float(selected_fleurs_arm["end_to_end_audio_seconds_per_wall_second"])
    final_duration_bucketing = True
    unbucketed_throughput = unbucketed.get("end_to_end_audio_seconds_per_wall_second")
    if (
        unbucketed.get("parity_eligible")
        and unbucketed_throughput is not None
        and float(unbucketed_throughput) > selected_throughput
    ):
        selected_throughput = float(unbucketed_throughput)
        final_duration_bucketing = False
    speedup = selected_throughput / batch1_throughput if batch1_throughput else None
    exact_above_one = any(int(arm["batch_size"]) > 1 and arm.get("parity_eligible") for arm in sweep["arms"])
    classification = scientific_classification(
        selected_batch=final_batch,
        exact_parity_above_one=exact_above_one,
        selected_speedup=speedup,
    )
    payload = {
        "stage": "validate-selected",
        "artur_j": {
            "batch_1": privacy_safe_arm_summary(batch1),
            "candidate_arms": [privacy_safe_arm_summary(arm) for arm in artur_arms if int(arm["batch_size"]) != 1],
            "selected": privacy_safe_arm_summary(selected_artur),
        },
        "fleurs_unbucketed_diagnostic": privacy_safe_arm_summary(unbucketed),
        "selected_policy": {
            "batch_size": final_batch,
            "duration_bucketing": final_duration_bucketing,
            "speedup_vs_batch_1": round_float(speedup),
            "classification": classification,
            "bucketed_throughput": round_float(float(selected_fleurs_arm["end_to_end_audio_seconds_per_wall_second"])),
            "unbucketed_throughput": round_float(float(unbucketed_throughput)) if unbucketed_throughput is not None else None,
        },
        "run_root": str(root),
    }
    atomic_write_json(stage_dir(config, "validate-selected") / "summary.local.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if selected_artur.get("parity_eligible") else 3


def metric_table(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    return {
        "raw_corpus_wer": metrics["raw"]["corpus_wer"],
        "raw_corpus_cer": metrics["raw"]["corpus_cer"],
        "normalized_corpus_wer": metrics["normalized"]["corpus_wer"],
        "normalized_corpus_cer": metrics["normalized"]["corpus_cer"],
        "mean_utterance_wer": metrics["raw"]["mean_utterance_wer"],
        "median_utterance_wer": metrics["raw"]["median_utterance_wer"],
        "mean_utterance_cer": metrics["raw"]["mean_utterance_cer"],
        "median_utterance_cer": metrics["raw"]["median_utterance_cer"],
        "empty_hypotheses": metrics["raw"]["empty_hypothesis_count"],
    }


def stage_summarize(config: dict[str, Any], config_path: Path) -> int:
    verify = read_json(stage_dir(config, "verify") / "summary.local.json")
    parity = read_json(stage_dir(config, "official-parity") / "summary.local.json")
    sweep = read_json(stage_dir(config, "sweep") / "summary.local.json")
    selected = read_json(stage_dir(config, "validate-selected") / "summary.local.json")
    selected_batch = int(selected["selected_policy"]["batch_size"])
    fleurs_batch1 = next(arm for arm in sweep["arms"] if int(arm["batch_size"]) == 1)
    selected_fleurs = next(arm for arm in sweep["arms"] if int(arm["batch_size"]) == selected_batch and arm.get("bucketed"))
    selected_policy = dict(selected["selected_policy"])
    selected_policy.setdefault(
        "bucketed_throughput",
        round_float(float(selected_fleurs["end_to_end_audio_seconds_per_wall_second"])),
    )
    unbucketed = selected["fleurs_unbucketed_diagnostic"]
    unbucketed_throughput = unbucketed.get("end_to_end_audio_seconds_per_wall_second")
    selected_policy.setdefault(
        "unbucketed_throughput",
        round_float(float(unbucketed_throughput)) if unbucketed_throughput is not None else None,
    )
    if (
        unbucketed.get("parity_eligible")
        and unbucketed_throughput is not None
        and float(unbucketed_throughput) > float(selected_fleurs["end_to_end_audio_seconds_per_wall_second"])
    ):
        selected_policy["duration_bucketing"] = False
        batch1_throughput = float(fleurs_batch1["end_to_end_audio_seconds_per_wall_second"])
        selected_policy["speedup_vs_batch_1"] = round_float(float(unbucketed_throughput) / batch1_throughput)
    selected_duration_bucketing = bool(selected_policy["duration_bucketing"])
    fleurs_baseline_report_arm = (
        unbucketed
        if selected_batch == 1 and not selected_duration_bucketing and unbucketed.get("parity_eligible")
        else fleurs_batch1
    )
    parity_report = {
        "rows": parity["rows"],
        "old_path_predictions": parity.get("old_path_predictions", parity.get("old_path", {}).get("prediction_count")),
        "new_path_predictions": parity.get("new_path_predictions", parity.get("new_path", {}).get("prediction_count")),
        "missing_ids": parity.get("missing_ids", parity.get("missing")),
        "duplicate_ids": parity.get("duplicate_ids", parity.get("duplicates")),
        "exact_mismatches": parity["exact_mismatches"],
        "normalized_mismatches": parity["normalized_mismatches"],
        "metric_differences": parity["metric_differences"],
        "result": parity["result"],
    }
    report = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_commit(),
        "configuration_sha256": config_sha256(config_path),
        "host": verify["host"],
        "runtime": {
            "python": verify["python"],
            "gpu": verify["gpu"],
            "precision": "FP32",
            "tf32": False,
            "normalizer": NORMALIZER_VERSION,
        },
        "input_integrity": verify["static_identities"],
        "official_batch_1_parity": parity_report,
        "fleurs_v2_sweep": sweep["arms"],
        "artur_j_confirmation": selected["artur_j"],
        "unbucketed_diagnostic": selected["fleurs_unbucketed_diagnostic"],
        "selected_policy": selected_policy,
        "fleurs_v2_untouched_base_metrics": {
            **metric_table(fleurs_baseline_report_arm),
            "rows": fleurs_baseline_report_arm["rows"],
            "audio_duration_seconds": fleurs_baseline_report_arm["audio_duration_seconds"],
            "wall_time_seconds": fleurs_baseline_report_arm["execution"]["wall_time_seconds"],
            "real_time_factor": fleurs_baseline_report_arm["end_to_end_real_time_factor"],
            "checkpoint_sha256": config["base_model"]["checkpoint_sha256"],
            "att_context_size": config["inference"]["att_context_size"],
            "target_lang": config["inference"]["target_lang"],
            "batch_size": 1,
            "duration_bucketing": selected_duration_bucketing if selected_batch == 1 else True,
        },
        "limitations": [
            "Batch size 1 remains the scientific reference mode.",
            "The selected A100 policy is not an RTX 2080 Ti policy.",
            "The corpus-v2 candidate reservoir was not scored in this work order.",
            "Raw references, hypotheses, local manifests, logs, and monitor CSVs remain ignored local artifacts.",
        ],
    }
    json_path = repo_root() / "docs" / "experiments" / "0006-a100-batched-streaming-evaluation.json"
    md_path = repo_root() / "docs" / "experiments" / "0006-a100-batched-streaming-evaluation.md"
    atomic_write_json(json_path, report)
    lines = [
        "# Experiment 0006: A100 Batched Streaming Evaluation",
        "",
        "Status: **completed in PR; pending strategic review**",
        "",
        "This experiment establishes the first valid untouched-base FLEURS-v2 ASR baseline and a parity-proven A100 streaming batch policy. It does not score the corpus-v2 candidate reservoir and does not train a model.",
        "",
        "## Input Identity",
        "",
        f"- Checkpoint SHA256: `{config['base_model']['checkpoint_sha256']}`",
        f"- NeMo revision: `{config['nemo']['revision']}`",
        f"- FLEURS-v2 manifest SHA256: `{config['gates']['fleurs']['manifest_sha256']}`",
        f"- ARTUR-J manifest SHA256: `{config['gates']['artur_j']['manifest_sha256']}`",
        f"- Context: `{config['inference']['att_context_size']}`",
        "- Target language: `sl-SI`",
        "- Precision: FP32, TF32 disabled",
        "",
        "## Official Batch-1 Parity",
        "",
        f"- Rows: {parity['rows']}",
        f"- Exact mismatches: {parity['exact_mismatches']}",
        f"- Normalized mismatches: {parity['normalized_mismatches']}",
        f"- Metric differences: {len(parity['metric_differences'])}",
        f"- Result: {parity['result']}",
        "",
        "## FLEURS-v2 Sweep",
        "",
        "| Batch | Bucketed | Status | Exact mismatch | End-to-end RTF | Active RTF | Speedup | Padding ratio | Mean util | P95 util | Peak memory MiB |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base_throughput = float(fleurs_batch1["end_to_end_audio_seconds_per_wall_second"])
    for arm in sweep["arms"]:
        throughput = arm.get("end_to_end_audio_seconds_per_wall_second")
        speedup = float(throughput) / base_throughput if throughput and base_throughput else 0.0
        monitor = arm.get("execution", {}).get("monitor", {})
        layout = arm.get("layout", {})
        lines.append(
            "| {batch} | {bucketed} | {status} | {mismatch} | {rtf} | {active} | {speedup:.3f} | {padding} | {mean} | {p95} | {mem} |".format(
                batch=arm["batch_size"],
                bucketed="yes" if arm["bucketed"] else "no",
                status=arm["status"],
                mismatch=arm.get("exact_mismatch_count", ""),
                rtf=arm.get("end_to_end_real_time_factor"),
                active=arm.get("active_real_time_factor"),
                speedup=speedup,
                padding=layout.get("padding_ratio"),
                mean=monitor.get("mean_utilization_percent"),
                p95=monitor.get("p95_utilization_percent"),
                mem=monitor.get("peak_memory_mib"),
            )
        )
    lines.extend(
        [
            "",
            "## Selected Policy",
            "",
            f"- Batch size: {selected_batch}",
            f"- Duration bucketing: {'enabled' if selected_duration_bucketing else 'disabled'}",
            f"- Scientific classification: `{selected_policy['classification']}`",
            f"- Speedup vs batch 1: {selected_policy['speedup_vs_batch_1']}",
            f"- Bucketed throughput: {selected_policy.get('bucketed_throughput')} audio seconds per wall second",
            f"- Unbucketed throughput: {selected_policy.get('unbucketed_throughput')} audio seconds per wall second",
            "",
            "## FLEURS-v2 Untouched Base Metrics",
            "",
            f"- Raw corpus WER/CER: {fleurs_baseline_report_arm['metrics']['raw']['corpus_wer']} / {fleurs_baseline_report_arm['metrics']['raw']['corpus_cer']}",
            f"- Normalized corpus WER/CER: {fleurs_baseline_report_arm['metrics']['normalized']['corpus_wer']} / {fleurs_baseline_report_arm['metrics']['normalized']['corpus_cer']}",
            f"- Mean utterance WER/CER: {fleurs_baseline_report_arm['metrics']['raw']['mean_utterance_wer']} / {fleurs_baseline_report_arm['metrics']['raw']['mean_utterance_cer']}",
            f"- Median utterance WER/CER: {fleurs_baseline_report_arm['metrics']['raw']['median_utterance_wer']} / {fleurs_baseline_report_arm['metrics']['raw']['median_utterance_cer']}",
            f"- Empty hypotheses: {fleurs_baseline_report_arm['metrics']['raw']['empty_hypothesis_count']}",
            f"- Rows: {fleurs_baseline_report_arm['rows']}",
            f"- Audio duration: {fleurs_baseline_report_arm['audio_duration_seconds']} s",
            f"- Wall time: {fleurs_baseline_report_arm['execution']['wall_time_seconds']} s",
            f"- RTF: {fleurs_baseline_report_arm['end_to_end_real_time_factor']}",
            "",
            "## ARTUR-J Confirmation",
            "",
            f"- Selected batch exact mismatches: {selected['artur_j']['selected'].get('exact_mismatch_count', 0)}",
            f"- Selected batch metric differences: {len(selected['artur_j']['selected'].get('metric_differences', {}))}",
            "",
            "## Notes",
            "",
            "- Batch size 1 remains the scientific reference mode.",
            "- The selected A100 policy is not an RTX 2080 Ti policy.",
            "- The corpus-v2 candidate reservoir was not scored.",
            "- Raw references, hypotheses, local manifests, logs, and monitor CSVs remain ignored.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    policy = read_json(repo_root() / "configs" / "evaluation" / "a100_streaming_batch_policy.json")
    policy.update(
        {
            "status": "measured",
            "batch_size": selected_batch,
            "duration_bucketing": selected_duration_bucketing,
            "selection_reason": selected_policy,
            "fleurs_v2_manifest_sha256": config["gates"]["fleurs"]["manifest_sha256"],
            "artur_j_manifest_sha256": config["gates"]["artur_j"]["manifest_sha256"],
            "experiment_report": "docs/experiments/0006-a100-batched-streaming-evaluation.json",
        }
    )
    atomic_write_json(repo_root() / "configs" / "evaluation" / "a100_streaming_batch_policy.json", policy)
    print(json.dumps({"report": str(json_path), "policy_batch_size": selected_batch}, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A100 batched cache-aware streaming benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("verify", "official-parity", "sweep", "validate-selected", "summarize"), required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "verify":
        return stage_verify(config, args.config)
    if args.stage == "official-parity":
        return stage_official_parity(config)
    if args.stage == "sweep":
        return stage_sweep(config)
    if args.stage == "validate-selected":
        return stage_validate_selected(config)
    if args.stage == "summarize":
        return stage_summarize(config, args.config)
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
