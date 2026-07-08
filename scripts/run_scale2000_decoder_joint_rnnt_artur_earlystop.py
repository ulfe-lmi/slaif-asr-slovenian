#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.artur_earlystop import (
    ARM_NAME,
    CERTIFICATE_ID,
    PR36_DECODER_JOINT_METRICS,
    assert_no_raw_report_material,
    classify_artur_earlystop,
    concurrent_gpu_contract,
    load_controller_dev_records,
    redacted_checkpoint_row,
    select_round,
    validate_agents_controller_dev_exception,
    validate_earlystop_config,
)
from slaif_asr.batched_streaming import load_local_predictions, metrics_for, run_batched_arm
from slaif_asr.corpus_v2_scoring import nemo_streaming_script, runtime_environment
from slaif_asr.batched_streaming import file_sha256
from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.directional_evaluation import load_directional_suite, normalized_metric_row, split_predictions, write_privacy_safe_suite_manifest
from slaif_asr.emission_rnnt_finetune import BASE_DIRECTIONAL_METRICS, SCALE2000_JOINT_ADAPTER_METRICS, verify_all_inputs


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/scale2000_decoder_joint_rnnt_artur_earlystop.json"
BASE_CONFIG = REPO_ROOT / "configs/experiments/scale2000_decoder_joint_rnnt_v1.json"
FAST_DIRECTIONAL_CONFIG = REPO_ROOT / "configs/experiments/fast_batched_directional_replay_v1.json"
REPORT_JSON = REPO_ROOT / "docs/experiments/0019-scale2000-decoder-joint-rnnt-artur-earlystop.json"
REPORT_MD = REPO_ROOT / "docs/experiments/0019-scale2000-decoder-joint-rnnt-artur-earlystop.md"
CERTIFICATE_PATH = REPO_ROOT / "docs/data-certificates/sl-corpus-v4-decoder-joint-rnnt-artur-earlystop-diagnostic-v1.json"


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def local_runs_root() -> Path:
    override = os.environ.get("SLAIF_ASR_RUNS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "runs"


def local_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "runs":
        return local_runs_root().joinpath(*path.parts[1:])
    return REPO_ROOT / path


def run_root(config: dict[str, Any]) -> Path:
    return local_path(config["local_outputs"]["run_root"])


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    validate_earlystop_config(config)
    return config


def stage_verify_inputs(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    base_config = read_json(BASE_CONFIG)
    base_inputs = verify_all_inputs(base_config)
    certificate_path = REPO_ROOT / config["controller_dev"]["certificate"]
    certificate = read_json(certificate_path)
    if certificate.get("partition_id") != "artur-controller-dev-v1":
        raise RuntimeError("ARTUR controller-dev certificate partition mismatch")
    if certificate.get("manifest_sha256") != config["controller_dev"]["manifest_sha256"]:
        raise RuntimeError("ARTUR controller-dev manifest SHA mismatch")
    if int(certificate.get("row_count", 0)) != int(config["controller_dev"]["rows"]):
        raise RuntimeError("ARTUR controller-dev row count mismatch")
    if not validate_agents_controller_dev_exception((REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")):
        raise RuntimeError("AGENTS.md does not preserve the ADR 0008 exception wording")
    payload = {
        "status": "PASSED",
        "work_order_id": "0032",
        "repository_commit": git_head(),
        "base_inputs": base_inputs,
        "controller_dev_certificate_sha256": file_sha256(certificate_path),
        "controller_dev_manifest_sha256": certificate["manifest_sha256"],
    }
    atomic_write_json(run_root(config) / "verification" / "inputs.local.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_probe_run_control(config_path: Path, *, training_gpu: str, validation_gpu: str, sequential: bool) -> dict[str, Any]:
    config = load_config(config_path)
    payload = {
        "status": "PASSED",
        "gpu_contract": concurrent_gpu_contract(training_gpu, validation_gpu, sequential=sequential),
        "training_gpu_selector": training_gpu,
        "validation_gpu_selector": validation_gpu if not sequential else None,
        "sequential_validation": sequential,
    }
    atomic_write_json(run_root(config) / "verification" / "run-control.local.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _derived_training_config(config: dict[str, Any]) -> Path:
    derived = read_json(BASE_CONFIG)
    derived["local_outputs"]["run_root"] = config["local_outputs"]["run_root"]
    path = run_root(config) / "configuration" / "scale2000-decoder-joint-rnnt-derived-0030.local.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, derived)
    return path


def _run_base_stage(config: dict[str, Any], *, stage: str, progress_interval_seconds: float, retain_round_checkpoints: bool) -> dict[str, Any]:
    derived_config = _derived_training_config(config)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    if retain_round_checkpoints:
        env["SLAIF_RETAIN_ROUND_CHECKPOINTS"] = "1"
    command = [
        sys.executable,
        "-u",
        str(REPO_ROOT / "scripts/run_scale2000_decoder_joint_rnnt.py"),
        "--config",
        str(derived_config),
        "--stage",
        stage,
        "--progress-interval-seconds",
        str(progress_interval_seconds),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, check=False)
    payload = {"stage": stage, "exit_code": completed.returncode, "derived_config": "local_ignored"}
    atomic_write_json(run_root(config) / "orchestration" / f"{stage}.local.json", payload)
    if completed.returncode != 0:
        raise RuntimeError(f"base decoder+joint stage failed: {stage}")
    return payload


def stage_probe_microbatch(config_path: Path, progress_interval_seconds: float) -> dict[str, Any]:
    config = load_config(config_path)
    return _run_base_stage(config, stage="probe-microbatch", progress_interval_seconds=progress_interval_seconds, retain_round_checkpoints=False)


def stage_train(config_path: Path, progress_interval_seconds: float) -> dict[str, Any]:
    config = load_config(config_path)
    payload = _run_base_stage(config, stage="train", progress_interval_seconds=progress_interval_seconds, retain_round_checkpoints=True)
    checkpoint_rows = run_root(config) / "scale2000_augmented_decoder_joint_rnnt" / "controller-dev" / "round-checkpoints.local.json"
    target = run_root(config) / "controller-dev" / "round-metrics.local.json"
    if checkpoint_rows.exists():
        source = read_json(checkpoint_rows)
        rows = []
        for row in source.get("rounds", []):
            rows.append({**row, "artur_controller_dev_wer": None, "artur_controller_dev_cer": None, "empty_count": None, "delete": None, "insert": None, "substitute": None})
        atomic_write_json(target, {"rounds": rows, "controller_dev_status": "NOT_RUN_PENDING_BATCH1_EVALUATION"})
    return payload


def _controller_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized = metrics["normalized"]
    raw = metrics["raw"]
    return {
        "wer": round(float(normalized["corpus_wer"]), 3),
        "cer": round(float(normalized["corpus_cer"]), 3),
        "empty": int(raw["empty_hypothesis_count"]),
        "delete": round(float(normalized.get("delete_rate", normalized.get("deletion_rate", 0.0))), 6),
        "insert": round(float(normalized.get("insert_rate", normalized.get("insertion_rate", 0.0))), 6),
        "substitute": round(float(normalized.get("substitute_rate", normalized.get("substitution_rate", 0.0))), 6),
    }


def _round_output_dir(config: dict[str, Any], round_index: int) -> Path:
    return run_root(config) / "controller-dev" / f"round_{round_index:02d}"


def _checkpoint_path(checkpoint_root: Path, round_index: int) -> Path:
    if round_index == 0:
        return checkpoint_root / "round_00_base" / "model.local.nemo"
    return checkpoint_root / f"round_{round_index:02d}" / "model.local.nemo"


def _row_from_metrics(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    round_index: int,
    metrics: dict[str, Any],
    arm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    controller = _controller_metric_row(metrics)
    row: dict[str, Any] = {
        "round": round_index,
        "checkpoint_sha256": file_sha256(checkpoint),
        "available": True,
        **controller,
        "artur_controller_dev_wer": controller["wer"],
        "artur_controller_dev_cer": controller["cer"],
        "empty_count": controller["empty"],
    }
    if arm is not None:
        row.update(
            {
                "wall_time_seconds": arm["execution"]["wall_time_seconds"],
                "rows_per_second": arm["utterances_per_second"],
                "real_time_factor": arm["end_to_end_real_time_factor"],
                "peak_validation_gpu_memory_mib": arm["execution"]["monitor"].get("peak_memory_mib"),
            }
        )
    marker = checkpoint.parent / "checkpoint-complete.local.json"
    if marker.exists():
        marker_payload = read_json(marker)
        row.update(
            {
                "optimizer_step": int(marker_payload.get("optimizer_step", 0)),
                "exposures_seen": int(marker_payload.get("exposures_seen", 0)),
                "train_loss": marker_payload.get("train_loss"),
                "synthetic_anchor_probe_loss": marker_payload.get("synthetic_anchor_probe_loss"),
                "synthetic_scale_probe_loss": marker_payload.get("synthetic_scale_probe_loss"),
            }
        )
    return row


def _existing_controller_rows(config: dict[str, Any], records: list[dict[str, Any]], checkpoints: list[tuple[int, Path]]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    metrics_path = run_root(config) / "controller-dev" / "round-metrics.local.json"
    if metrics_path.exists():
        payload = read_json(metrics_path)
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            if row.get("wer") is None or row.get("cer") is None or row.get("empty") is None:
                continue
            rows[int(row["round"])] = row
    for round_index, checkpoint in checkpoints:
        if round_index in rows:
            continue
        predictions_path = _round_output_dir(config, round_index) / "predictions.local.jsonl"
        if not predictions_path.exists():
            continue
        predictions = load_local_predictions(predictions_path)
        if len(predictions) != len(records):
            continue
        rows[round_index] = _row_from_metrics(
            config=config,
            checkpoint=checkpoint,
            round_index=round_index,
            metrics=metrics_for(records, predictions),
        )
    return rows


def stage_evaluate_controller_dev(config_path: Path, *, validation_gpu: str) -> dict[str, Any]:
    config = load_config(config_path)
    manifest = local_path(config["controller_dev"]["manifest"])
    records = load_controller_dev_records(
        manifest,
        expected_sha256=config["controller_dev"]["manifest_sha256"],
        expected_rows=int(config["controller_dev"]["rows"]),
    )
    arm_root = run_root(config) / "scale2000_augmented_decoder_joint_rnnt"
    checkpoint_root = arm_root / "checkpoints"
    checkpoints = []
    for round_index in range(0, int(config["training"]["max_rounds"]) + 1):
        checkpoint = _checkpoint_path(checkpoint_root, round_index)
        if checkpoint.exists():
            checkpoints.append((round_index, checkpoint))
    if not checkpoints:
        raise RuntimeError("no retained round checkpoints found")
    env = runtime_environment()
    env["CUDA_VISIBLE_DEVICES"] = validation_gpu
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    row_map = _existing_controller_rows(config, records, checkpoints)
    rows = [row_map[index] for index, _checkpoint in checkpoints if index in row_map]
    if rows:
        atomic_write_json(run_root(config) / "controller-dev" / "round-metrics.local.json", {"rounds": rows, "controller_dev_status": "PARTIAL" if len(rows) < len(checkpoints) else "PASSED"})
    for round_index, checkpoint in checkpoints:
        if round_index in row_map:
            print(json.dumps({"status": "SKIPPED_ALREADY_SCORED", "round": round_index, "wer": row_map[round_index]["wer"], "cer": row_map[round_index]["cer"], "empty": row_map[round_index]["empty"]}, ensure_ascii=False, sort_keys=True), flush=True)
            continue
        output_dir = _round_output_dir(config, round_index)
        arm = run_batched_arm(
            records=records,
            batch_size=1,
            bucketed=False,
            run_dir=output_dir,
            python_executable=Path(sys.executable),
            nemo_script=nemo_streaming_script(),
            checkpoint=checkpoint,
            context=config["model"]["att_context_size"],
            env=env,
            physical_gpu_index=validation_gpu,
            monitor_interval_seconds=1.0,
        )
        if arm.get("status") != "PASSED":
            raise RuntimeError(f"controller-dev evaluation failed at round {round_index}: {arm.get('status')}")
        predictions = load_local_predictions(output_dir / "predictions.local.jsonl")
        row = _row_from_metrics(config=config, checkpoint=checkpoint, round_index=round_index, metrics=metrics_for(records, predictions), arm=arm)
        row_map[round_index] = row
        rows = [row_map[index] for index, _checkpoint in checkpoints if index in row_map]
        atomic_write_json(run_root(config) / "controller-dev" / "round-metrics.local.json", {"rounds": rows, "controller_dev_status": "PARTIAL" if len(rows) < len(checkpoints) else "PASSED"})
        print(json.dumps({"status": "PASSED", "round": round_index, "wer": row["wer"], "cer": row["cer"], "empty": row["empty"]}, ensure_ascii=False, sort_keys=True), flush=True)
    base_empty = int(next(row for row in rows if row["round"] == 0)["empty"])
    selected = select_round(rows, base_empty_count=base_empty)
    for row in rows:
        row["selected_by_rule"] = selected is not None and int(row["round"]) == int(selected["round"])
    payload = {
        "status": "PASSED",
        "partition_id": config["controller_dev"]["partition_id"],
        "rows": rows,
        "selected_round": int(selected["round"]) if selected else None,
        "base_empty_count": base_empty,
        "checkpoint_count": len(checkpoints),
    }
    atomic_write_json(run_root(config) / "controller-dev" / "round-metrics.local.json", payload)
    print(json.dumps({"status": "PASSED", "selected_round": payload["selected_round"], "checkpoint_count": len(checkpoints)}, ensure_ascii=False, sort_keys=True))
    return payload


def _round_table_from_local(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_root(config) / "controller-dev" / "round-metrics.local.json"
    if not path.exists():
        return []
    payload = read_json(path)
    rows = payload.get("rounds", payload.get("rows", []))
    if not isinstance(rows, list):
        raise RuntimeError("round-metrics.local.json has malformed rounds")
    return rows


def _directional_metrics_from_local(config: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    path = run_root(config) / "post-selection-directional" / "summary.local.json"
    if not path.exists():
        return None
    payload = read_json(path)
    metrics = payload.get("metric_table")
    if not isinstance(metrics, dict):
        raise RuntimeError("post-selection directional summary has malformed metric_table")
    return metrics


def _directional_suite_summary_from_local(config: dict[str, Any]) -> dict[str, Any] | None:
    path = run_root(config) / "post-selection-directional" / "summary.local.json"
    if not path.exists():
        return None
    suite = read_json(path).get("suite")
    if not isinstance(suite, dict):
        return None
    layout = suite.get("layout", {})
    return {
        "rows": suite.get("rows"),
        "prediction_count": suite.get("prediction_count"),
        "audio_duration_seconds": suite.get("audio_duration_seconds"),
        "wall_time_seconds": suite.get("wall_time_seconds"),
        "real_time_factor": suite.get("real_time_factor"),
        "rows_per_second": suite.get("rows_per_second"),
        "audio_seconds_per_wall_second": suite.get("audio_seconds_per_wall_second"),
        "batch_size": layout.get("batch_size"),
        "duration_bucketing": layout.get("bucketed"),
        "batch_count": layout.get("batch_count"),
        "padding_ratio": layout.get("padding_ratio"),
        "gpu_monitor": suite.get("gpu_monitor"),
        "validation_gpu_selector": suite.get("validation_gpu_selector"),
        "sharded_evaluation": suite.get("sharded_evaluation"),
    }


def _base_empty_count(round_rows: list[dict[str, Any]]) -> int:
    for row in round_rows:
        if int(row.get("round", -1)) == 0:
            value = row.get("empty_count")
            if value is None:
                value = row.get("empty")
            return int(value)
    raise RuntimeError("round 0/base controller-dev metrics are required")


def stage_evaluate_directional(config_path: Path, *, validation_gpu: str) -> dict[str, Any]:
    config = load_config(config_path)
    round_rows = _round_table_from_local(config)
    if not round_rows:
        raise RuntimeError("controller-dev round metrics are required before post-selection directional evaluation")
    selected = select_round(round_rows, base_empty_count=_base_empty_count(round_rows))
    if selected is None:
        raise RuntimeError("controller-dev rule did not select a checkpoint")
    selected_round = int(selected["round"])
    checkpoint_root = run_root(config) / "scale2000_augmented_decoder_joint_rnnt" / "checkpoints"
    checkpoint = _checkpoint_path(checkpoint_root, selected_round)
    if not checkpoint.exists():
        raise RuntimeError(f"selected checkpoint for round {selected_round} is missing")

    fast_config = read_json(FAST_DIRECTIONAL_CONFIG)
    suite_records, split_records = load_directional_suite(fast_config)
    output_dir = run_root(config) / "post-selection-directional"
    suite_manifest_sha = write_privacy_safe_suite_manifest(output_dir / "suite-plan.local.jsonl", suite_records)
    arm_dir = output_dir / f"selected_round_{selected_round:02d}"

    env = runtime_environment()
    env["CUDA_VISIBLE_DEVICES"] = validation_gpu
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    arm = run_batched_arm(
        records=suite_records,
        batch_size=int(config["post_selection_directional"]["batch_size"]),
        bucketed=bool(config["post_selection_directional"]["duration_bucketing"]),
        run_dir=arm_dir,
        python_executable=Path(sys.executable),
        nemo_script=nemo_streaming_script(),
        checkpoint=checkpoint,
        context=config["post_selection_directional"]["att_context_size"],
        env=env,
        physical_gpu_index=validation_gpu,
        monitor_interval_seconds=0.5,
    )
    if arm.get("status") != "PASSED":
        raise RuntimeError(f"post-selection directional evaluation failed: {arm.get('status')}")
    predictions = load_local_predictions(arm_dir / "predictions.local.jsonl")
    split_predictions_map = split_predictions(suite_records, split_records, predictions)
    split_summaries = {}
    metric_table = {}
    for split, records in split_records.items():
        metrics = metrics_for(records, split_predictions_map[split])
        split_summaries[split] = {
            "rows": len(records),
            "audio_duration_seconds": round(sum(row.duration for row in records), 6),
            "metrics": metrics,
        }
        metric_table[split] = normalized_metric_row(split_summaries[split])

    before = int(selected["round"])
    after = int(select_round(round_rows, base_empty_count=_base_empty_count(round_rows))["round"])
    if before != after:
        raise RuntimeError("post-selection directional metrics must not change selected round")

    payload = {
        "status": "PASSED",
        "selected_round": selected_round,
        "selected_checkpoint_sha256": file_sha256(checkpoint),
        "suite_rows": int(arm["rows"]),
        "suite_manifest_sha256": suite_manifest_sha,
        "policy": config["post_selection_directional"],
        "suite": {
            "rows": int(arm["rows"]),
            "prediction_count": int(arm["prediction_count"]),
            "audio_duration_seconds": arm["audio_duration_seconds"],
            "wall_time_seconds": arm["execution"]["wall_time_seconds"],
            "real_time_factor": arm["end_to_end_real_time_factor"],
            "rows_per_second": arm["utterances_per_second"],
            "audio_seconds_per_wall_second": arm["end_to_end_audio_seconds_per_wall_second"],
            "layout": arm["layout"],
            "gpu_monitor": arm["execution"]["monitor"],
            "validation_gpu_selector": validation_gpu,
            "sharded_evaluation": False,
        },
        "splits": split_summaries,
        "metric_table": metric_table,
    }
    atomic_write_json(output_dir / "summary.local.json", payload)
    print(json.dumps({"status": "PASSED", "selected_round": selected_round, "metric_table": metric_table}, ensure_ascii=False, sort_keys=True))
    return payload


def _summary_rows(round_rows: list[dict[str, Any]], selected_round: int | None) -> list[dict[str, Any]]:
    rows = []
    for row in round_rows:
        rows.append(
            {
                "round": int(row["round"]),
                "optimizer_step": int(row.get("optimizer_step", 0)),
                "exposures_seen": int(row.get("exposures_seen", 0)),
                "train_loss": row.get("train_loss"),
                "synthetic_anchor_probe": row.get("synthetic_anchor_probe_loss"),
                "synthetic_scale_probe": row.get("synthetic_scale_probe_loss"),
                "artur_controller_dev_wer": row.get("artur_controller_dev_wer") or row.get("wer"),
                "artur_controller_dev_cer": row.get("artur_controller_dev_cer") or row.get("cer"),
                "empty": row.get("empty_count") or row.get("empty"),
                "delete": row.get("delete"),
                "insert": row.get("insert"),
                "substitute": row.get("substitute"),
                "available": bool(row.get("available")),
                "selected_eligible": selected_round is not None and int(row["round"]) == selected_round,
            }
        )
    return rows


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    input_summary_path = run_root(config) / "verification" / "inputs.local.json"
    inputs = read_json(input_summary_path) if input_summary_path.exists() else stage_verify_inputs(config_path)
    round_rows = _round_table_from_local(config)
    if not round_rows:
        round_rows = [{"round": 0, "available": False, "status": "NOT_RUN"}]
    selected = select_round(round_rows, base_empty_count=_base_empty_count(round_rows))
    selected_round = int(selected["round"]) if selected is not None else None
    directional_metrics = _directional_metrics_from_local(config)
    decision = classify_artur_earlystop(
        selected_round=selected_round,
        max_round=int(config["training"]["max_rounds"]),
        controller_rows=round_rows,
        selected_directional_metrics=directional_metrics,
    )
    public_rows = _summary_rows(round_rows, selected_round)
    checkpoint_rows = [redacted_checkpoint_row({**row, "checkpoint_sha256": row.get("checkpoint_sha256")}) for row in round_rows if "round" in row]
    stopped_round = max(int(row["round"]) for row in round_rows if "round" in row)
    final_round = max((row for row in round_rows if "round" in row), key=lambda row: int(row["round"]))
    certificate = {
        "schema_version": "1.0",
        "certificate_id": CERTIFICATE_ID,
        "status": "DIAGNOSTIC_ONLY",
        "work_order_id": "0032",
        "experiment_id": config["experiment_id"],
        "controller_dev_partition_id": config["controller_dev"]["partition_id"],
        "controller_dev_manifest_sha256": config["controller_dev"]["manifest_sha256"],
        "training_source": "scale-2000 augmented corpus v4",
        "trainable_surface": "model.decoder + model.joint only",
        "objective": "audio-conditioned RNNT loss only",
        "accepted_parent": "none",
        "TRAINING_ELIGIBLE_issued": False,
        "prohibited_artifacts_committed": False,
    }
    report = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "work_order_id": "0032",
        "repository_commit": git_head(),
        "classification": decision["classification"],
        "accepted_parent": "none",
        "promotion_eligible": False,
        "training_eligible_issued": False,
        "checkpoint_accepted": False,
        "model_published": False,
        "input_integrity": inputs,
        "governance": {
            "adr_0008_used": True,
            "immutable_gates_used_for_early_stopping": False,
            "controller_dev_used_for_aggregate_run_control_only": bool(round_rows and round_rows[0].get("available")),
        },
        "training": {
            "base_model_revision": config["model"]["revision"],
            "base_checkpoint_sha256": config["model"]["checkpoint_sha256"],
            "nemo_revision": config["model"]["nemo_revision"],
            "training_source": "scale-2000 augmented corpus v4",
            "semantic_rows": config["training"]["semantic_rows"],
            "exposure_records": config["training"]["sample_exposures"],
            "trainable_surface": config["training"]["trainable_surface"],
            "effective_batch": config["training"]["effective_batch_size"],
            "max_rounds": config["training"]["max_rounds"],
            "stopped_round": stopped_round,
            "optimizer_steps_completed": final_round.get("optimizer_step"),
            "exposures_seen": final_round.get("exposures_seen"),
            "operational_stop_rule": config["early_stop_rule"].get("human_runtime_override"),
        },
        "controller_dev_curve": public_rows,
        "selection": {
            "selected_round": selected_round,
            "selected_checkpoint_sha256": selected.get("checkpoint_sha256") if selected else None,
            "reason": decision["classification"],
        },
        "post_selection_directional_metrics": {
            "base": BASE_DIRECTIONAL_METRICS,
            "scale2000_joint_adapter": SCALE2000_JOINT_ADAPTER_METRICS,
            "pr36_round20_decoder_joint": PR36_DECODER_JOINT_METRICS,
            "selected_early_stop_checkpoint": directional_metrics,
        },
        "post_selection_directional_evaluation": _directional_suite_summary_from_local(config),
        "round_checkpoint_manifest": checkpoint_rows,
        "artifact_integrity": {
            "per_round_checkpoints_retained_locally": any(row.get("checkpoint_sha256") for row in round_rows),
            "local_checkpoint_paths_committed": False,
            "selected_checkpoint_committed": False,
            "predictions_committed": False,
            "raw_refs_hyps_committed": False,
            "local_manifests_committed": False,
        },
        "safety_confirmations": {
            "no_real_data_used_for_training": True,
            "no_immutable_gate_used_for_early_stopping": True,
            "no_raw_controller_dev_references_or_hypotheses_public": True,
            "no_encoder_prompt_tokenizer_changes_expected": True,
            "no_text_only_objective": True,
            "no_training_eligible": True,
            "accepted_parent": "none",
        },
        "known_limitations": [
            "Controller-dev is spent development data once used for selection.",
            "Directional post-selection metrics are noncanonical and promotion-ineligible.",
            "This report is not model acceptance evidence.",
        ],
    }
    assert_no_raw_report_material(certificate)
    atomic_write_json(CERTIFICATE_PATH, certificate)
    report["authorization_certificate_sha256"] = file_sha256(CERTIFICATE_PATH)
    assert_no_raw_report_material(report)
    atomic_write_json(REPORT_JSON, report)
    _write_md(report, REPORT_MD)
    print(json.dumps({"status": "PASSED", "classification": decision["classification"], "selected_round": selected_round}, ensure_ascii=False, sort_keys=True))
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_md(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Experiment 0019: Scale-2000 Decoder+Joint RNNT with ARTUR Controller-Dev Early Stopping",
        "",
        f"Classification: `{report['classification']}`",
        "",
        "This is diagnostic-only evidence. ARTUR controller-dev may be used for aggregate run-control under ADR 0008; immutable gates remain unavailable for checkpoint selection.",
        "",
        f"Stopped round: `{report['training']['stopped_round']}`. Selected round: `{report['selection']['selected_round']}`.",
        "",
        "Operational stop rule: after the human runtime override, training stopped after three further evaluated rounds failed to produce a new raw best ARTUR controller-dev WER. The checkpoint selection rule remained the predeclared earliest-within-tolerance ARTUR controller-dev rule.",
        "",
        "## Controller-Dev Early-Stop Curve",
        "",
        "| Round | Optimizer step | Exposures seen | Train loss | Synthetic anchor probe | Synthetic scale probe | ARTUR controller-dev WER | CER | Empty | Delete | Insert | Substitute | Selected eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["controller_dev_curve"]:
        lines.append(
            "| {round} | {step} | {exposures} | {train} | {anchor} | {scale} | {wer} | {cer} | {empty} | {delete} | {insert} | {substitute} | {selected} |".format(
                round=row["round"],
                step=row["optimizer_step"],
                exposures=row["exposures_seen"],
                train=_fmt(row["train_loss"]),
                anchor=_fmt(row["synthetic_anchor_probe"]),
                scale=_fmt(row["synthetic_scale_probe"]),
                wer=_fmt(row["artur_controller_dev_wer"]),
                cer=_fmt(row["artur_controller_dev_cer"]),
                empty=_fmt(row["empty"]),
                delete=_fmt(row["delete"]),
                insert=_fmt(row["insert"]),
                substitute=_fmt(row["substitute"]),
                selected="yes" if row["selected_eligible"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Post-Selection Directional Metrics",
            "",
            "| Split | Base directional WER/CER/empty | Scale-2000 joint-adapter WER/CER/empty | PR #36 round-20 decoder+joint WER/CER/empty | Selected early-stop checkpoint WER/CER/empty |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    metrics = report["post_selection_directional_metrics"]
    for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout", "fleurs_v2", "artur_j"):
        base = metrics["base"][split]
        joint = metrics["scale2000_joint_adapter"][split]
        pr36 = metrics["pr36_round20_decoder_joint"][split]
        selected = metrics["selected_early_stop_checkpoint"]
        if selected is None:
            selected_text = "not run"
        else:
            value = selected[split]
            selected_text = f"{value['wer']}/{value['cer']}/{value['empty']}"
        lines.append(
            f"| {split} | {base['wer']}/{base['cer']}/{base['empty']} | {joint['wer']}/{joint['cer']}/{joint['empty']} | {pr36['wer']}/{pr36['cer']}/{pr36['empty']} | {selected_text} |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- No immutable gate may be used for early stopping.",
            "- No raw controller-dev references or hypotheses are included.",
            "- No checkpoint, prediction, local manifest, audio, or model artifact is committed.",
            "- `accepted_parent` remains `none`.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", required=True, choices=["verify-inputs", "probe-run-control", "probe-microbatch", "train", "evaluate-controller-dev", "evaluate-directional", "summarize"])
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    parser.add_argument("--training-gpu", default=os.environ.get("SLAIF_TRAINING_GPU", "0"))
    parser.add_argument("--validation-gpu", default=os.environ.get("SLAIF_VALIDATION_GPU", "1"))
    parser.add_argument("--sequential-validation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if args.stage == "verify-inputs":
        stage_verify_inputs(config_path)
    elif args.stage == "probe-run-control":
        stage_probe_run_control(config_path, training_gpu=args.training_gpu, validation_gpu=args.validation_gpu, sequential=args.sequential_validation)
    elif args.stage == "probe-microbatch":
        stage_probe_microbatch(config_path, args.progress_interval_seconds)
    elif args.stage == "train":
        stage_train(config_path, args.progress_interval_seconds)
    elif args.stage == "evaluate-controller-dev":
        stage_evaluate_controller_dev(config_path, validation_gpu=args.validation_gpu)
    elif args.stage == "evaluate-directional":
        stage_evaluate_directional(config_path, validation_gpu=args.validation_gpu)
    elif args.stage == "summarize":
        stage_summarize(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
