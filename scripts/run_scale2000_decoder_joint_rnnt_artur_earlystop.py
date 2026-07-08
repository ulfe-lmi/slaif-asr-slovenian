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
    redacted_checkpoint_row,
    select_round,
    validate_agents_controller_dev_exception,
    validate_earlystop_config,
)
from slaif_asr.batched_streaming import file_sha256
from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.emission_rnnt_finetune import BASE_DIRECTIONAL_METRICS, SCALE2000_JOINT_ADAPTER_METRICS, verify_all_inputs


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/scale2000_decoder_joint_rnnt_artur_earlystop.json"
BASE_CONFIG = REPO_ROOT / "configs/experiments/scale2000_decoder_joint_rnnt_v1.json"
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


def _round_table_from_local(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = run_root(config) / "controller-dev" / "round-metrics.local.json"
    if not path.exists():
        return []
    rows = read_json(path).get("rounds", [])
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
    selected = select_round(round_rows, base_empty_count=0)
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
    parser.add_argument("--stage", required=True, choices=["verify-inputs", "probe-run-control", "probe-microbatch", "train", "summarize"])
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
    elif args.stage == "summarize":
        stage_summarize(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
