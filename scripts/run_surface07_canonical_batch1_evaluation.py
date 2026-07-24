#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.batched_streaming import (
    StreamingRecord,
    ensure_gpu_idle,
    file_sha256,
    load_gate_records,
    load_local_predictions,
    metrics_for,
    run_batched_arm,
)
from slaif_asr.canonical_challenger_evaluation import (
    CANDIDATE_IDS,
    REAL_GATE_SPLITS,
    assert_public_report_safe,
    classify_canonical,
    metric_row,
    validate_canonical_config,
)
from slaif_asr.config import REPO_ROOT
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.real_eval import atomic_write_json, atomic_write_text


DEFAULT_CONFIG = Path("configs/experiments/surface07_canonical_batch1_evaluation_v1.json")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def local_path(path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "runs":
        runs_root = os.environ.get("SLAIF_ASR_RUNS_ROOT")
        if runs_root:
            return Path(runs_root).expanduser().resolve() / Path(*path.parts[1:])
    return REPO_ROOT / path


def git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def load_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(config_path)
    policy = read_json(repo_path(config["evaluation_policy"]))
    validate_canonical_config(config, policy)
    return config, policy


def run_root(config: dict[str, Any]) -> Path:
    return local_path(config["local_outputs"]["run_root"])


def candidate_entry(config: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    return next(row for row in config["candidates"] if row["candidate_id"] == candidate_id)


def candidate_path(entry: dict[str, Any]) -> Path:
    return local_path(entry["checkpoint"])


def remap_gate_records(split: str, records: Sequence[StreamingRecord], start_index: int) -> list[StreamingRecord]:
    return [
        StreamingRecord(
            sample_id=f"{split}:{index:04d}",
            audio_filepath=row.audio_filepath,
            duration=row.duration,
            reference=row.reference,
            original_index=start_index + index,
            row={"split": split, "source_order": row.original_index},
        )
        for index, row in enumerate(records)
    ]


def load_suite(config: dict[str, Any]) -> tuple[list[StreamingRecord], dict[str, list[StreamingRecord]]]:
    suite: list[StreamingRecord] = []
    splits: dict[str, list[StreamingRecord]] = {}
    for split in REAL_GATE_SPLITS:
        gate = config["gates"][split]
        records = load_gate_records(
            local_path(gate["manifest"]),
            expected_sha256=gate["manifest_sha256"],
            expected_rows=int(gate["rows"]),
            gate_id=gate["gate_id"],
        )
        remapped = remap_gate_records(split, records, len(suite))
        splits[split] = remapped
        suite.extend(remapped)
    if len(suite) != 1090:
        raise RuntimeError(f"canonical real-gate suite must contain 1090 rows, found {len(suite)}")
    return suite, splits


def verify_nemo(config: dict[str, Any]) -> str:
    nemo_root = repo_path(config["nemo"]["source_tree"])
    completed = subprocess.run(
        ["git", "-C", str(nemo_root), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    actual = completed.stdout.strip()
    if actual != config["nemo"]["revision"]:
        raise RuntimeError(f"NeMo revision mismatch: {actual}")
    script = repo_path(config["nemo"]["streaming_script"])
    if not script.is_file():
        raise FileNotFoundError("pinned NeMo streaming script is unavailable")
    return actual


def configure_runtime() -> tuple[Any, dict[str, Any]]:
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be exactly 0")
    hardware = require_single_visible_cuda()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("TF32 must remain disabled")
    idle = ensure_gpu_idle(
        physical_gpu_index=hardware.physical_selector,
        max_memory_mib=1024,
        max_utilization_percent=10,
    )
    runtime = {
        "gpu_model": hardware.device_name,
        "cuda_visible_devices": hardware.cuda_visible_devices,
        "visible_device_count": hardware.visible_device_count,
        "logical_device": hardware.logical_device,
        "total_vram_mib": hardware.total_vram_mib,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "idle_probe": {
            "memory_used_mib": idle["memory_used_mib"],
            "utilization_percent": idle["utilization_percent"],
        },
    }
    return hardware, runtime


def verify_candidates(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for entry in config["candidates"]:
        candidate_id = entry["candidate_id"]
        checkpoint = candidate_path(entry)
        if not checkpoint.is_file():
            inventory[candidate_id] = {
                "available": False,
                "required": bool(entry["required"]),
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "source_experiment": entry["source_experiment"],
                "status": "NOT_RUN_CHECKPOINT_UNAVAILABLE",
            }
            continue
        actual = file_sha256(checkpoint)
        if actual != entry["checkpoint_sha256"]:
            raise RuntimeError(f"{candidate_id} checkpoint SHA256 mismatch: {actual}")
        inventory[candidate_id] = {
            "available": True,
            "required": bool(entry["required"]),
            "checkpoint_sha256": actual,
            "source_experiment": entry["source_experiment"],
            "status": "VERIFIED",
        }
    return inventory


def stage_verify(config_path: Path) -> dict[str, Any]:
    config, policy = load_config(config_path)
    hardware, runtime = configure_runtime()
    suite, splits = load_suite(config)
    candidates = verify_candidates(config)
    payload = {
        "status": "PASSED",
        "repository_commit": git_head(),
        "configuration_sha256": file_sha256(config_path),
        "policy_sha256": file_sha256(repo_path(config["evaluation_policy"])),
        "policy": policy,
        "nemo_revision": verify_nemo(config),
        "runtime": runtime,
        "candidates": candidates,
        "splits": {
            split: {
                "gate_id": config["gates"][split]["gate_id"],
                "manifest_sha256": config["gates"][split]["manifest_sha256"],
                "rows": len(records),
                "audio_duration_seconds": round(sum(row.duration for row in records), 6),
            }
            for split, records in splits.items()
        },
        "suite": {
            "rows": len(suite),
            "audio_duration_seconds": round(sum(row.duration for row in suite), 6),
        },
        "training_started": False,
        "controller_development_used": False,
    }
    atomic_write_json(run_root(config) / "verification.local.json", payload)
    if not candidates["surface07_round13"]["available"]:
        raise RuntimeError("CANONICAL_BLOCKED_SURFACE07_CHECKPOINT_UNAVAILABLE")
    print(
        json.dumps(
            {
                "status": "PASSED",
                "gpu": hardware.device_name,
                "suite_rows": len(suite),
                "candidates": {
                    candidate_id: row["status"]
                    for candidate_id, row in candidates.items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def split_predictions(
    split_records: dict[str, list[StreamingRecord]],
    predictions: dict[str, str],
) -> dict[str, dict[str, str]]:
    expected = {row.sample_id for records in split_records.values() for row in records}
    if set(predictions) != expected:
        raise RuntimeError("canonical prediction identity mismatch")
    return {
        split: {row.sample_id: predictions[row.sample_id] for row in records}
        for split, records in split_records.items()
    }


def stage_evaluate(config_path: Path, candidate_id: str, *, force: bool = False) -> dict[str, Any]:
    config, policy = load_config(config_path)
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError(f"unknown candidate: {candidate_id}")
    verification_path = run_root(config) / "verification.local.json"
    if not verification_path.exists():
        raise FileNotFoundError("run --stage verify before evaluation")
    verification = read_json(verification_path)
    candidate = verification["candidates"][candidate_id]
    if not candidate["available"]:
        print(json.dumps({"candidate": candidate_id, "status": "NOT_RUN_CHECKPOINT_UNAVAILABLE"}), flush=True)
        return candidate

    candidate_dir = run_root(config) / "candidates" / candidate_id
    summary_path = candidate_dir / "canonical-summary.local.json"
    if summary_path.exists() and not force:
        prior = read_json(summary_path)
        if prior.get("status") == "PASSED" and prior.get("checkpoint_sha256") == candidate["checkpoint_sha256"]:
            print(json.dumps({"candidate": candidate_id, "status": "ALREADY_PASSED"}), flush=True)
            return prior

    hardware, runtime = configure_runtime()
    suite, split_records = load_suite(config)
    entry = candidate_entry(config, candidate_id)
    checkpoint = candidate_path(entry)
    if file_sha256(checkpoint) != entry["checkpoint_sha256"]:
        raise RuntimeError(f"{candidate_id} checkpoint identity changed after verification")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": hardware.physical_selector,
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONUNBUFFERED": "1",
            "NEMO_ROOT": str(repo_path(config["nemo"]["source_tree"])),
        }
    )
    print(
        json.dumps(
            {
                "candidate": candidate_id,
                "event": "canonical_evaluation_start",
                "rows": len(suite),
                "batch_size": 1,
                "duration_bucketing": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    arm = run_batched_arm(
        records=suite,
        batch_size=int(policy["batch_size"]),
        bucketed=bool(policy["duration_bucketing"]),
        run_dir=candidate_dir / "nemo-run",
        python_executable=Path(sys.executable),
        nemo_script=repo_path(config["nemo"]["streaming_script"]),
        checkpoint=checkpoint,
        context=policy["att_context_size"],
        env=env,
        physical_gpu_index=hardware.physical_selector,
        monitor_interval_seconds=0.5,
    )
    if arm.get("status") != "PASSED":
        raise RuntimeError(f"{candidate_id} canonical evaluation failed: {arm.get('status')}")
    predictions = load_local_predictions(candidate_dir / "nemo-run" / "predictions.local.jsonl")
    predictions_by_split = split_predictions(split_records, predictions)
    split_summaries: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        metrics = metrics_for(records, predictions_by_split[split])
        split_summaries[split] = {
            "rows": len(records),
            "audio_duration_seconds": round(sum(row.duration for row in records), 6),
            "metrics": metrics,
            "metric_row": metric_row({"metrics": metrics}),
        }
    summary = {
        "status": "PASSED",
        "candidate_id": candidate_id,
        "source_experiment": entry["source_experiment"],
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "policy": policy,
        "runtime": runtime,
        "suite": {
            "rows": int(arm["rows"]),
            "prediction_count": int(arm["prediction_count"]),
            "audio_duration_seconds": arm["audio_duration_seconds"],
            "wall_time_seconds": arm["execution"]["wall_time_seconds"],
            "rows_per_second": arm["utterances_per_second"],
            "real_time_factor": arm["end_to_end_real_time_factor"],
            "audio_seconds_per_wall_second": arm["end_to_end_audio_seconds_per_wall_second"],
            "gpu_monitor": arm["execution"]["monitor"],
            "layout": {
                "batch_size": arm["layout"]["batch_size"],
                "bucketed": arm["layout"]["bucketed"],
                "batch_count": arm["layout"]["batch_count"],
                "padding_ratio": arm["layout"]["padding_ratio"],
            },
        },
        "splits": split_summaries,
        "training_started": False,
        "controller_development_used": False,
    }
    atomic_write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "candidate": candidate_id,
                "event": "canonical_evaluation_complete",
                "metrics": {split: row["metric_row"] for split, row in split_summaries.items()},
                "wall_time_seconds": summary["suite"]["wall_time_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def directional_context() -> dict[str, dict[str, dict[str, Any]]]:
    pr36 = read_json(REPO_ROOT / "docs/experiments/0017-scale2000-decoder-joint-rnnt-directional.json")
    surface06 = read_json(REPO_ROOT / "docs/experiments/0026-fixed-scale2000-surface06-last-four-encoder-blocks.json")
    surface07 = read_json(REPO_ROOT / "docs/experiments/0027-fixed-scale2000-surface07-topencoder-fusion.json")
    return {
        "pr36_round20": {
            split: pr36["directional_evaluation"]["metrics"]["decoder_joint_rnnt"][split]
            for split in REAL_GATE_SPLITS
        },
        "surface06_round05": {
            split: surface06["directional_evaluation"]["metrics"]["surface06_selected"][split]
            for split in REAL_GATE_SPLITS
        },
        "surface07_round13": {
            split: surface07["directional_evaluation"]["metrics"]["surface07_selected"][split]
            for split in REAL_GATE_SPLITS
        },
    }


def format_metric(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NOT_RUN_CHECKPOINT_UNAVAILABLE"
    return f"{float(row['wer']):.3f} / {float(row['cer']):.3f} / {int(row['empty'])}"


def markdown_report(public: dict[str, Any]) -> str:
    metrics = public["canonical_metrics"]
    directional = public["directional_context"]
    lines = [
        "# Experiment 0028: Surface07 Canonical Batch-1 Evaluation",
        "",
        f"Classification: `{public['classification']}`",
        "",
        "This evaluation-only proof compares named, preselected challengers under the scientific reference protocol. It did not train, tune, or reselect any checkpoint.",
        "",
        "## Candidate Inventory",
        "",
        "| Candidate | Available | SHA256 | Source experiment | Evaluated |",
        "|---|---:|---|---|---:|",
    ]
    for row in public["candidate_inventory"]:
        lines.append(
            f"| {row['candidate_id']} | {str(row['available']).lower()} | `{row['checkpoint_sha256']}` | {row['source_experiment']} | {str(row['evaluated']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Canonical Metrics",
            "",
            "| Split | Base canonical WER/CER/empty | PR #36 canonical WER/CER/empty | Surface06 canonical WER/CER/empty | Surface07 canonical WER/CER/empty |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in REAL_GATE_SPLITS:
        lines.append(
            f"| {split} | {format_metric(metrics.get('base', {}).get(split))} | {format_metric(metrics.get('pr36_round20', {}).get(split))} | {format_metric(metrics.get('surface06_round05', {}).get(split))} | {format_metric(metrics.get('surface07_round13', {}).get(split))} |"
        )
    lines.extend(
        [
            "",
            "Values are normalized corpus WER / CER / empty-hypothesis count using `sl-asr-normalization-v1`.",
            "",
            "## Directional Context",
            "",
            "| Split | PR #36 directional | Surface06 directional | Surface07 directional | Canonical interpretation |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for split in REAL_GATE_SPLITS:
        lines.append(
            f"| {split} | {format_metric(directional['pr36_round20'][split])} | {format_metric(directional['surface06_round05'][split])} | {format_metric(directional['surface07_round13'][split])} | Canonical batch-1 ordering is reported above; directional metrics did not alter candidate selection. |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            f"- Policy: `{public['protocol']['policy_id']}`.",
            f"- Batch size: {public['protocol']['batch_size']}.",
            f"- Duration bucketing: {str(public['protocol']['duration_bucketing']).lower()}.",
            f"- Precision: {public['protocol']['precision']}.",
            f"- TF32: {str(public['protocol']['tf32']).lower()}.",
            f"- Target language: `{public['protocol']['target_lang']}`.",
            f"- Attention context: `{public['protocol']['att_context_size']}`.",
            f"- Normalization: `{public['protocol']['normalization']}`.",
            f"- GPU: {public['runtime']['gpu_model']}; one visible CUDA device.",
            "",
            "## Boundaries",
            "",
            "- `accepted_parent` remains `none`.",
            "- Promotion eligibility and `TRAINING_ELIGIBLE` remain false.",
            "- No controller-development partition was loaded.",
            "- No checkpoint was accepted and no model was published.",
            "- Checkpoints, predictions, manifests, logs, raw references, and raw hypotheses remain ignored local artifacts.",
            "",
        ]
    )
    return "\n".join(lines)


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config, policy = load_config(config_path)
    verification = read_json(run_root(config) / "verification.local.json")
    summaries: dict[str, dict[str, Any]] = {}
    candidate_inventory = []
    for entry in config["candidates"]:
        candidate_id = entry["candidate_id"]
        verified = verification["candidates"][candidate_id]
        summary_path = run_root(config) / "candidates" / candidate_id / "canonical-summary.local.json"
        evaluated = summary_path.exists()
        if evaluated:
            summary = read_json(summary_path)
            if summary.get("status") != "PASSED":
                raise RuntimeError(f"{candidate_id} local summary is not passed")
            if summary.get("checkpoint_sha256") != entry["checkpoint_sha256"]:
                raise RuntimeError(f"{candidate_id} local summary checkpoint mismatch")
            if summary["policy"] != policy:
                raise RuntimeError(f"{candidate_id} local summary protocol mismatch")
            summaries[candidate_id] = summary
        candidate_inventory.append(
            {
                "candidate_id": candidate_id,
                "available": bool(verified["available"]),
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "source_experiment": entry["source_experiment"],
                "evaluated": evaluated,
                "status": "PASSED" if evaluated else verified["status"],
            }
        )
    if "surface07_round13" not in summaries:
        classification = "CANONICAL_BLOCKED_SURFACE07_CHECKPOINT_UNAVAILABLE"
    else:
        metric_table = {
            candidate_id: {
                split: summary["splits"][split]["metric_row"]
                for split in REAL_GATE_SPLITS
            }
            for candidate_id, summary in summaries.items()
        }
        classification = classify_canonical(metric_table)
    metric_table = {
        candidate_id: {
            split: summary["splits"][split]["metric_row"]
            for split in REAL_GATE_SPLITS
        }
        for candidate_id, summary in summaries.items()
    }
    public = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "EVALUATION_ONLY",
        "classification": classification,
        "repository_commit": git_head(),
        "configuration_sha256": file_sha256(config_path),
        "accepted_parent": "none",
        "promotion_eligible": False,
        "training_eligible": False,
        "checkpoint_accepted": False,
        "model_published": False,
        "training_started": False,
        "checkpoint_selection_changed": False,
        "controller_development_used": False,
        "candidate_inventory": candidate_inventory,
        "canonical_metrics": metric_table,
        "directional_context": directional_context(),
        "protocol": policy,
        "runtime": {
            **verification["runtime"],
            "nemo_revision": verification["nemo_revision"],
        },
        "splits": verification["splits"],
        "evaluation_runs": {
            candidate_id: {
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "rows": summary["suite"]["rows"],
                "audio_duration_seconds": summary["suite"]["audio_duration_seconds"],
                "wall_time_seconds": summary["suite"]["wall_time_seconds"],
                "rows_per_second": summary["suite"]["rows_per_second"],
                "real_time_factor": summary["suite"]["real_time_factor"],
                "peak_gpu_memory_mib": summary["suite"]["gpu_monitor"]["peak_memory_mib"],
                "mean_gpu_utilization_percent": summary["suite"]["gpu_monitor"]["mean_utilization_percent"],
                "p95_gpu_utilization_percent": summary["suite"]["gpu_monitor"]["p95_utilization_percent"],
            }
            for candidate_id, summary in summaries.items()
        },
        "safety": {
            "training_started": False,
            "controller_development_used": False,
            "raw_references_or_hypotheses_committed": False,
            "predictions_committed": False,
            "checkpoint_or_model_committed": False,
            "local_manifests_committed": False,
            "training_eligible_issued": False,
        },
        "limitations": [
            "Only the two immutable Slovenian real gates were evaluated; optional synthetic holdouts were out of scope.",
            "Canonical evidence remains evaluation-only and does not accept or publish a checkpoint.",
        ],
    }
    assert_public_report_safe(public)
    json_path = repo_path(config["public_outputs"]["json"])
    markdown_path = repo_path(config["public_outputs"]["markdown"])
    atomic_write_json(json_path, public)
    atomic_write_text(markdown_path, markdown_report(public))
    print(
        json.dumps(
            {
                "status": "PASSED",
                "classification": classification,
                "canonical_metrics": metric_table,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate named Slovenian ASR challengers in canonical batch-1 mode.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("verify", "evaluate", "summarize"), required=True)
    parser.add_argument("--candidate", choices=(*CANDIDATE_IDS, "all"))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = repo_path(args.config)
    if args.stage == "verify":
        stage_verify(config_path)
    elif args.stage == "evaluate":
        if args.candidate is None:
            raise SystemExit("--candidate is required for --stage evaluate")
        candidate_ids = CANDIDATE_IDS if args.candidate == "all" else (args.candidate,)
        for candidate_id in candidate_ids:
            stage_evaluate(config_path, candidate_id, force=args.force)
    else:
        stage_summarize(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
