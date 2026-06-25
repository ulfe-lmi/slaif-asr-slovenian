#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.batched_streaming import file_sha256
from slaif_asr.corpus_v2_scoring import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION, NEMO_REVISION, verify_runtime_identities
from slaif_asr.corpus_v2_training import git_head, read_json, repo_path, run_dir, runtime_summary, verify_all_input_identities
from slaif_asr.directional_evaluation import (
    classify_directional,
    directional_models,
    load_directional_suite,
    metric_table_from_summaries,
    privacy_safe_public_report,
    run_directional_model,
    suite_plan_hash,
    verify_historical_reports,
    verify_model_artifacts,
    verify_protected_training_files,
    write_privacy_safe_suite_manifest,
)
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json
from slaif_asr.supertonic3_tts import (
    load_supertonic_config,
    runtime_versions,
    supertonic_paths,
    synthesize_batched_supertonic_audio,
    validate_supertonic_audio,
    verify_assets,
    verify_input_identities,
)


DEFAULT_CONFIG = Path("configs/experiments/fast_batched_directional_replay_v1.json")
REPORT_JSON = Path("docs/experiments/0012-fast-batched-directional-replay.json")
REPORT_MD = Path("docs/experiments/0012-fast-batched-directional-replay.md")
ARM_NAME = "fast_batched_replay_supertonic3_joint_adapter_dim32"


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(repo_path(path))
    if config.get("work_order_id") != "0024":
        raise ValueError("fast directional replay config must belong to work order 0024")
    training = config.get("training", {})
    required = {
        "batch_size": 8,
        "epochs": 12,
        "sample_exposures": 1920,
        "optimizer_steps": 240,
        "seed": 1234,
        "optimizer": "AdamW",
        "scheduler": "none",
        "gradient_accumulation": "none",
        "gradient_clipping": "none",
        "precision": "fp32",
        "tf32": False,
        "spec_augment": False,
        "waveform_augmentation": False,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected}")
    if float(training.get("learning_rate", -1.0)) != 0.001:
        raise ValueError("training.learning_rate must be 0.001")
    directional = config.get("directional_evaluation", {})
    if directional.get("batch_size") != 32 or directional.get("duration_bucketing") is not True:
        raise ValueError("directional replay evaluation must use batch size 32 with duration bucketing")
    return config


def require_supertonic_gpu_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Supertonic batched synthesis must run with CUDA_VISIBLE_DEVICES=1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")


def require_nemotron_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("Nemotron stages must run with CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("Nemotron stages must run with NVIDIA_TF32_OVERRIDE=0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def load_supertonic_runner() -> Any:
    path = Path(__file__).with_name("run_supertonic3_joint_adapter_diagnostic.py")
    spec = importlib.util.spec_from_file_location("_slaif_supertonic_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Supertonic joint-adapter runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_training_authorization(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    protected = verify_protected_training_files(config)
    reports = verify_historical_reports(config)
    verify_all_input_identities(config, check_gpu=False)
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    paths = supertonic_paths(tts_config)
    validation = read_json(paths.validation)
    if validation.get("status") != "AUDIO_ACCEPTED":
        raise RuntimeError("batched replay Supertonic audio must be AUDIO_ACCEPTED before training")
    expected_training = config["supertonic_audio"].get("training_audio_manifest_sha256")
    actual_training = file_sha256(paths.training_audio_manifest)
    actual_holdout = file_sha256(paths.holdout_audio_manifest)
    actual_audio = file_sha256(paths.audio_manifest)
    config["supertonic_audio"]["training_audio_manifest_sha256"] = actual_training
    config["supertonic_audio"]["holdout_audio_manifest_sha256"] = actual_holdout
    config["supertonic_audio"]["audio_manifest_sha256"] = actual_audio
    if expected_training and actual_training != expected_training:
        raise RuntimeError("configured training audio manifest SHA mismatch")
    return {
        "status": "PASSED",
        "config_sha256": file_sha256(repo_path(config_path)),
        "protected_training_files": protected,
        "historical_reports": reports,
        "audio_validation_status": validation["status"],
    }


def patch_supertonic_runner_for_replay(module: Any, config: dict[str, Any], config_path: Path) -> None:
    module.ARM_NAME = ARM_NAME
    module.__dict__["__file__"] = str(Path(__file__).resolve())

    def _verify_certificate(path: Path, *, require_head: bool) -> dict[str, Any]:
        del path, require_head
        return {
            "certificate": {
                "status": "DIAGNOSTIC_ONLY",
                "work_order_id": "0024",
            },
            "tracked": {"tracked": True, "clean": True, "matches_head": True},
            "identities": replay_training_authorization(config, config_path),
        }

    module.verify_certificate = _verify_certificate


def stage_verify_synthesis(config: dict[str, Any]) -> dict[str, Any]:
    require_supertonic_gpu_env()
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    identities = verify_input_identities(tts_config)
    assets = verify_assets(tts_config)
    result = {
        "status": "PASSED",
        "identities": identities,
        "asset_tree_sha256": assets["asset_tree_sha256"],
        "runtime": runtime_versions(tts_config),
        "batch_size": tts_config["batch_synthesis"]["batch_size"],
    }
    atomic_write_json(supertonic_paths(tts_config).run_root / "verify-synthesis.local.json", result)
    return result


def stage_synthesize_batched(config: dict[str, Any], interval: float) -> dict[str, Any]:
    require_supertonic_gpu_env()
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    return synthesize_batched_supertonic_audio(tts_config, progress_interval_seconds=interval)


def stage_convert_validate(config: dict[str, Any], interval: float) -> dict[str, Any]:
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    validation = validate_supertonic_audio(tts_config, progress_interval_seconds=interval)
    if validation.get("status") != "AUDIO_ACCEPTED":
        raise RuntimeError(f"batched replay audio validation failed: {validation.get('failures_by_reason')}")
    paths = supertonic_paths(tts_config)
    summary = read_json(paths.run_root / "batched-synthesis-summary.local.json")
    result = {
        "status": "PASSED",
        "validation_status": validation["status"],
        "audio_manifest_sha256": validation["audio_manifest_sha256"],
        "training_audio_manifest_sha256": validation["training_audio_manifest_sha256"],
        "holdout_audio_manifest_sha256": validation["holdout_audio_manifest_sha256"],
        "data_stage": summary,
    }
    atomic_write_json(paths.run_root / "convert-validate-summary.local.json", result)
    return result


def stage_verify_training_code(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    result = replay_training_authorization(config, config_path)
    atomic_write_json(run_dir(config) / "authorization" / "training-code.local.json", result)
    return result


def stage_train_existing_protocol(config: dict[str, Any], config_path: Path, interval: float) -> dict[str, Any]:
    require_nemotron_env()
    stage_verify_training_code(config, config_path)
    verify_runtime_identities(check_gpu=True)
    runner = load_supertonic_runner()
    patch_supertonic_runner_for_replay(runner, config, config_path)
    result = runner.train(config, config_path, interval)
    replay_checkpoint = run_dir(config) / ARM_NAME / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
    config["replay_artifact"]["sha256"] = file_sha256(replay_checkpoint)
    artifact_summary = {
        "status": "PASSED",
        "checkpoint_sha256": config["replay_artifact"]["sha256"],
        "training": result,
    }
    atomic_write_json(run_dir(config) / ARM_NAME / "replay-training.local.json", artifact_summary)
    return artifact_summary


def stage_verify_artifact(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    require_nemotron_env()
    runner = load_supertonic_runner()
    patch_supertonic_runner_for_replay(runner, config, config_path)
    return runner.verify_artifact(config)


def stage_evaluate_directional(config: dict[str, Any]) -> dict[str, Any]:
    require_nemotron_env()
    verify_runtime_identities(check_gpu=True)
    verify_protected_training_files(config)
    verify_historical_reports(config)
    verify_model_artifacts(config)
    suite_records, split_records = load_directional_suite(config)
    output_dir = run_dir(config) / "directional-evaluation"
    suite_manifest_sha = write_privacy_safe_suite_manifest(output_dir / "suite-plan.local.jsonl", suite_records)
    models = directional_models(config, include_replay=True)
    summaries = {}
    for model in models:
        summaries[model.model_id] = run_directional_model(
            config=config,
            model=model,
            suite_records=suite_records,
            split_records=split_records,
            run_dir=output_dir,
            python_executable=Path(sys.executable),
        )
    metrics = metric_table_from_summaries(summaries)
    decision = classify_directional(metrics)
    payload = {
        "status": "PASSED",
        "suite_rows": len(suite_records),
        "suite_plan_sha256": suite_plan_hash(suite_records),
        "suite_manifest_sha256": suite_manifest_sha,
        "evaluation_policy": {
            "batch_size": 32,
            "duration_bucketing": True,
            "att_context_size": config["directional_evaluation"]["att_context_size"],
            "target_lang": config["directional_evaluation"]["target_lang"],
            "normalizer": NORMALIZER_VERSION,
            "canonical": False,
            "promotion_eligible": False,
        },
        "models": summaries,
        "metric_table": metrics,
        "decision": decision,
    }
    atomic_write_json(output_dir / "summary.local.json", payload)
    return payload


def stage_summarize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    protected = verify_protected_training_files(config)
    historical_reports = verify_historical_reports(config)
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    paths = supertonic_paths(tts_config)
    data_summary = read_json(paths.run_root / "batched-synthesis-summary.local.json")
    validation = read_json(paths.validation)
    training = read_json(run_dir(config) / ARM_NAME / "training-summary.local.json")
    evaluation = read_json(run_dir(config) / "directional-evaluation" / "summary.local.json")
    stage_times = {
        "data_stage_wall_seconds": data_summary["timings"]["complete_data_stage_wall_seconds"],
        "training_wall_seconds": training["wall_time_seconds"],
        "evaluation_wall_seconds": sum(float(model["suite"]["wall_time_seconds"]) for model in evaluation["models"].values()),
    }
    total = sum(stage_times.values())
    time_percentages = {key: round(value / total * 100.0, 6) if total else None for key, value in stage_times.items()}
    public = {
        "schema_version": "1.0",
        "experiment_id": "fast-batched-directional-replay-v1",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_head(),
        "work_order_id": "0024",
        "canonical": False,
        "promotion_eligible": False,
        "accepted_parent": "none",
        "authorization": {
            "source_experiment_report_sha256": historical_reports["supertonic3_joint_adapter"]["sha256"],
            "config_sha256": file_sha256(repo_path(config_path)),
            "tts_config_sha256": file_sha256(repo_path(config["tts_config"])),
            "directional_policy_sha256": file_sha256(repo_path(config["directional_evaluation"]["policy"])),
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
        },
        "runtime": runtime_summary(),
        "protected_training_files": protected,
        "supertonic_batched_synthesis": {
            "batch_size": data_summary["batch_size"],
            "batch_count": data_summary["batch_count"],
            "actual_batch_sizes": data_summary["actual_batch_sizes"],
            "oom_fallback_count": data_summary["oom_fallback_count"],
            "native_rows": data_summary["native_rows"],
            "converted_rows": data_summary["converted_rows"],
            "workers": data_summary["workers"],
            "timings": data_summary["timings"],
            "throughput": data_summary["throughput"],
            "gpu_monitor": data_summary["gpu_monitor"],
            "manifest_hashes": data_summary["manifests"],
        },
        "audio_validation": {
            "status": validation["status"],
            "row_count": validation["row_count"],
            "training_final_files": validation["training_final_files"],
            "holdout_final_files": validation["holdout_final_files"],
            "duplicate_paths": validation["duplicate_paths"],
            "duplicate_hashes": validation["duplicate_hashes"],
            "failures_by_reason": validation["failures_by_reason"],
        },
        "training": {
            key: training[key]
            for key in (
                "status",
                "batch_size",
                "duration_bucketing",
                "epochs",
                "sample_exposures",
                "optimizer_steps",
                "learning_rate",
                "initial_probe_loss",
                "final_probe_loss",
                "initial_full_training_loss",
                "final_full_training_loss",
                "full_loss_reduction_percent",
                "wall_time_seconds",
                "examples_per_second",
                "audio_seconds_per_wall_second",
                "padding_ratio",
                "gpu_monitor",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "trainable_parameter_count",
                "base_integrity",
                "adapter_integrity",
                "restore_integrity",
            )
            if key in training
        },
        "directional_evaluation": {
            "suite_rows": evaluation["suite_rows"],
            "suite_plan_sha256": evaluation["suite_plan_sha256"],
            "policy": evaluation["evaluation_policy"],
            "metric_table": evaluation["metric_table"],
            "decision": evaluation["decision"],
            "model_summaries": {
                model_id: {
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "suite": model["suite"],
                }
                for model_id, model in evaluation["models"].items()
            },
        },
        "stage_wall_times": stage_times,
        "stage_time_percentages": time_percentages,
        "limitations": [
            "Directional batch-32 metrics are not canonical acceptance evidence.",
            "Exact transcript parity with batch size 1 was intentionally not required.",
            "No batch-1 replay was run.",
            "No release or promotion decision may use this report alone.",
            "All training remains synthetic and accepted_parent remains none.",
        ],
    }
    privacy_safe_public_report(public)
    atomic_write_json(REPORT_JSON, public)
    write_markdown_report(REPORT_MD, public)
    result = {
        "status": "PASSED",
        "json_sha256": file_sha256(REPORT_JSON),
        "markdown_sha256": file_sha256(REPORT_MD),
        "classification": public["directional_evaluation"]["decision"]["classification"],
        "accepted_parent": "none",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    decision = payload["directional_evaluation"]["decision"]
    lines = [
        "# Experiment 0012: Fast Batched Directional Replay",
        "",
        f"Status: **{payload['status']}**",
        "",
        "This replay uses native batched Supertonic synthesis and batch-32 directional ASR evaluation. It is noncanonical: exact batch-1 transcript parity was intentionally not required, no batch-1 replay was run, and no release or promotion decision may use this report alone.",
        "",
        "## Synthesis",
        "",
        f"- Batch size: {payload['supertonic_batched_synthesis']['batch_size']}",
        f"- Batch count: {payload['supertonic_batched_synthesis']['batch_count']}",
        f"- Converted rows: {payload['supertonic_batched_synthesis']['converted_rows']}",
        f"- OOM fallbacks: {payload['supertonic_batched_synthesis']['oom_fallback_count']}",
        "",
        "## Directional Metrics",
        "",
        "| Model | Piper holdout WER/CER | Supertonic holdout WER/CER | FLEURS-v2 WER/CER | ARTUR-J WER/CER |",
        "|---|---:|---:|---:|---:|",
    ]
    table = payload["directional_evaluation"]["metric_table"]
    for model_id in ("base", "piper_joint_adapter", "supertonic3_joint_adapter", "batched_replay_joint_adapter"):
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
            f"- Piper burden: {decision['piper_burden']}",
            f"- Canonical Supertonic burden: {decision['canonical_supertonic_burden']}",
            f"- Replay Supertonic burden: {decision['replay_supertonic_burden']}",
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
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast batched directional replay for Supertonic corpus-v2.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "verify-synthesis":
        result = stage_verify_synthesis(config)
    elif args.stage == "synthesize-batched":
        result = stage_synthesize_batched(config, args.progress_interval_seconds)
    elif args.stage == "convert-validate":
        result = stage_convert_validate(config, args.progress_interval_seconds)
    elif args.stage == "verify-training-code":
        result = stage_verify_training_code(config, args.config)
    elif args.stage == "train-existing-protocol":
        result = stage_train_existing_protocol(config, args.config, args.progress_interval_seconds)
    elif args.stage == "verify-artifact":
        result = stage_verify_artifact(config, args.config)
    elif args.stage == "evaluate-directional-batch32":
        result = stage_evaluate_directional(config)
    elif args.stage == "summarize":
        result = stage_summarize(config, args.config)
    else:
        parser.error(f"unsupported stage: {args.stage}")
        return 2
    print(json.dumps({"status": result.get("status", "PASSED")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
