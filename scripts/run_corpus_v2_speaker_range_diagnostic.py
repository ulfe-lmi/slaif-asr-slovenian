#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.batched_streaming import NvidiaSmiMonitor, file_sha256, parse_monitor_csv
from slaif_asr.corpus_v2_scoring import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION, NEMO_REVISION, checkpoint_path, verify_runtime_identities
from slaif_asr.corpus_v2_training import (
    EXPECTED_TRAINABLE_PARAMETERS,
    assert_epoch_covers_once,
    deterministic_epoch_batches,
    evaluate_prompt_column_integrity,
    git_head,
    load_real_gate_eval_records,
    load_synthetic_eval_records,
    load_training_records,
    make_training_batch,
    metric_pair,
    optimizer_parameter_ids,
    original_state_dict_from_prompt_delta_model,
    parameter_integrity_before_merge,
    repo_path,
    rnnt_loss,
    run_dir,
    run_evaluation_arm,
    runtime_summary,
    select_probe_records,
    state_dict_cpu,
)
from slaif_asr.prompt_column import install_prompt_delta, merge_prompt_delta, trainable_delta_parameters
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json
from slaif_asr.speaker_range_augmentation import (
    BASELINE_REPORT_SHA256,
    DEFAULT_AUGMENTATION_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    EXPECTED_BASE_METRICS,
    EXPECTED_CLEAN_METRICS,
    REPORT_JSON_PATH,
    REPORT_MD_PATH,
    SPEAKER_RANGE_CERTIFICATE_PATH,
    augmentation_paths,
    classify_speaker_range_augmented,
    load_augmentation_config,
    load_experiment_config,
    privacy_safe_experiment_report,
    summarize_local_augmentation,
    training_records_for_epoch,
    validate_augmentations,
    verify_baseline_report,
    verify_data_identities,
    verify_speaker_range_certificate,
    write_markdown_report,
)


ARM_NAME = "speaker_range_augmented_batch8"


def require_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be exactly 0")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def configure_torch() -> Any:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def restore_base_model(config: dict[str, Any]):
    import nemo.collections.asr as nemo_asr

    checkpoint = repo_path(config["model"]["checkpoint_path"]).resolve()
    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0")
    model = model.cuda().eval()
    if hasattr(model, "spec_augmentation"):
        model.spec_augmentation = None
    return model


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def finite_grad_norm(parameters: list[Any]) -> tuple[float, bool]:
    import torch

    total = 0.0
    finite = True
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all())
        total += float(torch.sum(grad * grad).detach().cpu())
    return total**0.5, finite


def mean_loss(model: Any, selection: Any, records: list[Any], *, device: str) -> float:
    import torch

    losses = []
    with torch.no_grad():
        for record in records:
            loss = rnnt_loss(model, make_training_batch(model, [record], device=device), selection.prompt_index)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def profile_schedule_rows(augmentation_config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = augmentation_paths(augmentation_config)
    if not paths.schedule.exists():
        raise RuntimeError("speaker-range exposure schedule is missing")
    with paths.schedule.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def profile_counts_for_schedule(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["profile_id"])] = counts.get(str(row["profile_id"]), 0) + 1
    return dict(sorted(counts.items()))


def train_augmented_arm(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    verify_speaker_range_certificate(config_path, require_head=True)
    augmentation_config = load_augmentation_config(repo_path(config["augmentation"]["config"]))
    validate_augmentations(config, augmentation_config)
    runtime = verify_runtime_identities(check_gpu=True)
    torch = configure_torch()
    clean_records = load_training_records(config)
    schedule_rows = profile_schedule_rows(augmentation_config)
    profile_counts = profile_counts_for_schedule(schedule_rows)
    model = restore_base_model(config)
    model.eval()
    base_state = state_dict_cpu(model)
    selection, wrapper = install_prompt_delta(model, "sl-SI")
    if selection.effective_trainable_parameters != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("unexpected prompt-column trainable count")
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable != [(selection.first_linear_name + ".delta", EXPECTED_TRAINABLE_PARAMETERS)]:
        raise RuntimeError(f"unexpected trainable parameters: {trainable}")
    optimizer = torch.optim.AdamW(
        trainable_delta_parameters(wrapper, weight_decay=0),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=0.0,
    )
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != optimizer_parameter_ids(wrapper):
        raise RuntimeError("optimizer contains parameters other than the prompt-column delta")

    probe_records = select_probe_records(clean_records, int(config["training"]["probe_rows"]))
    initial_probe = mean_loss(model, selection, probe_records, device="cuda")
    initial_full = mean_loss(model, selection, clean_records, device="cuda")
    probe_curve = [{"epoch": 0, "mean_loss": round(initial_probe, 6)}]
    delta_norm_curve = []
    grad_norms = []
    profile_losses: dict[str, list[float]] = {profile: [] for profile in profile_counts}
    optimizer_steps = 0
    sample_exposures = 0
    actual_audio_seconds = 0.0
    padded_audio_seconds = 0.0
    arm_dir = run_dir(config) / ARM_NAME
    monitor_path = arm_dir / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index="1", output_csv=monitor_path, interval_seconds=0.2)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    monitor.start()
    try:
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            layout = deterministic_epoch_batches(
                clean_records,
                batch_size=int(config["training"]["batch_size"]),
                epoch=epoch,
                seed=int(config["training"]["seed"]),
                bucketed=True,
            )
            assert_epoch_covers_once(layout, len(clean_records))
            scheduled_by_id = training_records_for_epoch(clean_records, schedule_rows, epoch=epoch)
            epoch_profile = {str(row["selected_training_id"]): str(row["profile_id"]) for row in schedule_rows if int(row["epoch"]) == epoch}
            for batch_indices in layout.batches:
                batch_records = [scheduled_by_id[clean_records[index].selected_training_id] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                loss = rnnt_loss(model, make_training_batch(model, batch_records, device="cuda"), selection.prompt_index)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite training loss")
                loss.backward()
                grad_norm, grads_ok = finite_grad_norm(trainable_delta_parameters(wrapper, weight_decay=0))
                if not grads_ok:
                    raise RuntimeError("non-finite prompt-column gradient")
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                optimizer_steps += 1
                sample_exposures += len(batch_records)
                actual_audio_seconds += sum(row.duration for row in batch_records)
                padded_audio_seconds += max(row.duration for row in batch_records) * len(batch_records)
                grad_norms.append(grad_norm)
                for row in batch_records:
                    profile_losses[epoch_profile[row.selected_training_id]].append(loss_value)
            probe_loss = mean_loss(model, selection, probe_records, device="cuda")
            probe_curve.append({"epoch": epoch, "mean_loss": round(probe_loss, 6)})
            delta_norm_curve.append({"epoch": epoch, "delta_norm": round(float(torch.linalg.vector_norm(wrapper.delta.detach()).cpu()), 6)})
    finally:
        monitor.stop()
    wall = time.perf_counter() - start
    final_probe = mean_loss(model, selection, probe_records, device="cuda")
    final_full = mean_loss(model, selection, clean_records, device="cuda")
    pre_merge = parameter_integrity_before_merge(base_state, original_state_dict_from_prompt_delta_model(model, selection), selection=selection)
    delta_path = arm_dir / "artifacts" / "prompt-column-delta.pt"
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "1.0", "selection": selection.__dict__, "delta": wrapper.delta.detach().cpu()}, delta_path)
    merge_prompt_delta(model, selection)
    merged_state = state_dict_cpu(model)
    merged_integrity = evaluate_prompt_column_integrity(base_state, merged_state, selection=selection)
    checkpoint_out = arm_dir / "artifacts" / f"{ARM_NAME}.nemo"
    model.save_to(str(checkpoint_out))
    verify_command = [sys.executable, __file__, "--config", str(config_path), "--stage", "verify-checkpoint"]
    completed = subprocess.run(verify_command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
    (arm_dir / "verify-checkpoint.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{ARM_NAME}: restored checkpoint integrity failed")
    restored_report_path = arm_dir / "restore-integrity.local.json"
    restored_integrity = json.loads(restored_report_path.read_text(encoding="utf-8"))
    payload = {
        "arm": ARM_NAME,
        "status": "PASSED",
        "batch_size": int(config["training"]["batch_size"]),
        "duration_bucketing": True,
        "epochs": int(config["training"]["epochs"]),
        "sample_exposures": sample_exposures,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(config["training"]["learning_rate"]),
        "initial_probe_loss": round(initial_probe, 6),
        "final_probe_loss": round(final_probe, 6),
        "initial_full_training_loss": round(initial_full, 6),
        "final_full_training_loss": round(final_full, 6),
        "full_loss_reduction_percent": round((initial_full - final_full) / initial_full * 100.0, 6) if initial_full else None,
        "probe_curve": probe_curve,
        "profile_exposure_counts": profile_counts,
        "profile_training_loss": {
            profile: {
                "counted_batches": len(values),
                "mean_loss": round(sum(values) / len(values), 6) if values else None,
            }
            for profile, values in sorted(profile_losses.items())
        },
        "gradient_norm": {
            "min": round(min(grad_norms), 6),
            "max": round(max(grad_norms), 6),
            "final": round(grad_norms[-1], 6),
        },
        "delta_norm_curve": delta_norm_curve,
        "wall_time_seconds": round(wall, 6),
        "examples_per_second": round(sample_exposures / wall, 6) if wall else None,
        "audio_seconds_per_wall_second": round(actual_audio_seconds / wall, 6) if wall else None,
        "padding_ratio": round(padded_audio_seconds / actual_audio_seconds, 6) if actual_audio_seconds else None,
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "selection": selection.__dict__,
        "trainable_parameter_count": selection.effective_trainable_parameters,
        "pre_merge_integrity": pre_merge,
        "integrity": merged_integrity,
        "restored_checkpoint_integrity": restored_integrity,
        "checkpoint_local_sha256": file_sha256(checkpoint_out),
        "delta_artifact_local_sha256": file_sha256(delta_path),
        "runtime": runtime,
    }
    write_json(arm_dir / "training-summary.local.json", payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def verify_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    configure_torch()
    base = restore_base_model(config)
    base_state = state_dict_cpu(base)
    from slaif_asr.prompt_column import derive_prompt_column_selection

    selection = derive_prompt_column_selection(base, "sl-SI")
    del base
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    import nemo.collections.asr as nemo_asr

    checkpoint = run_dir(config) / ARM_NAME / "artifacts" / f"{ARM_NAME}.nemo"
    restored = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0").cuda().eval()
    restored_state = state_dict_cpu(restored)
    report = evaluate_prompt_column_integrity(base_state, restored_state, selection=selection)
    report["checkpoint_restored"] = True
    write_json(run_dir(config) / ARM_NAME / "restore-integrity.local.json", report)
    return report


def evaluate_augmented(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    verify_speaker_range_certificate(config_path, require_head=True)
    verify_runtime_identities(check_gpu=True)
    checkpoint = run_dir(config) / ARM_NAME / "artifacts" / f"{ARM_NAME}.nemo"
    if not checkpoint.exists():
        raise RuntimeError("augmented checkpoint is missing")
    splits = {
        "selected_training": load_synthetic_eval_records(config, "selected_training"),
        "synthetic_holdout": load_synthetic_eval_records(config, "synthetic_holdout"),
        "fleurs_v2": load_real_gate_eval_records(config, "fleurs_v2"),
        "artur_j": load_real_gate_eval_records(config, "artur_j"),
    }
    output: dict[str, Any] = {
        "status": "PASSED",
        "models": {ARM_NAME: {"checkpoint_sha256": file_sha256(checkpoint), "splits": {}}},
        "evaluation_policy": {
            "batch_size": 1,
            "duration_bucketing": False,
            "att_context_size": config["evaluation"]["att_context_size"],
            "target_lang": config["evaluation"]["target_lang"],
            "normalizer": NORMALIZER_VERSION,
        },
    }
    for split_name, records in splits.items():
        arm = run_evaluation_arm(
            records=records,
            checkpoint=checkpoint,
            run_dir=run_dir(config) / "evaluation" / ARM_NAME / split_name,
            python_executable=Path(sys.executable),
        )
        output["models"][ARM_NAME]["splits"][split_name] = {
            "rows": int(arm["rows"]),
            "prediction_count": int(arm["prediction_count"]),
            "audio_duration_seconds": arm["audio_duration_seconds"],
            "wall_time_seconds": arm["execution"]["wall_time_seconds"],
            "real_time_factor": arm["end_to_end_real_time_factor"],
            "metrics": arm["metrics"],
            "gpu_monitor": arm["execution"]["monitor"],
        }
    write_json(run_dir(config) / "evaluation" / "summary.local.json", output)
    return output


def augmented_normalized_metrics(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    splits = evaluation["models"][ARM_NAME]["splits"]
    metrics = {}
    for split_name, split in splits.items():
        normalized = split["metrics"]["normalized"]
        metrics[split_name] = {
            "wer": round(float(normalized["corpus_wer"]), 3),
            "cer": round(float(normalized["corpus_cer"]), 3),
            "empty": int(normalized["empty_hypothesis_count"]),
        }
    return metrics


def summarize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    auth = verify_speaker_range_certificate(config_path, require_head=True)
    baseline = verify_baseline_report()
    input_integrity = verify_data_identities(config)
    augmentation_config = load_augmentation_config(repo_path(config["augmentation"]["config"]))
    augmentation_validation = validate_augmentations(config, augmentation_config)
    augmentation_summary = summarize_local_augmentation(config, augmentation_config)
    training = json.loads((run_dir(config) / ARM_NAME / "training-summary.local.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run_dir(config) / "evaluation" / "summary.local.json").read_text(encoding="utf-8"))
    augmented_metrics = augmented_normalized_metrics(evaluation)
    decision = classify_speaker_range_augmented(augmented_metrics)
    comparison = {
        split: {
            "base": EXPECTED_BASE_METRICS[split],
            "clean": EXPECTED_CLEAN_METRICS[split],
            "augmented": augmented_metrics[split],
        }
        for split in ("selected_training", "synthetic_holdout", "fleurs_v2", "artur_j")
    }
    public = {
        "schema_version": "1.0",
        "experiment_id": "corpus-v2-speaker-range-diagnostic-v1",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_head(),
        "authorization": {
            "status": auth["certificate"]["status"],
            "sha256": file_sha256(SPEAKER_RANGE_CERTIFICATE_PATH),
            "work_order_id": auth["certificate"]["work_order_id"],
            "baseline_report_sha256": BASELINE_REPORT_SHA256,
            "tracked_before_execution": auth["tracked"],
        },
        "runtime": runtime_summary(),
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
        },
        "input_integrity": {
            "selected_training": input_integrity["selected_training"],
            "synthetic_holdout_audio_manifest_sha256": input_integrity["synthetic_holdout_audio_manifest_sha256"],
            "synthetic_holdout_rows": input_integrity["synthetic_holdout_rows"],
            "candidate_holdout_overlap_counts": input_integrity["candidate_holdout_overlap_counts"],
        },
        "baseline": baseline,
        "augmentation": {
            "policy_sha256": file_sha256(repo_path(config["augmentation"]["config"])),
            "source_rows": augmentation_validation["source_rows"],
            "non_clean_files": augmentation_validation["non_clean_files"],
            "total_profile_records": augmentation_validation["total_profile_records"],
            "scheduled_exposures": augmentation_validation["schedule"]["scheduled_exposures"],
            "exposures_by_profile": augmentation_validation["schedule"]["exposures_by_profile"],
            "manifest_sha256": augmentation_validation["manifest_sha256"],
            "exposure_schedule_sha256": augmentation_validation["exposure_schedule_sha256"],
            "validation_status": augmentation_validation["status"],
            "duration_seconds_by_profile": augmentation_summary["duration_seconds_by_profile"],
            "peak_by_profile": augmentation_summary["peak_by_profile"],
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
                "probe_curve",
                "profile_exposure_counts",
                "profile_training_loss",
                "gradient_norm",
                "delta_norm_curve",
                "wall_time_seconds",
                "examples_per_second",
                "audio_seconds_per_wall_second",
                "padding_ratio",
                "gpu_monitor",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "selection",
                "trainable_parameter_count",
                "pre_merge_integrity",
                "integrity",
                "restored_checkpoint_integrity",
            )
            if key in training
        },
        "evaluation": evaluation,
        "metric_comparison": comparison,
        "decision": decision,
        "accepted_parent": "none",
        "limitations": [
            "One original Piper voice family.",
            "Resampling is only a child/high/low/elder voice proxy.",
            "No real calibration speech.",
            "Synthetic holdout is not real-generalization evidence.",
            "FLEURS-v2 and ARTUR-J are development gates.",
        ],
    }
    privacy_safe_experiment_report(public)
    atomic_write_json(REPORT_JSON_PATH, public)
    write_markdown_report(REPORT_MD_PATH, public)
    result = {
        "status": "PASSED",
        "json_sha256": file_sha256(REPORT_JSON_PATH),
        "markdown_sha256": file_sha256(REPORT_MD_PATH),
        "scientific_classification": decision["classification"],
        "accepted_parent": "none",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the corpus-v2 speaker-range augmentation diagnostic.")
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.stage in {"verify", "train", "evaluate", "verify-checkpoint"}:
        require_env()
    if args.stage == "verify":
        payload = {
            "certificate": verify_speaker_range_certificate(args.config, require_head=True),
            "baseline": verify_baseline_report(),
            "data": verify_data_identities(config),
            "gpu_runtime": verify_runtime_identities(check_gpu=True),
            "checkpoint_sha256": file_sha256(checkpoint_path()),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.stage == "train":
        payload = train_augmented_arm(config, args.config)
        print(json.dumps({"status": payload["status"], "arm": payload["arm"]}, indent=2, sort_keys=True))
        return 0
    if args.stage == "verify-checkpoint":
        payload = verify_checkpoint(config)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload.get("passed") else 1
    if args.stage == "evaluate":
        payload = evaluate_augmented(config, args.config)
        print(json.dumps({"status": payload["status"], "models": sorted(payload["models"])}, indent=2, sort_keys=True))
        return 0
    if args.stage == "summarize":
        summarize(config, args.config)
        return 0
    parser.error(f"unsupported stage: {args.stage}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
