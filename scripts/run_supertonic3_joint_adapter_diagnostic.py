#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.batched_streaming import NvidiaSmiMonitor, StreamingRecord, file_sha256, parse_monitor_csv
from slaif_asr.corpus_v2_scoring import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION, NEMO_REVISION, checkpoint_path, verify_runtime_identities
from slaif_asr.corpus_v2_training import (
    TrainingRecord,
    assert_epoch_covers_once,
    assert_public_report_safe,
    deterministic_epoch_batches,
    git_head,
    git_tracked_and_clean_at_head,
    load_real_gate_eval_records,
    load_synthetic_eval_records,
    load_training_records,
    make_training_batch,
    read_json,
    repo_path,
    rnnt_loss,
    run_dir,
    run_evaluation_arm,
    runtime_summary,
    select_probe_records,
    verify_all_input_identities,
)
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread
from slaif_asr.prompt_column import derive_prompt_column_selection
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json
from slaif_asr.slovenian_joint_adapter import (
    ADAPTER_NAME,
    adapter_parameters,
    compare_adapter_state,
    compare_base_state,
    enabled_joint_adapters,
    expected_trainable_count,
    load_adapter_artifact,
    load_adapter_spec,
    save_adapter_artifact,
    state_dict_cpu,
    verify_optimizer_scope,
)
from slaif_asr.supertonic3_tts import (
    HELD_OUT_STYLES,
    TRAINING_STYLES,
    load_holdout_items,
    load_supertonic_config,
    load_supertonic_training_schedule,
    read_jsonl,
    supertonic_paths,
    training_records_for_supertonic_epoch,
)
from slaif_asr.tts import validate_wav


_JOINT_PATH = Path(__file__).with_name("run_corpus_v2_joint_adapter_diagnostic.py")
_JOINT_SPEC = importlib.util.spec_from_file_location("_slaif_joint_runner", _JOINT_PATH)
if _JOINT_SPEC is None or _JOINT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import joint-adapter runner")
_JOINT = importlib.util.module_from_spec(_JOINT_SPEC)
_JOINT_SPEC.loader.exec_module(_JOINT)
finite_grad_norm = _JOINT.finite_grad_norm
mean_loss = _JOINT.mean_loss


DEFAULT_CONFIG = Path("configs/experiments/corpus_v2_supertonic3_multivoice_v1.json")
REPORT_JSON = Path("docs/experiments/0011-corpus-v2-supertonic3-multivoice-joint-adapter.json")
REPORT_MD = Path("docs/experiments/0011-corpus-v2-supertonic3-multivoice-joint-adapter.md")
CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v2-supertonic3-joint-adapter-diagnostic-v1.json")
ARM_NAME = "supertonic3_multivoice_joint_adapter_dim32"
EXPECTED_PIPER_JOINT_BURDEN = 28.275

EXPECTED_BASE_METRICS = {
    "piper_selected_training": {"wer": 93.032, "cer": 61.623, "empty": 41},
    "piper_synthetic_holdout": {"wer": 84.317, "cer": 47.295, "empty": 17},
    "fleurs_v2": {"wer": 52.703, "cer": 16.423, "empty": 1},
    "artur_j": {"wer": 67.453, "cer": 29.016, "empty": 12},
}
EXPECTED_PIPER_JOINT_METRICS = {
    "piper_selected_training": {"wer": 24.253, "cer": 11.083, "empty": 0},
    "piper_synthetic_holdout": {"wer": 69.876, "cer": 29.156, "empty": 0},
    "fleurs_v2": {"wer": 64.733, "cer": 25.541, "empty": 0},
    "artur_j": {"wer": 73.263, "cer": 30.333, "empty": 0},
}


def require_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be exactly 0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def configure_torch() -> Any:
    return _JOINT.configure_torch()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(repo_path(path))
    if config.get("work_order_id") != "0023":
        raise ValueError("Supertonic diagnostic config must belong to work order 0023")
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
        "waveform_augmentation": False,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected}")
    if float(training.get("learning_rate", -1.0)) != 0.001:
        raise ValueError("learning_rate must be 0.001")
    return config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def restore_base_model(config: dict[str, Any], *, reporter: LiveProgressReporter | None = None):
    return _JOINT.restore_base_model(config, reporter=reporter)


def prepare_adapter_model(model: Any, config: dict[str, Any], *, enable: bool) -> dict[str, Any]:
    return _JOINT.prepare_adapter_model(model, config, enable=enable)


def verify_certificate(config_path: Path, *, require_head: bool) -> dict[str, Any]:
    config = load_config(config_path)
    cert_path = repo_path(config["authorization"]["certificate_path"])
    tracked = git_tracked_and_clean_at_head(cert_path) if require_head else {"tracked": False, "clean": False, "matches_head": False}
    cert = read_json(cert_path)
    if cert.get("status") != "DIAGNOSTIC_ONLY":
        raise RuntimeError("Supertonic diagnostic certificate status must be DIAGNOSTIC_ONLY")
    if cert.get("work_order_id") != "0023":
        raise RuntimeError("Supertonic diagnostic certificate work-order mismatch")
    if cert.get("experiment_config_sha256") != file_sha256(repo_path(config_path)):
        raise RuntimeError("Supertonic diagnostic certificate experiment config SHA mismatch")
    if cert.get("adapter_config_sha256") != file_sha256(repo_path(config["adapter"]["config"])):
        raise RuntimeError("Supertonic diagnostic certificate adapter config SHA mismatch")
    audio_cert = repo_path(config["supertonic_audio"]["audio_certificate"])
    if cert.get("supertonic_audio", {}).get("audio_certificate_sha256") != file_sha256(audio_cert):
        raise RuntimeError("Supertonic diagnostic certificate audio certificate SHA mismatch")
    identities = verify_all_input_identities(config, check_gpu=False)
    return {"certificate": cert, "tracked": tracked, "identities": identities}


def verify_baseline_report() -> dict[str, Any]:
    path = Path("docs/experiments/0010-corpus-v2-slovenian-joint-adapter-diagnostic.json")
    expected_sha = "e956154e2a6a2012b7a852a8bd8bb90ba3a794911df0bfd29267f2d7df3b2e0e"
    if file_sha256(path) != expected_sha:
        raise RuntimeError("Experiment 0010 report SHA mismatch")
    report = read_json(path)
    for split, expected in EXPECTED_PIPER_JOINT_METRICS.items():
        old_split = split.replace("piper_", "")
        if old_split == "selected_training":
            old_split = "selected_training"
        if old_split == "synthetic_holdout":
            old_split = "synthetic_holdout"
        metrics = report["metric_comparison"][old_split]["joint_adapter"]
        if round(float(metrics["wer"]), 3) != expected["wer"] or round(float(metrics["cer"]), 3) != expected["cer"]:
            raise RuntimeError(f"Experiment 0010 joint metric mismatch for {split}")
        if int(metrics["empty"]) != expected["empty"]:
            raise RuntimeError(f"Experiment 0010 joint empty count mismatch for {split}")
    burden = real_regression_burden(EXPECTED_PIPER_JOINT_METRICS)
    if round(burden, 3) != EXPECTED_PIPER_JOINT_BURDEN:
        raise RuntimeError("Experiment 0010 regression burden mismatch")
    return {"path": str(path), "sha256": expected_sha, "piper_joint_burden": burden}


def _training_text_by_id(config: dict[str, Any]) -> dict[str, TrainingRecord]:
    return {row.selected_training_id: row for row in load_training_records(config)}


def _supertonic_training_record(clean: TrainingRecord, row: dict[str, Any]) -> TrainingRecord:
    path = Path(str(row["audio_filepath"]))
    validate_wav(path, sample_rate=16000)
    if file_sha256(path) != str(row["audio_sha256"]):
        raise RuntimeError("Supertonic training audio hash mismatch")
    return TrainingRecord(
        selected_training_id=clean.selected_training_id,
        audio_filepath=str(path),
        duration=float(row["duration_seconds"]),
        text=clean.text,
        text_sha256=clean.text_sha256,
        audio_sha256=str(row["audio_sha256"]),
        selection_reason=clean.selection_reason,
        selection_rank=clean.selection_rank,
    )


def supertonic_training_probe_records(config: dict[str, Any]) -> list[TrainingRecord]:
    clean = _training_text_by_id(config)
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    rows = read_jsonl(supertonic_paths(tts_config).training_probe_manifest)
    records = []
    for row in rows:
        source_key = str(row["source_key"])
        records.append(_supertonic_training_record(clean[source_key], row))
    return sorted(records, key=lambda row: row.selected_training_id)


def supertonic_heldout_records(config: dict[str, Any], *, voice_style: str | None = None) -> list[StreamingRecord]:
    tts_config = load_supertonic_config(repo_path(config["tts_config"]))
    holdout_text = {item.source_key: item for item in load_holdout_items(tts_config)}
    rows = read_jsonl(supertonic_paths(tts_config).holdout_audio_manifest)
    records: list[StreamingRecord] = []
    for index, row in enumerate(rows):
        source_key = str(row["source_key"])
        voice = str(row["voice_style_id"])
        if voice not in HELD_OUT_STYLES:
            raise RuntimeError("training voice leaked into Supertonic held-out split")
        if voice_style is not None and voice != voice_style:
            continue
        item = holdout_text[source_key]
        path = Path(str(row["audio_filepath"]))
        validate_wav(path, sample_rate=16000)
        if file_sha256(path) != str(row["audio_sha256"]):
            raise RuntimeError("Supertonic held-out audio hash mismatch")
        records.append(
            StreamingRecord(
                sample_id=f"{source_key}.{voice}",
                audio_filepath=str(path),
                duration=float(row["duration_seconds"]),
                reference=item.text,
                original_index=len(records),
                row={"split": "supertonic_heldout_voice_holdout", "voice_style_id": voice},
            )
        )
    if voice_style is not None and len(records) != 96:
        raise RuntimeError(f"expected 96 held-out records for {voice_style}, found {len(records)}")
    return records


def supertonic_training_voice_records(config: dict[str, Any]) -> list[StreamingRecord]:
    records = []
    for index, row in enumerate(supertonic_training_probe_records(config)):
        records.append(
            StreamingRecord(
                sample_id=row.selected_training_id,
                audio_filepath=row.audio_filepath,
                duration=row.duration,
                reference=row.text,
                original_index=index,
                row={"split": "supertonic_training_voice_probe"},
            )
        )
    return records


def run_evaluation_with_progress(
    config: dict[str, Any],
    *,
    records: Sequence[StreamingRecord],
    checkpoint: Path,
    split: str,
    run_subdir: Path,
) -> dict[str, Any]:
    reporter = LiveProgressReporter(
        stage="evaluate",
        arm=ARM_NAME,
        split=split,
        ndjson_path=run_dir(config) / "progress" / f"evaluate-{split}.local.ndjson",
    )
    reporter.start(f"evaluating {split}")
    try:
        with heartbeat_thread(
            reporter,
            interval_seconds=5.0,
            message="evaluation subprocess active",
            fields=lambda: {"processed_rows": 0, "total_rows": len(records)},
        ):
            arm = run_evaluation_arm(
                records=records,
                checkpoint=checkpoint,
                run_dir=run_subdir,
                python_executable=Path(sys.executable),
            )
        reporter.complete(processed_rows=len(records), total_rows=len(records), message="evaluation complete")
        return arm
    except Exception as exc:
        reporter.failed(message="evaluation failed", error_type=type(exc).__name__)
        raise


def run_verify(config: dict[str, Any], config_path: Path, interval: float) -> dict[str, Any]:
    _JOINT.ARM_NAME = ARM_NAME
    cert = verify_certificate(config_path, require_head=True)
    runtime = verify_runtime_identities(check_gpu=True)
    baseline = verify_baseline_report()
    reporter = LiveProgressReporter(stage="verify", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "verify.local.ndjson")
    reporter.start("verifying Supertonic joint-adapter diagnostic")
    parity = _JOINT.zero_init_parity_probe(config, config_path, reporter=reporter)
    reporter.complete("verification complete")
    result = {"status": "PASSED", "authorization": cert["tracked"], "runtime": runtime, "baseline": baseline, "zero_init_parity": parity}
    write_json(run_dir(config) / "authorization" / "verify.local.json", result)
    return result


def train(config: dict[str, Any], config_path: Path, interval: float) -> dict[str, Any]:
    verify_certificate(config_path, require_head=True)
    runtime = verify_runtime_identities(check_gpu=True)
    torch = configure_torch()
    clean_records = load_training_records(config)
    schedule = load_supertonic_training_schedule(config)
    voice_by_audio = {str(row["audio_filepath"]): str(row["voice_style_id"]) for row in schedule}
    reporter = LiveProgressReporter(stage="train", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "train.local.ndjson")
    reporter.start("training Supertonic joint adapter")
    model = restore_base_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "restore.local.ndjson"))
    model.eval()
    adapter_summary = prepare_adapter_model(model, config, enable=True)
    initial_state = state_dict_cpu(model)
    prompt_selection = derive_prompt_column_selection(model, "sl-SI")
    if adapter_summary["trainable_parameters"] != expected_trainable_count(adapter_summary["joint_hidden"], 32):
        raise RuntimeError("joint-adapter trainable parameter count mismatch")
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=float(config["training"]["learning_rate"]), weight_decay=0.0)
    verify_optimizer_scope(optimizer, model)

    probe_pool = supertonic_training_probe_records(config)
    probe_records = select_probe_records(probe_pool, int(config["training"]["probe_rows"]))
    initial_probe = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
    initial_full = mean_loss(model, prompt_selection.prompt_index, probe_pool, device="cuda")
    probe_curve = [{"epoch": 0, "mean_loss": round(initial_probe, 6)}]
    adapter_norm_curve = []
    grad_norms = []
    optimizer_steps = 0
    sample_exposures = 0
    audio_seconds = 0.0
    padded_audio_seconds = 0.0
    rolling_losses: list[float] = []
    exposure_counts: Counter[str] = Counter()
    arm_dir = run_dir(config) / ARM_NAME
    monitor_path = arm_dir / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index="1", output_csv=monitor_path, interval_seconds=0.2)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    monitor.start()
    try:
        total_steps = int(config["training"]["optimizer_steps"])
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            epoch_records = training_records_for_supertonic_epoch(clean_records, schedule, epoch)
            layout = deterministic_epoch_batches(
                clean_records,
                batch_size=int(config["training"]["batch_size"]),
                epoch=epoch,
                seed=int(config["training"]["seed"]),
                bucketed=True,
            )
            assert_epoch_covers_once(layout, len(clean_records))
            for batch_indices in layout.batches:
                batch_records = [epoch_records[clean_records[index].selected_training_id] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                loss = rnnt_loss(model, make_training_batch(model, batch_records, device="cuda"), prompt_selection.prompt_index)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite Supertonic joint-adapter training loss")
                loss.backward()
                grad_norm, grads_ok = finite_grad_norm(adapter_parameters(model))
                if not grads_ok:
                    raise RuntimeError("non-finite Supertonic joint-adapter gradient")
                for name, parameter in model.named_parameters():
                    if not name.startswith(f"joint.adapter_layer.{ADAPTER_NAME}.") and parameter.grad is not None:
                        raise RuntimeError(f"pretrained parameter received gradient: {name}")
                optimizer.step()
                optimizer_steps += 1
                sample_exposures += len(batch_records)
                audio_seconds += sum(row.duration for row in batch_records)
                padded_audio_seconds += max(row.duration for row in batch_records) * len(batch_records)
                for row in batch_records:
                    exposure_counts[voice_by_audio[row.audio_filepath]] += 1
                loss_value = float(loss.detach().cpu())
                rolling_losses.append(loss_value)
                rolling_losses = rolling_losses[-10:]
                grad_norms.append(grad_norm)
                if optimizer_steps % 5 == 0:
                    elapsed = time.perf_counter() - start
                    reporter.progress(
                        epoch=epoch,
                        total_epochs=int(config["training"]["epochs"]),
                        step=optimizer_steps,
                        total_steps=total_steps,
                        current_loss=round(loss_value, 6),
                        rolling_mean_loss=round(sum(rolling_losses) / len(rolling_losses), 6),
                        examples_per_second=round(sample_exposures / elapsed, 6) if elapsed else None,
                        audio_seconds_per_wall_second=round(audio_seconds / elapsed, 6) if elapsed else None,
                        cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    )
            probe_loss = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
            probe_curve.append({"epoch": epoch, "mean_loss": round(probe_loss, 6)})
            adapter_norm = sum(float(torch.linalg.vector_norm(parameter.detach()).cpu()) for parameter in adapter_parameters(model))
            adapter_norm_curve.append({"epoch": epoch, "adapter_parameter_norm": round(adapter_norm, 6)})
    except Exception as exc:
        reporter.failed(message="training failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()
    reporter.heartbeat(message="post-training validation starting", step=optimizer_steps, total_steps=int(config["training"]["optimizer_steps"]))
    with heartbeat_thread(reporter, interval_seconds=interval, message="post-training validation in progress"):
        wall = time.perf_counter() - start
        final_probe = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
        final_full = mean_loss(model, prompt_selection.prompt_index, probe_pool, device="cuda")
        trained_state = state_dict_cpu(model)
        base_integrity = compare_base_state(initial_state, trained_state)
        adapter_integrity = compare_adapter_state(initial_state, trained_state)
        if not base_integrity["base_tensors_identical"]:
            raise RuntimeError("pretrained tensor changed during Supertonic joint-adapter training")
        artifact_path = arm_dir / "artifacts" / "sl-si-joint-adapter-v1.pt"
        artifact_sha = save_adapter_artifact(
            artifact_path,
            model=model,
            spec=load_adapter_spec(config["adapter"]["config"]),
            metadata={
                "base_checkpoint_sha256": CHECKPOINT_SHA256,
                "nemo_revision": NEMO_REVISION,
                "supertonic_training_audio_manifest_sha256": config["supertonic_audio"].get("training_audio_manifest_sha256"),
                "experiment_config_sha256": file_sha256(repo_path(config_path)),
                "adapter_config_sha256": file_sha256(repo_path(config["adapter"]["config"])),
                "adapter_config": read_json(repo_path(config["adapter"]["config"])),
            },
        )
        verify_command = [sys.executable, "-u", __file__, "--config", str(config_path), "--stage", "verify-artifact"]
        completed = subprocess.run(verify_command, cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
        (arm_dir / "verify-artifact.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError("adapter artifact restore verification failed")
        _JOINT.enable_for_target_language(model, "sl-SI")
        checkpoint_out = arm_dir / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
        model.save_to(str(checkpoint_out))
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
        "gradient_norm": {"min": round(min(grad_norms), 6), "max": round(max(grad_norms), 6), "final": round(grad_norms[-1], 6)},
        "adapter_norm_curve": adapter_norm_curve,
        "wall_time_seconds": round(wall, 6),
        "examples_per_second": round(sample_exposures / wall, 6) if wall else None,
        "audio_seconds_per_wall_second": round(audio_seconds / wall, 6) if wall else None,
        "padding_ratio": round(padded_audio_seconds / audio_seconds, 6) if audio_seconds else None,
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "adapter": adapter_summary,
        "trainable_parameter_count": adapter_summary["trainable_parameters"],
        "base_integrity": base_integrity,
        "adapter_integrity": adapter_integrity,
        "artifact_sha256": artifact_sha,
        "restore_integrity": read_json(arm_dir / "restore-integrity.local.json"),
        "evaluation_checkpoint_sha256": file_sha256(checkpoint_out),
        "exposure_counts_by_training_voice": dict(sorted(exposure_counts.items())),
        "runtime": runtime,
    }
    write_json(arm_dir / "training-summary.local.json", payload)
    reporter.complete("training complete", step=optimizer_steps, total_steps=int(config["training"]["optimizer_steps"]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def verify_artifact(config: dict[str, Any]) -> dict[str, Any]:
    _JOINT.ARM_NAME = ARM_NAME
    return _JOINT.verify_artifact(config)


def evaluate_base_new_splits(config: dict[str, Any]) -> dict[str, Any]:
    verify_certificate(DEFAULT_CONFIG, require_head=True)
    verify_runtime_identities(check_gpu=True)
    splits = {
        "supertonic_training_voice_probe": supertonic_training_voice_records(config),
        "supertonic_heldout_voice_holdout": supertonic_heldout_records(config),
        "supertonic_heldout_voice_m5": supertonic_heldout_records(config, voice_style="M5"),
        "supertonic_heldout_voice_f5": supertonic_heldout_records(config, voice_style="F5"),
    }
    output = {"status": "PASSED", "models": {"base": {"checkpoint_sha256": CHECKPOINT_SHA256, "splits": {}}}}
    for split_name, records in splits.items():
        split_dir = run_dir(config) / "evaluation" / "base" / split_name
        completed = _completed_evaluation_summary(split_dir, len(records))
        if completed is not None:
            output["models"]["base"]["splits"][split_name] = completed
            continue
        arm = run_evaluation_with_progress(
            config,
            records=records,
            checkpoint=checkpoint_path(),
            split=f"base_{split_name}",
            run_subdir=split_dir,
        )
        output["models"]["base"]["splits"][split_name] = _evaluation_summary(arm)
    write_json(run_dir(config) / "evaluation" / "base-new-splits.local.json", output)
    return output


def _evaluation_summary(arm: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(arm["rows"]),
        "prediction_count": int(arm["prediction_count"]),
        "audio_duration_seconds": arm["audio_duration_seconds"],
        "wall_time_seconds": arm["execution"]["wall_time_seconds"],
        "real_time_factor": arm["end_to_end_real_time_factor"],
        "metrics": arm["metrics"],
        "gpu_monitor": arm["execution"]["monitor"],
    }


def _completed_evaluation_summary(run_subdir: Path, expected_rows: int) -> dict[str, Any] | None:
    summary_path = run_subdir / "summary.local.json"
    if not summary_path.exists():
        return None
    arm = read_json(summary_path)
    if arm.get("status") != "PASSED":
        return None
    if int(arm.get("rows", -1)) != expected_rows:
        return None
    if int(arm.get("prediction_count", -1)) != expected_rows:
        return None
    return _evaluation_summary(arm)


def evaluate(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    verify_certificate(config_path, require_head=True)
    verify_runtime_identities(check_gpu=True)
    checkpoint = run_dir(config) / ARM_NAME / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
    if not checkpoint.exists():
        raise RuntimeError("Supertonic joint-adapter evaluation checkpoint is missing")
    splits = {
        "supertonic_training_voice_probe": supertonic_training_voice_records(config),
        "piper_selected_training": load_synthetic_eval_records(config, "selected_training"),
        "piper_synthetic_holdout": load_synthetic_eval_records(config, "synthetic_holdout"),
        "supertonic_heldout_voice_holdout": supertonic_heldout_records(config),
        "supertonic_heldout_voice_m5": supertonic_heldout_records(config, voice_style="M5"),
        "supertonic_heldout_voice_f5": supertonic_heldout_records(config, voice_style="F5"),
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
        arm = run_evaluation_with_progress(
            config,
            records=records,
            checkpoint=checkpoint,
            split=split_name,
            run_subdir=run_dir(config) / "evaluation" / ARM_NAME / split_name,
        )
        output["models"][ARM_NAME]["splits"][split_name] = _evaluation_summary(arm)
    write_json(run_dir(config) / "evaluation" / "summary.local.json", output)
    return output


def _metric_row(split: dict[str, Any]) -> dict[str, Any]:
    normalized = split["metrics"]["normalized"]
    raw = split["metrics"]["raw"]
    return {"wer": round(float(normalized["corpus_wer"]), 3), "cer": round(float(normalized["corpus_cer"]), 3), "empty": int(raw["empty_hypothesis_count"])}


def normalized_metrics(evaluation: dict[str, Any], model_name: str) -> dict[str, dict[str, Any]]:
    return {split_name: _metric_row(split) for split_name, split in evaluation["models"][model_name]["splits"].items()}


def synthetic_gain(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    wer_gain = (base["wer"] - candidate["wer"]) / base["wer"] * 100.0
    cer_gain = (base["cer"] - candidate["cer"]) / base["cer"] * 100.0
    return {"passes": wer_gain >= 10.0 or cer_gain >= 10.0, "wer_relative_gain": round(wer_gain, 6), "cer_relative_gain": round(cer_gain, 6)}


def real_regression_burden(metrics: dict[str, dict[str, Any]]) -> float:
    burden = 0.0
    for split in ("fleurs_v2", "artur_j"):
        base = EXPECTED_BASE_METRICS[split]
        candidate = metrics[split]
        burden += max(0.0, candidate["wer"] - base["wer"])
        burden += max(0.0, candidate["cer"] - base["cer"])
    return round(burden, 6)


def classify_supertonic(metrics: dict[str, dict[str, Any]], base_new: dict[str, dict[str, Any]]) -> dict[str, Any]:
    piper_holdout_gain = synthetic_gain(EXPECTED_BASE_METRICS["piper_synthetic_holdout"], metrics["piper_synthetic_holdout"])
    super_holdout_gain = synthetic_gain(base_new["supertonic_heldout_voice_holdout"], metrics["supertonic_heldout_voice_holdout"])
    both_gain = piper_holdout_gain["passes"] and super_holdout_gain["passes"]
    burden = real_regression_burden(metrics)
    piper_burden = real_regression_burden(EXPECTED_PIPER_JOINT_METRICS)
    non_regression = True
    improvement = False
    for split in ("fleurs_v2", "artur_j"):
        base = EXPECTED_BASE_METRICS[split]
        candidate = metrics[split]
        if candidate["wer"] - base["wer"] > 1.0 or candidate["cer"] - base["cer"] > 1.5 or candidate["empty"] > base["empty"]:
            non_regression = False
        if candidate["wer"] - base["wer"] <= -1.0 or candidate["cer"] - base["cer"] <= -1.5:
            improvement = True
    no_worse_than_piper = True
    for split in ("fleurs_v2", "artur_j"):
        piper = EXPECTED_PIPER_JOINT_METRICS[split]
        candidate = metrics[split]
        if candidate["wer"] - piper["wer"] > 0.5 or candidate["cer"] - piper["cer"] > 0.5:
            no_worse_than_piper = False
    burden_reduction = (piper_burden - burden) / piper_burden * 100.0 if piper_burden else 0.0
    if both_gain and non_regression and improvement:
        classification = "SUPERTONIC3_MULTIVOICE_REAL_GAIN_DIAGNOSTIC"
    elif both_gain and burden_reduction >= 30.0 and no_worse_than_piper:
        classification = "SUPERTONIC3_MULTIVOICE_MITIGATES_PIPER_REGRESSION"
    elif piper_holdout_gain["passes"] or super_holdout_gain["passes"]:
        classification = "SUPERTONIC3_MULTIVOICE_SYNTHETIC_ONLY"
    else:
        classification = "SUPERTONIC3_MULTIVOICE_NOT_SUPPORTED"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "piper_holdout_gain": piper_holdout_gain,
        "supertonic_heldout_voice_gain": super_holdout_gain,
        "piper_joint_burden": piper_burden,
        "supertonic_joint_burden": burden,
        "burden_reduction_percent": round(burden_reduction, 6),
        "real_non_regression": non_regression,
        "real_improvement": improvement,
    }


def summarize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    auth = verify_certificate(config_path, require_head=True)
    baseline = verify_baseline_report()
    input_integrity = verify_all_input_identities(config, check_gpu=False)
    audio_certificate = read_json(repo_path(config["supertonic_audio"]["audio_certificate"]))
    training = read_json(run_dir(config) / ARM_NAME / "training-summary.local.json")
    base_new_eval = read_json(run_dir(config) / "evaluation" / "base-new-splits.local.json")
    evaluation = read_json(run_dir(config) / "evaluation" / "summary.local.json")
    base_new_metrics = normalized_metrics(base_new_eval, "base")
    super_metrics = normalized_metrics(evaluation, ARM_NAME)
    decision = classify_supertonic(super_metrics, base_new_metrics)
    public = {
        "schema_version": "1.0",
        "experiment_id": "corpus-v2-supertonic3-multivoice-joint-adapter-v1",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_head(),
        "authorization": {
            "status": auth["certificate"]["status"],
            "sha256": file_sha256(CERTIFICATE_PATH),
            "work_order_id": auth["certificate"]["work_order_id"],
            "tracked_before_execution": auth["tracked"],
            "audio_certificate_sha256": auth["certificate"]["supertonic_audio"]["audio_certificate_sha256"],
            "adapter_config_sha256": auth["certificate"]["adapter_config_sha256"],
            "experiment_config_sha256": auth["certificate"]["experiment_config_sha256"],
        },
        "baseline": baseline,
        "runtime": runtime_summary(),
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256, "nemo_revision": NEMO_REVISION},
        "supertonic_audio": {
            "audio_status": audio_certificate["status"],
            "asset_tree_sha256": audio_certificate["tts"]["asset_tree_sha256"],
            "training_voice_styles": list(TRAINING_STYLES),
            "held_out_voice_styles": list(HELD_OUT_STYLES),
            "counts": audio_certificate["counts"],
            "hashes": audio_certificate["hashes"],
            "voice_counts": audio_certificate["voice_styles"]["counts"],
            "duration_by_style": audio_certificate["voice_styles"]["duration_by_style"],
            "license_boundary": {
                "code_license": audio_certificate["tts"]["package_license"],
                "model_license": audio_certificate["tts"]["model_license"],
                "generated_audio_publication": "prohibited by this work order",
                "adapter_checkpoint_publication": "prohibited by this work order",
                "future_legal_review_required": True,
            },
        },
        "input_integrity": {
            "selected_training": input_integrity["selected_training"],
            "synthetic_holdout_audio_manifest_sha256": input_integrity["synthetic_holdout_audio_manifest_sha256"],
            "synthetic_holdout_rows": input_integrity["synthetic_holdout_rows"],
            "candidate_holdout_overlap_counts": input_integrity["candidate_holdout_overlap_counts"],
        },
        "adapter": {
            "name": ADAPTER_NAME,
            "type": "nemo.collections.common.parts.adapter_modules.LinearAdapter",
            "module": "model.joint",
            "bottleneck_dimension": 32,
            "joint_hidden": training["adapter"]["joint_hidden"],
            "exact_trainable_parameters": training["trainable_parameter_count"],
            "zero_init_parity": read_json(run_dir(config) / "authorization" / "zero-init-parity" / "summary.local.json"),
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
                "gradient_norm",
                "adapter_norm_curve",
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
                "exposure_counts_by_training_voice",
            )
            if key in training
        },
        "evaluation": {"base_new_splits": base_new_eval, "supertonic_joint": evaluation},
        "metric_comparison": {
            "piper_selected_training": {
                "base": EXPECTED_BASE_METRICS["piper_selected_training"],
                "piper_joint": EXPECTED_PIPER_JOINT_METRICS["piper_selected_training"],
                "supertonic_joint": super_metrics["piper_selected_training"],
            },
            "piper_synthetic_holdout": {
                "base": EXPECTED_BASE_METRICS["piper_synthetic_holdout"],
                "piper_joint": EXPECTED_PIPER_JOINT_METRICS["piper_synthetic_holdout"],
                "supertonic_joint": super_metrics["piper_synthetic_holdout"],
            },
            "fleurs_v2": {
                "base": EXPECTED_BASE_METRICS["fleurs_v2"],
                "piper_joint": EXPECTED_PIPER_JOINT_METRICS["fleurs_v2"],
                "supertonic_joint": super_metrics["fleurs_v2"],
            },
            "artur_j": {
                "base": EXPECTED_BASE_METRICS["artur_j"],
                "piper_joint": EXPECTED_PIPER_JOINT_METRICS["artur_j"],
                "supertonic_joint": super_metrics["artur_j"],
            },
            "supertonic_training_voice_probe": {"base": base_new_metrics["supertonic_training_voice_probe"], "supertonic_joint": super_metrics["supertonic_training_voice_probe"]},
            "supertonic_heldout_voice_holdout": {"base": base_new_metrics["supertonic_heldout_voice_holdout"], "supertonic_joint": super_metrics["supertonic_heldout_voice_holdout"]},
            "supertonic_heldout_voice_m5": {"base": base_new_metrics["supertonic_heldout_voice_m5"], "supertonic_joint": super_metrics["supertonic_heldout_voice_m5"]},
            "supertonic_heldout_voice_f5": {"base": base_new_metrics["supertonic_heldout_voice_f5"], "supertonic_joint": super_metrics["supertonic_heldout_voice_f5"]},
        },
        "decision": decision,
        "accepted_parent": "none",
        "limitations": [
            "All training remains synthetic.",
            "Supertonic preset styles are not real speakers or demographic evidence.",
            "Supertonic model-license obligations may attach to downstream trained models.",
            "No real calibration speech exists.",
            "FLEURS-v2 and ARTUR-J are development gates, not a final blind test.",
        ],
    }
    assert_public_report_safe(public)
    write_json(REPORT_JSON, public)
    write_markdown_report(REPORT_MD, public)
    result = {"status": "PASSED", "json_sha256": file_sha256(REPORT_JSON), "markdown_sha256": file_sha256(REPORT_MD), "scientific_classification": decision["classification"], "accepted_parent": "none"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Experiment 0011: Corpus-v2 Supertonic 3 Multi-voice Joint-adapter Diagnostic",
        "",
        f"Status: **{payload.get('status')}**",
        "",
        "This diagnostic trains one frozen-base RNNT joint adapter on Supertonic 3 preset voice-style synthetic audio. The data status is `DIAGNOSTIC_ONLY`; no checkpoint or adapter is accepted as a parent.",
        "",
        "## Authorization",
        "",
        f"- Certificate status: `{payload['authorization']['status']}`",
        f"- Certificate SHA256: `{payload['authorization']['sha256']}`",
        f"- Audio certificate SHA256: `{payload['authorization']['audio_certificate_sha256']}`",
        "",
        "## Synthetic Audio",
        "",
        f"- Training final WAVs: {payload['supertonic_audio']['counts']['final_training_files']}",
        f"- Held-out final WAVs: {payload['supertonic_audio']['counts']['final_holdout_files']}",
        f"- Training styles: `{', '.join(payload['supertonic_audio']['training_voice_styles'])}`",
        f"- Held-out styles: `{', '.join(payload['supertonic_audio']['held_out_voice_styles'])}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Split | Base WER/CER | Piper joint WER/CER | Supertonic joint WER/CER | Empty base/Piper/Supertonic |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("piper_selected_training", "piper_synthetic_holdout", "fleurs_v2", "artur_j"):
        row = payload["metric_comparison"][split]
        lines.append(
            f"| {split} | {row['base']['wer']}/{row['base']['cer']} | {row['piper_joint']['wer']}/{row['piper_joint']['cer']} | "
            f"{row['supertonic_joint']['wer']}/{row['supertonic_joint']['cer']} | {row['base']['empty']}/{row['piper_joint']['empty']}/{row['supertonic_joint']['empty']} |"
        )
    decision = payload["decision"]
    lines.extend(
        [
            "",
            "## Supertonic Diagnostics",
            "",
            f"- Training-voice probe base/adapter WER/CER: {payload['metric_comparison']['supertonic_training_voice_probe']['base']['wer']}/{payload['metric_comparison']['supertonic_training_voice_probe']['base']['cer']} -> {payload['metric_comparison']['supertonic_training_voice_probe']['supertonic_joint']['wer']}/{payload['metric_comparison']['supertonic_training_voice_probe']['supertonic_joint']['cer']}",
            f"- Held-out voice holdout base/adapter WER/CER: {payload['metric_comparison']['supertonic_heldout_voice_holdout']['base']['wer']}/{payload['metric_comparison']['supertonic_heldout_voice_holdout']['base']['cer']} -> {payload['metric_comparison']['supertonic_heldout_voice_holdout']['supertonic_joint']['wer']}/{payload['metric_comparison']['supertonic_heldout_voice_holdout']['supertonic_joint']['cer']}",
            f"- M5 held-out base/adapter WER/CER: {payload['metric_comparison']['supertonic_heldout_voice_m5']['base']['wer']}/{payload['metric_comparison']['supertonic_heldout_voice_m5']['base']['cer']} -> {payload['metric_comparison']['supertonic_heldout_voice_m5']['supertonic_joint']['wer']}/{payload['metric_comparison']['supertonic_heldout_voice_m5']['supertonic_joint']['cer']}",
            f"- F5 held-out base/adapter WER/CER: {payload['metric_comparison']['supertonic_heldout_voice_f5']['base']['wer']}/{payload['metric_comparison']['supertonic_heldout_voice_f5']['base']['cer']} -> {payload['metric_comparison']['supertonic_heldout_voice_f5']['supertonic_joint']['wer']}/{payload['metric_comparison']['supertonic_heldout_voice_f5']['supertonic_joint']['cer']}",
            "",
            "## Decision",
            "",
            f"- Piper holdout gain: `{decision['piper_holdout_gain']['passes']}`",
            f"- Supertonic held-out voice gain: `{decision['supertonic_heldout_voice_gain']['passes']}`",
            f"- Piper joint burden: {decision['piper_joint_burden']}",
            f"- Supertonic joint burden: {decision['supertonic_joint_burden']}",
            f"- Burden reduction: {decision['burden_reduction_percent']}%",
            f"- Scientific classification: `{decision['classification']}`",
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
    parser = argparse.ArgumentParser(description="Run the corpus-v2 Supertonic 3 multi-voice joint-adapter diagnostic.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"verify", "evaluate-base-new-splits", "train", "evaluate", "verify-artifact"}:
        require_env()
    if args.stage == "verify":
        result = run_verify(config, args.config, args.progress_interval_seconds)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.stage == "evaluate-base-new-splits":
        result = evaluate_base_new_splits(config)
        print(json.dumps({"status": result["status"], "models": sorted(result["models"])}, indent=2, sort_keys=True))
        return 0
    if args.stage == "train":
        result = train(config, args.config, args.progress_interval_seconds)
        print(json.dumps({"status": result["status"], "arm": result["arm"]}, indent=2, sort_keys=True))
        return 0
    if args.stage == "verify-artifact":
        result = verify_artifact(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "PASSED" else 1
    if args.stage == "evaluate":
        result = evaluate(config, args.config)
        print(json.dumps({"status": result["status"], "models": sorted(result["models"])}, indent=2, sort_keys=True))
        return 0
    if args.stage == "summarize":
        summarize(config, args.config)
        return 0
    parser.error(f"unsupported stage: {args.stage}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
