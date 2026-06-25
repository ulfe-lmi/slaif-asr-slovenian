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

from slaif_asr.batched_streaming import (
    NvidiaSmiMonitor,
    compare_predictions,
    file_sha256,
    load_local_predictions,
    parse_monitor_csv,
)
from slaif_asr.corpus_v2_scoring import (
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NEMO_REVISION,
    checkpoint_path,
    verify_runtime_identities,
)
from slaif_asr.corpus_v2_training import (
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
    add_joint_adapter,
    adapter_parameters,
    compare_adapter_state,
    compare_base_state,
    configure_only_adapter_trainable,
    disable_joint_adapters,
    enable_for_target_language,
    enabled_joint_adapters,
    expected_trainable_count,
    load_adapter_artifact,
    load_adapter_spec,
    optimizer_parameter_ids,
    save_adapter_artifact,
    state_dict_cpu,
    verify_optimizer_scope,
)


DEFAULT_CONFIG = Path("configs/experiments/corpus_v2_slovenian_joint_adapter_v1.json")
REPORT_JSON = Path("docs/experiments/0010-corpus-v2-slovenian-joint-adapter-diagnostic.json")
REPORT_MD = Path("docs/experiments/0010-corpus-v2-slovenian-joint-adapter-diagnostic.md")
CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v2-joint-adapter-diagnostic-v1.json")
ARM_NAME = "sl_si_joint_adapter_dim32"
EXPECTED_PROMPT_BURDEN = 16.361

EXPECTED_BASE_METRICS = {
    "selected_training": {"wer": 93.032, "cer": 61.623, "empty": 41},
    "synthetic_holdout": {"wer": 84.317, "cer": 47.295, "empty": 17},
    "fleurs_v2": {"wer": 52.703, "cer": 16.423, "empty": 1},
    "artur_j": {"wer": 67.453, "cer": 29.016, "empty": 12},
}
EXPECTED_PROMPT_METRICS = {
    "selected_training": {"wer": 69.955, "cer": 26.405, "empty": 0},
    "synthetic_holdout": {"wer": 73.137, "cer": 27.474, "empty": 2},
    "fleurs_v2": {"wer": 61.470, "cer": 20.347, "empty": 0},
    "artur_j": {"wer": 71.123, "cer": 25.796, "empty": 0},
}


def require_env() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE must be exactly 0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def configure_torch() -> Any:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(repo_path(path))
    if config.get("work_order_id") != "0022":
        raise ValueError("joint-adapter diagnostic config must belong to work order 0022")
    training = config.get("training", {})
    required = {
        "batch_size": 8,
        "epochs": 12,
        "sample_exposures": 1920,
        "optimizer_steps": 240,
        "seed": 1234,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected}")
    if float(training.get("learning_rate", -1.0)) != 0.001:
        raise ValueError("learning_rate must be 0.001")
    if training.get("precision") != "fp32" or training.get("tf32") is not False:
        raise ValueError("training must use FP32 with TF32 disabled")
    return config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def suppress_nemo_stream_logging() -> None:
    try:
        from nemo.utils import logging as nemo_logging

        nemo_logging.remove_stream_handlers()
        nemo_logging.set_verbosity(nemo_logging.ERROR)
    except Exception:
        return


def restore_base_model(config: dict[str, Any], *, reporter: LiveProgressReporter | None = None):
    import nemo.collections.asr as nemo_asr

    suppress_nemo_stream_logging()
    checkpoint = repo_path(config["model"]["checkpoint_path"]).resolve()
    if reporter:
        reporter.start("restoring untouched base model")
    with heartbeat_thread(reporter, interval_seconds=5.0, message="base model restore in progress") if reporter else nullcontext():
        model = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0")
    model = model.cuda().eval()
    if hasattr(model, "spec_augmentation"):
        model.spec_augmentation = None
    if reporter:
        reporter.complete("base model restored")
    return model


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


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


def mean_loss(model: Any, prompt_index: int, records: list[Any], *, device: str) -> float:
    import torch

    losses = []
    with torch.no_grad():
        for record in records:
            loss = rnnt_loss(model, make_training_batch(model, [record], device=device), prompt_index)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def verify_certificate(config_path: Path, *, require_head: bool) -> dict[str, Any]:
    config = load_config(config_path)
    cert_path = repo_path(config["authorization"]["certificate_path"])
    if require_head:
        tracked = git_tracked_and_clean_at_head(cert_path)
    else:
        tracked = {"path": str(cert_path.relative_to(Path.cwd())), "tracked": False, "clean": False, "matches_head": False}
    cert = read_json(cert_path)
    if cert.get("status") != "DIAGNOSTIC_ONLY":
        raise RuntimeError("joint-adapter certificate status must be DIAGNOSTIC_ONLY")
    if cert.get("work_order_id") != "0022":
        raise RuntimeError("joint-adapter certificate work-order mismatch")
    if cert.get("experiment_config_sha256") != file_sha256(repo_path(config_path)):
        raise RuntimeError("joint-adapter certificate experiment config SHA mismatch")
    adapter_config = repo_path(config["adapter"]["config"])
    if cert.get("adapter_config_sha256") != file_sha256(adapter_config):
        raise RuntimeError("joint-adapter certificate adapter config SHA mismatch")
    identities = verify_all_input_identities(config, check_gpu=False)
    return {"certificate": cert, "tracked": tracked, "identities": identities}


def verify_baseline_report() -> dict[str, Any]:
    path = Path("docs/experiments/0008-corpus-v2-prompt-column-diagnostic.json")
    if file_sha256(path) != "117ec8bbb97580db3e9ccf13a118a8472aa06930f42417171046e487e8ba411a":
        raise RuntimeError("Experiment 0008 report SHA mismatch")
    report = read_json(path)
    for model_name, expected in (("base", EXPECTED_BASE_METRICS), ("a100_batched", EXPECTED_PROMPT_METRICS)):
        for split, row in expected.items():
            metrics = report["evaluation"]["models"][model_name]["splits"][split]["metrics"]["normalized"]
            empty = report["evaluation"]["models"][model_name]["splits"][split]["metrics"]["raw"]["empty_hypothesis_count"]
            if round(float(metrics["corpus_wer"]), 3) != row["wer"]:
                raise RuntimeError(f"{model_name} {split} WER mismatch")
            if round(float(metrics["corpus_cer"]), 3) != row["cer"]:
                raise RuntimeError(f"{model_name} {split} CER mismatch")
            if int(empty) != row["empty"]:
                raise RuntimeError(f"{model_name} {split} empty count mismatch")
    return {"path": str(path), "sha256": file_sha256(path)}


def prepare_adapter_model(model: Any, config: dict[str, Any], *, enable: bool) -> dict[str, Any]:
    spec = load_adapter_spec(config["adapter"]["config"])
    summary = add_joint_adapter(model, spec)
    trainable = configure_only_adapter_trainable(model, spec.name)
    if trainable["trainable_parameters"] != expected_trainable_count(summary["joint_hidden"], spec.bottleneck_dim):
        raise RuntimeError("joint-adapter trainable parameter count mismatch")
    if enable:
        enable_for_target_language(model, "sl-SI", adapter_name=spec.name)
    else:
        disable_joint_adapters(model)
    return summary | trainable


def zero_init_parity_probe(config: dict[str, Any], config_path: Path, *, reporter: LiveProgressReporter | None = None) -> dict[str, Any]:
    torch = configure_torch()
    stage_dir = run_dir(config) / "authorization" / "zero-init-parity"
    model = restore_base_model(config, reporter=reporter)
    spec = load_adapter_spec(config["adapter"]["config"])
    hidden = int(model.joint.joint_hidden)
    torch.manual_seed(1234)
    f = torch.randn(2, 5, hidden, device="cuda")
    g = torch.randn(2, 4, hidden, device="cuda")
    with torch.no_grad():
        base_logits = model.joint.joint_after_projection(f, g).detach().cpu()
    add_joint_adapter(model, spec)
    enable_for_target_language(model, "sl-SI", adapter_name=spec.name)
    with torch.no_grad():
        enabled_logits = model.joint.joint_after_projection(f, g).detach().cpu()
    disable_joint_adapters(model)
    with torch.no_grad():
        disabled_logits = model.joint.joint_after_projection(f, g).detach().cpu()
    enable_for_target_language(model, "en-US", adapter_name=spec.name)
    non_sl_enabled = enabled_joint_adapters(model)
    logit_parity = bool(torch.equal(base_logits, enabled_logits) and torch.equal(base_logits, disabled_logits))
    if not logit_parity or non_sl_enabled:
        raise RuntimeError("zero-init or language-gating parity failed")
    enable_for_target_language(model, "sl-SI", adapter_name=spec.name)
    checkpoint = stage_dir / "zero-init-enabled.local.nemo"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter.heartbeat(message="saving zero-init adapter checkpoint")
        with heartbeat_thread(reporter, interval_seconds=5.0, message="zero-init checkpoint save in progress"):
            model.save_to(str(checkpoint))
    else:
        model.save_to(str(checkpoint))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    records = load_synthetic_eval_records(config, "selected_training")[:8]
    base_arm = run_evaluation_with_progress(
        config,
        records=records,
        checkpoint=checkpoint_path(),
        split="zero_init_base",
        run_subdir=stage_dir / "base",
        reporter_parent=reporter,
    )
    adapter_arm = run_evaluation_with_progress(
        config,
        records=records,
        checkpoint=checkpoint,
        split="zero_init_adapter",
        run_subdir=stage_dir / "adapter",
        reporter_parent=reporter,
    )
    base_predictions = load_local_predictions(stage_dir / "base" / "predictions.local.jsonl")
    adapter_predictions = load_local_predictions(stage_dir / "adapter" / "predictions.local.jsonl")
    comparison = compare_predictions(
        records,
        base_predictions,
        adapter_predictions,
        baseline_metrics=base_arm["metrics"],
        candidate_metrics=adapter_arm["metrics"],
    )
    result = {
        "logit_parity": logit_parity,
        "disabled_parity": True,
        "non_slovenian_enabled_adapters": non_sl_enabled,
        "hypothesis_probe_rows": len(records),
        "exact_hypothesis_mismatches": comparison.exact_mismatch_count,
        "normalized_hypothesis_mismatches": comparison.normalized_mismatch_count,
        "metric_differences": comparison.metric_differences,
        "passed": comparison.exact_parity,
    }
    write_json(stage_dir / "summary.local.json", result)
    if not result["passed"]:
        raise RuntimeError("zero-init hypothesis parity failed")
    return result


def run_evaluation_with_progress(
    config: dict[str, Any],
    *,
    records,
    checkpoint: Path,
    split: str,
    run_subdir: Path,
    reporter_parent: LiveProgressReporter | None = None,
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
    cert = verify_certificate(config_path, require_head=True)
    runtime = verify_runtime_identities(check_gpu=True)
    baseline = verify_baseline_report()
    reporter = LiveProgressReporter(stage="verify", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "verify.local.ndjson")
    reporter.start("verifying joint adapter diagnostic")
    parity = zero_init_parity_probe(config, config_path, reporter=reporter)
    reporter.complete("verification complete")
    result = {"status": "PASSED", "authorization": cert["tracked"], "runtime": runtime, "baseline": baseline, "zero_init_parity": parity}
    write_json(run_dir(config) / "authorization" / "verify.local.json", result)
    return result


def train(config: dict[str, Any], config_path: Path, interval: float) -> dict[str, Any]:
    verify_certificate(config_path, require_head=True)
    runtime = verify_runtime_identities(check_gpu=True)
    torch = configure_torch()
    records = load_training_records(config)
    reporter = LiveProgressReporter(stage="train", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "train.local.ndjson")
    reporter.start("training joint adapter")
    model = restore_base_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "restore.local.ndjson"))
    model.eval()
    base_state = state_dict_cpu(model)
    adapter_summary = prepare_adapter_model(model, config, enable=True)
    initial_state = state_dict_cpu(model)
    prompt_selection = derive_prompt_column_selection(model, "sl-SI")
    trainable_count = adapter_summary["trainable_parameters"]
    expected_count = adapter_summary["expected_trainable_parameters"]
    if trainable_count != expected_count:
        raise RuntimeError("trainable count mismatch")
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=float(config["training"]["learning_rate"]), weight_decay=0.0)
    verify_optimizer_scope(optimizer, model)

    probe_records = select_probe_records(records, int(config["training"]["probe_rows"]))
    initial_probe = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
    initial_full = mean_loss(model, prompt_selection.prompt_index, records, device="cuda")
    probe_curve = [{"epoch": 0, "mean_loss": round(initial_probe, 6)}]
    adapter_norm_curve = []
    grad_norms = []
    optimizer_steps = 0
    sample_exposures = 0
    audio_seconds = 0.0
    padded_audio_seconds = 0.0
    rolling_losses: list[float] = []
    arm_dir = run_dir(config) / ARM_NAME
    monitor_path = arm_dir / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index="1", output_csv=monitor_path, interval_seconds=0.2)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    monitor.start()
    try:
        total_steps = int(config["training"]["optimizer_steps"])
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            layout = deterministic_epoch_batches(
                records,
                batch_size=int(config["training"]["batch_size"]),
                epoch=epoch,
                seed=int(config["training"]["seed"]),
                bucketed=True,
            )
            assert_epoch_covers_once(layout, len(records))
            for batch_indices in layout.batches:
                batch_records = [records[index] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                loss = rnnt_loss(model, make_training_batch(model, batch_records, device="cuda"), prompt_selection.prompt_index)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite joint-adapter training loss")
                loss.backward()
                grad_norm, grads_ok = finite_grad_norm(adapter_parameters(model))
                if not grads_ok:
                    raise RuntimeError("non-finite joint-adapter gradient")
                for name, parameter in model.named_parameters():
                    if not name.startswith(f"joint.adapter_layer.{ADAPTER_NAME}.") and parameter.grad is not None:
                        raise RuntimeError(f"pretrained parameter received gradient: {name}")
                optimizer.step()
                optimizer_steps += 1
                sample_exposures += len(batch_records)
                audio_seconds += sum(row.duration for row in batch_records)
                padded_audio_seconds += max(row.duration for row in batch_records) * len(batch_records)
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
    reporter.heartbeat(
        message="post-training validation starting",
        step=optimizer_steps,
        total_steps=int(config["training"]["optimizer_steps"]),
    )
    post_context = heartbeat_thread(
        reporter,
        interval_seconds=interval,
        message="post-training validation in progress",
        fields=lambda: {"step": optimizer_steps, "total_steps": int(config["training"]["optimizer_steps"])},
    )
    post_context.__enter__()
    wall = time.perf_counter() - start
    final_probe = mean_loss(model, prompt_selection.prompt_index, probe_records, device="cuda")
    final_full = mean_loss(model, prompt_selection.prompt_index, records, device="cuda")
    trained_state = state_dict_cpu(model)
    base_integrity = compare_base_state(initial_state, trained_state)
    adapter_integrity = compare_adapter_state(initial_state, trained_state)
    if not base_integrity["base_tensors_identical"]:
        raise RuntimeError("pretrained tensor changed during joint-adapter training")
    artifact_path = arm_dir / "artifacts" / "sl-si-joint-adapter-v1.pt"
    artifact_sha = save_adapter_artifact(
        artifact_path,
        model=model,
        spec=load_adapter_spec(config["adapter"]["config"]),
        metadata={
            "base_checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
            "training_manifest_sha256": config["data"]["selected_training_manifest_sha256"],
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
    enable_for_target_language(model, "sl-SI")
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
        "gradient_norm": {
            "min": round(min(grad_norms), 6),
            "max": round(max(grad_norms), 6),
            "final": round(grad_norms[-1], 6),
        },
        "adapter_norm_curve": adapter_norm_curve,
        "wall_time_seconds": round(wall, 6),
        "examples_per_second": round(sample_exposures / wall, 6) if wall else None,
        "audio_seconds_per_wall_second": round(audio_seconds / wall, 6) if wall else None,
        "padding_ratio": round(padded_audio_seconds / audio_seconds, 6) if audio_seconds else None,
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "adapter": adapter_summary,
        "trainable_parameter_count": trainable_count,
        "base_integrity": base_integrity,
        "adapter_integrity": adapter_integrity,
        "artifact_sha256": artifact_sha,
        "restore_integrity": read_json(arm_dir / "restore-integrity.local.json"),
        "evaluation_checkpoint_sha256": file_sha256(checkpoint_out),
        "runtime": runtime,
    }
    write_json(arm_dir / "training-summary.local.json", payload)
    post_context.__exit__(None, None, None)
    reporter.complete("training complete", step=optimizer_steps, total_steps=int(config["training"]["optimizer_steps"]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def verify_artifact(config: dict[str, Any]) -> dict[str, Any]:
    configure_torch()
    spec = load_adapter_spec(config["adapter"]["config"])
    model = restore_base_model(config)
    base_state = state_dict_cpu(model)
    artifact = run_dir(config) / ARM_NAME / "artifacts" / "sl-si-joint-adapter-v1.pt"
    payload = load_adapter_artifact(artifact, model=model, spec=spec)
    restored_state = state_dict_cpu(model)
    base_integrity = compare_base_state(base_state, restored_state)
    if not base_integrity["base_tensors_identical"]:
        raise RuntimeError("adapter restore changed base tensors")
    if enabled_joint_adapters(model):
        raise RuntimeError("adapter must be disabled by default after restore")
    report = {
        "status": "PASSED",
        "artifact_name": payload["adapter_name"],
        "base_integrity": base_integrity,
        "disabled_after_restore": True,
    }
    write_json(run_dir(config) / ARM_NAME / "restore-integrity.local.json", report)
    return report


def evaluate(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    verify_certificate(config_path, require_head=True)
    verify_runtime_identities(check_gpu=True)
    checkpoint = run_dir(config) / ARM_NAME / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
    if not checkpoint.exists():
        raise RuntimeError("joint-adapter evaluation checkpoint is missing")
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
        arm = run_evaluation_with_progress(
            config,
            records=records,
            checkpoint=checkpoint,
            split=split_name,
            run_subdir=run_dir(config) / "evaluation" / ARM_NAME / split_name,
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


def synthetic_holdout_gain(joint_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = EXPECTED_BASE_METRICS["synthetic_holdout"]
    joint = joint_metrics["synthetic_holdout"]
    wer_gain = (base["wer"] - joint["wer"]) / base["wer"] * 100.0
    cer_gain = (base["cer"] - joint["cer"]) / base["cer"] * 100.0
    return {"passes": wer_gain >= 10.0 or cer_gain >= 10.0, "wer_relative_gain": round(wer_gain, 6), "cer_relative_gain": round(cer_gain, 6)}


def real_regression_burden(metrics: dict[str, dict[str, Any]]) -> float:
    burden = 0.0
    for split in ("fleurs_v2", "artur_j"):
        base = EXPECTED_BASE_METRICS[split]
        candidate = metrics[split]
        burden += max(0.0, candidate["wer"] - base["wer"])
        burden += max(0.0, candidate["cer"] - base["cer"])
    return round(burden, 6)


def classify_joint_adapter(joint_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    holdout = synthetic_holdout_gain(joint_metrics)
    burden = real_regression_burden(joint_metrics)
    prompt_burden = real_regression_burden(EXPECTED_PROMPT_METRICS)
    non_regression = True
    improvement = False
    for split in ("fleurs_v2", "artur_j"):
        base = EXPECTED_BASE_METRICS[split]
        joint = joint_metrics[split]
        if joint["wer"] - base["wer"] > 1.0 or joint["cer"] - base["cer"] > 1.5 or joint["empty"] > base["empty"]:
            non_regression = False
        if joint["wer"] - base["wer"] <= -1.0 or joint["cer"] - base["cer"] <= -1.5:
            improvement = True
    mitigates = False
    if holdout["passes"] and not (non_regression and improvement):
        burden_reduction = (prompt_burden - burden) / prompt_burden * 100.0 if prompt_burden else 0.0
        no_worse_than_prompt = True
        for split in ("fleurs_v2", "artur_j"):
            prompt = EXPECTED_PROMPT_METRICS[split]
            joint = joint_metrics[split]
            if joint["wer"] - prompt["wer"] > 0.5 or joint["cer"] - prompt["cer"] > 0.5:
                no_worse_than_prompt = False
        mitigates = burden_reduction >= 30.0 and no_worse_than_prompt
    if holdout["passes"] and non_regression and improvement:
        classification = "SL_JOINT_ADAPTER_REAL_GAIN_DIAGNOSTIC"
    elif mitigates:
        classification = "SL_JOINT_ADAPTER_MITIGATES_PROMPT_REGRESSION"
    elif holdout["passes"]:
        classification = "SL_JOINT_ADAPTER_SYNTHETIC_ONLY"
    else:
        classification = "SL_JOINT_ADAPTER_NOT_SUPPORTED"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "synthetic_holdout_gain": holdout,
        "prompt_column_burden": prompt_burden,
        "joint_adapter_burden": burden,
        "burden_reduction_percent": round((prompt_burden - burden) / prompt_burden * 100.0, 6) if prompt_burden else None,
        "real_non_regression": non_regression,
        "real_improvement": improvement,
    }


def normalized_metrics(evaluation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = {}
    for split_name, split in evaluation["models"][ARM_NAME]["splits"].items():
        normalized = split["metrics"]["normalized"]
        raw = split["metrics"]["raw"]
        metrics[split_name] = {
            "wer": round(float(normalized["corpus_wer"]), 3),
            "cer": round(float(normalized["corpus_cer"]), 3),
            "empty": int(raw["empty_hypothesis_count"]),
        }
    return metrics


def summarize(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    auth = verify_certificate(config_path, require_head=True)
    baseline = verify_baseline_report()
    input_integrity = verify_all_input_identities(config, check_gpu=False)
    training = read_json(run_dir(config) / ARM_NAME / "training-summary.local.json")
    evaluation = read_json(run_dir(config) / "evaluation" / "summary.local.json")
    joint_metrics = normalized_metrics(evaluation)
    decision = classify_joint_adapter(joint_metrics)
    public = {
        "schema_version": "1.0",
        "experiment_id": "corpus-v2-slovenian-joint-adapter-diagnostic-v1",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_head(),
        "authorization": {
            "status": auth["certificate"]["status"],
            "sha256": file_sha256(CERTIFICATE_PATH),
            "work_order_id": auth["certificate"]["work_order_id"],
            "tracked_before_execution": auth["tracked"],
            "adapter_config_sha256": auth["certificate"]["adapter_config_sha256"],
            "experiment_config_sha256": auth["certificate"]["experiment_config_sha256"],
        },
        "baseline": baseline,
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
            )
            if key in training
        },
        "evaluation": evaluation,
        "metric_comparison": {
            split: {
                "base": EXPECTED_BASE_METRICS[split],
                "prompt_column": EXPECTED_PROMPT_METRICS[split],
                "joint_adapter": joint_metrics[split],
            }
            for split in ("selected_training", "synthetic_holdout", "fleurs_v2", "artur_j")
        },
        "decision": decision,
        "accepted_parent": "none",
        "limitations": [
            "Single original Piper voice family.",
            "No real training or calibration speech.",
            "Synthetic holdout is diagnostic only and not real-generalization evidence.",
            "FLEURS-v2 and ARTUR-J are development gates, not a final blind test.",
        ],
    }
    assert_public_report_safe(public)
    write_json(REPORT_JSON, public)
    write_markdown_report(REPORT_MD, public)
    result = {
        "status": "PASSED",
        "json_sha256": file_sha256(REPORT_JSON),
        "markdown_sha256": file_sha256(REPORT_MD),
        "scientific_classification": decision["classification"],
        "accepted_parent": "none",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Experiment 0010: Corpus-v2 Slovenian Joint-adapter Diagnostic",
        "",
        f"Status: **{payload.get('status')}**",
        "",
        "This diagnostic trains one NeMo-native residual adapter in the frozen RNNT joint hidden layer. The data status is `DIAGNOSTIC_ONLY`; no checkpoint or adapter is accepted as a parent.",
        "",
        "## Authorization",
        "",
        f"- Certificate status: `{payload['authorization']['status']}`",
        f"- Certificate SHA256: `{payload['authorization']['sha256']}`",
        f"- Adapter config SHA256: `{payload['authorization']['adapter_config_sha256']}`",
        "",
        "## Adapter",
        "",
        f"- Module: `{payload['adapter']['module']}`",
        f"- Name: `{payload['adapter']['name']}`",
        f"- Joint hidden dimension: {payload['adapter']['joint_hidden']}",
        f"- Trainable parameters: {payload['adapter']['exact_trainable_parameters']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Split | Base WER/CER | Prompt-column WER/CER | Joint-adapter WER/CER | Empty base/prompt/joint |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in payload["metric_comparison"].items():
        lines.append(
            f"| {split} | {row['base']['wer']}/{row['base']['cer']} | "
            f"{row['prompt_column']['wer']}/{row['prompt_column']['cer']} | "
            f"{row['joint_adapter']['wer']}/{row['joint_adapter']['cer']} | "
            f"{row['base']['empty']}/{row['prompt_column']['empty']}/{row['joint_adapter']['empty']} |"
        )
    decision = payload["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Synthetic-holdout gain: `{decision['synthetic_holdout_gain']['passes']}`",
            f"- Prompt-column burden: {decision['prompt_column_burden']}",
            f"- Joint-adapter burden: {decision['joint_adapter_burden']}",
            f"- Scientific classification: `{decision['classification']}`",
            "- Accepted parent: `none`",
            "",
            "## Limitations",
            "",
            "- Single original Piper voice family.",
            "- No real calibration speech.",
            "- Synthetic holdout is not real-generalization evidence.",
            "- Development gates are not a final blind test.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the corpus-v2 Slovenian joint-adapter diagnostic.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"verify", "train", "evaluate", "verify-artifact"}:
        require_env()
    if args.stage == "verify":
        result = run_verify(config, args.config, args.progress_interval_seconds)
        print(json.dumps(result, indent=2, sort_keys=True))
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
