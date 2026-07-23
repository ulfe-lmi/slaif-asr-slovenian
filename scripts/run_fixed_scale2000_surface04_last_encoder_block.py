#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from slaif_asr.artur_earlystop import load_controller_dev_records
from slaif_asr.batched_streaming import (
    NvidiaSmiMonitor,
    file_sha256,
    load_local_predictions,
    metrics_for,
    parse_monitor_csv,
    run_batched_arm,
)
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_scoring import (
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NEMO_REVISION,
    nemo_streaming_script,
    runtime_environment,
    verify_runtime_identities,
)
from slaif_asr.corpus_v2_training import assert_epoch_covers_once, deterministic_epoch_batches, make_training_batch
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.directional_evaluation import load_directional_suite, split_predictions, write_privacy_safe_suite_manifest
from slaif_asr.emission_rnnt_finetune import (
    BASE_DIRECTIONAL_METRICS,
    SCALE2000_JOINT_ADAPTER_METRICS,
    finite_grad_norm,
    git_head,
    load_scheduled_round_records,
    local_path,
    metric_row,
    probe_records,
    protected_file_fingerprints,
    read_json,
    rnnt_audio_loss,
    verify_all_inputs,
    verify_protected_file_fingerprints,
    write_json,
)
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread
from slaif_asr.prompt_column import derive_prompt_column_selection
from slaif_asr.rtx2080ti_policy import nvidia_smi_inventory, require_single_visible_rtx2080ti
from slaif_asr.trainable_surface_sweep import (
    ALLOWED_TRAINABLE_PREFIXES,
    PR36_METRICS,
    SURFACE_ID,
    assert_public_report_safe,
    bind_post_selection_metrics,
    changed_tensor_summary,
    classify_surface04,
    component_or_not_recorded,
    configure_surface04_trainable,
    load_config,
    mark_controller_selection,
    microbatch_plan,
    optimizer_parameter_groups,
    select_microbatch,
    set_surface04_training_mode,
    should_stop_controller_curve,
    verify_optimizer_scope,
)


DEFAULT_CONFIG = Path("configs/experiments/fixed-scale2000-surface04-last-encoder-block.json")
ARM_NAME = "fixed_scale2000_surface04_last_encoder_block"
FAST_DIRECTIONAL_CONFIG = Path("configs/experiments/fast_batched_directional_replay_v1.json")
REPORT_JSON = Path("docs/experiments/0024-fixed-scale2000-surface04-last-encoder-block.json")
REPORT_MD = Path("docs/experiments/0024-fixed-scale2000-surface04-last-encoder-block.md")
CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v4-fixed-scale2000-surface04-diagnostic-v1.json")

PR39_METRICS = {
    "piper_synthetic_holdout": {"wer": 44.565, "cer": 16.428, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 18.711, "cer": 6.196, "empty": 0},
    "fleurs_v2": {"wer": 48.023, "cer": 15.946, "empty": 0},
    "artur_j": {"wer": 57.274, "cer": 20.375, "empty": 0},
}

_BASE_PATH = Path(__file__).with_name("run_corpus_v2_joint_adapter_diagnostic.py")
_BASE_SPEC = importlib.util.spec_from_file_location("_slaif_surface04_model_restore", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import model restore helper")
_BASE = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_BASE)


def ensure_cuda_nvcc_process_env() -> None:
    cuda_home = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_nvcc"
    if not cuda_home.exists():
        return
    cuda_bin = str(cuda_home / "bin")
    nvvm_lib = str(cuda_home / "nvvm" / "lib64")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    ld_entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry]
    needs_reexec = (
        os.environ.get("CUDA_HOME") != str(cuda_home)
        or os.environ.get("CUDA_PATH") != str(cuda_home)
        or cuda_bin not in path_entries
        or nvvm_lib not in ld_entries
    )
    if needs_reexec and os.environ.get("SLAIF_NVCC_ENV_READY") != "1":
        env = os.environ.copy()
        env["CUDA_HOME"] = str(cuda_home)
        env["CUDA_PATH"] = str(cuda_home)
        env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = nvvm_lib + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        env["SLAIF_NVCC_ENV_READY"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    os.environ.setdefault("CUDA_HOME", str(cuda_home))
    os.environ.setdefault("CUDA_PATH", str(cuda_home))
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")


def run_dir(config: dict[str, Any]) -> Path:
    return local_path(config["local_outputs"]["run_root"])


def configure_torch() -> Any:
    ensure_cuda_nvcc_process_env()
    torch = _BASE.configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def restore_base_model(config: dict[str, Any], reporter: LiveProgressReporter | None = None) -> Any:
    return _BASE.restore_base_model(config, reporter=reporter)


def restore_local_checkpoint(path: Path, reporter: LiveProgressReporter | None = None) -> Any:
    import nemo.collections.asr as nemo_asr

    _BASE.suppress_nemo_stream_logging()
    if reporter:
        reporter.start("restoring completed round checkpoint")
    context = heartbeat_thread(reporter, interval_seconds=10.0, message="checkpoint restore") if reporter else nullcontext()
    with context:
        model = nemo_asr.models.ASRModel.restore_from(restore_path=str(path), map_location="cuda:0")
    model = model.cuda()
    if reporter:
        reporter.complete("checkpoint restored")
    return model


def _trainable_parameters(model: Any) -> list[Any]:
    return [parameter for _name, parameter in model.named_parameters() if parameter.requires_grad]


def _zero_grad(model: Any) -> None:
    for parameter in _trainable_parameters(model):
        parameter.grad = None


def _assert_gradient_scope(model: Any) -> None:
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and not name.startswith(ALLOWED_TRAINABLE_PREFIXES)
    ]
    if bad:
        raise RuntimeError(f"frozen parameters received gradients: {bad[:10]}")


def model_fingerprints(model: Any) -> dict[str, str]:
    # All pinned model tensors are FP32 in this work order, so NumPy byte views
    # provide stable bitwise fingerprints without retaining a second model copy.
    result: dict[str, str] = {}
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def fingerprint_integrity(before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    class Fingerprint:
        shape = ()

        def __init__(self, value: str):
            self.value = value

        def __eq__(self, other: object) -> "Fingerprint":
            return Fingerprint(self.value == getattr(other, "value", None))

        def all(self) -> bool:
            return bool(self.value)

    return changed_tensor_summary(
        {name: Fingerprint(value) for name, value in before.items()},
        {name: Fingerprint(value) for name, value in after.items()},
    )


def names_fingerprint(fingerprints: dict[str, str], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(fingerprints[name].encode("ascii"))
    return digest.hexdigest()


def runtime_summary(hardware: Any, torch: Any) -> dict[str, Any]:
    return {
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nemo_revision": NEMO_REVISION,
        "physical_selector": hardware.physical_selector,
        "logical_device": hardware.logical_device,
        "gpu": hardware.device_name,
        "visible_gpu_count": hardware.visible_device_count,
        "precision": "fp32",
        "tf32": False,
    }


def stage_verify_inputs(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    payload = {"status": "PASSED", "work_order_id": "0037", **verify_all_inputs(config)}
    controller_certificate = read_json(REPO_ROOT / config["controller_dev"]["certificate"])
    controller_manifest = local_path(config["controller_dev"]["manifest"])
    if controller_certificate.get("partition_id") != "artur-controller-dev-v1":
        raise RuntimeError("controller-dev certificate partition mismatch")
    if controller_certificate.get("manifest_sha256") != config["controller_dev"]["manifest_sha256"]:
        raise RuntimeError("controller-dev certificate hash mismatch")
    if file_sha256(controller_manifest) != config["controller_dev"]["manifest_sha256"]:
        raise RuntimeError("local controller-dev manifest hash mismatch")
    payload["controller_dev"] = {
        "partition_id": "artur-controller-dev-v1",
        "manifest_sha256": config["controller_dev"]["manifest_sha256"],
        "rows": sum(1 for _line in controller_manifest.open("r", encoding="utf-8")),
    }
    if payload["controller_dev"]["rows"] != 256:
        raise RuntimeError("controller-dev row count mismatch")
    write_json(run_dir(config) / "verification" / "inputs.local.json", payload)
    print(json.dumps({"status": "PASSED", "data": payload["local_artifacts"], "controller_dev": payload["controller_dev"]}, sort_keys=True))
    return payload


def stage_probe_hardware(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inventory = [row.to_dict() for row in nvidia_smi_inventory()]
    rtx = [row for row in inventory if "RTX 2080 Ti" in row["name"]]
    payload = {
        "status": "PASSED" if rtx else "ENVIRONMENT_BLOCKED",
        "inventory": inventory,
        "rtx2080ti_count": len(rtx),
        "second_2080ti_detected": len(rtx) >= 2,
    }
    write_json(run_dir(config) / "verification" / "hardware.local.json", payload)
    print(json.dumps(payload, sort_keys=True))
    if not rtx:
        raise RuntimeError("no RTX 2080 Ti available")
    return payload


def stage_probe_surface(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    hardware = require_single_visible_rtx2080ti()
    torch = configure_torch()
    verify_runtime_identities(check_gpu=False)
    reporter = LiveProgressReporter(stage="probe_surface", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "surface.local.ndjson")
    model = restore_base_model(config, reporter=reporter)
    summary = configure_surface04_trainable(model)
    expected = {
        "decoder_parameter_count": 14940160,
        "joint_parameter_count": 9455648,
        "final_encoder_block_parameter_count": 25189376,
        "trainable_parameter_count": 49585184,
    }
    for key, value in expected.items():
        if getattr(summary, key) != value:
            raise RuntimeError(f"live model {key} mismatch: {getattr(summary, key)} != {value}")
    payload = {"status": "PASSED", "surface": summary.to_dict(), "runtime": runtime_summary(hardware, torch)}
    write_json(run_dir(config) / "verification" / "surface.local.json", payload)
    print(json.dumps(payload, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _representative_records(config: dict[str, Any], count: int = 8, *, longest: bool = True) -> list[Any]:
    rounds, _meta, _summary = load_scheduled_round_records(config)
    return sorted(
        rounds[1],
        key=(lambda row: (-row.duration, row.selected_training_id)) if longest else (lambda row: (row.duration, row.selected_training_id)),
    )[:count]


def _run_gradient_partition(config: dict[str, Any], records: Sequence[Any], physical: int, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    model = restore_base_model(config)
    configure_surface04_trainable(model)
    # Keep RNNTDecoder itself in eval mode so random_state_sampling cannot
    # replace the deterministic zero LSTM state. cuDNN still requires its
    # underlying LSTM in training mode for backward, with probe-only dropout
    # disabled to make grouped and singleton partitions comparable.
    model.eval()
    decoder_rnn = model.decoder.prediction["dec_rnn"]
    decoder_rnn.train()
    if getattr(decoder_rnn, "dropout", None) is not None:
        decoder_rnn.dropout.eval()
    if hasattr(decoder_rnn, "lstm"):
        decoder_rnn.lstm.train()
        decoder_rnn.lstm.dropout = 0.0
    prompt = derive_prompt_column_selection(model, "sl-SI")
    _zero_grad(model)
    weighted_loss = 0.0
    for start in range(0, len(records), physical):
        micro = records[start : start + physical]
        batch = make_training_batch(model, micro, device="cuda")
        loss = rnnt_audio_loss(model, batch, prompt.prompt_index, frozen_encoder_no_grad=False)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite accumulation probe loss")
        scale = len(micro) / len(records)
        (loss * scale).backward()
        weighted_loss += float(loss.detach().cpu()) * scale
        del loss, batch
    _assert_gradient_scope(model)
    norm, finite = finite_grad_norm(_trainable_parameters(model))
    if not finite:
        raise RuntimeError("non-finite accumulation probe gradient")
    gradients = {name: parameter.grad.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None}
    result = {"weighted_loss": weighted_loss, "gradient_norm": norm}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result, gradients


def stage_probe_microbatch(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    require_single_visible_rtx2080ti()
    torch = configure_torch()
    verify_runtime_identities(check_gpu=False)
    records = _representative_records(config, longest=True)
    outcomes: dict[int, dict[str, Any]] = {}
    reporter = LiveProgressReporter(stage="probe_microbatch", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "microbatch.local.ndjson")
    reporter.start("probing SURFACE_04 microbatch")
    for candidate in config["training"]["physical_microbatch_candidates"]:
        reporter.progress(step=len(outcomes), total_steps=3, message=f"candidate_{candidate}")
        model = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            model = restore_base_model(config)
            configure_surface04_trainable(model)
            set_surface04_training_mode(model)
            prompt = derive_prompt_column_selection(model, "sl-SI")
            batch = make_training_batch(model, records[:candidate], device="cuda")
            loss = rnnt_audio_loss(model, batch, prompt.prompt_index, frozen_encoder_no_grad=False)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite microbatch loss")
            loss.backward()
            _assert_gradient_scope(model)
            norm, finite = finite_grad_norm(_trainable_parameters(model))
            if not finite:
                raise RuntimeError("non-finite microbatch gradient")
            free, _total = torch.cuda.mem_get_info(0)
            free_mib = int(free / 1024 / 1024)
            if free_mib < 500:
                raise RuntimeError(f"only {free_mib} MiB free after warmup")
            outcomes[int(candidate)] = {
                "status": "PASSED",
                "loss": round(float(loss.detach().cpu()), 6),
                "gradient_norm": round(norm, 6),
                "free_vram_mib": free_mib,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
            }
            del loss, batch
        except Exception as exc:
            outcomes[int(candidate)] = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc).splitlines()[0][:240]}
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
    selected = select_microbatch(outcomes)
    if selected["status"] != "PASSED":
        payload = {"status": selected["status"], "candidate_outcomes": outcomes, "selected": selected}
        write_json(run_dir(config) / "verification" / "microbatch.local.json", payload)
        raise RuntimeError("SURFACE_04 does not fit physical microbatch 1")

    physical = int(selected["physical_microbatch"])
    correctness_records = _representative_records(config, longest=False)
    comparison_physical = 2 if physical == 1 else physical
    torch.manual_seed(int(config["training"]["seed"]) + 3700)
    torch.cuda.manual_seed_all(int(config["training"]["seed"]) + 3700)
    first, first_grads = _run_gradient_partition(config, correctness_records, comparison_physical, torch)
    torch.manual_seed(int(config["training"]["seed"]) + 3700)
    torch.cuda.manual_seed_all(int(config["training"]["seed"]) + 3700)
    second, second_grads = _run_gradient_partition(config, correctness_records, 1, torch)
    squared_diff = 0.0
    squared_ref = 0.0
    for name in first_grads:
        delta = first_grads[name] - second_grads[name]
        squared_diff += float(torch.sum(delta * delta))
        squared_ref += float(torch.sum(second_grads[name] * second_grads[name]))
    relative_gradient_difference = (squared_diff**0.5 / squared_ref**0.5) if squared_ref else 0.0
    relative_loss_difference = abs(first["weighted_loss"] - second["weighted_loss"]) / second["weighted_loss"] if second["weighted_loss"] else 0.0
    correctness = {
        "selected_training_microbatch": physical,
        "comparison_partition": comparison_physical,
        "reference_partition": 1,
        "relative_loss_difference": relative_loss_difference,
        "relative_gradient_difference": relative_gradient_difference,
        "passed": relative_loss_difference <= 0.005 and relative_gradient_difference <= 0.01,
    }
    if not correctness["passed"]:
        payload = {
            "status": "FAILED_ACCUMULATION_CORRECTNESS",
            "candidate_outcomes": {str(key): value for key, value in outcomes.items()},
            "selected": selected,
            "correctness": correctness,
        }
        write_json(run_dir(config) / "verification" / "microbatch.local.json", payload)
        raise RuntimeError("gradient accumulation correctness probe failed")
    payload = {"status": "PASSED", "candidate_outcomes": {str(key): value for key, value in outcomes.items()}, "selected": selected, "correctness": correctness}
    write_json(run_dir(config) / "verification" / "microbatch.local.json", payload)
    reporter.complete("microbatch probe complete")
    print(json.dumps(payload, sort_keys=True))
    return payload


def mean_probe_loss(
    model: Any,
    prompt_index: int,
    records: Sequence[Any],
    *,
    torch: Any,
    reporter: LiveProgressReporter | None = None,
    message: str,
    interval: float,
) -> float:
    model.eval()
    losses: list[float] = []
    started = time.perf_counter()
    last_emit = started
    ordered = sorted(records, key=lambda row: (-row.duration, row.selected_training_id))
    with torch.no_grad():
        for index, record in enumerate(ordered, start=1):
            batch = make_training_batch(model, [record], device="cuda")
            loss = rnnt_audio_loss(model, batch, prompt_index, frozen_encoder_no_grad=False)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite synthetic probe loss")
            losses.append(float(loss.detach().cpu()))
            del loss, batch
            now = time.perf_counter()
            if reporter and (now - last_emit >= interval or index == len(ordered)):
                reporter.progress(
                    step=index,
                    total_steps=len(ordered),
                    examples_per_second=round(index / (now - started), 6),
                    cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                    cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    message=message,
                )
                last_emit = now
    set_surface04_training_mode(model)
    return sum(losses) / len(losses)


def _controller_records(config: dict[str, Any]) -> list[Any]:
    return load_controller_dev_records(
        local_path(config["controller_dev"]["manifest"]),
        expected_sha256=config["controller_dev"]["manifest_sha256"],
        expected_rows=int(config["controller_dev"]["rows"]),
    )


def _controller_metric_row(summary: dict[str, Any]) -> dict[str, Any]:
    normalized = summary["normalized"]
    raw = summary["raw"]
    return {
        "wer": round(float(normalized["corpus_wer"]), 3),
        "cer": round(float(normalized["corpus_cer"]), 3),
        "empty": int(raw["empty_hypothesis_count"]),
        "delete": component_or_not_recorded(normalized, "deletion_rate"),
        "insert": component_or_not_recorded(normalized, "insertion_rate"),
        "substitute": component_or_not_recorded(normalized, "substitution_rate"),
    }


def evaluate_controller_checkpoint(config: dict[str, Any], checkpoint: Path, round_index: int, validation_gpu: str) -> dict[str, Any]:
    records = _controller_records(config)
    output_dir = run_dir(config) / "controller-dev" / f"round_{round_index:02d}"
    predictions_path = output_dir / "predictions.local.jsonl"
    if predictions_path.exists():
        predictions = load_local_predictions(predictions_path)
        if len(predictions) == len(records):
            return {"round": round_index, **_controller_metric_row(metrics_for(records, predictions)), "available": True, "reused": True}
    env = runtime_environment()
    env.update({"CUDA_VISIBLE_DEVICES": validation_gpu, "NVIDIA_TF32_OVERRIDE": "0", "PYTHONUNBUFFERED": "1"})
    arm = run_batched_arm(
        records=records,
        batch_size=1,
        bucketed=False,
        run_dir=output_dir,
        python_executable=Path(sys.executable),
        nemo_script=nemo_streaming_script(),
        checkpoint=checkpoint,
        context=config["controller_dev"]["att_context_size"],
        env=env,
        physical_gpu_index=validation_gpu,
        monitor_interval_seconds=1.0,
    )
    if arm.get("status") != "PASSED":
        raise RuntimeError(f"controller-dev evaluation failed at round {round_index}: {arm.get('status')}")
    predictions = load_local_predictions(predictions_path)
    return {
        "round": round_index,
        **_controller_metric_row(metrics_for(records, predictions)),
        "available": True,
        "reused": False,
        "wall_time_seconds": arm["execution"]["wall_time_seconds"],
        "rows_per_second": arm["utterances_per_second"],
        "real_time_factor": arm["end_to_end_real_time_factor"],
        "peak_gpu_memory_mib": arm["execution"]["monitor"].get("peak_memory_mib"),
    }


def _select_and_mark_controller_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = next(row for row in rows if int(row["round"]) == 0)
    marked = mark_controller_selection(rows, base_empty_count=int(base["empty"]))
    rows[:] = marked["rows"]
    return {
        "status": "PASSED",
        "partition_id": "artur-controller-dev-v1",
        "rows": rows,
        "selected_round": marked["selected_round"],
        "best_raw_wer_round": marked["best_raw_wer_round"],
        "base_empty_count": int(base["empty"]),
    }


def _refresh_training_controller_selection(config: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    controller = _select_and_mark_controller_rows(training["controller_curve"])
    training["controller_curve"] = controller["rows"]
    training["selected_round"] = controller["selected_round"]
    write_json(run_dir(config) / "controller-dev" / "round-metrics.local.json", controller)
    write_json(run_dir(config) / "training-summary.local.json", training)
    return training


def _checkpoint_dir(config: dict[str, Any], round_index: int) -> Path:
    name = "round_00_base" if round_index == 0 else f"round_{round_index:02d}"
    return run_dir(config) / "checkpoints" / name


def _save_round_checkpoint(
    config: dict[str, Any],
    model: Any,
    optimizer: Any,
    torch: Any,
    *,
    row: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir = _checkpoint_dir(config, int(row["round"]))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "model.local.nemo"
    optimizer_path = checkpoint_dir / "optimizer.local.pt"
    marker_path = checkpoint_dir / "checkpoint-complete.local.json"
    if marker_path.exists() and model_path.exists() and optimizer_path.exists():
        marker = read_json(marker_path)
        if file_sha256(model_path) == marker.get("checkpoint_sha256"):
            return marker
        raise RuntimeError(f"completed round {row['round']} checkpoint hash mismatch")
    model.save_to(str(model_path))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(0),
            "optimizer_step": int(row["optimizer_step"]),
            "exposures_seen": int(row["exposures_seen"]),
        },
        optimizer_path,
    )
    marker = {**row, "checkpoint_sha256": file_sha256(model_path), "optimizer_state_sha256": file_sha256(optimizer_path)}
    write_json(marker_path, marker)
    return marker


def _latest_complete_round(config: dict[str, Any]) -> int | None:
    complete: list[int] = []
    for round_index in range(0, 21):
        directory = _checkpoint_dir(config, round_index)
        marker = directory / "checkpoint-complete.local.json"
        model = directory / "model.local.nemo"
        optimizer = directory / "optimizer.local.pt"
        if marker.exists() and model.exists() and optimizer.exists():
            payload = read_json(marker)
            if file_sha256(model) != payload.get("checkpoint_sha256") or file_sha256(optimizer) != payload.get("optimizer_state_sha256"):
                raise RuntimeError(f"round {round_index} completed checkpoint identity mismatch")
            complete.append(round_index)
    return max(complete) if complete else None


def _training_state_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "training-state.local.json"


def _load_or_initialize_training(
    config: dict[str, Any],
    torch: Any,
    reporter: LiveProgressReporter,
) -> tuple[Any, Any, dict[str, Any], dict[str, str], int]:
    latest = _latest_complete_round(config)
    rates = config["training"]["learning_rates"]
    if latest is None:
        model = restore_base_model(config, reporter=reporter)
        surface = configure_surface04_trainable(model)
        set_surface04_training_mode(model)
        optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, rates), weight_decay=0.0)
        verify_optimizer_scope(optimizer, model, rates)
        initial = model_fingerprints(model)
        write_json(run_dir(config) / "initial-fingerprints.local.json", initial)
        state = {
            "optimizer_steps": 0,
            "exposures_seen": 0,
            "probe_curve": [],
            "round_rows": [],
            "controller_rows": [],
            "gradient_norms": [],
            "norm_curve": [],
            "wall_time_seconds": 0.0,
            "surface": surface.to_dict(),
        }
        return model, optimizer, state, initial, 0
    checkpoint_dir = _checkpoint_dir(config, latest)
    model = restore_local_checkpoint(checkpoint_dir / "model.local.nemo", reporter=reporter)
    surface = configure_surface04_trainable(model)
    set_surface04_training_mode(model)
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, rates), weight_decay=0.0)
    saved = torch.load(checkpoint_dir / "optimizer.local.pt", map_location="cuda:0", weights_only=False)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_rng_state"])
    torch.cuda.set_rng_state(saved["cuda_rng_state"], 0)
    verify_optimizer_scope(optimizer, model, rates)
    if not _training_state_path(config).exists():
        if latest != 0:
            raise RuntimeError("completed training checkpoint exists without resumable training state")
        marker = read_json(checkpoint_dir / "checkpoint-complete.local.json")
        state = {
            "optimizer_steps": 0,
            "exposures_seen": 0,
            "probe_curve": [
                {
                    "round": 0,
                    "anchor_probe_loss": marker["synthetic_anchor_probe_loss"],
                    "scale_probe_loss": marker["synthetic_scale_probe_loss"],
                }
            ],
            "round_rows": [marker],
            "controller_rows": [],
            "gradient_norms": [],
            "norm_curve": [],
            "wall_time_seconds": 0.0,
            "surface": surface.to_dict(),
        }
    else:
        state = read_json(_training_state_path(config))
    if int(state["optimizer_steps"]) != int(saved["optimizer_step"]):
        marker = read_json(checkpoint_dir / "checkpoint-complete.local.json")
        if int(saved["optimizer_step"]) < int(state["optimizer_steps"]):
            raise RuntimeError("resume optimizer state is older than training state")
        state["optimizer_steps"] = int(saved["optimizer_step"])
        state["exposures_seen"] = int(saved["exposures_seen"])
        state["round_rows"] = [row for row in state["round_rows"] if int(row["round"]) != latest]
        state["round_rows"].append(marker)
        if not any(int(row["round"]) == latest for row in state["probe_curve"]):
            state["probe_curve"].append(
                {
                    "round": latest,
                    "anchor_probe_loss": marker["synthetic_anchor_probe_loss"],
                    "scale_probe_loss": marker["synthetic_scale_probe_loss"],
                }
            )
        write_json(_training_state_path(config), state)
    initial = read_json(run_dir(config) / "initial-fingerprints.local.json")
    state["surface"] = surface.to_dict()
    return model, optimizer, state, initial, latest


def stage_train(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    inputs = verify_all_inputs(config)
    protected = protected_file_fingerprints(config)
    hardware = require_single_visible_rtx2080ti()
    runtime_id = verify_runtime_identities(check_gpu=False)
    torch = configure_torch()
    micro = read_json(run_dir(config) / "verification" / "microbatch.local.json")
    if micro.get("status") != "PASSED":
        raise RuntimeError("passing microbatch probe is required")
    physical = int(micro["selected"]["physical_microbatch"])
    accumulation = int(micro["selected"]["gradient_accumulation_steps"])
    rounds, meta_by_audio, schedule = load_scheduled_round_records(config)
    anchor_probe, scale_probe = probe_records(config)
    reporter = LiveProgressReporter(stage="train", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "train.local.ndjson")
    reporter.start("training fixed scale-2000 SURFACE_04")
    restore_reporter = LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "restore.local.ndjson")
    model, optimizer, state, initial_fingerprints, resume_round = _load_or_initialize_training(config, torch, restore_reporter)
    prompt = derive_prompt_column_selection(model, "sl-SI")
    if not state["probe_curve"]:
        initial_anchor = mean_probe_loss(model, prompt.prompt_index, anchor_probe, torch=torch, reporter=reporter, message="round_0_anchor_probe", interval=interval)
        initial_scale = mean_probe_loss(model, prompt.prompt_index, scale_probe, torch=torch, reporter=reporter, message="round_0_scale_probe", interval=interval)
        state["probe_curve"].append({"round": 0, "anchor_probe_loss": round(initial_anchor, 6), "scale_probe_loss": round(initial_scale, 6)})
        base_row = {
            "round": 0,
            "optimizer_step": 0,
            "exposures_seen": 0,
            "train_loss": None,
            "synthetic_anchor_probe_loss": round(initial_anchor, 6),
            "synthetic_scale_probe_loss": round(initial_scale, 6),
        }
        base_marker = _save_round_checkpoint(config, model, optimizer, torch, row=base_row)
        state["round_rows"].append(base_marker)
        controller = evaluate_controller_checkpoint(config, _checkpoint_dir(config, 0) / "model.local.nemo", 0, os.environ.get("SLAIF_VALIDATION_GPU", "1"))
        state["controller_rows"].append({**base_row, **controller, "checkpoint_sha256": base_marker["checkpoint_sha256"]})
        write_json(_training_state_path(config), state)

    optimizer_steps = int(state["optimizer_steps"])
    exposures_seen = int(state["exposures_seen"])
    grad_norms = list(state["gradient_norms"])
    validation_gpu = os.environ.get("SLAIF_VALIDATION_GPU", "1")
    if validation_gpu == hardware.physical_selector:
        raise RuntimeError("training and validation GPU selectors must differ")
    evaluated_rounds = {int(row["round"]) for row in state["controller_rows"]}
    if resume_round not in evaluated_rounds:
        marker = read_json(_checkpoint_dir(config, resume_round) / "checkpoint-complete.local.json")
        controller = evaluate_controller_checkpoint(
            config,
            _checkpoint_dir(config, resume_round) / "model.local.nemo",
            resume_round,
            validation_gpu,
        )
        state["controller_rows"].append({**marker, **controller})
        write_json(_training_state_path(config), state)
    monitor_path = run_dir(config) / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index=hardware.physical_selector, output_csv=monitor_path, interval_seconds=0.5)
    torch.cuda.reset_peak_memory_stats(0)
    segment_started = time.perf_counter()
    last_progress = segment_started
    segment_exposures = 0
    audio_seconds = 0.0
    padded_audio_seconds = 0.0
    voice_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    stopped_reason = "max_rounds"
    stopped_round = resume_round
    monitor.start()
    try:
        for round_index in range(resume_round + 1, int(config["training"]["max_rounds"]) + 1):
            set_surface04_training_mode(model)
            layout = deterministic_epoch_batches(rounds[round_index], batch_size=8, epoch=round_index, seed=int(config["training"]["seed"]), bucketed=True)
            assert_epoch_covers_once(layout, len(rounds[round_index]))
            round_losses: list[float] = []
            for batch_indices in layout.batches:
                batch_records = [rounds[round_index][index] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for start in range(0, 8, physical):
                    micro_records = batch_records[start : start + physical]
                    batch = make_training_batch(model, micro_records, device="cuda")
                    loss = rnnt_audio_loss(model, batch, prompt.prompt_index, frozen_encoder_no_grad=False)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite SURFACE_04 RNNT loss")
                    scale = len(micro_records) / 8.0
                    (loss * scale).backward()
                    step_loss += float(loss.detach().cpu()) * scale
                    del loss, batch
                _assert_gradient_scope(model)
                grad_norm, finite = finite_grad_norm(_trainable_parameters(model))
                if not finite:
                    raise RuntimeError("non-finite SURFACE_04 gradient")
                optimizer.step()
                optimizer_steps += 1
                exposures_seen += 8
                segment_exposures += 8
                round_losses.append(step_loss)
                grad_norms.append(grad_norm)
                audio_seconds += sum(row.duration for row in batch_records)
                padded_audio_seconds += max(row.duration for row in batch_records) * 8
                for record in batch_records:
                    meta = meta_by_audio[record.audio_filepath]
                    voice_counts[str(meta["voice"])] += 1
                    profile_counts[str(meta["profile_id"])] += 1
                now = time.perf_counter()
                if optimizer_steps % 500 == 0 or now - last_progress >= interval:
                    elapsed = now - segment_started
                    reporter.progress(
                        epoch=round_index,
                        total_epochs=20,
                        step=optimizer_steps,
                        total_steps=40000,
                        current_loss=round(step_loss, 6),
                        rolling_mean_loss=round(sum(round_losses[-25:]) / len(round_losses[-25:]), 6),
                        examples_per_second=round(segment_exposures / elapsed, 6),
                        audio_seconds_per_wall_second=round(audio_seconds / elapsed, 6),
                        cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    )
                    last_progress = now

            anchor_loss = mean_probe_loss(model, prompt.prompt_index, anchor_probe, torch=torch, reporter=reporter, message=f"round_{round_index}_anchor_probe", interval=interval)
            scale_loss = mean_probe_loss(model, prompt.prompt_index, scale_probe, torch=torch, reporter=reporter, message=f"round_{round_index}_scale_probe", interval=interval)
            state["probe_curve"].append({"round": round_index, "anchor_probe_loss": round(anchor_loss, 6), "scale_probe_loss": round(scale_loss, 6)})
            norm_row = {
                "round": round_index,
                "decoder_norm": round(sum(float(torch.linalg.vector_norm(p.detach()).cpu()) for n, p in model.named_parameters() if n.startswith("decoder.")), 6),
                "joint_norm": round(sum(float(torch.linalg.vector_norm(p.detach()).cpu()) for n, p in model.named_parameters() if n.startswith("joint.")), 6),
                "final_encoder_block_norm": round(sum(float(torch.linalg.vector_norm(p.detach()).cpu()) for n, p in model.named_parameters() if n.startswith("encoder.layers.23.")), 6),
            }
            state["norm_curve"].append(norm_row)
            row = {
                "round": round_index,
                "optimizer_step": optimizer_steps,
                "exposures_seen": exposures_seen,
                "train_loss": round(sum(round_losses) / len(round_losses), 6),
                "synthetic_anchor_probe_loss": round(anchor_loss, 6),
                "synthetic_scale_probe_loss": round(scale_loss, 6),
            }
            marker = _save_round_checkpoint(config, model, optimizer, torch, row=row)
            state["round_rows"].append(marker)
            state.update({"optimizer_steps": optimizer_steps, "exposures_seen": exposures_seen, "gradient_norms": grad_norms})
            write_json(_training_state_path(config), state)
            controller = evaluate_controller_checkpoint(config, _checkpoint_dir(config, round_index) / "model.local.nemo", round_index, validation_gpu)
            state["controller_rows"].append({**row, **controller, "checkpoint_sha256": marker["checkpoint_sha256"]})
            write_json(_training_state_path(config), state)
            controller_payload = _select_and_mark_controller_rows(state["controller_rows"])
            write_json(run_dir(config) / "controller-dev" / "round-metrics.local.json", controller_payload)
            stopped_round = round_index
            reporter.progress(epoch=round_index, total_epochs=20, step=optimizer_steps, total_steps=40000, message=f"ARTUR-dev WER={controller['wer']} CER={controller['cer']} empty={controller['empty']}")
            stop = should_stop_controller_curve(state["controller_rows"])
            if stop["stop"]:
                stopped_reason = stop["reason"]
                break
    except Exception as exc:
        reporter.failed("SURFACE_04 training failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()

    segment_wall = time.perf_counter() - segment_started
    state["wall_time_seconds"] = float(state.get("wall_time_seconds", 0.0)) + segment_wall
    state.update({"optimizer_steps": optimizer_steps, "exposures_seen": exposures_seen, "gradient_norms": grad_norms})
    write_json(_training_state_path(config), state)
    controller_payload = _select_and_mark_controller_rows(state["controller_rows"])
    selected_round = int(controller_payload["selected_round"])
    after = model_fingerprints(model)
    integrity = fingerprint_integrity(initial_fingerprints, after)
    if not integrity["only_surface04_changed"]:
        raise RuntimeError("parameter-integrity failure: unauthorized tensor changed")
    verify_protected_file_fingerprints(config, protected)
    parameter_table = []
    groups = {
        "decoder": [name for name in initial_fingerprints if name.startswith("decoder.")],
        "joint": [name for name in initial_fingerprints if name.startswith("joint.")],
        "final_encoder_block": [name for name in initial_fingerprints if name.startswith("encoder.layers.23.")],
        "lower_encoder_and_frontend": [
            name
            for name in initial_fingerprints
            if (name.startswith("encoder.") and not name.startswith("encoder.layers.23.")) or name.startswith("preprocessor.")
        ],
        "prompt_path": [name for name in initial_fingerprints if "prompt" in name.lower()],
    }
    for name, tensor_names in groups.items():
        before_fp = names_fingerprint(initial_fingerprints, tensor_names)
        after_fp = names_fingerprint(after, tensor_names)
        expected = "trainable" if name in {"decoder", "joint", "final_encoder_block"} else "frozen"
        parameter_table.append({"surface": name, "expected_status": expected, "changed": before_fp != after_fp, "before_fingerprint": before_fp, "after_fingerprint": after_fp, "notes": "bitwise aggregate fingerprint"})
    payload = {
        "status": "PASSED",
        "surface_id": SURFACE_ID,
        "surface": state["surface"],
        "semantic_rows": 16000,
        "sample_exposures": exposures_seen,
        "optimizer_steps": optimizer_steps,
        "stopped_round": stopped_round,
        "stopped_reason": stopped_reason,
        "selected_round": selected_round,
        "physical_microbatch": physical,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": 8,
        "learning_rates": config["training"]["learning_rates"],
        "schedule_sha256": schedule["schedule_sha256"],
        "probe_curve": state["probe_curve"],
        "controller_curve": controller_payload["rows"],
        "norm_curve": state["norm_curve"],
        "gradient_norm": {"min": min(grad_norms), "max": max(grad_norms), "final": grad_norms[-1]},
        "wall_time_seconds": state["wall_time_seconds"],
        "examples_per_second": exposures_seen / state["wall_time_seconds"],
        "audio_seconds_per_wall_second": audio_seconds / segment_wall if segment_wall else None,
        "padding_ratio": padded_audio_seconds / audio_seconds if audio_seconds else None,
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "parameter_integrity": integrity,
        "parameter_integrity_table": parameter_table,
        "runtime": runtime_summary(hardware, torch),
        "runtime_identities": runtime_id,
        "input_integrity": inputs,
        "protected_file_fingerprints": protected,
        "exposure_counts_by_voice_current_segment": dict(sorted(voice_counts.items())),
        "exposure_counts_by_profile_current_segment": dict(sorted(profile_counts.items())),
        "per_round_checkpoints_retained": True,
    }
    write_json(run_dir(config) / "training-summary.local.json", payload)
    reporter.complete("SURFACE_04 training and controller selection complete", step=optimizer_steps, total_steps=40000)
    print(json.dumps({"status": "PASSED", "stopped_round": stopped_round, "selected_round": selected_round, "optimizer_steps": optimizer_steps}, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def stage_evaluate_directional(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    hardware = require_single_visible_rtx2080ti()
    verify_all_inputs(config)
    verify_runtime_identities(check_gpu=False)
    training = _refresh_training_controller_selection(
        config,
        read_json(run_dir(config) / "training-summary.local.json"),
    )
    selected_round = int(training["selected_round"])
    checkpoint = _checkpoint_dir(config, selected_round) / "model.local.nemo"
    marker = read_json(_checkpoint_dir(config, selected_round) / "checkpoint-complete.local.json")
    if file_sha256(checkpoint) != marker["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint identity mismatch")
    fast_config = read_json(REPO_ROOT / FAST_DIRECTIONAL_CONFIG)
    suite_records, split_records = load_directional_suite(fast_config)
    output_dir = run_dir(config) / "directional-evaluation"
    suite_manifest_sha = write_privacy_safe_suite_manifest(output_dir / "suite-plan.local.jsonl", suite_records)
    env = runtime_environment()
    env.update({"CUDA_VISIBLE_DEVICES": hardware.physical_selector, "NVIDIA_TF32_OVERRIDE": "0", "PYTHONUNBUFFERED": "1"})
    arm = run_batched_arm(
        records=suite_records,
        batch_size=32,
        bucketed=True,
        run_dir=output_dir / ARM_NAME,
        python_executable=Path(sys.executable),
        nemo_script=nemo_streaming_script(),
        checkpoint=checkpoint,
        context=config["evaluation"]["att_context_size"],
        env=env,
        physical_gpu_index=hardware.physical_selector,
        monitor_interval_seconds=0.5,
    )
    if arm.get("status") != "PASSED":
        raise RuntimeError(f"directional evaluation failed: {arm.get('status')}")
    predictions = load_local_predictions(output_dir / ARM_NAME / "predictions.local.jsonl")
    by_split = split_predictions(suite_records, split_records, predictions)
    metric_table: dict[str, dict[str, Any]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    for split, records in split_records.items():
        metrics = metrics_for(records, by_split[split])
        split_summaries[split] = {
            "rows": len(records),
            "audio_duration_seconds": round(sum(row.duration for row in records), 6),
            "metrics": metrics,
        }
        metric_table[split] = metric_row(split_summaries[split])
    classification = classify_surface04(
        metric_table,
        parameter_integrity=bool(training["parameter_integrity"]["only_surface04_changed"]),
        selected_round=selected_round,
    )
    binding = bind_post_selection_metrics(selected_round, metric_table)
    if binding["selected_round"] != selected_round:
        raise RuntimeError("post-selection metrics changed selected round")
    payload = {
        "status": "PASSED",
        "selected_round": selected_round,
        "checkpoint_sha256": marker["checkpoint_sha256"],
        "suite_manifest_sha256": suite_manifest_sha,
        "policy": config["evaluation"],
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
            "selected_gpu": hardware.to_dict(),
            "sharded_evaluation": False,
        },
        "splits": split_summaries,
        "metric_table": metric_table,
        "classification": classification,
        "accepted_parent": "none",
    }
    write_json(output_dir / "summary.local.json", payload)
    print(json.dumps({"status": "PASSED", "selected_round": selected_round, "classification": classification, "metrics": metric_table}, sort_keys=True))
    return payload


def _load_pr42_metrics() -> dict[str, Any] | None:
    path = REPO_ROOT / "docs/experiments/0023-scale2000-decoder-joint-rnnt-s6tts-hardvoice20.json"
    if not path.exists():
        return None
    payload = read_json(path)
    return payload.get("directional_evaluation", {}).get("metrics", {}).get("s6_hard20_candidate")


def _public_suite_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    suite = evaluation["suite"]
    return {
        "rows": suite["rows"],
        "prediction_count": suite["prediction_count"],
        "audio_duration_seconds": suite["audio_duration_seconds"],
        "wall_time_seconds": suite["wall_time_seconds"],
        "real_time_factor": suite["real_time_factor"],
        "rows_per_second": suite["rows_per_second"],
        "audio_seconds_per_wall_second": suite["audio_seconds_per_wall_second"],
        "gpu_monitor": suite["gpu_monitor"],
        "layout": {
            "batch_size": suite["layout"]["batch_size"],
            "bucketed": suite["layout"]["bucketed"],
            "batch_count": suite["layout"]["batch_count"],
            "padding_ratio": suite["layout"]["padding_ratio"],
        },
        "sharded_evaluation": False,
    }


def _format_metric(value: dict[str, Any] | None) -> str:
    if value is None:
        return "NOT_AVAILABLE"
    return f"{value['wer']:.3f} / {value['cer']:.3f} / {int(value['empty'])}"


def _markdown_report(public: dict[str, Any]) -> str:
    training = public["training"]
    controller_rows = public["controller_dev"]["curve"]
    metrics = public["directional_evaluation"]["metrics"]
    selected_round = int(training["selected_round"])
    selected = next(row for row in controller_rows if int(row["round"]) == selected_round)
    surface = public["surface"]
    lines = [
        "# Experiment 0024: Fixed Scale-2000 Surface04 Last Encoder Block",
        "",
        f"Classification: `{public['classification']}`",
        "",
        "This diagnostic changed only the trainable model surface. It used the original scale-2000 augmented corpus and its fixed exposure schedule.",
        "",
        "## Result",
        "",
        f"- Surface: `{surface['surface_id']}` (`{surface['final_encoder_block']}`).",
        f"- Trainable parameters: {surface['trainable_parameter_count']:,} total; {surface['decoder_parameter_count']:,} decoder, {surface['joint_parameter_count']:,} joint, and {surface['final_encoder_block_parameter_count']:,} final encoder block.",
        f"- Training stopped after round {training['stopped_round']} ({training['optimizer_steps']:,} optimizer steps and {training['sample_exposures']:,} exposures): `{training['stopped_reason']}`.",
        f"- ARTUR controller-dev selected round {selected_round} at {selected['wer']:.3f} WER / {selected['cer']:.3f} CER / {int(selected['empty'])} empty hypotheses.",
        f"- Selected checkpoint SHA256: `{public['directional_evaluation']['selected_checkpoint_sha256']}`.",
        f"- Training hardware: {training['runtime']['gpu']}, FP32, TF32 disabled, one visible CUDA device; peak allocated/reserved VRAM {training['peak_allocated_mib']:.3f}/{training['peak_reserved_mib']:.3f} MiB.",
        "- Surface04 did not beat PR #36 cleanly, but it is a credible real-gate tradeoff candidate under the one-sided non-regression tolerance. Surface05 is justified as the next controlled diagnostic, subject to a separate work order and review.",
        "",
        "## ARTUR Controller-Dev Curve",
        "",
        "| Round | Step | Exposures | Train loss | Synthetic anchor | Synthetic scale | ARTUR-dev WER | CER | Empty | Delete | Insert | Substitute | Eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controller_rows:
        lines.append(
            "| {round} | {optimizer_step} | {exposures_seen} | {train_loss} | {synthetic_anchor_probe_loss} | {synthetic_scale_probe_loss} | {wer} | {cer} | {empty} | {delete} | {insert} | {substitute} | {eligible} |".format(**row)
        )
    lines.extend(
        [
            "",
            "`Train loss` is the mean RNNT loss over the completed 16,000-exposure round. `Synthetic anchor` is the fixed 32-row inherited probe; `Synthetic scale` is the fixed 320-row scale probe. ARTUR-dev columns are aggregate real controller-development metrics and alone select the checkpoint.",
            "",
            "## Post-Selection Directional Metrics",
            "",
            "| Split | Base | Scale-2000 joint-adapter | PR #36 round20 | PR #39 round6 | PR #42 S6TTS hardvoice | SURFACE_04 selected |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout", "fleurs_v2", "artur_j"):
        lines.append(
            f"| {split} | {_format_metric(metrics['base'][split])} | {_format_metric(metrics['scale2000_joint_adapter'][split])} | {_format_metric(metrics['pr36_round20'][split])} | {_format_metric(metrics['pr39_round6'][split])} | {_format_metric((metrics.get('pr42_s6tts_hardvoice') or {}).get(split))} | {_format_metric(metrics['surface04_selected'][split])} |"
        )
    lines.extend(
        [
            "",
            "Values are normalized WER / CER / empty-hypothesis count. Directional batch-32 gates were run only after ARTUR-dev fixed the selected round.",
            "",
            "## Parameter Integrity",
            "",
            "| Surface | Expected status | Changed | Before fingerprint | After fingerprint | Notes |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in training["parameter_integrity_table"]:
        lines.append(f"| {row['surface']} | {row['expected_status']} | {str(row['changed']).lower()} | `{row['before_fingerprint']}` | `{row['after_fingerprint']}` | {row['notes']} |")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- `accepted_parent` remains `none`.",
            "- The result is diagnostic, noncanonical, and promotion-ineligible.",
            "- No real speech was used for training; ARTUR controller-dev was aggregate run-control only.",
            "- No checkpoint, audio, prediction, raw reference/hypothesis, or local manifest is committed.",
            "- No `TRAINING_ELIGIBLE` status or model publication is issued.",
            "",
        ]
    )
    return "\n".join(lines)


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inputs = verify_all_inputs(config)
    micro = read_json(run_dir(config) / "verification" / "microbatch.local.json")
    training = read_json(run_dir(config) / "training-summary.local.json")
    evaluation = read_json(run_dir(config) / "directional-evaluation" / "summary.local.json")
    evaluation["classification"] = classify_surface04(
        evaluation["metric_table"],
        parameter_integrity=bool(training["parameter_integrity"]["only_surface04_changed"]),
        selected_round=int(training["selected_round"]),
    )
    write_json(run_dir(config) / "directional-evaluation" / "summary.local.json", evaluation)
    pr42 = _load_pr42_metrics()
    public = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "work_order_id": "0037",
        "status": "DIAGNOSTIC_ONLY",
        "classification": evaluation["classification"],
        "repository_commit": git_head(),
        "accepted_parent": "none",
        "promotion_eligible": False,
        "training_eligible": False,
        "checkpoint_accepted": False,
        "model_published": False,
        "governance": {
            "adr_0009": "docs/adr/0009-fixed-scale2000-surface-sweep.md",
            "adr_0008_controller_dev_used": True,
            "immutable_gates_used_for_selection": False,
        },
        "input_integrity": inputs,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
        },
        "data": {
            "corpus_id": config["data"]["corpus_id"],
            "semantic_rows": 16000,
            "exposure_records": 320000,
            "fixed_text_sha256": config["data"]["fixed_text_sha256"],
            "all_views_sha256": config["data"]["all_views_sha256"],
            "exposure_schedule_sha256": config["data"]["exposure_schedule_sha256"],
            "s6tts_used": False,
            "scale8000_used": False,
            "real_speech_used_for_training": False,
        },
        "surface": training["surface"],
        "training": {
            key: training[key]
            for key in (
                "status",
                "semantic_rows",
                "sample_exposures",
                "optimizer_steps",
                "stopped_round",
                "stopped_reason",
                "selected_round",
                "physical_microbatch",
                "gradient_accumulation_steps",
                "effective_batch_size",
                "learning_rates",
                "schedule_sha256",
                "probe_curve",
                "norm_curve",
                "gradient_norm",
                "wall_time_seconds",
                "examples_per_second",
                "audio_seconds_per_wall_second",
                "padding_ratio",
                "gpu_monitor",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "parameter_integrity",
                "parameter_integrity_table",
                "runtime",
                "per_round_checkpoints_retained",
            )
        },
        "microbatch_probe": micro,
        "controller_dev": {
            "partition_id": "artur-controller-dev-v1",
            "policy": config["controller_dev"],
            "curve": training["controller_curve"],
            "selected_round": training["selected_round"],
            "immutable_gate_metrics_could_change_selection": False,
        },
        "directional_evaluation": {
            "policy": evaluation["policy"],
            "suite": _public_suite_summary(evaluation),
            "metrics": {
                "base": BASE_DIRECTIONAL_METRICS,
                "scale2000_joint_adapter": SCALE2000_JOINT_ADAPTER_METRICS,
                "pr36_round20": PR36_METRICS,
                "pr39_round6": PR39_METRICS,
                "pr42_s6tts_hardvoice": pr42,
                "surface04_selected": evaluation["metric_table"],
            },
            "selected_round": training["selected_round"],
            "selected_checkpoint_sha256": evaluation["checkpoint_sha256"],
            "classification": evaluation["classification"],
        },
        "limitations": [
            "Synthetic-only training remains diagnostic.",
            "ARTUR controller-dev is spent development data, not immutable acceptance evidence.",
            "Directional batch-32 evaluation is noncanonical and promotion-ineligible.",
            "Other-language behavior was not evaluated.",
            "Surface05 is justified for a separate controlled work order and review; this report does not authorize its execution.",
        ],
        "safety": {
            "real_data_used_for_training": False,
            "s6tts_used": False,
            "scale8000_used": False,
            "immutable_gate_used_for_early_stopping": False,
            "raw_references_or_hypotheses_committed": False,
            "checkpoint_or_model_committed": False,
            "audio_or_predictions_committed": False,
            "local_manifest_committed": False,
            "training_eligible_issued": False,
        },
    }
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v4-fixed-scale2000-surface04-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "work_order_id": "0037",
        "corpus_id": config["data"]["corpus_id"],
        "fixed_text_sha256": config["data"]["fixed_text_sha256"],
        "all_views_sha256": config["data"]["all_views_sha256"],
        "exposure_schedule_sha256": config["data"]["exposure_schedule_sha256"],
        "surface_id": SURFACE_ID,
        "selected_round": training["selected_round"],
        "selected_checkpoint_sha256": evaluation["checkpoint_sha256"],
        "classification": evaluation["classification"],
        "parameter_integrity_passed": training["parameter_integrity"]["only_surface04_changed"],
        "accepted_parent": "none",
        "promotion_eligible": False,
        "training_eligible": False,
        "checkpoint_accepted": False,
        "model_published": False,
        "local_checkpoint_committed": False,
        "generated_audio_committed": False,
        "predictions_committed": False,
        "prohibited_statuses": ["TRAINING_ELIGIBLE"],
    }
    assert_public_report_safe(public)
    assert_public_report_safe(certificate)
    for path, payload in ((CERTIFICATE_PATH, certificate), (REPORT_JSON, public)):
        absolute = REPO_ROOT / path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(absolute, payload)
    atomic_write_text(REPO_ROOT / REPORT_MD, _markdown_report(public))
    print(json.dumps({"status": "PASSED", "classification": public["classification"], "selected_round": training["selected_round"]}, sort_keys=True))
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Work Order 0037 Surface04 diagnostic")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("verify-inputs", "probe-hardware", "probe-surface", "probe-microbatch", "train", "evaluate-directional", "summarize"),
    )
    parser.add_argument("--progress-interval", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    stages = {
        "verify-inputs": lambda: stage_verify_inputs(config_path),
        "probe-hardware": lambda: stage_probe_hardware(config_path),
        "probe-surface": lambda: stage_probe_surface(config_path),
        "probe-microbatch": lambda: stage_probe_microbatch(config_path, args.progress_interval),
        "train": lambda: stage_train(config_path, args.progress_interval),
        "evaluate-directional": lambda: stage_evaluate_directional(config_path),
        "summarize": lambda: stage_summarize(config_path),
    }
    stages[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
