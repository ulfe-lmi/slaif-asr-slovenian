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
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from slaif_asr.batched_streaming import NvidiaSmiMonitor, file_sha256, load_local_predictions, metrics_for, parse_monitor_csv, run_batched_arm
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_scoring import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION, NEMO_REVISION, checkpoint_path, nemo_streaming_script, runtime_environment, verify_runtime_identities
from slaif_asr.corpus_v2_training import assert_epoch_covers_once, deterministic_epoch_batches, make_training_batch
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.directional_evaluation import load_directional_suite, split_predictions, write_privacy_safe_suite_manifest
from slaif_asr.emission_rnnt_finetune import (
    ARM_NAME,
    BASE_DIRECTIONAL_METRICS,
    SCALE2000_JOINT_ADAPTER_METRICS,
    assert_public_report_safe,
    changed_tensor_summary,
    classify_decoder_joint_rnnt,
    configure_decoder_joint_trainable,
    finite_grad_norm,
    git_head,
    has_forbidden_text_only_modules,
    load_config,
    load_scheduled_round_records,
    local_path,
    metric_row,
    microbatch_plan,
    optimizer_scope_summary,
    probe_records,
    protected_file_fingerprints,
    read_json,
    read_jsonl,
    rnnt_audio_loss,
    run_dir,
    trainable_parameters,
    validate_microbatch_selection,
    verify_all_inputs,
    verify_optimizer_scope,
    verify_protected_file_fingerprints,
    write_json,
)
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread
from slaif_asr.prompt_column import derive_prompt_column_selection
from slaif_asr.rtx2080ti_policy import nvidia_smi_inventory, require_single_visible_rtx2080ti
from slaif_asr.slovenian_joint_adapter import state_dict_cpu


DEFAULT_CONFIG = Path("configs/experiments/scale2000_decoder_joint_rnnt_v1.json")
REPORT_JSON = Path("docs/experiments/0017-scale2000-decoder-joint-rnnt-directional.json")
REPORT_MD = Path("docs/experiments/0017-scale2000-decoder-joint-rnnt-directional.md")
CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v4-decoder-joint-rnnt-diagnostic-v1.json")
FAST_DIRECTIONAL_CONFIG = Path("configs/experiments/fast_batched_directional_replay_v1.json")

_JOINT_PATH = Path(__file__).with_name("run_corpus_v2_joint_adapter_diagnostic.py")
_JOINT_SPEC = importlib.util.spec_from_file_location("_slaif_joint_runner_decoder_joint", _JOINT_PATH)
if _JOINT_SPEC is None or _JOINT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import model restore helper")
_JOINT = importlib.util.module_from_spec(_JOINT_SPEC)
_JOINT_SPEC.loader.exec_module(_JOINT)


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
    if cuda_bin not in path_entries:
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    if nvvm_lib not in ld_entries:
        os.environ["LD_LIBRARY_PATH"] = nvvm_lib + (os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")


def configure_torch() -> Any:
    ensure_cuda_nvcc_process_env()
    torch = _JOINT.configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_base_model(config: dict[str, Any], *, reporter: LiveProgressReporter | None = None) -> Any:
    return _JOINT.restore_base_model(config, reporter=reporter)


def write_public_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def runtime_summary(hardware: Any, torch: Any) -> dict[str, Any]:
    return {
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
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
    payload = {"status": "PASSED", "work_order_id": "0030", **verify_all_inputs(config)}
    write_json(run_dir(config) / "verification" / "inputs.local.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_probe_hardware(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inventory = [row.to_dict() for row in nvidia_smi_inventory()]
    rtx_count = sum(1 for row in inventory if "RTX 2080 Ti" in row["name"])
    payload: dict[str, Any] = {
        "status": "PASSED" if rtx_count >= 1 else "ENVIRONMENT_BLOCKED",
        "nvidia_smi_inventory": inventory,
        "physical_gpu_count": len(inventory),
        "rtx2080ti_count": rtx_count,
        "second_2080ti_detected": rtx_count >= 2,
    }
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        try:
            payload["selected_visible_device"] = require_single_visible_rtx2080ti().to_dict()
        except Exception as exc:
            payload["selected_visible_device_error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"
    write_json(run_dir(config) / "verification" / "hardware.local.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if payload["status"] != "PASSED":
        raise RuntimeError("no RTX 2080 Ti available")
    return payload


def _representative_records(config: dict[str, Any], count: int = 8):
    rounds, _meta, _summary = load_scheduled_round_records(config)
    return sorted(rounds[1], key=lambda record: (-record.duration, record.selected_training_id))[:count]


def _zero_grad(model: Any) -> None:
    for parameter in trainable_parameters(model):
        parameter.grad = None


def _accumulated_probe(model: Any, prompt_index: int, records: Sequence[Any], *, physical_microbatch: int, torch: Any) -> dict[str, Any]:
    _zero_grad(model)
    weighted_loss = 0.0
    for start in range(0, len(records), physical_microbatch):
        micro = records[start : start + physical_microbatch]
        batch = make_training_batch(model, micro, device="cuda")
        loss = rnnt_audio_loss(model, batch, prompt_index, frozen_encoder_no_grad=True)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite accumulation probe loss")
        scale = len(micro) / len(records)
        (loss * scale).backward()
        weighted_loss += float(loss.detach().cpu()) * scale
        del loss
        del batch
    grad_norm, finite = finite_grad_norm(trainable_parameters(model))
    if not finite:
        raise RuntimeError("non-finite accumulation probe gradient")
    return {"weighted_loss": round(weighted_loss, 6), "gradient_norm": round(grad_norm, 6)}


def stage_probe_microbatch(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    hardware = require_single_visible_rtx2080ti()
    torch = configure_torch()
    verify_runtime_identities(check_gpu=False)
    records = _representative_records(config, 8)
    reporter = LiveProgressReporter(stage="probe_microbatch", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "probe-microbatch.local.ndjson")
    reporter.start("probing decoder+joint RNNT microbatch")
    model = restore_base_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "probe-restore.local.ndjson"))
    model.train()
    surface = configure_decoder_joint_trainable(model)
    if has_forbidden_text_only_modules(model):
        raise RuntimeError("text-only LM module is present")
    prompt = derive_prompt_column_selection(model, "sl-SI")
    outcomes: dict[int, dict[str, Any]] = {}
    for candidate in config["training"]["physical_microbatch_candidates"]:
        reporter.progress(step=len(outcomes), total_steps=4, message=f"candidate_microbatch_{candidate}")
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            _zero_grad(model)
            batch = make_training_batch(model, records[:candidate], device="cuda")
            loss = rnnt_audio_loss(model, batch, prompt.prompt_index, frozen_encoder_no_grad=True)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            loss.backward()
            grad_norm, finite = finite_grad_norm(trainable_parameters(model))
            if not finite:
                raise RuntimeError("non-finite probe gradient")
            for name, parameter in model.named_parameters():
                if not (name.startswith("decoder.") or name.startswith("joint.")) and parameter.grad is not None:
                    raise RuntimeError(f"frozen parameter received gradient: {name}")
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            free_mib = int(free_bytes // 1024 // 1024)
            if free_mib < int(config["training"]["minimum_free_vram_mib_after_warmup"]):
                raise RuntimeError("insufficient free VRAM after warmup")
            outcomes[int(candidate)] = {
                "status": "PASSED",
                "loss": round(float(loss.detach().cpu()), 6),
                "gradient_norm": round(grad_norm, 6),
                "free_vram_mib_after_warmup": free_mib,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
            }
            del loss
            del batch
        except Exception as exc:
            outcomes[int(candidate)] = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc).splitlines()[0][:240]}
            torch.cuda.empty_cache()
        finally:
            _zero_grad(model)
    selected = validate_microbatch_selection(config["training"]["physical_microbatch_candidates"], outcomes)
    correctness = None
    if selected["status"] == "PASSED":
        physical = int(selected["physical_microbatch"])
        seed_torch(torch, int(config["training"]["seed"]) + 3000)
        correctness = _accumulated_probe(model, prompt.prompt_index, records, physical_microbatch=physical, torch=torch)
        seed_torch(torch, int(config["training"]["seed"]) + 3000)
        singleton = _accumulated_probe(model, prompt.prompt_index, records, physical_microbatch=1, torch=torch)
        rel = abs(correctness["weighted_loss"] - singleton["weighted_loss"]) / singleton["weighted_loss"] if singleton["weighted_loss"] else 0.0
        correctness.update({"singleton_weighted_loss": singleton["weighted_loss"], "relative_loss_difference_vs_singletons": round(rel, 8), "passed": rel <= 0.005})
    payload = {
        "status": selected["status"],
        "hardware": hardware.to_dict(),
        "surface": surface.to_dict(),
        "candidate_outcomes": {str(key): value for key, value in outcomes.items()},
        "selected": selected,
        "correctness": correctness,
    }
    write_json(run_dir(config) / "verification" / "microbatch.local.json", payload)
    if selected["status"] != "PASSED":
        raise RuntimeError("physical microbatch 1 failed; environment blocked")
    if correctness is None or not correctness["passed"]:
        raise RuntimeError("gradient accumulation correctness probe failed")
    reporter.complete("microbatch probe complete")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def mean_loss(
    model: Any,
    prompt_index: int,
    records: Sequence[Any],
    *,
    torch: Any,
    reporter: LiveProgressReporter | None = None,
    message: str = "loss_probe",
    interval_seconds: float = 10.0,
) -> float:
    losses = []
    ordered = sorted(records, key=lambda record: (-record.duration, record.selected_training_id))
    started = time.perf_counter()
    last_emit = started
    with torch.no_grad():
        for index, record in enumerate(ordered, start=1):
            batch = make_training_batch(model, [record], device="cuda")
            loss = rnnt_audio_loss(model, batch, prompt_index, frozen_encoder_no_grad=True)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            losses.append(float(loss.detach().cpu()))
            del loss
            del batch
            if index % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            now = time.perf_counter()
            if reporter is not None and (now - last_emit >= interval_seconds or index == len(ordered)):
                elapsed = now - started
                reporter.progress(
                    step=index,
                    total_steps=len(ordered),
                    examples_per_second=round(index / elapsed, 6) if elapsed else None,
                    cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                    cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    message=message,
                )
                last_emit = now
    return sum(losses) / len(losses)


def stage_train(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    verify_all_inputs(config)
    hardware = require_single_visible_rtx2080ti()
    runtime_id = verify_runtime_identities(check_gpu=False)
    torch = configure_torch()
    micro_path = run_dir(config) / "verification" / "microbatch.local.json"
    if not micro_path.exists():
        raise RuntimeError("microbatch probe must run before training")
    selected = read_json(micro_path)["selected"]
    physical_microbatch = int(selected["physical_microbatch"])
    accumulation = int(selected["gradient_accumulation_steps"])
    plan = microbatch_plan(physical_microbatch)
    rounds, meta_by_audio, schedule_summary = load_scheduled_round_records(config)
    anchor_probe, scale_probe = probe_records(config)
    reporter = LiveProgressReporter(stage="train", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "train.local.ndjson")
    reporter.start("training scale-2000 decoder+joint RNNT")
    model = restore_base_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "restore.local.ndjson"))
    model.train()
    surface = configure_decoder_joint_trainable(model)
    if has_forbidden_text_only_modules(model):
        raise RuntimeError("text-only LM module is present")
    initial_state = state_dict_cpu(model)
    prompt = derive_prompt_column_selection(model, "sl-SI")
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=float(config["training"]["learning_rate"]), weight_decay=0.0)
    verify_optimizer_scope(optimizer, model)
    initial_anchor = mean_loss(model, prompt.prompt_index, anchor_probe, torch=torch, reporter=reporter, message="initial_anchor_probe", interval_seconds=interval)
    initial_scale = mean_loss(model, prompt.prompt_index, scale_probe, torch=torch, reporter=reporter, message="initial_scale_probe", interval_seconds=interval)
    probe_curve = [{"round": 0, "anchor_probe_loss": round(initial_anchor, 6), "scale_probe_loss": round(initial_scale, 6)}]
    decoder_joint_norm_curve: list[dict[str, Any]] = []
    grad_norms: list[float] = []
    optimizer_steps = 0
    sample_exposures = 0
    audio_seconds = 0.0
    padded_audio_seconds = 0.0
    rolling_losses: list[float] = []
    voice_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    view_type_counts: Counter[str] = Counter()
    arm_dir = run_dir(config) / ARM_NAME
    monitor_path = arm_dir / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index=hardware.physical_selector, output_csv=monitor_path, interval_seconds=0.5)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    last_progress = started
    monitor.start()
    try:
        total_steps = int(config["training"]["optimizer_steps"])
        for round_index in range(1, 21):
            records = rounds[round_index]
            layout = deterministic_epoch_batches(records, batch_size=8, epoch=round_index, seed=int(config["training"]["seed"]), bucketed=True)
            assert_epoch_covers_once(layout, len(records))
            for batch_indices in layout.batches:
                batch_records = [records[index] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for start_index in range(0, len(batch_records), physical_microbatch):
                    micro_records = batch_records[start_index : start_index + physical_microbatch]
                    batch = make_training_batch(model, micro_records, device="cuda")
                    loss = rnnt_audio_loss(model, batch, prompt.prompt_index, frozen_encoder_no_grad=True)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite decoder+joint RNNT training loss")
                    scale = len(micro_records) / 8.0
                    (loss * scale).backward()
                    step_loss += float(loss.detach().cpu()) * scale
                    del loss
                    del batch
                    now = time.perf_counter()
                    if now - last_progress >= interval:
                        elapsed = now - started
                        reporter.progress(
                            epoch=round_index,
                            total_epochs=20,
                            step=optimizer_steps,
                            total_steps=total_steps,
                            current_loss=round(step_loss, 6) if step_loss else None,
                            rolling_mean_loss=round(sum(rolling_losses) / len(rolling_losses), 6) if rolling_losses else None,
                            examples_per_second=round(sample_exposures / elapsed, 6) if elapsed else None,
                            audio_seconds_per_wall_second=round(audio_seconds / elapsed, 6) if elapsed else None,
                            cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                            cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                            message="training microbatch in progress",
                        )
                        last_progress = now
                grad_norm, finite = finite_grad_norm(trainable_parameters(model))
                if not finite:
                    raise RuntimeError("non-finite decoder+joint RNNT training gradient")
                for name, parameter in model.named_parameters():
                    if not (name.startswith("decoder.") or name.startswith("joint.")) and parameter.grad is not None:
                        raise RuntimeError(f"frozen parameter received gradient: {name}")
                optimizer.step()
                optimizer_steps += 1
                sample_exposures += len(batch_records)
                audio_seconds += sum(row.duration for row in batch_records)
                padded_audio_seconds += max(row.duration for row in batch_records) * len(batch_records)
                for row in batch_records:
                    meta = meta_by_audio[row.audio_filepath]
                    voice_counts[str(meta["voice"])] += 1
                    profile_counts[str(meta["profile_id"])] += 1
                    view_type_counts[str(meta["view_type"])] += 1
                rolling_losses.append(step_loss)
                rolling_losses = rolling_losses[-25:]
                grad_norms.append(grad_norm)
                now = time.perf_counter()
                if optimizer_steps % 500 == 0 or now - last_progress >= interval:
                    elapsed = time.perf_counter() - started
                    reporter.progress(
                        epoch=round_index,
                        total_epochs=20,
                        step=optimizer_steps,
                        total_steps=total_steps,
                        current_loss=round(step_loss, 6),
                        rolling_mean_loss=round(sum(rolling_losses) / len(rolling_losses), 6),
                        examples_per_second=round(sample_exposures / elapsed, 6) if elapsed else None,
                        audio_seconds_per_wall_second=round(audio_seconds / elapsed, 6) if elapsed else None,
                        cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    )
                    last_progress = now
            anchor_loss = mean_loss(model, prompt.prompt_index, anchor_probe, torch=torch, reporter=reporter, message=f"round_{round_index}_anchor_probe", interval_seconds=interval)
            scale_loss = mean_loss(model, prompt.prompt_index, scale_probe, torch=torch, reporter=reporter, message=f"round_{round_index}_scale_probe", interval_seconds=interval)
            probe_curve.append({"round": round_index, "anchor_probe_loss": round(anchor_loss, 6), "scale_probe_loss": round(scale_loss, 6)})
            decoder_norm = sum(float(torch.linalg.vector_norm(parameter.detach()).cpu()) for name, parameter in model.named_parameters() if name.startswith("decoder."))
            joint_norm = sum(float(torch.linalg.vector_norm(parameter.detach()).cpu()) for name, parameter in model.named_parameters() if name.startswith("joint."))
            decoder_joint_norm_curve.append({"round": round_index, "decoder_norm": round(decoder_norm, 6), "joint_norm": round(joint_norm, 6)})
    except Exception as exc:
        reporter.failed(message="training failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()
    if optimizer_steps != int(config["training"]["optimizer_steps"]):
        raise RuntimeError(f"optimizer step count mismatch: {optimizer_steps}")
    with heartbeat_thread(reporter, interval_seconds=interval, message="post-training integrity checks"):
        wall = time.perf_counter() - started
        final_anchor = mean_loss(model, prompt.prompt_index, anchor_probe, torch=torch, reporter=reporter, message="final_anchor_probe", interval_seconds=interval)
        final_scale = mean_loss(model, prompt.prompt_index, scale_probe, torch=torch, reporter=reporter, message="final_scale_probe", interval_seconds=interval)
        trained_state = state_dict_cpu(model)
        integrity = changed_tensor_summary(initial_state, trained_state)
        if not integrity["only_decoder_joint_changed"]:
            raise RuntimeError("unexpected tensor changed during decoder+joint RNNT training")
        checkpoint_out = arm_dir / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        model.save_to(str(checkpoint_out))
    payload = {
        "arm": ARM_NAME,
        "status": "PASSED",
        "semantic_rows": int(config["training"]["semantic_rows"]),
        "sample_exposures": sample_exposures,
        "effective_batch_size": 8,
        "physical_microbatch": physical_microbatch,
        "gradient_accumulation_steps": accumulation,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(config["training"]["learning_rate"]),
        "schedule_sha256": schedule_summary["schedule_sha256"],
        "initial_anchor_probe_loss": round(initial_anchor, 6),
        "final_anchor_probe_loss": round(final_anchor, 6),
        "initial_scale_probe_loss": round(initial_scale, 6),
        "final_scale_probe_loss": round(final_scale, 6),
        "probe_curve": probe_curve,
        "gradient_norm": {"min": round(min(grad_norms), 6), "max": round(max(grad_norms), 6), "final": round(grad_norms[-1], 6)},
        "decoder_joint_norm_curve": decoder_joint_norm_curve,
        "wall_time_seconds": round(wall, 6),
        "examples_per_second": round(sample_exposures / wall, 6) if wall else None,
        "audio_seconds_per_wall_second": round(audio_seconds / wall, 6) if wall else None,
        "padding_ratio": round(padded_audio_seconds / audio_seconds, 6) if audio_seconds else None,
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "trainable_surface": surface.to_dict(),
        "optimizer_scope": optimizer_scope_summary(model),
        "parameter_integrity": integrity,
        "checkpoint_sha256": file_sha256(checkpoint_out),
        "exposure_counts_by_voice": dict(sorted(voice_counts.items())),
        "exposure_counts_by_profile": dict(sorted(profile_counts.items())),
        "exposure_counts_by_view_type": dict(sorted(view_type_counts.items())),
        "microbatch_plan": plan,
        "runtime": runtime_summary(hardware, torch),
        "runtime_identities": runtime_id,
    }
    write_json(arm_dir / "training-summary.local.json", payload)
    reporter.complete("training complete", step=optimizer_steps, total_steps=int(config["training"]["optimizer_steps"]))
    print(json.dumps({"status": "PASSED", "arm": ARM_NAME, "optimizer_steps": optimizer_steps}, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def stage_evaluate_directional(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    hardware = require_single_visible_rtx2080ti()
    verify_all_inputs(config)
    verify_runtime_identities(check_gpu=False)
    checkpoint = run_dir(config) / ARM_NAME / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
    if not checkpoint.exists():
        raise RuntimeError("decoder+joint RNNT checkpoint is missing")
    fast_config = read_json(REPO_ROOT / FAST_DIRECTIONAL_CONFIG)
    suite_records, split_records = load_directional_suite(fast_config)
    output_dir = run_dir(config) / "directional-evaluation"
    suite_manifest_sha = write_privacy_safe_suite_manifest(output_dir / "suite-plan.local.jsonl", suite_records)
    env = runtime_environment()
    env["CUDA_VISIBLE_DEVICES"] = hardware.physical_selector
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
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
    split_predictions_map = split_predictions(suite_records, split_records, predictions)
    split_summaries = {}
    metric_table = {}
    for split, records in split_records.items():
        metrics = metrics_for(records, split_predictions_map[split])
        split_summaries[split] = {"rows": len(records), "audio_duration_seconds": round(sum(row.duration for row in records), 6), "metrics": metrics}
        metric_table[split] = metric_row(split_summaries[split])
    decision = classify_decoder_joint_rnnt(metric_table)
    payload = {
        "status": "PASSED",
        "suite_rows": len(suite_records),
        "suite_manifest_sha256": suite_manifest_sha,
        "checkpoint_sha256": file_sha256(checkpoint),
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
        "decision": decision,
    }
    write_json(output_dir / "summary.local.json", payload)
    print(json.dumps({"status": "PASSED", "classification": decision["classification"], "metric_table": metric_table}, ensure_ascii=False, sort_keys=True))
    return payload


def public_suite_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
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
        "sharded_evaluation": suite["sharded_evaluation"],
    }


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inputs = verify_all_inputs(config)
    micro = read_json(run_dir(config) / "verification" / "microbatch.local.json")
    training = read_json(run_dir(config) / ARM_NAME / "training-summary.local.json")
    evaluation = read_json(run_dir(config) / "directional-evaluation" / "summary.local.json")
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v4-decoder-joint-rnnt-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "work_order_id": "0030",
        "corpus_id": config["data"]["corpus_id"],
        "fixed_text_sha256": config["data"]["fixed_text_sha256"],
        "all_views_sha256": config["data"]["all_views_sha256"],
        "exposure_schedule_sha256": config["data"]["exposure_schedule_sha256"],
        "training_arm": ARM_NAME,
        "selected_physical_microbatch": training["physical_microbatch"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "effective_batch_size": 8,
        "optimizer_steps": training["optimizer_steps"],
        "trainable_parameter_count": training["trainable_surface"]["trainable_parameter_count"],
        "decoder_parameter_count": training["trainable_surface"]["decoder_parameter_count"],
        "joint_parameter_count": training["trainable_surface"]["joint_parameter_count"],
        "encoder_and_prompt_unchanged": training["parameter_integrity"]["encoder_unchanged"] and training["parameter_integrity"]["prompt_kernel_unchanged"],
        "directional_batch_size": 32,
        "classification": evaluation["decision"]["classification"],
        "accepted_parent": "none",
        "prohibited_actions": ["TRAINING_ELIGIBLE", "model publication", "checkpoint acceptance", "text-only objective", "temporary LM head"],
    }
    public = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "work_order_id": "0030",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_head(),
        "accepted_parent": "none",
        "canonical": False,
        "promotion_eligible": False,
        "authorization": {
            "status": "DIAGNOSTIC_ONLY",
            "certificate": str(CERTIFICATE_PATH),
            "certificate_sha256": "",
        },
        "input_integrity": inputs,
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256, "nemo_revision": NEMO_REVISION},
        "trainable_surface": {
            "type": "decoder+joint RNNT audio-loss fine-tuning",
            "decoder_parameter_count": training["trainable_surface"]["decoder_parameter_count"],
            "joint_parameter_count": training["trainable_surface"]["joint_parameter_count"],
            "trainable_parameter_count": training["trainable_surface"]["trainable_parameter_count"],
            "frozen_parameter_count": training["trainable_surface"]["frozen_parameter_count"],
            "no_text_only_path": True,
            "no_temporary_lm_head": True,
            "no_adapter_installed": True,
        },
        "training": {
            key: training[key]
            for key in (
                "status",
                "semantic_rows",
                "sample_exposures",
                "effective_batch_size",
                "physical_microbatch",
                "gradient_accumulation_steps",
                "optimizer_steps",
                "learning_rate",
                "schedule_sha256",
                "initial_anchor_probe_loss",
                "final_anchor_probe_loss",
                "initial_scale_probe_loss",
                "final_scale_probe_loss",
                "probe_curve",
                "gradient_norm",
                "decoder_joint_norm_curve",
                "wall_time_seconds",
                "examples_per_second",
                "audio_seconds_per_wall_second",
                "padding_ratio",
                "gpu_monitor",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "optimizer_scope",
                "parameter_integrity",
                "exposure_counts_by_voice",
                "exposure_counts_by_profile",
                "exposure_counts_by_view_type",
                "runtime",
            )
        },
        "microbatch_probe": micro,
        "directional_evaluation": {
            "policy": evaluation["policy"],
            "suite": public_suite_summary(evaluation),
            "metrics": {
                "base": BASE_DIRECTIONAL_METRICS,
                "scale2000_joint_adapter": SCALE2000_JOINT_ADAPTER_METRICS,
                "decoder_joint_rnnt": evaluation["metric_table"],
            },
            "decision": evaluation["decision"],
        },
        "limitations": [
            "Synthetic-only training remains diagnostic.",
            "Decoder and joint base parameters changed; other-language behavior is intentionally not protected by this work order.",
            "Directional batch-32 metrics are not canonical acceptance evidence.",
            "No batch-1 canonical evaluation was run.",
            "No checkpoint is accepted as a parent.",
        ],
        "safety": {
            "training_eligible_issued": False,
            "accepted_parent": "none",
            "generated_audio_committed": False,
            "model_or_checkpoint_committed": False,
            "raw_predictions_committed": False,
            "text_only_path_invoked": False,
        },
    }
    assert_public_report_safe(certificate)
    write_public_json(CERTIFICATE_PATH, certificate)
    public["authorization"]["certificate_sha256"] = file_sha256(CERTIFICATE_PATH)
    assert_public_report_safe(public)
    write_public_json(REPORT_JSON, public)
    lines = [
        "# Experiment 0017: Scale-2000 Decoder+Joint RNNT Directional Diagnostic",
        "",
        f"Classification: `{evaluation['decision']['classification']}`",
        "",
        "This is synthetic-only, directional batch-32 evidence. The acoustic encoder and Slovenian prompt pathway were frozen; decoder and joint base parameters were intentionally trainable. No checkpoint is accepted and `accepted_parent` remains `none`.",
        "",
        "## Data",
        "",
        f"- Corpus: `{config['data']['corpus_id']}`",
        f"- Fixed text SHA256: `{config['data']['fixed_text_sha256']}`",
        f"- All views SHA256: `{config['data']['all_views_sha256']}`",
        f"- Exposure schedule SHA256: `{config['data']['exposure_schedule_sha256']}`",
        f"- Exposures: {training['sample_exposures']}",
        "",
        "## Training",
        "",
        f"- Arm: `{ARM_NAME}`",
        f"- Physical microbatch: {training['physical_microbatch']}",
        f"- Gradient accumulation: {training['gradient_accumulation_steps']}",
        f"- Effective batch size: {training['effective_batch_size']}",
        f"- Optimizer steps: {training['optimizer_steps']}",
        f"- Trainable parameters: {training['trainable_surface']['trainable_parameter_count']}",
        "",
        "## Directional Metrics",
        "",
        "| Split | Base WER/CER | Scale-2000 joint WER/CER | Decoder+Joint RNNT WER/CER | Empty base/scale2000/decoder+joint |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics = public["directional_evaluation"]["metrics"]
    for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout", "fleurs_v2", "artur_j"):
        base = metrics["base"][split]
        scale = metrics["scale2000_joint_adapter"][split]
        challenger = metrics["decoder_joint_rnnt"][split]
        lines.append(f"| {split} | {base['wer']}/{base['cer']} | {scale['wer']}/{scale['cer']} | {challenger['wer']}/{challenger['cer']} | {base['empty']}/{scale['empty']}/{challenger['empty']} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Real-regression burden: {evaluation['decision']['real_burden']}",
            f"- Accepted parent: `{evaluation['decision']['accepted_parent']}`",
            "",
            "## Limitations",
            "",
            "- Synthetic-only training remains diagnostic.",
            "- Directional batch-32 metrics cannot promote a checkpoint.",
            "- Real speech remains validation-only and decisive for acceptance.",
        ]
    )
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(REPORT_MD, "\n".join(lines) + "\n")
    print(json.dumps({"status": "PASSED", "classification": evaluation["decision"]["classification"], "accepted_parent": "none"}, ensure_ascii=False, sort_keys=True))
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", required=True, choices=["verify-inputs", "probe-hardware", "probe-microbatch", "train", "evaluate-directional", "summarize"])
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if args.stage == "verify-inputs":
        stage_verify_inputs(config_path)
    elif args.stage == "probe-hardware":
        stage_probe_hardware(config_path)
    elif args.stage == "probe-microbatch":
        stage_probe_microbatch(config_path, args.progress_interval_seconds)
    elif args.stage == "train":
        stage_train(config_path, args.progress_interval_seconds)
    elif args.stage == "evaluate-directional":
        stage_evaluate_directional(config_path)
    elif args.stage == "summarize":
        stage_summarize(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
