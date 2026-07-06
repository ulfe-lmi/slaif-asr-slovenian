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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from slaif_asr.batched_streaming import NvidiaSmiMonitor, file_sha256, load_local_predictions, metrics_for, parse_monitor_csv, run_batched_arm
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_scoring import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION, NEMO_REVISION, checkpoint_path, nemo_streaming_script, runtime_environment, verify_runtime_identities
from slaif_asr.corpus_v2_training import assert_epoch_covers_once, deterministic_epoch_batches, make_training_batch, rnnt_loss, select_probe_records
from slaif_asr.data_quality import atomic_write_json
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread
from slaif_asr.prompt_column import derive_prompt_column_selection
from slaif_asr.rtx2080ti_policy import nvidia_smi_inventory, require_single_visible_rtx2080ti, select_microbatch
from slaif_asr.scale8000_clean_training import (
    ARM_NAME,
    BASE_DIRECTIONAL_METRICS,
    SCALE2000_DIRECTIONAL_METRICS,
    assert_public_report_safe,
    build_clean_exposure_schedule,
    classify_scale8000_clean,
    clean_training_records_for_round,
    directional_suite,
    load_clean_training_bank,
    load_config,
    local_path,
    metric_row,
    microbatch_plan,
    protected_file_fingerprints,
    read_json,
    read_jsonl,
    run_dir,
    stable_sha256,
    validate_clean_exposure_schedule,
    verify_local_scale8000_inputs,
    verify_protected_file_fingerprints,
    verify_scale8000_public_evidence,
    write_clean_exposure_schedule,
)
from slaif_asr.slovenian_joint_adapter import (
    ADAPTER_NAME,
    adapter_parameters,
    compare_adapter_state,
    compare_base_state,
    expected_trainable_count,
    load_adapter_artifact,
    load_adapter_spec,
    save_adapter_artifact,
    state_dict_cpu,
    verify_optimizer_scope,
)


DEFAULT_CONFIG = Path("configs/experiments/scale8000_clean_only_2080ti_v1.json")
REPORT_JSON = Path("docs/experiments/0015-scale8000-clean-only-2080ti-directional.json")
REPORT_MD = Path("docs/experiments/0015-scale8000-clean-only-2080ti-directional.md")
CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v5-scale8000-clean-only-diagnostic-v1.json")

_JOINT_PATH = Path(__file__).with_name("run_corpus_v2_joint_adapter_diagnostic.py")
_JOINT_SPEC = importlib.util.spec_from_file_location("_slaif_joint_runner_scale8000_clean", _JOINT_PATH)
if _JOINT_SPEC is None or _JOINT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import joint-adapter runner")
_JOINT = importlib.util.module_from_spec(_JOINT_SPEC)
_JOINT_SPEC.loader.exec_module(_JOINT)


def ensure_cuda_nvcc_process_env() -> None:
    cuda_home = REPO_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cuda_nvcc"
    if cuda_home.exists():
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
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch = _JOINT.configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def restore_base_model(config: dict[str, Any], *, reporter: LiveProgressReporter | None = None) -> Any:
    return _JOINT.restore_base_model(config, reporter=reporter)


def prepare_adapter_model(model: Any, config: dict[str, Any], *, enable: bool) -> dict[str, Any]:
    return _JOINT.prepare_adapter_model(model, config, enable=enable)


def finite_grad_norm(parameters: list[Any]) -> tuple[float, bool]:
    return _JOINT.finite_grad_norm(parameters)


def enable_for_target_language(model: Any, target_lang: str) -> None:
    _JOINT.enable_for_target_language(model, target_lang)


def rtx_runtime_summary(hardware: Any, torch: Any) -> dict[str, Any]:
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
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def verify_common_inputs(config: dict[str, Any]) -> dict[str, Any]:
    public = verify_scale8000_public_evidence(config)
    local = verify_local_scale8000_inputs(config)
    reference = verify_scale2000_reference(config)
    protected = protected_file_fingerprints(config)
    return {"public_evidence": public, "local_inputs": local, "scale2000_reference": reference, "protected_file_fingerprints": protected}


def verify_scale2000_reference(config: dict[str, Any]) -> dict[str, Any]:
    ref = config["scale2000_reference_report"]
    path = REPO_ROOT / ref["path"]
    actual = file_sha256(path)
    if actual != ref["sha256"]:
        raise RuntimeError(f"Experiment 0014 report SHA mismatch: {actual}")
    report = read_json(path)
    decision = report["directional_evaluation"]["decision"]
    if decision["classification"] != "SCALE2000_TEXT_REAL_GAIN_DIRECTIONAL":
        raise RuntimeError("Experiment 0014 classification mismatch")
    if round(float(decision["scale2000_burden"]), 6) != float(ref["scale2000_burden"]):
        raise RuntimeError("Experiment 0014 burden mismatch")
    return {"path": ref["path"], "sha256": actual, "classification": decision["classification"], "scale2000_burden": decision["scale2000_burden"]}


def stage_verify_inputs(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    payload = {"status": "PASSED", "work_order_id": config["work_order_id"], **verify_common_inputs(config)}
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
            payload["selected_visible_device_error"] = type(exc).__name__
    write_json(run_dir(config) / "verification" / "hardware.local.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if payload["status"] != "PASSED":
        raise RuntimeError("no RTX 2080 Ti available")
    return payload


def _representative_records(config: dict[str, Any], count: int = 8):
    bank = load_clean_training_bank(config, validate_audio=False)
    records = [record for voices in bank.values() for record in voices.values()]
    return sorted(records, key=lambda record: (-record.duration, stable_sha256(record.selected_training_id)))[:count]


def _zero_grad(model: Any) -> None:
    for parameter in adapter_parameters(model):
        parameter.grad = None


def _accumulated_loss_and_grad(model: Any, prompt_index: int, records: Sequence[Any], *, physical_microbatch: int, torch: Any) -> dict[str, Any]:
    _zero_grad(model)
    weighted_loss = 0.0
    for start in range(0, len(records), physical_microbatch):
        micro = records[start : start + physical_microbatch]
        loss = rnnt_loss(model, make_training_batch(model, micro, device="cuda"), prompt_index)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite microbatch loss")
        scale = len(micro) / len(records)
        (loss * scale).backward()
        weighted_loss += float(loss.detach().cpu()) * scale
    grad_norm, grads_ok = finite_grad_norm(adapter_parameters(model))
    if not grads_ok:
        raise RuntimeError("non-finite accumulated gradient")
    return {"weighted_loss": round(weighted_loss, 6), "gradient_norm": round(grad_norm, 6), "finite": True}


def stage_probe_microbatch(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    info = require_single_visible_rtx2080ti()
    torch = configure_torch()
    records = _representative_records(config, 8)
    reporter = LiveProgressReporter(stage="probe_microbatch", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "probe-microbatch.local.ndjson")
    reporter.start("probing RTX 2080 Ti microbatch")
    model = restore_base_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "probe-restore.local.ndjson"))
    model.eval()
    adapter_summary = prepare_adapter_model(model, config, enable=True)
    prompt = derive_prompt_column_selection(model, "sl-SI")
    outcomes: dict[int, dict[str, Any]] = {}
    for candidate in config["training"]["physical_microbatch_candidates"]:
        reporter.progress(step=len(outcomes), total_steps=4, message=f"candidate_microbatch_{candidate}")
        try:
            torch.cuda.empty_cache()
            _zero_grad(model)
            batch_records = records[:candidate]
            loss = rnnt_loss(model, make_training_batch(model, batch_records, device="cuda"), prompt.prompt_index)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            loss.backward()
            grad_norm, grads_ok = finite_grad_norm(adapter_parameters(model))
            free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
            if not grads_ok:
                raise RuntimeError("non-finite probe gradient")
            free_mib = int(free_bytes // 1024 // 1024)
            if free_mib < int(config["hardware"]["minimum_free_vram_mib_after_warmup"]):
                raise RuntimeError("insufficient free VRAM after warmup")
            outcomes[int(candidate)] = {
                "status": "PASSED",
                "loss": round(float(loss.detach().cpu()), 6),
                "gradient_norm": round(grad_norm, 6),
                "free_vram_mib_after_warmup": free_mib,
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
            }
        except Exception as exc:
            outcomes[int(candidate)] = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc).splitlines()[0][:240]}
            torch.cuda.empty_cache()
        finally:
            _zero_grad(model)
    selected = select_microbatch(config["training"]["physical_microbatch_candidates"], outcomes)
    correctness = None
    if selected["status"] == "PASSED":
        correctness = _accumulated_loss_and_grad(model, prompt.prompt_index, records, physical_microbatch=int(selected["physical_microbatch"]), torch=torch)
        singleton = _accumulated_loss_and_grad(model, prompt.prompt_index, records, physical_microbatch=1, torch=torch)
        rel_diff = abs(correctness["weighted_loss"] - singleton["weighted_loss"]) / singleton["weighted_loss"] if singleton["weighted_loss"] else 0.0
        correctness.update({"singleton_weighted_loss": singleton["weighted_loss"], "relative_loss_difference_vs_singletons": round(rel_diff, 8), "passed": rel_diff <= 0.005})
    payload = {
        "status": selected["status"],
        "hardware": info.to_dict(),
        "adapter": adapter_summary,
        "candidate_outcomes": {str(key): value for key, value in outcomes.items()},
        "selected": selected,
        "correctness": correctness,
    }
    write_json(run_dir(config) / "verification" / "microbatch.local.json", payload)
    if selected["status"] != "PASSED":
        raise RuntimeError("physical microbatch 1 failed; environment blocked")
    if not correctness["passed"]:
        raise RuntimeError("gradient accumulation correctness probe failed")
    reporter.complete("microbatch probe complete")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def _mean_loss(model: Any, prompt_index: int, records: Sequence[Any], *, torch: Any) -> float:
    losses = []
    ordered_records = sorted(records, key=lambda record: (-record.duration, stable_sha256(record.selected_training_id)))
    with torch.no_grad():
        for index, record in enumerate(ordered_records, start=1):
            batch = make_training_batch(model, [record], device="cuda")
            loss = rnnt_loss(model, batch, prompt_index)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            losses.append(float(loss.detach().cpu()))
            del loss
            del batch
            if index % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    return sum(losses) / len(losses)


def _anchor_probe(bank: dict[str, dict[str, Any]], count: int) -> list[Any]:
    inherited_keys = [key for key in bank if not key.startswith("gamsv5-")]
    piper_records = [bank[key]["piper-sl_SI-artur-medium"] for key in inherited_keys]
    return select_probe_records(piper_records, count)


def _scale_probe(bank: dict[str, dict[str, Any]], count: int) -> list[Any]:
    piper_records = [voices["piper-sl_SI-artur-medium"] for voices in bank.values()]
    return select_probe_records(piper_records, count)


def stage_train(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    verify_common_inputs(config)
    hardware = require_single_visible_rtx2080ti()
    verify_runtime_identities(check_gpu=False)
    torch = configure_torch()
    micro_path = run_dir(config) / "verification" / "microbatch.local.json"
    if not micro_path.exists():
        raise RuntimeError("microbatch probe must run before training")
    micro = read_json(micro_path)["selected"]
    physical_microbatch = int(micro["physical_microbatch"])
    accumulation = int(micro["gradient_accumulation_steps"])
    plan = microbatch_plan(physical_microbatch)
    bank = load_clean_training_bank(config, validate_audio=False)
    schedule, schedule_summary = build_clean_exposure_schedule(config, bank)
    schedule_path = run_dir(config) / "training" / "clean-exposure-schedule.local.jsonl"
    schedule_sha = write_clean_exposure_schedule(schedule_path, schedule)
    reporter = LiveProgressReporter(stage="train", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "train.local.ndjson")
    reporter.start("training scale-8000 clean-only joint adapter")
    model = restore_base_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "restore.local.ndjson"))
    model.eval()
    adapter_summary = prepare_adapter_model(model, config, enable=True)
    if adapter_summary["trainable_parameters"] != expected_trainable_count(adapter_summary["joint_hidden"], 32):
        raise RuntimeError("joint-adapter trainable parameter count mismatch")
    initial_state = state_dict_cpu(model)
    prompt = derive_prompt_column_selection(model, "sl-SI")
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=float(config["training"]["learning_rate"]), weight_decay=0.0)
    verify_optimizer_scope(optimizer, model)
    anchor_probe = _anchor_probe(bank, int(config["training"]["anchor_probe_rows"]))
    scale_probe = _scale_probe(bank, int(config["training"]["scale_probe_rows"]))
    initial_anchor = _mean_loss(model, prompt.prompt_index, anchor_probe, torch=torch)
    initial_scale = _mean_loss(model, prompt.prompt_index, scale_probe, torch=torch)
    probe_curve = [{"round": 0, "anchor_probe_loss": round(initial_anchor, 6), "scale_probe_loss": round(initial_scale, 6)}]
    grad_norms: list[float] = []
    adapter_norm_curve: list[dict[str, Any]] = []
    loss_by_voice: Counter[str] = Counter()
    count_by_voice: Counter[str] = Counter()
    optimizer_steps = 0
    sample_exposures = 0
    audio_seconds = 0.0
    padded_audio_seconds = 0.0
    rolling_losses: list[float] = []
    arm_dir = run_dir(config) / ARM_NAME
    monitor_path = arm_dir / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(physical_gpu_index=hardware.physical_selector, output_csv=monitor_path, interval_seconds=0.5)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    monitor.start()
    try:
        for round_index in range(1, 10):
            records = clean_training_records_for_round(bank, round_index)
            layout = deterministic_epoch_batches(records, batch_size=8, epoch=round_index, seed=int(config["training"]["seed"]), bucketed=True)
            assert_epoch_covers_once(layout, len(records))
            for batch_indices in layout.batches:
                batch_records = [records[index] for index in batch_indices]
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for start_index in range(0, len(batch_records), physical_microbatch):
                    micro_records = batch_records[start_index : start_index + physical_microbatch]
                    loss = rnnt_loss(model, make_training_batch(model, micro_records, device="cuda"), prompt.prompt_index)
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite scale-8000 training loss")
                    scale = len(micro_records) / 8.0
                    (loss * scale).backward()
                    step_loss += float(loss.detach().cpu()) * scale
                grad_norm, grads_ok = finite_grad_norm(adapter_parameters(model))
                if not grads_ok:
                    raise RuntimeError("non-finite scale-8000 training gradient")
                for name, parameter in model.named_parameters():
                    if not name.startswith(f"joint.adapter_layer.{ADAPTER_NAME}.") and parameter.grad is not None:
                        raise RuntimeError(f"pretrained parameter received gradient: {name}")
                optimizer.step()
                optimizer_steps += 1
                sample_exposures += len(batch_records)
                audio_seconds += sum(row.duration for row in batch_records)
                padded_audio_seconds += max(row.duration for row in batch_records) * len(batch_records)
                for row in batch_records:
                    loss_by_voice[row.voice] += step_loss
                    count_by_voice[row.voice] += 1
                grad_norms.append(grad_norm)
                rolling_losses.append(step_loss)
                rolling_losses = rolling_losses[-25:]
                if optimizer_steps % 500 == 0:
                    elapsed = time.perf_counter() - start
                    reporter.progress(
                        step=optimizer_steps,
                        total_steps=int(config["training"]["optimizer_steps"]),
                        current_loss=round(step_loss, 6),
                        rolling_mean_loss=round(sum(rolling_losses) / len(rolling_losses), 6),
                        examples_per_second=round(sample_exposures / elapsed, 6) if elapsed else None,
                        audio_seconds_per_wall_second=round(audio_seconds / elapsed, 6) if elapsed else None,
                        cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    )
            round_anchor = _mean_loss(model, prompt.prompt_index, anchor_probe, torch=torch)
            round_scale = _mean_loss(model, prompt.prompt_index, scale_probe, torch=torch)
            probe_curve.append({"round": round_index, "anchor_probe_loss": round(round_anchor, 6), "scale_probe_loss": round(round_scale, 6)})
            adapter_norm = sum(float(torch.linalg.vector_norm(parameter.detach()).cpu()) for parameter in adapter_parameters(model))
            adapter_norm_curve.append({"round": round_index, "adapter_parameter_norm": round(adapter_norm, 6)})
    except Exception as exc:
        reporter.failed(message="training failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()
    if optimizer_steps != int(config["training"]["optimizer_steps"]):
        raise RuntimeError(f"optimizer step count mismatch: {optimizer_steps}")
    with heartbeat_thread(reporter, interval_seconds=interval, message="post-training integrity checks"):
        wall = time.perf_counter() - start
        final_anchor = _mean_loss(model, prompt.prompt_index, anchor_probe, torch=torch)
        final_scale = _mean_loss(model, prompt.prompt_index, scale_probe, torch=torch)
        trained_state = state_dict_cpu(model)
        base_integrity = compare_base_state(initial_state, trained_state)
        adapter_integrity = compare_adapter_state(initial_state, trained_state)
        if not base_integrity["base_tensors_identical"]:
            raise RuntimeError("pretrained tensor changed during scale-8000 clean-only training")
        artifact_path = arm_dir / "artifacts" / "sl-si-joint-adapter-v1.pt"
        artifact_sha = save_adapter_artifact(
            artifact_path,
            model=model,
            spec=load_adapter_spec(config["adapter"]["config"]),
            metadata={
                "base_checkpoint_sha256": CHECKPOINT_SHA256,
                "nemo_revision": NEMO_REVISION,
                "scale8000_clean_schedule_sha256": schedule_sha,
                "experiment_config_sha256": file_sha256(config_path),
                "adapter_config_sha256": file_sha256(REPO_ROOT / config["adapter"]["config"]),
                "adapter_config": read_json(REPO_ROOT / config["adapter"]["config"]),
            },
        )
        verify_cmd = [sys.executable, "-u", __file__, "--config", str(config_path), "--stage", "verify-artifact"]
        completed = subprocess.run(verify_cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy(), check=False)
        (arm_dir / "verify-artifact.log").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError("adapter artifact restore verification failed")
        enable_for_target_language(model, "sl-SI")
        checkpoint_out = arm_dir / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
        model.save_to(str(checkpoint_out))
    payload = {
        "arm": ARM_NAME,
        "status": "PASSED",
        "semantic_rows": int(config["training"]["semantic_rows"]),
        "exposure_rounds": 9,
        "sample_exposures": sample_exposures,
        "effective_batch_size": 8,
        "physical_microbatch": physical_microbatch,
        "gradient_accumulation_steps": accumulation,
        "optimizer_steps": optimizer_steps,
        "learning_rate": float(config["training"]["learning_rate"]),
        "schedule_sha256": schedule_sha,
        "schedule": schedule_summary,
        "initial_anchor_probe_loss": round(initial_anchor, 6),
        "final_anchor_probe_loss": round(final_anchor, 6),
        "initial_scale_probe_loss": round(initial_scale, 6),
        "final_scale_probe_loss": round(final_scale, 6),
        "probe_curve": probe_curve,
        "gradient_norm": {"min": round(min(grad_norms), 6), "max": round(max(grad_norms), 6), "final": round(grad_norms[-1], 6)},
        "adapter_norm_curve": adapter_norm_curve,
        "loss_by_voice": {voice: round(loss_by_voice[voice] / max(1, count_by_voice[voice]), 6) for voice in sorted(count_by_voice)},
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
        "runtime": rtx_runtime_summary(hardware, torch),
        "hardware": hardware.to_dict(),
    }
    write_json(arm_dir / "training-summary.local.json", payload)
    reporter.complete("training complete", step=optimizer_steps, total_steps=int(config["training"]["optimizer_steps"]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def stage_verify_artifact(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
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
    report = {"status": "PASSED", "artifact_name": payload["adapter_name"], "base_integrity": base_integrity, "disabled_after_restore": True}
    write_json(run_dir(config) / ARM_NAME / "restore-integrity.local.json", report)
    return report


def _split_predictions(suite: Sequence[Any], predictions: dict[str, str]) -> dict[str, dict[str, str]]:
    expected = {row.sample_id for row in suite}
    if set(predictions) != expected:
        raise RuntimeError("directional prediction set mismatch")
    output: dict[str, dict[str, str]] = {}
    for row in suite:
        split = str(row.row["split"])
        output.setdefault(split, {})[row.sample_id] = predictions[row.sample_id]
    return output


def _available_rtx2080ti_indices() -> list[str]:
    inventory = nvidia_smi_inventory()
    indices = [str(gpu.index) for gpu in inventory if "RTX 2080 Ti" in gpu.name]
    if len(indices) < 2:
        raise RuntimeError(f"dual-GPU directional evaluation requires two RTX 2080 Ti GPUs, found {indices}")
    return indices[:2]


def _evaluation_shards(suite: Sequence[Any], gpu_indices: Sequence[str]) -> list[tuple[str, list[Any]]]:
    shards = [(gpu, []) for gpu in gpu_indices]
    for index, record in enumerate(sorted(suite, key=lambda row: row.original_index)):
        shards[index % len(shards)][1].append(record)
    all_ids = [row.sample_id for _gpu, records in shards for row in records]
    if len(all_ids) != len(suite) or len(set(all_ids)) != len(suite):
        raise RuntimeError("directional evaluation shard split produced missing or duplicate rows")
    return shards


def _run_directional_shard(
    *,
    config: dict[str, Any],
    checkpoint: Path,
    records: Sequence[Any],
    gpu_index: str,
    shard_index: int,
) -> dict[str, Any]:
    env = runtime_environment()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["NVIDIA_TF32_OVERRIDE"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    shard_dir = run_dir(config) / "evaluation" / ARM_NAME / "directional-suite" / f"shard-{shard_index:02d}-gpu{gpu_index}"
    arm = run_batched_arm(
        records=records,
        batch_size=32,
        bucketed=True,
        run_dir=shard_dir,
        python_executable=Path(sys.executable),
        nemo_script=nemo_streaming_script(),
        checkpoint=checkpoint,
        context=config["evaluation"]["att_context_size"],
        env=env,
        physical_gpu_index=str(gpu_index),
        monitor_interval_seconds=0.5,
    )
    if arm.get("status") != "PASSED":
        raise RuntimeError(f"directional evaluation shard {shard_index} on GPU {gpu_index} failed: {arm.get('status')}")
    predictions = load_local_predictions(shard_dir / "predictions.local.jsonl")
    return {
        "shard_index": shard_index,
        "physical_gpu_index": str(gpu_index),
        "rows": len(records),
        "sample_ids": [row.sample_id for row in records],
        "predictions": predictions,
        "summary": arm,
        "run_dir_token": f"shard-{shard_index:02d}-gpu{gpu_index}",
    }


def stage_evaluate_directional(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    gpu_indices = _available_rtx2080ti_indices()
    verify_common_inputs(config)
    checkpoint = run_dir(config) / ARM_NAME / "artifacts" / f"{ARM_NAME}-enabled.local.nemo"
    if not checkpoint.exists():
        raise RuntimeError("scale-8000 clean-only checkpoint is missing")
    suite, split_records = directional_suite(config)
    shards = _evaluation_shards(suite, gpu_indices)
    started = time.perf_counter()
    shard_results = []
    with ThreadPoolExecutor(max_workers=len(shards)) as executor:
        futures = [
            executor.submit(
                _run_directional_shard,
                config=config,
                checkpoint=checkpoint,
                records=records,
                gpu_index=gpu_index,
                shard_index=index,
            )
            for index, (gpu_index, records) in enumerate(shards)
        ]
        for future in as_completed(futures):
            shard_results.append(future.result())
    wall_time = time.perf_counter() - started
    predictions = {}
    for result in shard_results:
        overlap = set(predictions).intersection(result["predictions"])
        if overlap:
            raise RuntimeError(f"duplicate directional predictions across shards: {sorted(overlap)[:5]}")
        predictions.update(result["predictions"])
    expected_ids = {row.sample_id for row in suite}
    if set(predictions) != expected_ids:
        missing = sorted(expected_ids - set(predictions))
        unexpected = sorted(set(predictions) - expected_ids)
        raise RuntimeError(f"directional sharded prediction mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    merged_predictions_path = run_dir(config) / "evaluation" / ARM_NAME / "directional-suite" / "predictions.local.jsonl"
    merged_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    from slaif_asr.real_eval import atomic_write_jsonl

    atomic_write_jsonl(
        merged_predictions_path,
        [{"sample_id": row.sample_id, "hypothesis": predictions[row.sample_id]} for row in sorted(suite, key=lambda row: row.original_index)],
    )
    by_split = _split_predictions(suite, predictions)
    split_summaries = {}
    metric_table = {}
    for split, records in split_records.items():
        safe_records = [row for row in suite if row.row["split"] == split]
        metrics = metrics_for(safe_records, by_split[split])
        split_summaries[split] = {"rows": len(records), "audio_duration_seconds": round(sum(row.duration for row in records), 6), "metrics": metrics}
        metric_table[split] = metric_row(split_summaries[split])
    decision = classify_scale8000_clean(metric_table)
    payload = {
        "status": "PASSED",
        "policy": config["evaluation"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "suite_rows": len(suite),
        "suite_summary": {
            "wall_time_seconds": round(wall_time, 6),
            "audio_duration_seconds": round(sum(row.duration for row in suite), 6),
            "real_time_factor": round(wall_time / sum(row.duration for row in suite), 6) if suite else None,
            "utterances_per_second": round(len(suite) / wall_time, 6) if wall_time else None,
            "second_gpu_used": True,
            "sharded_evaluation": True,
            "shards": [
                {
                    "shard_index": result["shard_index"],
                    "physical_gpu_index": result["physical_gpu_index"],
                    "rows": result["rows"],
                    "run_dir_token": result["run_dir_token"],
                    "wall_time_seconds": result["summary"]["execution"]["wall_time_seconds"],
                    "audio_duration_seconds": result["summary"]["audio_duration_seconds"],
                    "utterances_per_second": result["summary"]["utterances_per_second"],
                    "layout": result["summary"]["layout"],
                    "gpu_monitor": result["summary"]["execution"]["monitor"],
                }
                for result in sorted(shard_results, key=lambda item: item["shard_index"])
            ],
        },
        "splits": split_summaries,
        "metric_table": metric_table,
        "decision": decision,
        "hardware": {
            "inventory": [row.to_dict() for row in nvidia_smi_inventory()],
            "evaluation_gpus": gpu_indices,
            "mode": "two independent RTX 2080 Ti replicas on disjoint row shards",
        },
    }
    write_json(run_dir(config) / "evaluation" / "summary.local.json", payload)
    print(json.dumps({"status": "PASSED", "decision": decision, "metric_table": metric_table}, ensure_ascii=False, sort_keys=True))
    return payload


def _public_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "base": BASE_DIRECTIONAL_METRICS,
        "scale2000_augmented": SCALE2000_DIRECTIONAL_METRICS,
        "scale8000_clean_only": evaluation["metric_table"],
    }


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inputs = verify_common_inputs(config)
    micro = read_json(run_dir(config) / "verification" / "microbatch.local.json")
    training = read_json(run_dir(config) / ARM_NAME / "training-summary.local.json")
    evaluation = read_json(run_dir(config) / "evaluation" / "summary.local.json")
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v5-scale8000-clean-only-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "work_order_id": "0028",
        "corpus_id": config["data"]["corpus_id"],
        "text_sha256": config["data"]["text_sha256"],
        "piper_manifest_sha256": config["data"]["piper_manifest_sha256"],
        "supertonic_manifest_sha256": config["data"]["supertonic_manifest_sha256"],
        "clean_views": config["data"]["clean_views"],
        "augmented_views": 0,
        "training_arm": ARM_NAME,
        "selected_physical_microbatch": training["physical_microbatch"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "effective_batch_size": 8,
        "optimizer_steps": training["optimizer_steps"],
        "adapter_trainable_parameters": training["trainable_parameter_count"],
        "pretrained_tensors_unchanged": training["base_integrity"]["base_tensors_identical"],
        "directional_batch_size": 32,
        "classification": evaluation["decision"]["classification"],
        "accepted_parent": "none",
        "prohibited_actions": ["TRAINING_ELIGIBLE", "model publication", "adapter acceptance", "checkpoint acceptance"],
    }
    public = {
        "schema_version": "1.0",
        "experiment_id": "scale8000-clean-only-2080ti-directional-v1",
        "status": "completed in PR; pending strategic review",
        "repository_commit": git_head(),
        "accepted_parent": "none",
        "authorization": {"status": "DIAGNOSTIC_ONLY", "work_order_id": "0028"},
        "input_integrity": inputs,
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256, "nemo_revision": NEMO_REVISION},
        "hardware": {
            "inventory": read_json(run_dir(config) / "verification" / "hardware.local.json"),
            "selected": training["hardware"],
            "second_gpu_used": bool(evaluation["suite_summary"].get("second_gpu_used")),
            "evaluation_gpus": evaluation.get("hardware", {}).get("evaluation_gpus", []),
        },
        "training": {
            key: training[key]
            for key in (
                "status",
                "semantic_rows",
                "exposure_rounds",
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
                "adapter_norm_curve",
                "loss_by_voice",
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
        },
        "microbatch_probe": micro,
        "directional_evaluation": {
            "policy": evaluation["policy"],
            "suite_summary": evaluation["suite_summary"],
            "metrics": _public_metrics(evaluation),
            "decision": evaluation["decision"],
        },
        "limitations": [
            "Clean-only scale-8000 uses synthetic TTS audio only.",
            "The 3600x figure is exposure count relative to the original 160-item reference, not independent linguistic information.",
            "Directional batch-32 evaluation is not canonical acceptance evidence.",
            "No batch-1 canonical evaluation was run.",
        ],
        "safety": {
            "training_eligible_issued": False,
            "accepted_parent": "none",
            "generated_audio_committed": False,
            "model_or_adapter_committed": False,
            "raw_predictions_committed": False,
        },
    }
    assert_public_report_safe(public)
    assert_public_report_safe(certificate)
    write_json(REPORT_JSON, public)
    write_json(CERTIFICATE_PATH, certificate)
    lines = [
        "# Experiment 0015: Scale-8000 Clean-Only RTX 2080 Ti Directional",
        "",
        f"Classification: `{evaluation['decision']['classification']}`",
        "",
        "This is directional, noncanonical batch-32 evidence. No batch-1 canonical evaluation was run, no checkpoint or adapter is accepted, and `accepted_parent` remains `none`.",
        "",
        "## Data",
        "",
        f"- Corpus: `{config['data']['corpus_id']}`",
        f"- Text SHA256: `{config['data']['text_sha256']}`",
        f"- Semantic rows: {config['data']['text_rows']}",
        f"- Clean views: {config['data']['clean_views']}",
        "- Augmented views: 0",
        "",
        "## Training",
        "",
        f"- Arm: `{ARM_NAME}`",
        f"- Physical microbatch: {training['physical_microbatch']}",
        f"- Gradient accumulation: {training['gradient_accumulation_steps']}",
        f"- Effective batch size: {training['effective_batch_size']}",
        f"- Optimizer steps: {training['optimizer_steps']}",
        f"- Trainable parameters: {training['trainable_parameter_count']}",
        "",
        "## Directional Metrics",
        "",
        "| Split | Base WER/CER | Scale-2000 WER/CER | Scale-8000 clean WER/CER | Empty base/scale2000/scale8000 |",
        "|---|---:|---:|---:|---:|",
    ]
    metrics = _public_metrics(evaluation)
    for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout", "fleurs_v2", "artur_j"):
        base = metrics["base"][split]
        s2000 = metrics["scale2000_augmented"][split]
        s8000 = metrics["scale8000_clean_only"][split]
        lines.append(f"| {split} | {base['wer']}/{base['cer']} | {s2000['wer']}/{s2000['cer']} | {s8000['wer']}/{s8000['cer']} | {base['empty']}/{s2000['empty']}/{s8000['empty']} |")
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
            "- Synthetic-only training remains a diagnostic signal.",
            "- Directional batch-32 metrics cannot promote a model.",
            "- Real speech remains validation-only and decisive for acceptance.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASSED", "classification": evaluation["decision"]["classification"], "accepted_parent": "none"}, ensure_ascii=False, sort_keys=True))
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", required=True, choices=["verify-inputs", "probe-hardware", "probe-microbatch", "train", "verify-artifact", "evaluate-directional", "summarize"])
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    ensure_cuda_nvcc_process_env()
    args = parse_args()
    config_path = Path(args.config)
    try:
        if args.stage == "verify-inputs":
            stage_verify_inputs(config_path)
        elif args.stage == "probe-hardware":
            stage_probe_hardware(config_path)
        elif args.stage == "probe-microbatch":
            stage_probe_microbatch(config_path)
        elif args.stage == "train":
            stage_train(config_path, args.progress_interval_seconds)
        elif args.stage == "verify-artifact":
            stage_verify_artifact(config_path)
        elif args.stage == "evaluate-directional":
            stage_evaluate_directional(config_path)
        elif args.stage == "summarize":
            stage_summarize(config_path)
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "stage": args.stage, "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
