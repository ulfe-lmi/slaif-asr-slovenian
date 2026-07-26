#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import Counter
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
    NEMO_REVISION,
    nemo_streaming_script,
    runtime_environment,
    verify_runtime_identities,
)
from slaif_asr.corpus_v2_training import TrainingRecord, token_ids
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.directional_evaluation import (
    load_directional_suite,
    split_predictions,
    write_privacy_safe_suite_manifest,
)
from slaif_asr.emission_rnnt_finetune import (
    BASE_DIRECTIONAL_METRICS,
    metric_row,
    probe_records,
    protected_file_fingerprints,
    rnnt_audio_loss,
    verify_protected_file_fingerprints,
    write_json,
)
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.live_progress import LiveProgressReporter
from slaif_asr.otf_augmentation_dataset import (
    PrefetchedMicrobatch,
    iter_prefetched_microbatches,
    percentile,
    run_determinism_checks_for_exposures,
)
from slaif_asr.prompt_column import derive_prompt_column_selection
from slaif_asr.scale8000_otf_surface08 import (
    ALGORITHM_VERSION,
    AUGMENTATION_KEY,
    CORPUS_ID,
    EXPERIMENT_ID,
    SEMANTIC_ROWS,
    SURFACE_ID,
    auxiliary_runs_root,
    build_round_tasks,
    diversity_statistics,
    load_clean_pool,
    load_config,
    load_profiles,
    resolve_under,
    scale8000_runs_root,
    validate_loader_telemetry,
    validate_public_report,
    verify_committed_scale8000_evidence,
)
from slaif_asr.trainable_surface_sweep import (
    PR36_METRICS,
    SURFACE06_METRICS,
    SURFACE07_METRICS,
    mark_controller_selection,
    should_stop_controller_curve,
)


DEFAULT_CONFIG = Path("configs/experiments/surface08-scale8000-otf-augmented.json")
ARM_NAME = "surface08_scale8000_otf_augmented"
REPORT_JSON = Path("docs/experiments/0031-surface08-scale8000-otf-augmented.json")
REPORT_MD = Path("docs/experiments/0031-surface08-scale8000-otf-augmented.md")
CERTIFICATE_PATH = Path("docs/data-certificates/sl-corpus-v5-scale8000-otf-surface08-diagnostic-v1.json")
SURFACE08_SCALE2000_METRICS = {
    "piper_synthetic_holdout": {"wer": 20.497, "cer": 6.112, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 5.202, "cer": 1.850, "empty": 0},
    "fleurs_v2": {"wer": 41.878, "cer": 13.186, "empty": 0},
    "artur_j": {"wer": 41.765, "cer": 13.553, "empty": 0},
}


_SURFACE08_PATH = Path(__file__).with_name("run_fixed_scale2000_surface08_full_encoder.py")
_SURFACE08_SPEC = importlib.util.spec_from_file_location("_slaif_surface08_reuse", _SURFACE08_PATH)
if _SURFACE08_SPEC is None or _SURFACE08_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import Surface08 runner")
_SURFACE08 = importlib.util.module_from_spec(_SURFACE08_SPEC)
_SURFACE08_SPEC.loader.exec_module(_SURFACE08)


def run_dir(config: dict[str, Any]) -> Path:
    return _SURFACE08.run_dir(config)


def checkpoint_dir(config: dict[str, Any], round_index: int) -> Path:
    return _SURFACE08._checkpoint_dir(config, round_index)


def configure_torch() -> Any:
    torch = _SURFACE08.configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch


def load_probe_records(config: dict[str, Any]) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    root = auxiliary_runs_root()
    probes = config["diagnostic_probes"]
    proxy = {
        "data": {
            "fixed_text": str(resolve_under(root, probes["fixed_text_relative_path"])),
            "all_views": str(resolve_under(root, probes["all_views_relative_path"])),
        }
    }
    return probe_records(proxy)


def make_waveform_batch(
    model: Any,
    delivery: PrefetchedMicrobatch,
    *,
    device: str,
) -> tuple[Any, Any, Any, Any]:
    import torch

    records = [record for record, _exposure_id in delivery.task.exposures]
    if len(records) != len(delivery.prepared.waveforms):
        raise RuntimeError("OTF record/waveform count mismatch")
    audios = []
    audio_lengths = []
    transcripts = []
    transcript_lengths = []
    for record, waveform, sample_rate in zip(
        records,
        delivery.prepared.waveforms,
        delivery.prepared.sample_rates,
        strict=True,
    ):
        if sample_rate != 16_000:
            raise RuntimeError("OTF waveform sample rate mismatch")
        audio = torch.from_numpy(waveform)
        if audio.dtype != torch.float32 or audio.ndim != 1 or audio.numel() == 0:
            raise RuntimeError("OTF waveform tensor is invalid")
        audios.append(audio)
        audio_lengths.append(audio.numel())
        ids = token_ids(model, record.transcript)
        transcripts.append(torch.tensor(ids, dtype=torch.long))
        transcript_lengths.append(len(ids))
    signal = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True, padding_value=0.0).to(device)
    max_tokens = max(len(item) for item in transcripts)
    padded_targets = torch.zeros((len(transcripts), max_tokens), dtype=torch.long)
    for index, item in enumerate(transcripts):
        padded_targets[index, : len(item)] = item
    return (
        signal,
        torch.tensor(audio_lengths, dtype=torch.long, device=device),
        padded_targets.to(device),
        torch.tensor(transcript_lengths, dtype=torch.long, device=device),
    )


def _controller_records(config: dict[str, Any]) -> list[Any]:
    control = config["controller_dev"]
    manifest = resolve_under(auxiliary_runs_root(), control["manifest_relative_path"])
    return load_controller_dev_records(
        manifest,
        expected_sha256=control["manifest_sha256"],
        expected_rows=int(control["rows"]),
    )


def evaluate_controller_checkpoint(
    config: dict[str, Any],
    checkpoint: Path,
    round_index: int,
    validation_gpu: str,
) -> dict[str, Any]:
    records = _controller_records(config)
    output_dir = run_dir(config) / "controller-dev" / f"round_{round_index:02d}"
    predictions_path = output_dir / "predictions.local.jsonl"
    if predictions_path.exists():
        predictions = load_local_predictions(predictions_path)
        if len(predictions) == len(records):
            return {
                "round": round_index,
                **_SURFACE08._controller_metric_row(metrics_for(records, predictions)),
                "available": True,
                "reused": True,
            }
    env = runtime_environment()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": validation_gpu,
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
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
        **_SURFACE08._controller_metric_row(metrics_for(records, predictions)),
        "available": True,
        "reused": False,
        "wall_time_seconds": arm["execution"]["wall_time_seconds"],
        "rows_per_second": arm["utterances_per_second"],
        "real_time_factor": arm["end_to_end_real_time_factor"],
        "peak_gpu_memory_mib": arm["execution"]["monitor"].get("peak_memory_mib"),
    }


def mark_controller_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def verify_inputs(config: dict[str, Any]) -> dict[str, Any]:
    committed = verify_committed_scale8000_evidence(config)
    profiles = load_profiles(config)
    pool = load_clean_pool(config, runs_root=scale8000_runs_root())
    model_path = (REPO_ROOT / config["model"]["checkpoint_path"]).resolve()
    if file_sha256(model_path) != config["model"]["checkpoint_sha256"]:
        raise RuntimeError("untouched base checkpoint SHA256 mismatch")
    control = config["controller_dev"]
    controller_certificate = json.loads((REPO_ROOT / control["certificate"]).read_text(encoding="utf-8"))
    controller_manifest = resolve_under(auxiliary_runs_root(), control["manifest_relative_path"])
    if controller_certificate.get("partition_id") != "artur-controller-dev-v1":
        raise RuntimeError("controller-dev certificate identity mismatch")
    if file_sha256(controller_manifest) != control["manifest_sha256"]:
        raise RuntimeError("controller-dev manifest SHA256 mismatch")
    probes = config["diagnostic_probes"]
    probe_fixed = resolve_under(auxiliary_runs_root(), probes["fixed_text_relative_path"])
    probe_views = resolve_under(auxiliary_runs_root(), probes["all_views_relative_path"])
    if file_sha256(probe_fixed) != probes["fixed_text_sha256"]:
        raise RuntimeError("diagnostic anchor text SHA256 mismatch")
    if file_sha256(probe_views) != probes["all_views_sha256"]:
        raise RuntimeError("diagnostic probe view SHA256 mismatch")
    return {
        "status": "PASSED",
        "experiment_id": EXPERIMENT_ID,
        "committed_evidence": committed,
        "local_scale8000": {
            "semantic_rows": pool.semantic_rows,
            "clean_files": pool.clean_files,
            "fixed_text_sha256": pool.fixed_text_sha256,
            "piper_restored_manifest_sha256": pool.piper_manifest_sha256,
            "supertonic_restored_manifest_sha256": pool.supertonic_manifest_sha256,
            "full_audio_integrity": "576000_OF_576000_PASSED",
        },
        "augmentation": {
            "augmentation_key": AUGMENTATION_KEY,
            "profile_count": len(profiles),
            "disk_output": False,
        },
        "controller_dev": {
            "partition_id": "artur-controller-dev-v1",
            "manifest_sha256": control["manifest_sha256"],
            "rows": int(control["rows"]),
        },
        "base_checkpoint_sha256": config["model"]["checkpoint_sha256"],
    }


def stage_verify_inputs(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    payload = verify_inputs(config)
    write_json(run_dir(config) / "verification" / "inputs.local.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def stage_probe_hardware(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    hardware = require_single_visible_cuda()
    torch = configure_torch()
    if hardware.device_name != "NVIDIA GeForce RTX 3090":
        raise RuntimeError("this work order requires exactly one visible RTX 3090")
    total_mib = int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    if total_mib < 22 * 1024:
        raise RuntimeError("RTX 3090 VRAM is below 22 GiB")
    payload = {
        "status": "PASSED",
        "gpu": hardware.device_name,
        "visible_device_count": hardware.visible_device_count,
        "physical_selector": hardware.physical_selector,
        "vram_mib": total_mib,
        "precision": "fp32",
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    write_json(run_dir(config) / "verification" / "hardware.local.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def stage_probe_surface(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    hardware = require_single_visible_cuda()
    torch = configure_torch()
    verify_runtime_identities(check_gpu=False)
    reporter = LiveProgressReporter(
        stage="probe_surface",
        arm=ARM_NAME,
        ndjson_path=run_dir(config) / "progress" / "surface.local.ndjson",
    )
    model = _SURFACE08.restore_base_model(config, reporter=reporter)
    summary = _SURFACE08.configure_surface08_trainable(model)
    expected = {
        "decoder_parameter_count": 14_940_160,
        "joint_parameter_count": 9_455_648,
        "encoder_all_layers_parameter_count": 604_545_024,
        "fusion_bridge_parameter_count": 4_459_520,
        "trainable_parameter_count": 633_400_352,
        "frozen_parameter_count": 4_596_736,
    }
    for key, value in expected.items():
        if getattr(summary, key) != value:
            raise RuntimeError(f"live model {key} mismatch")
    payload = {
        "status": "PASSED",
        "surface": summary.to_dict(),
        "runtime": _SURFACE08.runtime_summary(hardware, torch),
    }
    write_json(run_dir(config) / "verification" / "surface.local.json", payload)
    print(json.dumps(payload, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def stage_probe_otf(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    pool = load_clean_pool(config, runs_root=scale8000_runs_root())
    profiles = load_profiles(config)
    tasks = build_round_tasks(pool, 1, physical_microbatch=2)
    exposures = [tasks[index].exposures[0] for index in range(16)]
    checks = run_determinism_checks_for_exposures(
        exposures,
        profiles,
        corpus_id=CORPUS_ID,
        augmentation_key=AUGMENTATION_KEY,
        algorithm_version=ALGORITHM_VERSION,
        num_workers=3,
    )
    if checks["status"] != "PASSED":
        raise RuntimeError("OTF_AUG_DETERMINISM_INVALID")
    payload = {
        "status": "PASSED",
        "augmentation_key": AUGMENTATION_KEY,
        "algorithm_version": ALGORITHM_VERSION,
        "workers": 3,
        "profile_count": len(profiles),
        "checks": checks,
    }
    write_json(run_dir(config) / "verification" / "otf-determinism.local.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def stage_probe_microbatch(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    require_single_visible_cuda()
    torch = configure_torch()
    pool = load_clean_pool(config, runs_root=scale8000_runs_root())
    profiles = load_profiles(config)
    reporter = LiveProgressReporter(
        stage="probe_microbatch",
        arm=ARM_NAME,
        ndjson_path=run_dir(config) / "progress" / "microbatch.local.ndjson",
    )
    model = _SURFACE08.restore_base_model(config, reporter=reporter)
    _SURFACE08.configure_surface08_trainable(model)
    _SURFACE08.set_surface08_training_mode(model)
    prompt = derive_prompt_column_selection(model, "sl-SI")
    tasks = build_round_tasks(pool, 1, physical_microbatch=2)[:4]
    torch.cuda.reset_peak_memory_stats(0)
    _SURFACE08._zero_grad(model)
    losses = []
    deliveries = iter_prefetched_microbatches(
        tasks,
        profiles,
        num_workers=3,
        prefetch_microbatches=4,
        corpus_id=CORPUS_ID,
        augmentation_key=AUGMENTATION_KEY,
        algorithm_version=ALGORITHM_VERSION,
    )
    for delivery in deliveries:
        batch = make_waveform_batch(model, delivery, device="cuda")
        loss = rnnt_audio_loss(model, batch, prompt.prompt_index, frozen_encoder_no_grad=False)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Surface08 OTF microbatch probe loss")
        (loss * 0.25).backward()
        losses.append(float(loss.detach().cpu()))
        del loss, batch
    _SURFACE08._assert_gradient_scope(model)
    grad_norm, finite = _SURFACE08.finite_grad_norm(_SURFACE08._trainable_parameters(model))
    if not finite:
        raise RuntimeError("non-finite Surface08 OTF probe gradient")
    payload = {
        "status": "PASSED",
        "selected": {
            "physical_microbatch": 2,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 8,
        },
        "outcomes": {
            "4": {
                "status": "FAILED",
                "error_type": "PriorObservedTrainingOOM",
                "source_experiment": "0029-fixed-scale2000-surface08-full-encoder",
            },
            "2": {
                "status": "PASSED",
                "mean_loss": round(statistics.fmean(losses), 6),
                "gradient_norm": round(grad_norm, 6),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
            },
            "1": {"status": "NOT_RUN_SMALLER_THAN_PASSING_CANDIDATE"},
        },
    }
    write_json(run_dir(config) / "verification" / "microbatch.local.json", payload)
    print(json.dumps(payload, sort_keys=True))
    del prompt, model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def cumulative_fill_rate(
    events: Sequence[dict[str, Any]],
    wall_seconds: float,
) -> float:
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    wait_total = sum(float(row["consumer_wait_seconds"]) for row in events)
    return max(0.0, 1.0 - wait_total / wall_seconds)


def _round_telemetry(
    *,
    round_index: int,
    events: Sequence[dict[str, Any]],
    wall_seconds: float,
    examples: int,
) -> dict[str, Any]:
    waits = [float(row["consumer_wait_seconds"]) for row in events]
    queues = [int(row["queue_depth_after"]) for row in events]
    wait_total = sum(waits)
    payload = {
        "round": round_index,
        "examples_per_second": round(examples / wall_seconds, 6),
        "otf_fill_rate": round(cumulative_fill_rate(events, wall_seconds), 9),
        "consumer_wait_percent": round(wait_total / wall_seconds * 100.0, 6),
        "ready_microbatch_rate": round(
            sum(bool(row["ready_without_wait"]) for row in events) / len(events),
            9,
        ),
        "consumer_wait_p50_ms": round(percentile(waits, 0.50) * 1000.0, 6),
        "consumer_wait_p95_ms": round(percentile(waits, 0.95) * 1000.0, 6),
        "consumer_wait_p99_ms": round(percentile(waits, 0.99) * 1000.0, 6),
        "queue_p50": round(percentile(queues, 0.50), 6),
        "queue_p95": round(percentile(queues, 0.95), 6),
        "worker_count": 3,
        "microbatch_count": len(events),
        "worker_compute_seconds": round(
            sum(float(row["processing_wall_seconds"]) for row in events),
            6,
        ),
        "worker_cpu_seconds": round(
            sum(float(row["processing_cpu_seconds"]) for row in events),
            6,
        ),
        "output_audio_seconds": round(
            sum(float(row["output_audio_seconds"]) for row in events),
            6,
        ),
        "peak_worker_rss_mib": round(
            max(float(row["worker_peak_rss_mib"]) for row in events),
            3,
        ),
    }
    validate_loader_telemetry(payload)
    return payload


def stage_train(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    input_path = run_dir(config) / "verification" / "inputs.local.json"
    micro_path = run_dir(config) / "verification" / "microbatch.local.json"
    if not input_path.exists() or json.loads(input_path.read_text(encoding="utf-8")).get("status") != "PASSED":
        raise RuntimeError("passing input verification is required")
    if not micro_path.exists():
        raise RuntimeError("passing microbatch probe is required")
    micro = json.loads(micro_path.read_text(encoding="utf-8"))
    if micro.get("status") != "PASSED":
        raise RuntimeError("passing microbatch probe is required")
    physical = int(micro["selected"]["physical_microbatch"])
    accumulation = int(micro["selected"]["gradient_accumulation_steps"])
    if physical != 2 or accumulation != 4:
        raise RuntimeError("Surface08 scale-8000 OTF requires effective batch 8 via 2 x 4")

    protected = protected_file_fingerprints(config)
    hardware = require_single_visible_cuda()
    runtime_identities = verify_runtime_identities(check_gpu=False)
    torch = configure_torch()
    pool = load_clean_pool(config, runs_root=scale8000_runs_root())
    profiles = load_profiles(config)
    anchor_probe, scale_probe = load_probe_records(config)
    reporter = LiveProgressReporter(
        stage="train",
        arm=ARM_NAME,
        ndjson_path=run_dir(config) / "progress" / "train.local.ndjson",
    )
    reporter.start("training Surface08 on scale-8000 deterministic OTF augmentation")
    restore_reporter = LiveProgressReporter(
        stage="restore",
        arm=ARM_NAME,
        ndjson_path=run_dir(config) / "progress" / "restore.local.ndjson",
    )
    model, optimizer, state, initial_fingerprints, resume_round = _SURFACE08._load_or_initialize_training(
        config,
        torch,
        restore_reporter,
    )
    state.setdefault("loader_telemetry", [])
    state.setdefault("profile_counts", {})
    state.setdefault("voice_counts", {})
    state.setdefault("training_compute_wall_seconds", 0.0)
    prompt = derive_prompt_column_selection(model, "sl-SI")
    validation_gpu = hardware.physical_selector

    if not state["probe_curve"]:
        initial_anchor = _SURFACE08.mean_probe_loss(
            model,
            prompt.prompt_index,
            anchor_probe,
            torch=torch,
            reporter=reporter,
            message="round_0_anchor_probe",
            interval=interval,
        )
        initial_scale = _SURFACE08.mean_probe_loss(
            model,
            prompt.prompt_index,
            scale_probe,
            torch=torch,
            reporter=reporter,
            message="round_0_scale_probe",
            interval=interval,
        )
        state["probe_curve"].append(
            {
                "round": 0,
                "anchor_probe_loss": round(initial_anchor, 6),
                "scale_probe_loss": round(initial_scale, 6),
            }
        )
        base_row = {
            "round": 0,
            "optimizer_step": 0,
            "exposures_seen": 0,
            "unique_semantic_rows": 0,
            "train_loss": None,
            "synthetic_anchor_probe_loss": round(initial_anchor, 6),
            "synthetic_scale_probe_loss": round(initial_scale, 6),
        }
        marker = _SURFACE08._save_round_checkpoint(config, model, optimizer, torch, row=base_row)
        state["round_rows"].append(marker)
        controller = evaluate_controller_checkpoint(
            config,
            checkpoint_dir(config, 0) / "model.local.nemo",
            0,
            validation_gpu,
        )
        state["controller_rows"].append(
            {**base_row, **controller, "checkpoint_sha256": marker["checkpoint_sha256"]}
        )
        write_json(_SURFACE08._training_state_path(config), state)

    optimizer_steps = int(state["optimizer_steps"])
    exposures_seen = int(state["exposures_seen"])
    grad_norms = list(state["gradient_norms"])
    evaluated_rounds = {int(row["round"]) for row in state["controller_rows"]}
    if resume_round not in evaluated_rounds:
        marker = json.loads(
            (checkpoint_dir(config, resume_round) / "checkpoint-complete.local.json").read_text(encoding="utf-8")
        )
        del prompt, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
        controller = evaluate_controller_checkpoint(
            config,
            checkpoint_dir(config, resume_round) / "model.local.nemo",
            resume_round,
            validation_gpu,
        )
        state["controller_rows"].append({**marker, **controller})
        write_json(_SURFACE08._training_state_path(config), state)
        model, optimizer, state, initial_fingerprints, restored_round = _SURFACE08._load_or_initialize_training(
            config,
            torch,
            restore_reporter,
        )
        if restored_round != resume_round:
            raise RuntimeError("round changed while restoring after controller validation")
        prompt = derive_prompt_column_selection(model, "sl-SI")

    monitor_path = run_dir(config) / "gpu-monitor.local.csv"
    monitor = NvidiaSmiMonitor(
        physical_gpu_index=hardware.physical_selector,
        output_csv=monitor_path,
        interval_seconds=0.5,
    )
    torch.cuda.reset_peak_memory_stats(0)
    segment_started = time.perf_counter()
    total_training_wall = float(state.get("training_compute_wall_seconds", 0.0))
    profile_counts = Counter(state.get("profile_counts", {}))
    voice_counts = Counter(state.get("voice_counts", {}))
    stopped_reason = "max_rounds"
    stopped_round = resume_round
    active_round = resume_round
    monitor.start()
    try:
        for round_index in range(resume_round + 1, int(config["training"]["max_rounds"]) + 1):
            active_round = round_index
            _SURFACE08.set_surface08_training_mode(model)
            tasks = build_round_tasks(
                pool,
                round_index,
                physical_microbatch=physical,
                effective_batch=8,
                round_size=int(config["training"]["round_size_exposures"]),
            )
            events: list[dict[str, Any]] = []
            round_losses: list[float] = []
            effective_loss = 0.0
            round_started = time.perf_counter()
            last_progress = round_started
            optimizer.zero_grad(set_to_none=True)
            deliveries = iter_prefetched_microbatches(
                tasks,
                profiles,
                num_workers=int(config["otf_augmentation"]["workers"]),
                prefetch_microbatches=int(config["otf_augmentation"]["prefetch_microbatches"]),
                corpus_id=CORPUS_ID,
                augmentation_key=AUGMENTATION_KEY,
                algorithm_version=ALGORITHM_VERSION,
                timeout_seconds=180.0,
            )
            for delivery_index, delivery in enumerate(deliveries, 1):
                events.append(
                    {
                        "consumer_wait_seconds": delivery.consumer_wait_seconds,
                        "ready_without_wait": delivery.ready_without_wait,
                        "queue_depth_after": delivery.queue_depth_after,
                        "processing_wall_seconds": delivery.prepared.processing_wall_seconds,
                        "processing_cpu_seconds": delivery.prepared.processing_cpu_seconds,
                        "output_audio_seconds": delivery.prepared.output_audio_seconds,
                        "worker_peak_rss_mib": delivery.prepared.worker_peak_rss_mib,
                    }
                )
                batch = make_waveform_batch(model, delivery, device="cuda")
                loss = rnnt_audio_loss(
                    model,
                    batch,
                    prompt.prompt_index,
                    frozen_encoder_no_grad=False,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite Surface08 scale-8000 OTF RNNT loss")
                scale = len(delivery.task.exposures) / 8.0
                (loss * scale).backward()
                effective_loss += float(loss.detach().cpu()) * scale
                for record, _exposure_id in delivery.task.exposures:
                    voice_counts[record.voice] += 1
                for spec in delivery.prepared.specs:
                    profile_counts[spec.profile_id] += 1
                del loss, batch, delivery

                if delivery_index % accumulation == 0:
                    _SURFACE08._assert_gradient_scope(model)
                    grad_norm, finite = _SURFACE08.finite_grad_norm(
                        _SURFACE08._trainable_parameters(model)
                    )
                    if not finite:
                        raise RuntimeError("non-finite Surface08 scale-8000 OTF gradient")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    exposures_seen += 8
                    grad_norms.append(grad_norm)
                    round_losses.append(effective_loss)
                    effective_loss = 0.0

                now = time.perf_counter()
                if optimizer_steps % 500 == 0 or now - last_progress >= interval:
                    elapsed = now - round_started
                    reporter.progress(
                        epoch=round_index,
                        total_epochs=int(config["training"]["max_rounds"]),
                        step=optimizer_steps,
                        total_steps=int(config["training"]["max_optimizer_steps"]),
                        current_loss=round(round_losses[-1], 6) if round_losses else None,
                        rolling_mean_loss=round(statistics.fmean(round_losses[-25:]), 6)
                        if round_losses
                        else None,
                        examples_per_second=round(
                            (delivery_index * physical) / elapsed,
                            6,
                        ),
                        otf_fill_rate=round(
                            cumulative_fill_rate(events, elapsed),
                            6,
                        ),
                        cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    )
                    last_progress = now

            if len(events) != len(tasks) or optimizer_steps % 2_000 != 0:
                raise RuntimeError("scale-8000 OTF round did not complete exactly")
            round_training_wall = time.perf_counter() - round_started
            total_training_wall += round_training_wall
            telemetry = _round_telemetry(
                round_index=round_index,
                events=events,
                wall_seconds=round_training_wall,
                examples=16_000,
            )
            state["loader_telemetry"].append(telemetry)

            anchor_loss = _SURFACE08.mean_probe_loss(
                model,
                prompt.prompt_index,
                anchor_probe,
                torch=torch,
                reporter=reporter,
                message=f"round_{round_index}_anchor_probe",
                interval=interval,
            )
            scale_loss = _SURFACE08.mean_probe_loss(
                model,
                prompt.prompt_index,
                scale_probe,
                torch=torch,
                reporter=reporter,
                message=f"round_{round_index}_scale_probe",
                interval=interval,
            )
            state["probe_curve"].append(
                {
                    "round": round_index,
                    "anchor_probe_loss": round(anchor_loss, 6),
                    "scale_probe_loss": round(scale_loss, 6),
                }
            )
            unique_rows = min(exposures_seen, SEMANTIC_ROWS)
            row = {
                "round": round_index,
                "optimizer_step": optimizer_steps,
                "exposures_seen": exposures_seen,
                "unique_semantic_rows": unique_rows,
                "train_loss": round(statistics.fmean(round_losses), 6),
                "synthetic_anchor_probe_loss": round(anchor_loss, 6),
                "synthetic_scale_probe_loss": round(scale_loss, 6),
            }
            marker = _SURFACE08._save_round_checkpoint(
                config,
                model,
                optimizer,
                torch,
                row=row,
            )
            state["round_rows"].append(marker)
            state.update(
                {
                    "optimizer_steps": optimizer_steps,
                    "exposures_seen": exposures_seen,
                    "gradient_norms": grad_norms,
                    "profile_counts": dict(profile_counts),
                    "voice_counts": dict(voice_counts),
                    "training_compute_wall_seconds": total_training_wall,
                }
            )
            write_json(_SURFACE08._training_state_path(config), state)
            del prompt, optimizer, model
            gc.collect()
            torch.cuda.empty_cache()
            controller = evaluate_controller_checkpoint(
                config,
                checkpoint_dir(config, round_index) / "model.local.nemo",
                round_index,
                validation_gpu,
            )
            state["controller_rows"].append(
                {**row, **controller, "checkpoint_sha256": marker["checkpoint_sha256"]}
            )
            controller_payload = mark_controller_rows(state["controller_rows"])
            write_json(run_dir(config) / "controller-dev" / "round-metrics.local.json", controller_payload)
            write_json(_SURFACE08._training_state_path(config), state)
            stopped_round = round_index
            reporter.progress(
                epoch=round_index,
                total_epochs=int(config["training"]["max_rounds"]),
                step=optimizer_steps,
                total_steps=int(config["training"]["max_optimizer_steps"]),
                message=(
                    f"ARTUR-dev WER={controller['wer']} CER={controller['cer']} "
                    f"empty={controller['empty']} OTF-fill={telemetry['otf_fill_rate']}"
                ),
            )
            stop = should_stop_controller_curve(state["controller_rows"])
            model, optimizer, state, initial_fingerprints, restored_round = (
                _SURFACE08._load_or_initialize_training(
                    config,
                    torch,
                    restore_reporter,
                )
            )
            if restored_round != round_index:
                raise RuntimeError("round changed while restoring after controller validation")
            prompt = derive_prompt_column_selection(model, "sl-SI")
            if stop["stop"]:
                stopped_reason = stop["reason"]
                break
    except Exception as exc:
        if type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower():
            _SURFACE08._record_training_oom(
                config,
                physical_microbatch=physical,
                optimizer_step=optimizer_steps,
                round_index=active_round,
            )
        reporter.failed("Surface08 scale-8000 OTF training failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()

    segment_wall = time.perf_counter() - segment_started
    state["wall_time_seconds"] = float(state.get("wall_time_seconds", 0.0)) + segment_wall
    state.update(
        {
            "optimizer_steps": optimizer_steps,
            "exposures_seen": exposures_seen,
            "gradient_norms": grad_norms,
            "profile_counts": dict(profile_counts),
            "voice_counts": dict(voice_counts),
            "training_compute_wall_seconds": total_training_wall,
        }
    )
    write_json(_SURFACE08._training_state_path(config), state)
    controller_payload = mark_controller_rows(state["controller_rows"])
    selected_round = int(controller_payload["selected_round"])
    after = _SURFACE08.model_fingerprints(model)
    integrity = _SURFACE08.fingerprint_integrity(initial_fingerprints, after)
    if not integrity["only_surface08_changed"]:
        raise RuntimeError("parameter-integrity failure: unauthorized tensor changed")
    initial_config_fingerprints = json.loads(
        (run_dir(config) / "initial-configuration-fingerprints.local.json").read_text(encoding="utf-8")
    )
    final_config_fingerprints = _SURFACE08.configuration_fingerprints(model)
    protected_configuration_unchanged = {
        name: initial_config_fingerprints.get(name) == final_config_fingerprints.get(name)
        for name in sorted(set(initial_config_fingerprints) | set(final_config_fingerprints))
    }
    if not all(protected_configuration_unchanged.values()):
        raise RuntimeError("protected model configuration changed")
    verify_protected_file_fingerprints(config, protected)
    diversity = diversity_statistics(
        pool,
        exposures_seen,
        profile_counts=dict(profile_counts),
    )
    payload = {
        "status": "PASSED",
        "surface_id": SURFACE_ID,
        "surface": state["surface"],
        "semantic_rows_available": SEMANTIC_ROWS,
        "sample_exposures": exposures_seen,
        "optimizer_steps": optimizer_steps,
        "stopped_round": stopped_round,
        "stopped_reason": stopped_reason,
        "selected_round": selected_round,
        "physical_microbatch": physical,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": 8,
        "learning_rates": config["training"]["learning_rates"],
        "probe_curve": state["probe_curve"],
        "controller_curve": controller_payload["rows"],
        "gradient_norm": {
            "min": min(grad_norms),
            "max": max(grad_norms),
            "final": grad_norms[-1],
        },
        "wall_time_seconds": state["wall_time_seconds"],
        "training_compute_wall_seconds": total_training_wall,
        "examples_per_second": exposures_seen / total_training_wall,
        "loader_telemetry": state["loader_telemetry"],
        "loader_fill_rate": round(
            1.0
            - sum(
                row["consumer_wait_percent"] / 100.0
                * (16_000 / row["examples_per_second"])
                for row in state["loader_telemetry"]
            )
            / sum(16_000 / row["examples_per_second"] for row in state["loader_telemetry"]),
            9,
        ),
        "data_diversity": diversity,
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "parameter_integrity": integrity,
        "protected_configuration_unchanged": protected_configuration_unchanged,
        "runtime": _SURFACE08.runtime_summary(hardware, torch),
        "runtime_identities": runtime_identities,
        "protected_file_fingerprints": protected,
        "per_round_checkpoints_retained": True,
    }
    write_json(run_dir(config) / "training-summary.local.json", payload)
    reporter.complete(
        "Surface08 scale-8000 OTF training and controller selection complete",
        step=optimizer_steps,
        total_steps=int(config["training"]["max_optimizer_steps"]),
    )
    print(
        json.dumps(
            {
                "status": "PASSED",
                "stopped_round": stopped_round,
                "selected_round": selected_round,
                "optimizer_steps": optimizer_steps,
                "loader_fill_rate": payload["loader_fill_rate"],
            },
            sort_keys=True,
        )
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def classify_result(
    metrics: dict[str, dict[str, Any]],
    *,
    training: dict[str, Any],
) -> str:
    if float(training["loader_fill_rate"]) < 0.95:
        return "SCALE8000_OTF_LOADER_THROUGHPUT_BLOCKED"
    if not bool(training["parameter_integrity"]["only_surface08_changed"]):
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    if any(int(metrics[split]["empty"]) > 0 for split in real_splits):
        return "SCALE8000_OTF_SURFACE08_REAL_REGRESSION"
    synthetic_safe = all(
        float(metrics[split][metric]) < float(BASE_DIRECTIONAL_METRICS[split][metric])
        for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout")
        for metric in ("wer", "cer")
    )
    comparisons = [
        (
            split,
            metric,
            float(metrics[split][metric]),
            float(SURFACE08_SCALE2000_METRICS[split][metric]),
            0.50 if metric == "wer" else 0.25,
        )
        for split in real_splits
        for metric in ("wer", "cer")
    ]
    improvements = sum(candidate < prior for _split, _metric, candidate, prior, _tol in comparisons)
    within = all(candidate <= prior + tolerance for _split, _metric, candidate, prior, tolerance in comparisons)
    if improvements >= 3 and within and synthetic_safe:
        return "SCALE8000_OTF_SURFACE08_NEW_BEST_DIRECTIONAL"
    if within and synthetic_safe and int(training["data_diversity"]["unique_semantic_rows_seen"]) > 16_000:
        return "SCALE8000_OTF_SURFACE08_MATCHES_SCALE2000_WITH_LINGUISTIC_GAIN"
    beats_base = all(
        float(metrics[split][metric]) < float(BASE_DIRECTIONAL_METRICS[split][metric])
        for split in real_splits
        for metric in ("wer", "cer")
    )
    if beats_base:
        return "SCALE8000_OTF_SURFACE08_BEATS_BASE_BUT_NOT_SCALE2000"
    return "SCALE8000_OTF_SURFACE08_REAL_REGRESSION"


def stage_evaluate_directional(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    hardware = require_single_visible_cuda()
    verify_runtime_identities(check_gpu=False)
    training = json.loads(
        (run_dir(config) / "training-summary.local.json").read_text(encoding="utf-8")
    )
    selected_round = int(training["selected_round"])
    checkpoint = checkpoint_dir(config, selected_round) / "model.local.nemo"
    marker = json.loads(
        (checkpoint_dir(config, selected_round) / "checkpoint-complete.local.json").read_text(
            encoding="utf-8"
        )
    )
    if file_sha256(checkpoint) != marker["checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint identity mismatch")
    extra_roots = [
        str(auxiliary_runs_root()),
        *[item for item in os.environ.get("SLAIF_ASR_EXTRA_RUNS_ROOTS", "").split(os.pathsep) if item],
    ]
    os.environ["SLAIF_ASR_EXTRA_RUNS_ROOTS"] = os.pathsep.join(dict.fromkeys(extra_roots))
    suite_config = json.loads(
        (REPO_ROOT / config["evaluation"]["suite_config"]).read_text(encoding="utf-8")
    )
    suite_records, split_records = load_directional_suite(suite_config)
    output_dir = run_dir(config) / "directional-evaluation"
    suite_manifest_sha = write_privacy_safe_suite_manifest(
        output_dir / "suite-plan.local.jsonl",
        suite_records,
    )
    env = runtime_environment()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": hardware.physical_selector,
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
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
    metric_table = {}
    split_summaries = {}
    for split, records in split_records.items():
        summary = {
            "rows": len(records),
            "audio_duration_seconds": round(sum(row.duration for row in records), 6),
            "metrics": metrics_for(records, by_split[split]),
        }
        split_summaries[split] = summary
        metric_table[split] = metric_row(summary)
    classification = classify_result(metric_table, training=training)
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
        },
        "splits": split_summaries,
        "metric_table": metric_table,
        "classification": classification,
        "accepted_parent": "none",
    }
    write_json(output_dir / "summary.local.json", payload)
    print(
        json.dumps(
            {
                "status": "PASSED",
                "selected_round": selected_round,
                "classification": classification,
                "metrics": metric_table,
            },
            sort_keys=True,
        )
    )
    return payload


def _format_metric(value: dict[str, Any]) -> str:
    return f"{value['wer']:.3f} / {value['cer']:.3f} / {int(value['empty'])}"


def _markdown_report(public: dict[str, Any]) -> str:
    training = public["training"]
    rows = public["controller_dev"]["curve"]
    metrics = public["directional_evaluation"]["metrics"]
    lines = [
        "# Experiment 0031: Surface08 Scale-8000 OTF Augmented",
        "",
        f"Classification: `{public['classification']}`",
        "",
        "This stacked diagnostic changes the training data axis while retaining the Surface08 model surface. It uses scale-8000 clean synthetic audio and deterministic in-memory transcript-preserving augmentation from PR #50 infrastructure. No augmented WAV was rendered or used.",
        "",
        "## Result",
        "",
        f"- Selected round: {training['selected_round']}; stopped round: {training['stopped_round']} (`{training['stopped_reason']}`).",
        f"- Training exposures: {training['sample_exposures']:,}; unique semantic rows: {training['data_diversity']['unique_semantic_rows_seen']:,}.",
        f"- OTF workers: 3 spawned processes; aggregate fill rate: {training['loader_fill_rate']:.6f}.",
        f"- Selected checkpoint SHA256: `{public['directional_evaluation']['selected_checkpoint_sha256']}`.",
        "- `accepted_parent` remains `none`; this result is noncanonical and promotion-ineligible.",
        "",
        "## Controller-Dev Curve",
        "",
        "| Round | Step | Exposures | Unique semantic rows | Train loss | Anchor | Scale | ARTUR-dev WER | CER | Empty | Eligible |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        train_loss = "NOT_APPLICABLE" if row["train_loss"] is None else row["train_loss"]
        lines.append(
            f"| {row['round']} | {row['optimizer_step']} | {row['exposures_seen']} | {row.get('unique_semantic_rows', 0)} | {train_loss} | {row['synthetic_anchor_probe_loss']} | {row['synthetic_scale_probe_loss']} | {row['wer']} | {row['cer']} | {row['empty']} | {str(row['eligible']).lower()} |"
        )
    lines.extend(
        [
            "",
            "ARTUR controller-dev aggregate metrics alone selected the checkpoint. Anchor and scale are inherited fixed synthetic probes and were not training sources.",
            "",
            "## Data Diversity",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    diversity = training["data_diversity"]
    for label, key in (
        ("Total virtual exposures seen", "total_virtual_exposures_seen"),
        ("Unique semantic rows seen", "unique_semantic_rows_seen"),
        ("Mean exposures per semantic row", "mean_exposures_per_semantic_row"),
        ("P95 exposures per semantic row", "p95_exposures_per_semantic_row"),
    ):
        lines.append(f"| {label} | {diversity[key]} |")
    lines.extend(
        [
            f"| Voice/style distribution | `{json.dumps(diversity['voice_distribution'], sort_keys=True)}` |",
            f"| Augmentation profile distribution | `{json.dumps(diversity['augmentation_profile_distribution'], sort_keys=True)}` |",
            "",
            "## Live OTF Loader Telemetry",
            "",
            "| Round | Examples/s | OTF fill rate | Consumer wait % | Queue p50/p95 | Worker count | Notes |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in training["loader_telemetry"]:
        lines.append(
            f"| {row['round']} | {row['examples_per_second']} | {row['otf_fill_rate']} | {row['consumer_wait_percent']} | {row['queue_p50']} / {row['queue_p95']} | {row['worker_count']} | in-memory only |"
        )
    lines.extend(
        [
            "",
            "## Directional Metrics",
            "",
            "| Split | Base | PR #36 | Surface07 | Surface08 scale-2000 | Surface08 scale-8000 OTF |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in (
        "piper_synthetic_holdout",
        "supertonic_heldout_voice_holdout",
        "fleurs_v2",
        "artur_j",
    ):
        lines.append(
            f"| {split} | {_format_metric(metrics['base'][split])} | {_format_metric(metrics['pr36'][split])} | {_format_metric(metrics['surface07'][split])} | {_format_metric(metrics['surface08_scale2000'][split])} | {_format_metric(metrics['surface08_scale8000_otf'][split])} |"
        )
    lines.extend(
        [
            "",
            "Values are normalized WER / CER / empty hypotheses. Directional batch-32 evaluation ran only after ARTUR-dev fixed the selected round.",
            "",
            "## Boundaries",
            "",
            "- No real speech, S6TTS, scale-2000 audio, or immutable gate was used for training.",
            "- No pre-rendered augmented WAV was used or written.",
            "- No checkpoint, model, audio, prediction, local manifest, raw reference, or hypothesis is committed.",
            "- No `TRAINING_ELIGIBLE`, checkpoint acceptance, accepted-parent change, or publication is issued.",
            "- The local scale-8000 manifests are restored identities; the historical committed manifest hashes remain separately recorded.",
            "",
        ]
    )
    return "\n".join(lines)


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    inputs = json.loads(
        (run_dir(config) / "verification" / "inputs.local.json").read_text(encoding="utf-8")
    )
    determinism = json.loads(
        (run_dir(config) / "verification" / "otf-determinism.local.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (run_dir(config) / "training-summary.local.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (run_dir(config) / "directional-evaluation" / "summary.local.json").read_text(encoding="utf-8")
    )
    selected_round = int(training["selected_round"])
    controller_rows = training["controller_curve"]
    if selected_round != int(evaluation["selected_round"]):
        raise RuntimeError("post-selection directional evaluation changed selected_round")
    candidate = evaluation["metric_table"]
    evaluation_suite = evaluation["suite"]
    public_evaluation_suite = {
        key: evaluation_suite[key]
        for key in (
            "rows",
            "prediction_count",
            "audio_duration_seconds",
            "wall_time_seconds",
            "real_time_factor",
            "rows_per_second",
            "audio_seconds_per_wall_second",
            "gpu_monitor",
        )
    }
    public = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "classification": evaluation["classification"],
        "status": "DIAGNOSTIC_ONLY",
        "accepted_parent": "none",
        "promotion_eligible": False,
        "training_eligible": False,
        "checkpoint_accepted": False,
        "model_published": False,
        "stacked_on": "PR #50 OTF augmentation infrastructure",
        "model": {
            "repository": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "base_checkpoint_sha256": config["model"]["checkpoint_sha256"],
            "nemo_revision": config["model"]["nemo_revision"],
            "initialization": "untouched_base",
        },
        "data": {
            "corpus_id": CORPUS_ID,
            "semantic_rows_available": SEMANTIC_ROWS,
            "clean_files": 576_000,
            "combined_text_sha256": config["data"]["combined_text_sha256"],
            "historical_manifest_identities": {
                "piper_sha256": config["data"]["piper_committed_manifest_sha256"],
                "supertonic_sha256": config["data"]["supertonic_committed_manifest_sha256"],
            },
            "restored_local_manifest_identities": {
                "piper_sha256": config["data"]["piper_restored_manifest_sha256"],
                "supertonic_sha256": config["data"]["supertonic_restored_manifest_sha256"],
            },
            "full_audio_integrity": inputs["local_scale8000"]["full_audio_integrity"],
            "diversity": training["data_diversity"],
        },
        "otf_augmentation": {
            "augmentation_key": AUGMENTATION_KEY,
            "algorithm_version": ALGORITHM_VERSION,
            "profiles": training["data_diversity"]["augmentation_profile_distribution"],
            "workers": 3,
            "prefetch_microbatches": 16,
            "disk_output": False,
            "determinism": determinism["checks"],
        },
        "surface": training["surface"],
        "training": {
            "physical_microbatch": training["physical_microbatch"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "effective_batch_size": training["effective_batch_size"],
            "learning_rates": training["learning_rates"],
            "sample_exposures": training["sample_exposures"],
            "optimizer_steps": training["optimizer_steps"],
            "stopped_round": training["stopped_round"],
            "stopped_reason": training["stopped_reason"],
            "selected_round": training["selected_round"],
            "wall_time_seconds": training["wall_time_seconds"],
            "training_compute_wall_seconds": training["training_compute_wall_seconds"],
            "examples_per_second": training["examples_per_second"],
            "loader_fill_rate": training["loader_fill_rate"],
            "loader_telemetry": training["loader_telemetry"],
            "data_diversity": training["data_diversity"],
            "peak_allocated_mib": training["peak_allocated_mib"],
            "peak_reserved_mib": training["peak_reserved_mib"],
            "runtime": training["runtime"],
            "parameter_integrity": training["parameter_integrity"],
            "protected_configuration_unchanged": training[
                "protected_configuration_unchanged"
            ],
        },
        "controller_dev": {
            "partition_id": "artur-controller-dev-v1",
            "policy": {
                "batch_size": 1,
                "duration_bucketing": False,
                "precision": "fp32",
                "tf32": False,
            },
            "curve": controller_rows,
            "selected_round": selected_round,
        },
        "directional_evaluation": {
            "policy": evaluation["policy"],
            "selected_checkpoint_sha256": evaluation["checkpoint_sha256"],
            "suite": public_evaluation_suite,
            "metrics": {
                "base": BASE_DIRECTIONAL_METRICS,
                "pr36": PR36_METRICS,
                "surface07": SURFACE07_METRICS,
                "surface08_scale2000": SURFACE08_SCALE2000_METRICS,
                "surface08_scale8000_otf": candidate,
            },
        },
        "safety": {
            "real_speech_used_for_training": False,
            "s6tts_used": False,
            "scale2000_used_as_training_source": False,
            "pre_rendered_augmented_wavs_used": False,
            "immutable_gate_used_for_selection": False,
            "generated_audio_committed": False,
            "checkpoint_or_model_committed": False,
            "predictions_committed": False,
            "local_manifest_committed": False,
            "training_eligible_issued": False,
            "checkpoint_accepted": False,
            "model_published": False,
        },
        "limitations": [
            "Directional batch-32 evaluation is noncanonical.",
            "The restored local manifests have new byte identities; historical committed identities and deterministic restoration evidence remain separate.",
            "The OTF profile schedule is deterministic but is intentionally not an exact replay of the old scale-2000 offline assignment.",
            "Synthetic-only training cannot accept a checkpoint.",
        ],
    }
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v5-scale8000-otf-surface08-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "corpus_id": CORPUS_ID,
        "combined_text_sha256": config["data"]["combined_text_sha256"],
        "semantic_rows_available": SEMANTIC_ROWS,
        "virtual_exposures_seen": training["sample_exposures"],
        "unique_semantic_rows_seen": training["data_diversity"]["unique_semantic_rows_seen"],
        "augmentation_key": AUGMENTATION_KEY,
        "augmentation_profile_config_sha256": config["otf_augmentation"][
            "profile_config_sha256"
        ],
        "surface_id": SURFACE_ID,
        "base_checkpoint_sha256": config["model"]["checkpoint_sha256"],
        "selected_round": selected_round,
        "selected_checkpoint_sha256": evaluation["checkpoint_sha256"],
        "classification": evaluation["classification"],
        "parameter_integrity_passed": training["parameter_integrity"][
            "only_surface08_changed"
        ],
        "otf_determinism_passed": determinism["checks"]["status"] == "PASSED",
        "otf_loader_fill_rate": training["loader_fill_rate"],
        "pre_rendered_augmented_audio_used": False,
        "generated_audio_committed": False,
        "local_manifest_committed": False,
        "checkpoint_committed": False,
        "accepted_parent": "none",
        "promotion_eligible": False,
        "training_eligible": False,
        "checkpoint_accepted": False,
        "model_published": False,
        "prohibited_statuses": ["TRAINING_ELIGIBLE"],
    }
    validate_public_report(public)
    validate_public_report(certificate)
    for relative, payload in ((REPORT_JSON, public), (CERTIFICATE_PATH, certificate)):
        path = REPO_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, payload)
    atomic_write_text(REPO_ROOT / REPORT_MD, _markdown_report(public))
    print(
        json.dumps(
            {
                "status": "PASSED",
                "classification": public["classification"],
                "selected_round": selected_round,
            },
            sort_keys=True,
        )
    )
    return public


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Surface08 on scale-8000 with deterministic OTF augmentation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "verify-inputs",
            "probe-hardware",
            "probe-surface",
            "probe-otf",
            "probe-microbatch",
            "train",
            "evaluate-directional",
            "summarize",
        ),
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
        "probe-otf": lambda: stage_probe_otf(config_path),
        "probe-microbatch": lambda: stage_probe_microbatch(config_path),
        "train": lambda: stage_train(config_path, args.progress_interval),
        "evaluate-directional": lambda: stage_evaluate_directional(config_path),
        "summarize": lambda: stage_summarize(config_path),
    }
    stages[args.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
