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
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from slaif_asr.batched_streaming import NvidiaSmiMonitor, make_batches, parse_monitor_csv
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_scoring import CHECKPOINT_SHA256, MODEL_REPOSITORY, MODEL_REVISION, NEMO_REVISION, verify_runtime_identities
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl
from slaif_asr.decoder_lm_adapter import (
    TemporaryLMHead,
    compare_pretrained_state,
    configure_text_only_trainable,
    disable_decoder_lm_adapter,
    enable_decoder_lm_adapter,
    enabled_decoder_lm_adapters,
    install_decoder_lm_adapter,
    load_decoder_lm_adapter_spec,
    load_text_only_artifact,
    pretrained_parameters_with_grad,
    save_text_only_artifact,
    state_dict_cpu,
    text_only_optimizer_parameters,
    verify_text_only_optimizer_scope,
)
from slaif_asr.live_progress import LiveProgressReporter, heartbeat_thread
from slaif_asr.text_only_decoder_lm import (
    ARM_NAME,
    BASE_DIRECTIONAL_METRICS,
    SCALE2000_DIRECTIONAL_METRICS,
    SCALE8000_CLEAN_DIRECTIONAL_METRICS,
    assert_text_only_public_report_safe,
    accepted_text_path,
    batch_order,
    classify_text_only,
    decoder_lm_forward_loss,
    deterministic_text_split,
    directional_suite,
    load_accepted_text_rows,
    load_config,
    local_path,
    make_lm_batch,
    metrics_from_predictions,
    perplexity,
    real_regression_burden,
    run_dir,
    split_token_counts,
    tokenize_split,
    tokenizer_special_id,
    tokenizer_vocab_size,
)


DEFAULT_CONFIG = Path("configs/experiments/text_only_decoder_lm_adapter_v1.json")
REPORT_JSON = Path("docs/experiments/0016-text-only-decoder-lm-adaptation.json")
REPORT_MD = Path("docs/experiments/0016-text-only-decoder-lm-adaptation.md")
CERTIFICATE_PATH = Path("docs/data-certificates/sl-text-only-decoder-lm-adaptation-diagnostic-v1.json")

_JOINT_PATH = Path(__file__).with_name("run_corpus_v2_joint_adapter_diagnostic.py")
_JOINT_SPEC = importlib.util.spec_from_file_location("_slaif_joint_runner_text_only", _JOINT_PATH)
if _JOINT_SPEC is None or _JOINT_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("cannot import joint-adapter runner")
_JOINT = importlib.util.module_from_spec(_JOINT_SPEC)
_JOINT_SPEC.loader.exec_module(_JOINT)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True).stdout.strip()


def file_sha256(path: Path) -> str:
    from slaif_asr.batched_streaming import file_sha256 as _file_sha256

    return _file_sha256(path)


def configure_torch() -> Any:
    torch = _JOINT.configure_torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def restore_base_model(config: dict[str, Any], *, reporter: LiveProgressReporter | None = None) -> Any:
    return _JOINT.restore_base_model(config, reporter=reporter)


def runtime_summary(torch: Any) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nemo_revision": NEMO_REVISION,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu": gpu_name,
        "logical_device": "cuda:0" if torch.cuda.is_available() else None,
        "precision": "fp32",
        "tf32": False,
    }


def verify_reference_reports(config: dict[str, Any]) -> dict[str, Any]:
    scale2000_path = REPO_ROOT / config["reference_reports"]["scale2000"]["path"]
    scale8000_path = REPO_ROOT / config["reference_reports"]["scale8000_clean"]["path"]
    scale8000 = read_json(scale8000_path)
    if scale8000["directional_evaluation"]["decision"]["classification"] != "SCALE8000_CLEAN_BEATS_BASE_BUT_NOT_SCALE2000":
        raise RuntimeError("Experiment 0015 classification mismatch")
    return {
        "scale2000_report": {"path": str(scale2000_path.relative_to(REPO_ROOT)), "sha256": file_sha256(scale2000_path)},
        "scale8000_clean_report": {"path": str(scale8000_path.relative_to(REPO_ROOT)), "sha256": file_sha256(scale8000_path)},
    }


def stage_verify_text(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    rows = load_accepted_text_rows(config)
    split = deterministic_text_split(rows, config["data"]["corpus_id"])
    payload = {
        "status": "PASSED",
        "corpus_id": config["data"]["corpus_id"],
        "fixed_text_sha256": file_sha256(accepted_text_path(config)),
        "rows": len(rows),
        "train_rows": len(split["train"]),
        "validation_rows": len(split["validation"]),
        "whole_file_decision": config["data"]["whole_file_decision"],
        "decision_id": config["data"]["decision_id"],
        "reference_reports": verify_reference_reports(config),
    }
    write_json(run_dir(config) / "verification" / "text.local.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _special_ids(tokenizer: Any, vocab_size: int) -> dict[str, int]:
    pad = tokenizer_special_id(tokenizer, ("pad_id", "pad"), 0)
    bos = tokenizer_special_id(tokenizer, ("bos_id", "bos"), pad)
    eos = tokenizer_special_id(tokenizer, ("eos_id", "eos"), vocab_size - 1)
    for name, value in {"pad_id": pad, "bos_id": bos, "eos_id": eos}.items():
        if value < 0 or value >= vocab_size:
            raise RuntimeError(f"{name}={value} is outside vocabulary size {vocab_size}")
    return {"pad_id": pad, "bos_id": bos, "eos_id": eos}


def _prepare_text_model(config: dict[str, Any], *, reporter: LiveProgressReporter | None = None):
    torch = configure_torch()
    model = restore_base_model(config, reporter=reporter)
    model.eval()
    spec = load_decoder_lm_adapter_spec(REPO_ROOT / "configs/experiments/text_only_decoder_lm_adapter_v1.json")
    adapter_summary = install_decoder_lm_adapter(model, spec)
    enable_decoder_lm_adapter(model)
    vocab_size = tokenizer_vocab_size(model.tokenizer)
    lm_head = TemporaryLMHead(adapter_summary["hidden_size"], vocab_size).to("cuda")
    trainable = configure_text_only_trainable(model, lm_head)
    return torch, model, lm_head, adapter_summary | trainable


def stage_probe_model_surface(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    reporter = LiveProgressReporter(stage="probe_model_surface", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "probe-model-surface.local.ndjson")
    reporter.start("probing text-only decoder LM surface")
    torch, model, lm_head, summary = _prepare_text_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "probe-restore.local.ndjson"))
    y = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long, device="cuda")
    disable_decoder_lm_adapter(model)
    with torch.no_grad():
        base, _ = model.decoder.predict(y, add_sos=False)
    enable_decoder_lm_adapter(model)
    with torch.no_grad():
        enabled, _ = model.decoder.predict(y, add_sos=False)
    disable_decoder_lm_adapter(model)
    with torch.no_grad():
        disabled, _ = model.decoder.predict(y, add_sos=False)
    parity = bool(torch.equal(base.detach().cpu(), enabled.detach().cpu()) and torch.equal(base.detach().cpu(), disabled.detach().cpu()))
    optimizer = torch.optim.AdamW(text_only_optimizer_parameters(model, lm_head), lr=0.001, weight_decay=0.01)
    verify_text_only_optimizer_scope(optimizer, model, lm_head)
    payload = {
        "status": "PASSED" if parity else "FAILED",
        "adapter": summary,
        "zero_effect_enabled_parity": parity,
        "disabled_parity": parity,
        "enabled_adapters_after_disable": enabled_decoder_lm_adapters(model),
        "tokenizer": {"vocabulary_size": tokenizer_vocab_size(model.tokenizer)},
        "runtime": runtime_summary(torch),
    }
    write_json(run_dir(config) / "verification" / "model-surface.local.json", payload)
    reporter.complete("model surface probe complete")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not parity:
        raise RuntimeError("decoder LM adapter zero-effect parity failed")
    return payload


def _evaluate_loss(model: Any, lm_head: Any, rows: Sequence[Any], *, batch_size: int, ids: dict[str, int], torch: Any) -> float:
    model.eval()
    losses = []
    weights = []
    with torch.no_grad():
        for batch_rows in batch_order(rows, epoch=0, seed=1234, batch_size=batch_size):
            batch = make_lm_batch(batch_rows, bos_id=ids["bos_id"], eos_id=ids["eos_id"], pad_id=ids["pad_id"], device="cuda")
            loss = decoder_lm_forward_loss(model, lm_head, batch, pad_id=ids["pad_id"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite text validation loss")
            token_count = int(batch["mask"].sum().detach().cpu())
            losses.append(float(loss.detach().cpu()) * token_count)
            weights.append(token_count)
    model.train()
    return sum(losses) / sum(weights)


def _select_microbatch(config: dict[str, Any], model: Any, lm_head: Any, tokenized: dict[str, list[Any]], ids: dict[str, int], torch: Any) -> dict[str, Any]:
    outcomes = {}
    for candidate in config["training"]["microbatch_candidates"]:
        try:
            batch_rows = tokenized["train"][: int(candidate)]
            batch = make_lm_batch(batch_rows, bos_id=ids["bos_id"], eos_id=ids["eos_id"], pad_id=ids["pad_id"], device="cuda")
            for parameter in text_only_optimizer_parameters(model, lm_head):
                parameter.grad = None
            loss = decoder_lm_forward_loss(model, lm_head, batch, pad_id=ids["pad_id"])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite microbatch loss")
            loss.backward()
            bad = pretrained_parameters_with_grad(model)
            if bad:
                raise RuntimeError(f"pretrained gradient detected: {bad[0]}")
            outcomes[str(candidate)] = {"status": "PASSED", "loss": round(float(loss.detach().cpu()), 6)}
            for parameter in text_only_optimizer_parameters(model, lm_head):
                parameter.grad = None
            return {"status": "PASSED", "physical_microbatch": int(candidate), "gradient_accumulation_steps": 128 // int(candidate), "candidate_outcomes": outcomes}
        except Exception as exc:
            outcomes[str(candidate)] = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc).splitlines()[0][:240]}
            torch.cuda.empty_cache()
    return {"status": "ENVIRONMENT_BLOCKED", "candidate_outcomes": outcomes}


def _lr_lambda(step: int, *, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, float(step + 1) / max(1, warmup_steps))
    progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math

    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def stage_train_primary(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    verify_runtime_identities(check_gpu=False)
    stage_verify_text(config_path)
    reporter = LiveProgressReporter(stage="train_primary", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "train-primary.local.ndjson")
    reporter.start("training text-only decoder LM adapter")
    torch, model, lm_head, adapter_summary = _prepare_text_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "train-restore.local.ndjson"))
    tokenizer = model.tokenizer
    rows = load_accepted_text_rows(config)
    split = deterministic_text_split(rows, config["data"]["corpus_id"])
    tokenized, token_stats = tokenize_split(split, tokenizer)
    if token_stats["rows_rejected_by_tokenization"]:
        raise RuntimeError("tokenization rejected accepted text rows")
    vocab_size = tokenizer_vocab_size(tokenizer)
    ids = _special_ids(tokenizer, vocab_size)
    micro = _select_microbatch(config, model, lm_head, tokenized, ids, torch)
    write_json(run_dir(config) / "verification" / "microbatch.local.json", micro)
    if micro["status"] != "PASSED":
        raise RuntimeError("no text-only microbatch candidate fit")
    physical_microbatch = int(micro["physical_microbatch"])
    accumulation = int(micro["gradient_accumulation_steps"])
    train_batches = len(tokenized["train"]) // int(config["training"]["effective_batch_size"])
    total_steps = train_batches * int(config["training"]["epochs"])
    warmup_steps = max(1, int(total_steps * float(config["training"]["warmup_fraction"])))
    optimizer = torch.optim.AdamW(text_only_optimizer_parameters(model, lm_head), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    verify_text_only_optimizer_scope(optimizer, model, lm_head)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: _lr_lambda(step, total_steps=total_steps, warmup_steps=warmup_steps))
    initial_state = state_dict_cpu(model)
    initial_train_loss = _evaluate_loss(model, lm_head, tokenized["train"], batch_size=physical_microbatch, ids=ids, torch=torch)
    initial_val_loss = _evaluate_loss(model, lm_head, tokenized["validation"], batch_size=physical_microbatch, ids=ids, torch=torch)
    curve = [{"epoch": 0, "train_loss": round(initial_train_loss, 6), "validation_loss": round(initial_val_loss, 6)}]
    grad_norms: list[float] = []
    adapter_norm_curve: list[dict[str, Any]] = []
    optimizer_steps = 0
    token_count = 0
    start = time.perf_counter()
    monitor_path = run_dir(config) / ARM_NAME / "gpu-monitor.local.csv"
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    monitor = NvidiaSmiMonitor(physical_gpu_index=physical_gpu, output_csv=monitor_path, interval_seconds=0.5)
    torch.cuda.reset_peak_memory_stats(0)
    monitor.start()
    try:
        model.train()
        lm_head.train()
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            batches = batch_order(tokenized["train"], epoch=epoch, seed=int(config["training"]["seed"]), batch_size=int(config["training"]["effective_batch_size"]))
            for effective_batch in batches:
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for start_index in range(0, len(effective_batch), physical_microbatch):
                    micro_rows = effective_batch[start_index : start_index + physical_microbatch]
                    batch = make_lm_batch(micro_rows, bos_id=ids["bos_id"], eos_id=ids["eos_id"], pad_id=ids["pad_id"], device="cuda")
                    loss = decoder_lm_forward_loss(model, lm_head, batch, pad_id=ids["pad_id"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("non-finite text-only training loss")
                    scale = len(micro_rows) / float(config["training"]["effective_batch_size"])
                    (loss * scale).backward()
                    step_loss += float(loss.detach().cpu()) * scale
                    token_count += int(batch["mask"].sum().detach().cpu())
                bad = pretrained_parameters_with_grad(model)
                if bad:
                    raise RuntimeError(f"pretrained parameter received gradient: {bad[0]}")
                grad_norm = float(torch.nn.utils.clip_grad_norm_(text_only_optimizer_parameters(model, lm_head), max_norm=float(config["training"]["gradient_clipping"])))
                if not math_is_finite(grad_norm):
                    raise RuntimeError("non-finite text-only gradient")
                optimizer.step()
                scheduler.step()
                optimizer_steps += 1
                grad_norms.append(grad_norm)
                if optimizer_steps % 500 == 0:
                    elapsed = time.perf_counter() - start
                    reporter.progress(
                        step=optimizer_steps,
                        total_steps=total_steps,
                        current_loss=round(step_loss, 6),
                        examples_per_second=round((optimizer_steps * int(config["training"]["effective_batch_size"])) / elapsed, 6) if elapsed else None,
                        message=f"epoch={epoch}",
                        cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
                        cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
                    )
            epoch_val = _evaluate_loss(model, lm_head, tokenized["validation"], batch_size=physical_microbatch, ids=ids, torch=torch)
            curve.append({"epoch": epoch, "validation_loss": round(epoch_val, 6)})
            adapter_norm = sum(float(torch.linalg.vector_norm(parameter.detach()).cpu()) for parameter in text_only_optimizer_parameters(model, lm_head))
            adapter_norm_curve.append({"epoch": epoch, "adapter_and_lm_head_norm": round(adapter_norm, 6)})
    except Exception as exc:
        reporter.failed(message="text-only training failed", error_type=type(exc).__name__)
        raise
    finally:
        monitor.stop()
    wall = time.perf_counter() - start
    final_train_loss = _evaluate_loss(model, lm_head, tokenized["train"], batch_size=physical_microbatch, ids=ids, torch=torch)
    final_val_loss = _evaluate_loss(model, lm_head, tokenized["validation"], batch_size=physical_microbatch, ids=ids, torch=torch)
    trained_state = state_dict_cpu(model)
    integrity = compare_pretrained_state(initial_state, trained_state)
    if not integrity["pretrained_tensors_unchanged"]:
        raise RuntimeError("pretrained tensor changed during text-only training")
    artifact_path = run_dir(config) / ARM_NAME / "artifacts" / "sl-si-decoder-lm-adapter-v1.pt"
    artifact_sha = save_text_only_artifact(
        artifact_path,
        model=model,
        lm_head=lm_head,
        metadata={
            "base_checkpoint_sha256": CHECKPOINT_SHA256,
            "nemo_revision": NEMO_REVISION,
            "experiment_config_sha256": file_sha256(config_path),
            "temporary_lm_head_retained_for_asr_inference": False,
            "tokenizer": {"vocabulary_size": vocab_size, **ids},
        },
    )
    payload = {
        "status": "PASSED",
        "arm": ARM_NAME,
        "epochs": int(config["training"]["epochs"]),
        "optimizer_steps": optimizer_steps,
        "effective_batch_size": int(config["training"]["effective_batch_size"]),
        "physical_microbatch": physical_microbatch,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": float(config["training"]["learning_rate"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "scheduler": config["training"]["scheduler"],
        "tokenizer": {"vocabulary_size": vocab_size, **ids, **token_stats, "token_counts": split_token_counts(tokenized)},
        "initial_train_loss": round(initial_train_loss, 6),
        "final_train_loss": round(final_train_loss, 6),
        "initial_train_perplexity": perplexity(initial_train_loss),
        "final_train_perplexity": perplexity(final_train_loss),
        "initial_validation_loss": round(initial_val_loss, 6),
        "final_validation_loss": round(final_val_loss, 6),
        "initial_validation_perplexity": perplexity(initial_val_loss),
        "final_validation_perplexity": perplexity(final_val_loss),
        "loss_curve": curve,
        "validation_loss_improved": final_val_loss < initial_val_loss,
        "token_count_processed": token_count,
        "tokens_per_second": round(token_count / wall, 6) if wall else None,
        "rows_per_second": round((optimizer_steps * int(config["training"]["effective_batch_size"])) / wall, 6) if wall else None,
        "wall_time_seconds": round(wall, 6),
        "gradient_norm": {"min": round(min(grad_norms), 6), "max": round(max(grad_norms), 6), "final": round(grad_norms[-1], 6)},
        "adapter_norm_curve": adapter_norm_curve,
        "adapter": adapter_summary,
        "pretrained_integrity": integrity,
        "artifact_sha256": artifact_sha,
        "runtime": runtime_summary(torch),
        "gpu_monitor": parse_monitor_csv(monitor_path),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 3),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024 / 1024, 3),
        "secondary_arm": "SECONDARY_ARM_SKIPPED_UNSUPPORTED",
    }
    write_json(run_dir(config) / ARM_NAME / "training-summary.local.json", payload)
    reporter.complete("text-only training complete")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def math_is_finite(value: float) -> bool:
    import math

    return math.isfinite(float(value))


def _prediction_text(item: Any) -> str:
    if hasattr(item, "text"):
        return str(item.text)
    return str(item)


def stage_evaluate_directional(config_path: Path, interval: float) -> dict[str, Any]:
    config = load_config(config_path)
    torch = configure_torch()
    artifact = run_dir(config) / ARM_NAME / "artifacts" / "sl-si-decoder-lm-adapter-v1.pt"
    if not artifact.exists():
        raise RuntimeError("text-only adapter artifact is missing; train-primary must run first")
    reporter = LiveProgressReporter(stage="evaluate_directional", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "evaluate-directional.local.ndjson")
    reporter.start("evaluating text-only decoder LM adapter")
    model = restore_base_model(config, reporter=LiveProgressReporter(stage="restore", arm=ARM_NAME, ndjson_path=run_dir(config) / "progress" / "eval-restore.local.ndjson"))
    adapter_summary = install_decoder_lm_adapter(model)
    lm_head = TemporaryLMHead(adapter_summary["hidden_size"], tokenizer_vocab_size(model.tokenizer)).to("cuda")
    load_text_only_artifact(artifact, model=model, lm_head=lm_head)
    enable_decoder_lm_adapter(model)
    if hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(att_context_size=config["evaluation"]["att_context_size"])
    suite, split_records = directional_suite(config)
    layout = make_batches(suite, batch_size=32, bucketed=True)
    predictions = {}
    started = time.perf_counter()
    for index, batch in enumerate(layout.batches, start=1):
        batch_manifest = run_dir(config) / "evaluation" / ARM_NAME / "directional-suite" / "batch-manifests" / f"batch-{index:04d}.local.jsonl"
        atomic_write_jsonl(
            batch_manifest,
            [
                {"audio_filepath": record.audio_filepath, "duration": record.duration, "text": "", "lang": "sl-SI"}
                for record in batch
            ],
        )
        with torch.no_grad():
            outputs = model.transcribe(audio=str(batch_manifest), batch_size=len(batch), target_lang="sl-SI", verbose=False)
        for record, output in zip(batch, outputs):
            predictions[record.sample_id] = _prediction_text(output)
        elapsed = time.perf_counter() - started
        reporter.progress(
            step=index,
            total_steps=layout.batch_count,
            examples_per_second=round(len(predictions) / elapsed, 6) if elapsed else None,
            audio_seconds_per_wall_second=round(sum(row.duration for row in suite[: len(predictions)]) / elapsed, 6) if elapsed else None,
            cuda_alloc_mib=round(torch.cuda.memory_allocated(0) / 1024 / 1024, 3),
            cuda_reserved_mib=round(torch.cuda.memory_reserved(0) / 1024 / 1024, 3),
        )
    wall = time.perf_counter() - started
    pred_path = run_dir(config) / "evaluation" / ARM_NAME / "directional-suite" / "predictions.local.jsonl"
    atomic_write_jsonl(pred_path, [{"sample_id": row.sample_id, "hypothesis": predictions[row.sample_id]} for row in sorted(suite, key=lambda item: item.original_index)])
    split_summaries, metric_table = metrics_from_predictions(suite, split_records, predictions)
    training = read_json(run_dir(config) / ARM_NAME / "training-summary.local.json")
    decision = classify_text_only(metric_table, text_validation_improved=bool(training["validation_loss_improved"]))
    payload = {
        "status": "PASSED",
        "policy": config["evaluation"],
        "suite_rows": len(suite),
        "suite_summary": {
            "wall_time_seconds": round(wall, 6),
            "audio_duration_seconds": round(sum(row.duration for row in suite), 6),
            "real_time_factor": round(wall / sum(row.duration for row in suite), 6),
            "utterances_per_second": round(len(suite) / wall, 6) if wall else None,
            "padding_ratio": layout.padding_ratio,
        },
        "adapter": {"artifact_sha256": file_sha256(artifact), **adapter_summary},
        "splits": split_summaries,
        "metrics": metric_table,
        "decision": decision,
        "reference_metrics": {
            "base": BASE_DIRECTIONAL_METRICS,
            "scale2000_augmented": SCALE2000_DIRECTIONAL_METRICS,
            "scale8000_clean_only": SCALE8000_CLEAN_DIRECTIONAL_METRICS,
        },
        "burdens": {
            "text_only": decision["real_regression_burden"],
            "scale2000_augmented": real_regression_burden(SCALE2000_DIRECTIONAL_METRICS),
            "scale8000_clean_only": real_regression_burden(SCALE8000_CLEAN_DIRECTIONAL_METRICS),
        },
        "runtime": runtime_summary(torch),
    }
    write_json(run_dir(config) / "evaluation" / ARM_NAME / "directional-summary.local.json", payload)
    reporter.complete("directional evaluation complete")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    training_path = run_dir(config) / ARM_NAME / "training-summary.local.json"
    evaluation_path = run_dir(config) / "evaluation" / ARM_NAME / "directional-summary.local.json"
    if not training_path.exists() or not evaluation_path.exists():
        raise RuntimeError("training and evaluation summaries are required before summarize")
    training = read_json(training_path)
    evaluation = read_json(evaluation_path)
    cert_payload = {
        "schema_version": "1.0",
        "certificate_id": "sl-text-only-decoder-lm-adaptation-diagnostic-v1",
        "status": "DIAGNOSTIC_ONLY",
        "work_order_id": "0029",
        "corpus_id": config["data"]["corpus_id"],
        "text_sha256": config["data"]["text_sha256"],
        "text_rows": config["data"]["text_rows"],
        "adapter_name": config["adapter"]["name"],
        "accepted_parent": "none",
        "training_eligible_issued": False,
        "authorized_actions": ["train text-only decoder LM adapter", "run directional batch-32 ASR evaluation", "write privacy-safe aggregate evidence"],
        "prohibited_actions": ["use audio for training", "use real-gate transcripts", "train pretrained tensors", "publish model or adapter", "issue TRAINING_ELIGIBLE"],
    }
    write_json(CERTIFICATE_PATH, cert_payload)
    payload = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "work_order_id": "0029",
        "status": "completed in PR; pending strategic review",
        "accepted_parent": "none",
        "promotion_eligible": False,
        "canonical": False,
        "repository_commit": git_head(),
        "authorization": {"certificate": str(CERTIFICATE_PATH), "status": "DIAGNOSTIC_ONLY"},
        "data": {
            "corpus_id": config["data"]["corpus_id"],
            "text_sha256": config["data"]["text_sha256"],
            "rows": config["data"]["text_rows"],
            "train_rows": config["data"]["train_rows"],
            "validation_rows": config["data"]["validation_rows"],
            "whole_file_decision": config["data"]["whole_file_decision"],
            "decision_id": config["data"]["decision_id"],
        },
        "model": config["model"],
        "adapter": {
            "name": config["adapter"]["name"],
            "location": config["adapter"]["location"],
            "type": config["adapter"]["type"],
            "bottleneck_dimension": config["adapter"]["bottleneck_dimension"],
            "trainable_parameters": training["adapter"]["adapter_trainable_parameters"],
            "temporary_lm_head": "training_only_excluded_from_asr_inference",
        },
        "training": {
            key: training[key]
            for key in (
                "epochs",
                "optimizer_steps",
                "effective_batch_size",
                "physical_microbatch",
                "gradient_accumulation_steps",
                "learning_rate",
                "weight_decay",
                "scheduler",
                "initial_train_loss",
                "final_train_loss",
                "initial_train_perplexity",
                "final_train_perplexity",
                "initial_validation_loss",
                "final_validation_loss",
                "initial_validation_perplexity",
                "final_validation_perplexity",
                "wall_time_seconds",
                "tokens_per_second",
                "rows_per_second",
                "peak_allocated_mib",
                "peak_reserved_mib",
            )
        },
        "tokenizer": training["tokenizer"],
        "parameter_integrity": training["pretrained_integrity"],
        "directional_evaluation": {
            "policy": evaluation["policy"],
            "suite_summary": evaluation["suite_summary"],
            "metrics": {
                "base": BASE_DIRECTIONAL_METRICS,
                "scale2000_augmented": SCALE2000_DIRECTIONAL_METRICS,
                "scale8000_clean_only": SCALE8000_CLEAN_DIRECTIONAL_METRICS,
                "text_only_decoder_lm": evaluation["metrics"],
            },
            "burdens": evaluation["burdens"],
            "decision": evaluation["decision"],
        },
        "secondary_arm": training["secondary_arm"],
        "limitations": [
            "Text-only diagnostic adaptation used accepted synthetic text but no acoustic learning.",
            "Directional batch-32 ASR evaluation is noncanonical and promotion-ineligible.",
            "No batch-1 canonical evaluation was run.",
            "No model, adapter, checkpoint, raw text, tokenized text, or predictions are committed.",
        ],
        "safety": {
            "training_eligible_issued": False,
            "accepted_parent": "none",
            "audio_training_used": False,
            "real_gate_transcripts_used_for_training": False,
            "pretrained_tensors_changed": not training["pretrained_integrity"]["pretrained_tensors_unchanged"],
        },
    }
    assert_text_only_public_report_safe(payload)
    write_json(REPORT_JSON, payload)
    lines = [
        "# Experiment 0016: Text-only Decoder-LM Adaptation",
        "",
        "Classification: `" + evaluation["decision"]["classification"] + "`",
        "",
        "This diagnostic trains only a decoder-side residual LM adapter with a temporary next-token LM head. No audio, TTS, real-gate transcripts, RNNT loss, or acoustic encoder training was used.",
        "",
        "## Data",
        f"- Corpus: `{config['data']['corpus_id']}`",
        f"- Text SHA256: `{config['data']['text_sha256']}`",
        f"- Train/validation rows: {config['data']['train_rows']} / {config['data']['validation_rows']}",
        "",
        "## Training",
        f"- Adapter: `{config['adapter']['name']}`",
        f"- Adapter trainable parameters: {training['adapter']['adapter_trainable_parameters']}",
        f"- Temporary LM head: training only; excluded from ASR inference",
        f"- Initial/final validation loss: {training['initial_validation_loss']} / {training['final_validation_loss']}",
        f"- Initial/final validation perplexity: {training['initial_validation_perplexity']} / {training['final_validation_perplexity']}",
        "",
        "## Directional Metrics",
        "| Split | Base WER/CER | Scale-2000 WER/CER | Scale-8000 clean WER/CER | Text-only WER/CER | Empty base/scale2000/scale8000/text |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout", "fleurs_v2", "artur_j"):
        base = BASE_DIRECTIONAL_METRICS[split]
        s2 = SCALE2000_DIRECTIONAL_METRICS[split]
        s8 = SCALE8000_CLEAN_DIRECTIONAL_METRICS[split]
        text = evaluation["metrics"][split]
        lines.append(
            f"| {split} | {base['wer']}/{base['cer']} | {s2['wer']}/{s2['cer']} | {s8['wer']}/{s8['cer']} | {text['wer']}/{text['cer']} | {base['empty']}/{s2['empty']}/{s8['empty']}/{text['empty']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            f"- Real-regression burden: {evaluation['decision']['real_regression_burden']}",
            f"- Accepted parent: `none`",
            "",
            "This report is diagnostic-only. Canonical batch-1 evaluation would be required before any acceptance discussion.",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run text-only Slovenian decoder-LM adaptation diagnostic.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", required=True, choices=["verify-text", "probe-model-surface", "train-primary", "evaluate-directional", "summarize"])
    parser.add_argument("--progress-interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.stage == "verify-text":
        stage_verify_text(args.config)
    elif args.stage == "probe-model-surface":
        stage_probe_model_surface(args.config)
    elif args.stage == "train-primary":
        stage_train_primary(args.config, args.progress_interval_seconds)
    elif args.stage == "evaluate-directional":
        stage_evaluate_directional(args.config, args.progress_interval_seconds)
    elif args.stage == "summarize":
        stage_summarize(args.config)


if __name__ == "__main__":
    main()
