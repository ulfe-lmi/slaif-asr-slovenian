#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.config import load_runtime_config
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.metrics import CorpusMetricSummary, corpus_metric_summary
from slaif_asr.real_eval import normalize_sl_asr_text, summarize_predictions, validate_gate_manifest
from slaif_asr.slovenian_adapter import (
    ResidualIntegrityReport,
    adapter_state_hashes,
    changed_adapter_tensors,
    compare_base_hashes,
    count_changed_elements,
    install_slovenian_residual_adapter,
    load_adapter_artifact,
    original_state_dict_from_wrapped_model,
    state_hashes,
    trainable_adapter_parameters,
    write_adapter_artifact,
    write_residual_integrity_report,
)
from slaif_asr.slovenian_curriculum import assert_training_disjoint, file_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_training_helpers() -> Any:
    helper_path = Path(__file__).resolve().parent / "train_prompt_column.py"
    spec = importlib.util.spec_from_file_location("slaif_train_prompt_column_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load training helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_streaming_helpers() -> Any:
    helper_path = (
        Path(__file__).resolve().parents[1]
        / ".external"
        / "NeMo"
        / "examples"
        / "asr"
        / "asr_cache_aware_streaming"
        / "speech_to_text_cache_aware_streaming_infer.py"
    )
    spec = importlib.util.spec_from_file_location("slaif_streaming_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load streaming helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING_HELPERS = load_training_helpers()
make_batch = TRAINING_HELPERS.make_batch
rnnt_loss = TRAINING_HELPERS.rnnt_loss
transcribe_extract = TRAINING_HELPERS.extract_transcript


def atomic_write_text(path: Path, text: str) -> None:
    from slaif_asr.prompt_experiment import atomic_write_text as write_text

    write_text(path, text)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def run_nvidia_smi(path: Path) -> None:
    completed = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout, encoding="utf-8")


def git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def verify_hash(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")
    return actual


def resolve_transferred_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path.resolve()
    parts = path.parts
    if "slaif-asr-slovenian" in parts:
        index = parts.index("slaif-asr-slovenian")
        candidate = REPO_ROOT.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate.resolve()
    if "runs" in parts:
        index = parts.index("runs")
        candidate = REPO_ROOT.joinpath(*parts[index:])
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(path_text)


def localize_manifest(source: Path, destination: Path) -> str:
    rows = []
    for row in read_jsonl(source):
        row = dict(row)
        if "audio_filepath" in row:
            row["audio_filepath"] = str(resolve_transferred_path(str(row["audio_filepath"])))
        rows.append(row)
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(destination, text)
    return file_sha256(destination)


def localized_manifest_path(config: dict[str, Any], key: str) -> Path:
    name = {
        "selected_training_manifest": "selected-training.local.jsonl",
        "synthetic_holdout_manifest": "synthetic-holdout.local.jsonl",
        "fleurs_manifest": "fleurs.local.jsonl",
        "artur_j_manifest": "artur-j.local.jsonl",
    }[key]
    local = Path(config["paths"]["run_dir"]) / "manifests" / name
    return local if local.exists() else Path(config["paths"][key])


def verify_data_integrity(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    hashes = config["data_hashes"]
    actual = {
        "candidate_pool_sha256": verify_hash(Path(paths["candidate_pool_jsonl"]), hashes["candidate_pool_sha256"], "candidate pool"),
        "synthetic_holdout_sha256": verify_hash(
            Path(paths["synthetic_holdout_jsonl"]), hashes["synthetic_holdout_sha256"], "synthetic holdout"
        ),
        "training_manifest_sha256": verify_hash(
            Path(paths["selected_training_manifest"]), hashes["selected_training_manifest_sha256"], "selected training manifest"
        ),
        "fleurs_manifest_sha256": verify_hash(Path(paths["fleurs_manifest"]), hashes["fleurs_manifest_sha256"], "FLEURS manifest"),
        "artur_j_manifest_sha256": verify_hash(
            Path(paths["artur_j_manifest"]), hashes["artur_j_manifest_sha256"], "ARTUR-J manifest"
        ),
    }
    local_manifest_dir = Path(paths["run_dir"]) / "manifests"
    local_hashes = {
        "selected_training_manifest_local_sha256": localize_manifest(
            Path(paths["selected_training_manifest"]), local_manifest_dir / "selected-training.local.jsonl"
        ),
        "synthetic_holdout_manifest_local_sha256": localize_manifest(
            Path(paths["synthetic_holdout_manifest"]), local_manifest_dir / "synthetic-holdout.local.jsonl"
        ),
        "fleurs_manifest_local_sha256": localize_manifest(Path(paths["fleurs_manifest"]), local_manifest_dir / "fleurs.local.jsonl"),
        "artur_j_manifest_local_sha256": localize_manifest(Path(paths["artur_j_manifest"]), local_manifest_dir / "artur-j.local.jsonl"),
    }
    training_rows = read_jsonl(localized_manifest_path(config, "selected_training_manifest"))
    holdout_rows = read_jsonl(localized_manifest_path(config, "synthetic_holdout_manifest"))
    fleurs_rows = validate_gate_manifest(localized_manifest_path(config, "fleurs_manifest"))
    artur_rows = validate_gate_manifest(localized_manifest_path(config, "artur_j_manifest"))
    training_ids = {str(row["sample_id"]) for row in training_rows}
    holdout_ids = {str(row["sample_id"]) for row in holdout_rows}
    fleurs_ids = {str(row["sample_id"]) for row in fleurs_rows}
    artur_ids = {str(row["sample_id"]) for row in artur_rows}
    assert_training_disjoint(training_ids, holdout_ids=holdout_ids, real_gate_ids=fleurs_ids | artur_ids)
    actual.update(local_hashes)
    actual["training_rows"] = len(training_rows)
    actual["holdout_rows"] = len(holdout_rows)
    actual["fleurs_rows"] = len(fleurs_rows)
    actual["artur_j_rows"] = len(artur_rows)
    actual["training_unique_sample_ids"] = len(training_ids)
    actual["holdout_unique_sample_ids"] = len(holdout_ids)
    actual["fleurs_unique_sample_ids"] = len(fleurs_ids)
    actual["artur_j_unique_sample_ids"] = len(artur_ids)
    return actual


def restore_base_model(checkpoint: Path) -> Any:
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0").cuda().eval()
    if hasattr(model, "spec_augmentation"):
        model.spec_augmentation = None
    return model


def prompt_dictionary(model: Any) -> dict[str, int]:
    from slaif_asr.prompt_column import prompt_dictionary as get_prompt_dictionary

    return get_prompt_dictionary(model)


def tokenizer_identity(model: Any) -> dict[str, Any]:
    tokenizer = getattr(model, "tokenizer", None)
    return {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}" if tokenizer is not None else None,
        "vocab_size": int(getattr(tokenizer, "vocab_size", -1)),
        "prompt_dictionary": prompt_dictionary(model),
    }


def adapter_probe_results(adapter: Any, selection: Any, *, input_features: int, device: str) -> dict[str, bool]:
    import torch

    with torch.no_grad():
        torch.manual_seed(17)
        probe = torch.randn(2, 3, input_features, device=device)
        sl_probe = probe.clone()
        sl_probe[..., selection.selected_column] = 1.0
        other_probe = probe.clone()
        other_probe[..., selection.selected_column] = 0.0
        base_sl = adapter.base_prompt_kernel(sl_probe)
        wrapped_sl = adapter(sl_probe)
        base_other = adapter.base_prompt_kernel(other_probe)
        wrapped_other = adapter(other_probe)
    return {
        "step_zero_equivalent": bool(torch.allclose(base_sl, wrapped_sl, atol=1e-7, rtol=1e-7)),
        "non_sl_residual_zero": bool(torch.equal(base_other, wrapped_other)),
    }


def load_training_records(manifest: Path) -> list[Any]:
    from slaif_asr.prompt_experiment import ManifestRecord
    from slaif_asr.tts import validate_wav

    records = []
    for row in read_jsonl(manifest):
        audio_path = Path(row["audio_filepath"]).resolve()
        info = validate_wav(audio_path, sample_rate=16000)
        records.append(
            ManifestRecord(
                sample_id=str(row["sample_id"]),
                audio_filepath=audio_path,
                duration=round(info.duration_seconds, 6),
                text=str(row["text"]),
                lang="sl-SI",
                target_lang="sl-SI",
                partition_role=str(row.get("partition_role", "synthetic_candidate")),
                source_type=str(row.get("source_type", "synthetic_tts")),
            )
        )
    return records


def train_attempt(
    *,
    model: Any,
    adapter: Any,
    selection: Any,
    records: list[Any],
    learning_rate: float,
    max_steps: int,
    fallback_check_step: int,
    fallback_min_reduction_percent: float,
    allow_fallback_check: bool,
    seed: int,
    log_every_steps: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    optimizer = torch.optim.AdamW(trainable_adapter_parameters(adapter, weight_decay=0), lr=learning_rate, weight_decay=0)
    batches = [make_batch(model, record, "cuda") for record in records]
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    losses: list[float] = []
    logs: list[dict[str, Any]] = []
    stopped_reason = "completed max_optimizer_steps"
    fallback_required = False

    for step in range(1, max_steps + 1):
        index = order[(step - 1) % len(order)]
        record = records[index]
        batch = batches[index]
        optimizer.zero_grad(set_to_none=True)
        loss = rnnt_loss(model, batch, selection.prompt_index, "32")
        if not torch.isfinite(loss):
            stopped_reason = "non-finite loss"
            break
        loss.backward()
        grad_norm = 0.0
        total_grad_sq = 0.0
        for parameter in adapter.adapter_parameters():
            if parameter.grad is not None:
                total_grad_sq += float(torch.sum(parameter.grad.detach() ** 2).cpu())
        grad_norm = total_grad_sq ** 0.5
        total_delta_sq = sum(float(torch.sum(parameter.detach() ** 2).cpu()) for parameter in adapter.adapter_parameters())
        delta_norm = total_delta_sq ** 0.5
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % log_every_steps == 0 or step == max_steps:
            logs.append(
                {
                    "step": step,
                    "candidate_id": record.sample_id,
                    "loss": loss_value,
                    "learning_rate": learning_rate,
                    "gradient_norm": grad_norm,
                    "adapter_norm": delta_norm,
                    "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
                    "nan_or_overflow": False,
                }
            )
        if allow_fallback_check and step == fallback_check_step and losses:
            initial = losses[0]
            reduction = 0.0 if initial == 0 else (initial - losses[-1]) / initial * 100.0
            if reduction < fallback_min_reduction_percent:
                fallback_required = True
                stopped_reason = (
                    f"fallback required: {reduction:.3f}% loss reduction at step {fallback_check_step} "
                    f"< {fallback_min_reduction_percent:.3f}%"
                )
                break

    initial_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None
    reduction = None if initial_loss in (None, 0) else (initial_loss - (final_loss or initial_loss)) / initial_loss * 100.0
    return {
        "learning_rate": learning_rate,
        "steps": len(losses),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_percent": None if reduction is None else round(reduction, 3),
        "wall_time_seconds": round(time.perf_counter() - start, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
        "stopped_reason": stopped_reason,
        "fallback_required": fallback_required,
        "logs": logs,
    }


def train_rank(config: dict[str, Any], rank: int, gpu: dict[str, Any]) -> dict[str, Any]:
    import torch

    runtime = load_runtime_config()
    paths = config["paths"]
    training = config["training"]
    run_dir = Path(paths["run_dir"]) / f"rank-{rank}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_nvidia_smi(run_dir / "nvidia-smi-before.txt")
    checkpoint = Path(paths["checkpoint"]).resolve()
    records = load_training_records(localized_manifest_path(config, "selected_training_manifest"))

    def run_once(learning_rate: float, *, allow_fallback_check: bool) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        model = restore_base_model(checkpoint)
        model.eval()
        base_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        base_hashes = state_hashes(base_state)
        tokenizer_before = tokenizer_identity(model)
        selection, adapter = install_slovenian_residual_adapter(model, rank=rank, prompt_name=training["target_prompt"])
        probes = adapter_probe_results(
            adapter,
            selection,
            input_features=selection.encoder_width + selection.num_prompts,
            device="cuda",
        )
        initial_adapter_hashes = adapter_state_hashes(adapter)
        attempt = train_attempt(
            model=model,
            adapter=adapter,
            selection=selection,
            records=records,
            learning_rate=learning_rate,
            max_steps=int(training["max_optimizer_steps"]),
            fallback_check_step=int(training["fallback_check_step"]),
            fallback_min_reduction_percent=float(training["fallback_min_loss_reduction_percent"]),
            allow_fallback_check=allow_fallback_check,
            seed=int(training["seed"]),
            log_every_steps=int(training["log_every_steps"]),
        )
        return model, selection, adapter, base_state, base_hashes, tokenizer_before, probes | {"attempt": attempt, "initial_adapter_hashes": initial_adapter_hashes}

    model, selection, adapter, base_state, base_hashes, tokenizer_before, probe_payload = run_once(
        float(training["learning_rate"]),
        allow_fallback_check=True,
    )
    attempts = [probe_payload["attempt"]]
    if attempts[-1]["fallback_required"]:
        del model, adapter
        gc.collect()
        torch.cuda.empty_cache()
        model, selection, adapter, base_state, base_hashes, tokenizer_before, probe_payload = run_once(
            float(training["fallback_learning_rate"]),
            allow_fallback_check=False,
        )
        attempts.append(probe_payload["attempt"])

    current_state = original_state_dict_from_wrapped_model(model)
    current_hashes = state_hashes(current_state)
    unexpected, missing, changed = compare_base_hashes(base_hashes, current_hashes)
    tokenizer_after = tokenizer_identity(model)
    final_adapter_hashes = adapter_state_hashes(adapter)
    adapter_changes = changed_adapter_tensors(probe_payload["initial_adapter_hashes"], final_adapter_hashes)
    non_sl_probe = adapter_probe_results(
        adapter,
        selection,
        input_features=selection.encoder_width + selection.num_prompts,
        device="cuda",
    )
    optimizer_count = sum(parameter.numel() for parameter in trainable_adapter_parameters(adapter, weight_decay=0))
    integrity = ResidualIntegrityReport(
        selected_prompt=selection.prompt_name,
        prompt_index=selection.prompt_index,
        selected_column=selection.selected_column,
        rank=rank,
        trainable_parameters=selection.trainable_parameters,
        base_tensors_identical=not unexpected and not missing and not changed,
        prompt_kernel_identical=not any(name.startswith("prompt_kernel.") for name in changed),
        encoder_identical=not any(name.startswith("encoder.") for name in changed),
        decoder_joint_identical=not any(name.startswith(("decoder.", "joint.")) for name in changed),
        tokenizer_config_identical=tokenizer_before == tokenizer_after,
        changed_adapter_tensors=adapter_changes,
        unexpected_base_tensors=unexpected,
        missing_base_tensors=missing,
        unexpected_changed_base_tensors=changed,
        unexpected_changed_elements=count_changed_elements(base_state, current_state, changed),
        prompt_dictionary_unchanged=tokenizer_before["prompt_dictionary"] == tokenizer_after["prompt_dictionary"],
        step_zero_equivalent=bool(probe_payload["step_zero_equivalent"]),
        non_sl_residual_zero=bool(non_sl_probe["non_sl_residual_zero"]),
        optimizer_parameter_count=optimizer_count,
    )
    integrity_path = run_dir / "integrity-report.json"
    write_residual_integrity_report(integrity, integrity_path)
    artifact_path = run_dir / f"sl-si-residual-adapter-rank-{rank}.pt"
    write_adapter_artifact(
        artifact_path,
        adapter=adapter,
        selection=selection,
        metadata={
            "base_model_repository": runtime["base_model"]["repository"],
            "base_model_revision": runtime["base_model"]["revision"],
            "base_checkpoint_sha256": runtime["base_model"]["sha256"],
            "nemo_revision": runtime["nemo"]["revision"],
            "training_config": training,
            "data_hashes": config["data_hashes"],
            "gpu": gpu,
            "repository_commit": git_head(),
        },
    )
    run_nvidia_smi(run_dir / "nvidia-smi-after.txt")
    final_attempt = attempts[-1]
    summary = {
        "schema_version": "1.0",
        "rank": rank,
        "selection": asdict(selection),
        "trainable_parameters": selection.trainable_parameters,
        "attempts": attempts,
        "learning_rate": final_attempt["learning_rate"],
        "steps": final_attempt["steps"],
        "initial_loss": final_attempt["initial_loss"],
        "final_loss": final_attempt["final_loss"],
        "peak_vram_mib": final_attempt["peak_vram_mib"],
        "wall_time_seconds": final_attempt["wall_time_seconds"],
        "integrity_passed": integrity.passed(),
        "integrity_report": str(integrity_path),
        "adapter_artifact": str(artifact_path),
        "gpu": gpu,
    }
    atomic_write_json(run_dir / "training-summary.json", summary)
    del model, adapter, base_state, current_state
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def prepare_model_for_streaming(model: Any, *, context: list[int]) -> Any:
    import torch
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

    if hasattr(model.encoder, "set_default_att_context_size"):
        model.encoder.set_default_att_context_size(att_context_size=context)
    if hasattr(model, "change_decoding_strategy") and hasattr(model, "joint"):
        model.change_decoding_strategy(RNNTDecodingConfig(fused_batch_size=-1))
    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt("sl-SI")
        model.decoding.set_strip_lang_tags(True, lang_tag_pattern=None)
    model = model.to(device=torch.device("cuda:0"), dtype=torch.float32)
    model.eval()
    return model


def streaming_transcribe_manifest(model: Any, manifest: Path, output_dir: Path, *, context: list[int]) -> dict[str, Any]:
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    helpers = load_streaming_helpers()
    rows = read_jsonl(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = prepare_model_for_streaming(model, context=context)
    buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False, pad_and_drop_preencoded=False)
    predictions: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        buffer.append_audio_file(row["audio_filepath"], stream_id=-1)
        streaming, _ = helpers.perform_streaming(
            asr_model=model,
            streaming_buffer=buffer,
            compute_dtype=torch.float32,
            compare_vs_offline=False,
            debug_mode=False,
            pad_and_drop_preencoded=False,
        )
        buffer.reset_buffer()
        hypothesis = str(streaming[0]) if streaming else ""
        predictions.append(
            {
                "sample_id": row.get("sample_id") or row.get("candidate_id") or str(index),
                "reference": str(row.get("text", "")),
                "hypothesis": hypothesis,
                "pipeline_status": "PASSED",
                "empty_hypothesis": not hypothesis.strip(),
            }
        )
    wall = time.perf_counter() - start
    output_path = output_dir / "predictions.local.jsonl"
    atomic_write_text(
        output_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions),
    )
    summary = summarize_predictions(predictions)
    payload = {
        "manifest": str(manifest),
        "rows": len(rows),
        "context": context,
        "wall_time_seconds": round(wall, 3),
        "audio_duration_seconds": round(sum(float(row.get("duration", 0.0)) for row in rows), 3),
        "real_time_factor": None,
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
        "predictions": str(output_path),
        "summary": summary,
    }
    if payload["audio_duration_seconds"]:
        payload["real_time_factor"] = round(wall / payload["audio_duration_seconds"], 4)
    atomic_write_json(output_dir / "summary.json", payload)
    return payload


def install_adapter_from_artifact(model: Any, artifact: Path) -> tuple[Any, Any]:
    payload = load_adapter_artifact(artifact)
    selection = payload["selection"]
    rank = int(selection["rank"])
    derived, adapter = install_slovenian_residual_adapter(model, rank=rank, prompt_name=selection["prompt_name"])
    if asdict(derived) != selection:
        raise RuntimeError(f"adapter artifact selection does not match live model: {asdict(derived)} != {selection}")
    adapter.load_state_dict(payload["adapter_state_dict"], strict=False)
    return derived, adapter


def evaluate_model(
    *,
    config: dict[str, Any],
    model_label: str,
    adapter_artifact: Path | None,
    checkpoint: Path,
) -> dict[str, Any]:
    import torch

    root = Path(config["paths"]["run_dir"]) / "evaluation" / model_label
    run_nvidia_smi(root / "nvidia-smi-before.txt")
    model = restore_base_model(checkpoint)
    if adapter_artifact is not None:
        install_adapter_from_artifact(model, adapter_artifact)
    manifests = {
        "selected_synthetic_training": localized_manifest_path(config, "selected_training_manifest"),
        "fixed_synthetic_holdout": localized_manifest_path(config, "synthetic_holdout_manifest"),
        "fleurs": localized_manifest_path(config, "fleurs_manifest"),
        "artur_j": localized_manifest_path(config, "artur_j_manifest"),
    }
    results: dict[str, Any] = {}
    for split, manifest in manifests.items():
        results[split] = streaming_transcribe_manifest(
            model,
            manifest,
            root / split,
            context=list(config["evaluation"]["att_context_size"]),
        )
    run_nvidia_smi(root / "nvidia-smi-after.txt")
    peak = max(result["peak_vram_mib"] for result in results.values())
    payload = {"schema_version": "1.0", "model_label": model_label, "peak_vram_mib": peak, "splits": results}
    atomic_write_json(root / "model-summary.json", payload)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return payload


def metric(summary: dict[str, Any], split: str, model_label: str) -> CorpusMetricSummary:
    return CorpusMetricSummary(**summary[model_label]["splits"][split]["summary"]["normalized"])


def relative_improvement(base: float, challenger: float) -> float:
    return 0.0 if base == 0 else (base - challenger) / base * 100.0


def classify_adapter(
    *,
    integrity_passed: bool,
    synthetic_base: CorpusMetricSummary,
    synthetic_adapter: CorpusMetricSummary,
    fleurs_base: CorpusMetricSummary,
    fleurs_adapter: CorpusMetricSummary,
    artur_base: CorpusMetricSummary,
    artur_adapter: CorpusMetricSummary,
    thresholds: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not integrity_passed:
        return "EXPERIMENT_INVALID", ["parameter integrity failed"]
    syn_wer = relative_improvement(synthetic_base.corpus_wer, synthetic_adapter.corpus_wer)
    syn_cer = relative_improvement(synthetic_base.corpus_cer, synthetic_adapter.corpus_cer)
    synthetic_ok = syn_wer >= thresholds["synthetic_holdout_relative_wer_or_cer_improvement_percent"] or syn_cer >= thresholds[
        "synthetic_holdout_relative_wer_or_cer_improvement_percent"
    ]
    if not synthetic_ok:
        reasons.append("synthetic holdout improvement below threshold")
    real_regression = False
    if fleurs_adapter.corpus_wer - fleurs_base.corpus_wer > thresholds["fleurs_max_absolute_wer_regression"]:
        real_regression = True
        reasons.append("FLEURS WER regression beyond threshold")
    if artur_adapter.corpus_wer - artur_base.corpus_wer > thresholds["artur_j_max_absolute_wer_regression"]:
        real_regression = True
        reasons.append("ARTUR-J WER regression beyond threshold")
    if fleurs_adapter.corpus_cer - fleurs_base.corpus_cer > thresholds["fleurs_max_absolute_cer_regression"]:
        real_regression = True
        reasons.append("FLEURS CER regression beyond threshold")
    if artur_adapter.corpus_cer - artur_base.corpus_cer > thresholds["artur_j_max_absolute_cer_regression"]:
        real_regression = True
        reasons.append("ARTUR-J CER regression beyond threshold")
    if thresholds["empty_hypotheses_must_not_increase"]:
        if fleurs_adapter.empty_hypothesis_count > fleurs_base.empty_hypothesis_count:
            real_regression = True
            reasons.append("FLEURS empty-hypothesis count increased")
        if artur_adapter.empty_hypothesis_count > artur_base.empty_hypothesis_count:
            real_regression = True
            reasons.append("ARTUR-J empty-hypothesis count increased")
    real_improved = (
        fleurs_base.corpus_wer - fleurs_adapter.corpus_wer >= thresholds["real_improvement_wer_abs"]
        or artur_base.corpus_wer - artur_adapter.corpus_wer >= thresholds["real_improvement_wer_abs"]
        or fleurs_base.corpus_cer - fleurs_adapter.corpus_cer >= thresholds["real_improvement_cer_abs"]
        or artur_base.corpus_cer - artur_adapter.corpus_cer >= thresholds["real_improvement_cer_abs"]
    )
    if synthetic_ok and not real_regression and real_improved:
        return "eligible", ["synthetic holdout improved and at least one real gate improved without regression"]
    if synthetic_ok and not real_regression:
        return "synthetic_only", ["synthetic holdout improved but no real promotion improvement occurred"]
    if real_regression:
        return "rejected", reasons
    return "not_supported", reasons


def promotion_decision(config: dict[str, Any], training: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["promotion"]
    decisions: dict[str, Any] = {}
    eligible: list[int] = []
    synthetic_only = False
    synthetic_material = False
    invalid = False
    for rank in config["adapter"]["ranks"]:
        label = f"rank_{rank}"
        synthetic_base = metric(evaluation, "fixed_synthetic_holdout", "base")
        synthetic_adapter = metric(evaluation, "fixed_synthetic_holdout", label)
        syn_wer = relative_improvement(synthetic_base.corpus_wer, synthetic_adapter.corpus_wer)
        syn_cer = relative_improvement(synthetic_base.corpus_cer, synthetic_adapter.corpus_cer)
        synthetic_material = synthetic_material or (
            syn_wer >= thresholds["synthetic_holdout_relative_wer_or_cer_improvement_percent"]
            or syn_cer >= thresholds["synthetic_holdout_relative_wer_or_cer_improvement_percent"]
        )
        status, reasons = classify_adapter(
            integrity_passed=bool(training[label]["integrity_passed"]),
            synthetic_base=synthetic_base,
            synthetic_adapter=synthetic_adapter,
            fleurs_base=metric(evaluation, "fleurs", "base"),
            fleurs_adapter=metric(evaluation, "fleurs", label),
            artur_base=metric(evaluation, "artur_j", "base"),
            artur_adapter=metric(evaluation, "artur_j", label),
            thresholds=thresholds,
        )
        decisions[label] = {"status": status, "reasons": reasons}
        if status == "eligible":
            eligible.append(rank)
        if status == "synthetic_only":
            synthetic_only = True
        if status == "EXPERIMENT_INVALID":
            invalid = True
    selected_adapter = None
    accepted_parent = "none"
    if invalid:
        conclusion = "EXPERIMENT_INVALID"
    elif eligible:
        if len(eligible) == 2:
            r16 = metric(evaluation, "fleurs", "rank_16").corpus_wer + metric(evaluation, "artur_j", "rank_16").corpus_wer
            r64 = metric(evaluation, "fleurs", "rank_64").corpus_wer + metric(evaluation, "artur_j", "rank_64").corpus_wer
            rank16_close = (
                abs(metric(evaluation, "fleurs", "rank_16").corpus_wer - metric(evaluation, "fleurs", "rank_64").corpus_wer)
                <= 0.5
                and abs(metric(evaluation, "artur_j", "rank_16").corpus_wer - metric(evaluation, "artur_j", "rank_64").corpus_wer)
                <= 0.5
            )
            selected_adapter = 16 if rank16_close else (16 if r16 <= r64 else 64)
        else:
            selected_adapter = eligible[0]
        accepted_parent = f"rank_{selected_adapter}"
        conclusion = "SL_RESIDUAL_GENERALIZATION_SUPPORTED"
    elif synthetic_only or synthetic_material:
        conclusion = "SL_RESIDUAL_SYNTHETIC_ONLY"
    else:
        conclusion = "SL_RESIDUAL_NOT_SUPPORTED"
    return {
        "schema_version": "1.0",
        "rank_decisions": decisions,
        "selected_adapter": selected_adapter,
        "accepted_parent": accepted_parent,
        "scientific_conclusion": conclusion,
    }


def summarize_public(config: dict[str, Any], data: dict[str, Any], training: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    decision = promotion_decision(config, training, evaluation)
    public = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "repository_commit": git_head(),
        "data_integrity": data,
        "adapter_arms": training,
        "evaluation": {
            label: {
                split: {
                    "normalized": result["summary"]["normalized"],
                    "raw": result["summary"]["raw"],
                    "wall_time_seconds": result["wall_time_seconds"],
                    "real_time_factor": result["real_time_factor"],
                    "peak_vram_mib": result["peak_vram_mib"],
                }
                for split, result in model_result["splits"].items()
            }
            for label, model_result in evaluation.items()
        },
        "promotion_decision": decision,
        "safety": {
            "gams_used": False,
            "new_corpus_generated": False,
            "real_speech_entered_training": False,
            "synthetic_holdout_entered_training": False,
            "model_published": False,
        },
    }
    atomic_write_json(Path(config["paths"]["run_dir"]) / "residual-adapter-public-summary.json", public)
    return public


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Slovenian residual-adapter proof experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/slovenian_residual_adapter_proof.json"))
    parser.add_argument(
        "--stage",
        choices=["verify-data", "train", "evaluate", "summarize", "all"],
        required=True,
    )
    args = parser.parse_args()

    gpu = require_single_visible_cuda().to_dict()
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json(args.config)
    run_dir = Path(config["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    stages = ["verify-data", "train", "evaluate", "summarize"] if args.stage == "all" else [args.stage]
    if "verify-data" in stages:
        results["verify-data"] = verify_data_integrity(config)
        atomic_write_json(run_dir / "data-integrity.json", results["verify-data"])
    if "train" in stages:
        training: dict[str, Any] = {}
        for rank in config["adapter"]["ranks"]:
            training[f"rank_{rank}"] = train_rank(config, int(rank), gpu)
        results["train"] = training
        atomic_write_json(run_dir / "training-summary.json", training)
    if "evaluate" in stages:
        checkpoint = Path(config["paths"]["checkpoint"]).resolve()
        training = read_json(run_dir / "training-summary.json")
        evaluation: dict[str, Any] = {}
        base_eval = evaluate_model(config=config, model_label="base", adapter_artifact=None, checkpoint=checkpoint)
        evaluation["base"] = base_eval
        for rank in config["adapter"]["ranks"]:
            artifact = Path(training[f"rank_{rank}"]["adapter_artifact"]).resolve()
            evaluation[f"rank_{rank}"] = evaluate_model(
                config=config,
                model_label=f"rank_{rank}",
                adapter_artifact=artifact,
                checkpoint=checkpoint,
            )
        results["evaluate"] = evaluation
        atomic_write_json(run_dir / "evaluation-summary.json", evaluation)
    if "summarize" in stages:
        data = read_json(run_dir / "data-integrity.json")
        training = read_json(run_dir / "training-summary.json")
        evaluation = read_json(run_dir / "evaluation-summary.json")
        results["summarize"] = summarize_public(config, data, training, evaluation)
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
