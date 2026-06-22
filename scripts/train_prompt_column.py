#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
import wave
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".external" / "NeMo"))

from slaif_asr.config import load_runtime_config
from slaif_asr.metrics import raw_cer, raw_wer
from slaif_asr.prompt_column import (
    compare_prompt_column_state_dicts,
    derive_prompt_column_selection,
    install_prompt_delta,
    merge_prompt_delta,
    trainable_delta_parameters,
    write_integrity_report,
)
from slaif_asr.prompt_experiment import (
    load_json,
    load_real_public_smoke,
    load_rendered_records,
    repository_path,
    select_records,
    validate_experiment_config,
    write_json,
    write_manifest,
)
from slaif_asr.tts import sha256_file


def require_single_gpu() -> str:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, saw {torch.cuda.device_count()}")
    name = torch.cuda.get_device_name(0)
    if "2080 Ti" not in name:
        raise RuntimeError(f"expected RTX 2080 Ti, saw {name}")
    return name


def read_wav_float(path: Path):
    import torch

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16 kHz 16-bit PCM WAV")
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    audio = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.float32) / 32768.0
    return audio, frames


def extract_transcript(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return ""
        return extract_transcript(result[0])
    if hasattr(result, "text"):
        return str(result.text)
    return str(result)


def transcribe_one(model: Any, audio_path: Path) -> str:
    model.eval()
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import RNNTPromptTranscribeConfig

    result = model.transcribe(
        [str(audio_path)],
        override_config=RNNTPromptTranscribeConfig(
            use_lhotse=False,
            batch_size=1,
            num_workers=0,
            verbose=False,
            target_lang="sl-SI",
        ),
    )
    return extract_transcript(result)


def make_batch(model: Any, record: Any, device: str):
    import torch

    audio, frames = read_wav_float(record.audio_filepath)
    token_ids = model.tokenizer.text_to_ids(record.text)
    if not token_ids:
        raise ValueError(f"{record.sample_id}: tokenizer produced no target IDs")
    return (
        audio.unsqueeze(0).to(device),
        torch.tensor([frames], dtype=torch.long, device=device),
        torch.tensor([token_ids], dtype=torch.long, device=device),
        torch.tensor([len(token_ids)], dtype=torch.long, device=device),
    )


def rnnt_loss(model: Any, batch: tuple, prompt_index: int, precision: str):
    import torch

    signal, signal_len, transcript, transcript_len = batch
    prompt_indices = torch.full((signal.shape[0],), prompt_index, dtype=torch.long, device=signal.device)
    autocast_enabled = precision == "16-mixed"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
        encoded, encoded_len = model.forward(
            input_signal=signal,
            input_signal_length=signal_len,
            prompt_indices=prompt_indices,
        )
        decoder, target_length, _ = model.decoder(targets=transcript, target_length=transcript_len)
        if model.joint.fuse_loss_wer:
            loss_value, _, _, _ = model.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_length,
                compute_wer=False,
            )
        else:
            joint = model.joint(encoder_outputs=encoded, decoder_outputs=decoder)
            loss_value = model.loss(
                log_probs=joint,
                targets=transcript,
                input_lengths=encoded_len,
                target_lengths=target_length,
            )
        loss_value = model.add_auxiliary_losses(loss_value)
    return loss_value


def state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_arm(
    *,
    model: Any,
    wrapper: Any,
    selection: Any,
    records: list[Any],
    learning_rate: float,
    max_steps: int,
    seed: int,
    precision: str,
    log_every_steps: int,
    run_dir: Path,
    arm_name: str,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    wrapper.delta.data.zero_()
    optimizer = torch.optim.AdamW(
        trainable_delta_parameters(wrapper, weight_decay=0),
        lr=learning_rate,
        weight_decay=0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "16-mixed")
    batches = [make_batch(model, record, "cuda") for record in records]
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    logs: list[dict[str, Any]] = []
    initial_loss = None
    final_loss = None
    overflow_events = 0

    for step in range(1, max_steps + 1):
        record = records[(step - 1) % len(records)]
        batch = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = rnnt_loss(model, batch, selection.prompt_index, precision)
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        previous_scale = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad = wrapper.delta.grad.detach()
        grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu()) if grad is not None else 0.0
        delta_norm = float(torch.linalg.vector_norm(wrapper.delta.detach()).cpu())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < previous_scale:
            overflow_events += 1
        final_loss = float(loss.detach().cpu())
        if step == 1 or step % log_every_steps == 0 or step == max_steps:
            logs.append(
                {
                    "step": step,
                    "candidate_id": record.sample_id,
                    "loss": final_loss,
                    "learning_rate": learning_rate,
                    "delta_norm": delta_norm,
                    "gradient_norm": grad_norm,
                    "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
                    "loss_scale": float(scaler.get_scale()),
                }
            )
        if not torch.isfinite(loss):
            break

    wall_time = time.perf_counter() - start
    payload = {
        "arm": arm_name,
        "learning_rate": learning_rate,
        "steps": len(range(1, (logs[-1]["step"] if logs else 0) + 1)),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction_percent": None
        if initial_loss in (None, 0)
        else round((initial_loss - (final_loss or initial_loss)) / initial_loss * 100.0, 3),
        "wall_time_seconds": round(wall_time, 3),
        "peak_vram_mib": round(torch.cuda.max_memory_allocated(0) / 1024 / 1024, 1),
        "overflow_events": overflow_events,
        "logs": logs,
    }
    write_json(run_dir / f"{arm_name}.json", payload)
    return payload


def classify_phase_a(arm: dict[str, Any], base_hypothesis: str, adapted_hypothesis: str, reference: str) -> str:
    loss_ok = (arm.get("loss_reduction_percent") or 0.0) >= 50.0 and arm.get("overflow_events", 0) == 0
    if not loss_ok:
        return "Not supported"
    base_wer = raw_wer(reference, base_hypothesis).percent
    adapted_wer = raw_wer(reference, adapted_hypothesis).percent
    if adapted_wer < base_wer or (not base_hypothesis.strip() and adapted_hypothesis.strip()):
        return "Supported"
    return "Partially supported"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Slovenian prompt-column-only Nemotron delta.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/prompt_column_micro_overfit.json"))
    args = parser.parse_args()

    gpu_name = require_single_gpu()
    config = load_json(args.config)
    validate_experiment_config(config)
    runtime_cfg = load_runtime_config()
    run_dir = repository_path(config["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = run_dir / "manifests"
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    records_by_id = load_rendered_records(repository_path(config["paths"]["synthetic_provenance"]))
    one_record = records_by_id[config["phase_a"]["candidate_id"]]
    train_records = select_records(records_by_id, config["phase_b"]["train_candidate_ids"])
    holdout_records = select_records(records_by_id, config["holdout_candidate_ids"])
    real_smoke = load_real_public_smoke(config)
    manifest_hashes = {
        "phase_a": write_manifest(manifest_dir / "phase_a.jsonl", [one_record]),
        "synthetic_training": write_manifest(manifest_dir / "synthetic_training.jsonl", train_records),
        "synthetic_holdout": write_manifest(manifest_dir / "synthetic_holdout.jsonl", holdout_records),
        "real_public_smoke": write_manifest(manifest_dir / "real_public_smoke.jsonl", [real_smoke]),
    }

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    import nemo.collections.asr as nemo_asr
    import torch

    torch.manual_seed(int(config["seed"]))
    checkpoint = repository_path(config["paths"]["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != runtime_cfg["base_model"]["sha256"]:
        raise RuntimeError("checkpoint SHA256 does not match runtime configuration")

    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(checkpoint), map_location="cuda:0")
    model = model.cuda()
    model.eval()
    if hasattr(model, "spec_augmentation"):
        model.spec_augmentation = None

    base_state = state_dict_cpu(model)
    selection, wrapper = install_prompt_delta(model, "sl-SI")
    base_hypothesis = transcribe_one(model, one_record.audio_filepath)
    phase_a_arms = []
    best_arm: dict[str, Any] | None = None
    best_delta = None
    best_adapted_hypothesis = ""

    for index, learning_rate in enumerate(config["phase_a"]["learning_rates"], start=1):
        arm = train_arm(
            model=model,
            wrapper=wrapper,
            selection=selection,
            records=[one_record],
            learning_rate=float(learning_rate),
            max_steps=int(config["phase_a"]["max_optimizer_steps"]),
            seed=int(config["seed"]),
            precision=config["training"]["precision"],
            log_every_steps=int(config["phase_a"]["log_every_steps"]),
            run_dir=run_dir,
            arm_name=f"phase_a_lr_{learning_rate:g}".replace(".", "_"),
        )
        adapted_hypothesis = transcribe_one(model, one_record.audio_filepath)
        arm["base_hypothesis"] = base_hypothesis
        arm["adapted_hypothesis"] = adapted_hypothesis
        arm["base_wer"] = raw_wer(one_record.text, base_hypothesis).percent
        arm["adapted_wer"] = raw_wer(one_record.text, adapted_hypothesis).percent
        arm["base_cer"] = raw_cer(one_record.text, base_hypothesis).percent
        arm["adapted_cer"] = raw_cer(one_record.text, adapted_hypothesis).percent
        arm["classification"] = classify_phase_a(arm, base_hypothesis, adapted_hypothesis, one_record.text)
        phase_a_arms.append(arm)
        if best_arm is None or (arm.get("final_loss") or float("inf")) < (best_arm.get("final_loss") or float("inf")):
            best_arm = arm
            best_delta = wrapper.delta.detach().cpu().clone()
            best_adapted_hypothesis = adapted_hypothesis
        if arm["classification"] in {"Supported", "Partially supported"}:
            break

    phase_b = {"executed": False, "reason": "Phase A did not reach supported or partially supported status."}
    final_delta = best_delta
    if best_arm and best_arm["classification"] in {"Supported", "Partially supported"}:
        phase_b = train_arm(
            model=model,
            wrapper=wrapper,
            selection=selection,
            records=train_records,
            learning_rate=float(best_arm["learning_rate"]),
            max_steps=int(config["phase_b"]["max_optimizer_steps"]),
            seed=int(config["seed"]),
            precision=config["training"]["precision"],
            log_every_steps=10,
            run_dir=run_dir,
            arm_name="phase_b_six_sample",
        )
        phase_b["executed"] = True
        phase_b["reason"] = "Phase A permitted six-sample micro-training."
        final_delta = wrapper.delta.detach().cpu().clone()

    if final_delta is None:
        raise RuntimeError("no delta was produced")
    wrapper.delta.data.copy_(final_delta.to(wrapper.delta.device))
    delta_path = artifact_dir / "sl-si-prompt-column-delta.pt"
    torch.save(
        {
            "selection": asdict(selection),
            "delta": wrapper.delta.detach().cpu(),
            "checkpoint_sha256": checkpoint_sha256,
        },
        delta_path,
    )
    merge_prompt_delta(model, selection)
    merged_checkpoint = artifact_dir / "sl-si-prompt-column-adapted.nemo"
    model.save_to(str(merged_checkpoint))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    restored = nemo_asr.models.ASRModel.restore_from(restore_path=str(merged_checkpoint), map_location="cuda:0")
    restored = restored.cuda().eval()
    adapted_state = state_dict_cpu(restored)
    integrity = compare_prompt_column_state_dicts(
        base_state,
        adapted_state,
        first_linear_weight_name=f"{selection.first_linear_name}.weight",
        first_linear_bias_name=f"{selection.first_linear_name}.bias",
        selected_column=selection.selected_column,
        selected_prompt=selection.prompt_name,
        prompt_index=selection.prompt_index,
        effective_trainable_parameters=selection.effective_trainable_parameters,
    )
    integrity_path = run_dir / "integrity-report.json"
    write_integrity_report(integrity, integrity_path)
    restored_hypothesis = transcribe_one(restored, one_record.audio_filepath)

    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "repository_commit": subprocess_output(["git", "rev-parse", "HEAD"]),
        "gpu": gpu_name,
        "precision": config["training"]["precision"],
        "base_model_revision": runtime_cfg["base_model"]["revision"],
        "base_checkpoint_sha256": checkpoint_sha256,
        "nemo_revision": runtime_cfg["nemo"]["revision"],
        "selection": asdict(selection),
        "manifest_hashes": manifest_hashes,
        "phase_a": phase_a_arms,
        "phase_b": phase_b,
        "delta_artifact": str(delta_path),
        "merged_checkpoint": str(merged_checkpoint),
        "integrity_report": str(integrity_path),
        "integrity_passed": (
            integrity.tensor_shapes_match
            and not integrity.unexpected_changed_tensors
            and integrity.unexpected_changed_elements == 0
            and integrity.selected_column_changed
            and integrity.other_columns_bitwise_identical
            and integrity.bias_bitwise_identical
        ),
        "checkpoint_restore": "PASSED",
        "restored_one_sample_hypothesis": restored_hypothesis,
    }
    write_json(run_dir / "training-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["integrity_passed"] else 2


def subprocess_output(command: list[str]) -> str:
    import subprocess

    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        return completed.stderr.strip()
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
