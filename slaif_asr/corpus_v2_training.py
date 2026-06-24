from __future__ import annotations

import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.batched_streaming import (
    StreamingRecord,
    file_sha256,
    load_gate_records,
    parse_monitor_csv,
    run_batched_arm,
)
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_scoring import (
    ATT_CONTEXT_SIZE,
    CHECKPOINT_SHA256,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    NEMO_REVISION,
    TARGET_LANG,
    checkpoint_path,
    nemo_root,
    nemo_streaming_script,
    runtime_environment,
    verify_batch_policy,
    verify_runtime_identities,
)
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.prompt_column import (
    PromptColumnDelta,
    PromptColumnSelection,
    compare_prompt_column_state_dicts,
    derive_prompt_column_selection,
    install_prompt_delta,
    merge_prompt_delta,
    trainable_delta_parameters,
)
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json, atomic_write_jsonl, summarize_predictions
from slaif_asr.tts import validate_wav


DIAGNOSTIC_CERTIFICATE_PATH = REPO_ROOT / "docs/data-certificates/sl-corpus-v2-prompt-column-diagnostic-v1.json"
SELECTED_TRAINING_CERTIFICATE_PATH = REPO_ROOT / "docs/data-certificates/sl-corpus-v2-selected-training-v1.json"
EXPECTED_SELECTED_CERTIFICATE_SHA256 = "a561ee4c76ddbc5baacca1d5f10aa3beb1749dded7f2f6a1b8fd0e893ab79602"
EXPECTED_SELECTED_MANIFEST_SHA256 = "84e10587af184be92571ab84e3bd58cd676866e2bd944534c759f0fc9a07fa13"
EXPECTED_SELECTED_AUDIO_MANIFEST_SHA256 = "4fe8ab008dd9725c65da510ed801a46299e1c03db0c00cb3fbf5dea40ff0be7b"
EXPECTED_HOLDOUT_TEXT_SHA256 = "078fab68fe82914fb1dfb0755c3fcc3f1603dae2dc52adf9397c9d5080c08fc5"
EXPECTED_HOLDOUT_AUDIO_MANIFEST_SHA256 = "7848f57e1fb65a2ef514815eec8092cd0a205b29819f6afeb767ea951473990d"
EXPECTED_FLEURS_MANIFEST_SHA256 = "8e1a17bc8269b22e05699a9e7ee9f6a5e3ce3018b39a61af2f87f06372877513"
EXPECTED_ARTUR_MANIFEST_SHA256 = "66691acd85107cc095ce648acca1f14b5cf0fd25ce1c355399283d3e7ab9a763"
DIAGNOSTIC_STATUS = "DIAGNOSTIC_ONLY"
EXPECTED_TRAINABLE_PARAMETERS = 2048
EXPERIMENT_ID = "corpus-v2-prompt-column-diagnostic-v1"
REPORT_SCHEMA_VERSION = "1.0"

PUBLIC_FORBIDDEN_KEYS = {
    "audio_filepath",
    "candidate_id",
    "candidate_ids",
    "hypothesis",
    "hypotheses",
    "local_path",
    "reference",
    "references",
    "sample_id",
    "sample_ids",
    "selected_training_id",
    "source_candidate_id",
    "text",
}
PUBLIC_FORBIDDEN_MARKERS = (
    "gamsv2-",
    "gams9holdout-",
    "/" + "home" + "/",
    "/" + "mnt" + "/",
    "/" + "tmp" + "/",
)


@dataclass(frozen=True)
class TrainingRecord:
    selected_training_id: str
    audio_filepath: str
    duration: float
    text: str
    text_sha256: str
    audio_sha256: str
    selection_reason: str
    selection_rank: int

    @property
    def sample_id(self) -> str:
        return self.selected_training_id


@dataclass(frozen=True)
class EvalRecord:
    sample_id: str
    audio_filepath: str
    duration: float
    reference: str
    original_index: int
    row: dict[str, Any]


@dataclass(frozen=True)
class BatchLayout:
    batch_size: int
    epoch: int
    batches: list[list[int]]
    actual_audio_seconds: float
    padded_audio_seconds: float
    padding_ratio: float
    final_partial_batch_size: int


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_sha256(value: str) -> str:
    return sha256_text(value)


def git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def git_show_head(path: Path) -> bytes:
    rel = str(path.relative_to(REPO_ROOT))
    completed = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT, stdout=subprocess.PIPE, check=True)
    return completed.stdout


def git_tracked_and_clean_at_head(path: Path) -> dict[str, Any]:
    rel = str(path.relative_to(REPO_ROOT))
    subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    subprocess.run(["git", "diff", "--quiet", "--", rel], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "diff", "--cached", "--quiet", "--", rel], cwd=REPO_ROOT, check=True)
    head_bytes = git_show_head(path)
    current = path.read_bytes()
    if head_bytes != current:
        raise RuntimeError(f"{rel} differs from HEAD")
    return {
        "path": rel,
        "tracked": True,
        "clean": True,
        "matches_head": True,
        "head_sha256": hashlib.sha256(head_bytes).hexdigest(),
    }


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("work_order_id") != "0020":
        raise ValueError("experiment config must belong to work order 0020")
    if config.get("target_lang") != TARGET_LANG:
        raise ValueError("target_lang must be sl-SI")
    training = config.get("training", {})
    if training.get("precision") != "fp32" or training.get("tf32") is not False:
        raise ValueError("training must use FP32 with TF32 disabled")
    if float(training.get("weight_decay", -1)) != 0.0:
        raise ValueError("weight decay must be zero")
    if int(training.get("sample_exposures", -1)) != int(training.get("epochs", 0)) * int(training.get("rows_per_epoch", 0)):
        raise ValueError("sample exposures must equal epochs * rows_per_epoch")
    surface = config.get("trainable_surface", {})
    if surface.get("type") != "sl-si-prompt-column-delta":
        raise ValueError("unsupported trainable surface")
    return config


def verify_selected_training_certificate(path: Path = SELECTED_TRAINING_CERTIFICATE_PATH) -> dict[str, Any]:
    actual = file_sha256(path)
    if actual != EXPECTED_SELECTED_CERTIFICATE_SHA256:
        raise RuntimeError(f"selected-training certificate SHA256 mismatch: {actual}")
    payload = read_json(path)
    if payload.get("status") != "SELECTED_TRAINING_MANIFEST_READY":
        raise RuntimeError("selected-training certificate is not ready")
    if int(payload.get("selected_row_count", -1)) != 160:
        raise RuntimeError("selected-training row count mismatch")
    if int(payload.get("hard_count", -1)) != 120 or int(payload.get("control_count", -1)) != 40:
        raise RuntimeError("selected-training hard/control counts mismatch")
    if payload.get("selected_manifest_sha256") != EXPECTED_SELECTED_MANIFEST_SHA256:
        raise RuntimeError("selected-training manifest SHA mismatch in certificate")
    if payload.get("selected_audio_manifest_sha256") != EXPECTED_SELECTED_AUDIO_MANIFEST_SHA256:
        raise RuntimeError("selected-training audio manifest SHA mismatch in certificate")
    return {"certificate": payload, "sha256": actual}


def _resolve_transferred_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
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


def verify_selected_training_manifests(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    manifest = repo_path(data["selected_training_manifest"])
    audio_manifest = repo_path(data["selected_training_audio_manifest"])
    if file_sha256(manifest) != data["selected_training_manifest_sha256"]:
        raise RuntimeError("selected-training manifest SHA256 mismatch")
    if file_sha256(audio_manifest) != data["selected_training_audio_manifest_sha256"]:
        raise RuntimeError("selected-training audio manifest SHA256 mismatch")
    rows = read_jsonl(manifest)
    audio_rows = read_jsonl(audio_manifest)
    if len(rows) != 160 or len(audio_rows) != 160:
        raise RuntimeError("selected-training row count mismatch")
    ids = [str(row["selected_training_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate selected-training IDs")
    audio_by_selected = {str(row["selected_training_id"]): row for row in audio_rows}
    if set(audio_by_selected) != set(ids):
        raise RuntimeError("selected-training manifest/audio manifest ID mismatch")
    hard = sum(1 for row in rows if row.get("selection_reason") == "hard")
    control = sum(1 for row in rows if row.get("selection_reason") == "control")
    if hard != 120 or control != 40:
        raise RuntimeError("selected-training hard/control mismatch")
    for row in rows:
        audio_row = audio_by_selected[str(row["selected_training_id"])]
        if str(row["text_sha256"]) != str(audio_row["target_text_sha256"]):
            raise RuntimeError("selected-training text hash mismatch")
        if str(row["audio_sha256"]) != str(audio_row["audio_sha256"]):
            raise RuntimeError("selected-training audio hash mismatch")
        audio_path = _resolve_transferred_path(str(row["audio_filepath"]))
        validate_wav(audio_path, sample_rate=16000)
        if file_sha256(audio_path) != row["audio_sha256"]:
            raise RuntimeError("selected-training audio file hash mismatch")
    return {
        "selected_manifest_sha256": file_sha256(manifest),
        "selected_audio_manifest_sha256": file_sha256(audio_manifest),
        "rows": len(rows),
        "hard": hard,
        "control": control,
    }


def load_training_records(config: dict[str, Any]) -> list[TrainingRecord]:
    rows = read_jsonl(repo_path(config["data"]["selected_training_manifest"]))
    records: list[TrainingRecord] = []
    for row in rows:
        audio_path = _resolve_transferred_path(str(row["audio_filepath"]))
        validate_wav(audio_path, sample_rate=16000)
        records.append(
            TrainingRecord(
                selected_training_id=str(row["selected_training_id"]),
                audio_filepath=str(audio_path),
                duration=float(row["duration"]),
                text=str(row["text"]),
                text_sha256=str(row["text_sha256"]),
                audio_sha256=str(row["audio_sha256"]),
                selection_reason=str(row["selection_reason"]),
                selection_rank=int(row["selection_rank"]),
            )
        )
    return records


def load_holdout_audio_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    manifest = repo_path(config["data"]["synthetic_holdout_audio_manifest"])
    if file_sha256(manifest) != config["data"]["synthetic_holdout_audio_manifest_sha256"]:
        raise RuntimeError("synthetic holdout audio manifest SHA256 mismatch")
    rows = read_jsonl(manifest)
    if len(rows) != int(config["data"]["synthetic_holdout_rows"]):
        raise RuntimeError("synthetic holdout row count mismatch")
    for row in rows:
        audio_path = _resolve_transferred_path(str(row["audio_filepath"]))
        validate_wav(audio_path, sample_rate=16000)
        if file_sha256(audio_path) != row["audio_sha256"]:
            raise RuntimeError("synthetic holdout audio hash mismatch")
    return rows


def load_synthetic_eval_records(config: dict[str, Any], split: str) -> list[StreamingRecord]:
    if split == "selected_training":
        records = [
            StreamingRecord(
                sample_id=row.selected_training_id,
                audio_filepath=row.audio_filepath,
                duration=row.duration,
                reference=row.text,
                original_index=index,
                row={"split": split},
            )
            for index, row in enumerate(load_training_records(config))
        ]
        return records
    if split == "synthetic_holdout":
        records = []
        for index, row in enumerate(load_holdout_audio_rows(config)):
            audio_path = _resolve_transferred_path(str(row["audio_filepath"]))
            records.append(
                StreamingRecord(
                    sample_id=str(row["candidate_id"]),
                    audio_filepath=str(audio_path),
                    duration=float(row["duration_seconds"]),
                    reference=str(row["text"]),
                    original_index=index,
                    row={"split": split},
                )
            )
        return records
    raise ValueError(f"unsupported synthetic split: {split}")


def load_real_gate_eval_records(config: dict[str, Any], split: str) -> list[StreamingRecord]:
    data = config["data"]
    if split == "fleurs_v2":
        return load_gate_records(
            repo_path(data["fleurs_v2_manifest"]),
            expected_sha256=data["fleurs_v2_manifest_sha256"],
            expected_rows=int(data["fleurs_v2_rows"]),
            gate_id="fleurs-sl-si-test-full-v2",
        )
    if split == "artur_j":
        return load_gate_records(
            repo_path(data["artur_j_manifest"]),
            expected_sha256=data["artur_j_manifest_sha256"],
            expected_rows=int(data["artur_j_rows"]),
            gate_id="artur-j-public-gate-v1",
        )
    raise ValueError(f"unsupported real gate split: {split}")


def candidate_holdout_overlap_counts(config: dict[str, Any]) -> dict[str, int]:
    selected_rows = read_jsonl(repo_path(config["data"]["selected_training_audio_manifest"]))
    holdout_rows = load_holdout_audio_rows(config)
    selected_ids = {str(row.get("source_candidate_id") or row.get("candidate_id")) for row in selected_rows}
    holdout_ids = {str(row["candidate_id"]) for row in holdout_rows}
    selected_text = {str(row["target_text_sha256"]) for row in selected_rows}
    holdout_text = {str(row["target_text_sha256"]) for row in holdout_rows}
    selected_audio = {str(row["audio_sha256"]) for row in selected_rows}
    holdout_audio = {str(row["audio_sha256"]) for row in holdout_rows}
    selected_utterance = {str(row.get("utterance_family_id")) for row in selected_rows}
    holdout_utterance = {str(row.get("utterance_family_id")) for row in holdout_rows}
    selected_template = {str(row.get("template_family_id", "")) for row in selected_rows if row.get("template_family_id")}
    holdout_template = {str(row.get("template_family_id", "")) for row in holdout_rows if row.get("template_family_id")}
    return {
        "id_overlap": len(selected_ids & holdout_ids),
        "text_hash_overlap": len(selected_text & holdout_text),
        "audio_hash_overlap": len(selected_audio & holdout_audio),
        "utterance_family_overlap": len(selected_utterance & holdout_utterance),
        "template_family_overlap": len(selected_template & holdout_template),
    }


def verify_all_input_identities(config: dict[str, Any], *, check_gpu: bool = False) -> dict[str, Any]:
    selected_certificate = verify_selected_training_certificate()
    selected = verify_selected_training_manifests(config)
    holdout_rows = load_holdout_audio_rows(config)
    overlaps = candidate_holdout_overlap_counts(config)
    if any(value != 0 for value in overlaps.values()):
        raise RuntimeError(f"candidate/holdout overlap detected: {overlaps}")
    runtime = verify_runtime_identities(check_gpu=check_gpu)
    return {
        "selected_training_certificate_sha256": selected_certificate["sha256"],
        "selected_training": selected,
        "synthetic_holdout_audio_manifest_sha256": file_sha256(repo_path(config["data"]["synthetic_holdout_audio_manifest"])),
        "synthetic_holdout_rows": len(holdout_rows),
        "candidate_holdout_overlap_counts": overlaps,
        "runtime": runtime,
    }


def build_diagnostic_certificate(config_path: Path, *, selected_certificate_path: Path, work_order_id: str) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    if work_order_id != "0020":
        raise ValueError("this diagnostic certificate is authorized only for work order 0020")
    selected_certificate_path = repo_path(selected_certificate_path).resolve()
    selected = verify_selected_training_certificate(selected_certificate_path)
    identities = verify_all_input_identities(config, check_gpu=False)
    config_sha = file_sha256(config_path)
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v2-prompt-column-diagnostic-v1",
        "status": DIAGNOSTIC_STATUS,
        "decision_date": "2026-06-24",
        "work_order_id": work_order_id,
        "named_exception": "corpus-v2 prompt-column diagnostic exception",
        "human_approved_exception_statement": (
            "This certificate authorizes only the named Work Order 0020 prompt-column diagnostic on "
            "single-voice synthetic selected-training data. It does not issue TRAINING_ELIGIBLE and "
            "cannot be reused by another experiment."
        ),
        "selected_training_certificate": {
            "path": str(selected_certificate_path.relative_to(REPO_ROOT)),
            "sha256": selected["sha256"],
            "status": selected["certificate"]["status"],
        },
        "selected_training": {
            "manifest_sha256": identities["selected_training"]["selected_manifest_sha256"],
            "audio_manifest_sha256": identities["selected_training"]["selected_audio_manifest_sha256"],
            "rows": identities["selected_training"]["rows"],
            "hard_rows": identities["selected_training"]["hard"],
            "control_rows": identities["selected_training"]["control"],
        },
        "holdout": {
            "text_sha256": config["data"]["synthetic_holdout_text_sha256"],
            "audio_manifest_sha256": identities["synthetic_holdout_audio_manifest_sha256"],
            "rows": identities["synthetic_holdout_rows"],
        },
        "candidate_holdout_exclusion_evidence": identities["candidate_holdout_overlap_counts"],
        "candidate_and_holdout_certificates": {
            "candidate_audio_certificate_sha256": selected["certificate"]["candidate_source_hashes"]["audio_certificate_sha256"],
            "holdout_audio_certificate_sha256": selected["certificate"]["holdout_hashes"]["audio_certificate_sha256"],
        },
        "generator_revisions": {
            "candidate_source": "cjvt/GaMS3-12B-Instruct@1d0b27af5748784482600d24779409e7e1dc9adc",
            "synthetic_holdout": "cjvt/GaMS-9B-Instruct@292744023fa0b7ccc7ae2c3c885a67468e49fa03",
        },
        "piper": {
            "engine": "OHF-Voice/piper1-gpl",
            "engine_revision": "b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6",
            "voice": "sl_SI-artur-medium",
            "voice_revision": "217ddc79818708b078d0d14a8fae9608b9d77141",
            "voice_concentration": "single voice synthetic audio",
        },
        "model": config["model"],
        "nemo_revision": config["model"]["nemo_revision"],
        "trainable_surface": config["trainable_surface"],
        "expected_trainable_count": EXPECTED_TRAINABLE_PARAMETERS,
        "experiment_config_sha256": config_sha,
        "maximum_permitted_arms": config["authorization"]["maximum_arms"],
        "maximum_permitted_epochs": config["training"]["epochs"],
        "maximum_sample_exposures": config["training"]["sample_exposures"],
        "permitted_evaluation_sets": ["selected_training", "synthetic_holdout", "fleurs_v2", "artur_j"],
        "authorized_actions": config["authorization"]["authorized_actions"],
        "prohibited_actions": config["authorization"]["prohibited_actions"],
        "scientific_limitations": [
            "Data status is DIAGNOSTIC_ONLY, not TRAINING_ELIGIBLE.",
            "Selected training and synthetic holdout are single-voice synthetic Piper audio.",
            "Synthetic holdout improvement is diagnostic only and is not real-speech generalization evidence.",
            "No checkpoint from this experiment may become an accepted parent.",
        ],
    }
    assert_public_report_safe(certificate)
    atomic_write_json(DIAGNOSTIC_CERTIFICATE_PATH, certificate)
    return certificate


def verify_diagnostic_certificate(config_path: Path) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    path = repo_path(config["authorization"]["certificate_path"])
    tracked = git_tracked_and_clean_at_head(path)
    payload = read_json(path)
    if payload.get("status") != DIAGNOSTIC_STATUS:
        raise RuntimeError("diagnostic certificate status must be DIAGNOSTIC_ONLY")
    if payload.get("work_order_id") != "0020":
        raise RuntimeError("diagnostic certificate work-order ID mismatch")
    if payload.get("experiment_config_sha256") != file_sha256(config_path):
        raise RuntimeError("diagnostic certificate config SHA mismatch")
    if payload.get("selected_training_certificate", {}).get("sha256") != EXPECTED_SELECTED_CERTIFICATE_SHA256:
        raise RuntimeError("diagnostic certificate selected-training certificate SHA mismatch")
    identities = verify_all_input_identities(config, check_gpu=False)
    return {"certificate": payload, "tracked": tracked, "identities": identities}


def read_wav_tensor(path: Path):
    import torch

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16 kHz 16-bit PCM WAV")
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)
    audio = torch.frombuffer(bytearray(raw), dtype=torch.int16).to(torch.float32) / 32768.0
    return audio, frame_count


def token_ids(model: Any, text: str) -> list[int]:
    ids = model.tokenizer.text_to_ids(text)
    if not ids:
        raise ValueError("tokenizer produced no IDs")
    return [int(item) for item in ids]


def make_training_batch(model: Any, records: Sequence[TrainingRecord], *, device: str):
    import torch

    audios = []
    audio_lengths = []
    transcripts = []
    transcript_lengths = []
    for record in records:
        audio, frames = read_wav_tensor(Path(record.audio_filepath))
        audios.append(audio)
        audio_lengths.append(frames)
        ids = token_ids(model, record.text)
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


def rnnt_loss(model: Any, batch: tuple[Any, Any, Any, Any], prompt_index: int):
    import torch

    signal, signal_len, transcript, transcript_len = batch
    prompt_indices = torch.full((signal.shape[0],), prompt_index, dtype=torch.long, device=signal.device)
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
    return model.add_auxiliary_losses(loss_value)


def duration_sorted_indices(records: Sequence[TrainingRecord]) -> list[int]:
    return [
        index
        for index, _record in sorted(
            enumerate(records),
            key=lambda item: (float(item[1].duration), item[1].selected_training_id),
        )
    ]


def deterministic_epoch_batches(records: Sequence[TrainingRecord], *, batch_size: int, epoch: int, seed: int, bucketed: bool) -> BatchLayout:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if bucketed:
        ordered = duration_sorted_indices(records)
    else:
        ordered = list(range(len(records)))
    batches = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
    rng = random.Random(seed + epoch)
    rng.shuffle(batches)
    actual = sum(records[index].duration for batch in batches for index in batch)
    padded = 0.0
    for batch in batches:
        if not batch:
            continue
        padded += max(records[index].duration for index in batch) * len(batch)
    final_partial = 0 if not batches else (len(batches[-1]) if len(batches[-1]) != batch_size else 0)
    return BatchLayout(
        batch_size=batch_size,
        epoch=epoch,
        batches=batches,
        actual_audio_seconds=round(actual, 6),
        padded_audio_seconds=round(padded, 6),
        padding_ratio=round(padded / actual, 6) if actual else 0.0,
        final_partial_batch_size=final_partial,
    )


def assert_epoch_covers_once(layout: BatchLayout, row_count: int) -> None:
    flat = [index for batch in layout.batches for index in batch]
    if sorted(flat) != list(range(row_count)):
        raise ValueError("epoch layout does not cover every row exactly once")


def select_probe_records(records: Sequence[TrainingRecord], count: int) -> list[TrainingRecord]:
    if count > len(records):
        raise ValueError("probe count exceeds record count")
    return sorted(records, key=lambda item: (stable_sha256(item.selected_training_id), item.selected_training_id))[:count]


def compare_batched_loss_to_individual(
    model: Any,
    selection: PromptColumnSelection,
    records: Sequence[TrainingRecord],
    *,
    device: str,
) -> dict[str, Any]:
    import torch

    individual_losses = []
    for record in records:
        with torch.no_grad():
            loss = rnnt_loss(model, make_training_batch(model, [record], device=device), selection.prompt_index)
        individual_losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        batch_loss = rnnt_loss(model, make_training_batch(model, records, device=device), selection.prompt_index)
    individual_mean = sum(individual_losses) / len(individual_losses)
    batch_value = float(batch_loss.detach().cpu())
    relative_difference = abs(batch_value - individual_mean) / individual_mean if individual_mean else 0.0
    return {
        "rows": len(records),
        "individual_mean_loss": round(individual_mean, 6),
        "batch_loss": round(batch_value, 6),
        "relative_difference": round(relative_difference, 8),
        "finite": bool(torch.isfinite(batch_loss)),
    }


def selection_from_benchmark(rows: Sequence[dict[str, Any]], *, within_best_fraction: float) -> dict[str, Any]:
    valid = [
        row
        for row in rows
        if int(row["batch_size"]) > 1
        and row.get("status") == "PASSED"
        and row.get("correctness", {}).get("passed") is True
        and row.get("audio_seconds_per_wall_second") is not None
    ]
    if not valid:
        return {"selected_batch_size": None, "reason": "no valid batch size above 1"}
    best = max(float(row["audio_seconds_per_wall_second"]) for row in valid)
    threshold = best * within_best_fraction
    selected = min(
        [row for row in valid if float(row["audio_seconds_per_wall_second"]) >= threshold],
        key=lambda item: int(item["batch_size"]),
    )
    return {
        "selected_batch_size": int(selected["batch_size"]),
        "best_audio_seconds_per_wall_second": round(best, 6),
        "selected_audio_seconds_per_wall_second": selected["audio_seconds_per_wall_second"],
        "within_best_fraction": within_best_fraction,
    }


def optimizer_parameter_ids(wrapper: PromptColumnDelta) -> set[int]:
    return {id(parameter) for parameter in trainable_delta_parameters(wrapper, weight_decay=0)}


def state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def original_state_dict_from_prompt_delta_model(model: Any, selection: PromptColumnSelection) -> dict[str, Any]:
    state: dict[str, Any] = {}
    linear_prefix = selection.first_linear_name + ".linear."
    delta_name = selection.first_linear_name + ".delta"
    original_prefix = selection.first_linear_name + "."
    for name, tensor in model.state_dict().items():
        if name == delta_name:
            continue
        if name.startswith(linear_prefix):
            original_name = original_prefix + name.removeprefix(linear_prefix)
            state[original_name] = tensor.detach().cpu().clone()
        else:
            state[name] = tensor.detach().cpu().clone()
    return state


def first_linear_bias_name(selection: PromptColumnSelection, state: dict[str, Any]) -> str | None:
    candidate = selection.first_linear_name + ".bias"
    return candidate if candidate in state else None


def parameter_integrity_before_merge(
    base_state: dict[str, Any],
    current_state: dict[str, Any],
    *,
    selection: PromptColumnSelection,
) -> dict[str, Any]:
    unexpected = []
    if set(base_state) != set(current_state):
        unexpected.extend(sorted(set(base_state).symmetric_difference(current_state)))
    changed = []
    for name in sorted(set(base_state) & set(current_state)):
        if not (base_state[name].shape == current_state[name].shape and (base_state[name] == current_state[name]).all()):
            changed.append(name)
    return {
        "pretrained_tensors_identical": not unexpected and not changed,
        "changed_pretrained_tensors": changed,
        "unexpected_tensors": unexpected,
        "selection": asdict(selection),
    }


def evaluate_prompt_column_integrity(
    base_state: dict[str, Any],
    adapted_state: dict[str, Any],
    *,
    selection: PromptColumnSelection,
) -> dict[str, Any]:
    report = compare_prompt_column_state_dicts(
        base_state,
        adapted_state,
        first_linear_weight_name=selection.first_linear_name + ".weight",
        first_linear_bias_name=first_linear_bias_name(selection, base_state),
        selected_column=selection.selected_column,
        selected_prompt=selection.prompt_name,
        prompt_index=selection.prompt_index,
        effective_trainable_parameters=selection.effective_trainable_parameters,
    )
    return asdict(report) | {"passed": not report.unexpected_changed_tensors and report.unexpected_changed_elements == 0}


def extract_transcript(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return ""
        return extract_transcript(result[0])
    if hasattr(result, "text"):
        return str(result.text)
    return str(result)


def public_metrics_from_local_predictions(records: Sequence[StreamingRecord], predictions: dict[str, str]) -> dict[str, Any]:
    rows = [
        {
            "reference": record.reference,
            "hypothesis": predictions[record.sample_id],
            "pipeline_status": "PASSED",
            "empty_hypothesis": not predictions[record.sample_id].strip(),
        }
        for record in sorted(records, key=lambda item: item.original_index)
    ]
    return summarize_predictions(rows)


def run_evaluation_arm(
    *,
    records: Sequence[StreamingRecord],
    checkpoint: Path,
    run_dir: Path,
    python_executable: Path,
) -> dict[str, Any]:
    arm = run_batched_arm(
        records=records,
        batch_size=1,
        bucketed=False,
        run_dir=run_dir,
        python_executable=python_executable,
        nemo_script=nemo_streaming_script(),
        checkpoint=checkpoint,
        context=ATT_CONTEXT_SIZE,
        env=runtime_environment(),
        physical_gpu_index="1",
        monitor_interval_seconds=0.2,
    )
    if arm.get("status") != "PASSED":
        raise RuntimeError(f"evaluation failed for {run_dir}: {arm.get('status')}")
    return arm


def metric_pair(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["metrics"]
    return {
        "raw_wer": metrics["raw"]["corpus_wer"],
        "raw_cer": metrics["raw"]["corpus_cer"],
        "normalized_wer": metrics["normalized"]["corpus_wer"],
        "normalized_cer": metrics["normalized"]["corpus_cer"],
        "mean_utterance_wer": metrics["normalized"]["mean_utterance_wer"],
        "median_utterance_wer": metrics["normalized"]["median_utterance_wer"],
        "mean_utterance_cer": metrics["normalized"]["mean_utterance_cer"],
        "median_utterance_cer": metrics["normalized"]["median_utterance_cer"],
        "empty_hypotheses": metrics["raw"]["empty_hypothesis_count"],
    }


def relative_improvement(base: float, candidate: float) -> float:
    if base == 0:
        return 0.0
    return (base - candidate) / base * 100.0


def classify_scientific(evaluation: dict[str, Any], valid_arms: Sequence[str]) -> dict[str, Any]:
    reasons = []
    real_gain = False
    synthetic_gain = False
    for arm in valid_arms:
        metrics = evaluation["models"][arm]["splits"]
        base = evaluation["models"]["base"]["splits"]
        holdout_base = base["synthetic_holdout"]["metrics"]["normalized"]
        holdout_arm = metrics["synthetic_holdout"]["metrics"]["normalized"]
        holdout_wer_gain = relative_improvement(holdout_base["corpus_wer"], holdout_arm["corpus_wer"])
        holdout_cer_gain = relative_improvement(holdout_base["corpus_cer"], holdout_arm["corpus_cer"])
        arm_synthetic = holdout_wer_gain >= 10.0 or holdout_cer_gain >= 10.0
        synthetic_gain = synthetic_gain or arm_synthetic
        non_regression = True
        improvement = False
        for split in ("fleurs_v2", "artur_j"):
            base_norm = base[split]["metrics"]["normalized"]
            arm_norm = metrics[split]["metrics"]["normalized"]
            base_empty = base[split]["metrics"]["raw"]["empty_hypothesis_count"]
            arm_empty = metrics[split]["metrics"]["raw"]["empty_hypothesis_count"]
            wer_delta = arm_norm["corpus_wer"] - base_norm["corpus_wer"]
            cer_delta = arm_norm["corpus_cer"] - base_norm["corpus_cer"]
            if wer_delta > 1.0 or cer_delta > 1.5 or arm_empty > base_empty:
                non_regression = False
            if wer_delta <= -1.0 or cer_delta <= -1.5:
                improvement = True
        if non_regression and improvement:
            real_gain = True
        reasons.append(
            {
                "arm": arm,
                "synthetic_holdout_wer_relative_gain": round(holdout_wer_gain, 6),
                "synthetic_holdout_cer_relative_gain": round(holdout_cer_gain, 6),
                "synthetic_gain": arm_synthetic,
                "real_non_regression": non_regression,
                "real_improvement": improvement,
            }
        )
    if real_gain:
        classification = "CORPUS_V2_PROMPT_COLUMN_REAL_GAIN_DIAGNOSTIC"
    elif synthetic_gain:
        classification = "CORPUS_V2_PROMPT_COLUMN_SYNTHETIC_ONLY"
    else:
        classification = "CORPUS_V2_PROMPT_COLUMN_NOT_SUPPORTED"
    return {"classification": classification, "accepted_parent": "none", "arm_decisions": reasons}


def classify_batching(report: dict[str, Any]) -> dict[str, Any]:
    reference = report["training"].get("reference_batch1")
    batched = report["training"].get("a100_batched")
    if not batched:
        return {"classification": "A100_PROMPT_TRAINING_BATCH_UNAVAILABLE", "reason": "no valid batch above 1"}
    if batched.get("integrity", {}).get("passed") is not True:
        return {"classification": "EXPERIMENT_INVALID", "reason": "batched integrity did not pass"}
    ref_eps = float(reference.get("examples_per_second", 0.0))
    batched_eps = float(batched.get("examples_per_second", 0.0))
    throughput_ok = ref_eps > 0 and batched_eps >= ref_eps * 1.25
    evals = report["evaluation"]["models"]
    equivalent = True
    for split, tolerance in (("synthetic_holdout", 1.0), ("fleurs_v2", 0.5), ("artur_j", 0.5)):
        ref_norm = evals["reference_batch1"]["splits"][split]["metrics"]["normalized"]
        bat_norm = evals["a100_batched"]["splits"][split]["metrics"]["normalized"]
        if abs(ref_norm["corpus_wer"] - bat_norm["corpus_wer"]) > tolerance:
            equivalent = False
        if abs(ref_norm["corpus_cer"] - bat_norm["corpus_cer"]) > tolerance:
            equivalent = False
        ref_empty = evals["reference_batch1"]["splits"][split]["metrics"]["raw"]["empty_hypothesis_count"]
        bat_empty = evals["a100_batched"]["splits"][split]["metrics"]["raw"]["empty_hypothesis_count"]
        if split in {"fleurs_v2", "artur_j"} and ref_empty != bat_empty:
            equivalent = False
    if throughput_ok and equivalent:
        classification = "A100_PROMPT_TRAINING_BATCH_SUPPORTED"
    else:
        classification = "A100_PROMPT_TRAINING_BATCH_NOT_EQUIVALENT"
    return {
        "classification": classification,
        "reference_examples_per_second": round(ref_eps, 6),
        "batched_examples_per_second": round(batched_eps, 6),
        "throughput_ratio": round(batched_eps / ref_eps, 6) if ref_eps else None,
        "equivalent": equivalent,
    }


def assert_public_report_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def walk(value: Any, key: str = "") -> None:
        if key in PUBLIC_FORBIDDEN_KEYS:
            raise ValueError(f"public payload contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    if any(marker in serialized for marker in PUBLIC_FORBIDDEN_MARKERS):
        raise ValueError("public payload contains row IDs or local paths")


def run_dir(config: dict[str, Any]) -> Path:
    return repo_path(config["run_dir"])


def manifest_for_evaluation(records: Sequence[StreamingRecord], path: Path) -> str:
    rows = [
        {
            "sample_id": record.sample_id,
            "audio_filepath": record.audio_filepath,
            "duration": record.duration,
            "text": record.reference,
            "lang": "sl-SI",
            "target_lang": "sl-SI",
        }
        for record in records
    ]
    atomic_write_jsonl(path, rows)
    return file_sha256(path)


def runtime_summary() -> dict[str, Any]:
    import torch

    gpu = require_single_visible_cuda(allowed_name_fragments=("A100",))
    return {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nemo_revision": NEMO_REVISION,
        "physical_selector": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": "cuda:0",
        "gpu": gpu.device_name,
        "visible_gpu_count": gpu.visible_device_count,
        "precision": "fp32",
        "tf32": False,
    }


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    decisions = payload.get("decisions", {})
    lines = [
        "# Experiment 0008: Corpus-v2 Prompt-column Diagnostic",
        "",
        f"Status: **{payload.get('status', 'completed in PR; pending strategic review')}**",
        "",
        "This diagnostic trains only the 2,048-value Slovenian prompt-column delta on the corpus-v2 selected synthetic training manifest. The data status is `DIAGNOSTIC_ONLY`; no checkpoint is accepted as a parent.",
        "",
        "## Authorization",
        "",
        f"- Certificate status: `{payload['authorization']['status']}`",
        f"- Certificate SHA256: `{payload['authorization']['sha256']}`",
        f"- Selected-training manifest SHA256: `{payload['input_integrity']['selected_training']['selected_manifest_sha256']}`",
        "",
        "## Decisions",
        "",
        f"- Scientific classification: `{decisions.get('scientific', {}).get('classification', 'NOT_RUN')}`",
        f"- Batching classification: `{decisions.get('batching', {}).get('classification', 'NOT_RUN')}`",
        "- Accepted parent: `none`",
        "",
        "## Training",
        "",
        "| Arm | Batch | Epochs | Exposures | Initial probe loss | Final probe loss | Full-loss reduction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm_name, arm in payload.get("training", {}).items():
        lines.append(
            f"| {arm_name} | {arm.get('batch_size')} | {arm.get('epochs')} | {arm.get('sample_exposures')} | "
            f"{arm.get('initial_probe_loss')} | {arm.get('final_probe_loss')} | {arm.get('full_loss_reduction_percent')} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Model | Split | Normalized WER | Normalized CER | Empty hypotheses |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_name, model_payload in payload.get("evaluation", {}).get("models", {}).items():
        for split_name, split in model_payload.get("splits", {}).items():
            metrics = split["metrics"]
            lines.append(
                f"| {model_name} | {split_name} | {metrics['normalized']['corpus_wer']} | "
                f"{metrics['normalized']['corpus_cer']} | {metrics['raw']['empty_hypothesis_count']} |"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Single-voice synthetic training.",
            "- No real training or calibration speech.",
            "- Synthetic holdout metrics are diagnostic only.",
            "- Development real gates are not a final blind test.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
