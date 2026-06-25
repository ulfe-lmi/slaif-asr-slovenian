from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.acoustic_quality import read_audio_stats
from slaif_asr.batched_streaming import file_sha256
from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_training import (
    EXPECTED_ARTUR_MANIFEST_SHA256,
    EXPECTED_FLEURS_MANIFEST_SHA256,
    EXPECTED_HOLDOUT_AUDIO_MANIFEST_SHA256,
    EXPECTED_HOLDOUT_TEXT_SHA256,
    EXPECTED_SELECTED_AUDIO_MANIFEST_SHA256,
    EXPECTED_SELECTED_CERTIFICATE_SHA256,
    EXPECTED_SELECTED_MANIFEST_SHA256,
    EXPECTED_TRAINABLE_PARAMETERS,
    TrainingRecord,
    assert_public_report_safe,
    candidate_holdout_overlap_counts,
    git_tracked_and_clean_at_head,
    load_training_records,
    read_json,
    read_jsonl,
    repo_path,
    select_probe_records,
    stable_sha256,
    verify_all_input_identities,
    verify_selected_training_certificate,
)
from slaif_asr.real_eval import atomic_write_json, atomic_write_jsonl


SPEAKER_RANGE_CERTIFICATE_PATH = REPO_ROOT / "docs/data-certificates/sl-corpus-v2-speaker-range-diagnostic-v1.json"
DEFAULT_EXPERIMENT_CONFIG = REPO_ROOT / "configs/experiments/corpus_v2_speaker_range_diagnostic_v1.json"
DEFAULT_AUGMENTATION_CONFIG = REPO_ROOT / "configs/augmentation/corpus_v2_speaker_range_resampling_v1.json"
BASELINE_REPORT_PATH = REPO_ROOT / "docs/experiments/0008-corpus-v2-prompt-column-diagnostic.json"
BASELINE_REPORT_SHA256 = "117ec8bbb97580db3e9ccf13a118a8472aa06930f42417171046e487e8ba411a"
DIAGNOSTIC_STATUS = "DIAGNOSTIC_ONLY"
EXPERIMENT_ID = "corpus-v2-speaker-range-diagnostic-v1"
REPORT_JSON_PATH = REPO_ROOT / "docs/experiments/0009-corpus-v2-speaker-range-augmentation-diagnostic.json"
REPORT_MD_PATH = REPO_ROOT / "docs/experiments/0009-corpus-v2-speaker-range-augmentation-diagnostic.md"

EXPECTED_CLEAN_METRICS = {
    "selected_training": {"wer": 69.955, "cer": 26.405, "empty": 0},
    "synthetic_holdout": {"wer": 73.137, "cer": 27.474, "empty": 2},
    "fleurs_v2": {"wer": 61.470, "cer": 20.347, "empty": 0},
    "artur_j": {"wer": 71.123, "cer": 25.796, "empty": 0},
}
EXPECTED_BASE_METRICS = {
    "selected_training": {"wer": 93.032, "cer": 61.623, "empty": 41},
    "synthetic_holdout": {"wer": 84.317, "cer": 47.295, "empty": 17},
    "fleurs_v2": {"wer": 52.703, "cer": 16.423, "empty": 1},
    "artur_j": {"wer": 67.453, "cer": 29.016, "empty": 12},
}
EXPECTED_TRAINING_PROTOCOL = {
    "batch_size": 8,
    "epochs": 12,
    "sample_exposures": 1920,
    "optimizer_steps": 240,
    "learning_rate": 0.01,
    "trainable_parameter_count": 2048,
}
EXPECTED_PROFILE_FACTORS = [
    ("child_like_proxy", 0.8, 4, 5),
    ("high_voice_proxy", 0.9, 9, 10),
    ("clean", 1.0, 1, 1),
    ("low_voice_proxy", 1.1, 11, 10),
    ("elder_like_proxy", 1.2, 6, 5),
]
NON_CLEAN_PROFILE_IDS = {"child_like_proxy", "high_voice_proxy", "low_voice_proxy", "elder_like_proxy"}
PUBLIC_FORBIDDEN_MARKERS = ("gamsv2-", "gams9holdout-", "/" + "home" + "/", "/" + "mnt" + "/", "/" + "tmp" + "/")


@dataclass(frozen=True)
class SpeakerProfile:
    profile_id: str
    resampling_rate: float
    up: int
    down: int
    intended_proxy: str


@dataclass(frozen=True)
class AugmentationPaths:
    run_root: Path
    audio_root: Path
    manifest: Path
    schedule: Path
    validation: Path
    summary: Path
    listening_pack: Path


@dataclass(frozen=True)
class SourceAudioRecord:
    selected_training_id: str
    source_audio_filepath: str
    source_audio_sha256: str
    source_text_sha256: str
    utterance_family_id: str
    source_family_id: str
    duration: float


def sha256_json(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_json_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment_config(path: Path = DEFAULT_EXPERIMENT_CONFIG) -> dict[str, Any]:
    config = load_json_config(repo_path(path))
    if config.get("work_order_id") != "0021":
        raise ValueError("speaker-range diagnostic config must belong to work order 0021")
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
    if float(training.get("learning_rate", -1.0)) != 0.01:
        raise ValueError("learning rate must remain 0.01")
    if float(training.get("weight_decay", -1.0)) != 0.0:
        raise ValueError("weight decay must remain zero")
    if training.get("precision") != "fp32" or training.get("tf32") is not False:
        raise ValueError("training must use FP32 with TF32 disabled")
    if training.get("spec_augment") is not False:
        raise ValueError("SpecAugment must be disabled")
    surface = config.get("trainable_surface", {})
    if surface.get("type") != "sl-si-prompt-column-delta":
        raise ValueError("unsupported trainable surface")
    return config


def load_augmentation_config(path: Path = DEFAULT_AUGMENTATION_CONFIG) -> dict[str, Any]:
    config = load_json_config(repo_path(path))
    if config.get("policy_id") != "corpus-v2-speaker-range-resampling-v1":
        raise ValueError("unexpected speaker-range augmentation policy")
    profiles = parse_profiles(config)
    factors = [(item.profile_id, item.resampling_rate, item.up, item.down) for item in profiles]
    if factors != EXPECTED_PROFILE_FACTORS:
        raise ValueError("speaker-range profiles do not match the fixed policy")
    if config.get("transform", {}).get("algorithm") != "scipy.signal.resample_poly":
        raise ValueError("speaker-range transform must use scipy.signal.resample_poly")
    return config


def parse_profiles(config: dict[str, Any]) -> list[SpeakerProfile]:
    profiles = []
    for row in config.get("profiles", []):
        profiles.append(
            SpeakerProfile(
                profile_id=str(row["profile_id"]),
                resampling_rate=float(row["resampling_rate"]),
                up=int(row["up"]),
                down=int(row["down"]),
                intended_proxy=str(row["intended_proxy"]),
            )
        )
    return profiles


def augmentation_paths(config: dict[str, Any]) -> AugmentationPaths:
    outputs = config["local_outputs"]
    root = repo_path(outputs["run_root"])
    return AugmentationPaths(
        run_root=root,
        audio_root=root / "audio",
        manifest=repo_path(outputs["manifest"]),
        schedule=repo_path(outputs["exposure_schedule"]),
        validation=repo_path(outputs["validation"]),
        summary=repo_path(outputs["summary"]),
        listening_pack=repo_path(outputs["listening_pack"]),
    )


def verify_baseline_report(path: Path = BASELINE_REPORT_PATH) -> dict[str, Any]:
    actual = file_sha256(path)
    if actual != BASELINE_REPORT_SHA256:
        raise RuntimeError(f"Experiment 0008 baseline report SHA mismatch: {actual}")
    payload = read_json(path)
    clean = payload.get("evaluation", {}).get("models", {}).get("a100_batched", {}).get("splits", {})
    base = payload.get("evaluation", {}).get("models", {}).get("base", {}).get("splits", {})
    for split, expected in EXPECTED_CLEAN_METRICS.items():
        observed = _normalized_triplet(clean[split])
        if observed != expected:
            raise RuntimeError(f"clean baseline metric mismatch for {split}: {observed}")
    for split, expected in EXPECTED_BASE_METRICS.items():
        observed = _normalized_triplet(base[split])
        if observed != expected:
            raise RuntimeError(f"base metric mismatch for {split}: {observed}")
    training = payload.get("training", {}).get("a100_batched", {})
    for key, expected in EXPECTED_TRAINING_PROTOCOL.items():
        observed = training.get(key)
        if isinstance(expected, float):
            if float(observed) != expected:
                raise RuntimeError(f"clean training {key} mismatch: {observed}")
        elif observed != expected:
            raise RuntimeError(f"clean training {key} mismatch: {observed}")
    burden = real_regression_burden(EXPECTED_BASE_METRICS, EXPECTED_CLEAN_METRICS)
    return {
        "sha256": actual,
        "clean_comparison_arm": "a100_batched",
        "clean_real_regression_burden": burden,
        "scientific_classification": payload.get("decisions", {}).get("scientific", {}).get("classification"),
        "batching_classification": payload.get("decisions", {}).get("batching", {}).get("classification"),
    }


def _normalized_triplet(split_payload: dict[str, Any]) -> dict[str, Any]:
    metrics = split_payload["metrics"]["normalized"]
    return {
        "wer": round(float(metrics["corpus_wer"]), 3),
        "cer": round(float(metrics["corpus_cer"]), 3),
        "empty": int(metrics["empty_hypothesis_count"]),
    }


def real_regression_burden(base_metrics: dict[str, dict[str, Any]], model_metrics: dict[str, dict[str, Any]]) -> float:
    burden = 0.0
    for split in ("fleurs_v2", "artur_j"):
        burden += max(0.0, float(model_metrics[split]["wer"]) - float(base_metrics[split]["wer"]))
        burden += max(0.0, float(model_metrics[split]["cer"]) - float(base_metrics[split]["cer"]))
    return round(burden, 6)


def verify_data_identities(config: dict[str, Any]) -> dict[str, Any]:
    if config["data"]["selected_training_certificate_sha256"] != EXPECTED_SELECTED_CERTIFICATE_SHA256:
        raise RuntimeError("selected-training certificate SHA mismatch in experiment config")
    if config["data"]["selected_training_manifest_sha256"] != EXPECTED_SELECTED_MANIFEST_SHA256:
        raise RuntimeError("selected-training manifest SHA mismatch in experiment config")
    if config["data"]["selected_training_audio_manifest_sha256"] != EXPECTED_SELECTED_AUDIO_MANIFEST_SHA256:
        raise RuntimeError("selected-training audio manifest SHA mismatch in experiment config")
    if config["data"]["synthetic_holdout_text_sha256"] != EXPECTED_HOLDOUT_TEXT_SHA256:
        raise RuntimeError("synthetic holdout text SHA mismatch in experiment config")
    if config["data"]["synthetic_holdout_audio_manifest_sha256"] != EXPECTED_HOLDOUT_AUDIO_MANIFEST_SHA256:
        raise RuntimeError("synthetic holdout audio manifest SHA mismatch in experiment config")
    if config["data"]["fleurs_v2_manifest_sha256"] != EXPECTED_FLEURS_MANIFEST_SHA256:
        raise RuntimeError("FLEURS-v2 manifest SHA mismatch in experiment config")
    if config["data"]["artur_j_manifest_sha256"] != EXPECTED_ARTUR_MANIFEST_SHA256:
        raise RuntimeError("ARTUR-J manifest SHA mismatch in experiment config")
    identities = verify_all_input_identities(config, check_gpu=False)
    return identities


def build_speaker_range_certificate(
    *,
    experiment_config_path: Path,
    selected_certificate_path: Path,
    baseline_report_path: Path,
    augmentation_config_path: Path,
    work_order_id: str,
) -> dict[str, Any]:
    if work_order_id != "0021":
        raise ValueError("speaker-range diagnostic is authorized only for work order 0021")
    config_path = repo_path(experiment_config_path)
    config = load_experiment_config(config_path)
    augmentation_path = repo_path(augmentation_config_path)
    augmentation = load_augmentation_config(augmentation_path)
    baseline = verify_baseline_report(repo_path(baseline_report_path))
    selected = verify_selected_training_certificate(repo_path(selected_certificate_path))
    identities = verify_data_identities(config)
    certificate = {
        "schema_version": "1.0",
        "certificate_id": "sl-corpus-v2-speaker-range-diagnostic-v1",
        "status": DIAGNOSTIC_STATUS,
        "decision_date": "2026-06-25",
        "work_order_id": work_order_id,
        "named_exception": "corpus-v2 speaker-range augmentation diagnostic exception",
        "exception_rationale": [
            "Selected text and audio have passed the governed corpus-v2 admission process.",
            "The clean prompt-column diagnostic learned the synthetic holdout but regressed real gates.",
            "The only tested intervention is deterministic speaker-range resampling of training waveforms.",
            "The corpus remains one Piper voice family and no checkpoint may become an accepted parent."
        ],
        "selected_training_certificate": {
            "sha256": selected["sha256"],
            "status": selected["certificate"]["status"]
        },
        "selected_training": {
            "manifest_sha256": identities["selected_training"]["selected_manifest_sha256"],
            "audio_manifest_sha256": identities["selected_training"]["selected_audio_manifest_sha256"],
            "rows": identities["selected_training"]["rows"],
            "hard_rows": identities["selected_training"]["hard"],
            "control_rows": identities["selected_training"]["control"]
        },
        "synthetic_holdout": {
            "text_sha256": config["data"]["synthetic_holdout_text_sha256"],
            "audio_manifest_sha256": identities["synthetic_holdout_audio_manifest_sha256"],
            "rows": identities["synthetic_holdout_rows"]
        },
        "candidate_holdout_exclusion_evidence": identities["candidate_holdout_overlap_counts"],
        "baseline_report": baseline,
        "model": config["model"],
        "trainable_surface": config["trainable_surface"],
        "expected_trainable_count": EXPECTED_TRAINABLE_PARAMETERS,
        "augmentation_policy": {
            "sha256": file_sha256(augmentation_path),
            "policy_id": augmentation["policy_id"],
            "profiles": [
                {
                    "profile_id": profile.profile_id,
                    "resampling_rate": profile.resampling_rate,
                    "up": profile.up,
                    "down": profile.down,
                    "proxy_only": True
                }
                for profile in parse_profiles(augmentation)
            ]
        },
        "experiment_config_sha256": file_sha256(config_path),
        "training": {
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "sample_exposures": config["training"]["sample_exposures"],
            "optimizer_steps": config["training"]["optimizer_steps"],
            "optimizer": config["training"]["optimizer"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "seed": config["training"]["seed"],
            "precision": config["training"]["precision"],
            "tf32": config["training"]["tf32"]
        },
        "authorized_actions": config["authorization"]["authorized_actions"],
        "prohibited_actions": config["authorization"]["prohibited_actions"],
        "permitted_evaluation_sets": ["selected_training", "synthetic_holdout", "fleurs_v2", "artur_j"],
        "limitations": [
            "Data status is DIAGNOSTIC_ONLY, not TRAINING_ELIGIBLE.",
            "All variants remain derived from one Piper voice family.",
            "Resampling profiles are acoustic proxies only.",
            "Synthetic holdout improvement is diagnostic only.",
            "No checkpoint from this experiment may become an accepted parent."
        ]
    }
    assert_public_report_safe(certificate)
    atomic_write_json(SPEAKER_RANGE_CERTIFICATE_PATH, certificate)
    return certificate


def verify_speaker_range_certificate(config_path: Path = DEFAULT_EXPERIMENT_CONFIG, *, require_head: bool) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    certificate_path = repo_path(config["authorization"]["certificate_path"])
    payload = read_json(certificate_path)
    if payload.get("status") != DIAGNOSTIC_STATUS:
        raise RuntimeError("speaker-range certificate status must be DIAGNOSTIC_ONLY")
    if payload.get("work_order_id") != "0021":
        raise RuntimeError("speaker-range certificate work-order ID mismatch")
    if payload.get("experiment_config_sha256") != file_sha256(repo_path(config_path)):
        raise RuntimeError("speaker-range certificate experiment config SHA mismatch")
    if payload.get("augmentation_policy", {}).get("sha256") != file_sha256(repo_path(config["augmentation"]["config"])):
        raise RuntimeError("speaker-range certificate augmentation policy SHA mismatch")
    if payload.get("selected_training_certificate", {}).get("sha256") != EXPECTED_SELECTED_CERTIFICATE_SHA256:
        raise RuntimeError("speaker-range certificate selected-training SHA mismatch")
    tracked = git_tracked_and_clean_at_head(certificate_path) if require_head else {"matches_head": False, "tracked": certificate_path.exists()}
    return {"certificate": payload, "tracked": tracked}


def load_source_audio_records(config: dict[str, Any]) -> list[SourceAudioRecord]:
    rows = read_jsonl(repo_path(config["data"]["selected_training_audio_manifest"]))
    if len(rows) != 160:
        raise RuntimeError("selected-training audio manifest row count mismatch")
    records = []
    seen: set[str] = set()
    for row in rows:
        selected_id = str(row["selected_training_id"])
        if selected_id in seen:
            raise RuntimeError(f"duplicate selected-training ID: {selected_id}")
        seen.add(selected_id)
        path = repo_path(str(row["audio_filepath"]))
        if not path.exists():
            path = Path(str(row["audio_filepath"]))
        if file_sha256(path) != row["audio_sha256"]:
            raise RuntimeError("selected-training source audio SHA mismatch")
        records.append(
            SourceAudioRecord(
                selected_training_id=selected_id,
                source_audio_filepath=str(path.resolve()),
                source_audio_sha256=str(row["audio_sha256"]),
                source_text_sha256=str(row["target_text_sha256"]),
                utterance_family_id=str(row["utterance_family_id"]),
                source_family_id=str(row["source_family_id"]),
                duration=float(row["duration_seconds"]),
            )
        )
    return sorted(records, key=lambda item: item.selected_training_id)


def stable_ordered_ids(records: Sequence[TrainingRecord | SourceAudioRecord]) -> list[str]:
    ids = [getattr(record, "selected_training_id") for record in records]
    return sorted(ids, key=lambda value: (stable_sha256(value), value))


def build_exposure_schedule(
    records: Sequence[TrainingRecord | SourceAudioRecord],
    profiles: Sequence[SpeakerProfile],
    *,
    epochs: int,
) -> list[dict[str, Any]]:
    ordered_ids = stable_ordered_ids(records)
    profile_ids = [profile.profile_id for profile in profiles]
    offsets = {selected_id: index % len(profiles) for index, selected_id in enumerate(ordered_ids)}
    rows = []
    for epoch in range(epochs):
        for selected_id in ordered_ids:
            profile_index = (offsets[selected_id] + epoch) % len(profiles)
            rows.append(
                {
                    "schema_version": "1.0",
                    "epoch": epoch + 1,
                    "selected_training_id": selected_id,
                    "profile_index": profile_index,
                    "profile_id": profile_ids[profile_index],
                }
            )
    return rows


def validate_exposure_schedule(schedule: Sequence[dict[str, Any]], records: Sequence[TrainingRecord | SourceAudioRecord], profiles: Sequence[SpeakerProfile], *, epochs: int) -> dict[str, Any]:
    record_ids = {getattr(record, "selected_training_id") for record in records}
    profile_ids = [profile.profile_id for profile in profiles]
    if len(schedule) != len(record_ids) * epochs:
        raise RuntimeError("exposure schedule length mismatch")
    total_by_profile = {profile_id: 0 for profile_id in profile_ids}
    per_epoch: dict[int, dict[str, int]] = {epoch: {profile_id: 0 for profile_id in profile_ids} for epoch in range(1, epochs + 1)}
    per_row_profiles: dict[str, set[str]] = {selected_id: set() for selected_id in record_ids}
    for row in schedule:
        selected_id = str(row["selected_training_id"])
        profile_id = str(row["profile_id"])
        epoch = int(row["epoch"])
        if selected_id not in record_ids:
            raise RuntimeError("schedule contains unknown selected-training ID")
        if profile_id not in profile_ids:
            raise RuntimeError("schedule contains unknown profile")
        total_by_profile[profile_id] += 1
        per_epoch[epoch][profile_id] += 1
        per_row_profiles[selected_id].add(profile_id)
    for epoch in range(1, epochs + 1):
        epoch_ids = [str(row["selected_training_id"]) for row in schedule if int(row["epoch"]) == epoch]
        if set(epoch_ids) != record_ids or len(epoch_ids) != len(record_ids):
            raise RuntimeError("each epoch must expose every row exactly once")
        for profile_id, count in per_epoch[epoch].items():
            if count != 32:
                raise RuntimeError(f"epoch {epoch} profile {profile_id} exposure count mismatch: {count}")
    for profile_id, count in total_by_profile.items():
        if count != 384:
            raise RuntimeError(f"profile {profile_id} total exposure count mismatch: {count}")
    for selected_id, seen_profiles in per_row_profiles.items():
        if set(profile_ids) - seen_profiles:
            raise RuntimeError(f"{selected_id}: missing profile exposure")
        counts = {profile_id: 0 for profile_id in profile_ids}
        for row in schedule:
            if row["selected_training_id"] == selected_id:
                counts[str(row["profile_id"])] += 1
        if min(counts.values()) < 2:
            raise RuntimeError(f"{selected_id}: each profile must appear at least twice")
    return {
        "scheduled_exposures": len(schedule),
        "exposures_by_profile": total_by_profile,
        "clean_fraction": round(total_by_profile["clean"] / len(schedule), 6),
        "epochs": epochs,
    }


def _rate_dir(profile: SpeakerProfile) -> str:
    return f"rate-{int(round(profile.resampling_rate * 100)):03d}"


def _read_wav_float(path: Path) -> tuple[Any, int]:
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono 16 kHz signed 16-bit PCM WAV")
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    return data, frames


def _write_wav_float(path: Path, samples: Any) -> None:
    import numpy as np

    clipped = np.clip(samples, -0.999969, 0.999969)
    pcm = np.rint(clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.part.{os.getpid()}{path.suffix}")
    with wave.open(str(temp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    os.replace(temp, path)


def _rms(samples: Any) -> float:
    import numpy as np

    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _peak(samples: Any) -> float:
    import numpy as np

    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def resample_variant(source_path: Path, output_path: Path, profile: SpeakerProfile, *, peak_limit: float) -> dict[str, Any]:
    import scipy
    from scipy.signal import resample_poly

    source, source_frames = _read_wav_float(source_path)
    source_rms = _rms(source)
    transformed = resample_poly(source, profile.up, profile.down)
    raw_rms = _rms(transformed)
    rms_gain = 1.0
    if source_rms > 0 and raw_rms > 0:
        rms_gain = source_rms / raw_rms
        transformed = transformed * rms_gain
    peak_before_safety = _peak(transformed)
    peak_safety_gain = 1.0
    if peak_before_safety > peak_limit:
        peak_safety_gain = peak_limit / peak_before_safety
        transformed = transformed * peak_safety_gain
    _write_wav_float(output_path, transformed)
    stats = read_audio_stats(output_path)
    observed_ratio = stats.frames / source_frames if source_frames else 0.0
    return {
        "source_frame_count": source_frames,
        "output_frame_count": stats.frames,
        "source_duration_seconds": round(source_frames / 16000.0, 6),
        "output_duration_seconds": round(stats.duration_seconds, 6),
        "expected_duration_ratio": profile.resampling_rate,
        "observed_duration_ratio": round(observed_ratio, 8),
        "source_rms": round(source_rms, 8),
        "output_rms": round(stats.rms_ratio, 8),
        "applied_rms_gain": round(rms_gain, 8),
        "peak_safety_gain": round(peak_safety_gain, 8),
        "output_peak": round(stats.peak_ratio, 8),
        "output_audio_sha256": stats.sha256,
        "scipy_version": scipy.__version__,
    }


def build_augmentations(experiment_config: dict[str, Any], augmentation_config: dict[str, Any]) -> dict[str, Any]:
    paths = augmentation_paths(augmentation_config)
    profiles = parse_profiles(augmentation_config)
    source_rows = load_source_audio_records(experiment_config)
    source_by_id = {row.selected_training_id: row for row in source_rows}
    profile_rows: list[dict[str, Any]] = []
    peak_limit = float(augmentation_config["transform"]["peak_safety_limit"])
    policy_sha = file_sha256(DEFAULT_AUGMENTATION_CONFIG)
    for source in source_rows:
        source_path = Path(source.source_audio_filepath)
        source_stats = read_audio_stats(source_path)
        for profile in profiles:
            common = {
                "schema_version": "1.0",
                "source_selected_training_id": source.selected_training_id,
                "source_audio_sha256": source.source_audio_sha256,
                "source_text_sha256": source.source_text_sha256,
                "utterance_family_id": source.utterance_family_id,
                "source_family_id": source.source_family_id,
                "partition_role": "selected_training",
                "profile_id": profile.profile_id,
                "resampling_rate": profile.resampling_rate,
                "rational": {"up": profile.up, "down": profile.down},
                "transform_implementation": "scipy.signal.resample_poly",
                "augmentation_policy_sha256": policy_sha,
            }
            if profile.profile_id == "clean":
                profile_rows.append(
                    common
                    | {
                        "audio_filepath": str(source_path.resolve()),
                        "source_frame_count": source_stats.frames,
                        "output_frame_count": source_stats.frames,
                        "source_duration_seconds": round(source_stats.duration_seconds, 6),
                        "output_duration_seconds": round(source_stats.duration_seconds, 6),
                        "expected_duration_ratio": 1.0,
                        "observed_duration_ratio": 1.0,
                        "source_rms": round(source_stats.rms_ratio, 8),
                        "output_rms": round(source_stats.rms_ratio, 8),
                        "applied_rms_gain": 1.0,
                        "peak_safety_gain": 1.0,
                        "output_peak": round(source_stats.peak_ratio, 8),
                        "output_audio_sha256": source.source_audio_sha256,
                        "clean_reference": True,
                    }
                )
                continue
            output_path = paths.audio_root / _rate_dir(profile) / f"{source.selected_training_id}.wav"
            details = resample_variant(source_path, output_path, profile, peak_limit=peak_limit)
            profile_rows.append(common | {"audio_filepath": str(output_path.resolve()), "clean_reference": False} | details)
    profile_rows = sorted(profile_rows, key=lambda row: (str(row["source_selected_training_id"]), str(row["profile_id"])))
    schedule = build_exposure_schedule(source_rows, profiles, epochs=int(experiment_config["training"]["epochs"]))
    schedule_by_key = {(row["source_selected_training_id"], row["profile_id"]): row for row in profile_rows}
    enriched_schedule = []
    for row in schedule:
        profile_row = schedule_by_key[(row["selected_training_id"], row["profile_id"])]
        enriched_schedule.append(
            row
            | {
                "audio_filepath": profile_row["audio_filepath"],
                "audio_sha256": profile_row["output_audio_sha256"],
                "duration": profile_row["output_duration_seconds"],
            }
        )
    atomic_write_jsonl(paths.manifest, profile_rows)
    atomic_write_jsonl(paths.schedule, enriched_schedule)
    write_listening_pack(paths.listening_pack, profile_rows, source_by_id)
    return summarize_local_augmentation(experiment_config, augmentation_config)


def write_listening_pack(path: Path, profile_rows: Sequence[dict[str, Any]], source_by_id: dict[str, SourceAudioRecord]) -> None:
    selected_ids = sorted(source_by_id, key=lambda value: (stable_sha256(value), value))[:8]
    by_id: dict[str, dict[str, str]] = {selected_id: {} for selected_id in selected_ids}
    for row in profile_rows:
        selected_id = str(row["source_selected_training_id"])
        if selected_id in by_id:
            by_id[selected_id][str(row["profile_id"])] = str(row["audio_filepath"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.stem}.part{path.suffix}")
    with temp.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(["selected_training_id", "clean", "child_like_proxy", "high_voice_proxy", "low_voice_proxy", "elder_like_proxy"])
        for selected_id in selected_ids:
            row = by_id[selected_id]
            writer.writerow(
                [
                    selected_id,
                    row.get("clean", ""),
                    row.get("child_like_proxy", ""),
                    row.get("high_voice_proxy", ""),
                    row.get("low_voice_proxy", ""),
                    row.get("elder_like_proxy", ""),
                ]
            )
    os.replace(temp, path)


def _db_ratio(value: float, reference: float) -> float:
    if value <= 0 or reference <= 0:
        return 0.0
    return 20.0 * math.log10(value / reference)


def validate_augmentations(experiment_config: dict[str, Any], augmentation_config: dict[str, Any]) -> dict[str, Any]:
    paths = augmentation_paths(augmentation_config)
    profiles = parse_profiles(augmentation_config)
    profile_ids = [profile.profile_id for profile in profiles]
    rows = read_jsonl(paths.manifest)
    schedule = read_jsonl(paths.schedule)
    source_rows = load_source_audio_records(experiment_config)
    source_ids = {row.selected_training_id for row in source_rows}
    by_source: dict[str, list[dict[str, Any]]] = {}
    output_paths = []
    non_clean_hashes = []
    failures: list[dict[str, Any]] = []
    ratio_tolerance = float(augmentation_config["validation"]["duration_ratio_tolerance_fraction"])
    rms_tolerance_db = float(augmentation_config["validation"]["rms_tolerance_db"])
    max_peak = float(augmentation_config["validation"]["maximum_peak_ratio"])
    for row in rows:
        selected_id = str(row.get("source_selected_training_id", ""))
        profile_id = str(row.get("profile_id", ""))
        by_source.setdefault(selected_id, []).append(row)
        if selected_id not in source_ids:
            failures.append({"reason": "unknown_source"})
        if profile_id not in profile_ids:
            failures.append({"reason": "unknown_profile"})
        if "gams9holdout-" in selected_id:
            failures.append({"reason": "holdout_id_encountered"})
        audio_path = Path(str(row["audio_filepath"]))
        if not audio_path.exists():
            failures.append({"reason": "missing_audio"})
            continue
        output_paths.append(str(audio_path.resolve()))
        stats = read_audio_stats(audio_path)
        if stats.sample_rate != 16000 or stats.channels != 1 or stats.sample_width != 2 or stats.frames <= 0:
            failures.append({"reason": "bad_wav_format"})
        if stats.clipping_fraction > 0.01:
            failures.append({"reason": "clipping_fraction"})
        if stats.active_frame_fraction < 0.02:
            failures.append({"reason": "active_frame_fraction"})
        if profile_id in NON_CLEAN_PROFILE_IDS:
            non_clean_hashes.append(str(row["output_audio_sha256"]))
            if stats.peak_ratio > max_peak:
                failures.append({"reason": "peak_limit", "profile": profile_id})
            expected_ratio = float(row["expected_duration_ratio"])
            observed_ratio = float(row["observed_duration_ratio"])
            if abs(observed_ratio - expected_ratio) > ratio_tolerance:
                failures.append({"reason": "duration_ratio", "profile": profile_id})
            source_rms = float(row["source_rms"])
            output_rms = float(row["output_rms"])
            peak_scaled = float(row.get("peak_safety_gain", 1.0)) < 0.999999
            if not peak_scaled and abs(_db_ratio(output_rms, source_rms)) > rms_tolerance_db:
                failures.append({"reason": "rms_match", "profile": profile_id})
    if len(rows) != 800:
        failures.append({"reason": "profile_record_count", "count": len(rows)})
    if len([row for row in rows if row.get("profile_id") != "clean"]) != 640:
        failures.append({"reason": "non_clean_count"})
    if len(set(output_paths)) != len(output_paths):
        failures.append({"reason": "duplicate_output_path"})
    if len(set(non_clean_hashes)) != len(non_clean_hashes):
        failures.append({"reason": "duplicate_non_clean_hash"})
    for selected_id in source_ids:
        if sorted(row["profile_id"] for row in by_source.get(selected_id, [])) != sorted(profile_ids):
            failures.append({"reason": "missing_profile_for_source"})
    schedule_summary = validate_exposure_schedule(schedule, source_rows, profiles, epochs=int(experiment_config["training"]["epochs"]))
    validation = {
        "schema_version": "1.0",
        "status": "PASSED" if not failures else "FAILED",
        "source_rows": len(source_rows),
        "non_clean_files": len(non_clean_hashes),
        "total_profile_records": len(rows),
        "schedule": schedule_summary,
        "failures": failures,
        "manifest_sha256": file_sha256(paths.manifest),
        "exposure_schedule_sha256": file_sha256(paths.schedule),
    }
    atomic_write_json(paths.validation, validation)
    if failures:
        raise RuntimeError(f"speaker-range augmentation validation failed: {failures[:5]}")
    return validation


def summarize_local_augmentation(experiment_config: dict[str, Any], augmentation_config: dict[str, Any]) -> dict[str, Any]:
    paths = augmentation_paths(augmentation_config)
    profiles = parse_profiles(augmentation_config)
    rows = read_jsonl(paths.manifest) if paths.manifest.exists() else []
    schedule = read_jsonl(paths.schedule) if paths.schedule.exists() else []
    counts = {profile.profile_id: 0 for profile in profiles}
    durations = {profile.profile_id: [] for profile in profiles}
    peaks = {profile.profile_id: [] for profile in profiles}
    for row in rows:
        profile_id = str(row["profile_id"])
        counts[profile_id] += 1
        durations[profile_id].append(float(row["output_duration_seconds"]))
        peaks[profile_id].append(float(row["output_peak"]))
    summary = {
        "schema_version": "1.0",
        "status": "PASSED" if paths.validation.exists() and read_json(paths.validation).get("status") == "PASSED" else "LOCAL_ONLY",
        "source_rows": len({row.get("source_selected_training_id") for row in rows}),
        "profiles": [asdict(profile) for profile in profiles],
        "profile_record_counts": counts,
        "non_clean_files": sum(count for profile, count in counts.items() if profile != "clean"),
        "total_profile_records": len(rows),
        "scheduled_exposures": len(schedule),
        "duration_seconds_by_profile": {
            profile: {
                "total": round(sum(values), 6),
                "mean": round(statistics.fmean(values), 6) if values else 0.0,
            }
            for profile, values in durations.items()
        },
        "peak_by_profile": {
            profile: {
                "max": round(max(values), 6) if values else 0.0,
                "mean": round(statistics.fmean(values), 6) if values else 0.0,
            }
            for profile, values in peaks.items()
        },
        "manifest_sha256": file_sha256(paths.manifest) if paths.manifest.exists() else None,
        "exposure_schedule_sha256": file_sha256(paths.schedule) if paths.schedule.exists() else None,
        "validation_sha256": file_sha256(paths.validation) if paths.validation.exists() else None,
    }
    atomic_write_json(paths.summary, summary)
    return summary


def training_records_for_epoch(
    clean_records: Sequence[TrainingRecord],
    schedule_rows: Sequence[dict[str, Any]],
    *,
    epoch: int,
) -> dict[str, TrainingRecord]:
    clean_by_id = {record.selected_training_id: record for record in clean_records}
    selected: dict[str, TrainingRecord] = {}
    for row in schedule_rows:
        if int(row["epoch"]) != epoch:
            continue
        selected_id = str(row["selected_training_id"])
        clean = clean_by_id[selected_id]
        selected[selected_id] = TrainingRecord(
            selected_training_id=clean.selected_training_id,
            audio_filepath=str(row["audio_filepath"]),
            duration=float(row["duration"]),
            text=clean.text,
            text_sha256=clean.text_sha256,
            audio_sha256=str(row["audio_sha256"]),
            selection_reason=clean.selection_reason,
            selection_rank=clean.selection_rank,
        )
    if set(selected) != set(clean_by_id):
        raise RuntimeError("epoch schedule does not cover all clean records")
    return selected


def privacy_safe_experiment_report(payload: dict[str, Any]) -> None:
    assert_public_report_safe(payload)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if any(marker in serialized for marker in PUBLIC_FORBIDDEN_MARKERS):
        raise ValueError("public report contains raw row IDs or local paths")


def synthetic_holdout_gain(base_metrics: dict[str, dict[str, Any]], model_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = base_metrics["synthetic_holdout"]
    model = model_metrics["synthetic_holdout"]
    wer_gain = (float(base["wer"]) - float(model["wer"])) / float(base["wer"]) * 100.0 if base["wer"] else 0.0
    cer_gain = (float(base["cer"]) - float(model["cer"])) / float(base["cer"]) * 100.0 if base["cer"] else 0.0
    return {
        "wer_relative_gain": round(wer_gain, 6),
        "cer_relative_gain": round(cer_gain, 6),
        "passes": wer_gain >= 10.0 or cer_gain >= 10.0,
    }


def classify_speaker_range_augmented(augmented_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    holdout = synthetic_holdout_gain(EXPECTED_BASE_METRICS, augmented_metrics)
    burden = real_regression_burden(EXPECTED_BASE_METRICS, augmented_metrics)
    clean_burden = real_regression_burden(EXPECTED_BASE_METRICS, EXPECTED_CLEAN_METRICS)
    prevents = bool(holdout["passes"])
    for split in ("fleurs_v2", "artur_j"):
        base = EXPECTED_BASE_METRICS[split]
        aug = augmented_metrics[split]
        if float(aug["wer"]) - float(base["wer"]) > 1.0:
            prevents = False
        if float(aug["cer"]) - float(base["cer"]) > 1.5:
            prevents = False
        if int(aug["empty"]) > int(base["empty"]):
            prevents = False
    mitigates = False
    if not prevents and holdout["passes"]:
        burden_reduction = (clean_burden - burden) / clean_burden * 100.0 if clean_burden else 0.0
        no_worse_than_clean = True
        for split in ("fleurs_v2", "artur_j"):
            clean = EXPECTED_CLEAN_METRICS[split]
            aug = augmented_metrics[split]
            if float(aug["wer"]) - float(clean["wer"]) > 0.5:
                no_worse_than_clean = False
            if float(aug["cer"]) - float(clean["cer"]) > 0.5:
                no_worse_than_clean = False
        mitigates = burden_reduction >= 30.0 and no_worse_than_clean
    if prevents:
        classification = "SPEAKER_RANGE_AUGMENTATION_PREVENTS_REAL_REGRESSION"
    elif mitigates:
        classification = "SPEAKER_RANGE_AUGMENTATION_MITIGATES_REAL_REGRESSION"
    else:
        classification = "SPEAKER_RANGE_AUGMENTATION_NOT_SUPPORTED"
    burden_reduction = (clean_burden - burden) / clean_burden * 100.0 if clean_burden else 0.0
    return {
        "classification": classification,
        "accepted_parent": "none",
        "synthetic_holdout_gain": holdout,
        "clean_real_regression_burden": clean_burden,
        "augmented_real_regression_burden": burden,
        "burden_reduction_percent": round(burden_reduction, 6),
    }


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    decision = payload.get("decision", {})
    lines = [
        "# Experiment 0009: Corpus-v2 Speaker-range Augmentation Diagnostic",
        "",
        f"Status: **{payload.get('status', 'completed in PR; pending strategic review')}**",
        "",
        "This diagnostic changes only deterministic speaker-range resampling of the selected-training waveforms relative to Experiment 0008's clean `a100_batched` arm. The data status remains `DIAGNOSTIC_ONLY`; no checkpoint is accepted as a parent.",
        "",
        "## Authorization",
        "",
        f"- Certificate status: `{payload['authorization']['status']}`",
        f"- Certificate SHA256: `{payload['authorization']['sha256']}`",
        f"- Baseline report SHA256: `{payload['authorization']['baseline_report_sha256']}`",
        "",
        "## Augmentation",
        "",
        f"- Source rows: {payload['augmentation']['source_rows']}",
        f"- Non-clean generated files: {payload['augmentation']['non_clean_files']}",
        f"- Scheduled exposures: {payload['augmentation']['scheduled_exposures']}",
        "",
        "## Aggregate Metrics",
        "",
        "| Split | Base WER/CER | Clean batch-8 WER/CER | Augmented WER/CER | Empty base/clean/augmented |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in payload.get("metric_comparison", {}).items():
        lines.append(
            f"| {split} | {row['base']['wer']}/{row['base']['cer']} | "
            f"{row['clean']['wer']}/{row['clean']['cer']} | {row['augmented']['wer']}/{row['augmented']['cer']} | "
            f"{row['base']['empty']}/{row['clean']['empty']}/{row['augmented']['empty']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Scientific classification: `{decision.get('classification', 'NOT_RUN')}`",
            "- Accepted parent: `none`",
            f"- Clean real-regression burden: {decision.get('clean_real_regression_burden')}",
            f"- Augmented real-regression burden: {decision.get('augmented_real_regression_burden')}",
            "",
            "## Limitations",
            "",
            "- One original Piper voice family.",
            "- Resampling is only an acoustic proxy and does not establish age, gender, or multi-speaker coverage.",
            "- No real calibration speech.",
            "- Synthetic holdout is not real-generalization evidence.",
            "- FLEURS-v2 and ARTUR-J are development gates.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
