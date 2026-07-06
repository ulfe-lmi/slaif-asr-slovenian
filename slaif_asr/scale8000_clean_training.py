from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.batched_streaming import StreamingRecord, file_sha256, load_gate_records, metrics_for, resolve_manifest_audio_path
from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_jsonl, sha256_file
from slaif_asr.scale8000_corpus import local_run_path, repo_path
from slaif_asr.tts import validate_wav


ARM_NAME = "scale8000_clean_only_joint_adapter_dim32"
CLEAN_VOICE_ORDER = (
    "piper-sl_SI-artur-medium",
    "supertonic-M1",
    "supertonic-M2",
    "supertonic-M3",
    "supertonic-M4",
    "supertonic-F1",
    "supertonic-F2",
    "supertonic-F3",
    "supertonic-F4",
)
TRAINING_STYLES = ("M1", "M2", "M3", "M4", "F1", "F2", "F3", "F4")
BASE_DIRECTIONAL_METRICS = {
    "piper_synthetic_holdout": {"wer": 86.025, "cer": 46.762, "empty": 17},
    "supertonic_heldout_voice_holdout": {"wer": 58.307, "cer": 27.712, "empty": 32},
    "fleurs_v2": {"wer": 52.685, "cer": 16.406, "empty": 1},
    "artur_j": {"wer": 67.322, "cer": 28.620, "empty": 12},
}
SCALE2000_DIRECTIONAL_METRICS = {
    "piper_synthetic_holdout": {"wer": 55.435, "cer": 20.073, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 27.407, "cer": 7.597, "empty": 0},
    "fleurs_v2": {"wer": 51.589, "cer": 16.238, "empty": 0},
    "artur_j": {"wer": 60.114, "cer": 20.630, "empty": 0},
}
PUBLIC_FORBIDDEN_KEYS = {
    "audio_filepath",
    "candidate_id",
    "candidate_ids",
    "holdout_id",
    "holdout_ids",
    "hypothesis",
    "hypotheses",
    "local_path",
    "reference",
    "references",
    "sample_id",
    "sample_ids",
    "semantic_key",
    "selected_training_id",
    "text",
}
PUBLIC_FORBIDDEN_MARKERS = (
    "gamsv",
    "gams9holdout-",
    "fleurs-sl-si-test-occ-",
    "artur-j-public-",
    "/" + "home" + "/",
    "/" + "tmp" + "/",
    "/" + "data-nvme" + "/",
    ".wav",
)


@dataclass(frozen=True)
class CleanTrainingRecord:
    semantic_key: str
    voice: str
    audio_filepath: str
    duration: float
    text: str
    text_sha256: str
    audio_sha256: str
    source_family_id: str
    utterance_family_id: str

    @property
    def selected_training_id(self) -> str:
        return f"{self.semantic_key}.{self.voice}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def local_path(path_text: str | Path) -> Path:
    return local_run_path(path_text)


def run_dir(config: dict[str, Any]) -> Path:
    return local_path(config["run_dir"])


def local_data_dir(config: dict[str, Any]) -> Path:
    return local_path(config["local_data_run_dir"])


def load_config(path: str | Path = "configs/experiments/scale8000_clean_only_2080ti_v1.json") -> dict[str, Any]:
    config = read_json(repo_path(path))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("work_order_id") != "0028":
        raise ValueError("work_order_id must be 0028")
    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("status must be DIAGNOSTIC_ONLY")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    training = config["training"]
    required = {
        "semantic_rows": 64000,
        "exposure_rounds": 9,
        "sample_exposures": 576000,
        "effective_batch_size": 8,
        "optimizer_steps": 72000,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": "none",
        "gradient_clipping": "none",
        "precision": "fp32",
        "tf32": False,
        "spec_augment": False,
        "waveform_augmentation": False,
        "early_stopping": False,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if training.get("physical_microbatch_candidates") != [8, 4, 2, 1]:
        raise ValueError("physical microbatch candidates must be [8, 4, 2, 1]")
    data = config["data"]
    if data.get("text_rows") != 64000 or data.get("clean_views") != 576000 or data.get("augmented_views") != 0:
        raise ValueError("scale-8000 clean-only counts are invalid")
    if data.get("piper_rows") != 64000 or data.get("supertonic_rows") != 512000:
        raise ValueError("scale-8000 clean manifest counts are invalid")
    evaluation = config["evaluation"]
    if evaluation.get("batch_size") != 32 or evaluation.get("duration_bucketing") is not True:
        raise ValueError("directional evaluation must use batch size 32 with bucketing")
    if evaluation.get("canonical") is not False or evaluation.get("promotion_eligible") is not False:
        raise ValueError("scale-8000 clean-only evaluation must be directional only")


def protected_file_fingerprints(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for path_text in config["protected_unchanged_files"]:
        path = repo_path(path_text)
        rel = str(path.relative_to(REPO_ROOT))
        completed = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True)
        output[path_text] = {"git_blob_sha": completed.stdout.strip(), "byte_sha256": file_sha256(path)}
    return output


def verify_protected_file_fingerprints(config: dict[str, Any], expected: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    current = protected_file_fingerprints(config)
    for path, fingerprint in expected.items():
        if current.get(path) != fingerprint:
            raise RuntimeError(f"protected file changed: {path}")
    return current


def verify_scale8000_public_evidence(config: dict[str, Any]) -> dict[str, Any]:
    certificate_path = repo_path(config["scale8000_certificate"])
    report_path = repo_path(config["scale8000_report"])
    cert = read_json(certificate_path)
    report = read_json(report_path)
    data = config["data"]
    if cert.get("corpus_id") != data["corpus_id"] or report.get("corpus_id") != data["corpus_id"]:
        raise RuntimeError("scale-8000 corpus ID mismatch")
    text = cert.get("text_corpus", {})
    if text.get("combined_sha256") != data["text_sha256"] or int(text.get("combined_rows", -1)) != data["text_rows"]:
        raise RuntimeError("scale-8000 text identity mismatch")
    clean = cert.get("clean_audio", {})
    if clean.get("status") != "CLEAN_AUDIO_GENERATED" or clean.get("clean_audio_complete") is not True:
        raise RuntimeError("scale-8000 clean audio is not complete")
    if int(clean.get("generated_clean_files", -1)) != data["clean_views"]:
        raise RuntimeError("scale-8000 clean view count mismatch")
    if int(clean.get("augmented_audio", {}).get("generated_augmented_files", -1)) != 0:
        raise RuntimeError("augmentation must not exist for this clean-only run")
    piper = clean.get("piper", {})
    supertonic = clean.get("supertonic", {})
    if piper.get("audio_manifest_sha256") != data["piper_manifest_sha256"] or int(piper.get("rows", -1)) != data["piper_rows"]:
        raise RuntimeError("scale-8000 Piper manifest identity mismatch")
    if supertonic.get("audio_manifest_sha256") != data["supertonic_manifest_sha256"] or int(supertonic.get("rows", -1)) != data["supertonic_rows"]:
        raise RuntimeError("scale-8000 Supertonic manifest identity mismatch")
    if supertonic.get("native_manifest_sha256") != data["supertonic_native_manifest_sha256"]:
        raise RuntimeError("scale-8000 Supertonic native manifest identity mismatch")
    return {
        "certificate_sha256": file_sha256(certificate_path),
        "report_sha256": file_sha256(report_path),
        "corpus_id": data["corpus_id"],
        "text_sha256": data["text_sha256"],
        "text_rows": data["text_rows"],
        "piper_manifest_sha256": data["piper_manifest_sha256"],
        "supertonic_manifest_sha256": data["supertonic_manifest_sha256"],
        "supertonic_native_manifest_sha256": data["supertonic_native_manifest_sha256"],
        "clean_views": data["clean_views"],
    }


def verify_local_scale8000_inputs(config: dict[str, Any]) -> dict[str, Any]:
    data = config["data"]
    fixed_text = local_path(data["fixed_text"])
    piper_manifest = local_path(data["piper_manifest"])
    super_manifest = local_path(data["supertonic_manifest"])
    super_native = local_path(data["supertonic_native_manifest"])
    if sha256_file(fixed_text) != data["text_sha256"]:
        raise RuntimeError("local scale-8000 fixed text SHA mismatch")
    if sha256_file(piper_manifest) != data["piper_manifest_sha256"]:
        raise RuntimeError("local scale-8000 Piper manifest SHA mismatch")
    if sha256_file(super_manifest) != data["supertonic_manifest_sha256"]:
        raise RuntimeError("local scale-8000 Supertonic manifest SHA mismatch")
    if sha256_file(super_native) != data["supertonic_native_manifest_sha256"]:
        raise RuntimeError("local scale-8000 Supertonic native manifest SHA mismatch")
    text_rows = read_jsonl(fixed_text)
    piper_rows = read_jsonl(piper_manifest)
    super_rows = read_jsonl(super_manifest)
    if len(text_rows) != data["text_rows"] or len(piper_rows) != data["piper_rows"] or len(super_rows) != data["supertonic_rows"]:
        raise RuntimeError("local scale-8000 row-count mismatch")
    return {
        "fixed_text_sha256": sha256_file(fixed_text),
        "piper_manifest_sha256": sha256_file(piper_manifest),
        "supertonic_manifest_sha256": sha256_file(super_manifest),
        "supertonic_native_manifest_sha256": sha256_file(super_native),
        "text_rows": len(text_rows),
        "piper_rows": len(piper_rows),
        "supertonic_rows": len(super_rows),
        "nfs_heavy_inputs_used": False,
    }


def _record_from_piper(row: dict[str, Any]) -> CleanTrainingRecord:
    return CleanTrainingRecord(
        semantic_key=str(row["candidate_id"]),
        voice="piper-sl_SI-artur-medium",
        audio_filepath=str(row["audio_filepath"]),
        duration=float(row["duration_seconds"]),
        text=str(row["text"]),
        text_sha256=str(row["target_text_sha256"]),
        audio_sha256=str(row["audio_sha256"]),
        source_family_id=str(row["source_family_id"]),
        utterance_family_id=str(row["utterance_family_id"]),
    )


def _record_from_supertonic(row: dict[str, Any]) -> CleanTrainingRecord:
    style = str(row["voice_style_id"])
    if style not in TRAINING_STYLES:
        raise RuntimeError("held-out Supertonic style leaked into clean training manifest")
    return CleanTrainingRecord(
        semantic_key=str(row["source_key"]),
        voice=f"supertonic-{style}",
        audio_filepath=str(row["audio_filepath"]),
        duration=float(row["duration_seconds"]),
        text=str(row.get("text") or ""),
        text_sha256=str(row["target_text_sha256"]),
        audio_sha256=str(row["audio_sha256"]),
        source_family_id=str(row["source_family_id"]),
        utterance_family_id=str(row["utterance_family_id"]),
    )


def load_clean_training_bank(config: dict[str, Any], *, validate_audio: bool = False) -> dict[str, dict[str, CleanTrainingRecord]]:
    text_rows = read_jsonl(local_path(config["data"]["fixed_text"]))
    text_by_key = {str(row["candidate_id"]): row for row in text_rows}
    bank: dict[str, dict[str, CleanTrainingRecord]] = {key: {} for key in text_by_key}
    for row in read_jsonl(local_path(config["data"]["piper_manifest"])):
        record = _record_from_piper(row)
        if record.semantic_key not in bank:
            raise RuntimeError("Piper manifest contains unexpected semantic key")
        bank[record.semantic_key][record.voice] = record
    for row in read_jsonl(local_path(config["data"]["supertonic_manifest"])):
        record = _record_from_supertonic(row)
        if not record.text:
            record = CleanTrainingRecord(
                semantic_key=record.semantic_key,
                voice=record.voice,
                audio_filepath=record.audio_filepath,
                duration=record.duration,
                text=str(text_by_key[record.semantic_key]["spoken_text"]),
                text_sha256=record.text_sha256,
                audio_sha256=record.audio_sha256,
                source_family_id=record.source_family_id,
                utterance_family_id=record.utterance_family_id,
            )
        if record.semantic_key not in bank:
            raise RuntimeError("Supertonic manifest contains unexpected semantic key")
        piper = bank[record.semantic_key].get("piper-sl_SI-artur-medium")
        if piper is not None and piper.text_sha256 != record.text_sha256:
            raise RuntimeError("Supertonic/Piper text hash mismatch")
        bank[record.semantic_key][record.voice] = record
    missing = {key: sorted(set(CLEAN_VOICE_ORDER) - set(voices)) for key, voices in bank.items() if set(voices) != set(CLEAN_VOICE_ORDER)}
    if missing:
        first = next(iter(missing.items()))
        raise RuntimeError(f"clean training bank is incomplete for {first[0]}: {first[1]}")
    if validate_audio:
        for voices in bank.values():
            for record in voices.values():
                path = Path(record.audio_filepath)
                validate_wav(path, sample_rate=16000)
                if file_sha256(path) != record.audio_sha256:
                    raise RuntimeError("clean audio hash mismatch")
    return bank


def clean_training_records_for_round(bank: dict[str, dict[str, CleanTrainingRecord]], round_index: int) -> list[CleanTrainingRecord]:
    if round_index < 1 or round_index > len(CLEAN_VOICE_ORDER):
        raise ValueError("round_index must be 1..9")
    voice = CLEAN_VOICE_ORDER[round_index - 1]
    return [bank[key][voice] for key in sorted(bank, key=lambda item: (stable_sha256(item), item))]


def build_clean_exposure_schedule(config: dict[str, Any], bank: dict[str, dict[str, CleanTrainingRecord]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_keys = sorted(bank, key=lambda item: (stable_sha256(item), item))
    schedule: list[dict[str, Any]] = []
    for round_index, voice in enumerate(CLEAN_VOICE_ORDER, start=1):
        for position, key in enumerate(ordered_keys):
            record = bank[key][voice]
            schedule.append(
                {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": key,
                    "voice": voice,
                    "audio_sha256": record.audio_sha256,
                    "text_sha256": record.text_sha256,
                    "duration_seconds": round(record.duration, 6),
                    "view_type": "clean",
                }
            )
    summary = validate_clean_exposure_schedule(schedule, semantic_rows=int(config["training"]["semantic_rows"]))
    return schedule, summary


def validate_clean_exposure_schedule(schedule: Sequence[dict[str, Any]], *, semantic_rows: int = 64000) -> dict[str, Any]:
    if len(schedule) != semantic_rows * 9:
        raise ValueError("scale-8000 clean-only schedule must contain exactly 576000 exposures")
    voice_counts = Counter(str(row["voice"]) for row in schedule)
    round_counts: dict[int, set[str]] = {index: set() for index in range(1, 10)}
    for row in schedule:
        if row.get("view_type") != "clean":
            raise ValueError("hidden augmentation view detected")
        voice = str(row["voice"])
        if voice in {"supertonic-M5", "supertonic-F5", "M5", "F5"}:
            raise ValueError("held-out voice leaked into training schedule")
        round_index = int(row["round"])
        key = str(row["semantic_key"])
        if key in round_counts[round_index]:
            raise ValueError(f"duplicate semantic key in clean round {round_index}")
        round_counts[round_index].add(key)
    issues = []
    for round_index, keys in round_counts.items():
        if len(keys) != semantic_rows:
            issues.append(f"round_{round_index}_count")
    for voice in CLEAN_VOICE_ORDER:
        if voice_counts[voice] != semantic_rows:
            issues.append(f"voice_{voice}_count")
    if issues:
        raise ValueError(f"invalid clean-only exposure schedule: {issues}")
    return {
        "status": "PASSED",
        "semantic_rows": semantic_rows,
        "exposures": len(schedule),
        "rounds": 9,
        "clean_views_per_semantic_row": 9,
        "augmented_views": 0,
        "effective_batch_size": 8,
        "optimizer_steps": len(schedule) // 8,
        "voice_counts": {voice: voice_counts[voice] for voice in CLEAN_VOICE_ORDER},
        "heldout_voice_exposures": {"supertonic-M5": voice_counts.get("supertonic-M5", 0), "supertonic-F5": voice_counts.get("supertonic-F5", 0)},
    }


def write_clean_exposure_schedule(path: Path, schedule: Sequence[dict[str, Any]]) -> str:
    atomic_write_jsonl(path, schedule)
    return file_sha256(path)


def microbatch_plan(physical_microbatch: int) -> dict[str, int]:
    if physical_microbatch not in {1, 2, 4, 8}:
        raise ValueError("physical microbatch must be one of 1,2,4,8")
    if 8 % physical_microbatch != 0:
        raise ValueError("physical microbatch must divide 8")
    return {"physical_microbatch": physical_microbatch, "gradient_accumulation_steps": 8 // physical_microbatch, "effective_batch_size": 8}


def burden(metrics: dict[str, dict[str, Any]], base_metrics: dict[str, dict[str, Any]] = BASE_DIRECTIONAL_METRICS) -> float:
    value = 0.0
    for split in ("fleurs_v2", "artur_j"):
        value += max(0.0, float(metrics[split]["wer"]) - float(base_metrics[split]["wer"]))
        value += max(0.0, float(metrics[split]["cer"]) - float(base_metrics[split]["cer"]))
    return round(value, 6)


def classify_scale8000_clean(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    piper_gain = (
        float(metrics["piper_synthetic_holdout"]["wer"]) < BASE_DIRECTIONAL_METRICS["piper_synthetic_holdout"]["wer"]
        or float(metrics["piper_synthetic_holdout"]["cer"]) < BASE_DIRECTIONAL_METRICS["piper_synthetic_holdout"]["cer"]
    )
    super_gain = (
        float(metrics["supertonic_heldout_voice_holdout"]["wer"]) < BASE_DIRECTIONAL_METRICS["supertonic_heldout_voice_holdout"]["wer"]
        or float(metrics["supertonic_heldout_voice_holdout"]["cer"]) < BASE_DIRECTIONAL_METRICS["supertonic_heldout_voice_holdout"]["cer"]
    )
    real_burden = burden(metrics)
    real_improvements = 0
    no_real_metric_worse = True
    real_within_half = True
    for split in ("fleurs_v2", "artur_j"):
        for metric in ("wer", "cer"):
            delta = float(metrics[split][metric]) - float(SCALE2000_DIRECTIONAL_METRICS[split][metric])
            if delta < 0:
                real_improvements += 1
            if delta > 0.5:
                no_real_metric_worse = False
            if abs(delta) > 0.5:
                real_within_half = False
    synthetic_no_more_than_one_worse = True
    synthetic_within_one = True
    for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout"):
        for metric in ("wer", "cer"):
            delta = float(metrics[split][metric]) - float(SCALE2000_DIRECTIONAL_METRICS[split][metric])
            if delta > 1.0:
                synthetic_no_more_than_one_worse = False
            if abs(delta) > 1.0:
                synthetic_within_one = False
    if piper_gain and super_gain and real_burden == 0.0 and real_improvements >= 2 and no_real_metric_worse and synthetic_no_more_than_one_worse:
        classification = "SCALE8000_CLEAN_BEATS_SCALE2000_AUGMENTED_DIRECTIONAL"
    elif piper_gain and super_gain and real_burden == 0.0 and real_within_half and synthetic_within_one:
        classification = "SCALE8000_CLEAN_MATCHES_SCALE2000_AUGMENTED_DIRECTIONAL"
    elif piper_gain and super_gain and real_burden == 0.0:
        classification = "SCALE8000_CLEAN_BEATS_BASE_BUT_NOT_SCALE2000"
    else:
        classification = "SCALE8000_CLEAN_UNDERPERFORMS_DIRECTIONALLY"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "piper_holdout_gain": piper_gain,
        "supertonic_holdout_gain": super_gain,
        "real_burden": real_burden,
        "real_metrics_improved_vs_scale2000": real_improvements,
        "no_real_metric_more_than_half_point_worse_than_scale2000": no_real_metric_worse,
        "synthetic_holdouts_no_more_than_one_point_worse_than_scale2000": synthetic_no_more_than_one_worse,
    }


def assert_public_report_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in PUBLIC_FORBIDDEN_KEYS:
                    raise ValueError(f"public report contains forbidden key: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(payload)
    for marker in PUBLIC_FORBIDDEN_MARKERS:
        if marker in serialized:
            raise ValueError(f"public report contains forbidden marker: {marker}")


def scale8000_synthetic_holdout_records(config: dict[str, Any]) -> list[StreamingRecord]:
    manifest = local_path(config["data"]["piper_synthetic_holdout_manifest"])
    rows = read_jsonl(manifest)
    if len(rows) != int(config["data"]["piper_synthetic_holdout_rows"]):
        raise RuntimeError("Piper synthetic holdout row count mismatch")
    records = []
    for index, row in enumerate(rows):
        path = resolve_manifest_audio_path(manifest, str(row["audio_filepath"]))
        validate_wav(path, sample_rate=16000)
        if file_sha256(path) != str(row["audio_sha256"]):
            raise RuntimeError("Piper synthetic holdout audio SHA mismatch")
        records.append(StreamingRecord(sample_id=f"piper_synthetic_holdout:{index:04d}", audio_filepath=str(path), duration=float(row["duration_seconds"]), reference=str(row["text"]), original_index=index, row={"split": "piper_synthetic_holdout"}))
    return records


def scale8000_supertonic_heldout_records(config: dict[str, Any]) -> list[StreamingRecord]:
    manifest = local_path(config["data"]["supertonic_heldout_manifest"])
    rows = read_jsonl(manifest)
    holdout_references = {
        str(row["candidate_id"]): {
            "text": str(row["text"]),
            "target_text_sha256": str(row["target_text_sha256"]),
        }
        for row in read_jsonl(local_path(config["data"]["piper_synthetic_holdout_manifest"]))
    }
    records = []
    for row in rows:
        voice = str(row.get("voice_style_id"))
        if voice not in {"M5", "F5"}:
            continue
        source_key = str(row["source_key"])
        reference = holdout_references.get(source_key)
        if not reference:
            raise RuntimeError(f"Supertonic held-out reference missing for source key {source_key}")
        if reference["target_text_sha256"] != str(row["target_text_sha256"]):
            raise RuntimeError(f"Supertonic held-out text hash mismatch for source key {source_key}")
        path = resolve_manifest_audio_path(manifest, str(row["audio_filepath"]))
        validate_wav(path, sample_rate=16000)
        if file_sha256(path) != str(row["audio_sha256"]):
            raise RuntimeError("Supertonic held-out audio SHA mismatch")
        records.append(
            StreamingRecord(
                sample_id=f"supertonic_heldout_voice_holdout:{len(records):04d}",
                audio_filepath=str(path),
                duration=float(row["duration_seconds"]),
                reference=reference["text"],
                original_index=len(records),
                row={"split": "supertonic_heldout_voice_holdout", "voice_style_id": voice},
            )
        )
    if len(records) != int(config["data"]["supertonic_heldout_rows"]):
        raise RuntimeError(f"expected 192 Supertonic held-out records, found {len(records)}")
    return records


def directional_suite(config: dict[str, Any]) -> tuple[list[StreamingRecord], dict[str, list[StreamingRecord]]]:
    split_records = {
        "piper_synthetic_holdout": scale8000_synthetic_holdout_records(config),
        "supertonic_heldout_voice_holdout": scale8000_supertonic_heldout_records(config),
        "fleurs_v2": load_gate_records(
            local_path(config["data"]["fleurs_v2_manifest"]),
            expected_sha256=config["data"]["fleurs_v2_manifest_sha256"],
            expected_rows=int(config["data"]["fleurs_v2_rows"]),
            gate_id="fleurs-sl-si-test-full-v2",
        ),
        "artur_j": load_gate_records(
            local_path(config["data"]["artur_j_manifest"]),
            expected_sha256=config["data"]["artur_j_manifest_sha256"],
            expected_rows=int(config["data"]["artur_j_rows"]),
            gate_id="artur-j-public-gate-v1",
        ),
    }
    suite = []
    for split, records in split_records.items():
        for index, record in enumerate(records):
            suite.append(
                StreamingRecord(
                    sample_id=f"{split}:{index:04d}",
                    audio_filepath=record.audio_filepath,
                    duration=record.duration,
                    reference=record.reference,
                    original_index=len(suite),
                    row={"split": split, "source_order": index},
                )
            )
    if len(suite) != 1378:
        raise RuntimeError(f"expected 1378 directional rows, found {len(suite)}")
    return suite, split_records


def metric_row(split_summary: dict[str, Any]) -> dict[str, Any]:
    normalized = split_summary["metrics"]["normalized"]
    raw = split_summary["metrics"]["raw"]
    return {
        "wer": round(float(normalized["corpus_wer"]), 3),
        "cer": round(float(normalized["corpus_cer"]), 3),
        "empty": int(raw["empty_hypothesis_count"]),
    }
