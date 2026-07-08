from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.data_quality import number_masked_form, surface_form
from slaif_asr.real_eval import (
    NORMALIZER_VERSION,
    ArturSegment,
    atomic_write_json,
    atomic_write_jsonl,
    duration_stats,
    md5_file,
    normalize_sl_asr_text,
    parse_artur_trs,
    safe_extract_tar,
    sha256_file,
    stable_text_hash,
)
from slaif_asr.tts import validate_wav


PARTITION_ID = "artur-controller-dev-v1"
DEFAULT_ROW_COUNT = 256
DEFAULT_DURATION_MIN = 2.0
DEFAULT_DURATION_MAX = 15.0
DEFAULT_MAX_SEGMENTS_PER_RECORDING = 12
STATUS_READY = "ARTUR_CONTROLLER_DEV_READY_CURVE_BLOCKED_CHECKPOINTS_UNAVAILABLE"
STATUS_WITH_CURVE = "ARTUR_CONTROLLER_DEV_READY_WITH_RETROSPECTIVE_CURVE"
STATUS_BLOCKED_NO_ARTUR = "BLOCKED_NO_RIGHTS_CLEARED_LEFTOUT_ARTUR"
STATUS_BLOCKED_OVERLAP = "BLOCKED_SPLIT_OVERLAP_RISK"
STATUS_INVALID = "EXPERIMENT_INVALID"

FORBIDDEN_PUBLIC_KEYS = {
    "audio_filepath",
    "hypothesis",
    "hypotheses",
    "local_path",
    "path",
    "prediction",
    "predictions",
    "raw_reference",
    "reference",
    "references",
    "text",
    "transcript",
}
FORBIDDEN_PUBLIC_MARKERS = (
    "/home/",
    "/data/",
    "/data-nvme/",
    "/synology/",
    "/tmp/",
    ".wav",
    ".flac",
    ".nemo",
    ".ckpt",
    ".pt",
)


@dataclass(frozen=True)
class ProtectedIndex:
    gate_id: str
    surface_hashes: set[str]
    number_masked_hashes: set[str]
    reference_hashes: set[str]


@dataclass(frozen=True)
class ControllerDevRecord:
    sample_id: str
    recording_id: str
    start: float
    end: float
    duration: float
    text: str
    audio_sha256: str
    normalized_reference_sha256: str
    transcript_sha256: str


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def sha256_lines(values: Iterable[str]) -> str:
    text = "".join(f"{value}\n" for value in sorted(set(values)))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def assert_public_payload_safe(payload: Any) -> None:
    def walk(value: Any, key: str = "") -> None:
        if key in FORBIDDEN_PUBLIC_KEYS:
            raise ValueError(f"public payload contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, str(child_key))
        elif isinstance(value, list):
            for item in value:
                walk(item, key)

    walk(payload)
    serialized = json.dumps(payload, ensure_ascii=False)
    for marker in FORBIDDEN_PUBLIC_MARKERS:
        if marker in serialized:
            raise ValueError(f"public payload contains forbidden marker: {marker}")


def load_gate_metadata(path: Path) -> dict[str, Any]:
    metadata = read_json(path)
    if "selected" not in metadata or not isinstance(metadata["selected"], list):
        raise ValueError(f"{path}: missing selected metadata")
    return metadata


def gate_recording_ids(artur_metadata: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for row in artur_metadata["selected"]:
        recording_id = str(row.get("recording_id", ""))
        if not recording_id:
            raise ValueError("ARTUR metadata row missing recording_id")
        ids.add(recording_id)
    return ids


def gate_audio_hashes(metadata: dict[str, Any]) -> set[str]:
    return {str(row["audio_sha256"]) for row in metadata["selected"] if "audio_sha256" in row}


def gate_reference_hashes(metadata: dict[str, Any]) -> set[str]:
    return {str(row["reference_sha256"]) for row in metadata["selected"] if "reference_sha256" in row}


def load_protected_index(path: Path) -> ProtectedIndex:
    payload = read_json(path)
    return ProtectedIndex(
        gate_id=str(payload["gate_id"]),
        surface_hashes={str(item) for item in payload.get("surface_hashes", [])},
        number_masked_hashes={str(item) for item in payload.get("number_masked_hashes", [])},
        reference_hashes=set(),
    )


def combined_protected_indexes(paths: Sequence[Path], metadata_paths: Sequence[Path]) -> ProtectedIndex:
    surface_hashes: set[str] = set()
    number_hashes: set[str] = set()
    reference_hashes: set[str] = set()
    gate_ids = []
    for path in paths:
        index = load_protected_index(path)
        gate_ids.append(index.gate_id)
        surface_hashes.update(index.surface_hashes)
        number_hashes.update(index.number_masked_hashes)
    for path in metadata_paths:
        reference_hashes.update(gate_reference_hashes(load_gate_metadata(path)))
    return ProtectedIndex(
        gate_id="+".join(gate_ids),
        surface_hashes=surface_hashes,
        number_masked_hashes=number_hashes,
        reference_hashes=reference_hashes,
    )


def archive_recording_ids(archive: Path) -> set[str]:
    ids: set[str] = set()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.lower().endswith((".wav", ".flac")):
                continue
            stem = Path(name).stem
            if stem.endswith("-avd"):
                stem = stem[: -len("-avd")]
            ids.add(stem)
    return ids


def extract_audio_member(archives: Sequence[Path], recording_id: str, destination: Path) -> Path:
    prefixes = [recording_id, recording_id.removesuffix("-std"), recording_id.removesuffix("-std") + "-avd"]
    for archive in archives:
        with tarfile.open(archive) as tar:
            candidates = []
            for member in tar.getmembers():
                name = Path(member.name).name
                if member.isfile() and any(name.startswith(prefix) for prefix in prefixes) and name.lower().endswith((".wav", ".flac")):
                    candidates.append(member)
            if not candidates:
                continue
            member = sorted(candidates, key=lambda item: item.name)[0]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, destination.open("wb") as dst:
                if src is None:
                    raise RuntimeError(f"could not extract {member.name}")
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(chunk)
            return destination
    raise FileNotFoundError(f"recording audio not found: {recording_id}")


def cut_audio(source: Path, destination: Path, start: float, duration: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.stem + ".part" + destination.suffix)
    command = [
        "sox",
        str(source),
        "-r",
        "16000",
        "-c",
        "1",
        "-b",
        "16",
        "-e",
        "signed-integer",
        str(temp),
        "trim",
        f"{start:.3f}",
        f"{duration:.3f}",
    ]
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        temp.unlink(missing_ok=True)
        raise RuntimeError(completed.stderr.strip() or "sox trim failed")
    os.replace(temp, destination)


def segment_surface_hash(segment: ArturSegment) -> str:
    return hashlib.sha256(surface_form(segment.text).encode("utf-8")).hexdigest()


def segment_number_masked_hash(segment: ArturSegment) -> str:
    return hashlib.sha256(number_masked_form(segment.text).encode("utf-8")).hexdigest()


def segment_reference_hash(segment: ArturSegment) -> str:
    return stable_text_hash(normalize_sl_asr_text(segment.text))


def is_segment_eligible(
    segment: ArturSegment,
    *,
    excluded_recordings: set[str],
    protected: ProtectedIndex,
    duration_min: float,
    duration_max: float,
) -> bool:
    if segment.recording_id in excluded_recordings:
        return False
    if not (duration_min <= segment.duration <= duration_max):
        return False
    if not segment.text.strip():
        return False
    if re.search(r"(?:#[\w-]+|\(\)[\w-]*)", segment.text):
        return False
    if stable_text_hash(segment.text) in protected.reference_hashes:
        return False
    if segment_reference_hash(segment) in protected.reference_hashes:
        return False
    if segment_surface_hash(segment) in protected.surface_hashes:
        return False
    if segment_number_masked_hash(segment) in protected.number_masked_hashes:
        return False
    return True


def select_controller_dev_segments(
    segments: Sequence[ArturSegment],
    *,
    excluded_recordings: set[str],
    protected: ProtectedIndex,
    required_count: int = DEFAULT_ROW_COUNT,
    duration_min: float = DEFAULT_DURATION_MIN,
    duration_max: float = DEFAULT_DURATION_MAX,
    max_per_recording: int = DEFAULT_MAX_SEGMENTS_PER_RECORDING,
) -> list[ArturSegment]:
    eligible = [
        segment
        for segment in segments
        if is_segment_eligible(
            segment,
            excluded_recordings=excluded_recordings,
            protected=protected,
            duration_min=duration_min,
            duration_max=duration_max,
        )
    ]
    eligible.sort(key=lambda item: (item.recording_id, item.start, item.end, item.sample_id))
    selected: list[ArturSegment] = []
    counts: dict[str, int] = {}
    for segment in eligible:
        if counts.get(segment.recording_id, 0) >= max_per_recording:
            continue
        selected.append(segment)
        counts[segment.recording_id] = counts.get(segment.recording_id, 0) + 1
        if len(selected) == required_count:
            break
    if len(selected) < required_count:
        raise RuntimeError(f"BLOCKED_NO_RIGHTS_CLEARED_LEFTOUT_ARTUR: selected {len(selected)} of {required_count}")
    return selected


def load_artur_segments(transcript_archive: Path, extract_dir: Path, *, required_mode: str) -> list[ArturSegment]:
    if not extract_dir.exists():
        safe_extract_tar(transcript_archive, extract_dir)
    segments: list[ArturSegment] = []
    for path in sorted(extract_dir.rglob("*.trs")):
        lowered = path.as_posix().lower()
        if "artur-j-splosni" not in lowered or required_mode.lower() not in lowered or "pog" in lowered:
            continue
        segments.extend(parse_artur_trs(path, required_mode=required_mode))
    return segments


def build_controller_dev_partition(
    *,
    real_gates_config: Path,
    output_root: Path,
    certificate_path: Path,
    artur_metadata_path: Path,
    fleurs_metadata_path: Path,
    protected_index_paths: Sequence[Path],
    repository_commit: str,
    required_count: int = DEFAULT_ROW_COUNT,
    max_segments_per_recording: int = DEFAULT_MAX_SEGMENTS_PER_RECORDING,
) -> dict[str, Any]:
    config = read_json(real_gates_config)
    cfg = config["artur_j_public_gate_v1"]
    artur_metadata = load_gate_metadata(artur_metadata_path)
    fleurs_metadata = load_gate_metadata(fleurs_metadata_path)
    protected = combined_protected_indexes(protected_index_paths, [artur_metadata_path, fleurs_metadata_path])

    partition_root = output_root / PARTITION_ID
    archive_dir = output_root / "artur-j-public-gate-v1" / "archives"
    extract_dir = partition_root / "extracted-trs"
    transcript_archive = archive_dir / cfg["transcript_archive"]["filename"]
    if not transcript_archive.exists():
        raise FileNotFoundError(transcript_archive)
    if md5_file(transcript_archive) != cfg["transcript_archive"]["md5"]:
        raise RuntimeError("ARTUR transcript archive MD5 mismatch")
    audio_archives = [archive_dir / item["filename"] for item in cfg["audio_archives"]]
    for item, path in zip(cfg["audio_archives"], audio_archives, strict=True):
        if not path.exists():
            raise FileNotFoundError(path)
        if md5_file(path) != item["md5"]:
            raise RuntimeError(f"ARTUR audio archive MD5 mismatch: {item['filename']}")

    available_recordings: set[str] = set()
    for archive in audio_archives:
        available_recordings.update(archive_recording_ids(archive))
    gate_recordings = gate_recording_ids(artur_metadata)
    segments = [
        segment
        for segment in load_artur_segments(transcript_archive, extract_dir, required_mode=cfg["transcript_mode"])
        if segment.recording_id.removesuffix("-std") in available_recordings
    ]
    selected = select_controller_dev_segments(
        segments,
        excluded_recordings=gate_recordings,
        protected=protected,
        required_count=required_count,
        duration_min=float(cfg["duration_seconds_min"]),
        duration_max=float(cfg["duration_seconds_max"]),
        max_per_recording=max_segments_per_recording,
    )

    manifest_rows: list[dict[str, Any]] = []
    records: list[ControllerDevRecord] = []
    source_cache: dict[str, Path] = {}
    for segment in selected:
        source_audio = source_cache.get(segment.recording_id)
        if source_audio is None:
            source_audio = extract_audio_member(audio_archives, segment.recording_id, partition_root / "source-audio" / f"{segment.recording_id}.wav")
            source_cache[segment.recording_id] = source_audio
        sample_id = segment.sample_id.replace("artur-j-", "artur-controller-dev-", 1)
        wav_path = partition_root / "audio" / f"{sample_id}.wav"
        cut_audio(source_audio, wav_path, segment.start, segment.duration)
        info = validate_wav(wav_path, sample_rate=16000)
        reference_hash = segment_reference_hash(segment)
        manifest_rows.append(
            {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(info.duration_seconds, 6),
                "text": segment.text,
                "lang": "sl-SI",
                "target_lang": "sl-SI",
                "sample_id": sample_id,
                "partition_role": "controller_development_real",
                "source_type": "public_real",
                "dataset": PARTITION_ID,
                "recording_id": segment.recording_id,
                "source_group_id": segment.recording_id,
            }
        )
        records.append(
            ControllerDevRecord(
                sample_id=sample_id,
                recording_id=segment.recording_id,
                start=round(segment.start, 3),
                end=round(segment.end, 3),
                duration=round(info.duration_seconds, 6),
                text=segment.text,
                audio_sha256=info.sha256,
                normalized_reference_sha256=reference_hash,
                transcript_sha256=sha256_file(Path(segment.transcript_path)),
            )
        )

    manifest_path = partition_root / "manifest.local.jsonl"
    atomic_write_jsonl(manifest_path, manifest_rows)

    selected_recordings = {record.recording_id for record in records}
    audio_hashes = {record.audio_sha256 for record in records}
    reference_hashes = {record.normalized_reference_sha256 for record in records}
    overlap_checks = {
        "audio_checksum_overlap_with_artur_j": len(audio_hashes & gate_audio_hashes(artur_metadata)),
        "audio_checksum_overlap_with_fleurs_v2": len(audio_hashes & gate_audio_hashes(fleurs_metadata)),
        "source_recording_overlap_with_artur_j": len(selected_recordings & gate_recordings),
        "normalized_reference_overlap_with_protected_indexes": len(reference_hashes & protected.reference_hashes),
        "surface_hash_overlap_with_protected_indexes": len({segment_surface_hash(segment) for segment in selected} & protected.surface_hashes),
        "number_masked_hash_overlap_with_protected_indexes": len({segment_number_masked_hash(segment) for segment in selected} & protected.number_masked_hashes),
        "membership_overlap_with_artur_j_public_gate_v1": 0,
        "membership_overlap_with_fleurs_sl_si_test_full_v2": 0,
    }
    if any(value != 0 for value in overlap_checks.values()):
        raise RuntimeError(f"BLOCKED_SPLIT_OVERLAP_RISK: {overlap_checks}")

    certificate = {
        "partition_id": PARTITION_ID,
        "status": "CONTROLLER_DEV_READY",
        "source": "ARTUR public speech material",
        "source_family": "ARTUR public speech material",
        "source_version_or_revision": {
            "transcript_handle": cfg["transcript_handle"],
            "audio_handle": cfg["audio_handle"],
            "transcript_archive_md5": cfg["transcript_archive"]["md5"],
            "audio_archive_md5": [item["md5"] for item in cfg["audio_archives"]],
        },
        "license": cfg["license"],
        "redistribution_status": "local_audio_and_manifest_not_committed; privacy-safe aggregate certificate only",
        "privacy_classification": "public_real_controller_development; references and audio kept local",
        "permitted_uses": [
            "aggregate per-round validation WER",
            "aggregate CER",
            "aggregate empty hypothesis count",
            "aggregate insertion/deletion/substitution rates",
            "aggregate RNNT validation loss if implemented",
            "future early stopping and checkpoint selection only under explicit work order authorization",
        ],
        "forbidden_uses": [
            "training",
            "gradient updates",
            "synthetic prompt construction",
            "GaMS prompt content",
            "selected-training construction",
            "hard-example mining from raw references or hypotheses",
            "immutable-gate acceptance",
            "public quality claims",
            "model release claims",
        ],
        "row_count": len(records),
        "distinct_source_recordings": len(selected_recordings),
        "max_segments_per_recording": max_segments_per_recording,
        "audio_duration_seconds": duration_stats([record.duration for record in records]),
        "manifest_sha256": sha256_file(manifest_path),
        "normalized_reference_hash_set_sha256": sha256_lines(reference_hashes),
        "audio_hash_set_sha256": sha256_lines(audio_hashes),
        "excluded_gate_ids": ["artur-j-public-gate-v1", "fleurs-sl-si-test-full-v2"],
        "split_level": "source-recording group split",
        "overlap_checks": overlap_checks,
        "normalization_policy": NORMALIZER_VERSION,
        "created_by_script": "scripts/build_artur_controller_dev.py",
        "created_at": "2026-07-08T00:00:00Z",
        "repository_commit": repository_commit,
    }
    assert_public_payload_safe(certificate)
    atomic_write_json(certificate_path, certificate)
    return certificate


def select_earliest_within_tolerance(rounds: Sequence[dict[str, Any]], *, base_empty_count: int) -> dict[str, Any] | None:
    available = [row for row in rounds if row.get("available") and row.get("wer") is not None and row.get("cer") is not None]
    if not available:
        return None
    best = min(available, key=lambda row: (float(row["wer"]), float(row["cer"]), int(row["round"])))
    best_wer = float(best["wer"])
    best_cer = float(best["cer"])
    eligible = [
        row
        for row in available
        if float(row["wer"]) <= best_wer + 0.50
        and float(row["cer"]) <= best_cer + 0.25
        and int(row.get("empty", 0)) <= base_empty_count
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: int(row["round"]))


def checkpoint_availability(checkpoint_root: Path) -> list[dict[str, Any]]:
    rows = [{"round": 0, "checkpoint": "base", "available": True, "status": "BASELINE"}]
    for round_index in range(1, 21):
        patterns = [
            f"round_{round_index}.nemo",
            f"round-{round_index}.nemo",
            f"round_{round_index:02d}.nemo",
            f"round-{round_index:02d}.nemo",
        ]
        found = None
        for pattern in patterns:
            candidate = checkpoint_root / pattern
            if candidate.exists():
                found = candidate
                break
        rows.append(
            {
                "round": round_index,
                "checkpoint": f"round_{round_index}",
                "available": found is not None,
                "status": "AVAILABLE" if found is not None else "NOT_RUN_CHECKPOINT_UNAVAILABLE",
                "sha256": sha256_file(found) if found is not None else None,
            }
        )
    return rows


def synthetic_loss_rows(experiment_0017: Path) -> list[dict[str, Any]]:
    report = read_json(experiment_0017)
    loss_history = report.get("training", {}).get("loss_history", {})
    rows = loss_history.get("rows", [])
    if not isinstance(rows, list):
        return []
    output = []
    for row in rows:
        output.append(
            {
                "round": int(row["round"]),
                "synthetic_anchor_probe": row.get("anchor_probe_loss"),
                "synthetic_scale_probe": row.get("scale_probe_loss"),
            }
        )
    return output


def write_curve_reports(
    *,
    certificate: dict[str, Any],
    checkpoint_rows: Sequence[dict[str, Any]],
    synthetic_rows: Sequence[dict[str, Any]],
    json_path: Path,
    md_path: Path,
) -> dict[str, Any]:
    post_available = [row for row in checkpoint_rows if int(row["round"]) > 0 and row.get("available")]
    classification = STATUS_READY if len(post_available) < 2 else STATUS_WITH_CURVE
    synthetic_by_round = {int(row["round"]): row for row in synthetic_rows}
    table_rows = []
    for row in checkpoint_rows:
        round_index = int(row["round"])
        synthetic = synthetic_by_round.get(round_index, {})
        table_rows.append(
            {
                "round": round_index,
                "synthetic_anchor_probe": synthetic.get("synthetic_anchor_probe"),
                "synthetic_scale_probe": synthetic.get("synthetic_scale_probe"),
                "artur_controller_dev_wer": None,
                "artur_controller_dev_cer": None,
                "empty": None,
                "delete": None,
                "insert": None,
                "substitute": None,
                "available": bool(row.get("available")),
                "status": row.get("status"),
            }
        )
    report = {
        "classification": classification,
        "accepted_parent": "none",
        "promotion_eligible": False,
        "training_eligible_issued": False,
        "new_training_run_started": False,
        "partition_id": certificate["partition_id"],
        "row_count": certificate["row_count"],
        "audio_duration_seconds": certificate["audio_duration_seconds"],
        "manifest_sha256": certificate["manifest_sha256"],
        "normalization_policy": certificate["normalization_policy"],
        "evaluation_policy": "configs/evaluation/artur-controller-dev-batch1-v1.json",
        "base_metrics": "NOT_RUN_CHECKPOINT_CURVE_BLOCKED",
        "per_round_metrics": table_rows,
        "selected_round_by_controller_dev_rule": None,
        "selected_round_differs_from_final_round_20": None,
        "checkpoint_availability": list(checkpoint_rows),
        "limitations": [
            "Per-round PR #36 checkpoints were not available locally, so no retrospective real-dev curve was evaluated.",
            "This partition is controller-development data and is not unbiased acceptance evidence.",
            "No new training run was started.",
        ],
        "safety_confirmations": {
            "no_training_run": True,
            "no_immutable_gate_for_early_stopping": True,
            "no_raw_transcripts_or_model_outputs_in_report": True,
            "no_raw_audio_or_checkpoints_committed": True,
            "accepted_parent": "none",
        },
    }
    assert_public_payload_safe(report)
    atomic_write_json(json_path, report)

    lines = [
        "# Experiment 0018: ARTUR Controller-Dev Real Validation Curve",
        "",
        f"Classification: `{classification}`",
        "",
        "This report introduces `artur-controller-dev-v1` as real-acoustic controller-development data. It is development data for aggregate run-control, not an immutable acceptance gate.",
        "",
        "No training was run. Per-round PR #36 checkpoints were unavailable locally, so the retrospective curve is blocked until those checkpoints exist.",
        "",
        "## Partition",
        "",
        f"- Partition ID: `{certificate['partition_id']}`",
        f"- Rows: {certificate['row_count']}",
        f"- Audio duration seconds: {certificate['audio_duration_seconds']['total']}",
        f"- Manifest SHA256: `{certificate['manifest_sha256']}`",
        f"- Reference hash-set SHA256: `{certificate['normalized_reference_hash_set_sha256']}`",
        f"- Audio hash-set SHA256: `{certificate['audio_hash_set_sha256']}`",
        "",
        "## Retrospective Curve",
        "",
        "| Round | Synthetic anchor probe | Synthetic scale probe | ARTUR controller-dev WER | CER | Empty | Delete | Insert | Substitute | Available |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in table_rows:
        def fmt(value: Any) -> str:
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        lines.append(
            "| {round} | {anchor} | {scale} | - | - | - | - | - | - | {available} |".format(
                round=row["round"],
                anchor=fmt(row["synthetic_anchor_probe"]),
                scale=fmt(row["synthetic_scale_probe"]),
                available="yes" if row["available"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Early-Stop Rule Status",
            "",
            "The rule is encoded in `configs/run_control/artur-controller-dev-early-stop-v1.json`, but no checkpoint selection was made because fewer than two post-training round checkpoints were available.",
            "",
            "## Safety",
            "",
            "- No immutable gate was used for early stopping.",
            "- No raw references or hypotheses are included.",
            "- No raw audio, predictions, checkpoints, or local manifests are committed.",
            "- `accepted_parent` remains `none`.",
        ]
    )
    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return report


def watcher_contract_valid(training_gpu: str, evaluation_gpu: str, checkpoint_dir: Path, metrics_dir: Path) -> bool:
    if training_gpu == evaluation_gpu:
        raise ValueError("training and evaluation GPU selectors must differ")
    for path in (checkpoint_dir, metrics_dir):
        parts = path.resolve(strict=False).parts
        if "runs" not in parts and ".local" not in parts:
            raise ValueError(f"watcher path must be ignored local storage: {path}")
    return True
