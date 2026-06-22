from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import tarfile
import tempfile
import unicodedata
import wave
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slaif_asr.metrics import corpus_metric_summary
from slaif_asr.tts import sha256_file, validate_wav


NORMALIZER_VERSION = "sl-asr-normalization-v1"
PUNCTUATION_PATTERN = re.compile(r"[^\w\sčšžćđČŠŽĆĐ]", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
SAFE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
UNRESOLVED_TRANSCRIPT_TAG_PATTERN = re.compile(r"(?:#[\w-]+|\(\)[\w-]*)")


@dataclass(frozen=True)
class ArturSegment:
    sample_id: str
    recording_id: str
    start: float
    end: float
    text: str
    transcript_path: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class GateSummary:
    gate_id: str
    sample_count: int
    total_duration_seconds: float
    manifest_sha256: str
    metadata_path: Path


def normalize_sl_asr_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'").replace("´", "'")
    text = text.replace("-", " ")
    text = text.lower()
    text = PUNCTUATION_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def stable_text_hash(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fp:
        fp.write(text)
        temp_name = fp.name
    os.replace(temp_name, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_safe_member(name: str) -> None:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        for member in members:
            assert_safe_member(member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"unexpected archive link: {member.name}")
        tar.extractall(destination, members=members)


def parse_artur_trs(path: Path, *, required_mode: str = "std") -> list[ArturSegment]:
    lowered = path.as_posix().lower()
    if "pog" in lowered:
        raise ValueError(f"pronunciation transcript is forbidden: {path}")
    if required_mode and required_mode.lower() not in lowered:
        raise ValueError(f"expected {required_mode} transcript path: {path}")
    if "artur-j-splosni" not in lowered:
        raise ValueError(f"expected Artur-J-Splosni transcript path: {path}")

    tree = ET.parse(path)
    root = tree.getroot()
    recording_id = path.stem
    segments: list[ArturSegment] = []
    previous_end = -1.0
    for turn_index, turn in enumerate(root.iter("Turn")):
        syncs = list(turn.iter("Sync"))
        if len(syncs) < 2:
            continue
        pieces: list[tuple[float, str]] = []
        for sync in syncs:
            time_text = sync.attrib.get("time")
            if time_text is None:
                raise ValueError(f"{path}: Sync without time")
            pieces.append((float(time_text), sync.tail or ""))
        for index, (start, text) in enumerate(pieces[:-1]):
            end = pieces[index + 1][0]
            cleaned = WHITESPACE_PATTERN.sub(" ", text).strip()
            if not cleaned:
                continue
            if end <= start:
                raise ValueError(f"{path}: non-positive interval {start}..{end}")
            if start < previous_end:
                raise ValueError(f"{path}: overlapping interval at {start}")
            previous_end = end
            sample_id = f"artur-j-{recording_id}-{turn_index:04d}-{index:04d}".lower()
            sample_id = re.sub(r"[^a-z0-9._-]+", "-", sample_id).strip("-")
            if not SAFE_ID_PATTERN.fullmatch(sample_id):
                raise ValueError(f"{path}: unsafe derived sample id {sample_id}")
            segments.append(
                ArturSegment(
                    sample_id=sample_id,
                    recording_id=recording_id,
                    start=start,
                    end=end,
                    text=unicodedata.normalize("NFC", cleaned),
                    transcript_path=path.as_posix(),
                )
            )
    return segments


def select_artur_segments(
    segments: list[ArturSegment],
    *,
    required_count: int,
    duration_min: float,
    duration_max: float,
    max_per_recording: int,
) -> list[ArturSegment]:
    eligible = [
        item
        for item in segments
        if duration_min <= item.duration <= duration_max
        and item.text.strip()
        and not UNRESOLVED_TRANSCRIPT_TAG_PATTERN.search(item.text)
    ]
    eligible.sort(key=lambda item: (item.recording_id, item.start, item.end, item.sample_id))
    counts: dict[str, int] = {}
    selected: list[ArturSegment] = []
    for segment in eligible:
        count = counts.get(segment.recording_id, 0)
        if count >= max_per_recording:
            continue
        selected.append(segment)
        counts[segment.recording_id] = count + 1
        if len(selected) == required_count:
            break
    return selected


def duration_stats(durations: list[float]) -> dict[str, float]:
    if not durations:
        return {"count": 0, "total": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": len(durations),
        "total": round(sum(durations), 6),
        "min": round(min(durations), 6),
        "median": round(float(statistics.median(durations)), 6),
        "max": round(max(durations), 6),
    }


def public_manifest_hash(rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ensure_no_references_or_paths(metadata: dict[str, Any]) -> None:
    serialized = json.dumps(metadata, ensure_ascii=False)
    forbidden_keys = {"text", "reference", "raw_transcription", "transcription", "audio_filepath"}
    def walk(value: Any, key: str = "") -> None:
        if key in forbidden_keys:
            raise ValueError(f"metadata contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(metadata)
    local_home = "/" + "home" + "/"
    local_mnt_data = "/" + "mnt" + "/" + "data"
    if local_home in serialized or local_mnt_data in serialized:
        raise ValueError("metadata contains local absolute path")


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_pairs = [(str(row["reference"]), str(row["hypothesis"])) for row in rows]
    normalized_pairs = [
        (normalize_sl_asr_text(str(row["reference"])), normalize_sl_asr_text(str(row["hypothesis"]))) for row in rows
    ]
    raw = corpus_metric_summary(raw_pairs)
    normalized = corpus_metric_summary(normalized_pairs)
    return {
        "normalizer": NORMALIZER_VERSION,
        "raw": raw.__dict__,
        "normalized": normalized.__dict__,
    }


def validate_gate_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition_role") != "immutable_real_gate":
                raise ValueError(f"{path}:{line_number}: not an immutable real gate")
            if row.get("source_type") != "public_real":
                raise ValueError(f"{path}:{line_number}: not public real source")
            if row.get("target_lang") != "sl-SI":
                raise ValueError(f"{path}:{line_number}: target_lang must be sl-SI")
            audio_path = Path(row["audio_filepath"])
            validate_wav(audio_path, sample_rate=16000)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def reject_real_gate_for_training(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("partition_role") == "immutable_real_gate" or row.get("source_type") == "public_real":
            raise ValueError("immutable real-gate rows must not enter training")


def reject_real_gate_for_generation(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row.get("partition_role") == "immutable_real_gate":
            raise ValueError("immutable real-gate rows must not enter generation prompts")
        if "reference" in row or "text" in row:
            raise ValueError("raw reference text must not enter generation prompts")
