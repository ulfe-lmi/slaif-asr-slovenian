#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import urllib.request
import wave
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.real_eval import (
    NORMALIZER_VERSION,
    atomic_write_json,
    atomic_write_jsonl,
    duration_stats,
    ensure_no_references_or_paths,
    plan_fleurs_occurrences,
    md5_file,
    parse_artur_trs,
    public_manifest_hash,
    safe_extract_tar,
    select_artur_segments,
    sha256_file,
    stable_text_hash,
)
from slaif_asr.tts import validate_wav


def download_atomic(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    part = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as fp:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fp.write(chunk)
    os.replace(part, path)


def write_wav(path: Path, array: Any, sample_rate: int) -> None:
    import numpy as np
    import soxr

    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = soxr.resample(audio, sample_rate, 16000)
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    with wave.open(str(temp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm.tobytes())
    os.replace(temp, path)


def fleurs_audio_array(row: dict[str, Any]) -> tuple[Any, int]:
    import soundfile as sf
    from io import BytesIO

    audio = row["audio"]
    if audio.get("bytes") is not None:
        array, sample_rate = sf.read(BytesIO(audio["bytes"]), dtype="float32")
        return array, int(sample_rate)
    if audio.get("path"):
        array, sample_rate = sf.read(audio["path"], dtype="float32")
        return array, int(sample_rate)
    raise ValueError("FLEURS audio row has neither bytes nor path")


def build_fleurs(config: dict[str, Any], output_root: Path, metadata_root: Path) -> dict[str, Any]:
    from datasets import Audio, load_dataset

    cfg = config["fleurs_sl_si_test_full_v2"]
    dataset = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"], revision=cfg["revision"])
    dataset = dataset.cast_column("audio", Audio(decode=False))
    rows = list(dataset)
    plans = plan_fleurs_occurrences(rows)
    manifest_rows: list[dict[str, Any]] = []
    local_reference_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    gender_counts: Counter[str] = Counter()
    durations: list[float] = []
    for row, plan in zip(rows, plans, strict=True):
        array, sample_rate = fleurs_audio_array(row)
        wav_path = output_root / cfg["gate_id"] / plan.relative_audio_path
        write_wav(wav_path, array, sample_rate)
        info = validate_wav(wav_path, sample_rate=16000)
        text = row["transcription"]
        raw_text = row.get("raw_transcription")
        duration = round(info.duration_seconds, 6)
        durations.append(duration)
        gender = str(row.get("gender", "unknown"))
        gender_counts[gender] += 1
        manifest_rows.append(
            {
                "audio_filepath": str(wav_path.resolve()),
                "duration": duration,
                "text": text,
                "lang": "sl-SI",
                "target_lang": "sl-SI",
                "sample_id": plan.sample_id,
                "partition_role": "immutable_real_gate",
                "source_type": "public_real",
                "dataset": cfg["gate_id"],
                "source_row_index": plan.source_row_index,
                "source_id": plan.source_id,
            }
        )
        local_reference_rows.append(
            {
                "sample_id": plan.sample_id,
                "source_row_index": plan.source_row_index,
                "source_id": plan.source_id,
                "reference": text,
                "raw_reference": raw_text,
                "normalized_reference_sha256": stable_text_hash(text),
            }
        )
        public_rows.append(
            {
                "source_row_index": plan.source_row_index,
                "source_id": plan.source_id,
                "sample_id": plan.sample_id,
                "audio_sha256": info.sha256,
                "reference_sha256": stable_text_hash(text),
                "duration_seconds": duration,
                "gender": gender,
            }
        )
    gate_root = output_root / cfg["gate_id"]
    manifest_path = gate_root / "manifest.jsonl"
    references_path = gate_root / "references.local.jsonl"
    atomic_write_jsonl(manifest_path, manifest_rows)
    atomic_write_jsonl(references_path, local_reference_rows)
    metadata = {
        "gate_id": cfg["gate_id"],
        "dataset": cfg["dataset"],
        "config": cfg["config"],
        "split": cfg["split"],
        "revision": cfg["revision"],
        "license": cfg["license"],
        "construction_algorithm": cfg["construction_algorithm"],
        "normalizer": NORMALIZER_VERSION,
        "complete_test_split_used": True,
        "rows": len(public_rows),
        "duration_seconds": duration_stats(durations),
        "gender_counts": dict(sorted(gender_counts.items())),
        "selected": public_rows,
        "manifest_sha256": sha256_file(manifest_path),
        "reference_manifest_sha256": sha256_file(references_path),
        "public_metadata_hash": public_manifest_hash(public_rows),
    }
    ensure_no_references_or_paths(metadata)
    metadata_path = metadata_root / f"{cfg['gate_id']}.metadata.json"
    atomic_write_json(metadata_path, metadata)
    return {"gate": cfg["gate_id"], "manifest": str(manifest_path), "metadata": str(metadata_path), "rows": len(public_rows)}


def extract_audio_member(archive: Path, recording_id: str, destination: Path) -> Path | None:
    candidates = []
    audio_prefixes = [
        recording_id,
        recording_id.removesuffix("-std"),
        recording_id.removesuffix("-std") + "-avd",
    ]
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and any(name.startswith(prefix) for prefix in audio_prefixes) and name.lower().endswith((".wav", ".flac")):
                candidates.append(member)
        if not candidates:
            return None
        member = sorted(candidates, key=lambda item: item.name)[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as src, destination.open("wb") as dst:
            if src is None:
                return None
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    return destination


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


def build_artur(config: dict[str, Any], output_root: Path, metadata_root: Path, *, include_optional_audio: bool) -> dict[str, Any]:
    cfg = config["artur_j_public_gate_v1"]
    gate_root = output_root / cfg["gate_id"]
    archive_dir = gate_root / "archives"
    extract_dir = gate_root / "extracted"
    transcript_archive = archive_dir / cfg["transcript_archive"]["filename"]
    download_atomic(cfg["transcript_archive"]["url"], transcript_archive)
    if md5_file(transcript_archive) != cfg["transcript_archive"]["md5"]:
        raise RuntimeError(f"MD5 mismatch for {transcript_archive}")
    transcript_extract = extract_dir / "trs"
    if not transcript_extract.exists():
        safe_extract_tar(transcript_archive, transcript_extract)
    segments = []
    for path in sorted(transcript_extract.rglob("*.trs")):
        lowered = path.as_posix().lower()
        if "artur-j-splosni" not in lowered or "std" not in lowered or "pog" in lowered:
            continue
        segments.extend(parse_artur_trs(path, required_mode=cfg["transcript_mode"]))
    audio_archives = [item for item in cfg["audio_archives"] if include_optional_audio or not item.get("optional")]
    archive_paths = []
    for item in audio_archives:
        path = archive_dir / item["filename"]
        download_atomic(item["url"], path)
        if md5_file(path) != item["md5"]:
            raise RuntimeError(f"MD5 mismatch for {path}")
        archive_paths.append(path)
    available_recordings: set[str] = set()
    for path in archive_paths:
        available_recordings.update(archive_recording_ids(path))
    segments = [segment for segment in segments if segment.recording_id.removesuffix("-std") in available_recordings]
    selected = select_artur_segments(
        segments,
        required_count=cfg["required_segments"],
        duration_min=cfg["duration_seconds_min"],
        duration_max=cfg["duration_seconds_max"],
        max_per_recording=cfg["max_segments_per_recording"],
    )

    manifest_rows: list[dict[str, Any]] = []
    local_reference_rows: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    source_cache: dict[str, Path] = {}
    missing_recordings: list[str] = []
    for segment in selected:
        source_audio = source_cache.get(segment.recording_id)
        if source_audio is None:
            source_audio = gate_root / "source-audio" / f"{segment.recording_id}.wav"
            for archive in archive_paths:
                found = extract_audio_member(archive, segment.recording_id, source_audio)
                if found is not None:
                    source_audio = found
                    break
            if not source_audio.exists():
                missing_recordings.append(segment.recording_id)
                continue
            source_cache[segment.recording_id] = source_audio
        wav_path = gate_root / "audio" / f"{segment.sample_id}.wav"
        cut_audio(source_audio, wav_path, segment.start, segment.duration)
        info = validate_wav(wav_path, sample_rate=16000)
        manifest_rows.append(
            {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(info.duration_seconds, 6),
                "text": segment.text,
                "lang": "sl-SI",
                "target_lang": "sl-SI",
                "sample_id": segment.sample_id,
                "partition_role": "immutable_real_gate",
                "source_type": "public_real",
                "dataset": cfg["gate_id"],
            }
        )
        local_reference_rows.append({"sample_id": segment.sample_id, "reference": segment.text})
        public_rows.append(
            {
                "sample_id": segment.sample_id,
                "recording_id": segment.recording_id,
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "duration_seconds": round(info.duration_seconds, 6),
                "audio_sha256": info.sha256,
                "reference_sha256": stable_text_hash(segment.text),
                "transcript_sha256": sha256_file(Path(segment.transcript_path)),
            }
        )
    if len(manifest_rows) < cfg["required_segments"]:
        raise RuntimeError(
            f"ARTUR gate shortfall: built {len(manifest_rows)} of {cfg['required_segments']} segments; "
            f"missing recordings: {sorted(set(missing_recordings))[:10]}"
        )
    manifest_path = gate_root / "manifest.jsonl"
    references_path = gate_root / "references.local.jsonl"
    atomic_write_jsonl(manifest_path, manifest_rows)
    atomic_write_jsonl(references_path, local_reference_rows)
    durations = [float(row["duration"]) for row in manifest_rows]
    metadata = {
        "gate_id": cfg["gate_id"],
        "handles": {"transcriptions": cfg["transcript_handle"], "audio": cfg["audio_handle"]},
        "license": cfg["license"],
        "construction_algorithm": cfg["construction_algorithm"],
        "normalizer": NORMALIZER_VERSION,
        "transcript_archive": {
            "filename": cfg["transcript_archive"]["filename"],
            "md5": cfg["transcript_archive"]["md5"],
            "actual_md5": md5_file(transcript_archive),
        },
        "audio_archives": [
            {"filename": item["filename"], "md5": item["md5"], "actual_md5": md5_file(archive_dir / item["filename"])}
            for item in audio_archives
        ],
        "transcript_mode": cfg["transcript_mode"],
        "domain": cfg["preferred_domain"],
        "excluded_domains": cfg["excluded_domains"],
        "segments": len(manifest_rows),
        "distinct_source_recordings": len({row["recording_id"] for row in public_rows}),
        "max_segments_per_recording": cfg["max_segments_per_recording"],
        "duration_seconds": duration_stats(durations),
        "selected": public_rows,
        "manifest_sha256": sha256_file(manifest_path),
        "reference_manifest_sha256": sha256_file(references_path),
        "public_metadata_hash": public_manifest_hash(public_rows),
    }
    ensure_no_references_or_paths(metadata)
    metadata_path = metadata_root / f"{cfg['gate_id']}.metadata.json"
    atomic_write_json(metadata_path, metadata)
    return {"gate": cfg["gate_id"], "manifest": str(manifest_path), "metadata": str(metadata_path), "rows": len(public_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable real Slovenian evaluation gates.")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/real_gates.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/evaluation-gates"))
    parser.add_argument("--metadata-root", type=Path, default=Path("docs/evaluation-gates"))
    parser.add_argument("--gate", choices=["fleurs", "artur", "all"], default="all")
    parser.add_argument("--include-optional-artur-audio", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = []
    if args.gate in {"fleurs", "all"}:
        results.append(build_fleurs(config, args.output_root, args.metadata_root))
    if args.gate in {"artur", "all"}:
        results.append(build_artur(config, args.output_root, args.metadata_root, include_optional_audio=args.include_optional_artur_audio))
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
