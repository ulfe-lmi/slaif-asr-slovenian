from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slaif_asr.config import REPO_ROOT
from slaif_asr.tts import sha256_file, validate_wav


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    audio_filepath: Path
    duration: float
    text: str
    lang: str
    target_lang: str
    partition_role: str
    source_type: str


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fp:
        fp.write(text)
        temp_name = fp.name
    os.replace(temp_name, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def validate_experiment_config(config: dict[str, Any]) -> None:
    if config.get("target_lang") != "sl-SI":
        raise ValueError("target_lang must be sl-SI")
    training = config.get("training", {})
    if training.get("prompt_mode") != "langID":
        raise ValueError("training.prompt_mode must be langID")
    if training.get("weight_decay") != 0:
        raise ValueError("training.weight_decay must be 0")
    phase_a = config.get("phase_a", {})
    phase_b = config.get("phase_b", {})
    one_sample = phase_a.get("candidate_id")
    train_ids = list(phase_b.get("train_candidate_ids", []))
    holdout_ids = list(config.get("holdout_candidate_ids", []))
    if one_sample not in train_ids:
        raise ValueError("phase_a candidate must be part of the synthetic training set")
    overlap = sorted(set(train_ids).intersection(holdout_ids))
    if overlap:
        raise ValueError(f"holdout IDs cannot enter training: {overlap}")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("duplicate training candidate ID")
    if len(set(holdout_ids)) != len(holdout_ids):
        raise ValueError("duplicate holdout candidate ID")
    real_smoke = config.get("real_public_smoke", {})
    if real_smoke.get("sample_id") in set(train_ids):
        raise ValueError("real public smoke sample cannot enter training")


def repository_path(path_text: str) -> Path:
    return (REPO_ROOT / path_text).resolve()


def load_rendered_records(path: Path) -> dict[str, ManifestRecord]:
    records: dict[str, ManifestRecord] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line_number, line in enumerate(fp, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_id = row["candidate_id"]
            if candidate_id in records:
                raise ValueError(f"{path}:{line_number}: duplicate candidate_id {candidate_id}")
            audio_path = Path(row["audio_filepath"]).expanduser().resolve()
            info = validate_wav(audio_path, sample_rate=16000)
            if row.get("audio_sha256") != info.sha256:
                raise ValueError(f"{candidate_id}: audio SHA256 does not match provenance")
            if row.get("tts", {}).get("engine_revision") != "b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6":
                raise ValueError(f"{candidate_id}: unexpected Piper revision")
            if row.get("tts", {}).get("voice_revision") != "217ddc79818708b078d0d14a8fae9608b9d77141":
                raise ValueError(f"{candidate_id}: unexpected voice revision")
            records[candidate_id] = ManifestRecord(
                sample_id=candidate_id,
                audio_filepath=audio_path,
                duration=float(row["duration_seconds"]),
                text=str(row["text"]),
                lang=str(row["language"]),
                target_lang=str(row["target_lang"]),
                partition_role=str(row["partition_role"]),
                source_type=str(row["source_type"]),
            )
    return records


def select_records(records: dict[str, ManifestRecord], sample_ids: list[str]) -> list[ManifestRecord]:
    selected: list[ManifestRecord] = []
    for sample_id in sample_ids:
        if sample_id not in records:
            raise KeyError(f"missing sample ID {sample_id}")
        selected.append(records[sample_id])
    return selected


def load_real_public_smoke(config: dict[str, Any]) -> ManifestRecord:
    item = config["real_public_smoke"]
    audio_path = repository_path(item["audio_filepath"])
    info = validate_wav(audio_path, sample_rate=16000)
    text = str(item["reference_text"])
    if item.get("license") != "CC BY 4.0":
        raise ValueError("real public smoke sample must record CC BY 4.0 license")
    return ManifestRecord(
        sample_id=str(item["sample_id"]),
        audio_filepath=audio_path,
        duration=round(info.duration_seconds, 6),
        text=text,
        lang="sl-SI",
        target_lang="sl-SI",
        partition_role="public_real_smoke",
        source_type="public_real",
    )


def write_manifest(path: Path, records: list[ManifestRecord]) -> str:
    rows = []
    seen: set[str] = set()
    for record in records:
        if record.sample_id in seen:
            raise ValueError(f"duplicate sample ID {record.sample_id}")
        seen.add(record.sample_id)
        validate_wav(record.audio_filepath, sample_rate=16000)
        rows.append(
            {
                "audio_filepath": str(record.audio_filepath.resolve()),
                "duration": round(record.duration, 6),
                "text": record.text,
                "lang": record.lang,
                "target_lang": record.target_lang,
                "sample_id": record.sample_id,
                "partition_role": record.partition_role,
                "source_type": record.source_type,
            }
        )
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)
    return sha256_file(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
