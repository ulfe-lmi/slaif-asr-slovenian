#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import wave
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.prompt_experiment import atomic_write_text
from slaif_asr.tts import sha256_file, validate_wav


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
    temp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an ignored deterministic FLEURS Slovenian real gate.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/prompt_column_active_curriculum.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gates/fleurs-sl-si-64"))
    args = parser.parse_args()

    from datasets import load_dataset

    config = json.loads(args.config.read_text(encoding="utf-8"))["fixed_gates"]["real_fleurs"]
    dataset = load_dataset(config["dataset"], config["config"], split=config["split"], revision=config["revision"])
    candidates = []
    excluded = set(config.get("exclude_known_row_ids", []))
    for index, row in enumerate(dataset):
        if index in excluded:
            continue
        audio = row["audio"]
        duration = len(audio["array"]) / audio["sampling_rate"]
        if config["duration_seconds_min"] <= duration <= config["duration_seconds_max"]:
            candidates.append((index, row, duration))
    rng = random.Random(config["seed"])
    rng.shuffle(candidates)
    selected = sorted(candidates[: config["size"]], key=lambda item: item[0])
    manifest_rows = []
    public_rows = []
    for row_id, row, duration in selected:
        wav_path = args.output_dir / "audio" / f"fleurs-sl-si-{row_id:05d}.wav"
        write_wav(wav_path, row["audio"]["array"], row["audio"]["sampling_rate"])
        info = validate_wav(wav_path, sample_rate=16000)
        manifest_rows.append(
            {
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(info.duration_seconds, 6),
                "text": row["transcription"],
                "lang": "sl-SI",
                "target_lang": "sl-SI",
                "sample_id": f"fleurs-sl-si-{row_id:05d}",
                "partition_role": "immutable_real_gate",
                "source_type": "public_real",
                "license": config["license"],
            }
        )
        public_rows.append(
            {
                "row_id": row_id,
                "sample_id": f"fleurs-sl-si-{row_id:05d}",
                "audio_sha256": info.sha256,
                "duration": round(duration, 6),
            }
        )
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows)
    public_text = json.dumps(
        {
            "dataset": config["dataset"],
            "config": config["config"],
            "revision": config["revision"],
            "split": config["split"],
            "license": config["license"],
            "seed": config["seed"],
            "selected": public_rows,
            "manifest_sha256": "",
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    manifest_path = args.output_dir / "manifest.jsonl"
    atomic_write_text(manifest_path, manifest_text)
    payload = json.loads(public_text)
    payload["manifest_sha256"] = sha256_file(manifest_path)
    atomic_write_text(args.output_dir / "gate-metadata.json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
