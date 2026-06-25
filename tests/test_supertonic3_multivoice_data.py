from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from slaif_asr.real_eval import atomic_write_json, atomic_write_jsonl
from slaif_asr.supertonic3_tts import (
    ALL_STYLES,
    HELD_OUT_STYLES,
    TRAINING_STYLES,
    validate_supertonic_audio,
)
from tests.test_supertonic3_tts import selected_row, selected_audio_row


def write_fixture_wav(path: Path, *, sample_rate: int, amplitude: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(200, int(sample_rate * 0.25))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((int(amplitude).to_bytes(2, "little", signed=True)) * frames)


class Supertonic3MultivoiceDataTests(unittest.TestCase):
    def test_full_manifest_validation_rejects_duplicate_final_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = self.fixture_config(tmp)
            self.write_assets(tmp / "model")
            rows = self.audio_rows(tmp, duplicate_first_two=True)
            atomic_write_jsonl(Path(config["local_outputs"]["audio_manifest"]), rows)
            atomic_write_jsonl(Path(config["local_outputs"]["exposure_schedule"]), self.schedule_rows(rows))
            result = validate_supertonic_audio(config, progress_interval_seconds=999.0)
            self.assertEqual(result["status"], "AUDIO_REJECTED")
            self.assertGreater(result["failures_by_reason"].get("duplicate_audio_sha256", 0), 0)

    def test_full_manifest_validation_accepts_expected_counts_and_voice_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = self.fixture_config(tmp)
            self.write_assets(tmp / "model")
            rows = self.audio_rows(tmp, duplicate_first_two=False)
            atomic_write_jsonl(Path(config["local_outputs"]["audio_manifest"]), rows)
            atomic_write_jsonl(Path(config["local_outputs"]["exposure_schedule"]), self.schedule_rows(rows))
            result = validate_supertonic_audio(config, progress_interval_seconds=999.0)
            self.assertEqual(result["status"], "AUDIO_ACCEPTED")
            self.assertEqual(result["training_final_files"], 1280)
            self.assertEqual(result["holdout_final_files"], 192)
            self.assertEqual(result["voice_counts"]["selected_training"]["M1"], 160)
            self.assertEqual(result["voice_counts"]["synthetic_holdout"]["M5"], 96)
            self.assertEqual(result["duplicate_paths"], 0)
            self.assertEqual(result["duplicate_hashes"], 0)

    def audio_rows(self, tmp: Path, *, duplicate_first_two: bool) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(160):
            selected_id = f"sl-corpus-v2-selected-training-v1-{index:03d}"
            for voice_index, voice in enumerate(TRAINING_STYLES):
                rows.append(self.audio_row(tmp, "selected_training", selected_id, voice, index * 20 + voice_index, duplicate=False))
        for index in range(96):
            holdout_id = f"gams9holdout-cell01-a00-o{index:03d}"
            for voice_index, voice in enumerate(HELD_OUT_STYLES):
                rows.append(self.audio_row(tmp, "synthetic_holdout", holdout_id, voice, 5000 + index * 20 + voice_index, duplicate=False))
        if duplicate_first_two:
            Path(rows[1]["audio_filepath"]).write_bytes(Path(rows[0]["audio_filepath"]).read_bytes())
        return rows

    def audio_row(self, tmp: Path, partition: str, source_key: str, voice: str, amplitude: int, *, duplicate: bool) -> dict[str, object]:
        native = tmp / "native" / partition / voice / f"{source_key}.{voice}.wav"
        final = tmp / "final" / partition / voice / f"{source_key}.{voice}.wav"
        write_fixture_wav(native, sample_rate=44100, amplitude=amplitude + 1000)
        write_fixture_wav(final, sample_rate=16000, amplitude=amplitude + 1100)
        return {
            "source_key": source_key,
            "partition_role": partition,
            "voice_style_id": voice,
            "voice_style_json_sha256": f"voice-{voice}",
            "source_text_sha256": f"text-{source_key}",
            "target_text_sha256": f"text-{source_key}",
            "source_audio_sha256": f"piper-{source_key}",
            "utterance_family_id": f"utt-{source_key}",
            "source_family_id": f"family-{source_key}",
            "source_id": f"source-{source_key}",
            "native_audio_filepath": str(native),
            "native_audio_sha256": "native-placeholder",
            "native_sample_rate": 44100,
            "native_channels": 1,
            "native_sample_width": 2,
            "native_frames": 2205,
            "native_duration_seconds": 0.05,
            "supertonic_duration_seconds": 0.05,
            "audio_filepath": str(final),
            "audio_sha256": "placeholder",
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "frames": 800,
            "duration_seconds": 0.05,
            "tts": {"asset_tree_sha256": "assets", "voice_style_id": voice},
        }

    def schedule_rows(self, audio_rows: list[dict[str, object]]) -> list[dict[str, object]]:
        by_id_voice = {(str(row["source_key"]), str(row["voice_style_id"])): row for row in audio_rows if row["partition_role"] == "selected_training"}
        selected_ids = sorted({source_key for source_key, _ in by_id_voice}, key=lambda value: __import__("hashlib").sha256(value.encode()).hexdigest())
        schedule = []
        for epoch in range(1, 13):
            for position, source_key in enumerate(selected_ids):
                group_index = position // 20
                voice = TRAINING_STYLES[(group_index + epoch - 1) % len(TRAINING_STYLES)]
                row = by_id_voice[(source_key, voice)]
                schedule.append(
                    {
                        "epoch": epoch,
                        "source_key": source_key,
                        "voice_style_id": voice,
                        "audio_filepath": row["audio_filepath"],
                        "audio_sha256": row["audio_sha256"],
                        "duration": row["duration_seconds"],
                        "target_text_sha256": row["target_text_sha256"],
                        "utterance_family_id": row["utterance_family_id"],
                        "source_family_id": row["source_family_id"],
                    }
                )
        return schedule

    def write_assets(self, root: Path) -> None:
        for rel in [
            "onnx/duration_predictor.onnx",
            "onnx/text_encoder.onnx",
            "onnx/vector_estimator.onnx",
            "onnx/vocoder.onnx",
            "onnx/tts.json",
            "onnx/unicode_indexer.json",
            *(f"voice_styles/{style}.json" for style in ALL_STYLES),
        ]:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rel, encoding="utf-8")

    def fixture_config(self, tmp: Path) -> dict[str, object]:
        selected = tmp / "selected.jsonl"
        selected_audio = tmp / "selected-audio.jsonl"
        holdout = tmp / "holdout.jsonl"
        atomic_write_jsonl(selected, [selected_row(index) for index in range(160)])
        atomic_write_jsonl(selected_audio, [selected_audio_row(index) for index in range(160)])
        atomic_write_jsonl(
            holdout,
            [
                {
                    "candidate_id": f"gams9holdout-cell01-a00-o{index:03d}",
                    "target_text": f"Holdout stavek {index}.",
                    "source_id": f"holdout-source-{index:03d}",
                    "source_family_id": f"holdout-family-{index:03d}",
                    "utterance_family_id": f"holdout-utt-{index:03d}",
                    "domain": "fixture",
                    "phenomena": [],
                }
                for index in range(96)
            ],
        )
        run = tmp / "run"
        return {
            "tts_id": "supertonic3-sl-multivoice-v1",
            "package": {"name": "supertonic", "version": "1.3.1", "license": "MIT", "environment": str(tmp / ".venv-supertonic")},
            "model": {
                "name": "supertonic-3",
                "repository": "Supertone/supertonic-3",
                "revision": "724fb5abbf5502583fb520898d45929e62f02c0b",
                "license": "BigScience OpenRAIL-M",
                "local_dir": str(tmp / "model"),
                "auto_download": False,
            },
            "language": {"code": "sl", "fallback_na_allowed": False},
            "voice_styles": {"available": list(ALL_STYLES), "training": list(TRAINING_STYLES), "held_out": list(HELD_OUT_STYLES)},
            "synthesis": {
                "total_steps": 8,
                "speed": 1.05,
                "max_chunk_length": 300,
                "silence_duration": 0.3,
                "native_sample_rate": 44100,
                "final_sample_rate": 16000,
                "expression_tags_allowed": False,
                "custom_voice_builder_allowed": False,
            },
            "inputs": {
                "selected_training_manifest": str(selected),
                "selected_training_audio_manifest": str(selected_audio),
                "synthetic_holdout_text": str(holdout),
                "selected_training_manifest_sha256": "unused",
                "selected_training_audio_manifest_sha256": "unused",
                "synthetic_holdout_text_sha256": "unused",
                "selected_rows": 160,
                "synthetic_holdout_rows": 96,
            },
            "local_outputs": {
                "run_root": str(run),
                "native_manifest": str(run / "native.local.jsonl"),
                "audio_manifest": str(run / "audio.local.jsonl"),
                "training_probe_manifest": str(run / "training-probe.local.jsonl"),
                "exposure_schedule": str(run / "exposure.local.jsonl"),
                "validation": str(run / "validation.local.json"),
                "summary": str(run / "summary.local.json"),
                "progress_dir": str(run / "progress"),
                "logs_dir": str(run / "logs"),
            },
        }


if __name__ == "__main__":
    unittest.main()
