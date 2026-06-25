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
    assert_public_payload_safe,
    build_exposure_schedule,
    build_variant_plan,
    load_supertonic_config,
    validate_exposure_schedule,
    write_training_probe_manifest,
)


def write_wav(path: Path, *, sample_rate: int, frames: int = 800, amplitude: int = 1200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((int(amplitude).to_bytes(2, "little", signed=True)) * frames)


def selected_row(index: int) -> dict[str, object]:
    selected_id = f"sl-corpus-v2-selected-training-v1-{index:03d}"
    return {
        "selected_training_id": selected_id,
        "text": f"Varna testna poved {index}.",
        "text_sha256": f"text-{index:03d}",
        "audio_sha256": f"piper-audio-{index:03d}",
        "duration": 1.0 + index / 1000.0,
        "selection_reason": "hard" if index < 120 else "control",
        "selection_rank": index,
    }


def selected_audio_row(index: int) -> dict[str, object]:
    selected_id = f"sl-corpus-v2-selected-training-v1-{index:03d}"
    return {
        "selected_training_id": selected_id,
        "source_id": f"source-{index:03d}",
        "source_family_id": f"source-family-{index:03d}",
        "utterance_family_id": f"utterance-family-{index:03d}",
        "domain": "fixture",
        "phenomena": ["ordinary"],
    }


def holdout_row(index: int) -> dict[str, object]:
    holdout_id = f"gams9holdout-cell01-a00-o{index:03d}"
    return {
        "candidate_id": holdout_id,
        "target_text": f"Neodvisna testna poved {index}.",
        "source_id": f"holdout-source-{index:03d}",
        "source_family_id": f"holdout-source-family-{index:03d}",
        "utterance_family_id": f"holdout-utterance-family-{index:03d}",
        "domain": "fixture",
        "phenomena": ["ordinary"],
    }


class Supertonic3TtsTests(unittest.TestCase):
    def test_committed_config_is_pinned_and_explicit_slovenian(self) -> None:
        config = load_supertonic_config()
        self.assertEqual(config["package"], {"name": "supertonic", "version": "1.3.1", "license": "MIT", "environment": ".venv-supertonic"})
        self.assertEqual(config["model"]["repository"], "Supertone/supertonic-3")
        self.assertEqual(config["model"]["revision"], "724fb5abbf5502583fb520898d45929e62f02c0b")
        self.assertIs(config["model"]["auto_download"], False)
        self.assertEqual(config["language"]["code"], "sl")
        self.assertIs(config["language"]["fallback_na_allowed"], False)
        self.assertEqual(config["runtime"]["execution_device"], "cuda")
        self.assertEqual(config["runtime"]["cuda_visible_devices"], "1")
        self.assertEqual(config["runtime"]["required_provider"], "CUDAExecutionProvider")
        self.assertIs(config["runtime"]["cpu_provider_fallback_allowed"], False)
        self.assertEqual(tuple(config["voice_styles"]["available"]), ALL_STYLES)
        self.assertEqual(tuple(config["voice_styles"]["training"]), TRAINING_STYLES)
        self.assertEqual(tuple(config["voice_styles"]["held_out"]), HELD_OUT_STYLES)

    def test_variant_plan_uses_eight_training_and_two_heldout_styles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            config = self.fixture_config(Path(tmp_text), selected_count=2, holdout_count=3)
            training = build_variant_plan(config, "training")
            holdout = build_variant_plan(config, "holdout")
            self.assertEqual(len(training), 16)
            self.assertEqual(len(holdout), 6)
            self.assertEqual({voice for _, voice in training}, set(TRAINING_STYLES))
            self.assertEqual({voice for _, voice in holdout}, set(HELD_OUT_STYLES))
            self.assertFalse({voice for _, voice in training} & set(HELD_OUT_STYLES))

    def test_exposure_schedule_is_balanced_and_excludes_heldout_styles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = self.fixture_config(tmp, selected_count=160, holdout_count=0)
            rows = []
            for index in range(160):
                selected_id = f"sl-corpus-v2-selected-training-v1-{index:03d}"
                for voice in TRAINING_STYLES:
                    rows.append(
                        {
                            "source_key": selected_id,
                            "partition_role": "selected_training",
                            "voice_style_id": voice,
                            "audio_filepath": str(tmp / f"{selected_id}.{voice}.wav"),
                            "audio_sha256": f"audio-{index:03d}-{voice}",
                            "duration_seconds": 1.0,
                            "target_text_sha256": f"text-{index:03d}",
                            "utterance_family_id": f"utt-{index:03d}",
                            "source_family_id": f"family-{index:03d}",
                        }
                    )
            summary = build_exposure_schedule(config, rows)
            self.assertEqual(summary["status"], "PASSED")
            self.assertEqual(summary["scheduled_exposures"], 1920)
            for style in TRAINING_STYLES:
                self.assertEqual(summary["exposures_by_training_style"][style], 240)
            self.assertNotIn("M5", summary["exposures_by_training_style"])
            self.assertNotIn("F5", summary["exposures_by_training_style"])
            schedule = [
                {
                    "epoch": row["epoch"],
                    "source_key": row["source_key"],
                    "voice_style_id": row["voice_style_id"],
                }
                for row in self.read_jsonl(Path(config["local_outputs"]["exposure_schedule"]))
            ]
            self.assertEqual(validate_exposure_schedule(config, schedule)["status"], "PASSED")

    def test_training_probe_manifest_has_one_row_per_text_and_balanced_voices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            config = self.fixture_config(tmp, selected_count=160, holdout_count=0)
            rows = []
            for index in range(160):
                selected_id = f"sl-corpus-v2-selected-training-v1-{index:03d}"
                for voice in TRAINING_STYLES:
                    rows.append(
                        {
                            "source_key": selected_id,
                            "partition_role": "selected_training",
                            "voice_style_id": voice,
                            "audio_filepath": str(tmp / f"{selected_id}.{voice}.wav"),
                            "audio_sha256": f"audio-{index:03d}-{voice}",
                            "duration_seconds": 1.0,
                        }
                    )
            write_training_probe_manifest(config, rows)
            probe = self.read_jsonl(Path(config["local_outputs"]["training_probe_manifest"]))
            self.assertEqual(len(probe), 160)
            counts = {voice: sum(1 for row in probe if row["voice_style_id"] == voice) for voice in TRAINING_STYLES}
            self.assertEqual(set(counts.values()), {20})

    def test_public_payload_rejects_raw_text_ids_and_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            assert_public_payload_safe({"text": "To se ne sme objaviti."})
        with self.assertRaisesRegex(ValueError, "row IDs"):
            assert_public_payload_safe({"message": "gams9holdout-cell01-a00-o001"})
        with self.assertRaisesRegex(ValueError, "row IDs|local paths"):
            assert_public_payload_safe({"message": "/home/example/audio.wav"})

    def fixture_config(self, tmp: Path, *, selected_count: int, holdout_count: int) -> dict[str, object]:
        selected = tmp / "selected.jsonl"
        selected_audio = tmp / "selected-audio.jsonl"
        holdout = tmp / "holdout.jsonl"
        atomic_write_jsonl(selected, [selected_row(index) for index in range(selected_count)])
        atomic_write_jsonl(selected_audio, [selected_audio_row(index) for index in range(selected_count)])
        atomic_write_jsonl(holdout, [holdout_row(index) for index in range(holdout_count)])
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
                "selected_rows": selected_count,
                "synthetic_holdout_rows": holdout_count,
            },
            "local_outputs": {
                "run_root": str(tmp / "run"),
                "native_manifest": str(tmp / "run/native.local.jsonl"),
                "audio_manifest": str(tmp / "run/audio.local.jsonl"),
                "training_probe_manifest": str(tmp / "run/training-probe.local.jsonl"),
                "exposure_schedule": str(tmp / "run/exposure.local.jsonl"),
                "validation": str(tmp / "run/validation.local.json"),
                "summary": str(tmp / "run/summary.local.json"),
                "progress_dir": str(tmp / "run/progress"),
                "logs_dir": str(tmp / "run/logs"),
            },
        }

    def read_jsonl(self, path: Path) -> list[dict[str, object]]:
        with path.open("r", encoding="utf-8") as fp:
            return [__import__("json").loads(line) for line in fp if line.strip()]


if __name__ == "__main__":
    unittest.main()
