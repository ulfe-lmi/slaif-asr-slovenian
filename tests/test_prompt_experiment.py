from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slaif_asr.prompt_experiment import validate_experiment_config, write_manifest
from slaif_asr.prompt_experiment import ManifestRecord


def minimal_config() -> dict:
    return {
        "target_lang": "sl-SI",
        "training": {"prompt_mode": "langID", "weight_decay": 0},
        "phase_a": {"candidate_id": "piper-smoke-0007"},
        "phase_b": {"train_candidate_ids": ["piper-smoke-0007"]},
        "holdout_candidate_ids": ["piper-smoke-0002"],
        "real_public_smoke": {"sample_id": "fleurs-sl-si-smoke"},
    }


class PromptExperimentTests(unittest.TestCase):
    def test_training_config_rejects_auto_prompt_mode(self) -> None:
        cfg = minimal_config()
        cfg["training"]["prompt_mode"] = "auto"
        with self.assertRaises(ValueError):
            validate_experiment_config(cfg)

    def test_training_config_rejects_holdout_in_training(self) -> None:
        cfg = minimal_config()
        cfg["phase_b"]["train_candidate_ids"].append("piper-smoke-0002")
        with self.assertRaises(ValueError):
            validate_experiment_config(cfg)

    def test_training_config_rejects_real_sample_in_training(self) -> None:
        cfg = minimal_config()
        cfg["phase_a"]["candidate_id"] = "fleurs-sl-si-smoke"
        cfg["phase_b"]["train_candidate_ids"] = ["fleurs-sl-si-smoke"]
        with self.assertRaises(ValueError):
            validate_experiment_config(cfg)

    def test_valid_config_passes(self) -> None:
        validate_experiment_config(minimal_config())

    def test_write_manifest_is_deterministic_and_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "a.wav"
            # Minimal valid mono 16 kHz 16-bit WAV.
            import wave

            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(b"\x01\x00" * 160)
            manifest = Path(tmp) / "manifest.jsonl"
            record = ManifestRecord(
                sample_id="sample-1",
                audio_filepath=wav_path,
                duration=0.01,
                text="Besedilo.",
                lang="sl-SI",
                target_lang="sl-SI",
                partition_role="synthetic_smoke",
                source_type="synthetic_tts",
            )
            first_hash = write_manifest(manifest, [record])
            first_text = manifest.read_text(encoding="utf-8")
            second_hash = write_manifest(manifest, [record])
            self.assertEqual(first_hash, second_hash)
            row = json.loads(first_text)
            self.assertEqual(row["sample_id"], "sample-1")
            self.assertTrue(Path(row["audio_filepath"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
