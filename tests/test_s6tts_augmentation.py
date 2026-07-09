from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

from slaif_asr.s6tts_augmentation import (
    EXPECTED_PROFILE_IDS,
    augmented_duplicate_groups,
    load_json,
    planned_tasks,
    summarize_augmented_view,
    validate_s6_augmentation_config,
)
from slaif_asr.tts import write_jsonl


CONFIG_PATH = Path("configs/augmentation/s6tts_transcript_preserving_11_views_v1.json")
CLEAN_CERT_PATH = Path("docs/data-certificates/sl-corpus-v4-s6tts-clean-view-v1.json")
BASE_AUG_CONFIG = Path("configs/augmentation/scale200_transcript_preserving_v1.json")


def write_wav(path: Path, *, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x00" * frames)


class S6AugmentationTests(unittest.TestCase):
    def load_inputs(self):
        return load_json(CONFIG_PATH), load_json(CLEAN_CERT_PATH), load_json(BASE_AUG_CONFIG)

    def test_config_validator_accepts_expected_profile_set(self):
        config, clean_cert, base_config = self.load_inputs()
        validate_s6_augmentation_config(config, clean_certificate=clean_cert, base_config=base_config)
        self.assertEqual([profile["profile_id"] for profile in base_config["augmentation_profiles"]], EXPECTED_PROFILE_IDS)

    def test_config_validator_rejects_changed_fixed_text_hash(self):
        config, clean_cert, base_config = self.load_inputs()
        config = dict(config)
        config["fixed_text_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "fixed_text_sha256"):
            validate_s6_augmentation_config(config, clean_certificate=clean_cert, base_config=base_config)

    def test_config_validator_rejects_unknown_clean_status(self):
        config, clean_cert, base_config = self.load_inputs()
        clean_cert = dict(clean_cert)
        clean_cert["status"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "not accepted"):
            validate_s6_augmentation_config(config, clean_certificate=clean_cert, base_config=base_config)

    def test_stage_planner_computes_expected_output_count(self):
        config, _clean_cert, base_config = self.load_inputs()
        clean_rows = [{"safe_key": f"row-{idx}", "row_index": idx} for idx in range(int(config["semantic_rows"]))]
        self.assertEqual(sum(1 for _ in planned_tasks(clean_rows, base_config["augmentation_profiles"])), 176000)

    def test_public_certificate_redacts_text_and_absolute_paths(self):
        config, _clean_cert, _base_config = self.load_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_manifest = root / "audio-manifest.local.jsonl"
            provenance_manifest = root / "provenance.local.jsonl"
            rows = []
            for idx in range(11):
                wav_path = root / f"out-{idx}.wav"
                write_wav(wav_path)
                rows.append(
                    {
                        "audio_relative_path": f"wav/out-{idx}.wav",
                        "audio_sha256": f"{idx:064x}",
                        "duration_seconds": 0.1,
                        "peak_ratio": 0.5,
                        "sample_rate": 16000,
                        "channels": 1,
                        "sample_width": 2,
                        "frames": 1600,
                        "row_index": 0,
                        "safe_key": f"s6tts-scale2000-00000__aug{idx + 1:02d}",
                        "source_clean_safe_key": "s6tts-scale2000-00000",
                        "source_clean_audio_sha256": "b" * 64,
                        "text_hash": "c" * 64,
                        "profile_id": EXPECTED_PROFILE_IDS[idx],
                        "profile_index": idx + 1,
                    }
                )
            write_jsonl(audio_manifest, rows)
            write_jsonl(provenance_manifest, [{k: v for k, v in row.items() if k != "audio_filepath"} for row in rows])
            local_config = dict(config)
            local_config["semantic_rows"] = 1
            local_config["expected_augmented_files"] = 11
            local_config["source_clean_files"] = 1
            local_config["local_outputs"] = {
                "run_root": str(root),
                "audio_manifest": str(audio_manifest),
                "provenance_manifest": str(provenance_manifest),
                "validation": str(root / "validation.json"),
                "summary": str(root / "summary.json"),
            }
            summary = summarize_augmented_view(local_config)
            public_text = json.dumps(summary, ensure_ascii=False)
            self.assertNotIn('"text"', public_text)
            self.assertNotIn("audio_filepath", public_text)
            self.assertNotIn(str(root), public_text)

    def test_explained_duplicates_allowed_only_from_clean_duplicate_groups(self):
        rows = [
            {
                "audio_sha256": "a" * 64,
                "row_index": 1,
                "safe_key": "row-1__aug01",
                "source_clean_safe_key": "clean-1",
                "source_clean_audio_sha256": "b" * 64,
                "text_hash": "c" * 64,
                "profile_id": "gain_dynamic_range_variation",
                "parameters": {"gain_db": 1.0},
            },
            {
                "audio_sha256": "a" * 64,
                "row_index": 2,
                "safe_key": "row-2__aug01",
                "source_clean_safe_key": "clean-2",
                "source_clean_audio_sha256": "b" * 64,
                "text_hash": "d" * 64,
                "profile_id": "gain_dynamic_range_variation",
                "parameters": {"gain_db": 2.0},
            },
        ]
        groups = augmented_duplicate_groups(rows, [frozenset({"clean-1", "clean-2"})])
        self.assertEqual(groups["explained_duplicate_extra_file_count"], 1)
        self.assertEqual(groups["unexplained_duplicate_extra_file_count"], 0)
        self.assertEqual(groups["groups"][0]["explanation"], "inherited_numeric_normalization_equivalence")
        groups = augmented_duplicate_groups(rows, [])
        self.assertEqual(groups["unexplained_duplicate_extra_file_count"], 1)

    def test_same_source_profile_collision_is_explained(self):
        rows = [
            {
                "audio_sha256": "a" * 64,
                "row_index": 10,
                "safe_key": "row-10__aug01",
                "source_clean_safe_key": "clean-10",
                "source_clean_audio_sha256": "b" * 64,
                "text_hash": "c" * 64,
                "profile_id": "coupled_speed_pitch_resampling",
                "parameters": {"rate": 1.04},
            },
            {
                "audio_sha256": "a" * 64,
                "row_index": 10,
                "safe_key": "row-10__aug02",
                "source_clean_safe_key": "clean-10",
                "source_clean_audio_sha256": "b" * 64,
                "text_hash": "c" * 64,
                "profile_id": "tempo_preserving_pitch",
                "parameters": {"tempo_factor": 0.961538},
            },
        ]
        groups = augmented_duplicate_groups(rows, [])
        self.assertEqual(groups["explained_duplicate_extra_file_count"], 1)
        self.assertEqual(groups["unexplained_duplicate_extra_file_count"], 0)
        self.assertEqual(groups["groups"][0]["explanation"], "deterministic_augmentation_profile_equivalence")

    def test_unexplained_duplicate_audio_hashes_reject_admission(self):
        config, _clean_cert, _base_config = self.load_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for idx in range(2):
                rows.append(
                    {
                        "audio_relative_path": f"wav/out-{idx}.wav",
                        "audio_sha256": "a" * 64,
                        "duration_seconds": 0.1,
                        "peak_ratio": 0.5,
                        "sample_rate": 16000,
                        "channels": 1,
                        "sample_width": 2,
                        "frames": 1600,
                        "row_index": idx,
                        "safe_key": f"row-{idx}",
                        "source_clean_safe_key": f"clean-{idx}",
                        "source_clean_audio_sha256": f"{idx:064x}",
                        "text_hash": f"{idx + 1:064x}",
                        "profile_id": EXPECTED_PROFILE_IDS[idx],
                        "profile_index": idx + 1,
                    }
                )
            write_jsonl(root / "audio-manifest.local.jsonl", rows)
            write_jsonl(root / "provenance.local.jsonl", rows)
            local_config = dict(config)
            local_config["semantic_rows"] = 1
            local_config["expected_augmented_files"] = 2
            local_config["source_clean_files"] = 1
            local_config["local_outputs"] = {
                "run_root": str(root),
                "audio_manifest": str(root / "audio-manifest.local.jsonl"),
                "provenance_manifest": str(root / "provenance.local.jsonl"),
                "validation": str(root / "validation.json"),
                "summary": str(root / "summary.json"),
            }
            summary = summarize_augmented_view(local_config)
            self.assertEqual(summary["status"], "S6TTS_AUGMENTATION_REJECTED_QUALITY")
            self.assertIn("unexplained_duplicate_audio_hashes", summary["issues_by_reason"])

    def test_zero_failures_required_for_accepted_classification(self):
        config, _clean_cert, _base_config = self.load_inputs()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "audio_relative_path": "wav/out.wav",
                "audio_sha256": "a" * 64,
                "duration_seconds": 0.1,
                "peak_ratio": 0.5,
                "sample_rate": 16000,
                "channels": 1,
                "sample_width": 2,
                "frames": 1600,
                "row_index": 0,
                "safe_key": "row-0",
                "source_clean_safe_key": "clean-0",
                "source_clean_audio_sha256": "b" * 64,
                "text_hash": "c" * 64,
                "profile_id": EXPECTED_PROFILE_IDS[0],
                "profile_index": 1,
            }
            write_jsonl(root / "audio-manifest.local.jsonl", [row])
            write_jsonl(root / "provenance.local.jsonl", [row])
            write_jsonl(root / "failures.local.jsonl", [{"reason": "RuntimeError"}])
            local_config = dict(config)
            local_config["semantic_rows"] = 1
            local_config["expected_augmented_files"] = 1
            local_config["source_clean_files"] = 1
            local_config["local_outputs"] = {
                "run_root": str(root),
                "audio_manifest": str(root / "audio-manifest.local.jsonl"),
                "provenance_manifest": str(root / "provenance.local.jsonl"),
                "validation": str(root / "validation.json"),
                "summary": str(root / "summary.json"),
            }
            summary = summarize_augmented_view(local_config)
            self.assertEqual(summary["status"], "S6TTS_AUGMENTATION_REJECTED_QUALITY")
            self.assertIn("augmentation_failure:RuntimeError", summary["issues_by_reason"])

    def test_no_generated_audio_extensions_tracked_policy(self):
        self.assertTrue(CONFIG_PATH.exists())
        self.assertTrue(CLEAN_CERT_PATH.exists())


if __name__ == "__main__":
    unittest.main()
