from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

from slaif_asr.acoustic_quality import (
    assert_public_audio_payload_safe,
    audio_partition_overlap_counts,
    load_corpus_v2_tts_items,
    read_audio_stats,
    select_worker_count,
    validate_audio_manifest,
)
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, sha256_text


def write_wav(path: Path, *, frames: int = 1600, sample_rate: int = 16000, amplitude: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((int(amplitude).to_bytes(2, "little", signed=True)) * frames)


def text_record(candidate_id: str, text: str, *, partition_role: str = "synthetic_candidate") -> dict:
    return {
        "schema_version": "2.0",
        "candidate_id": candidate_id,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": partition_role,
        "source_type": "generated_text",
        "source_id": f"source-{candidate_id}",
        "source_family_id": f"family-{candidate_id}",
        "template_family_id": None,
        "utterance_family_id": candidate_id,
        "phenomena": ["ordinary"],
        "domain": "fixture",
        "license": "fixture",
        "generation": {"system": "project-generated"},
        "entities": [],
        "minimal_pair": None,
    }


class AcousticQualityTests(unittest.TestCase):
    def test_corpus_v2_bridge_accepts_schema_2_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "accepted.jsonl"
            atomic_write_jsonl(path, [text_record("row-001", "Danes je miren dan.")])
            items = load_corpus_v2_tts_items(path)
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].candidate_id, "row-001")

    def test_corpus_v2_bridge_accepts_holdout_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "accepted-holdout.jsonl"
            atomic_write_jsonl(path, [text_record("row-001", "Danes je miren dan.", partition_role="synthetic_holdout")])
            items = load_corpus_v2_tts_items(path, expected_partition_role="synthetic_holdout")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].partition_role, "synthetic_holdout")

    def test_corpus_v2_bridge_rejects_wrong_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "accepted-holdout.jsonl"
            atomic_write_jsonl(path, [text_record("row-001", "Danes je miren dan.", partition_role="synthetic_holdout")])
            with self.assertRaisesRegex(ValueError, "expected synthetic_candidate partition"):
                load_corpus_v2_tts_items(path)

    def test_corpus_v2_bridge_rejects_text_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "accepted.jsonl"
            row = text_record("row-001", "Danes je miren dan.")
            row["target_text"] = "Drugo besedilo."
            atomic_write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "spoken_text must equal target_text"):
                load_corpus_v2_tts_items(path)

    def test_worker_selection_chooses_smallest_within_five_percent(self) -> None:
        selected = select_worker_count(
            [
                {"worker_count": 1, "valid": True, "utterances_per_minute": 90.0},
                {"worker_count": 2, "valid": True, "utterances_per_minute": 100.0},
                {"worker_count": 4, "valid": True, "utterances_per_minute": 102.0},
                {"worker_count": 8, "valid": False, "utterances_per_minute": 140.0},
            ],
            threshold=0.95,
        )
        self.assertEqual(selected, 2)

    def test_audio_stats_detects_non_silence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "sample.wav"
            write_wav(path)
            stats = read_audio_stats(path)
            self.assertEqual(stats.sample_rate, 16000)
            self.assertGreater(stats.peak_ratio, 0)
            self.assertGreater(stats.active_frame_fraction, 0.9)

    def test_audio_manifest_validates_successful_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            accepted = tmp / "accepted.jsonl"
            manifest = tmp / "manifest.jsonl"
            report = tmp / "audio-report.json"
            config = tmp / "audio-config.json"
            run_root = tmp / "run"
            row = text_record("row-001", "Danes je miren dan.")
            audio = tmp / "audio" / "row-001.wav"
            write_wav(audio)
            atomic_write_jsonl(accepted, [row])
            atomic_write_json(config, self.audio_config())
            atomic_write_jsonl(manifest, [self.audio_manifest_row(row, audio)])

            with self.patch_validation_paths(run_root, accepted, manifest, report, config):
                payload, return_code = validate_audio_manifest(require_status="AUDIO_ACCEPTED")
            self.assertEqual(return_code, 0)
            self.assertEqual(payload["status"], "AUDIO_ACCEPTED")

    def test_duplicate_audio_hash_rejects_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            accepted = tmp / "accepted.jsonl"
            manifest = tmp / "manifest.jsonl"
            report = tmp / "audio-report.json"
            config = tmp / "audio-config.json"
            run_root = tmp / "run"
            rows = [text_record("row-001", "Danes je miren dan."), text_record("row-002", "Jutri pride nova pošiljka.")]
            audio_a = tmp / "audio" / "a.wav"
            audio_b = tmp / "audio" / "b.wav"
            write_wav(audio_a)
            write_wav(audio_b)
            atomic_write_jsonl(accepted, rows)
            atomic_write_json(config, self.audio_config())
            atomic_write_jsonl(manifest, [self.audio_manifest_row(rows[0], audio_a), self.audio_manifest_row(rows[1], audio_b)])

            with self.patch_validation_paths(run_root, accepted, manifest, report, config):
                payload, return_code = validate_audio_manifest(require_status="AUDIO_ACCEPTED")
            self.assertEqual(return_code, 1)
            self.assertEqual(payload["status"], "AUDIO_REJECTED")
            self.assertEqual(payload["failures_by_reason"]["duplicate_audio_sha256"], 1)

    def test_audio_partition_overlap_counts_paths_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            candidate = tmp / "candidate.jsonl"
            holdout = tmp / "holdout.jsonl"
            shared_path = str(tmp / "shared.wav")
            atomic_write_jsonl(
                candidate,
                [
                    {"audio_filepath": shared_path, "audio_sha256": "aaa"},
                    {"audio_filepath": str(tmp / "candidate-only.wav"), "audio_sha256": "bbb"},
                ],
            )
            atomic_write_jsonl(
                holdout,
                [
                    {"audio_filepath": shared_path, "audio_sha256": "ccc"},
                    {"audio_filepath": str(tmp / "holdout-only.wav"), "audio_sha256": "bbb"},
                ],
            )
            self.assertEqual(audio_partition_overlap_counts(candidate, holdout), {"audio_path_overlaps": 1, "audio_sha256_overlaps": 1})

    def test_public_audio_payload_rejects_raw_text_and_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            assert_public_audio_payload_safe({"text": "Danes je miren dan."})
        with self.assertRaisesRegex(ValueError, "local path"):
            assert_public_audio_payload_safe({"safe": "/home/example/file.wav"})

    def audio_manifest_row(self, source: dict, audio: Path) -> dict:
        return {
            "candidate_id": source["candidate_id"],
            "audio_filepath": str(audio),
            "duration_seconds": 0.1,
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "text": source["target_text"],
            "target_text_sha256": sha256_text(source["target_text"]),
            "language": "sl-SI",
            "target_lang": "sl-SI",
            "partition_role": "synthetic_candidate",
            "source_type": "synthetic_tts",
            "source_id": source["source_id"],
            "source_family_id": source["source_family_id"],
            "utterance_family_id": source["utterance_family_id"],
            "audio_sha256": "not-used-by-validator",
            "native_audio": {"sample_rate": 22050},
            "tts": {
                "engine": "OHF-Voice/piper1-gpl",
                "engine_revision": "b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6",
                "voice": "sl_SI-artur-medium",
                "voice_repository": "rhasspy/piper-voices",
                "voice_revision": "217ddc79818708b078d0d14a8fae9608b9d77141",
                "execution_provider": "CUDAExecutionProvider",
            },
        }

    def audio_config(self) -> dict:
        return {
            "schema_version": "1.0",
            "validator_algorithm_version": "synthetic-audio-validator-v1",
            "expected_format": {"channels": 1, "sample_rate": 16000, "sample_width_bytes": 2},
            "duration_bounds_seconds": {"minimum": 0.01, "maximum": 30.0},
            "waveform_thresholds": {
                "minimum_peak_ratio": 0.001,
                "minimum_rms_ratio": 0.0001,
                "minimum_active_frame_fraction": 0.1,
                "maximum_clipping_fraction": 0.01,
                "maximum_leading_silence_seconds": 3.0,
                "maximum_trailing_silence_seconds": 3.0,
            },
            "tts": {
                "required_execution_provider": "CUDAExecutionProvider",
                "engine": "OHF-Voice/piper1-gpl",
                "engine_revision": "b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6",
                "voice": "sl_SI-artur-medium",
                "voice_repository": "rhasspy/piper-voices",
                "voice_revision": "217ddc79818708b078d0d14a8fae9608b9d77141",
                "native_sample_rate": 22050,
                "limitations": [],
            },
            "limitations": [],
        }

    def patch_validation_paths(self, run_root: Path, accepted: Path, manifest: Path, report: Path, config: Path):
        class Paths:
            audio_manifest = manifest
            validation_report = report

        return mock.patch.multiple(
            "slaif_asr.acoustic_quality",
            load_generation_config=mock.Mock(return_value={"run_directory": str(run_root)}),
            synthetic_audio_config_path=mock.Mock(return_value=config),
            accepted_candidates_path=mock.Mock(return_value=accepted),
            audio_paths=mock.Mock(return_value=Paths()),
        )


if __name__ == "__main__":
    unittest.main()
