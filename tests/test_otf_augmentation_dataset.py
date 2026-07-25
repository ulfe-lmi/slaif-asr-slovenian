from __future__ import annotations

import copy
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from slaif_asr.otf_augmentation_dataset import (
    CleanAudioRecord,
    EXPECTED_AUGMENTATION_KEY,
    load_json,
    prepare_virtual_sample,
    select_source_record,
    validate_otf_config,
    validate_public_payload,
    virtual_augmentation_spec,
    waveform_sha256,
)
from slaif_asr.scale200_corpus import load_augmentation_config


CONFIG_PATH = Path("configs/data_loading/otf-augmentation-scale2000-fill-rate-v1.json")
PROFILE_PATH = Path("configs/augmentation/scale200_transcript_preserving_v1.json")


def write_fixture_wav(path: Path, *, sample_rate: int = 16000) -> None:
    t = np.arange(sample_rate // 5, dtype=np.float64) / sample_rate
    samples = np.rint(np.sin(2.0 * np.pi * 220.0 * t) * 4000.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples.tobytes())


class OtfAugmentationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = list(load_augmentation_config(PROFILE_PATH)["augmentation_profiles"])

    def record(self, path: Path, *, semantic_key: str = "row-0001") -> CleanAudioRecord:
        return CleanAudioRecord(
            semantic_key=semantic_key,
            voice="piper-sl_SI-artur-medium",
            source_audio_sha256="a" * 64,
            audio_path=path,
            duration_seconds=0.2,
            transcript="Preskusni zapis.",
        )

    def test_config_accepts_expected_three_worker_policy(self) -> None:
        config = load_json(CONFIG_PATH)
        profiles = load_json(PROFILE_PATH)
        validate_otf_config(config, profile_config=profiles)

    def test_config_rejects_worker_count_drift(self) -> None:
        config = copy.deepcopy(load_json(CONFIG_PATH))
        config["pipeline"]["num_workers"] = 2
        with self.assertRaisesRegex(ValueError, "exactly three workers"):
            validate_otf_config(config, profile_config=load_json(PROFILE_PATH))

    def test_same_virtual_exposure_spec_is_restart_and_worker_independent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            write_fixture_wav(path)
            record = self.record(path)
            first = virtual_augmentation_spec(record, 17, self.profiles)
            restarted = virtual_augmentation_spec(record, 17, self.profiles)
            worker_zero = virtual_augmentation_spec(record, 17, list(self.profiles))
            worker_two = virtual_augmentation_spec(record, 17, list(reversed(list(reversed(self.profiles)))))
            self.assertEqual(first, restarted)
            self.assertEqual(worker_zero, worker_two)
            self.assertEqual(first.parameter_seed, restarted.parameter_seed)

    def test_queue_order_does_not_change_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            write_fixture_wav(path)
            record = self.record(path)
            forward = {
                index: virtual_augmentation_spec(record, index, self.profiles)
                for index in range(12)
            }
            reverse = {
                index: virtual_augmentation_spec(record, index, self.profiles)
                for index in reversed(range(12))
            }
            self.assertEqual(forward, reverse)

    def test_different_exposures_produce_valid_varied_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            write_fixture_wav(path)
            record = self.record(path)
            specs = [
                virtual_augmentation_spec(record, index, self.profiles)
                for index in range(64)
            ]
            self.assertTrue(all(0 <= spec.profile_index < 11 for spec in specs))
            self.assertGreater(len({spec.profile_id for spec in specs}), 5)
            self.assertGreater(len({spec.augmentation_identity_sha256 for spec in specs}), 60)

    def test_source_selection_is_exposure_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.wav"
            write_fixture_wav(path)
            records = [self.record(path, semantic_key=f"row-{index}") for index in range(8)]
            first = select_source_record(records, 123, augmentation_key=EXPECTED_AUGMENTATION_KEY)
            second = select_source_record(records, 123, augmentation_key=EXPECTED_AUGMENTATION_KEY)
            self.assertEqual(first.semantic_key, second.semantic_key)

    def test_transform_preserves_transcript_and_sample_rate_without_writing_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.wav"
            write_fixture_wav(path)
            before = sorted(root.iterdir())
            record = self.record(path)
            first = prepare_virtual_sample(record, 3, self.profiles)
            second = prepare_virtual_sample(record, 3, self.profiles)
            after = sorted(root.iterdir())
            self.assertEqual(first.transcript, record.transcript)
            self.assertEqual(first.sample_rate, 16000)
            self.assertEqual(first.waveform.dtype, np.float32)
            self.assertEqual(waveform_sha256(first.waveform), waveform_sha256(second.waveform))
            self.assertEqual(before, after)

    def test_public_payload_rejects_raw_text_and_local_absolute_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            validate_public_payload({"transcript": "private"})
        with self.assertRaisesRegex(ValueError, "absolute path"):
            validate_public_payload({"storage": "/mnt/private/source.wav"})
        validate_public_payload(
            {
                "corpus_id": "sl-corpus-v4-gams-16000-training-v1",
                "augmentation_key": EXPECTED_AUGMENTATION_KEY,
            }
        )

    def test_config_contains_no_local_absolute_path(self) -> None:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        serialized = json.dumps(payload, sort_keys=True)
        for prefix in ("/data/", "/home/", "/mnt/", "/synology/"):
            self.assertNotIn(prefix, serialized)


if __name__ == "__main__":
    unittest.main()
