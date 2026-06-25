from __future__ import annotations

import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("NumPy is required for speaker-range augmentation tests") from exc

try:
    import scipy  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise unittest.SkipTest("SciPy is required for speaker-range augmentation tests") from exc

from slaif_asr.corpus_v2_training import TrainingRecord, deterministic_epoch_batches
from slaif_asr.speaker_range_augmentation import (
    EXPECTED_BASE_METRICS,
    EXPECTED_CLEAN_METRICS,
    EXPECTED_PROFILE_FACTORS,
    SourceAudioRecord,
    SpeakerProfile,
    build_exposure_schedule,
    classify_speaker_range_augmented,
    load_augmentation_config,
    real_regression_burden,
    resample_variant,
    stable_ordered_ids,
    training_records_for_epoch,
    validate_exposure_schedule,
)


def source_record(index: int) -> SourceAudioRecord:
    return SourceAudioRecord(
        selected_training_id=f"sl-corpus-v2-selected-training-v1-{index:03d}",
        source_audio_filepath=f"/ignored/{index}.wav",
        source_audio_sha256=f"audio-{index}",
        source_text_sha256=f"text-{index}",
        utterance_family_id=f"utt-{index}",
        source_family_id=f"source-{index}",
        duration=1.0 + index / 100.0,
    )


def training_record(index: int, *, duration: float | None = None) -> TrainingRecord:
    return TrainingRecord(
        selected_training_id=f"sl-corpus-v2-selected-training-v1-{index:03d}",
        audio_filepath=f"/clean/{index}.wav",
        duration=duration if duration is not None else 1.0 + index / 100.0,
        text=f"Besedilo {index}",
        text_sha256=f"text-{index}",
        audio_sha256=f"audio-{index}",
        selection_reason="hard",
        selection_rank=index,
    )


def write_sine_wav(path: Path, *, amplitude: float, seconds: float = 0.5) -> None:
    sample_rate = 16000
    frames = int(sample_rate * seconds)
    t = np.arange(frames, dtype=np.float64) / sample_rate
    signal = np.sin(2.0 * math.pi * 440.0 * t) * amplitude
    pcm = np.rint(np.clip(signal, -0.999, 0.999) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


class SpeakerRangeAugmentationTests(unittest.TestCase):
    def test_exact_five_profile_policy(self) -> None:
        config = load_augmentation_config()
        factors = [
            (row["profile_id"], float(row["resampling_rate"]), int(row["up"]), int(row["down"]))
            for row in config["profiles"]
        ]
        self.assertEqual(factors, EXPECTED_PROFILE_FACTORS)
        self.assertEqual(config["transform"]["algorithm"], "scipy.signal.resample_poly")

    def test_stable_row_ordering_is_deterministic(self) -> None:
        rows = [source_record(index) for index in range(160)]
        first = stable_ordered_ids(rows)
        second = stable_ordered_ids(list(reversed(rows)))
        self.assertEqual(first, second)

    def test_exposure_schedule_counts(self) -> None:
        config = load_augmentation_config()
        profiles = [
            SpeakerProfile(row["profile_id"], float(row["resampling_rate"]), int(row["up"]), int(row["down"]), row["intended_proxy"])
            for row in config["profiles"]
        ]
        rows = [source_record(index) for index in range(160)]
        schedule = build_exposure_schedule(rows, profiles, epochs=12)
        summary = validate_exposure_schedule(schedule, rows, profiles, epochs=12)
        self.assertEqual(summary["scheduled_exposures"], 1920)
        self.assertEqual(set(summary["exposures_by_profile"].values()), {384})
        self.assertEqual(summary["clean_fraction"], 0.2)
        for epoch in range(1, 13):
            epoch_rows = [row for row in schedule if row["epoch"] == epoch]
            self.assertEqual(len({row["selected_training_id"] for row in epoch_rows}), 160)
            counts = {}
            for row in epoch_rows:
                counts[row["profile_id"]] = counts.get(row["profile_id"], 0) + 1
            self.assertEqual(set(counts.values()), {32})

    def test_training_records_for_epoch_swaps_audio_without_text_mutation(self) -> None:
        clean = [training_record(index) for index in range(3)]
        schedule = [
            {
                "epoch": 1,
                "selected_training_id": row.selected_training_id,
                "profile_id": "clean",
                "audio_filepath": f"/aug/{index}.wav",
                "audio_sha256": f"aug-audio-{index}",
                "duration": row.duration + 0.1,
            }
            for index, row in enumerate(clean)
        ]
        swapped = training_records_for_epoch(clean, schedule, epoch=1)
        self.assertEqual(swapped[clean[0].selected_training_id].text, clean[0].text)
        self.assertEqual(swapped[clean[0].selected_training_id].audio_filepath, "/aug/0.wav")
        self.assertEqual(swapped[clean[0].selected_training_id].audio_sha256, "aug-audio-0")

    def test_batch_membership_uses_original_duration(self) -> None:
        clean = [training_record(index, duration=10.0 - index) for index in range(8)]
        layout = deterministic_epoch_batches(clean, batch_size=4, epoch=1, seed=1234, bucketed=True)
        augmented = [
            {
                "epoch": 1,
                "selected_training_id": row.selected_training_id,
                "profile_id": "child_like_proxy",
                "audio_filepath": f"/aug/{index}.wav",
                "audio_sha256": f"aug-{index}",
                "duration": 100.0 + index,
            }
            for index, row in enumerate(clean)
        ]
        _ = training_records_for_epoch(clean, augmented, epoch=1)
        repeated = deterministic_epoch_batches(clean, batch_size=4, epoch=1, seed=1234, bucketed=True)
        self.assertEqual(layout.batches, repeated.batches)

    def test_resample_variant_duration_rms_and_peak_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            tmp = Path(tmp_text)
            source = tmp / "source.wav"
            output = tmp / "out.wav"
            write_sine_wav(source, amplitude=0.95, seconds=0.5)
            profile = SpeakerProfile("child_like_proxy", 0.8, 4, 5, "proxy")
            details = resample_variant(source, output, profile, peak_limit=0.98)
            self.assertTrue(output.exists())
            self.assertAlmostEqual(details["observed_duration_ratio"], 0.8, delta=0.005)
            self.assertLessEqual(details["output_peak"], 0.9801)
            self.assertGreater(details["output_audio_sha256"], "")

    def test_regression_burden_and_classification(self) -> None:
        self.assertAlmostEqual(real_regression_burden(EXPECTED_BASE_METRICS, EXPECTED_CLEAN_METRICS), 16.361)
        mitigated = json.loads(json.dumps(EXPECTED_CLEAN_METRICS))
        mitigated["synthetic_holdout"] = {"wer": 70.0, "cer": 25.0, "empty": 0}
        mitigated["fleurs_v2"] = {"wer": 57.0, "cer": 18.0, "empty": 0}
        mitigated["artur_j"] = {"wer": 69.0, "cer": 27.0, "empty": 1}
        decision = classify_speaker_range_augmented(mitigated)
        self.assertIn(
            decision["classification"],
            {
                "SPEAKER_RANGE_AUGMENTATION_PREVENTS_REAL_REGRESSION",
                "SPEAKER_RANGE_AUGMENTATION_MITIGATES_REAL_REGRESSION",
                "SPEAKER_RANGE_AUGMENTATION_NOT_SUPPORTED",
            },
        )
        self.assertEqual(decision["accepted_parent"], "none")


if __name__ == "__main__":
    unittest.main()
