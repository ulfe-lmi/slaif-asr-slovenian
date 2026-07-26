from __future__ import annotations

import math
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from slaif_asr.config import REPO_ROOT
from slaif_asr.otf_augmentation_dataset import (
    CleanAudioRecord,
    prepare_virtual_sample,
    run_determinism_checks_for_exposures,
    virtual_augmentation_spec,
)
from slaif_asr.parametric_voice_augmentation import (
    ALGORITHM_VERSION,
    AUGMENTATION_KEY,
    DEPENDENCIES,
    apply_policy_transform,
    load_policy,
    phase_for_exposure,
    validate_audio,
    validate_chain,
    verify_dependencies,
)
from slaif_asr.scale200_corpus import load_augmentation_config


POLICY_PATH = REPO_ROOT / "configs/augmentation/parametric_voiceaug_v1.json"
PROFILE_PATH = REPO_ROOT / "configs/augmentation/scale200_transcript_preserving_v1.json"


def write_wav(path: Path) -> None:
    time = np.arange(6400, dtype=np.float64) / 16_000.0
    values = (
        0.25 * np.sin(2.0 * math.pi * 180.0 * time)
        + 0.08 * np.sin(2.0 * math.pi * 620.0 * time)
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(np.round(values * 32767.0).astype("<i2").tobytes())


def fixture_record(path: Path, index: int = 1) -> CleanAudioRecord:
    return CleanAudioRecord(
        semantic_key=f"semantic-{index}",
        voice=f"voice-{index % 3}",
        source_audio_sha256=f"{index:064x}",
        audio_path=path,
        duration_seconds=0.4,
        transcript=f"synthetic fixture {index}",
    )


class ParametricVoiceAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_policy(POLICY_PATH)
        cls.profiles = list(
            load_augmentation_config(PROFILE_PATH)["augmentation_profiles"]
        )

    def spec(self, exposure_id: int, *, index: int = 1):
        return virtual_augmentation_spec(
            fixture_record(Path("unused.wav"), index),
            exposure_id,
            self.profiles,
            corpus_id="sl-corpus-v5-scale8000-training-v1",
            augmentation_key=AUGMENTATION_KEY,
            algorithm_version=ALGORITHM_VERSION,
            augmentation_policy=self.policy,
        )

    def test_policy_schema_key_and_dependencies_validate(self) -> None:
        self.assertEqual(self.policy["augmentation_key"], AUGMENTATION_KEY)
        self.assertEqual(self.policy["augmentation_family"], "parametric_voiceaug_v1")
        self.assertFalse(self.policy["disk_output"])
        self.assertTrue(self.policy["transcript_preserved"])
        self.assertEqual(self.policy["dependencies"], list(DEPENDENCIES))
        resolved = verify_dependencies()
        self.assertEqual([row["status"] for row in resolved], ["PASSED"] * 5)
        self.assertTrue(all(not row["vendored"] for row in resolved))
        self.assertTrue(all(row["license"] for row in resolved))

    def test_curriculum_phase_boundaries_are_fixed(self) -> None:
        self.assertEqual(phase_for_exposure(0), (1, "coverage", 0.50))
        self.assertEqual(phase_for_exposure(63_999), (4, "coverage", 0.50))
        self.assertEqual(phase_for_exposure(64_000), (5, "robustness", 0.90))
        self.assertEqual(phase_for_exposure(143_999), (9, "robustness", 0.90))
        with self.assertRaises(ValueError):
            phase_for_exposure(144_000)

    def test_specs_replay_and_parameters_remain_bounded(self) -> None:
        families: set[str] = set()
        for exposure_id in range(400):
            first = self.spec(exposure_id, index=exposure_id + 1)
            second = self.spec(exposure_id, index=exposure_id + 1)
            self.assertEqual(first, second)
            if first.parameters["mode"] != "parametric":
                continue
            operations = first.parameters["operations"]
            self.assertGreaterEqual(len(operations), 2)
            self.assertLessEqual(len(operations), 3)
            validate_chain(operations)
            for operation in operations:
                family = operation["family"]
                values = operation["parameters"]
                families.add(family)
                if family == "pitch_shift":
                    self.assertLessEqual(abs(values["semitones"]), 3.0)
                    self.assertGreaterEqual(abs(values["semitones"]), 1.0)
                elif family == "speed":
                    self.assertTrue(0.90 <= values["rate"] <= 1.10)
                elif family == "formant_warp":
                    self.assertTrue(0.84 <= values["factor"] <= 1.16)
                    self.assertFalse(0.98 < values["factor"] < 1.02)
                elif family == "aperiodicity":
                    self.assertTrue(20.0 <= values["breathiness_snr_db"] <= 32.0)
                    self.assertTrue(2500.0 <= values["highpass_hz"] <= 5000.0)
                elif family == "reverb":
                    self.assertTrue(0.15 <= values["rt60_seconds"] <= 0.80)
                elif family == "additive_noise":
                    self.assertTrue(8.0 <= values["snr_db"] <= 25.0)
        self.assertEqual(
            families,
            {
                "pitch_shift",
                "speed",
                "formant_warp",
                "aperiodicity",
                "channel_filter",
                "codec_companding",
                "reverb",
                "additive_noise",
                "gain_compression",
            },
        )

    def test_destructive_chains_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_chain(
                [
                    {"family": "additive_noise", "parameters": {"snr_db": 8.0}},
                    {"family": "reverb", "parameters": {}},
                    {"family": "codec_companding", "parameters": {}},
                ]
            )
        with self.assertRaises(ValueError):
            validate_chain(
                [
                    {"family": "additive_noise", "parameters": {"snr_db": 8.0}},
                    {
                        "family": "channel_filter",
                        "parameters": {"filter": "telephone_like"},
                    },
                ]
            )

    def test_audio_validation_rejects_invalid_waveforms(self) -> None:
        source = np.full(1600, 0.1, dtype=np.float64)
        validate_audio(source, source.copy())
        invalid = (
            np.zeros(1600),
            np.full(1600, np.nan),
            np.full(1600, np.inf),
            np.full(1600, 0.98),
            np.full(1000, 0.1),
        )
        for values in invalid:
            with self.assertRaises(ValueError):
                validate_audio(source, values)

    def test_all_operation_families_produce_safe_audio(self) -> None:
        source = (
            np.sin(np.linspace(0.0, 100.0, 6400, endpoint=False)) * 0.25
        ).astype(np.float64)
        seen: set[str] = set()
        for exposure_id in range(64_000, 66_000):
            spec = self.spec(exposure_id, index=exposure_id + 1)
            if spec.parameters["mode"] != "parametric":
                continue
            operation_families = {
                item["family"] for item in spec.parameters["operations"]
            }
            if operation_families.issubset(seen):
                continue
            transformed, details = apply_policy_transform(
                source,
                parameters=spec.parameters,
                seed_text=spec.parameter_seed,
            )
            self.assertTrue(np.isfinite(transformed).all())
            self.assertTrue(0.88 <= len(transformed) / len(source) <= 1.12)
            self.assertTrue(details["validation"]["non_silent"])
            seen.update(operation_families)
            if len(seen) == 9:
                break
        self.assertEqual(len(seen), 9)

    def test_worker_count_order_restart_and_waveform_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.wav"
            write_wav(path)
            candidate = None
            for exposure_id in range(64_000, 65_000):
                spec = virtual_augmentation_spec(
                    fixture_record(path),
                    exposure_id,
                    self.profiles,
                    corpus_id="sl-corpus-v5-scale8000-training-v1",
                    augmentation_key=AUGMENTATION_KEY,
                    algorithm_version=ALGORITHM_VERSION,
                    augmentation_policy=self.policy,
                )
                families = {
                    item["family"]
                    for item in spec.parameters.get("operations", [])
                }
                if (
                    spec.parameters["mode"] == "parametric"
                    and "pitch_shift" not in families
                ):
                    candidate = exposure_id
                    break
            self.assertIsNotNone(candidate)
            records = [fixture_record(path, index) for index in range(1, 4)]
            exposures = [
                (record, int(candidate) + index)
                for index, record in enumerate(records)
            ]
            before = set(root.iterdir())
            checks = run_determinism_checks_for_exposures(
                exposures,
                self.profiles,
                corpus_id="sl-corpus-v5-scale8000-training-v1",
                augmentation_key=AUGMENTATION_KEY,
                algorithm_version=ALGORITHM_VERSION,
                num_workers=3,
                augmentation_policy=self.policy,
            )
            sample = prepare_virtual_sample(
                exposures[0][0],
                exposures[0][1],
                self.profiles,
                corpus_id="sl-corpus-v5-scale8000-training-v1",
                augmentation_key=AUGMENTATION_KEY,
                algorithm_version=ALGORITHM_VERSION,
                augmentation_policy=self.policy,
            )
            after = set(root.iterdir())
        self.assertEqual(checks["status"], "PASSED")
        self.assertTrue(checks["same_virtual_exposure_after_restart"])
        self.assertTrue(checks["worker_order_independent"])
        self.assertTrue(checks["worker_count_independent"])
        self.assertTrue(checks["transcript_preserved"])
        self.assertEqual(sample.transcript, exposures[0][0].transcript)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
