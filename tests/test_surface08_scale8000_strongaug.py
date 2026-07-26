from __future__ import annotations

import copy
import json
import tempfile
import unittest
import wave
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from scripts.run_surface08_scale8000_otf_strongaug_v1 import (
        STANDARD_OTF_METRICS,
        classify_result,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    STANDARD_OTF_METRICS = {}
    classify_result = None
from slaif_asr.config import REPO_ROOT
from slaif_asr.otf_augmentation_dataset import (
    CleanAudioRecord,
    prepare_virtual_sample,
    run_determinism_checks_for_exposures,
    virtual_augmentation_spec,
)
from slaif_asr.scale200_corpus import load_augmentation_config
from slaif_asr.scale8000_strongaug_surface08 import (
    ALGORITHM_VERSION,
    AUGMENTATION_KEY,
    load_config,
    load_policy_for_config,
    validate_config,
)
from slaif_asr.strongaug_v1 import (
    MAX_OPERATIONS,
    phase_for_exposure,
    summarize_profile_counts,
)


CONFIG_PATH = (
    REPO_ROOT / "configs/experiments/surface08-scale8000-otf-strongaug-v1.json"
)
PROFILE_PATH = REPO_ROOT / "configs/augmentation/scale200_transcript_preserving_v1.json"
requires_numpy = unittest.skipUnless(
    np is not None, "NumPy is required for waveform-backed StrongAug tests"
)
requires_training_runner = unittest.skipUnless(
    classify_result is not None,
    "Torch is required to import the StrongAug training runner",
)


def write_wav(path: Path) -> None:
    values = (
        np.sin(np.linspace(0.0, 24.0, 3200, endpoint=False)) * 8000.0
    ).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(values.tobytes())


def metric_table() -> dict[str, dict[str, float | int]]:
    return copy.deepcopy(STANDARD_OTF_METRICS)


def training_summary() -> dict[str, object]:
    return {
        "loader_fill_rate": 0.99,
        "parameter_integrity": {"only_surface08_changed": True},
    }


class Surface08Scale8000StrongAugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.loaded = load_config(CONFIG_PATH)
        cls.policy = load_policy_for_config(cls.loaded)
        cls.profiles = list(
            load_augmentation_config(PROFILE_PATH)["augmentation_profiles"]
        )

    def test_config_locks_surface_data_base_and_budget(self) -> None:
        self.assertEqual(self.loaded["model"]["initialization"], "untouched_base")
        self.assertEqual(
            self.loaded["data"]["training_source"], "scale8000_clean_otf_only"
        )
        self.assertEqual(self.loaded["training"]["physical_microbatch"], 2)
        self.assertEqual(self.loaded["training"]["gradient_accumulation_steps"], 4)
        self.assertEqual(self.loaded["training"]["effective_batch_size"], 8)
        self.assertEqual(self.loaded["training"]["max_rounds"], 9)
        self.assertEqual(self.loaded["training"]["max_exposures"], 144_000)
        self.assertEqual(self.loaded["training"]["max_optimizer_steps"], 18_000)

    def test_changed_base_surface_or_data_is_rejected(self) -> None:
        changes = (
            ("model", "initialization", "surface08_scale8000_standard_otf"),
            ("data", "training_source", "scale2000_training"),
            ("trainable_surface", "surface_id", "SURFACE_09_FULL_MODEL"),
        )
        for section, key, value in changes:
            changed = copy.deepcopy(self.config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_curriculum_phase_boundaries_are_fixed(self) -> None:
        self.assertEqual(phase_for_exposure(0), (1, "coverage", 0.30))
        self.assertEqual(phase_for_exposure(63_999), (4, "coverage", 0.30))
        self.assertEqual(phase_for_exposure(64_000), (5, "robustness", 0.75))
        self.assertEqual(phase_for_exposure(143_999), (9, "robustness", 0.75))
        with self.assertRaises(ValueError):
            phase_for_exposure(144_000)

    def test_spec_is_restart_stable_and_bounded(self) -> None:
        record = CleanAudioRecord(
            semantic_key="semantic-1",
            voice="F1",
            source_audio_sha256="1" * 64,
            audio_path=Path("unused.wav"),
            duration_seconds=1.0,
            transcript="synthetic fixture",
        )
        first = virtual_augmentation_spec(
            record,
            80_000,
            self.profiles,
            corpus_id="sl-corpus-v5-scale8000-training-v1",
            augmentation_key=AUGMENTATION_KEY,
            algorithm_version=ALGORITHM_VERSION,
            augmentation_policy=self.policy,
        )
        second = virtual_augmentation_spec(
            record,
            80_000,
            self.profiles,
            corpus_id="sl-corpus-v5-scale8000-training-v1",
            augmentation_key=AUGMENTATION_KEY,
            algorithm_version=ALGORITHM_VERSION,
            augmentation_policy=self.policy,
        )
        self.assertEqual(first, second)
        if first.parameters["mode"] == "strong":
            operations = first.parameters["operations"]
            self.assertGreaterEqual(len(operations), 2)
            self.assertLessEqual(len(operations), MAX_OPERATIONS)
            for operation in operations:
                if operation["family"] == "additive_noise":
                    self.assertGreaterEqual(operation["parameters"]["snr_db"], 8.0)

    @requires_numpy
    def test_waveform_replays_across_worker_order_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "source.wav"
            write_wav(path)
            records = [
                CleanAudioRecord(
                    semantic_key=f"semantic-{index}",
                    voice="F1",
                    source_audio_sha256=f"{index + 1:064x}",
                    audio_path=path,
                    duration_seconds=0.2,
                    transcript=f"synthetic transcript {index}",
                )
                for index in range(4)
            ]
            exposures = [
                (record, 64_000 + index) for index, record in enumerate(records)
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
        self.assertTrue(checks["worker_order_independent"])
        self.assertTrue(checks["worker_count_independent"])
        self.assertEqual(sample.transcript, exposures[0][0].transcript)
        self.assertEqual(sample.sample_rate, 16_000)
        self.assertEqual(before, after)

    def test_profile_summary_reports_phase_mixture_and_families(self) -> None:
        summary = summarize_profile_counts(
            {
                "standard_otf/coverage/reverb": 70,
                "strongaug_v1/coverage/additive_noise+reverb": 30,
                "standard_otf/robustness/reverb": 25,
                "strongaug_v1/robustness/additive_noise+channel_filter": 75,
            }
        )
        self.assertEqual(summary["standard_fraction"], 0.475)
        self.assertEqual(summary["strong_fraction"], 0.525)
        self.assertEqual(summary["phases"]["coverage"]["strong_fraction"], 0.3)
        self.assertEqual(summary["phases"]["robustness"]["strong_fraction"], 0.75)
        self.assertEqual(summary["family_sample_counts"]["additive_noise"], 105)

    @requires_training_runner
    def test_classifier_uses_standard_otf_as_critical_comparator(self) -> None:
        improved = metric_table()
        for split in ("fleurs_v2", "artur_j"):
            improved[split]["wer"] = float(improved[split]["wer"]) - 0.1
            improved[split]["cer"] = float(improved[split]["cer"]) - 0.1
        self.assertEqual(
            classify_result(improved, training=training_summary()),
            "STRONGAUG_V1_NEW_BEST_DIRECTIONAL",
        )

        mixed = metric_table()
        mixed["fleurs_v2"]["wer"] = 39.8
        self.assertEqual(
            classify_result(mixed, training=training_summary()),
            "STRONGAUG_V1_MATCHES_STANDARD_OTF",
        )

        regressed = metric_table()
        regressed["artur_j"]["wer"] = 40.6
        self.assertEqual(
            classify_result(regressed, training=training_summary()),
            "STRONGAUG_V1_REAL_GATE_REGRESSION",
        )

    def test_governance_names_work_order_0046(self) -> None:
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        adr = (REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("Work Order 0046", agents)
        self.assertIn("Phase 7 / Work Order 0046", adr)


if __name__ == "__main__":
    unittest.main()
