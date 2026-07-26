from __future__ import annotations

import copy
import json
import unittest

from scripts.run_surface08_scale8000_otf_parametric_voiceaug_v1 import (
    STANDARD_OTF_METRICS,
    classify_result,
)
from slaif_asr.config import REPO_ROOT
from slaif_asr.scale8000_otf_surface08 import validate_public_report
from slaif_asr.scale8000_parametric_voiceaug_surface08 import (
    AUGMENTATION_KEY,
    load_config,
    validate_config,
)


CONFIG_PATH = (
    REPO_ROOT
    / "configs/experiments/surface08-scale8000-otf-parametric-voiceaug-v1.json"
)


def training_summary() -> dict[str, object]:
    return {
        "loader_fill_rate": 0.99,
        "loader_telemetry": [{"worker_failures": 0}],
        "parameter_integrity": {"only_surface08_changed": True},
    }


class ParametricVoiceAugScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.config = load_config(CONFIG_PATH)

    def test_control_surface_data_base_and_budget_are_fixed(self) -> None:
        self.assertEqual(self.config["work_order_id"], "0047")
        self.assertEqual(self.config["model"]["initialization"], "untouched_base")
        self.assertEqual(
            self.config["data"]["training_source"], "scale8000_clean_otf_only"
        )
        self.assertEqual(self.config["data"]["semantic_rows"], 64_000)
        self.assertEqual(self.config["data"]["clean_files"], 576_000)
        self.assertEqual(
            self.config["trainable_surface"]["surface_id"],
            "SURFACE_08_FULL_ENCODER",
        )
        self.assertEqual(
            self.config["trainable_surface"]["encoder_layer_indices"],
            list(range(24)),
        )
        self.assertEqual(self.config["training"]["max_exposures"], 144_000)
        self.assertEqual(self.config["training"]["max_rounds"], 9)
        self.assertEqual(self.config["training"]["physical_microbatch"], 2)
        self.assertEqual(
            self.config["training"]["gradient_accumulation_steps"], 4
        )
        self.assertEqual(self.config["training"]["effective_batch_size"], 8)

    def test_parametric_policy_is_default_and_strongaug_is_comparator_only(self) -> None:
        augmentation = self.config["otf_augmentation"]
        self.assertEqual(augmentation["augmentation_key"], AUGMENTATION_KEY)
        self.assertEqual(
            augmentation["augmentation_family"], "parametric_voiceaug_v1"
        )
        self.assertNotIn("strongaug", json.dumps(augmentation).lower())
        self.assertEqual(
            self.config["standard_otf_comparator"]["experiment"],
            "0031-surface08-scale8000-otf-augmented",
        )
        self.assertEqual(
            self.config["strongaug_negative_comparator"]["classification"],
            "STRONGAUG_V1_REAL_GATE_REGRESSION",
        )

    def test_invalid_data_model_selection_or_policy_is_rejected(self) -> None:
        changes = (
            ("data", "training_source", "scale2000_training"),
            ("data", "training_source", "s6tts_training"),
            ("data", "training_source", "real_speech_training"),
            ("model", "initialization", "surface08_scale8000_standard_otf"),
            ("trainable_surface", "surface_id", "SURFACE_09_FULL_MODEL"),
            ("training", "effective_batch_size", 4),
            ("training", "max_exposures", 160_000),
            ("otf_augmentation", "augmentation_key", "scale8000-otf-strongaug-v1"),
        )
        for section, key, value in changes:
            changed = copy.deepcopy(self.raw)
            changed[section][key] = value
            with self.subTest(section=section, key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_config(changed)
        changed = copy.deepcopy(self.raw)
        changed["controller_dev"]["partition_id"] = "fleurs_v2"
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_dependency_provenance_is_required(self) -> None:
        changed = copy.deepcopy(self.raw)
        changed["otf_augmentation"]["dependencies"][0]["license"] = ""
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_classifier_uses_standard_otf_as_critical_comparator(self) -> None:
        improved = copy.deepcopy(STANDARD_OTF_METRICS)
        for split in ("fleurs_v2", "artur_j"):
            improved[split]["wer"] -= 0.1
            improved[split]["cer"] -= 0.1
        self.assertEqual(
            classify_result(improved, training=training_summary()),
            "PARAMETRIC_VOICEAUG_V1_NEW_BEST_DIRECTIONAL",
        )

        tradeoff = copy.deepcopy(improved)
        tradeoff["piper_synthetic_holdout"]["wer"] += 0.2
        self.assertEqual(
            classify_result(tradeoff, training=training_summary()),
            "PARAMETRIC_VOICEAUG_V1_REAL_GATE_GAIN_SYNTHETIC_TRADEOFF",
        )

        regressed = copy.deepcopy(STANDARD_OTF_METRICS)
        regressed["fleurs_v2"]["wer"] += 0.6
        self.assertEqual(
            classify_result(regressed, training=training_summary()),
            "PARAMETRIC_VOICEAUG_V1_REAL_GATE_REGRESSION",
        )

    def test_public_report_rejects_raw_fields_and_local_paths(self) -> None:
        validate_public_report(
            {
                "classification": "DIAGNOSTIC_ONLY",
                "augmentation_key": AUGMENTATION_KEY,
                "semantic_rows": 64_000,
            }
        )
        with self.assertRaises(ValueError):
            validate_public_report({"local_path": "/private/data"})
        with self.assertRaises(ValueError):
            validate_public_report({"reference": "raw transcript"})
        with self.assertRaises(ValueError):
            validate_public_report({"hypothesis": "raw hypothesis"})

    def test_governance_names_work_order_0047(self) -> None:
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        adr = (REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md").read_text(
            encoding="utf-8"
        )
        work_order = (
            REPO_ROOT
            / "docs/work-orders/0047-surface08-scale8000-otf-parametric-voiceaug-v1.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Work Order 0047", agents)
        self.assertIn("Phase 8 / Work Order 0047", adr)
        self.assertIn("scale8000-otf-parametric-voiceaug-v1", work_order)


if __name__ == "__main__":
    unittest.main()
