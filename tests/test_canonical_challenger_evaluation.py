from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from slaif_asr.canonical_challenger_evaluation import (
    assert_public_report_safe,
    classify_canonical,
    validate_canonical_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def metrics(
    *,
    fleurs_wer: float,
    fleurs_cer: float,
    artur_wer: float,
    artur_cer: float,
    empty: int = 0,
) -> dict[str, dict[str, float | int]]:
    return {
        "fleurs_v2": {"wer": fleurs_wer, "cer": fleurs_cer, "empty": empty},
        "artur_j": {"wer": artur_wer, "cer": artur_cer, "empty": empty},
    }


class CanonicalChallengerEvaluationTests(unittest.TestCase):
    def load_inputs(self):
        config = json.loads(
            (REPO_ROOT / "configs/experiments/surface07_canonical_batch1_evaluation_v1.json").read_text(
                encoding="utf-8"
            )
        )
        policy = json.loads(
            (REPO_ROOT / "configs/evaluation/single-gpu-canonical-batch1-v1.json").read_text(encoding="utf-8")
        )
        return config, policy

    def test_config_accepts_exact_canonical_protocol(self) -> None:
        validate_canonical_config(*self.load_inputs())

    def test_config_rejects_noncanonical_batch_or_bucketing(self) -> None:
        config, policy = self.load_inputs()
        policy["batch_size"] = 32
        with self.assertRaises(ValueError):
            validate_canonical_config(config, policy)
        policy["batch_size"] = 1
        policy["duration_bucketing"] = True
        with self.assertRaises(ValueError):
            validate_canonical_config(config, policy)

    def test_config_rejects_controller_dev(self) -> None:
        config, policy = self.load_inputs()
        config["controller"] = "artur-controller-dev-v1"
        with self.assertRaises(ValueError):
            validate_canonical_config(config, policy)

    def test_surface07_new_best_classification(self) -> None:
        result = classify_canonical(
            {
                "base": metrics(fleurs_wer=52, fleurs_cer=16, artur_wer=67, artur_cer=28, empty=2),
                "pr36_round20": metrics(fleurs_wer=46, fleurs_cer=15, artur_wer=56, artur_cer=20),
                "surface06_round05": metrics(fleurs_wer=44, fleurs_cer=14, artur_wer=51, artur_cer=16),
                "surface07_round13": metrics(fleurs_wer=43, fleurs_cer=13, artur_wer=49, artur_cer=15),
            }
        )
        self.assertEqual(result, "CANONICAL_SURFACE07_CONFIRMED_NEW_BEST")

    def test_surface07_mixed_and_prior_stronger_classifications(self) -> None:
        base = metrics(fleurs_wer=52, fleurs_cer=16, artur_wer=67, artur_cer=28, empty=2)
        mixed = classify_canonical(
            {
                "base": base,
                "surface06_round05": metrics(fleurs_wer=43, fleurs_cer=14, artur_wer=51, artur_cer=16),
                "surface07_round13": metrics(fleurs_wer=43.2, fleurs_cer=13.9, artur_wer=50.8, artur_cer=16.1),
            }
        )
        self.assertEqual(mixed, "CANONICAL_SURFACE07_CONFIRMED_BUT_MIXED")
        prior = classify_canonical(
            {
                "base": base,
                "surface06_round05": metrics(fleurs_wer=43, fleurs_cer=13, artur_wer=49, artur_cer=15),
                "surface07_round13": metrics(fleurs_wer=44, fleurs_cer=14, artur_wer=50, artur_cer=14.9),
            }
        )
        self.assertEqual(prior, "CANONICAL_SURFACE06_OR_PRIOR_STRONGER")

    def test_surface07_missing_or_base_regression_classifications(self) -> None:
        base = metrics(fleurs_wer=52, fleurs_cer=16, artur_wer=67, artur_cer=28, empty=1)
        self.assertEqual(
            classify_canonical({"base": base}),
            "CANONICAL_BLOCKED_SURFACE07_CHECKPOINT_UNAVAILABLE",
        )
        regressed = copy.deepcopy(base)
        regressed["artur_j"]["wer"] = 68
        self.assertEqual(
            classify_canonical({"base": base, "surface07_round13": regressed}),
            "CANONICAL_REJECTS_SURFACE07_DIRECTIONAL_GAIN",
        )

    def test_public_report_rejects_raw_fields_and_local_paths(self) -> None:
        assert_public_report_safe({"classification": "SAFE", "metrics": {"wer": 1.0}})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"reference": "forbidden"})
        with self.assertRaises(ValueError):
            assert_public_report_safe({"note": "/data/private/checkpoint"})


if __name__ == "__main__":
    unittest.main()
