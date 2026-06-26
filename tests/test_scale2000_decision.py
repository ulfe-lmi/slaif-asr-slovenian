from __future__ import annotations

import unittest

from slaif_asr.scale2000_corpus import classify_scale2000


BASE = {
    "piper_synthetic_holdout": {"wer": 86.025, "cer": 46.762, "empty": 17},
    "supertonic_heldout_voice_holdout": {"wer": 58.307, "cer": 27.712, "empty": 32},
    "fleurs_v2": {"wer": 52.685, "cer": 16.406, "empty": 1},
    "artur_j": {"wer": 67.322, "cer": 28.620, "empty": 12},
}
SCALE200 = {
    "piper_synthetic_holdout": {"wer": 63.509, "cer": 23.830, "empty": 1},
    "supertonic_heldout_voice_holdout": {"wer": 36.879, "cer": 11.088, "empty": 3},
    "fleurs_v2": {"wer": 54.386, "cer": 17.573, "empty": 0},
    "artur_j": {"wer": 64.176, "cer": 22.753, "empty": 1},
}


class Scale2000DecisionTests(unittest.TestCase):
    def test_real_gain_directional(self) -> None:
        metrics = {
            **SCALE200,
            "fleurs_v2": {"wer": 51.4, "cer": 15.0, "empty": 0},
            "artur_j": {"wer": 64.0, "cer": 22.0, "empty": 1},
        }
        decision = classify_scale2000(base_metrics=BASE, scale200_metrics=SCALE200, scale2000_metrics=metrics)
        self.assertEqual(decision["classification"], "SCALE2000_TEXT_REAL_GAIN_DIRECTIONAL")
        self.assertEqual(decision["accepted_parent"], "none")

    def test_improves_scale200(self) -> None:
        metrics = {
            **SCALE200,
            "fleurs_v2": {"wer": 53.8, "cer": 16.9, "empty": 0},
            "artur_j": {"wer": 64.2, "cer": 22.8, "empty": 1},
        }
        decision = classify_scale2000(base_metrics=BASE, scale200_metrics=SCALE200, scale2000_metrics=metrics)
        self.assertEqual(decision["classification"], "SCALE2000_TEXT_IMPROVES_SCALE200")
        self.assertLessEqual(decision["scale2000_burden"], 2.0076)

    def test_plateaus(self) -> None:
        decision = classify_scale2000(base_metrics=BASE, scale200_metrics=SCALE200, scale2000_metrics=SCALE200)
        self.assertEqual(decision["classification"], "SCALE2000_TEXT_PLATEAUS")

    def test_degrades_when_synthetic_gain_lost(self) -> None:
        metrics = {
            **SCALE200,
            "piper_synthetic_holdout": {"wer": 90.0, "cer": 50.0, "empty": 20},
        }
        decision = classify_scale2000(base_metrics=BASE, scale200_metrics=SCALE200, scale2000_metrics=metrics)
        self.assertEqual(decision["classification"], "SCALE2000_TEXT_DEGRADES")


if __name__ == "__main__":
    unittest.main()
