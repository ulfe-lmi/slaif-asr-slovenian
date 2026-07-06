import unittest

from slaif_asr.scale8000_clean_training import classify_scale8000_clean


BASE_BEATING = {
    "piper_synthetic_holdout": {"wer": 54.0, "cer": 19.5, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 26.9, "cer": 7.2, "empty": 0},
    "fleurs_v2": {"wer": 51.0, "cer": 16.0, "empty": 0},
    "artur_j": {"wer": 59.5, "cer": 20.0, "empty": 0},
}


class Scale8000CleanDecisionTests(unittest.TestCase):
    def test_beats_scale2000_when_real_metrics_improve(self):
        decision = classify_scale8000_clean(BASE_BEATING)
        self.assertEqual(decision["classification"], "SCALE8000_CLEAN_BEATS_SCALE2000_AUGMENTED_DIRECTIONAL")
        self.assertEqual(decision["accepted_parent"], "none")

    def test_matches_scale2000_within_thresholds(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 55.8, "cer": 20.4, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 27.8, "cer": 8.0, "empty": 0},
            "fleurs_v2": {"wer": 51.7, "cer": 16.3, "empty": 0},
            "artur_j": {"wer": 60.3, "cer": 20.8, "empty": 0},
        }
        decision = classify_scale8000_clean(metrics)
        self.assertEqual(decision["classification"], "SCALE8000_CLEAN_MATCHES_SCALE2000_AUGMENTED_DIRECTIONAL")

    def test_underperforms_when_real_burden_positive(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 55.0, "cer": 20.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 27.0, "cer": 7.5, "empty": 0},
            "fleurs_v2": {"wer": 54.0, "cer": 16.0, "empty": 0},
            "artur_j": {"wer": 60.0, "cer": 20.0, "empty": 0},
        }
        decision = classify_scale8000_clean(metrics)
        self.assertEqual(decision["classification"], "SCALE8000_CLEAN_UNDERPERFORMS_DIRECTIONALLY")


if __name__ == "__main__":
    unittest.main()
