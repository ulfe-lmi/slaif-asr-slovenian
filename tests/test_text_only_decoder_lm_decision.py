import unittest

from slaif_asr.text_only_decoder_lm import classify_text_only, real_regression_burden


BASE_BEATING = {
    "piper_synthetic_holdout": {"wer": 80.0, "cer": 40.0, "empty": 10},
    "supertonic_heldout_voice_holdout": {"wer": 50.0, "cer": 20.0, "empty": 10},
    "fleurs_v2": {"wer": 52.0, "cer": 16.0, "empty": 1},
    "artur_j": {"wer": 66.0, "cer": 28.0, "empty": 10},
}


class TextOnlyDecoderLMDecisionTests(unittest.TestCase):
    def test_real_gain_directional(self):
        decision = classify_text_only(BASE_BEATING, text_validation_improved=True)
        self.assertEqual(decision["classification"], "TEXT_ONLY_DECODER_LM_REAL_GAIN_DIRECTIONAL")
        self.assertEqual(decision["accepted_parent"], "none")
        self.assertEqual(decision["real_regression_burden"], 0.0)

    def test_synthetic_only_when_real_burden_positive(self):
        metrics = {
            **BASE_BEATING,
            "fleurs_v2": {"wer": 53.0, "cer": 16.0, "empty": 1},
        }
        decision = classify_text_only(metrics, text_validation_improved=True)
        self.assertEqual(decision["classification"], "TEXT_ONLY_DECODER_LM_HELPS_SYNTHETIC_ONLY")
        self.assertGreater(real_regression_burden(metrics), 0.0)

    def test_no_asr_gain_when_text_loss_only(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 86.025, "cer": 46.762, "empty": 17},
            "supertonic_heldout_voice_holdout": {"wer": 58.307, "cer": 27.712, "empty": 32},
            "fleurs_v2": {"wer": 52.685, "cer": 16.406, "empty": 1},
            "artur_j": {"wer": 67.322, "cer": 28.620, "empty": 12},
        }
        decision = classify_text_only(metrics, text_validation_improved=True)
        self.assertEqual(decision["classification"], "TEXT_ONLY_DECODER_LM_NO_ASR_GAIN")

    def test_degrades_when_real_gate_worsens(self):
        metrics = {
            **BASE_BEATING,
            "artur_j": {"wer": 70.0, "cer": 30.0, "empty": 20},
        }
        decision = classify_text_only(metrics, text_validation_improved=True)
        self.assertEqual(decision["classification"], "TEXT_ONLY_DECODER_LM_DEGRADES_ASR")


if __name__ == "__main__":
    unittest.main()
