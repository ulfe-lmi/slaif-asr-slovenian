import unittest

from slaif_asr.emission_rnnt_finetune import classify_decoder_joint_rnnt


BEATS = {
    "piper_synthetic_holdout": {"wer": 55.0, "cer": 19.5, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 26.8, "cer": 7.2, "empty": 0},
    "fleurs_v2": {"wer": 51.0, "cer": 16.0, "empty": 0},
    "artur_j": {"wer": 59.5, "cer": 20.0, "empty": 0},
}


class Scale2000DecoderJointDecisionTests(unittest.TestCase):
    def test_beats_scale2000_when_two_real_metrics_improve(self):
        decision = classify_decoder_joint_rnnt(BEATS)
        self.assertEqual(decision["classification"], "DECODER_JOINT_RNNT_BEATS_SCALE2000_DIRECTIONAL")
        self.assertEqual(decision["accepted_parent"], "none")

    def test_matches_scale2000_within_thresholds(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 55.8, "cer": 20.4, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 27.8, "cer": 8.0, "empty": 0},
            "fleurs_v2": {"wer": 51.7, "cer": 16.3, "empty": 0},
            "artur_j": {"wer": 60.3, "cer": 20.8, "empty": 0},
        }
        decision = classify_decoder_joint_rnnt(metrics)
        self.assertEqual(decision["classification"], "DECODER_JOINT_RNNT_MATCHES_SCALE2000_DIRECTIONAL")

    def test_beats_base_but_not_scale2000(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 60.0, "cer": 21.5, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 28.4, "cer": 8.3, "empty": 0},
            "fleurs_v2": {"wer": 51.8, "cer": 16.3, "empty": 0},
            "artur_j": {"wer": 61.0, "cer": 21.4, "empty": 0},
        }
        decision = classify_decoder_joint_rnnt(metrics)
        self.assertEqual(decision["classification"], "DECODER_JOINT_RNNT_BEATS_BASE_BUT_NOT_SCALE2000")

    def test_regresses_when_real_burden_positive(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 55.0, "cer": 20.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 27.0, "cer": 7.5, "empty": 0},
            "fleurs_v2": {"wer": 54.0, "cer": 16.0, "empty": 0},
            "artur_j": {"wer": 60.0, "cer": 20.0, "empty": 0},
        }
        decision = classify_decoder_joint_rnnt(metrics)
        self.assertEqual(decision["classification"], "DECODER_JOINT_RNNT_SYNTHETIC_ONLY_OR_REGRESSES")

    def test_regresses_when_synthetic_gain_disappears(self):
        metrics = {
            "piper_synthetic_holdout": {"wer": 87.0, "cer": 47.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 27.0, "cer": 7.5, "empty": 0},
            "fleurs_v2": {"wer": 51.0, "cer": 16.0, "empty": 0},
            "artur_j": {"wer": 60.0, "cer": 20.0, "empty": 0},
        }
        decision = classify_decoder_joint_rnnt(metrics)
        self.assertEqual(decision["classification"], "DECODER_JOINT_RNNT_SYNTHETIC_ONLY_OR_REGRESSES")


if __name__ == "__main__":
    unittest.main()
