import unittest

from slaif_asr.artur_earlystop import assert_no_raw_report_material, redacted_checkpoint_row, validate_earlystop_config
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

    def test_artur_earlystop_config_validates_fixed_protocol(self):
        config = {
            "experiment_id": "scale2000-decoder-joint-rnnt-artur-earlystop-v1",
            "work_order_id": "0032",
            "status": "DIAGNOSTIC_ONLY",
            "accepted_parent": "none",
            "training": {
                "semantic_rows": 16000,
                "sample_exposures": 320000,
                "effective_batch_size": 8,
                "max_optimizer_steps": 40000,
                "max_rounds": 20,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "scheduler": "none",
                "gradient_clipping": "none",
                "precision": "fp32",
                "tf32": False,
                "objective": "audio_conditioned_rnnt_loss",
                "trainable_surface": ["decoder.", "joint."],
                "forbid_text_only_path": True,
            },
            "controller_dev": {
                "partition_id": "artur-controller-dev-v1",
                "batch_size": 1,
                "duration_bucketing": False,
                "allowed_for": "aggregate_run_control_and_early_stopping_only",
            },
            "post_selection_directional": {
                "batch_size": 32,
                "duration_bucketing": True,
                "canonical": False,
                "promotion_eligible": False,
            },
        }
        validate_earlystop_config(config)
        config["controller_dev"]["batch_size"] = 32
        with self.assertRaises(ValueError):
            validate_earlystop_config(config)

    def test_round_checkpoint_manifest_redacts_local_path(self):
        row = redacted_checkpoint_row(
            {
                "round": 3,
                "checkpoint_sha256": "abc",
                "optimizer_step": 6000,
                "exposures_seen": 48000,
                "available": True,
                "local_checkpoint_path": "/tmp/not-public/round_03/model.nemo",
            }
        )
        self.assertNotIn("local_checkpoint_path", row)
        self.assertEqual(row["checkpoint_sha256"], "abc")

    def test_public_report_rejects_raw_references_and_hypotheses(self):
        assert_no_raw_report_material({"classification": "ARTUR_EARLYSTOP_NO_REAL_DEV_GAIN"})
        with self.assertRaises(ValueError):
            assert_no_raw_report_material({"raw_reference": "ne sme v javno porocilo"})


if __name__ == "__main__":
    unittest.main()
