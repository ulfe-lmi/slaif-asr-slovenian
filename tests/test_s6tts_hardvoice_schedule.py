import unittest

from slaif_asr.s6tts_hardvoice import (
    BASE_DIRECTIONAL_METRICS,
    EXPECTED_PROFILE_IDS,
    PR36_DECODER_JOINT_METRICS,
    classify_hardvoice,
    should_stop_for_controller_dev,
    validate_config,
    validate_hardvoice_schedule,
)


VALID_CONFIG = {
    "work_order_id": "0036",
    "status": "DIAGNOSTIC_ONLY",
    "accepted_parent": "none",
    "schedule": {
        "schedule_id": "scale2000_plus_s6tts_hardvoice_20pct_v1",
        "semantic_rows": 16000,
        "exposures_per_semantic_row": 20,
        "total_exposures": 320000,
        "original_scale2000_exposures_per_row": 16,
        "original_scale2000_exposures": 256000,
        "s6tts_exposures_per_row": 4,
        "s6tts_total_exposures": 64000,
        "s6tts_share": 0.2,
        "s6tts_clean_exposures_per_row": 1,
        "s6tts_augmented_exposures_per_row": 3,
        "s6tts_replacement_rounds": [5, 10, 15, 20],
    },
    "data": {
        "semantic_rows": 16000,
        "fixed_text_sha256": "dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14",
        "base_scale2000_all_views_sha256": "9207429fdd675d6a8ea491f6f6ce3647e1fc9ec22e439c9548ad1120268e3bca",
        "base_scale2000_schedule_sha256": "6757018f3306839ce8564ba758e13e231ab4784bf98049b65701b963b55e5842",
        "s6_clean_manifest_sha256": "355a85134e81d9e3ea4089ea9a941f62fb101902b4e151c394eaaf1d1de416d5",
        "s6_augmented_manifest_sha256": "8d39606dc276a7730e032e83c1811f6c71ece3de6f0b68aa1bd5f4c0a8f50251",
        "s6_augmented_provenance_manifest_sha256": "d18a2c8245e75d94d18c97ae02a3194ddcdc8fdf864bc44cc21fcec8941603ec",
        "s6tts_revision": "6e55c9dad7a9414d8f67e2612862e6fb8b7ff37c",
        "s6tts_runtime_data_hash": "2e71b1ed2df53b7959fa748d9cd1366478895202a874d884ed3986abb581e6dc",
    },
    "training": {
        "sample_exposures": 320000,
        "effective_batch_size": 8,
        "optimizer_steps": 40000,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": "none",
        "gradient_clipping": "none",
        "seed": 1234,
        "precision": "fp32",
        "tf32": False,
        "early_stopping": True,
        "early_stopping_partition": "artur-controller-dev-v1",
        "retain_per_round_checkpoints": True,
        "physical_microbatch_candidates": [8, 4, 2, 1],
    },
    "trainable_surface": {"allowed_prefixes": ["decoder.", "joint."], "text_only_path_allowed": False},
    "evaluation": {"batch_size": 32, "duration_bucketing": True, "canonical": False, "promotion_eligible": False},
    "controller_dev": {
        "partition_id": "artur-controller-dev-v1",
        "manifest_sha256": "7944cbd82107e4aa8cfd3c5ca991d652e4ec3450ba8805efbc98e7c3aeec34f9",
        "batch_size": 1,
        "duration_bucketing": False,
        "allowed_for": "aggregate_run_control_and_early_stopping_only",
    },
    "early_stop_rule": {
        "operational_stop_rule": "stop_after_three_evaluated_rounds_without_new_raw_best_controller_dev_wer",
        "patience_rounds_without_new_raw_best": 3,
    },
}


def make_schedule():
    rows = []
    for round_index in range(1, 21):
        for position in range(16000):
            semantic_key = f"row-{position:05d}"
            if round_index == 5:
                row = {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": semantic_key,
                    "source_schedule": "s6tts",
                    "voice": "s6tts-sl-si-s6-vintage",
                    "profile_id": "clean",
                    "view_type": "clean",
                }
            elif round_index in {10, 15, 20}:
                slot = {10: 0, 15: 1, 20: 2}[round_index]
                row = {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": semantic_key,
                    "source_schedule": "s6tts",
                    "voice": "s6tts-sl-si-s6-vintage",
                    "profile_id": EXPECTED_PROFILE_IDS[(position * 3 + slot) % len(EXPECTED_PROFILE_IDS)],
                    "view_type": "augmented",
                }
            else:
                row = {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": semantic_key,
                    "source_schedule": "scale2000",
                    "voice": "piper-sl_SI-artur-medium",
                    "profile_id": "clean",
                    "view_type": "clean",
                }
            rows.append(row)
    return rows


class S6TtsHardvoiceScheduleTests(unittest.TestCase):
    def test_config_requires_fixed_budget_and_s6_share(self):
        validate_config(VALID_CONFIG)
        bad = {**VALID_CONFIG, "schedule": {**VALID_CONFIG["schedule"], "total_exposures": 512000}}
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_schedule_counts_and_balanced_profiles(self):
        summary = validate_hardvoice_schedule(make_schedule())
        self.assertEqual(summary["total_exposures"], 320000)
        self.assertEqual(summary["original_scale2000_exposures"], 256000)
        self.assertEqual(summary["s6tts_exposures"], 64000)
        self.assertEqual(summary["s6_clean_exposures"], 16000)
        self.assertEqual(summary["s6_augmented_exposures"], 48000)
        counts = list(summary["profile_distribution"].values())
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_schedule_rejects_duplicate_in_round(self):
        schedule = make_schedule()
        schedule[1]["semantic_key"] = schedule[0]["semantic_key"]
        with self.assertRaises(ValueError):
            validate_hardvoice_schedule(schedule)

    def test_schedule_rejects_heldout_voice_leakage(self):
        schedule = make_schedule()
        schedule[0]["voice"] = "supertonic-M5"
        with self.assertRaises(ValueError):
            validate_hardvoice_schedule(schedule)

    def test_controller_dev_stop_after_three_rounds_without_raw_best(self):
        config = {**VALID_CONFIG, "early_stop_rule": {**VALID_CONFIG["early_stop_rule"], "min_rounds_before_stop": 3}}
        self.assertFalse(
            should_stop_for_controller_dev(
                config,
                [
                    {"round": 0, "wer": 66.0},
                    {"round": 1, "wer": 61.0},
                    {"round": 2, "wer": 62.0},
                    {"round": 3, "wer": 63.0},
                ],
            )
        )
        self.assertTrue(
            should_stop_for_controller_dev(
                config,
                [
                    {"round": 0, "wer": 66.0},
                    {"round": 1, "wer": 61.0},
                    {"round": 2, "wer": 62.0},
                    {"round": 3, "wer": 63.0},
                    {"round": 4, "wer": 64.0},
                ],
            )
        )
        self.assertFalse(
            should_stop_for_controller_dev(
                config,
                [
                    {"round": 0, "wer": 66.0},
                    {"round": 1, "wer": 61.0},
                    {"round": 2, "wer": 62.0},
                    {"round": 3, "wer": 60.5},
                    {"round": 4, "wer": 61.5},
                ],
            )
        )

    def test_positive_real_safe_classification(self):
        metrics = {
            "s6tts_clean_holdout": {"wer": 30.0, "cer": 10.0, "empty": 0},
            "s6tts_augmented_holdout": {"wer": 35.0, "cer": 12.0, "empty": 0},
            "piper_synthetic_holdout": PR36_DECODER_JOINT_METRICS["piper_synthetic_holdout"],
            "supertonic_heldout_voice_holdout": PR36_DECODER_JOINT_METRICS["supertonic_heldout_voice_holdout"],
            "fleurs_v2": PR36_DECODER_JOINT_METRICS["fleurs_v2"],
            "artur_j": PR36_DECODER_JOINT_METRICS["artur_j"],
        }
        base_s6 = {
            "s6tts_clean_holdout": {"wer": 80.0, "cer": 45.0, "empty": 10},
            "s6tts_augmented_holdout": {"wer": 82.0, "cer": 47.0, "empty": 12},
        }
        decision = classify_hardvoice(metrics, base_s6_metrics=base_s6)
        self.assertEqual(decision["classification"], "S6TTS_HARDVOICE_IMPACT_POSITIVE_REAL_SAFE")
        self.assertEqual(decision["accepted_parent"], "none")

    def test_real_regression_classification(self):
        metrics = {
            "s6tts_clean_holdout": {"wer": 30.0, "cer": 10.0, "empty": 0},
            "s6tts_augmented_holdout": {"wer": 35.0, "cer": 12.0, "empty": 0},
            "piper_synthetic_holdout": PR36_DECODER_JOINT_METRICS["piper_synthetic_holdout"],
            "supertonic_heldout_voice_holdout": PR36_DECODER_JOINT_METRICS["supertonic_heldout_voice_holdout"],
            "fleurs_v2": {"wer": BASE_DIRECTIONAL_METRICS["fleurs_v2"]["wer"] + 0.1, "cer": 15.0, "empty": 0},
            "artur_j": PR36_DECODER_JOINT_METRICS["artur_j"],
        }
        base_s6 = {
            "s6tts_clean_holdout": {"wer": 80.0, "cer": 45.0, "empty": 10},
            "s6tts_augmented_holdout": {"wer": 82.0, "cer": 47.0, "empty": 12},
        }
        decision = classify_hardvoice(metrics, base_s6_metrics=base_s6)
        self.assertEqual(decision["classification"], "S6TTS_HARDVOICE_REAL_REGRESSION")


if __name__ == "__main__":
    unittest.main()
