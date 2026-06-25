from __future__ import annotations

import unittest

from slaif_asr.batched_streaming import StreamingRecord
from slaif_asr.directional_evaluation import (
    classify_directional,
    heldout_synthetic_gain,
    privacy_safe_public_report,
    real_improvement,
    real_non_regression,
    real_regression_burden,
    split_predictions,
    suite_plan_hash,
)


def rec(sample_id: str, split: str, index: int) -> StreamingRecord:
    return StreamingRecord(
        sample_id=sample_id,
        audio_filepath=f"/tmp/{sample_id}.wav",
        duration=1.0 + index,
        reference="varna referenca",
        original_index=index,
        row={"split": split, "source_order": index},
    )


class DirectionalEvaluationTests(unittest.TestCase):
    def test_split_predictions_rejects_missing_and_unexpected_ids(self) -> None:
        records = [rec("a", "x", 0), rec("b", "y", 1)]
        split_records = {"x": [records[0]], "y": [records[1]]}
        self.assertEqual(split_predictions(records, split_records, {"a": "ena", "b": "dve"}), {"x": {"a": "ena"}, "y": {"b": "dve"}})
        with self.assertRaisesRegex(RuntimeError, "prediction mismatch"):
            split_predictions(records, split_records, {"a": "ena", "c": "tri"})

    def test_suite_plan_hash_is_stable(self) -> None:
        records = [rec("x:0000", "x", 0), rec("y:0000", "y", 1)]
        self.assertEqual(suite_plan_hash(records), suite_plan_hash(list(records)))

    def test_real_burden_non_regression_and_improvement_logic(self) -> None:
        base = {
            "fleurs_v2": {"wer": 50.0, "cer": 10.0, "empty": 1},
            "artur_j": {"wer": 60.0, "cer": 20.0, "empty": 2},
        }
        candidate = {
            "fleurs_v2": {"wer": 50.5, "cer": 11.0, "empty": 1},
            "artur_j": {"wer": 59.0, "cer": 18.0, "empty": 2},
        }
        self.assertEqual(real_regression_burden(candidate, base), 1.5)
        self.assertTrue(real_non_regression(candidate, base))
        self.assertTrue(real_improvement(candidate, base))
        candidate["artur_j"]["empty"] = 3
        self.assertFalse(real_non_regression(candidate, base))

    def test_heldout_gain_uses_batch32_base(self) -> None:
        base = {"supertonic_heldout_voice_holdout": {"wer": 70.0, "cer": 30.0, "empty": 0}}
        self.assertTrue(heldout_synthetic_gain({"supertonic_heldout_voice_holdout": {"wer": 69.0, "cer": 30.0, "empty": 0}}, base))
        self.assertFalse(heldout_synthetic_gain({"supertonic_heldout_voice_holdout": {"wer": 70.0, "cer": 30.0, "empty": 0}}, base))

    def test_directional_classification_confirms_when_both_supertonic_models_reduce_burden(self) -> None:
        models = {
            "base": {
                "supertonic_heldout_voice_holdout": {"wer": 70.0, "cer": 30.0, "empty": 0},
                "fleurs_v2": {"wer": 50.0, "cer": 10.0, "empty": 1},
                "artur_j": {"wer": 60.0, "cer": 20.0, "empty": 2},
            },
            "piper_joint_adapter": {
                "supertonic_heldout_voice_holdout": {"wer": 65.0, "cer": 25.0, "empty": 0},
                "fleurs_v2": {"wer": 60.0, "cer": 15.0, "empty": 0},
                "artur_j": {"wer": 66.0, "cer": 23.0, "empty": 0},
            },
            "supertonic3_joint_adapter": {
                "supertonic_heldout_voice_holdout": {"wer": 55.0, "cer": 20.0, "empty": 0},
                "fleurs_v2": {"wer": 57.0, "cer": 13.0, "empty": 0},
                "artur_j": {"wer": 63.0, "cer": 21.0, "empty": 0},
            },
            "batched_replay_joint_adapter": {
                "supertonic_heldout_voice_holdout": {"wer": 56.0, "cer": 21.0, "empty": 0},
                "fleurs_v2": {"wer": 56.0, "cer": 13.0, "empty": 0},
                "artur_j": {"wer": 63.0, "cer": 21.0, "empty": 0},
            },
        }
        self.assertEqual(classify_directional(models)["classification"], "FAST_DIRECTIONAL_REPLAY_CONFIRMS_CONCLUSION")

    def test_directional_classification_positive_when_replay_non_regresses_and_improves(self) -> None:
        models = {
            "base": {
                "supertonic_heldout_voice_holdout": {"wer": 70.0, "cer": 30.0, "empty": 0},
                "fleurs_v2": {"wer": 50.0, "cer": 10.0, "empty": 1},
                "artur_j": {"wer": 60.0, "cer": 20.0, "empty": 2},
            },
            "piper_joint_adapter": {
                "supertonic_heldout_voice_holdout": {"wer": 65.0, "cer": 25.0, "empty": 0},
                "fleurs_v2": {"wer": 60.0, "cer": 15.0, "empty": 0},
                "artur_j": {"wer": 66.0, "cer": 23.0, "empty": 0},
            },
            "supertonic3_joint_adapter": {
                "supertonic_heldout_voice_holdout": {"wer": 55.0, "cer": 20.0, "empty": 0},
                "fleurs_v2": {"wer": 57.0, "cer": 13.0, "empty": 0},
                "artur_j": {"wer": 63.0, "cer": 21.0, "empty": 0},
            },
            "batched_replay_joint_adapter": {
                "supertonic_heldout_voice_holdout": {"wer": 50.0, "cer": 20.0, "empty": 0},
                "fleurs_v2": {"wer": 49.0, "cer": 10.5, "empty": 1},
                "artur_j": {"wer": 60.5, "cer": 19.0, "empty": 2},
            },
        }
        self.assertEqual(classify_directional(models)["classification"], "FAST_DIRECTIONAL_REPLAY_CHANGES_CONCLUSION_POSITIVELY")

    def test_public_report_privacy_rejects_ids_text_and_paths(self) -> None:
        with self.assertRaises(ValueError):
            privacy_safe_public_report({"text": "Ne objavi."})
        with self.assertRaises(ValueError):
            privacy_safe_public_report({"message": "gams9holdout-cell01-a00-o001"})
        with self.assertRaises(ValueError):
            privacy_safe_public_report({"message": "/home/user/private.wav"})


if __name__ == "__main__":
    unittest.main()
