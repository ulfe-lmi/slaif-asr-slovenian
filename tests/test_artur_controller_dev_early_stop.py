from __future__ import annotations

import unittest

from slaif_asr.artur_controller_dev import select_earliest_within_tolerance, watcher_contract_valid
from slaif_asr.artur_earlystop import (
    assert_post_selection_does_not_change_selection,
    classify_artur_earlystop,
    concurrent_gpu_contract,
    validate_agents_controller_dev_exception,
)


class ArturControllerDevEarlyStopTests(unittest.TestCase):
    def test_agents_policy_contains_controller_dev_exception_without_gate_weakening(self) -> None:
        with open("AGENTS.md", encoding="utf-8") as fp:
            text = fp.read()
        self.assertTrue(validate_agents_controller_dev_exception(text))
        self.assertIn("Immutable gates and final blind tests remain unavailable", text)

    def test_early_stop_can_select_round_zero_base(self) -> None:
        rows = [
            {"round": 0, "available": True, "wer": 20.0, "cer": 8.0, "empty": 0},
            {"round": 1, "available": True, "wer": 21.0, "cer": 8.4, "empty": 0},
        ]
        selected = select_earliest_within_tolerance(rows, base_empty_count=0)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["round"], 0)

    def test_early_stop_chooses_earliest_checkpoint_within_tolerance(self) -> None:
        rows = [
            {"round": 1, "available": True, "wer": 20.4, "cer": 8.1, "empty": 0},
            {"round": 2, "available": True, "wer": 20.0, "cer": 8.0, "empty": 0},
            {"round": 3, "available": True, "wer": 19.9, "cer": 7.9, "empty": 0},
        ]
        selected = select_earliest_within_tolerance(rows, base_empty_count=0)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["round"], 1)


    def test_early_stop_rejects_empty_hypothesis_regression(self) -> None:
        rows = [
            {"round": 1, "available": True, "wer": 20.4, "cer": 8.1, "empty": 1},
            {"round": 2, "available": True, "wer": 20.0, "cer": 8.0, "empty": 0},
        ]
        selected = select_earliest_within_tolerance(rows, base_empty_count=0)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["round"], 2)

    def test_watcher_refuses_same_training_and_evaluation_gpu(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "runs" / "checkpoints"
            metrics_dir = Path(tmp) / "runs" / "metrics"
            with self.assertRaisesRegex(ValueError, "must differ"):
                watcher_contract_valid("0", "0", checkpoint_dir, metrics_dir)

    def test_concurrent_gpu_contract_requires_different_selectors(self) -> None:
        with self.assertRaisesRegex(ValueError, "must differ"):
            concurrent_gpu_contract("0", "0", sequential=False)
        self.assertEqual(concurrent_gpu_contract("0", "0", sequential=True)["mode"], "sequential")
        self.assertEqual(concurrent_gpu_contract("0", "1", sequential=False)["mode"], "concurrent")

    def test_post_selection_directional_metrics_cannot_change_selected_round(self) -> None:
        assert_post_selection_does_not_change_selection(3, 3)
        with self.assertRaisesRegex(RuntimeError, "must not change"):
            assert_post_selection_does_not_change_selection(3, 4)

    def test_artur_earlystop_classifies_base_selection_as_no_gain(self) -> None:
        decision = classify_artur_earlystop(
            selected_round=0,
            max_round=20,
            controller_rows=[{"available": True}],
            selected_directional_metrics=None,
        )
        self.assertEqual(decision["classification"], "ARTUR_EARLYSTOP_NO_REAL_DEV_GAIN")

    def test_artur_earlystop_classifies_earlier_selected_round(self) -> None:
        metrics = {
            "piper_synthetic_holdout": {"wer": 34.0, "cer": 13.0, "empty": 0},
            "supertonic_heldout_voice_holdout": {"wer": 14.0, "cer": 4.0, "empty": 0},
            "fleurs_v2": {"wer": 46.5, "cer": 15.7, "empty": 0},
            "artur_j": {"wer": 56.9, "cer": 20.2, "empty": 0},
        }
        decision = classify_artur_earlystop(
            selected_round=8,
            max_round=20,
            controller_rows=[{"available": True}],
            selected_directional_metrics=metrics,
        )
        self.assertEqual(decision["classification"], "ARTUR_EARLYSTOP_FINDS_EFFICIENT_EARLIER_CHECKPOINT")


if __name__ == "__main__":
    unittest.main()
