from __future__ import annotations

import unittest

from slaif_asr.artur_controller_dev import select_earliest_within_tolerance, watcher_contract_valid


class ArturControllerDevEarlyStopTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
