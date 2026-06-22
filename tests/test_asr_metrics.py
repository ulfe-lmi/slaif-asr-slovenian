from __future__ import annotations

import unittest

from slaif_asr.metrics import empty_status, raw_cer, raw_wer, recognition_change


class AsrMetricsTests(unittest.TestCase):
    def test_empty_hypothesis_has_explicit_status(self) -> None:
        self.assertEqual(empty_status(""), "EMPTY_HYPOTHESIS")
        self.assertEqual(empty_status("besedilo"), "NONEMPTY")

    def test_wer_and_cer_handle_empty_hypothesis(self) -> None:
        self.assertEqual(raw_wer("Kratek stavek.", "").percent, 100.0)
        self.assertGreater(raw_cer("abc", "").percent, 0.0)

    def test_recognition_change_is_distinct_from_pipeline_status(self) -> None:
        self.assertEqual(recognition_change("a b", "", "a b"), "EXACT_MATCH")
        self.assertEqual(recognition_change("a b", "", "a"), "IMPROVED")
        self.assertEqual(recognition_change("a b", "a", ""), "EMPTY_HYPOTHESIS")
        self.assertEqual(recognition_change("a b", "a b", "x y z"), "REGRESSED")


if __name__ == "__main__":
    unittest.main()
