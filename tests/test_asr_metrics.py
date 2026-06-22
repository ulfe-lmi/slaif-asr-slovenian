from __future__ import annotations

import unittest

from slaif_asr.metrics import (
    corpus_metric_summary,
    empty_status,
    raw_cer,
    raw_character_edit_counts,
    raw_wer,
    raw_word_edit_counts,
    recognition_change,
)


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

    def test_edit_counts_are_reported_separately(self) -> None:
        counts = raw_word_edit_counts("ena dva tri", "ena štiri")
        self.assertEqual(counts.deletions + counts.substitutions + counts.insertions, raw_wer("ena dva tri", "ena štiri").distance)
        char_counts = raw_character_edit_counts("abc", "adc")
        self.assertEqual(char_counts.substitutions, 1)

    def test_corpus_and_mean_metrics_are_distinct(self) -> None:
        summary = corpus_metric_summary([("ena dva tri štiri", "ena dva tri štiri"), ("pet", "")])
        self.assertEqual(summary.corpus_wer, 20.0)
        self.assertEqual(summary.mean_utterance_wer, 50.0)
        self.assertEqual(summary.empty_hypothesis_count, 1)


if __name__ == "__main__":
    unittest.main()
