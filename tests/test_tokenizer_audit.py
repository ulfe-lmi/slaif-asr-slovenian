from __future__ import annotations

import unittest

from slaif_asr.tokenizer_audit import SLOVENIAN_AUDIT_TEXTS, audit_tokenizer


class RoundTripTokenizer:
    def __init__(self):
        self.store = {}

    def text_to_ids(self, text):
        ids = [ord(char) for char in text]
        self.store[tuple(ids)] = text
        return ids

    def ids_to_text(self, ids):
        return self.store[tuple(ids)]


class LowercaseTokenizer(RoundTripTokenizer):
    def ids_to_text(self, ids):
        return super().ids_to_text(ids).lower()


class TokenizerAuditTests(unittest.TestCase):
    def test_tokenizer_audit_passes_exact_round_trip(self):
        report = audit_tokenizer(RoundTripTokenizer())

        self.assertTrue(report.passed)
        self.assertEqual(len(report.records), len(SLOVENIAN_AUDIT_TEXTS))
        self.assertEqual(report.records[2].decoded_text, "Čez cesto švigne žaba.")

    def test_tokenizer_audit_fails_when_case_changes(self):
        report = audit_tokenizer(LowercaseTokenizer(), texts=["ABCČŠŽ"])

        self.assertFalse(report.passed)
        self.assertEqual(report.records[0].decoded_text, "abcčšž")


if __name__ == "__main__":
    unittest.main()
