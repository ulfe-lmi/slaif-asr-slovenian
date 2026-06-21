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


class ExtendedSymbolUnknownTokenizer(RoundTripTokenizer):
    def text_to_ids(self, text):
        normalized = text.replace("€", "<unk>").replace("°", "<unk>")
        ids = [ord(char) for char in normalized]
        self.store[tuple(ids)] = normalized
        return ids


class TokenizerAuditTests(unittest.TestCase):
    def test_tokenizer_audit_passes_exact_round_trip(self):
        report = audit_tokenizer(RoundTripTokenizer())

        self.assertTrue(report.required_slovenian_passed)
        self.assertTrue(report.all_samples_passed)
        self.assertEqual(len(report.records), len(SLOVENIAN_AUDIT_TEXTS))
        self.assertEqual(report.records[2].decoded_text, "Čez cesto švigne žaba.")

    def test_tokenizer_audit_fails_when_case_changes(self):
        report = audit_tokenizer(LowercaseTokenizer(), texts=["ABCČŠŽ"])

        self.assertFalse(report.required_slovenian_passed)
        self.assertEqual(report.records[0].decoded_text, "abcčšž")

    def test_extended_symbol_failure_is_warning_not_required_failure(self):
        report = audit_tokenizer(ExtendedSymbolUnknownTokenizer())

        self.assertTrue(report.required_slovenian_passed)
        self.assertFalse(report.all_samples_passed)
        self.assertIn("extended_symbols: decoded output differs from input", report.warnings)


if __name__ == "__main__":
    unittest.main()
