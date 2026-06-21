from __future__ import annotations

import argparse
import unittest

from scripts.run_streaming_inference import parse_context
from slaif_asr.config import streaming_contexts


class StreamingCliTests(unittest.TestCase):
    def test_streaming_context_config_contains_required_settings(self):
        self.assertEqual(streaming_contexts(), [(56, 0), (56, 1), (56, 3), (56, 6), (56, 13)])

    def test_parse_context_accepts_supported_context(self):
        self.assertEqual(parse_context("[56,13]"), (56, 13))

    def test_parse_context_rejects_unsupported_context(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_context("[56,2]")


if __name__ == "__main__":
    unittest.main()
