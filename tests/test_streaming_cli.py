from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_streaming_inference import parse_context
from slaif_asr.config import streaming_contexts
from slaif_asr.inference import parse_final_transcript, resolve_existing_path, run_context


class StreamingCliTests(unittest.TestCase):
    def test_streaming_context_config_contains_required_settings(self):
        self.assertEqual(streaming_contexts(), [(56, 0), (56, 1), (56, 3), (56, 6), (56, 13)])

    def test_parse_context_accepts_supported_context(self):
        self.assertEqual(parse_context("[56,13]"), (56, 13))

    def test_parse_context_rejects_unsupported_context(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_context("[56,2]")

    def test_resolve_existing_path_returns_absolute_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio.wav"
            path.write_bytes(b"RIFF")
            self.assertTrue(resolve_existing_path(path, "audio").is_absolute())

    def test_parse_final_transcript_from_nemo_log(self):
        log = "INFO Final streaming transcriptions: ['dober dan']\n"
        self.assertEqual(parse_final_transcript(log), "dober dan")

    def test_run_context_persists_log_and_result_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            context_dir = Path(tmp) / "context_56_3"
            command = [
                sys.executable,
                "-c",
                "print(\"INFO Final streaming transcriptions: ['živjo']\")",
            ]

            result = run_context(
                command=command,
                context=(56, 3),
                context_dir=context_dir,
                checkpoint_sha256="210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74",
                cuda_index=None,
            )

            self.assertEqual(result.exit_status, 0)
            self.assertTrue(result.log_path.exists())
            self.assertTrue(result.result_path.exists())
            payload = json.loads(result.result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["transcript"], "živjo")
            self.assertEqual(payload["att_context_size"], [56, 3])
            self.assertIsNone(payload["reference_text"])
            self.assertIsNone(payload["wer"])


if __name__ == "__main__":
    unittest.main()
