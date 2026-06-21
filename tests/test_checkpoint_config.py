from __future__ import annotations

import re
import unittest

from slaif_asr.config import load_runtime_config


class CheckpointConfigTests(unittest.TestCase):
    def test_checkpoint_sha256_is_real_digest_shape(self):
        sha256 = load_runtime_config()["base_model"]["sha256"]
        self.assertRegex(sha256, re.compile(r"^[0-9a-f]{64}$"))
        self.assertNotEqual(sha256, load_runtime_config()["base_model"].get("hf_lfs_etag"))

    def test_checkpoint_byte_size_is_pinned(self):
        self.assertEqual(load_runtime_config()["base_model"]["byte_size"], 2368284501)


if __name__ == "__main__":
    unittest.main()
