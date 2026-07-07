import unittest

from slaif_asr.rtx2080ti_policy import parse_nvidia_smi_csv, select_microbatch


class Rtx2080TiPolicyTests(unittest.TestCase):
    def test_parse_inventory(self):
        rows = parse_nvidia_smi_csv(
            "0, NVIDIA GeForce RTX 2080 Ti, 11264, 12, 0\n"
            "1, NVIDIA A100-SXM4-80GB, 81920, 0, 0\n"
        )
        self.assertEqual(rows[0].index, 0)
        self.assertEqual(rows[0].name, "NVIDIA GeForce RTX 2080 Ti")
        self.assertEqual(rows[0].memory_total_mib, 11264)
        self.assertIn("A100", rows[1].name)

    def test_microbatch_selection_order_and_accumulation(self):
        selected = select_microbatch(
            [8, 4, 2, 1],
            {
                8: {"status": "FAILED"},
                4: {"status": "PASSED"},
                2: {"status": "PASSED"},
                1: {"status": "PASSED"},
            },
        )
        self.assertEqual(selected["physical_microbatch"], 4)
        self.assertEqual(selected["gradient_accumulation_steps"], 2)
        self.assertEqual(selected["effective_batch_size"], 8)

    def test_physical_batch_one_failure_blocks(self):
        selected = select_microbatch(
            [8, 4, 2, 1],
            {
                8: {"status": "FAILED"},
                4: {"status": "FAILED"},
                2: {"status": "FAILED"},
                1: {"status": "FAILED"},
            },
        )
        self.assertEqual(selected["status"], "ENVIRONMENT_BLOCKED")


if __name__ == "__main__":
    unittest.main()
