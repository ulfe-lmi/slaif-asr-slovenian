from __future__ import annotations

import unittest
from pathlib import Path

from slaif_asr.gams_retry_controller import AttemptTask
from slaif_asr.scale8000_corpus import (
    build_dual_gpu_generation_plan,
    build_new_record,
    load_scale8000_generation_config,
    scale8000_multiplier_table,
    verify_scale2000_prefix,
)


CONFIG_PATH = Path("configs/generation/gams_corpus_v5_scale8000_v1.json")


class Scale8000CorpusTests(unittest.TestCase):
    def test_config_counts_and_unbounded_retry(self) -> None:
        config = load_scale8000_generation_config(CONFIG_PATH)
        self.assertEqual(config["final_rows"], 64000)
        self.assertEqual(config["new_rows"], 48000)
        self.assertEqual(config["combined_rows_per_cell"], 1600)
        self.assertEqual(config["inherited_rows_per_cell"], 400)
        self.assertEqual(config["new_rows_per_cell"], 1200)
        self.assertTrue(config["retry_policy"]["retry_until_valid"])
        self.assertIsNone(config["retry_policy"]["max_total_attempts"])

    def test_build_new_record_uses_v5_identity_and_keeps_id_out_of_text(self) -> None:
        config = load_scale8000_generation_config(CONFIG_PATH)
        cell = {"cell_id": "cell01", "source_family_id": "sf-cell01", "domain": "fixture", "phenomena": ["dual"]}
        task = AttemptTask("cell01", "shard03", 2, 1, 60, 123, "targeted_refill")
        row = build_new_record(config, cell, task, 7, "Danes jasno povem kratko poved.")
        self.assertTrue(row["candidate_id"].startswith("gamsv5-cell01-shard03-a02-o007"))
        self.assertEqual(row["partition_role"], "selected_training")
        self.assertEqual(row["spoken_text"], row["target_text"])
        self.assertNotIn(row["candidate_id"], row["spoken_text"])

    def test_dual_gpu_shard_plan_is_balanced_and_single_visible_device(self) -> None:
        config = load_scale8000_generation_config(CONFIG_PATH)
        plan = build_dual_gpu_generation_plan(config)
        self.assertEqual(plan["total_tasks"], 1200)
        self.assertEqual(plan["total_requested_rows"], 72000)
        self.assertEqual(plan["workers"]["gpu0"]["task_count"], 600)
        self.assertEqual(plan["workers"]["gpu1"]["task_count"], 600)
        self.assertEqual(plan["workers"]["gpu0"]["cuda_visible_devices"], "0")
        self.assertEqual(plan["workers"]["gpu1"]["cuda_visible_devices"], "1")
        self.assertEqual(plan["workers"]["gpu0"]["logical_device"], "cuda:0")
        self.assertEqual(plan["workers"]["gpu1"]["logical_device"], "cuda:0")

    def test_scale2000_prefix_verification_rejects_mutation(self) -> None:
        inherited = [{"candidate_id": "a", "target_text": "Prva."}, {"candidate_id": "b", "target_text": "Druga."}]
        combined = [*inherited, {"candidate_id": "c", "target_text": "Tretja."}]
        result = verify_scale2000_prefix(combined, inherited)
        self.assertTrue(result["prefix_preserved"])
        mutated = [{"candidate_id": "a", "target_text": "Spremenjena."}, inherited[1]]
        with self.assertRaisesRegex(ValueError, "prefix mutated"):
            verify_scale2000_prefix(mutated, inherited)

    def test_multiplier_table_names_exposure_scale_not_independent_information(self) -> None:
        table = scale8000_multiplier_table()
        self.assertEqual(table["semantic_rows"], 64000)
        self.assertEqual(table["clean_files"], 576000)
        self.assertEqual(table["augmented_files"], 704000)
        self.assertEqual(table["total_views"], 1280000)
        self.assertEqual(table["optimizer_steps_at_batch8"], 160000)
        self.assertEqual(table["exposure_multiplier_vs_reference"], 8000)
        self.assertIn("not independent linguistic information", table["interpretation"])


if __name__ == "__main__":
    unittest.main()
