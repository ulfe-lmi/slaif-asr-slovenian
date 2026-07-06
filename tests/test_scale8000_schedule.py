from __future__ import annotations

import unittest

from slaif_asr.scale8000_corpus import estimate_scale8000_storage, scale8000_plan, storage_preflight


class Scale8000ScheduleTests(unittest.TestCase):
    def test_exact_scale_accounting(self) -> None:
        plan = scale8000_plan()
        self.assertEqual(plan.semantic_rows, 64000)
        self.assertEqual(plan.inherited_rows, 16000)
        self.assertEqual(plan.new_rows, 48000)
        self.assertEqual(plan.clean_files, 576000)
        self.assertEqual(plan.augmented_files, 704000)
        self.assertEqual(plan.total_views, 1280000)
        self.assertEqual(plan.optimizer_steps_at_batch8, 160000)
        self.assertEqual(plan.exposure_multiplier_vs_reference, 8000)

    def test_storage_estimate_requires_three_more_scale2000_units_plus_margin(self) -> None:
        one_scale2000 = 100
        result = estimate_scale8000_storage(
            inherited_scale2000_bytes=one_scale2000,
            available_bytes=374,
            safety_margin_fraction=0.25,
        )
        self.assertEqual(result["projected_new_bytes"], 300)
        self.assertEqual(result["required_free_bytes"], 375)
        self.assertFalse(result["sufficient"])
        sufficient = estimate_scale8000_storage(
            inherited_scale2000_bytes=one_scale2000,
            available_bytes=375,
            safety_margin_fraction=0.25,
        )
        self.assertTrue(sufficient["sufficient"])

    def test_storage_preflight_uses_recorded_inherited_size(self) -> None:
        config = {
            "storage_preflight": {
                "inherited_scale2000_bytes": 100,
                "measurement_source": "fixture-recorded-size",
                "runtime_storage": "fixture-storage",
            }
        }
        result = storage_preflight(config)
        self.assertEqual(result["inherited_scale2000_bytes"], 100)
        self.assertEqual(result["projected_new_bytes"], 300)
        self.assertEqual(result["measurement_source"], "fixture-recorded-size")
        self.assertEqual(result["runtime_storage"], "fixture-storage")


if __name__ == "__main__":
    unittest.main()
