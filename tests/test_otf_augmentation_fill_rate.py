from __future__ import annotations

import copy
import unittest

from slaif_asr.otf_augmentation_dataset import (
    classify_fill_rate,
    compute_fill_rate_metrics,
    validate_benchmark_result,
)


class OtfAugmentationFillRateTests(unittest.TestCase):
    def thresholds(self) -> dict:
        return {
            "minimum_fill_rate": 0.98,
            "minimum_ready_microbatch_rate": 0.98,
            "maximum_p95_consumer_wait_ms": 50.0,
        }

    def result(self) -> dict:
        return {
            "benchmark_id": "otf-augmentation-scale2000-fill-rate-v1",
            "classification": "OTF_AUG_FILL_RATE_PASSES_PRIMARY_TARGET",
            "data_source": {
                "corpus_id": "sl-corpus-v4-gams-16000-training-v1",
                "source_clean_files": 144000,
            },
            "determinism": {"status": "PASSED"},
            "pipeline": {"num_workers": 3},
            "primary_target_examples_per_second": 12.8,
            "thresholds": self.thresholds(),
            "rates": [
                {
                    "target_examples_per_second": 12.8,
                    "prepared_examples_per_second": 30.0,
                    "fill_rate": 0.999,
                    "microbatch_ready_rate": 0.999,
                    "p95_consumer_wait_ms": 0.5,
                }
            ],
            "runtime": {"cpu_model": "fixture"},
        }

    def test_fill_rate_metrics_from_synthetic_events(self) -> None:
        metrics = compute_fill_rate_metrics(
            wait_seconds=[0.0, 0.0, 0.002, 0.0],
            queue_fill_levels=[4, 3, 2, 1],
            target_examples_per_second=10.0,
            physical_microbatch=2,
            ready_wait_threshold_ms=1.0,
            ready_flags=[True, True, False, True],
        )
        self.assertAlmostEqual(metrics["fill_rate"], 1.0 - 0.002 / 0.802)
        self.assertEqual(metrics["microbatch_ready_rate"], 0.75)
        self.assertEqual(metrics["underrun_count"], 1)
        self.assertEqual(metrics["queue_fill"]["mean"], 2.5)

    def test_passing_primary_classification(self) -> None:
        payload = self.result()
        self.assertEqual(
            classify_fill_rate(payload, self.thresholds()),
            "OTF_AUG_FILL_RATE_PASSES_PRIMARY_TARGET",
        )

    def test_marginal_primary_classification(self) -> None:
        payload = self.result()
        payload["rates"][0]["p95_consumer_wait_ms"] = 70.0
        self.assertEqual(
            classify_fill_rate(payload, self.thresholds()),
            "OTF_AUG_FILL_RATE_MARGINAL_PRIMARY_TARGET",
        )

    def test_throughput_limited_primary_classification(self) -> None:
        payload = self.result()
        payload["rates"][0].update(
            {
                "prepared_examples_per_second": 8.0,
                "fill_rate": 0.8,
                "microbatch_ready_rate": 0.5,
                "p95_consumer_wait_ms": 100.0,
            }
        )
        self.assertEqual(
            classify_fill_rate(payload, self.thresholds()),
            "OTF_AUG_FILL_RATE_FAILS_PRIMARY_TARGET",
        )

    def test_determinism_failure_has_priority(self) -> None:
        payload = self.result()
        payload["determinism"]["status"] = "FAILED"
        self.assertEqual(
            classify_fill_rate(payload, self.thresholds()),
            "OTF_AUG_DETERMINISM_INVALID",
        )

    def test_result_schema_accepts_privacy_safe_result(self) -> None:
        validate_benchmark_result(self.result())

    def test_result_schema_enforces_three_workers(self) -> None:
        payload = self.result()
        payload["pipeline"]["num_workers"] = 2
        with self.assertRaisesRegex(ValueError, "exactly three workers"):
            validate_benchmark_result(payload)

    def test_result_schema_rejects_raw_fields_and_local_paths(self) -> None:
        raw = copy.deepcopy(self.result())
        raw["transcript"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "forbidden field"):
            validate_benchmark_result(raw)
        local = copy.deepcopy(self.result())
        local["runtime"]["storage"] = "/data/private"
        with self.assertRaisesRegex(ValueError, "absolute path"):
            validate_benchmark_result(local)


if __name__ == "__main__":
    unittest.main()
