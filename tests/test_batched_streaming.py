from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from slaif_asr.batched_streaming import (
    StreamingRecord,
    compare_predictions,
    make_batches,
    parse_monitor_csv,
    parse_sentinel_predictions,
    privacy_safe_arm_summary,
    scientific_classification,
    select_batch_policy,
    select_hash_subset,
    should_run_batch_256,
    stable_sha256,
    write_nemo_manifest,
)
from slaif_asr.real_eval import summarize_predictions


def record(sample_id: str, duration: float, index: int) -> StreamingRecord:
    return StreamingRecord(
        sample_id=sample_id,
        audio_filepath=f"/tmp/{sample_id}.wav",
        duration=duration,
        reference=f"referenca {sample_id}",
        original_index=index,
        row={},
    )


class BatchedStreamingTests(unittest.TestCase):
    def test_duration_bucketing_and_final_partial_batch(self) -> None:
        records = [record("c", 3.0, 0), record("a", 1.0, 1), record("b", 2.0, 2)]
        layout = make_batches(records, batch_size=2, bucketed=True)
        self.assertEqual([[item.sample_id for item in batch] for batch in layout.batches], [["a", "b"], ["c"]])
        self.assertEqual(layout.final_partial_batch_size, 1)
        self.assertEqual(layout.batch_count, 2)
        self.assertEqual(layout.padded_audio_seconds, 7.0)
        self.assertEqual(layout.actual_audio_seconds, 6.0)

    def test_unbucketed_preserves_source_order(self) -> None:
        records = [record("c", 3.0, 0), record("a", 1.0, 1), record("b", 2.0, 2)]
        layout = make_batches(records, batch_size=2, bucketed=False)
        self.assertEqual([item.sample_id for item in layout.ordered_records], ["c", "a", "b"])

    def test_hash_subset_is_deterministic(self) -> None:
        records = [record(f"id-{index}", float(index + 1), index) for index in range(10)]
        subset = select_hash_subset(records, 3)
        expected = sorted(records, key=lambda item: (stable_sha256(item.sample_id), item.sample_id))[:3]
        self.assertEqual({item.sample_id for item in subset}, {item.sample_id for item in expected})
        self.assertEqual([item.original_index for item in subset], sorted(item.original_index for item in subset))

    def test_sentinel_manifest_and_prediction_parsing(self) -> None:
        records = [record("sample-a", 1.0, 0), record("sample-b", 2.0, 1)]
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.jsonl"
            layout = make_batches(records, batch_size=2, bucketed=False)
            write_nemo_manifest(manifest, layout)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows[0]["text"].startswith("__SLAIF_SAMPLE_ID__:sample-a\t"))
            output = Path(temp) / "streaming.json"
            output.write_text(
                "\n".join(
                    [
                        json.dumps({"text": rows[0]["text"], "pred_text": "ena"}),
                        json.dumps({"text": rows[1]["text"], "pred_text": "dve"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_sentinel_predictions(output), {"sample-a": "ena", "sample-b": "dve"})

    def test_prediction_parity_comparison(self) -> None:
        records = [record("a", 1.0, 0), record("b", 1.0, 1)]
        baseline = {"a": "Dober dan.", "b": "Živjo"}
        candidate = {"a": "Dober dan", "b": "Živjo"}
        base_metrics = summarize_predictions(
            [{"reference": item.reference, "hypothesis": baseline[item.sample_id]} for item in records]
        )
        cand_metrics = summarize_predictions(
            [{"reference": item.reference, "hypothesis": candidate[item.sample_id]} for item in records]
        )
        comparison = compare_predictions(records, baseline, candidate, baseline_metrics=base_metrics, candidate_metrics=cand_metrics)
        self.assertEqual(comparison.exact_mismatch_count, 1)
        self.assertEqual(comparison.normalized_mismatch_count, 0)
        self.assertFalse(comparison.exact_parity)

    def test_select_batch_policy_uses_smallest_within_best(self) -> None:
        arms = [
            {"batch_size": 1, "bucketed": True, "parity_eligible": True, "end_to_end_audio_seconds_per_wall_second": 10.0},
            {"batch_size": 8, "bucketed": True, "parity_eligible": True, "end_to_end_audio_seconds_per_wall_second": 19.2},
            {"batch_size": 16, "bucketed": True, "parity_eligible": True, "end_to_end_audio_seconds_per_wall_second": 20.0},
        ]
        selected = select_batch_policy(arms, within_best_fraction=0.95)
        self.assertEqual(selected["batch_size"], 8)

    def test_conditional_batch_256_policy(self) -> None:
        b64 = {
            "status": "PASSED",
            "parity_eligible": True,
            "execution": {"monitor": {"peak_memory_mib": 1000}},
            "end_to_end_audio_seconds_per_wall_second": 20.0,
        }
        b128 = {
            "status": "PASSED",
            "parity_eligible": True,
            "execution": {"monitor": {"peak_memory_mib": 50000}},
            "end_to_end_audio_seconds_per_wall_second": 22.0,
        }
        self.assertTrue(should_run_batch_256(b128, b64, max_memory_mib=57344, min_gain=1.05))
        b128["execution"]["monitor"]["peak_memory_mib"] = 60000
        self.assertFalse(should_run_batch_256(b128, b64, max_memory_mib=57344, min_gain=1.05))

    def test_gpu_monitor_csv_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "monitor.csv"
            path.write_text(
                "timestamp,index,name,memory_used_mib,utilization_percent,power_watts\n"
                "1,1,A100,100,0,80\n"
                "2,1,A100,200,90,100\n",
                encoding="utf-8",
            )
            summary = parse_monitor_csv(path)
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["peak_memory_mib"], 200)
            self.assertEqual(summary["fraction_at_or_above_80_percent"], 0.5)

    def test_classifications(self) -> None:
        self.assertEqual(
            scientific_classification(selected_batch=8, exact_parity_above_one=True, selected_speedup=1.3),
            "A100_BATCHED_STREAMING_SUPPORTED",
        )
        self.assertEqual(
            scientific_classification(selected_batch=1, exact_parity_above_one=False, selected_speedup=1.0),
            "A100_BATCHED_STREAMING_NOT_EQUIVALENT",
        )

    def test_privacy_safe_arm_summary_excludes_paths(self) -> None:
        summary = privacy_safe_arm_summary(
            {
                "batch_size": 1,
                "bucketed": True,
                "status": "PASSED",
                "output_path": "/home/user/private.json",
                "layout": {"batch_count": 1, "full_batch_count": 1, "final_partial_batch_size": 0, "actual_audio_seconds": 1, "padded_audio_seconds": 1, "padding_ratio": 1, "max_padded_batch_duration": 1},
                "execution": {"exit_status": 0, "wall_time_seconds": 1, "active_wall_time_seconds": 1, "monitor": {}},
            }
        )
        serialized = json.dumps(summary)
        self.assertNotIn("/home/user", serialized)


if __name__ == "__main__":
    unittest.main()
