from __future__ import annotations

import json
import unittest

from slaif_asr.artur_controller_dev import checkpoint_availability, write_curve_reports


class ArturControllerDevCurveTests(unittest.TestCase):
    def test_checkpoint_availability_marks_missing_rounds(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            from pathlib import Path

            rows = checkpoint_availability(Path(tmp))
        self.assertEqual(rows[0]["status"], "BASELINE")
        self.assertIs(rows[0]["available"], True)
        self.assertEqual(rows[1]["status"], "NOT_RUN_CHECKPOINT_UNAVAILABLE")
        self.assertIs(rows[20]["available"], False)


    def test_curve_report_redacts_raw_references_and_hypotheses(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        certificate = {
            "partition_id": "artur-controller-dev-v1",
            "row_count": 2,
            "audio_duration_seconds": {"count": 2, "total": 6.0, "min": 3.0, "median": 3.0, "max": 3.0},
            "manifest_sha256": "a" * 64,
            "normalized_reference_hash_set_sha256": "b" * 64,
            "audio_hash_set_sha256": "c" * 64,
            "normalization_policy": "sl-asr-normalization-v1",
        }
        with TemporaryDirectory() as tmp:
            json_path = Path(tmp) / "report.json"
            md_path = Path(tmp) / "report.md"
            report = write_curve_reports(
                certificate=certificate,
                checkpoint_rows=[
                    {"round": 0, "checkpoint": "base", "available": True, "status": "BASELINE"},
                    {"round": 1, "checkpoint": "round_1", "available": False, "status": "NOT_RUN_CHECKPOINT_UNAVAILABLE"},
                ],
                synthetic_rows=[{"round": 0, "synthetic_anchor_probe": 1.0, "synthetic_scale_probe": 2.0}],
                json_path=json_path,
                md_path=md_path,
            )
            serialized = json.dumps(json.loads(json_path.read_text()), ensure_ascii=False)
        self.assertEqual(report["classification"], "ARTUR_CONTROLLER_DEV_READY_CURVE_BLOCKED_CHECKPOINTS_UNAVAILABLE")
        self.assertNotIn("hypotheses", serialized)
        self.assertNotIn("raw text", serialized)
        self.assertNotIn(".wav", serialized)


if __name__ == "__main__":
    unittest.main()
