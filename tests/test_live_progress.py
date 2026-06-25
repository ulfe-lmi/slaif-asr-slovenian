from __future__ import annotations

import io
import time
import tempfile
import unittest
from pathlib import Path

from slaif_asr.live_progress import LiveProgressReporter, format_progress_line, heartbeat_thread, run_streaming_child, sanitize_event


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class LiveProgressTests(unittest.TestCase):
    def test_stage_events_flush_to_stderr_style_stream(self) -> None:
        stream = io.StringIO()
        clock = FakeClock()
        reporter = LiveProgressReporter(stage="train", arm="joint_adapter", stream=stream, clock=clock)
        reporter.start("loading")
        clock.advance(5.0)
        reporter.progress(step=5, total_steps=20, current_loss=12.5, rolling_mean_loss=13.0)
        reporter.complete("done")
        output = stream.getvalue()
        self.assertIn("event=stage_start", output)
        self.assertIn("event=progress", output)
        self.assertIn("percent_complete=25.0", output)
        self.assertIn("event=stage_complete", output)

    def test_progress_ndjson_and_completed_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_text:
            path = Path(tmp_text) / "progress.ndjson"
            reporter = LiveProgressReporter(stage="evaluate", split="fleurs_v2", stream=io.StringIO(), ndjson_path=path)
            reporter.start()
            reporter.complete(processed_rows=25, total_rows=25)
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".ndjson.completed").exists())
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_sanitizer_blocks_raw_ids_text_and_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            sanitize_event({"event": "progress", "sample_id": "x"})
        with self.assertRaisesRegex(ValueError, "raw IDs"):
            sanitize_event({"event": "progress", "message": "gamsv2-cell01-a00-o001"})
        with self.assertRaisesRegex(ValueError, "local paths"):
            sanitize_event({"event": "progress", "message": "/home/example/file"})

    def test_format_has_core_fields(self) -> None:
        line = format_progress_line({"timestamp": "2026-06-25T00:00:00Z", "event": "heartbeat", "stage": "restore"})
        self.assertIn("event=heartbeat", line)
        self.assertIn("stage=restore", line)

    def test_failure_event_emission(self) -> None:
        stream = io.StringIO()
        reporter = LiveProgressReporter(stage="train", stream=stream)
        reporter.failed("boom")
        self.assertIn("event=stage_failed", stream.getvalue())

    def test_heartbeat_context_emits_periodically(self) -> None:
        stream = io.StringIO()
        reporter = LiveProgressReporter(stage="restore", stream=stream)
        with heartbeat_thread(reporter, interval_seconds=0.01, message="loading"):
            time.sleep(0.03)
        self.assertIn("event=heartbeat", stream.getvalue())

    def test_child_output_streams_immediately(self) -> None:
        stream = io.StringIO()
        code = "import sys; print('hello', flush=True)"
        status = run_streaming_child(["python3", "-c", code], stream=stream)
        self.assertEqual(status, 0)
        self.assertIn("hello", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
