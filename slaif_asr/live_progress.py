from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TextIO


EVENT_TYPES = {"stage_start", "heartbeat", "progress", "stage_complete", "stage_failed"}
FORBIDDEN_KEYS = {
    "audio_filepath",
    "candidate_id",
    "candidate_ids",
    "holdout_id",
    "holdout_ids",
    "hypothesis",
    "hypotheses",
    "reference",
    "references",
    "sample_id",
    "sample_ids",
    "selected_training_id",
    "text",
}
FORBIDDEN_MARKERS = ("gamsv2-", "gams9holdout-", "/" + "home" + "/", "/" + "mnt" + "/", "/" + "tmp" + "/")


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def sanitize_event(event: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in event.items():
        if key in FORBIDDEN_KEYS:
            raise ValueError(f"progress event contains forbidden key: {key}")
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    serialized = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    if any(marker in serialized for marker in FORBIDDEN_MARKERS):
        raise ValueError("progress event contains raw IDs or local paths")
    return sanitized


def format_progress_line(event: dict[str, Any]) -> str:
    parts = [str(event.get("timestamp", utc_timestamp()))]
    ordered = [
        "event",
        "stage",
        "arm",
        "split",
        "epoch",
        "total_epochs",
        "step",
        "total_steps",
        "processed_rows",
        "total_rows",
        "percent_complete",
        "current_loss",
        "rolling_mean_loss",
        "elapsed_seconds",
        "estimated_remaining_seconds",
        "examples_per_second",
        "audio_seconds_per_wall_second",
        "cuda_alloc_mib",
        "cuda_reserved_mib",
        "message",
    ]
    for key in ordered:
        if key not in event:
            continue
        value = event[key]
        if isinstance(value, float):
            value = round(value, 6)
        parts.append(f"{key}={value}")
    return " ".join(parts)


@dataclass
class LiveProgressReporter:
    stage: str
    arm: str | None = None
    split: str | None = None
    stream: TextIO = sys.stderr
    ndjson_path: Path | None = None
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        self._start = self.clock()
        self._last_progress = 0.0
        self._completed = False
        if self.ndjson_path is not None:
            self.ndjson_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported progress event type: {event_type}")
        event = {
            "timestamp": utc_timestamp(),
            "event": event_type,
            "stage": self.stage,
            "arm": self.arm,
            "split": self.split,
            "elapsed_seconds": round(self.clock() - self._start, 6),
        }
        event.update(fields)
        event = sanitize_event(event)
        print(format_progress_line(event), file=self.stream, flush=True)
        if self.ndjson_path is not None:
            with self.ndjson_path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                fp.flush()
        self._last_progress = self.clock()
        return event

    def start(self, message: str | None = None) -> dict[str, Any]:
        return self.emit("stage_start", message=message)

    def heartbeat(self, message: str | None = None, **fields: Any) -> dict[str, Any]:
        return self.emit("heartbeat", message=message, **fields)

    def progress(
        self,
        *,
        step: int | None = None,
        total_steps: int | None = None,
        processed_rows: int | None = None,
        total_rows: int | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        percent = None
        if step is not None and total_steps:
            percent = float(step) / float(total_steps) * 100.0
        elif processed_rows is not None and total_rows:
            percent = float(processed_rows) / float(total_rows) * 100.0
        eta = None
        elapsed = self.clock() - self._start
        if percent and percent > 0:
            eta = elapsed * (100.0 - percent) / percent
        return self.emit(
            "progress",
            step=step,
            total_steps=total_steps,
            processed_rows=processed_rows,
            total_rows=total_rows,
            percent_complete=round(percent, 6) if percent is not None else None,
            estimated_remaining_seconds=round(eta, 6) if eta is not None else None,
            **fields,
        )

    def complete(self, message: str | None = None, **fields: Any) -> dict[str, Any]:
        self._completed = True
        completed = self.emit("stage_complete", message=message, **fields)
        if self.ndjson_path is not None:
            marker = self.ndjson_path.with_suffix(self.ndjson_path.suffix + ".completed")
            marker.write_text(json.dumps({"completed": True, "timestamp": utc_timestamp()}) + "\n", encoding="utf-8")
        return completed

    def failed(self, message: str | None = None, **fields: Any) -> dict[str, Any]:
        return self.emit("stage_failed", message=message, **fields)


@contextmanager
def heartbeat_thread(
    reporter: LiveProgressReporter,
    *,
    interval_seconds: float,
    message: str,
    fields: Callable[[], dict[str, Any]] | None = None,
) -> Iterator[None]:
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(interval_seconds):
            payload = fields() if fields is not None else {}
            reporter.heartbeat(message=message, **payload)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval_seconds * 2.0))


def run_streaming_child(
    command: Sequence[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    log_path: Path | None = None,
    stream: TextIO = sys.stderr,
) -> int:
    child_env = dict(env or {})
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    if command and command[0].endswith("python") and "-u" not in command:
        command = [command[0], "-u", *command[1:]]
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    log_fp = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("w", encoding="utf-8")
    try:
        for line in process.stdout:
            print(line, end="", file=stream, flush=True)
            if log_fp is not None:
                log_fp.write(line)
                log_fp.flush()
        status = process.wait()
        process.stdout.close()
        return status
    finally:
        if log_fp is not None:
            log_fp.close()
