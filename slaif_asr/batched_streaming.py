from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.metrics import CorpusMetricSummary
from slaif_asr.real_eval import NORMALIZER_VERSION, atomic_write_json, atomic_write_jsonl, normalize_sl_asr_text, sha256_file, summarize_predictions
from slaif_asr.tts import validate_wav


SENTINEL_PREFIX = "__SLAIF_SAMPLE_ID__:"
ACTIVE_TIME_PATTERN = re.compile(r"The whole streaming process took:\s*([0-9.]+)s")
OOM_PATTERNS = (
    "CUDA out of memory",
    "out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
)
RAW_CHILD_STREAM_MARKERS = (
    SENTINEL_PREFIX,
    "gamsv2-",
    "gams9holdout-",
    "fleurs-sl-si-test-occ-",
    "artur-j-public-",
    "Final streaming transcriptions",
    "Added this sample",
    "audio_filepath",
    "dataset_manifest",
    "model_path",
    "output_path",
    "pred_text",
    "reference",
    "hypothesis",
    "/home/",
    "/mnt/",
    "/tmp/",
    ".wav",
    ".jsonl",
    ".nemo",
)
SAFE_CHILD_STREAM_PREFIXES = (
    "[NeMo I",
    "[NeMo W",
    "The whole streaming process took:",
    "WER% of streaming mode",
)


@dataclass(frozen=True)
class StreamingRecord:
    sample_id: str
    audio_filepath: str
    duration: float
    reference: str
    original_index: int
    row: dict[str, Any]


@dataclass(frozen=True)
class BatchLayout:
    batch_size: int
    bucketed: bool
    batches: list[list[StreamingRecord]]
    ordered_records: list[StreamingRecord]
    actual_audio_seconds: float
    padded_audio_seconds: float
    padding_ratio: float
    batch_count: int
    full_batch_count: int
    final_partial_batch_size: int
    max_padded_batch_duration: float


@dataclass(frozen=True)
class PredictionComparison:
    exact_mismatch_count: int
    normalized_mismatch_count: int
    missing_ids: list[str]
    duplicate_ids: list[str]
    unexpected_ids: list[str]
    metric_differences: dict[str, Any]
    empty_hypothesis_difference: int

    @property
    def exact_parity(self) -> bool:
        return (
            self.exact_mismatch_count == 0
            and self.normalized_mismatch_count == 0
            and not self.missing_ids
            and not self.duplicate_ids
            and not self.unexpected_ids
            and not self.metric_differences
            and self.empty_hypothesis_difference == 0
        )


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gate_records(path: Path, *, expected_sha256: str, expected_rows: int, gate_id: str) -> list[StreamingRecord]:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise ValueError(f"{gate_id}: manifest SHA256 mismatch: {actual_sha} != {expected_sha256}")
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"{gate_id}: expected {expected_rows} rows, found {len(rows)}")
    sample_ids: set[str] = set()
    audio_paths: set[str] = set()
    records = []
    for index, row in enumerate(rows):
        if row.get("dataset") != gate_id:
            raise ValueError(f"{gate_id}: row {index} has dataset {row.get('dataset')}")
        if row.get("partition_role") != "immutable_real_gate":
            raise ValueError(f"{gate_id}: row {index} is not an immutable real gate")
        if row.get("source_type") != "public_real":
            raise ValueError(f"{gate_id}: row {index} is not public real source")
        if row.get("target_lang") != "sl-SI":
            raise ValueError(f"{gate_id}: row {index} target_lang must be sl-SI")
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"{gate_id}: duplicate sample ID {sample_id}")
        sample_ids.add(sample_id)
        audio_path = resolve_manifest_audio_path(path, str(row["audio_filepath"]))
        normalized_audio_path = audio_path.as_posix()
        if normalized_audio_path in audio_paths:
            raise ValueError(f"{gate_id}: duplicate resolved audio path {normalized_audio_path}")
        audio_paths.add(normalized_audio_path)
        validate_wav(audio_path, sample_rate=16000)
        records.append(
            StreamingRecord(
                sample_id=sample_id,
                audio_filepath=str(audio_path),
                duration=float(row["duration"]),
                reference=str(row["text"]),
                original_index=index,
                row=row,
            )
        )
    return records


def resolve_manifest_audio_path(manifest: Path, audio_filepath: str) -> Path:
    path = Path(audio_filepath).expanduser()
    if path.is_absolute():
        runs_root = os.environ.get("SLAIF_ASR_RUNS_ROOT")
        if runs_root and "runs" in path.parts:
            suffix = Path(*path.parts[path.parts.index("runs") + 1 :])
            relocated_root = Path(runs_root).expanduser().resolve() / suffix
            if relocated_root.exists():
                return relocated_root.resolve()
        relocated = manifest.parent / "audio" / path.name
        if relocated.exists():
            return relocated.resolve()
    if path.exists():
        return path.resolve()
    relative = (manifest.parent / path).resolve()
    if relative.exists():
        return relative
    raise FileNotFoundError(f"audio file not found: {audio_filepath}")


def select_hash_subset(records: Sequence[StreamingRecord], count: int) -> list[StreamingRecord]:
    if count > len(records):
        raise ValueError(f"cannot select {count} rows from {len(records)} records")
    selected = sorted(records, key=lambda item: (stable_sha256(item.sample_id), item.sample_id))[:count]
    return sorted(selected, key=lambda item: item.original_index)


def make_batches(records: Sequence[StreamingRecord], *, batch_size: int, bucketed: bool) -> BatchLayout:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ordered = list(records)
    if bucketed:
        ordered.sort(key=lambda item: (item.duration, item.sample_id))
    batches = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
    actual = sum(item.duration for item in ordered)
    padded = 0.0
    max_padded_duration = 0.0
    for batch in batches:
        if not batch:
            continue
        max_duration = max(item.duration for item in batch)
        padded += max_duration * len(batch)
        max_padded_duration = max(max_padded_duration, max_duration)
    full_batches = sum(1 for batch in batches if len(batch) == batch_size)
    final_partial = 0 if not batches else (len(batches[-1]) if len(batches[-1]) != batch_size else 0)
    return BatchLayout(
        batch_size=batch_size,
        bucketed=bucketed,
        batches=batches,
        ordered_records=[item for batch in batches for item in batch],
        actual_audio_seconds=round(actual, 6),
        padded_audio_seconds=round(padded, 6),
        padding_ratio=round(padded / actual, 6) if actual else 0.0,
        batch_count=len(batches),
        full_batch_count=full_batches,
        final_partial_batch_size=final_partial,
        max_padded_batch_duration=round(max_padded_duration, 6),
    )


def sentinel_text(sample_id: str, reference: str) -> str:
    return f"{SENTINEL_PREFIX}{sample_id}\t{reference}"


def parse_sentinel_text(text: str) -> tuple[str, str]:
    if not text.startswith(SENTINEL_PREFIX):
        raise ValueError("missing sample-id sentinel in NeMo output")
    sample_id, _, reference = text[len(SENTINEL_PREFIX) :].partition("\t")
    if not sample_id or not _:
        raise ValueError("malformed sample-id sentinel in NeMo output")
    return sample_id, reference


def write_nemo_manifest(path: Path, layout: BatchLayout, *, with_sentinel: bool = True) -> str:
    rows = []
    for record in layout.ordered_records:
        text = sentinel_text(record.sample_id, record.reference) if with_sentinel else record.reference
        rows.append(
            {
                "audio_filepath": record.audio_filepath,
                "duration": record.duration,
                "text": text,
                "lang": "sl-SI",
                "target_lang": "sl-SI",
            }
        )
    atomic_write_jsonl(path, rows)
    return sha256_file(path)


def newest_streaming_output(context_dir: Path) -> Path:
    candidates = sorted(context_dir.glob("streaming_out_*.json"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"missing NeMo streaming output under {context_dir}")
    return candidates[-1]


def parse_sentinel_predictions(output_path: Path) -> dict[str, str]:
    predictions: dict[str, str] = {}
    duplicates: list[str] = []
    for row in read_jsonl(output_path):
        sample_id, _reference = parse_sentinel_text(str(row.get("text", "")))
        if sample_id in predictions:
            duplicates.append(sample_id)
        predictions[sample_id] = str(row.get("pred_text", ""))
    if duplicates:
        raise ValueError(f"duplicate predictions in NeMo output: {duplicates[:5]}")
    return predictions


def parse_ordered_predictions(output_path: Path, records: Sequence[StreamingRecord]) -> dict[str, str]:
    rows = read_jsonl(output_path)
    if len(rows) != len(records):
        raise ValueError(f"prediction count {len(rows)} != record count {len(records)}")
    return {record.sample_id: str(row.get("pred_text", "")) for record, row in zip(records, rows, strict=True)}


def validate_prediction_ids(predictions: dict[str, str], records: Sequence[StreamingRecord]) -> None:
    expected = {item.sample_id for item in records}
    actual = set(predictions)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(f"prediction ID mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def per_sample_rows(records: Sequence[StreamingRecord], predictions: dict[str, str]) -> list[dict[str, Any]]:
    validate_prediction_ids(predictions, records)
    rows = []
    for record in sorted(records, key=lambda item: item.original_index):
        hypothesis = predictions[record.sample_id]
        rows.append(
            {
                "sample_id": record.sample_id,
                "reference": record.reference,
                "hypothesis": hypothesis,
                "pipeline_status": "PASSED",
                "empty_hypothesis": not hypothesis.strip(),
            }
        )
    return rows


def metrics_for(records: Sequence[StreamingRecord], predictions: dict[str, str]) -> dict[str, Any]:
    return summarize_predictions(per_sample_rows(records, predictions))


def compare_predictions(
    records: Sequence[StreamingRecord],
    baseline: dict[str, str],
    candidate: dict[str, str],
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> PredictionComparison:
    expected = {item.sample_id for item in records}
    candidate_ids = set(candidate)
    missing = sorted(expected - candidate_ids)
    unexpected = sorted(candidate_ids - expected)
    exact = 0
    normalized = 0
    for sample_id in sorted(expected & set(baseline) & candidate_ids):
        if baseline[sample_id] != candidate[sample_id]:
            exact += 1
        if normalize_sl_asr_text(baseline[sample_id]) != normalize_sl_asr_text(candidate[sample_id]):
            normalized += 1
    metric_diffs: dict[str, Any] = {}
    for scope in ("raw", "normalized"):
        for key, baseline_value in baseline_metrics[scope].items():
            candidate_value = candidate_metrics[scope].get(key)
            if candidate_value != baseline_value:
                metric_diffs[f"{scope}.{key}"] = {"baseline": baseline_value, "candidate": candidate_value}
    empty_diff = (
        candidate_metrics["raw"]["empty_hypothesis_count"]
        - baseline_metrics["raw"]["empty_hypothesis_count"]
    )
    return PredictionComparison(
        exact_mismatch_count=exact,
        normalized_mismatch_count=normalized,
        missing_ids=missing,
        duplicate_ids=[],
        unexpected_ids=unexpected,
        metric_differences=metric_diffs,
        empty_hypothesis_difference=empty_diff,
    )


def round_float(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def parse_active_seconds(log_text: str) -> float | None:
    matches = ACTIVE_TIME_PATTERN.findall(log_text)
    if not matches:
        return None
    return float(matches[-1])


def command_as_shell(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def query_physical_gpu(index: str) -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi query failed")
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        if parts[0] == str(index):
            return {
                "index": parts[0],
                "name": parts[1],
                "memory_used_mib": float(parts[2]),
                "memory_total_mib": float(parts[3]),
                "utilization_percent": float(parts[4]),
                "power_watts": float(parts[5]),
            }
    raise RuntimeError(f"physical GPU {index} not found in nvidia-smi output")


class NvidiaSmiMonitor:
    def __init__(self, *, physical_gpu_index: str, output_csv: Path, interval_seconds: float = 0.2) -> None:
        self.physical_gpu_index = str(physical_gpu_index)
        self.output_csv = output_csv
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["timestamp", "index", "name", "memory_used_mib", "utilization_percent", "power_watts"])
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                row = query_physical_gpu(self.physical_gpu_index)
                with self.output_csv.open("a", encoding="utf-8", newline="") as fp:
                    writer = csv.writer(fp)
                    writer.writerow(
                        [
                            f"{time.time():.6f}",
                            row["index"],
                            row["name"],
                            row["memory_used_mib"],
                            row["utilization_percent"],
                            row["power_watts"],
                        ]
                    )
            except Exception:
                pass
            self._stop.wait(self.interval_seconds)


def parse_monitor_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "sample_count": 0,
            "mean_utilization_percent": 0.0,
            "median_utilization_percent": 0.0,
            "p95_utilization_percent": 0.0,
            "fraction_at_or_above_80_percent": 0.0,
            "peak_memory_mib": 0.0,
            "mean_power_watts": 0.0,
            "p95_power_watts": 0.0,
        }
    utilizations: list[float] = []
    memories: list[float] = []
    powers: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                utilizations.append(float(row["utilization_percent"]))
                memories.append(float(row["memory_used_mib"]))
                powers.append(float(row["power_watts"]))
            except (KeyError, ValueError):
                continue
    if not utilizations:
        return {
            "sample_count": 0,
            "mean_utilization_percent": 0.0,
            "median_utilization_percent": 0.0,
            "p95_utilization_percent": 0.0,
            "fraction_at_or_above_80_percent": 0.0,
            "peak_memory_mib": 0.0,
            "mean_power_watts": 0.0,
            "p95_power_watts": 0.0,
        }

    def p95(values: list[float]) -> float:
        sorted_values = sorted(values)
        index = min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))
        return sorted_values[index]

    return {
        "sample_count": len(utilizations),
        "mean_utilization_percent": round(statistics.mean(utilizations), 6),
        "median_utilization_percent": round(float(statistics.median(utilizations)), 6),
        "p95_utilization_percent": round(p95(utilizations), 6),
        "fraction_at_or_above_80_percent": round(sum(1 for value in utilizations if value >= 80.0) / len(utilizations), 6),
        "peak_memory_mib": round(max(memories), 6),
        "mean_power_watts": round(statistics.mean(powers), 6),
        "p95_power_watts": round(p95(powers), 6),
    }


def ensure_gpu_idle(*, physical_gpu_index: str, max_memory_mib: float, max_utilization_percent: float) -> dict[str, Any]:
    row = query_physical_gpu(physical_gpu_index)
    if row["memory_used_mib"] > max_memory_mib or row["utilization_percent"] > max_utilization_percent:
        raise RuntimeError(
            f"physical GPU {physical_gpu_index} is occupied: "
            f"{row['memory_used_mib']} MiB, {row['utilization_percent']}% utilization"
        )
    return row


def privacy_safe_child_stream_line(line: str) -> str | None:
    """Return a terminal-safe child-process line, or None when it may expose data."""
    stripped = line.strip()
    if not stripped:
        return None
    if any(marker in stripped for marker in RAW_CHILD_STREAM_MARKERS):
        return None
    if re.search(r"(^|\s)/(?:home|mnt|tmp|var|run|sata-ssd)\S*", stripped):
        return None
    if re.search(r"\b(?:text|reference|hypothesis|pred_text)\s*[:=]", stripped, flags=re.IGNORECASE):
        return None
    if any(stripped.startswith(prefix) for prefix in SAFE_CHILD_STREAM_PREFIXES):
        return line
    if stripped.startswith(("  ", "\t", "-", "{", "}", "[", "]")):
        return None
    if "streaming" in stripped.lower() or "transcribe" in stripped.lower():
        return line
    return None


def run_subprocess_with_monitor(
    command: Sequence[str],
    *,
    env: dict[str, str],
    log_path: Path,
    monitor_csv: Path,
    physical_gpu_index: str,
    monitor_interval_seconds: float,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    monitor = NvidiaSmiMonitor(
        physical_gpu_index=physical_gpu_index,
        output_csv=monitor_csv,
        interval_seconds=monitor_interval_seconds,
    )
    start = time.perf_counter()
    monitor.start()
    completed_output: list[str] = []
    return_code = 1
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        with log_path.open("w", encoding="utf-8") as log_fp:
            for line in process.stdout:
                completed_output.append(line)
                safe_line = privacy_safe_child_stream_line(line)
                if safe_line is not None:
                    print(safe_line, end="", file=sys.stderr, flush=True)
                log_fp.write(line)
                log_fp.flush()
        return_code = process.wait(timeout=timeout_seconds)
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise
    finally:
        monitor.stop()
    wall = time.perf_counter() - start
    log_text = "".join(completed_output)
    monitor_summary = parse_monitor_csv(monitor_csv)
    status = "PASSED" if return_code == 0 else "FAILED"
    if return_code != 0 and any(pattern in log_text for pattern in OOM_PATTERNS):
        status = "ENVIRONMENT_BLOCKED"
    return {
        "command": command_as_shell(command),
        "exit_status": return_code,
        "status": status,
        "wall_time_seconds": round(wall, 6),
        "active_wall_time_seconds": round_float(parse_active_seconds(log_text)),
        "monitor": monitor_summary,
        "log_path": str(log_path),
        "monitor_csv": str(monitor_csv),
    }


def build_nemo_command(
    *,
    python_executable: Path,
    nemo_script: Path,
    checkpoint: Path,
    manifest: Path,
    output_path: Path,
    batch_size: int,
    context: Sequence[int],
) -> list[str]:
    return [
        str(python_executable),
        str(nemo_script),
        f"model_path={checkpoint}",
        f"batch_size={batch_size}",
        "target_lang=sl-SI",
        "strip_lang_tags=true",
        f"att_context_size=[{context[0]},{context[1]}]",
        f"output_path={output_path}",
        "cuda=0",
        "compute_dtype=float32",
        "amp=false",
        "matmul_precision=highest",
        f"dataset_manifest={manifest}",
    ]


def run_batched_arm(
    *,
    records: Sequence[StreamingRecord],
    batch_size: int,
    bucketed: bool,
    run_dir: Path,
    python_executable: Path,
    nemo_script: Path,
    checkpoint: Path,
    context: Sequence[int],
    env: dict[str, str],
    physical_gpu_index: str,
    monitor_interval_seconds: float,
) -> dict[str, Any]:
    layout = make_batches(records, batch_size=batch_size, bucketed=bucketed)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "input.local.jsonl"
    manifest_sha = write_nemo_manifest(manifest, layout, with_sentinel=True)
    output_dir = run_dir / "nemo-output"
    command = build_nemo_command(
        python_executable=python_executable,
        nemo_script=nemo_script,
        checkpoint=checkpoint,
        manifest=manifest,
        output_path=output_dir,
        batch_size=batch_size,
        context=context,
    )
    execution = run_subprocess_with_monitor(
        command,
        env=env,
        log_path=run_dir / "nemo.log",
        monitor_csv=run_dir / "gpu-monitor.local.csv",
        physical_gpu_index=physical_gpu_index,
        monitor_interval_seconds=monitor_interval_seconds,
    )
    payload: dict[str, Any] = {
        "batch_size": batch_size,
        "bucketed": bucketed,
        "layout": asdict(layout),
        "input_manifest_sha256": manifest_sha,
        "execution": execution,
    }
    if execution["exit_status"] == 0:
        output_path = newest_streaming_output(output_dir)
        predictions = parse_sentinel_predictions(output_path)
        validate_prediction_ids(predictions, records)
        local_prediction_rows = [
            {"sample_id": item.sample_id, "hypothesis": predictions[item.sample_id]}
            for item in sorted(records, key=lambda value: value.original_index)
        ]
        atomic_write_jsonl(run_dir / "predictions.local.jsonl", local_prediction_rows)
        metrics = metrics_for(records, predictions)
        audio_duration = sum(item.duration for item in records)
        wall_time = float(execution["wall_time_seconds"])
        active_time = execution.get("active_wall_time_seconds")
        payload.update(
            {
                "status": "PASSED",
                "rows": len(records),
                "prediction_count": len(predictions),
                "metrics": metrics,
                "audio_duration_seconds": round(audio_duration, 6),
                "end_to_end_real_time_factor": round(wall_time / audio_duration, 6) if audio_duration else None,
                "active_real_time_factor": round(active_time / audio_duration, 6) if active_time and audio_duration else None,
                "end_to_end_audio_seconds_per_wall_second": round(audio_duration / wall_time, 6) if wall_time else None,
                "active_audio_seconds_per_wall_second": round(audio_duration / active_time, 6) if active_time else None,
                "utterances_per_second": round(len(records) / wall_time, 6) if wall_time else None,
                "output_path": str(output_path),
            }
        )
    else:
        payload["status"] = execution["status"]
    atomic_write_json(run_dir / "summary.local.json", payload)
    return payload


def run_old_ordered_arm(
    *,
    records: Sequence[StreamingRecord],
    run_dir: Path,
    python_executable: Path,
    repo_script: Path,
    checkpoint: Path,
    context: Sequence[int],
    env: dict[str, str],
) -> dict[str, Any]:
    layout = make_batches(records, batch_size=1, bucketed=False)
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "old-input.local.jsonl"
    write_nemo_manifest(manifest, layout, with_sentinel=False)
    output_root = run_dir / "old-wrapper-output"
    command = [
        str(python_executable),
        str(repo_script),
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(checkpoint),
        "--context",
        f"[{context[0]},{context[1]}]",
        "--batch-size",
        "1",
        "--cuda",
        "0",
        "--output-dir",
        str(output_root),
    ]
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
    wall = time.perf_counter() - start
    (run_dir / "old-wrapper.log").write_text(completed.stdout, encoding="utf-8")
    payload: dict[str, Any] = {
        "command": command_as_shell(command),
        "exit_status": completed.returncode,
        "wall_time_seconds": round(wall, 6),
    }
    if completed.returncode == 0:
        context_dirs = sorted(output_root.glob("*/context_*"), key=lambda item: item.stat().st_mtime)
        if not context_dirs:
            raise FileNotFoundError(f"old wrapper produced no context directory in {output_root}")
        output = newest_streaming_output(context_dirs[-1])
        predictions = parse_ordered_predictions(output, list(records))
        atomic_write_jsonl(
            run_dir / "old-predictions.local.jsonl",
            [{"sample_id": item.sample_id, "hypothesis": predictions[item.sample_id]} for item in records],
        )
        payload["status"] = "PASSED"
        payload["output_path"] = str(output)
        payload["prediction_count"] = len(predictions)
    else:
        payload["status"] = "FAILED"
    atomic_write_json(run_dir / "old-summary.local.json", payload)
    return payload


def load_local_predictions(path: Path) -> dict[str, str]:
    return {str(row["sample_id"]): str(row["hypothesis"]) for row in read_jsonl(path)}


def arm_is_parity_eligible(arm: dict[str, Any], comparison: PredictionComparison, *, max_peak_memory_mib: float) -> bool:
    if arm.get("status") != "PASSED":
        return False
    peak = float(arm.get("execution", {}).get("monitor", {}).get("peak_memory_mib", 0.0))
    return comparison.exact_parity and peak <= max_peak_memory_mib


def should_run_batch_256(batch_128: dict[str, Any], batch_64: dict[str, Any], *, max_memory_mib: float, min_gain: float) -> bool:
    if batch_128.get("status") != "PASSED" or not batch_128.get("parity_eligible"):
        return False
    peak = float(batch_128.get("execution", {}).get("monitor", {}).get("peak_memory_mib", 0.0))
    if peak > max_memory_mib:
        return False
    t128 = batch_128.get("end_to_end_audio_seconds_per_wall_second")
    t64 = batch_64.get("end_to_end_audio_seconds_per_wall_second")
    if not t128 or not t64:
        return False
    return float(t128) >= float(t64) * min_gain


def select_batch_policy(arms: Sequence[dict[str, Any]], *, within_best_fraction: float) -> dict[str, Any]:
    eligible = [
        arm
        for arm in arms
        if arm.get("parity_eligible") and arm.get("end_to_end_audio_seconds_per_wall_second") is not None
    ]
    if not eligible:
        return {"batch_size": 1, "duration_bucketing": True, "reason": "no batch above 1 was eligible"}
    best = max(float(arm["end_to_end_audio_seconds_per_wall_second"]) for arm in eligible)
    threshold = best * within_best_fraction
    selected = min(
        (arm for arm in eligible if float(arm["end_to_end_audio_seconds_per_wall_second"]) >= threshold),
        key=lambda item: int(item["batch_size"]),
    )
    return {
        "batch_size": int(selected["batch_size"]),
        "duration_bucketing": bool(selected["bucketed"]),
        "best_throughput": round(best, 6),
        "selected_throughput": selected["end_to_end_audio_seconds_per_wall_second"],
        "within_best_fraction": within_best_fraction,
    }


def scientific_classification(*, selected_batch: int, exact_parity_above_one: bool, selected_speedup: float | None) -> str:
    if selected_batch > 1 and exact_parity_above_one and selected_speedup is not None and selected_speedup >= 1.25:
        return "A100_BATCHED_STREAMING_SUPPORTED"
    if exact_parity_above_one:
        return "A100_BATCHED_STREAMING_EQUIVALENT_NO_MATERIAL_GAIN"
    return "A100_BATCHED_STREAMING_NOT_EQUIVALENT"


def privacy_safe_arm_summary(arm: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "batch_size",
        "bucketed",
        "status",
        "rows",
        "prediction_count",
        "audio_duration_seconds",
        "end_to_end_real_time_factor",
        "active_real_time_factor",
        "end_to_end_audio_seconds_per_wall_second",
        "active_audio_seconds_per_wall_second",
        "utterances_per_second",
        "parity_eligible",
        "exact_mismatch_count",
        "normalized_mismatch_count",
        "metric_differences",
        "empty_hypothesis_difference",
    ]
    summary = {key: arm[key] for key in keys if key in arm}
    if "layout" in arm:
        layout = arm["layout"]
        summary["layout"] = {
            "batch_count": layout["batch_count"],
            "full_batch_count": layout["full_batch_count"],
            "final_partial_batch_size": layout["final_partial_batch_size"],
            "actual_audio_seconds": layout["actual_audio_seconds"],
            "padded_audio_seconds": layout["padded_audio_seconds"],
            "padding_ratio": layout["padding_ratio"],
            "max_padded_batch_duration": layout["max_padded_batch_duration"],
        }
    if "execution" in arm:
        summary["execution"] = {
            "exit_status": arm["execution"]["exit_status"],
            "wall_time_seconds": arm["execution"]["wall_time_seconds"],
            "active_wall_time_seconds": arm["execution"].get("active_wall_time_seconds"),
            "monitor": arm["execution"].get("monitor", {}),
        }
    if "metrics" in arm:
        summary["metrics"] = arm["metrics"]
    return summary
