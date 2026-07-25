from __future__ import annotations

import concurrent.futures
import hashlib
import hmac
import json
import math
import multiprocessing
import os
import queue
import resource
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.scale200_corpus import load_augmentation_config
from slaif_asr.transcript_preserving_augmentation import (
    apply_profile_transform,
    parameters_for_profile,
    read_mono_pcm16,
)


EXPECTED_CORPUS_ID = "sl-corpus-v4-gams-16000-training-v1"
EXPECTED_FIXED_TEXT_SHA256 = "dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14"
EXPECTED_ALL_VIEWS_SHA256 = "9207429fdd675d6a8ea491f6f6ce3647e1fc9ec22e439c9548ad1120268e3bca"
EXPECTED_PROFILE_CONFIG_SHA256 = "ee93dce63a0b5f47b4bc2426d932fe04ccc4a161e13d40349578afabe4fdce40"
EXPECTED_AUGMENTATION_KEY = "scale2000-otf-transcript-preserving-v1"
EXPECTED_ALGORITHM_VERSION = "otf-transcript-preserving-v1"
EXPECTED_PROFILE_COUNT = 11
EXPECTED_CLEAN_FILES = 144_000
EXPECTED_SEMANTIC_ROWS = 16_000
EXPECTED_WORKERS = 3
EXPECTED_SAMPLE_RATE = 16_000
OFFLINE_REPLAY_STATUS = "SCALE2000_OFFLINE_AUGMENTATION_REPLAY_NOT_EXACT"

FORBIDDEN_PUBLIC_KEYS = {
    "audio_filepath",
    "hypothesis",
    "local_manifest",
    "local_path",
    "path",
    "raw_reference",
    "reference",
    "source_path",
    "spoken_text",
    "target_text",
    "text",
    "transcript",
}
ABSOLUTE_PATH_PREFIXES = (
    "/data/",
    "/home/",
    "/mnt/",
    "/opt/",
    "/root/",
    "/srv/",
    "/synology/",
    "/tmp/",
    "/var/",
)


@dataclass(frozen=True)
class CleanAudioRecord:
    semantic_key: str
    voice: str
    source_audio_sha256: str
    audio_path: Path
    duration_seconds: float
    transcript: str


@dataclass(frozen=True)
class VirtualAugmentationSpec:
    virtual_exposure_id: int
    profile_id: str
    profile_index: int
    parameters: dict[str, Any]
    parameter_seed: str
    augmentation_identity_sha256: str
    profile_space_sha256: str


@dataclass(frozen=True)
class PreparedSample:
    waveform: Any
    sample_rate: int
    transcript: str
    spec: VirtualAugmentationSpec
    source_duration_seconds: float
    output_duration_seconds: float
    source_bytes: int
    processing_wall_seconds: float
    processing_cpu_seconds: float


@dataclass(frozen=True)
class MicrobatchTask:
    microbatch_index: int
    exposures: tuple[tuple[CleanAudioRecord, int], ...]


@dataclass(frozen=True)
class PreparedMicrobatch:
    microbatch_index: int
    waveforms: tuple[Any, ...]
    sample_rates: tuple[int, ...]
    transcript_preserved: bool
    output_audio_seconds: float
    source_bytes: int
    processing_wall_seconds: float
    processing_cpu_seconds: float
    started_at: float
    finished_at: float
    worker_peak_rss_mib: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: expected a JSON object")
    return payload


def validate_otf_config(config: dict[str, Any], *, profile_config: dict[str, Any]) -> None:
    if config.get("schema_version") != "1.0":
        raise ValueError("unsupported on-the-fly augmentation config schema")
    if config.get("benchmark_id") != "otf-augmentation-scale2000-fill-rate-v1":
        raise ValueError("unexpected on-the-fly augmentation benchmark id")

    source = config.get("data_source", {})
    expected_source = {
        "corpus_id": EXPECTED_CORPUS_ID,
        "semantic_rows": EXPECTED_SEMANTIC_ROWS,
        "clean_files": EXPECTED_CLEAN_FILES,
        "fixed_text_sha256": EXPECTED_FIXED_TEXT_SHA256,
        "all_views_sha256": EXPECTED_ALL_VIEWS_SHA256,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"unexpected data source value for {key}")

    augmentation = config.get("augmentation", {})
    expected_augmentation = {
        "profile_config_sha256": EXPECTED_PROFILE_CONFIG_SHA256,
        "otf_aug_random_key": EXPECTED_AUGMENTATION_KEY,
        "algorithm_version": EXPECTED_ALGORITHM_VERSION,
        "offline_replay_status": OFFLINE_REPLAY_STATUS,
        "expected_sample_rate": EXPECTED_SAMPLE_RATE,
    }
    for key, expected in expected_augmentation.items():
        if augmentation.get(key) != expected:
            raise ValueError(f"unexpected augmentation value for {key}")
    load_augmentation_config_from_payload(profile_config)

    pipeline = config.get("pipeline", {})
    if int(pipeline.get("num_workers", 0)) != EXPECTED_WORKERS:
        raise ValueError("fill-rate benchmark requires exactly three workers")
    if pipeline.get("worker_type") != "multiprocessing_spawn":
        raise ValueError("fill-rate benchmark must use multiprocessing spawn workers")
    if int(pipeline.get("physical_microbatch", 0)) != 2:
        raise ValueError("physical microbatch must remain two")
    if int(pipeline.get("gradient_accumulation", 0)) != 4:
        raise ValueError("gradient accumulation must remain four")
    if int(pipeline.get("effective_batch", 0)) != 8:
        raise ValueError("effective batch must remain eight")
    if int(pipeline.get("prefetch_microbatches", 0)) < 1:
        raise ValueError("prefetch queue depth must be positive")

    benchmark = config.get("benchmark", {})
    if int(benchmark.get("warmup_microbatches", 0)) < 1:
        raise ValueError("warmup microbatches must be positive")
    if int(benchmark.get("measured_microbatches", 0)) < 1:
        raise ValueError("measured microbatches must be positive")
    rates = [float(value) for value in benchmark.get("target_examples_per_second", [])]
    if rates != [12.8, 16.0, 24.0, 32.0]:
        raise ValueError("unexpected target consumer rates")
    if float(benchmark.get("primary_target_examples_per_second", 0.0)) != 12.8:
        raise ValueError("primary target must remain 12.8 examples/s")


def load_augmentation_config_from_payload(profile_config: dict[str, Any]) -> None:
    if profile_config.get("policy_id") != "scale200-transcript-preserving-v1":
        raise ValueError("unexpected transcript-preserving augmentation policy")
    profiles = profile_config.get("augmentation_profiles", [])
    if len(profiles) != EXPECTED_PROFILE_COUNT:
        raise ValueError("exactly eleven augmentation profiles are required")
    profile_ids = [str(profile.get("profile_id", "")) for profile in profiles]
    if len(set(profile_ids)) != EXPECTED_PROFILE_COUNT or any(not value for value in profile_ids):
        raise ValueError("augmentation profile IDs must be unique and non-empty")


def load_otf_config(config_path: Path, *, repository_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = load_json(config_path)
    profile_path = repository_root / str(config["augmentation"]["profile_config"])
    if sha256_file(profile_path) != EXPECTED_PROFILE_CONFIG_SHA256:
        raise RuntimeError("augmentation profile config SHA256 mismatch")
    profile_config = load_augmentation_config(profile_path)
    validate_otf_config(config, profile_config=profile_config)
    return config, list(profile_config["augmentation_profiles"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}: expected JSON objects")
                rows.append(row)
    return rows


def load_scale2000_clean_records(config: dict[str, Any], *, runs_root: Path) -> list[CleanAudioRecord]:
    source = config["data_source"]
    fixed_text_path = runs_root / str(source["fixed_text_relative_path"])
    all_views_path = runs_root / str(source["all_views_relative_path"])
    if not fixed_text_path.is_file() or not all_views_path.is_file():
        raise FileNotFoundError("scale-2000 fixed text or all-views manifest is unavailable")
    if sha256_file(fixed_text_path) != EXPECTED_FIXED_TEXT_SHA256:
        raise RuntimeError("scale-2000 fixed text SHA256 mismatch")
    if sha256_file(all_views_path) != EXPECTED_ALL_VIEWS_SHA256:
        raise RuntimeError("scale-2000 all-views SHA256 mismatch")

    text_rows = _read_jsonl(fixed_text_path)
    if len(text_rows) != EXPECTED_SEMANTIC_ROWS:
        raise RuntimeError("scale-2000 fixed text row count mismatch")
    text_by_key: dict[str, str] = {}
    for row in text_rows:
        semantic_key = str(row["candidate_id"])
        transcript = str(row["spoken_text"])
        if semantic_key in text_by_key:
            raise RuntimeError("duplicate scale-2000 semantic key")
        text_by_key[semantic_key] = transcript

    records: list[CleanAudioRecord] = []
    seen: set[tuple[str, str]] = set()
    with all_views_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("view_type") != "clean":
                continue
            semantic_key = str(row["semantic_key"])
            voice = str(row["voice"])
            identity = (semantic_key, voice)
            if identity in seen:
                raise RuntimeError("duplicate scale-2000 clean view identity")
            seen.add(identity)
            transcript = text_by_key.get(semantic_key)
            if transcript is None:
                raise RuntimeError("clean view does not map to fixed text")
            audio_path = Path(str(row["audio_filepath"]))
            if not audio_path.is_file():
                raise FileNotFoundError("scale-2000 clean audio file is unavailable")
            source_hash = str(row.get("observed_audio_sha256") or row.get("audio_sha256") or "")
            if len(source_hash) != 64:
                raise RuntimeError("clean source audio SHA256 is unavailable")
            if int(row.get("sample_rate", 0)) != EXPECTED_SAMPLE_RATE:
                raise RuntimeError("clean source audio sample rate mismatch")
            records.append(
                CleanAudioRecord(
                    semantic_key=semantic_key,
                    voice=voice,
                    source_audio_sha256=source_hash,
                    audio_path=audio_path,
                    duration_seconds=float(row["duration_seconds"]),
                    transcript=transcript,
                )
            )
    if len(records) != EXPECTED_CLEAN_FILES:
        raise RuntimeError(f"expected {EXPECTED_CLEAN_FILES} clean records, found {len(records)}")
    records.sort(key=lambda row: (row.semantic_key, row.voice))
    return records


def profile_space_sha256(profiles: Sequence[dict[str, Any]]) -> str:
    return canonical_json_sha256(list(profiles))


def _keyed_digest(key: str, *parts: str) -> str:
    message = "\x1f".join(parts).encode("utf-8")
    return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()


def virtual_augmentation_spec(
    record: CleanAudioRecord,
    virtual_exposure_id: int,
    profiles: Sequence[dict[str, Any]],
    *,
    corpus_id: str = EXPECTED_CORPUS_ID,
    augmentation_key: str = EXPECTED_AUGMENTATION_KEY,
    algorithm_version: str = EXPECTED_ALGORITHM_VERSION,
) -> VirtualAugmentationSpec:
    if virtual_exposure_id < 0:
        raise ValueError("virtual exposure ID must be non-negative")
    if len(profiles) != EXPECTED_PROFILE_COUNT:
        raise ValueError("exactly eleven profiles are required")
    space_sha = profile_space_sha256(profiles)
    identity = _keyed_digest(
        augmentation_key,
        corpus_id,
        record.semantic_key,
        record.voice,
        record.source_audio_sha256,
        str(virtual_exposure_id),
        space_sha,
        augmentation_key,
        algorithm_version,
    )
    profile_index = int(identity[:16], 16) % len(profiles)
    profile = profiles[profile_index]
    parameter_seed = _keyed_digest(augmentation_key, identity, "parameters")
    parameters = parameters_for_profile(profile, semantic_key=parameter_seed)
    return VirtualAugmentationSpec(
        virtual_exposure_id=virtual_exposure_id,
        profile_id=str(profile["profile_id"]),
        profile_index=profile_index,
        parameters=parameters,
        parameter_seed=parameter_seed,
        augmentation_identity_sha256=identity,
        profile_space_sha256=space_sha,
    )


def select_source_record(
    records: Sequence[CleanAudioRecord],
    virtual_exposure_id: int,
    *,
    augmentation_key: str = EXPECTED_AUGMENTATION_KEY,
) -> CleanAudioRecord:
    if not records:
        raise ValueError("clean source records are required")
    digest = _keyed_digest(augmentation_key, EXPECTED_CORPUS_ID, str(virtual_exposure_id), "source")
    return records[int(digest[:16], 16) % len(records)]


def prepare_virtual_sample(
    record: CleanAudioRecord,
    virtual_exposure_id: int,
    profiles: Sequence[dict[str, Any]],
    *,
    augmentation_key: str = EXPECTED_AUGMENTATION_KEY,
    algorithm_version: str = EXPECTED_ALGORITHM_VERSION,
) -> PreparedSample:
    import numpy as np

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    source_bytes = record.audio_path.stat().st_size
    source, frames = read_mono_pcm16(record.audio_path)
    spec = virtual_augmentation_spec(
        record,
        virtual_exposure_id,
        profiles,
        augmentation_key=augmentation_key,
        algorithm_version=algorithm_version,
    )
    transformed, _details = apply_profile_transform(
        source,
        spec.profile_id,
        spec.parameters,
        seed_text=spec.parameter_seed,
    )
    waveform = np.ascontiguousarray(transformed, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise RuntimeError("on-the-fly augmentation produced an invalid waveform")
    if frames <= 0:
        raise RuntimeError("clean source waveform is empty")
    return PreparedSample(
        waveform=waveform,
        sample_rate=EXPECTED_SAMPLE_RATE,
        transcript=record.transcript,
        spec=spec,
        source_duration_seconds=record.duration_seconds,
        output_duration_seconds=float(waveform.size / EXPECTED_SAMPLE_RATE),
        source_bytes=source_bytes,
        processing_wall_seconds=time.perf_counter() - started_wall,
        processing_cpu_seconds=time.process_time() - started_cpu,
    )


def waveform_sha256(waveform: Any) -> str:
    return hashlib.sha256(memoryview(waveform).cast("B")).hexdigest()


def _fingerprint_exposure(
    task: tuple[CleanAudioRecord, int, list[dict[str, Any]], str, str],
) -> tuple[int, str, str, bool]:
    record, exposure_id, profiles, augmentation_key, algorithm_version = task
    sample = prepare_virtual_sample(
        record,
        exposure_id,
        profiles,
        augmentation_key=augmentation_key,
        algorithm_version=algorithm_version,
    )
    return (
        exposure_id,
        canonical_json_sha256(asdict(sample.spec)),
        waveform_sha256(sample.waveform),
        sample.transcript == record.transcript,
    )


def run_determinism_checks(
    records: Sequence[CleanAudioRecord],
    profiles: list[dict[str, Any]],
    *,
    example_count: int,
    augmentation_key: str = EXPECTED_AUGMENTATION_KEY,
    algorithm_version: str = EXPECTED_ALGORITHM_VERSION,
) -> dict[str, Any]:
    if example_count < 1:
        raise ValueError("determinism example count must be positive")
    tasks = [
        (
            select_source_record(records, index, augmentation_key=augmentation_key),
            index,
            profiles,
            augmentation_key,
            algorithm_version,
        )
        for index in range(example_count)
    ]
    context = multiprocessing.get_context("spawn")

    def execute(task_rows: list[tuple[CleanAudioRecord, int, list[dict[str, Any]], str, str]], workers: int) -> dict[int, tuple[str, str, bool]]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            rows = list(pool.map(_fingerprint_exposure, task_rows))
        return {exposure_id: (spec_hash, wave_hash, preserved) for exposure_id, spec_hash, wave_hash, preserved in rows}

    first = execute(tasks, EXPECTED_WORKERS)
    restart = execute(tasks, EXPECTED_WORKERS)
    reversed_order = execute(list(reversed(tasks)), EXPECTED_WORKERS)
    single_worker = execute(tasks, 1)
    same_worker_count = first == restart
    order_independent = first == reversed_order
    worker_count_independent = first == single_worker
    transcript_preserved = all(value[2] for value in first.values())
    passed = same_worker_count and order_independent and worker_count_independent and transcript_preserved
    return {
        "status": "PASSED" if passed else "FAILED",
        "examples": example_count,
        "same_virtual_exposure_same_worker_count": same_worker_count,
        "same_virtual_exposure_after_restart": same_worker_count,
        "worker_order_independent": order_independent,
        "worker_count_independent": worker_count_independent,
        "transcript_preserved": transcript_preserved,
        "augmentation_key_recorded": bool(augmentation_key),
    }


def build_microbatch_tasks(
    records: Sequence[CleanAudioRecord],
    *,
    total_microbatches: int,
    physical_microbatch: int,
    exposure_offset: int,
    augmentation_key: str = EXPECTED_AUGMENTATION_KEY,
) -> list[MicrobatchTask]:
    tasks: list[MicrobatchTask] = []
    for microbatch_index in range(total_microbatches):
        exposures: list[tuple[CleanAudioRecord, int]] = []
        for item_index in range(physical_microbatch):
            exposure_id = exposure_offset + microbatch_index * physical_microbatch + item_index
            record = select_source_record(records, exposure_id, augmentation_key=augmentation_key)
            exposures.append((record, exposure_id))
        tasks.append(MicrobatchTask(microbatch_index=microbatch_index, exposures=tuple(exposures)))
    return tasks


def _worker_main(
    input_queue: Any,
    output_queue: Any,
    profiles: list[dict[str, Any]],
    augmentation_key: str,
    algorithm_version: str,
) -> None:
    while True:
        task = input_queue.get()
        if task is None:
            return
        started = time.perf_counter()
        wall_total = 0.0
        cpu_total = 0.0
        audio_seconds = 0.0
        source_bytes = 0
        waveforms = []
        sample_rates = []
        transcript_preserved = True
        try:
            for record, exposure_id in task.exposures:
                sample = prepare_virtual_sample(
                    record,
                    exposure_id,
                    profiles,
                    augmentation_key=augmentation_key,
                    algorithm_version=algorithm_version,
                )
                waveforms.append(sample.waveform)
                sample_rates.append(sample.sample_rate)
                audio_seconds += sample.output_duration_seconds
                source_bytes += sample.source_bytes
                wall_total += sample.processing_wall_seconds
                cpu_total += sample.processing_cpu_seconds
                transcript_preserved = transcript_preserved and sample.transcript == record.transcript
            output_queue.put(
                (
                    "result",
                    PreparedMicrobatch(
                        microbatch_index=task.microbatch_index,
                        waveforms=tuple(waveforms),
                        sample_rates=tuple(sample_rates),
                        transcript_preserved=transcript_preserved,
                        output_audio_seconds=audio_seconds,
                        source_bytes=source_bytes,
                        processing_wall_seconds=wall_total,
                        processing_cpu_seconds=cpu_total,
                        started_at=started,
                        finished_at=time.perf_counter(),
                        worker_peak_rss_mib=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
                    ),
                )
            )
        except Exception as exc:
            output_queue.put(("error", task.microbatch_index, type(exc).__name__))
            return


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compute_fill_rate_metrics(
    *,
    wait_seconds: Sequence[float],
    queue_fill_levels: Sequence[int],
    target_examples_per_second: float,
    physical_microbatch: int,
    ready_wait_threshold_ms: float,
    ready_flags: Sequence[bool] | None = None,
) -> dict[str, Any]:
    if not wait_seconds:
        raise ValueError("consumer wait events are required")
    if target_examples_per_second <= 0:
        raise ValueError("target consumer rate must be positive")
    expected_consumer_seconds = len(wait_seconds) * physical_microbatch / target_examples_per_second
    total_wait = sum(wait_seconds)
    total_consumer_seconds = expected_consumer_seconds + total_wait
    if ready_flags is not None and len(ready_flags) != len(wait_seconds):
        raise ValueError("ready flags must match consumer wait events")
    ready_threshold = ready_wait_threshold_ms / 1000.0
    ready_count = (
        sum(bool(value) for value in ready_flags)
        if ready_flags is not None
        else sum(wait <= ready_threshold for wait in wait_seconds)
    )
    waits_ms = [wait * 1000.0 for wait in wait_seconds]
    return {
        "fill_rate": 1.0 - (total_wait / total_consumer_seconds),
        "microbatch_ready_rate": ready_count / len(wait_seconds),
        "underrun_count": len(wait_seconds) - ready_count,
        "consumer_wait_seconds": total_wait,
        "total_consumer_seconds": total_consumer_seconds,
        "p50_consumer_wait_ms": percentile(waits_ms, 0.50),
        "p95_consumer_wait_ms": percentile(waits_ms, 0.95),
        "p99_consumer_wait_ms": percentile(waits_ms, 0.99),
        "queue_fill": {
            "mean": statistics.fmean(queue_fill_levels) if queue_fill_levels else 0.0,
            "minimum": min(queue_fill_levels) if queue_fill_levels else 0,
            "p50": percentile(queue_fill_levels, 0.50),
            "p95": percentile(queue_fill_levels, 0.95),
            "maximum": max(queue_fill_levels) if queue_fill_levels else 0,
        },
    }


def _queue_size(value: Any) -> int:
    try:
        return max(0, int(value.qsize()))
    except (NotImplementedError, OSError):
        return 0


def _take_expected(
    output_queue: Any,
    expected_index: int,
    buffered: dict[int, PreparedMicrobatch],
) -> tuple[PreparedMicrobatch, bool]:
    if expected_index in buffered:
        return buffered.pop(expected_index), True
    ready_without_blocking = True
    while True:
        try:
            result = output_queue.get_nowait()
        except queue.Empty:
            ready_without_blocking = False
            result = output_queue.get(timeout=120.0)
        if result[0] == "error":
            raise RuntimeError(f"augmentation worker failed with {result[2]}")
        prepared = result[1]
        if prepared.microbatch_index == expected_index:
            return prepared, ready_without_blocking
        buffered[prepared.microbatch_index] = prepared


def run_fill_rate_benchmark(
    records: Sequence[CleanAudioRecord],
    profiles: list[dict[str, Any]],
    *,
    target_examples_per_second: float,
    num_workers: int,
    physical_microbatch: int,
    warmup_microbatches: int,
    measured_microbatches: int,
    prefetch_microbatches: int,
    ready_wait_threshold_ms: float,
    exposure_offset: int,
    augmentation_key: str = EXPECTED_AUGMENTATION_KEY,
    algorithm_version: str = EXPECTED_ALGORITHM_VERSION,
) -> dict[str, Any]:
    if num_workers != EXPECTED_WORKERS:
        raise ValueError("benchmark requires exactly three workers")
    total_microbatches = warmup_microbatches + measured_microbatches
    tasks = build_microbatch_tasks(
        records,
        total_microbatches=total_microbatches,
        physical_microbatch=physical_microbatch,
        exposure_offset=exposure_offset,
        augmentation_key=augmentation_key,
    )
    context = multiprocessing.get_context("spawn")
    input_queue = context.Queue()
    output_queue = context.Queue(maxsize=prefetch_microbatches)
    workers = [
        context.Process(
            target=_worker_main,
            args=(input_queue, output_queue, profiles, augmentation_key, algorithm_version),
        )
        for _ in range(num_workers)
    ]
    for worker in workers:
        worker.start()
    next_task_index = 0

    def submit_next() -> bool:
        nonlocal next_task_index
        if next_task_index >= len(tasks):
            return False
        input_queue.put(tasks[next_task_index])
        next_task_index += 1
        return True

    for _index in range(min(prefetch_microbatches, len(tasks))):
        submit_next()

    prefill_started = time.perf_counter()
    prefill_target = min(prefetch_microbatches, len(tasks))
    while _queue_size(output_queue) < prefill_target:
        if time.perf_counter() - prefill_started > 120.0:
            raise TimeoutError("prefetch queue did not fill before timeout")
        if any(not worker.is_alive() for worker in workers):
            raise RuntimeError("augmentation worker exited during prefill")
        time.sleep(0.01)

    interval = physical_microbatch / target_examples_per_second
    buffered: dict[int, PreparedMicrobatch] = {}
    try:
        for microbatch_index in range(warmup_microbatches):
            time.sleep(interval)
            warmup, _ready = _take_expected(output_queue, microbatch_index, buffered)
            del warmup
            submit_next()

        waits: list[float] = []
        ready_flags: list[bool] = []
        queue_levels: list[int] = []
        measured_rows: list[PreparedMicrobatch] = []
        measured_started = time.perf_counter()
        for microbatch_index in range(warmup_microbatches, total_microbatches):
            time.sleep(interval)
            wait_started = time.perf_counter()
            prepared, ready = _take_expected(output_queue, microbatch_index, buffered)
            waits.append(time.perf_counter() - wait_started)
            ready_flags.append(ready)
            queue_levels.append(
                min(prefetch_microbatches, _queue_size(output_queue) + len(buffered))
            )
            measured_rows.append(prepared)
            submit_next()
        measured_wall = time.perf_counter() - measured_started
    finally:
        for _worker in workers:
            input_queue.put(None)
        for worker in workers:
            worker.join(timeout=30.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5.0)
        input_queue.close()
        output_queue.close()

    if len(measured_rows) != measured_microbatches:
        raise RuntimeError("measured microbatch count mismatch")
    if not all(row.transcript_preserved for row in measured_rows):
        raise RuntimeError("transcript changed during on-the-fly augmentation")
    if any(rate != EXPECTED_SAMPLE_RATE for row in measured_rows for rate in row.sample_rates):
        raise RuntimeError("on-the-fly output sample rate mismatch")

    timing = compute_fill_rate_metrics(
        wait_seconds=waits,
        queue_fill_levels=queue_levels,
        target_examples_per_second=target_examples_per_second,
        physical_microbatch=physical_microbatch,
        ready_wait_threshold_ms=ready_wait_threshold_ms,
        ready_flags=ready_flags,
    )
    aggregate_compute_wall = sum(row.processing_wall_seconds for row in measured_rows)
    aggregate_compute_cpu = sum(row.processing_cpu_seconds for row in measured_rows)
    capacity_denominator = aggregate_compute_wall / num_workers
    measured_examples = measured_microbatches * physical_microbatch
    producer_span = max(row.finished_at for row in measured_rows) - min(row.started_at for row in measured_rows)
    total_audio_seconds = sum(row.output_audio_seconds for row in measured_rows)
    total_source_bytes = sum(row.source_bytes for row in measured_rows)
    return {
        "target_examples_per_second": target_examples_per_second,
        "measured_microbatches": measured_microbatches,
        "measured_examples": measured_examples,
        "prepared_examples_per_second": measured_examples / capacity_denominator,
        "prepared_audio_seconds_per_second": total_audio_seconds / capacity_denominator,
        "delivered_examples_per_second": measured_examples / measured_wall,
        "worker_compute_occupancy_estimate": min(1.0, aggregate_compute_wall / (max(producer_span, 1e-9) * num_workers)),
        "worker_cpu_utilization_estimate": min(1.0, aggregate_compute_cpu / (max(producer_span, 1e-9) * num_workers)),
        "source_io_megabytes_per_second": (total_source_bytes / (1024.0 * 1024.0)) / capacity_denominator,
        "peak_worker_rss_mib": max(row.worker_peak_rss_mib for row in measured_rows),
        "benchmark_wall_seconds": measured_wall,
        "transcript_preserved": True,
        **timing,
    }


def classify_fill_rate(result: dict[str, Any], thresholds: dict[str, Any]) -> str:
    determinism = result.get("determinism", {})
    if determinism.get("status") != "PASSED":
        return "OTF_AUG_DETERMINISM_INVALID"
    primary_rate = float(result["primary_target_examples_per_second"])
    primary = next(
        row for row in result["rates"] if float(row["target_examples_per_second"]) == primary_rate
    )
    fill_pass = float(primary["fill_rate"]) >= float(thresholds["minimum_fill_rate"])
    ready_pass = float(primary["microbatch_ready_rate"]) >= float(
        thresholds["minimum_ready_microbatch_rate"]
    )
    wait_pass = float(primary["p95_consumer_wait_ms"]) <= float(
        thresholds["maximum_p95_consumer_wait_ms"]
    )
    if fill_pass and ready_pass and wait_pass:
        return "OTF_AUG_FILL_RATE_PASSES_PRIMARY_TARGET"
    if float(primary["prepared_examples_per_second"]) >= primary_rate * 0.9:
        return "OTF_AUG_FILL_RATE_MARGINAL_PRIMARY_TARGET"
    return "OTF_AUG_FILL_RATE_FAILS_PRIMARY_TARGET"


def mark_rate_passes(result: dict[str, Any], thresholds: dict[str, Any]) -> None:
    for row in result["rates"]:
        row["pass"] = (
            float(row["fill_rate"]) >= float(thresholds["minimum_fill_rate"])
            and float(row["microbatch_ready_rate"]) >= float(
                thresholds["minimum_ready_microbatch_rate"]
            )
            and float(row["p95_consumer_wait_ms"])
            <= float(thresholds["maximum_p95_consumer_wait_ms"])
        )


def validate_benchmark_result(payload: dict[str, Any]) -> None:
    required = {
        "benchmark_id",
        "classification",
        "data_source",
        "determinism",
        "pipeline",
        "rates",
        "runtime",
        "thresholds",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"benchmark result missing fields: {sorted(missing)}")
    if payload["classification"] not in {
        "OTF_AUG_FILL_RATE_PASSES_PRIMARY_TARGET",
        "OTF_AUG_FILL_RATE_MARGINAL_PRIMARY_TARGET",
        "OTF_AUG_FILL_RATE_FAILS_PRIMARY_TARGET",
        "OTF_AUG_DETERMINISM_INVALID",
        "BLOCKED_SCALE2000_CLEAN_AUDIO_UNAVAILABLE",
        "EXPERIMENT_INVALID",
    }:
        raise ValueError("invalid benchmark classification")
    if int(payload["pipeline"]["num_workers"]) != EXPECTED_WORKERS:
        raise ValueError("benchmark result did not use exactly three workers")
    if not isinstance(payload["rates"], list) or not payload["rates"]:
        raise ValueError("benchmark result has no consumer rates")
    validate_public_payload(payload)


def validate_public_payload(payload: Any, *, key: str = "") -> None:
    if key in FORBIDDEN_PUBLIC_KEYS:
        raise ValueError(f"public payload contains forbidden field: {key}")
    if isinstance(payload, dict):
        for child_key, child_value in payload.items():
            validate_public_payload(child_value, key=str(child_key))
    elif isinstance(payload, list):
        for item in payload:
            validate_public_payload(item, key=key)
    elif isinstance(payload, str):
        if payload.startswith(ABSOLUTE_PATH_PREFIXES):
            raise ValueError("public payload contains a local absolute path")
