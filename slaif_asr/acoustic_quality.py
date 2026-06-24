from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import wave
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.config import REPO_ROOT
from slaif_asr.corpus_v2_generation import load_config as load_generation_config
from slaif_asr.corpus_v2_generation import run_dir
from slaif_asr.data_quality import atomic_write_json, atomic_write_jsonl, atomic_write_text, load_json, load_jsonl, sha256_file, sha256_text
from slaif_asr.gpu_policy import SingleGpuInfo, require_single_visible_cuda
from slaif_asr.tts import build_piper_command, convert_to_16k_pcm, load_tts_config, repo_resolve, run_piper_command, sox_version, validate_wav


AUDIO_VALIDATOR_VERSION = "synthetic-audio-validator-v1"
AUDIO_REPORT_SCHEMA_VERSION = "1.0"
AUDIO_CERTIFICATE_SCHEMA_VERSION = "1.0"
CORPUS_ID = "sl-corpus-v2-gams-candidate-reservoir-v1"
REVIEWED_CORPUS_ID = "sl-corpus-v2-gams-candidate-reservoir-v1-reviewed"
EXPECTED_ACCEPTED_SHA256 = "b8a5e4769ef881e90e94f45e36cb4bdbabd24feac0ebcb804fcf5fe760a301d6"
EXPECTED_ACCEPTED_REVIEW_SHA256 = "4dc16336dc9404d48cab196b862e4aa2a4558b20d2728fc954dd7fbb88fed732"
EXPECTED_TEXT_STATUS = "TEXT_ACCEPTED"
PUBLIC_FORBIDDEN_KEYS = {"candidate_id", "candidate_ids", "text", "spoken_text", "target_text", "audio_filepath", "local_path"}
PUBLIC_FORBIDDEN_VALUE_MARKERS = ("gamsv2-", "/" + "home" + "/", "/" + "mnt" + "/" + "data")


@dataclass(frozen=True)
class CorpusV2TtsItem:
    candidate_id: str
    spoken_text: str
    target_text: str
    language: str
    partition_role: str
    source_id: str
    source_family_id: str
    utterance_family_id: str
    domain: str
    phenomena: tuple[str, ...]


@dataclass(frozen=True)
class AudioPaths:
    run_root: Path
    native_dir: Path
    final_dir: Path
    log_dir: Path
    benchmark_dir: Path
    audio_manifest: Path
    validation_report: Path
    synthesis_summary: Path
    benchmark_summary: Path
    gpu_monitor: Path


@dataclass(frozen=True)
class AudioStats:
    sample_rate: int
    channels: int
    sample_width: int
    frames: int
    duration_seconds: float
    peak_ratio: float
    rms_ratio: float
    active_frame_fraction: float
    clipping_fraction: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    sha256: str


def audio_paths(generation_config: dict[str, Any]) -> AudioPaths:
    base = run_dir(generation_config)
    return AudioPaths(
        run_root=base,
        native_dir=base / "audio" / "native-22050",
        final_dir=base / "audio" / "final-16000",
        log_dir=base / "piper-logs",
        benchmark_dir=base / "piper-benchmark",
        audio_manifest=base / "audio-manifest.local.jsonl",
        validation_report=base / "audio-validation.local.json",
        synthesis_summary=base / "audio-synthesis-summary.local.json",
        benchmark_summary=base / "piper-benchmark" / "benchmark-summary.local.json",
        gpu_monitor=base / "gpu-monitor.local.csv",
    )


def synthetic_audio_config_path() -> Path:
    return REPO_ROOT / "configs/data_quality/synthetic_audio_v1.json"


def default_generation_config_path() -> Path:
    return REPO_ROOT / "configs/generation/slovenian_corpus_v2_candidate_reservoir.json"


def default_text_report_path() -> Path:
    return REPO_ROOT / "docs/data-reports/0002-corpus-v2-linguistic-review-admission.json"


def accepted_candidates_path(generation_config: dict[str, Any]) -> Path:
    return run_dir(generation_config) / "accepted-candidates.local.jsonl"


def accepted_review_path(generation_config: dict[str, Any]) -> Path:
    return run_dir(generation_config) / "accepted-linguistic-review.local.jsonl"


def load_corpus_v2_tts_items(path: Path) -> list[CorpusV2TtsItem]:
    rows = load_jsonl(path)
    items: list[CorpusV2TtsItem] = []
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"duplicate or blank candidate_id: {candidate_id!r}")
        seen.add(candidate_id)
        if row.get("schema_version") != "2.0":
            raise ValueError(f"{candidate_id}: expected schema_version 2.0")
        if row.get("language") != "sl-SI":
            raise ValueError(f"{candidate_id}: expected language sl-SI")
        if row.get("partition_role") != "synthetic_candidate":
            raise ValueError(f"{candidate_id}: expected synthetic_candidate partition")
        spoken = str(row.get("spoken_text", ""))
        target = str(row.get("target_text", ""))
        if not spoken or spoken != target:
            raise ValueError(f"{candidate_id}: spoken_text must equal target_text")
        items.append(
            CorpusV2TtsItem(
                candidate_id=candidate_id,
                spoken_text=spoken,
                target_text=target,
                language="sl-SI",
                partition_role="synthetic_candidate",
                source_id=str(row.get("source_id", "")),
                source_family_id=str(row.get("source_family_id", "")),
                utterance_family_id=str(row.get("utterance_family_id", "")),
                domain=str(row.get("domain", "")),
                phenomena=tuple(str(item) for item in row.get("phenomena", [])),
            )
        )
    return sorted(items, key=lambda item: item.candidate_id)


def verify_text_admission_inputs(generation_config: dict[str, Any]) -> dict[str, Any]:
    accepted_path = accepted_candidates_path(generation_config)
    review_path = accepted_review_path(generation_config)
    text_report = load_json(default_text_report_path())
    accepted_sha = sha256_file(accepted_path)
    review_sha = sha256_file(review_path)
    if accepted_sha != EXPECTED_ACCEPTED_SHA256:
        raise RuntimeError(f"accepted candidate partition SHA mismatch: {accepted_sha}")
    if review_sha != EXPECTED_ACCEPTED_REVIEW_SHA256:
        raise RuntimeError(f"accepted review SHA mismatch: {review_sha}")
    if text_report.get("validator", {}).get("status") != EXPECTED_TEXT_STATUS:
        raise RuntimeError("text-admission public report is not TEXT_ACCEPTED")
    rows = load_corpus_v2_tts_items(accepted_path)
    if len(rows) != 415:
        raise RuntimeError(f"expected 415 accepted rows, saw {len(rows)}")
    return {
        "accepted_candidates_sha256": accepted_sha,
        "accepted_review_sha256": review_sha,
        "text_report_sha256": sha256_file(default_text_report_path()),
        "rows": len(rows),
        "review_decision": text_report.get("review", {}).get("whole_file_decision"),
    }


def piper_runtime_env(piper_python: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_path = piper_python if piper_python.is_absolute() else REPO_ROOT / piper_python
    venv_root = python_path.parents[1]
    site_roots = sorted((venv_root / "lib").glob("python*/site-packages/nvidia/*/lib"))
    if site_roots:
        lib_path = ":".join(str(path) for path in site_roots)
        existing = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = f"{lib_path}:{existing}" if existing else lib_path
    return env


class GpuMonitor:
    def __init__(self, path: Path, *, physical_selector: str, interval_seconds: float) -> None:
        self.path = path
        self.physical_selector = physical_selector
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> GpuMonitor:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, "elapsed_seconds,utilization_gpu_percent,memory_used_mib,power_draw_watts\n")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        start = time.perf_counter()
        while not self._stop.is_set():
            row = self._sample(time.perf_counter() - start)
            if row is not None:
                with self.path.open("a", encoding="utf-8") as fp:
                    fp.write(row)
            self._stop.wait(self.interval_seconds)

    def _sample(self, elapsed: float) -> str | None:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={self.physical_selector}",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        if len(parts) < 3:
            return None
        try:
            util = float(parts[0])
            memory = float(parts[1])
            power = float(parts[2])
        except ValueError:
            return None
        return f"{elapsed:.3f},{util:.3f},{memory:.3f},{power:.3f}\n"


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return round(ordered[index], 6)


def monitor_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "sample_count": 0,
            "mean_utilization_percent": None,
            "median_utilization_percent": None,
            "p95_utilization_percent": None,
            "fraction_at_or_above_80_percent": None,
            "peak_memory_mib": None,
            "mean_power_watts": None,
            "p95_power_watts": None,
        }
    utils: list[float] = []
    memory: list[float] = []
    power: list[float] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                utils.append(float(row["utilization_gpu_percent"]))
                memory.append(float(row["memory_used_mib"]))
                power.append(float(row["power_draw_watts"]))
            except (KeyError, ValueError):
                continue
    return {
        "sample_count": len(utils),
        "mean_utilization_percent": round(statistics.fmean(utils), 6) if utils else None,
        "median_utilization_percent": round(statistics.median(utils), 6) if utils else None,
        "p95_utilization_percent": percentile(utils, 0.95),
        "fraction_at_or_above_80_percent": round(sum(1 for value in utils if value >= 80.0) / len(utils), 6) if utils else None,
        "peak_memory_mib": round(max(memory), 6) if memory else None,
        "mean_power_watts": round(statistics.fmean(power), 6) if power else None,
        "p95_power_watts": percentile(power, 0.95),
    }


def read_audio_stats(path: Path) -> AudioStats:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        comptype = wav.getcomptype()
        raw = wav.readframes(frames)
    if comptype != "NONE":
        raise ValueError(f"{path}: expected PCM WAV, got {comptype}")
    if frames <= 0 or not raw:
        raise ValueError(f"{path}: empty audio")
    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM")
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError(f"{path}: empty sample array")
    abs_values = [abs(int(sample)) for sample in samples]
    peak = max(abs_values)
    peak_ratio = peak / 32768.0
    rms_ratio = math.sqrt(sum(value * value for value in abs_values) / len(abs_values)) / 32768.0
    silence_threshold = int(0.001 * 32768)
    active = [value > silence_threshold for value in abs_values]
    active_fraction = sum(1 for value in active if value) / len(active)
    clipping_fraction = sum(1 for value in abs_values if value >= 32760) / len(abs_values)
    leading = 0
    for value in active:
        if value:
            break
        leading += 1
    trailing = 0
    for value in reversed(active):
        if value:
            break
        trailing += 1
    return AudioStats(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frames=frames,
        duration_seconds=frames / sample_rate,
        peak_ratio=peak_ratio,
        rms_ratio=rms_ratio,
        active_frame_fraction=active_fraction,
        clipping_fraction=clipping_fraction,
        leading_silence_seconds=leading / (sample_rate * channels),
        trailing_silence_seconds=trailing / (sample_rate * channels),
        sha256=sha256_file(path),
    )


def _safe_temp_output(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.stem}.part.{os.getpid()}.{threading.get_ident()}{path.suffix}")


def render_one_item(
    *,
    item: CorpusV2TtsItem,
    tts_config: dict[str, Any],
    paths: AudioPaths,
    output_root: Path | None,
) -> dict[str, Any]:
    voice_root = repo_resolve(tts_config["voice"]["local_storage_dir"])
    model_path = voice_root / next(entry["path"] for entry in tts_config["voice"]["files"] if entry["role"] == "model")
    config_path = voice_root / next(entry["path"] for entry in tts_config["voice"]["files"] if entry["role"] == "config")
    piper_python = repo_resolve(tts_config["engine"]["environment"]) / "bin" / "python"
    native_dir = (output_root / "native-22050") if output_root is not None else paths.native_dir
    final_dir = (output_root / "final-16000") if output_root is not None else paths.final_dir
    log_dir = (output_root / "logs") if output_root is not None else paths.log_dir
    native_path = native_dir / f"{item.candidate_id}.native.wav"
    final_path = final_dir / f"{item.candidate_id}.wav"
    log_path = log_dir / f"{item.candidate_id}.piper.log"
    for required in (piper_python, model_path, config_path):
        if not required.exists():
            raise FileNotFoundError(required)
    temp_native = _safe_temp_output(native_path)
    command = build_piper_command(
        piper_python=piper_python,
        model_path=model_path,
        config_path=config_path,
        output_file=temp_native,
        text=item.spoken_text,
    )
    start = time.perf_counter()
    completed = run_piper_command(command, env=piper_runtime_env(piper_python))
    wall = time.perf_counter() - start
    atomic_write_text(log_path, completed.stdout)
    if completed.returncode != 0:
        temp_native.unlink(missing_ok=True)
        raise RuntimeError(f"{item.candidate_id}: Piper failed with exit {completed.returncode}")
    if "Failed to create CUDAExecutionProvider" in completed.stdout or "CPUExecutionProvider" in completed.stdout:
        temp_native.unlink(missing_ok=True)
        raise RuntimeError(f"{item.candidate_id}: Piper attempted CPU provider or failed CUDA provider")
    if "Using CUDA" not in completed.stdout and "CUDAExecutionProvider" not in completed.stdout:
        temp_native.unlink(missing_ok=True)
        raise RuntimeError(f"{item.candidate_id}: Piper log did not confirm CUDA execution")
    native_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_native, native_path)
    native_info = validate_wav(native_path, sample_rate=int(tts_config["voice"]["native_sample_rate"]))
    convert_to_16k_pcm(native_path, final_path)
    final_info = validate_wav(final_path, sample_rate=int(tts_config["voice"]["final_asr_sample_rate"]))
    return {
        "schema_version": "1.0",
        "candidate_id": item.candidate_id,
        "audio_filepath": str(final_path.resolve()),
        "duration_seconds": round(final_info.duration_seconds, 6),
        "sample_rate": final_info.sample_rate,
        "channels": final_info.channels,
        "sample_width": final_info.sample_width,
        "text": item.target_text,
        "target_text_sha256": sha256_text(item.target_text),
        "language": item.language,
        "target_lang": item.language,
        "partition_role": item.partition_role,
        "source_type": "synthetic_tts",
        "source_id": item.source_id,
        "source_family_id": item.source_family_id,
        "utterance_family_id": item.utterance_family_id,
        "domain": item.domain,
        "phenomena": list(item.phenomena),
        "audio_sha256": final_info.sha256,
        "native_audio": {
            "path": str(native_path.resolve()),
            "sample_rate": native_info.sample_rate,
            "channels": native_info.channels,
            "sample_width": native_info.sample_width,
            "sha256": native_info.sha256,
        },
        "audio_validation": {
            "final_peak_ratio": round(final_info.peak_ratio, 6),
            "native_peak_ratio": round(native_info.peak_ratio, 6),
            "conversion": {
                "tool": "sox",
                "version": sox_version(),
                "parameters": ["-r", "16000", "-c", "1", "-b", "16", "-e", "signed-integer"],
            },
        },
        "tts": {
            "engine": tts_config["engine"]["repository_name"],
            "engine_revision": tts_config["engine"]["revision"],
            "engine_license": tts_config["engine"]["license"],
            "voice": tts_config["voice"]["name"],
            "voice_repository": tts_config["voice"]["repository"],
            "voice_revision": tts_config["voice"]["revision"],
            "execution_provider": tts_config["runtime"]["required_execution_provider"],
            "physical_gpu_selector": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "runtime": {
            "piper_wall_time_seconds": round(wall, 6),
            "log_sha256": sha256_file(log_path),
        },
    }


def render_items_concurrently(
    *,
    items: Sequence[CorpusV2TtsItem],
    worker_count: int,
    tts_config: dict[str, Any],
    paths: AudioPaths,
    output_root: Path | None = None,
    monitor_path: Path | None = None,
    monitor_interval_seconds: float = 0.2,
    physical_selector: str = "1",
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    monitor_context = (
        GpuMonitor(monitor_path, physical_selector=physical_selector, interval_seconds=monitor_interval_seconds)
        if monitor_path is not None
        else None
    )
    try:
        if monitor_context is not None:
            monitor_context.__enter__()
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(render_one_item, item=item, tts_config=tts_config, paths=paths, output_root=output_root): item
                for item in items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:  # noqa: BLE001 - recorded as local failure evidence
                    failures.append({"candidate_id": item.candidate_id, "reason": type(exc).__name__, "detail": str(exc)})
    finally:
        if monitor_context is not None:
            monitor_context.__exit__(None, None, None)
    wall = time.perf_counter() - start
    rows = sorted(rows, key=lambda row: str(row["candidate_id"]))
    total_duration = sum(float(row["duration_seconds"]) for row in rows)
    return {
        "worker_count": worker_count,
        "requested": len(items),
        "successful": len(rows),
        "failed": len(failures),
        "failures": failures,
        "wall_time_seconds": round(wall, 6),
        "utterances_per_minute": round((len(rows) / wall) * 60.0, 6) if wall > 0 else None,
        "audio_seconds_per_wall_second": round(total_duration / wall, 6) if wall > 0 else None,
        "total_audio_duration_seconds": round(total_duration, 6),
        "rows": rows,
        "monitor": monitor_summary(monitor_path) if monitor_path else None,
    }


def verify_piper_runtime() -> dict[str, Any]:
    gpu = require_single_visible_cuda(allowed_name_fragments=("A100",))
    tts_config = load_tts_config()
    piper_python = repo_resolve(tts_config["engine"]["environment"]) / "bin" / "python"
    code = "import onnxruntime as ort\nprint(ort.get_available_providers())\nassert 'CUDAExecutionProvider' in ort.get_available_providers()\n"
    completed = subprocess.run(
        [str(piper_python), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=piper_runtime_env(piper_python),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or "CUDAExecutionProvider unavailable")
    generation_config = load_generation_config(default_generation_config_path())
    text = verify_text_admission_inputs(generation_config)
    voice_root = repo_resolve(tts_config["voice"]["local_storage_dir"])
    required_voice_files = [
        voice_root / entry["path"]
        for entry in tts_config["voice"]["files"]
        if entry["role"] in {"model", "config", "model_card"}
    ]
    missing = [str(path) for path in required_voice_files if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    return {
        "gpu": gpu.to_dict(),
        "onnxruntime_providers": completed.stdout.strip(),
        "text_admission": text,
        "piper": {
            "engine_revision": tts_config["engine"]["revision"],
            "voice_revision": tts_config["voice"]["revision"],
            "native_sample_rate": tts_config["voice"]["native_sample_rate"],
            "final_sample_rate": tts_config["voice"]["final_asr_sample_rate"],
        },
    }


def run_worker_benchmark() -> dict[str, Any]:
    gpu = require_single_visible_cuda(allowed_name_fragments=("A100",))
    generation_config = load_generation_config(default_generation_config_path())
    audio_config = load_json(synthetic_audio_config_path())
    paths = audio_paths(generation_config)
    items = load_corpus_v2_tts_items(accepted_candidates_path(generation_config))[: int(audio_config["concurrency"]["benchmark_subset_rows"])]
    tts_config = load_tts_config()
    worker_counts = [int(value) for value in audio_config["concurrency"]["worker_counts"]]
    baseline_hashes: dict[str, str] | None = None
    results: list[dict[str, Any]] = []
    for workers in worker_counts:
        output_root = paths.benchmark_dir / f"workers-{workers}"
        monitor_path = paths.benchmark_dir / f"workers-{workers}.gpu.csv"
        result = render_items_concurrently(
            items=items,
            worker_count=workers,
            tts_config=tts_config,
            paths=paths,
            output_root=output_root,
            monitor_path=monitor_path,
            monitor_interval_seconds=float(audio_config["concurrency"]["monitor_interval_seconds"]),
            physical_selector=gpu.physical_selector,
        )
        hashes = {str(row["candidate_id"]): str(row["audio_sha256"]) for row in result["rows"]}
        parity = True if baseline_hashes is None else hashes == baseline_hashes
        if baseline_hashes is None:
            baseline_hashes = hashes
        valid = result["failed"] == 0 and result["successful"] == len(items) and parity
        results.append({key: value for key, value in result.items() if key != "rows"} | {"hash_parity_with_worker_1": parity, "valid": valid})
    selected = select_worker_count(results, threshold=float(audio_config["concurrency"]["selection_within_best_fraction"]))
    summary = {
        "schema_version": AUDIO_REPORT_SCHEMA_VERSION,
        "benchmark_version": "piper-worker-benchmark-v1",
        "subset_rows": len(items),
        "worker_results": results,
        "selected_worker_count": selected,
        "selection_policy": "smallest valid worker count within 5% of best valid throughput",
    }
    atomic_write_json(paths.benchmark_summary, summary)
    return summary


def select_worker_count(results: Sequence[dict[str, Any]], *, threshold: float = 0.95) -> int:
    valid = [row for row in results if row.get("valid") and row.get("utterances_per_minute") is not None]
    if not valid:
        raise RuntimeError("no valid worker-count benchmark result")
    best = max(float(row["utterances_per_minute"]) for row in valid)
    cutoff = best * threshold
    eligible = [row for row in valid if float(row["utterances_per_minute"]) >= cutoff]
    return min(int(row["worker_count"]) for row in eligible)


def run_full_synthesis() -> dict[str, Any]:
    gpu = require_single_visible_cuda(allowed_name_fragments=("A100",))
    generation_config = load_generation_config(default_generation_config_path())
    audio_config = load_json(synthetic_audio_config_path())
    paths = audio_paths(generation_config)
    benchmark = load_json(paths.benchmark_summary)
    worker_count = int(benchmark["selected_worker_count"])
    tts_config = load_tts_config()
    items = load_corpus_v2_tts_items(accepted_candidates_path(generation_config))
    result = render_items_concurrently(
        items=items,
        worker_count=worker_count,
        tts_config=tts_config,
        paths=paths,
        output_root=None,
        monitor_path=paths.gpu_monitor,
        monitor_interval_seconds=float(audio_config["concurrency"]["monitor_interval_seconds"]),
        physical_selector=gpu.physical_selector,
    )
    atomic_write_jsonl(paths.audio_manifest, result["rows"])
    summary = {key: value for key, value in result.items() if key != "rows"} | {
        "schema_version": AUDIO_REPORT_SCHEMA_VERSION,
        "synthesis_version": "corpus-v2-piper-synthesis-v1",
        "selected_worker_count": worker_count,
        "audio_manifest_sha256": sha256_file(paths.audio_manifest),
        "accepted_candidate_partition_sha256": sha256_file(accepted_candidates_path(generation_config)),
        "gpu": gpu.to_dict(),
    }
    atomic_write_json(paths.synthesis_summary, summary)
    if result["failed"]:
        raise RuntimeError(f"synthesis failures: {len(result['failures'])}")
    return summary


def validate_audio_manifest(*, require_status: str | None = None) -> tuple[dict[str, Any], int]:
    generation_config = load_generation_config(default_generation_config_path())
    audio_config = load_json(synthetic_audio_config_path())
    paths = audio_paths(generation_config)
    accepted_items = {item.candidate_id: item for item in load_corpus_v2_tts_items(accepted_candidates_path(generation_config))}
    manifest_rows = load_jsonl(paths.audio_manifest)
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    stats_rows: list[dict[str, Any]] = []
    thresholds = audio_config["waveform_thresholds"]
    expected = audio_config["expected_format"]
    tts_expected = audio_config["tts"]
    for row in manifest_rows:
        candidate_id = str(row.get("candidate_id", ""))
        item = accepted_items.get(candidate_id)
        if not item:
            issues.append({"reason": "unexpected_candidate", "candidate_id_hash": sha256_text(candidate_id)})
            continue
        if candidate_id in seen_ids:
            issues.append({"reason": "duplicate_candidate_id", "candidate_id_hash": sha256_text(candidate_id)})
        seen_ids.add(candidate_id)
        audio_path_text = str(row.get("audio_filepath", ""))
        if audio_path_text in seen_paths:
            issues.append({"reason": "duplicate_audio_path", "candidate_id_hash": sha256_text(candidate_id)})
        seen_paths.add(audio_path_text)
        path = Path(audio_path_text)
        if not path.exists():
            issues.append({"reason": "missing_audio", "candidate_id_hash": sha256_text(candidate_id)})
            continue
        try:
            stats = read_audio_stats(path)
        except Exception as exc:  # noqa: BLE001 - local report records reason
            issues.append({"reason": "malformed_audio", "candidate_id_hash": sha256_text(candidate_id), "detail": type(exc).__name__})
            continue
        if row.get("text") != item.target_text or row.get("target_text_sha256") != sha256_text(item.target_text):
            issues.append({"reason": "transcript_linkage_mismatch", "candidate_id_hash": sha256_text(candidate_id)})
        for field in ("source_id", "source_family_id", "utterance_family_id"):
            if row.get(field) != getattr(item, field):
                issues.append({"reason": f"{field}_mismatch", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.sha256 in seen_hashes:
            issues.append({"reason": "duplicate_audio_sha256", "candidate_id_hash": sha256_text(candidate_id)})
        seen_hashes.add(stats.sha256)
        if stats.channels != int(expected["channels"]):
            issues.append({"reason": "wrong_channels", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.sample_rate != int(expected["sample_rate"]):
            issues.append({"reason": "wrong_sample_rate", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.sample_width != int(expected["sample_width_bytes"]):
            issues.append({"reason": "wrong_sample_width", "candidate_id_hash": sha256_text(candidate_id)})
        if not float(audio_config["duration_bounds_seconds"]["minimum"]) <= stats.duration_seconds <= float(audio_config["duration_bounds_seconds"]["maximum"]):
            issues.append({"reason": "duration_out_of_bounds", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.peak_ratio < float(thresholds["minimum_peak_ratio"]):
            issues.append({"reason": "low_peak", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.rms_ratio < float(thresholds["minimum_rms_ratio"]):
            issues.append({"reason": "low_rms", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.active_frame_fraction < float(thresholds["minimum_active_frame_fraction"]):
            issues.append({"reason": "mostly_silent", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.clipping_fraction > float(thresholds["maximum_clipping_fraction"]):
            issues.append({"reason": "clipping", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.leading_silence_seconds > float(thresholds["maximum_leading_silence_seconds"]):
            issues.append({"reason": "leading_silence", "candidate_id_hash": sha256_text(candidate_id)})
        if stats.trailing_silence_seconds > float(thresholds["maximum_trailing_silence_seconds"]):
            issues.append({"reason": "trailing_silence", "candidate_id_hash": sha256_text(candidate_id)})
        if row.get("tts", {}).get("execution_provider") != tts_expected["required_execution_provider"]:
            issues.append({"reason": "tts_provider_mismatch", "candidate_id_hash": sha256_text(candidate_id)})
        for key, expected_value in (
            ("engine", tts_expected["engine"]),
            ("engine_revision", tts_expected["engine_revision"]),
            ("voice", tts_expected["voice"]),
            ("voice_repository", tts_expected["voice_repository"]),
            ("voice_revision", tts_expected["voice_revision"]),
        ):
            if row.get("tts", {}).get(key) != expected_value:
                issues.append({"reason": f"tts_{key}_mismatch", "candidate_id_hash": sha256_text(candidate_id)})
        if row.get("native_audio", {}).get("sample_rate") != int(tts_expected["native_sample_rate"]):
            issues.append({"reason": "native_sample_rate_mismatch", "candidate_id_hash": sha256_text(candidate_id)})
        stats_rows.append(asdict(stats))
    missing = sorted(set(accepted_items) - seen_ids)
    for candidate_id in missing:
        issues.append({"reason": "missing_candidate_audio", "candidate_id_hash": sha256_text(candidate_id)})
    status = "AUDIO_ACCEPTED" if not issues and len(manifest_rows) == len(accepted_items) else "AUDIO_REJECTED"
    durations = [row["duration_seconds"] for row in stats_rows]
    peak = [row["peak_ratio"] for row in stats_rows]
    rms = [row["rms_ratio"] for row in stats_rows]
    active = [row["active_frame_fraction"] for row in stats_rows]
    clipping = [row["clipping_fraction"] for row in stats_rows]
    leading = [row["leading_silence_seconds"] for row in stats_rows]
    trailing = [row["trailing_silence_seconds"] for row in stats_rows]
    summary = {
        "schema_version": AUDIO_REPORT_SCHEMA_VERSION,
        "validator_algorithm_version": AUDIO_VALIDATOR_VERSION,
        "corpus_id": CORPUS_ID,
        "status": status,
        "row_count": len(accepted_items),
        "manifest_rows": len(manifest_rows),
        "validated_audio_count": len(stats_rows),
        "audio_manifest_sha256": sha256_file(paths.audio_manifest),
        "accepted_candidate_partition_sha256": sha256_file(accepted_candidates_path(generation_config)),
        "failures_by_reason": dict(sorted(Counter(issue["reason"] for issue in issues).items())),
        "issues": issues,
        "distributions": {
            "duration_seconds": distribution(durations),
            "peak_ratio": distribution(peak),
            "rms_ratio": distribution(rms),
            "active_frame_fraction": distribution(active),
            "clipping_fraction": distribution(clipping),
            "leading_silence_seconds": distribution(leading),
            "trailing_silence_seconds": distribution(trailing),
        },
        "unique_audio_sha256_count": len(seen_hashes),
        "unique_audio_path_count": len(seen_paths),
        "voice_summary": {
            "voice_count": 1 if stats_rows else 0,
            "voice": tts_expected["voice"],
            "total_duration_seconds": round(sum(durations), 6),
        },
        "limitations": audio_config["limitations"],
    }
    atomic_write_json(paths.validation_report, summary)
    return summary, 0 if require_status is None or status == require_status else 1


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "minimum": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "p95": percentile(values, 0.95),
        "maximum": round(max(values), 6),
    }


def assert_public_audio_payload_safe(payload: Any) -> None:
    def walk(value: Any, key: str = "") -> None:
        if key in PUBLIC_FORBIDDEN_KEYS:
            raise ValueError(f"public audio payload contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, child_key)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            if any(marker in value for marker in PUBLIC_FORBIDDEN_VALUE_MARKERS):
                raise ValueError("public audio payload contains candidate ID or local path")

    walk(payload)


def build_audio_certificate_and_reports() -> dict[str, Any]:
    generation_config = load_generation_config(default_generation_config_path())
    paths = audio_paths(generation_config)
    validation = load_json(paths.validation_report)
    synthesis = load_json(paths.synthesis_summary)
    benchmark = load_json(paths.benchmark_summary)
    text_report = load_json(default_text_report_path())
    certificate_dir = REPO_ROOT / "docs/data-certificates"
    report_dir = REPO_ROOT / "docs/data-reports"
    certificate_path = certificate_dir / "sl-corpus-v2-gams-candidate-reservoir-v1-audio.json"
    report_json_path = report_dir / "0003-corpus-v2-acoustic-admission.json"
    report_md_path = report_dir / "0003-corpus-v2-acoustic-admission.md"
    certificate = {
        "schema_version": AUDIO_CERTIFICATE_SCHEMA_VERSION,
        "corpus_id": CORPUS_ID,
        "status": validation["status"],
        "accepted_text_partition_sha256": validation["accepted_candidate_partition_sha256"],
        "review_decision": text_report.get("review", {}).get("whole_file_decision"),
        "row_count": validation["row_count"],
        "audio_count": validation["validated_audio_count"],
        "audio_manifest_sha256": validation["audio_manifest_sha256"],
        "engine": {
            "name": "OHF-Voice/piper1-gpl",
            "revision": "b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6",
        },
        "voice": {
            "name": "sl_SI-artur-medium",
            "repository": "rhasspy/piper-voices",
            "revision": "217ddc79818708b078d0d14a8fae9608b9d77141",
            "count": validation["voice_summary"]["voice_count"],
            "total_duration_seconds": validation["voice_summary"]["total_duration_seconds"],
        },
        "audio_format": {
            "sample_rate": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
            "encoding": "signed-16-bit-pcm-wav",
        },
        "duration_distribution": validation["distributions"]["duration_seconds"],
        "waveform_validation": {
            "peak_ratio": validation["distributions"]["peak_ratio"],
            "rms_ratio": validation["distributions"]["rms_ratio"],
            "active_frame_fraction": validation["distributions"]["active_frame_fraction"],
            "clipping_fraction": validation["distributions"]["clipping_fraction"],
            "leading_silence_seconds": validation["distributions"]["leading_silence_seconds"],
            "trailing_silence_seconds": validation["distributions"]["trailing_silence_seconds"],
        },
        "duplicate_counts": {
            "audio_path_duplicates": validation["manifest_rows"] - validation["unique_audio_path_count"],
            "audio_sha256_duplicates": validation["validated_audio_count"] - validation["unique_audio_sha256_count"],
        },
        "failures_by_reason": validation["failures_by_reason"],
        "selected_worker_count": synthesis["selected_worker_count"],
        "concurrency_benchmark": [
            {
                "worker_count": row["worker_count"],
                "successful": row["successful"],
                "failed": row["failed"],
                "wall_time_seconds": row["wall_time_seconds"],
                "utterances_per_minute": row["utterances_per_minute"],
                "audio_seconds_per_wall_second": row["audio_seconds_per_wall_second"],
                "hash_parity_with_worker_1": row["hash_parity_with_worker_1"],
                "valid": row["valid"],
                "monitor": row["monitor"],
            }
            for row in benchmark["worker_results"]
        ],
        "gpu_utilization": synthesis["monitor"],
        "validator": {
            "algorithm_version": AUDIO_VALIDATOR_VERSION,
            "config_sha256": sha256_file(synthetic_audio_config_path()),
        },
        "limitations": [
            "This certificate is AUDIO_ACCEPTED or AUDIO_REJECTED only; it is never TRAINING_ELIGIBLE.",
            "The corpus has one Piper voice and no independent synthetic holdout.",
            "Waveform checks do not prove transcript correctness or natural prosody.",
            "TTS and audio validation do not authorize ASR scoring, selection, or model training.",
        ],
    }
    assert_public_audio_payload_safe(certificate)
    report_payload = {
        "schema_version": AUDIO_REPORT_SCHEMA_VERSION,
        "report": "corpus-v2-acoustic-admission",
        "certificate": certificate,
        "text_admission": {
            "status": text_report.get("validator", {}).get("status"),
            "accepted_count": text_report.get("review", {}).get("accepted_count"),
            "accepted_partition_sha256": text_report.get("accepted_partition", {}).get("sha256"),
        },
    }
    assert_public_audio_payload_safe(report_payload)
    atomic_write_json(certificate_path, certificate)
    atomic_write_json(report_json_path, report_payload)
    write_audio_markdown(report_md_path, report_payload)
    return {
        "certificate_path": str(certificate_path.relative_to(REPO_ROOT)),
        "certificate_sha256": sha256_file(certificate_path),
        "report_json_path": str(report_json_path.relative_to(REPO_ROOT)),
        "report_json_sha256": sha256_file(report_json_path),
        "report_markdown_path": str(report_md_path.relative_to(REPO_ROOT)),
        "report_markdown_sha256": sha256_file(report_md_path),
        "status": validation["status"],
        "audio_manifest_sha256": validation["audio_manifest_sha256"],
    }


def write_audio_markdown(path: Path, payload: dict[str, Any]) -> None:
    cert = payload["certificate"]
    duration = cert["duration_distribution"]
    gpu = cert["gpu_utilization"]
    lines = [
        "# Corpus-v2 Acoustic Admission",
        "",
        f"Status: `{cert['status']}`",
        "",
        "This privacy-safe report contains aggregate synthesis and waveform-validation evidence only. It does not include generated sentences, candidate IDs, audio paths, local paths, or reviewer identity.",
        "",
        "## Inputs",
        "",
        f"- Text status: `{payload['text_admission']['status']}`",
        f"- Accepted text rows: {payload['text_admission']['accepted_count']}",
        f"- Accepted text partition SHA256: `{payload['text_admission']['accepted_partition_sha256']}`",
        "",
        "## Audio",
        "",
        f"- Audio count: {cert['audio_count']} / {cert['row_count']}",
        f"- Audio manifest SHA256: `{cert['audio_manifest_sha256']}`",
        f"- Total duration seconds: {cert['voice']['total_duration_seconds']}",
        f"- Duration range seconds: {duration.get('minimum')} - {duration.get('maximum')}",
        f"- Selected worker count: {cert['selected_worker_count']}",
        "",
        "## GPU",
        "",
        f"- Monitor samples: {gpu.get('sample_count') if gpu else None}",
        f"- Mean utilization percent: {gpu.get('mean_utilization_percent') if gpu else None}",
        f"- Peak memory MiB: {gpu.get('peak_memory_mib') if gpu else None}",
        "",
        "## Limitations",
        "",
        "- This status is not `TRAINING_ELIGIBLE`.",
        "- No independent synthetic holdout exists.",
        "- The corpus remains single-voice synthetic audio.",
        "- Waveform validation does not prove transcript correctness or natural prosody.",
        "",
    ]
    atomic_write_text(path, "\n".join(lines))
