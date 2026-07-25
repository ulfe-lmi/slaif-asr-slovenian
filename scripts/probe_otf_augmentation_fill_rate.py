#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from slaif_asr.config import REPO_ROOT
from slaif_asr.otf_augmentation_dataset import (
    EXPECTED_AUGMENTATION_KEY,
    EXPECTED_CORPUS_ID,
    EXPECTED_WORKERS,
    OFFLINE_REPLAY_STATUS,
    classify_fill_rate,
    load_otf_config,
    load_scale2000_clean_records,
    mark_rate_passes,
    run_determinism_checks,
    run_fill_rate_benchmark,
    select_source_record,
    validate_benchmark_result,
    virtual_augmentation_spec,
)
from slaif_asr.real_eval import atomic_write_json, atomic_write_text


DEFAULT_CONFIG = REPO_ROOT / "configs" / "data_loading" / "otf-augmentation-scale2000-fill-rate-v1.json"


def runs_root() -> Path:
    override = os.environ.get("SLAIF_ASR_RUNS_ROOT")
    return Path(override) if override else REPO_ROOT / "runs"


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def cpu_inventory() -> dict[str, Any]:
    model = "unknown"
    physical_cores: set[tuple[str, str]] = set()
    physical_id = "0"
    core_id = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines() + [""]:
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
            elif line.startswith("physical id"):
                physical_id = line.split(":", 1)[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":", 1)[1].strip()
            elif not line and core_id:
                physical_cores.add((physical_id, core_id))
                core_id = ""
    logical = os.cpu_count() or 1
    physical = len(physical_cores) or logical
    available_ram = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    return {
        "cpu_model": model,
        "physical_cpu_count": physical,
        "logical_cpu_count": logical,
        "available_ram_gib_at_start": round(available_ram / (1024**3), 3),
    }


def storage_description(root: Path) -> str:
    try:
        source = subprocess.check_output(
            ["findmnt", "-n", "-o", "SOURCE", "-T", str(root)],
            text=True,
        ).strip()
        rotational = subprocess.check_output(
            ["lsblk", "-n", "-o", "ROTA", source],
            text=True,
        ).splitlines()[0].strip()
        return "non-rotational block storage" if rotational == "0" else "rotational block storage"
    except (OSError, subprocess.CalledProcessError):
        return "storage type not reported"


def verify_inputs(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any]]:
    config, profiles = load_otf_config(config_path, repository_root=REPO_ROOT)
    records = load_scale2000_clean_records(config, runs_root=runs_root())
    voices = Counter(record.voice for record in records)
    print(
        json.dumps(
            {
                "status": "PASSED",
                "corpus_id": EXPECTED_CORPUS_ID,
                "clean_records": len(records),
                "semantic_rows": int(config["data_source"]["semantic_rows"]),
                "voices": len(voices),
                "profiles": len(profiles),
                "raw_text_emitted": False,
                "local_paths_emitted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return config, profiles, records


def local_summary_path(config: dict[str, Any]) -> Path:
    return runs_root() / str(config["local_output"])


def stage_benchmark(config_path: Path) -> dict[str, Any]:
    config, profiles, records = verify_inputs(config_path)
    augmentation = config["augmentation"]
    pipeline = config["pipeline"]
    benchmark = config["benchmark"]
    thresholds = config["thresholds"]
    started = time.perf_counter()

    print(
        json.dumps(
            {
                "event": "determinism_start",
                "examples": int(benchmark["determinism_examples"]),
                "workers": int(pipeline["num_workers"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    determinism = run_determinism_checks(
        records,
        profiles,
        example_count=int(benchmark["determinism_examples"]),
        augmentation_key=str(augmentation["otf_aug_random_key"]),
        algorithm_version=str(augmentation["algorithm_version"]),
    )
    print(
        json.dumps(
            {
                "event": "determinism_complete",
                "status": determinism["status"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rate_results: list[dict[str, Any]] = []
    profile_distribution: Counter[str] = Counter()
    rates = [float(value) for value in benchmark["target_examples_per_second"]]
    for rate_index, rate in enumerate(rates):
        exposure_offset = (rate_index + 1) * 1_000_000
        total_examples = (
            int(benchmark["warmup_microbatches"]) + int(benchmark["measured_microbatches"])
        ) * int(pipeline["physical_microbatch"])
        for exposure_id in range(exposure_offset, exposure_offset + total_examples):
            record = select_source_record(
                records,
                exposure_id,
                augmentation_key=str(augmentation["otf_aug_random_key"]),
            )
            spec = virtual_augmentation_spec(
                record,
                exposure_id,
                profiles,
                augmentation_key=str(augmentation["otf_aug_random_key"]),
                algorithm_version=str(augmentation["algorithm_version"]),
            )
            profile_distribution[spec.profile_id] += 1

        print(
            json.dumps(
                {
                    "event": "rate_benchmark_start",
                    "target_examples_per_second": rate,
                    "workers": int(pipeline["num_workers"]),
                    "measured_microbatches": int(benchmark["measured_microbatches"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        result = run_fill_rate_benchmark(
            records,
            profiles,
            target_examples_per_second=rate,
            num_workers=int(pipeline["num_workers"]),
            physical_microbatch=int(pipeline["physical_microbatch"]),
            warmup_microbatches=int(benchmark["warmup_microbatches"]),
            measured_microbatches=int(benchmark["measured_microbatches"]),
            prefetch_microbatches=int(pipeline["prefetch_microbatches"]),
            ready_wait_threshold_ms=float(pipeline["ready_wait_threshold_ms"]),
            exposure_offset=exposure_offset,
            augmentation_key=str(augmentation["otf_aug_random_key"]),
            algorithm_version=str(augmentation["algorithm_version"]),
        )
        rate_results.append(result)
        print(
            json.dumps(
                {
                    "event": "rate_benchmark_complete",
                    "target_examples_per_second": rate,
                    "fill_rate": round(float(result["fill_rate"]), 6),
                    "ready_microbatch_rate": round(float(result["microbatch_ready_rate"]), 6),
                    "p95_wait_ms": round(float(result["p95_consumer_wait_ms"]), 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    runtime = {
        **cpu_inventory(),
        "num_workers": EXPECTED_WORKERS,
        "storage_type": storage_description(runs_root()),
        "python_version": platform.python_version(),
        "pytorch": "not used",
        "audio_decode": "Python wave plus NumPy",
        "augmentation_backend": "existing NumPy/SciPy/audioop transcript-preserving transforms",
        "benchmark_wall_seconds": time.perf_counter() - started,
    }
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": "0030-otf-augmentation-fill-rate-scale2000",
        "benchmark_id": config["benchmark_id"],
        "status": "DIAGNOSTIC_ONLY",
        "classification": "EXPERIMENT_INVALID",
        "repository_commit": git_revision(),
        "data_source": {
            "corpus_id": EXPECTED_CORPUS_ID,
            "source_clean_rows": int(config["data_source"]["semantic_rows"]),
            "source_clean_files": len(records),
            "fixed_text_sha256": config["data_source"]["fixed_text_sha256"],
            "all_views_sha256": config["data_source"]["all_views_sha256"],
            "source_audio_hash_policy": "manifest SHA256 per clean source; paths remain local",
            "source_type": "synthetic_tts_clean",
        },
        "augmentation": {
            "augmentation_key": augmentation["otf_aug_random_key"],
            "algorithm_version": augmentation["algorithm_version"],
            "profile_config_sha256": augmentation["profile_config_sha256"],
            "profile_count": len(profiles),
            "profile_sampling_method": augmentation["profile_sampling"],
            "profile_distribution": dict(sorted(profile_distribution.items())),
            "offline_replay_status": OFFLINE_REPLAY_STATUS,
            "disk_output": "none",
            "transcript_policy": "unchanged",
        },
        "pipeline": {
            "num_workers": int(pipeline["num_workers"]),
            "worker_type": pipeline["worker_type"],
            "physical_microbatch": int(pipeline["physical_microbatch"]),
            "gradient_accumulation": int(pipeline["gradient_accumulation"]),
            "effective_batch": int(pipeline["effective_batch"]),
            "prefetch_microbatches": int(pipeline["prefetch_microbatches"]),
            "warmup_microbatches": int(benchmark["warmup_microbatches"]),
            "measured_microbatches": int(benchmark["measured_microbatches"]),
            "measured_examples_per_rate": int(benchmark["measured_microbatches"])
            * int(pipeline["physical_microbatch"]),
        },
        "primary_target_examples_per_second": float(
            benchmark["primary_target_examples_per_second"]
        ),
        "thresholds": thresholds,
        "determinism": determinism,
        "rates": rate_results,
        "runtime": runtime,
        "limitations": [
            "The virtual stream reuses the admitted profile family but does not exactly replay the offline round and source-voice assignment.",
            "The consumer is a timed CPU simulator; no GPU kernels or model training ran.",
            "Results describe this host, clean-WAV storage, bounded queue depth, and three-worker configuration.",
            "SpecAugment is outside this waveform-only fill-rate probe.",
        ],
        "safety": {
            "training_started": False,
            "evaluation_started": False,
            "wer_cer_computed": False,
            "scale8000_used": False,
            "s6tts_used": False,
            "real_speech_used": False,
            "generated_audio_written": False,
            "raw_text_public": False,
            "local_paths_public": False,
        },
    }
    mark_rate_passes(payload, thresholds)
    payload["classification"] = classify_fill_rate(payload, thresholds)
    validate_benchmark_result(payload)
    atomic_write_json(local_summary_path(config), payload)
    print(
        json.dumps(
            {
                "status": "PASSED",
                "classification": payload["classification"],
                "primary_target_examples_per_second": payload[
                    "primary_target_examples_per_second"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Experiment 0030: Scale-2000 On-the-Fly Augmentation Fill-Rate Probe",
        "",
        f"- Classification: `{payload['classification']}`",
        "- Status: `DIAGNOSTIC_ONLY`",
        "- Training started: no",
        "- Evaluation started: no",
        "- WER/CER computed: no",
        "",
        "## Design",
        "",
        "Three spawned CPU workers decoded clean scale-2000 PCM WAVs, selected one of the existing 11 transcript-preserving profiles with a content-keyed HMAC identity, transformed the waveform in memory, and delivered physical microbatches of two through a bounded prefetch queue. The consumer slept at fixed target rates to simulate Surface08-style GPU demand.",
        "",
        f"- Source clean files: {payload['data_source']['source_clean_files']}",
        f"- Semantic rows: {payload['data_source']['source_clean_rows']}",
        f"- Workers: {payload['pipeline']['num_workers']} spawned processes",
        f"- Queue depth: {payload['pipeline']['prefetch_microbatches']} microbatches",
        f"- Augmentation key: `{payload['augmentation']['augmentation_key']}`",
        f"- Offline replay: `{payload['augmentation']['offline_replay_status']}`",
        "- Augmented WAVs written: none",
        "",
        "## Fill-Rate Results",
        "",
        "| Target examples/s | Prepared examples/s | Fill rate | Ready microbatch rate | Underruns | p50 wait ms | p95 wait ms | p99 wait ms | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rates"]:
        lines.append(
            "| {target} | {prepared} | {fill} | {ready} | {underruns} | {p50} | {p95} | {p99} | {passed} |".format(
                target=fmt(row["target_examples_per_second"], 1),
                prepared=fmt(row["prepared_examples_per_second"]),
                fill=fmt(row["fill_rate"], 6),
                ready=fmt(row["microbatch_ready_rate"], 6),
                underruns=int(row["underrun_count"]),
                p50=fmt(row["p50_consumer_wait_ms"]),
                p95=fmt(row["p95_consumer_wait_ms"]),
                p99=fmt(row["p99_consumer_wait_ms"]),
                passed="yes" if row["pass"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "Prepared examples/s estimates three-worker transform capacity from aggregate worker compute time. Fill rate is `1 - consumer_wait / total_consumer_time`. A microbatch is ready when the ordered item can be acquired without the queue becoming empty.",
            "",
            "## Determinism",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| Same virtual exposure, same worker count | {'passed' if payload['determinism']['same_virtual_exposure_same_worker_count'] else 'failed'} |",
            f"| Same virtual exposure after restart | {'passed' if payload['determinism']['same_virtual_exposure_after_restart'] else 'failed'} |",
            f"| Worker order independent | {'passed' if payload['determinism']['worker_order_independent'] else 'failed'} |",
            f"| Worker count independent | {'passed' if payload['determinism']['worker_count_independent'] else 'failed'} |",
            f"| Augmentation key recorded | {'passed' if payload['determinism']['augmentation_key_recorded'] else 'failed'} |",
            f"| Transcript preserved | {'passed' if payload['determinism']['transcript_preserved'] else 'failed'} |",
            "",
            "## Runtime",
            "",
            f"- CPU: {payload['runtime']['cpu_model']}",
            f"- Physical/logical CPU count: {payload['runtime']['physical_cpu_count']} / {payload['runtime']['logical_cpu_count']}",
            f"- Storage: {payload['runtime']['storage_type']}",
            f"- Available RAM at start: {payload['runtime']['available_ram_gib_at_start']} GiB",
            f"- Python: {payload['runtime']['python_version']}",
            f"- Audio decode: {payload['runtime']['audio_decode']}",
            f"- Augmentation backend: {payload['runtime']['augmentation_backend']}",
            f"- Benchmark wall time: {fmt(payload['runtime']['benchmark_wall_seconds'])} seconds",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in payload["limitations"])
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- No training or model evaluation ran.",
            "- No WER/CER was computed.",
            "- No scale-8000, S6TTS, or real speech was used.",
            "- No generated audio, checkpoints, predictions, raw transcripts, local manifests, or local paths are committed.",
            "",
        ]
    )
    return "\n".join(lines)


def stage_summarize(config_path: Path) -> dict[str, Any]:
    config, _profiles = load_otf_config(config_path, repository_root=REPO_ROOT)
    summary = local_summary_path(config)
    if not summary.is_file():
        raise FileNotFoundError("local benchmark summary is unavailable")
    with summary.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    validate_benchmark_result(payload)
    public = config["public_outputs"]
    atomic_write_json(REPO_ROOT / str(public["json"]), payload)
    atomic_write_text(REPO_ROOT / str(public["markdown"]), markdown_report(payload))
    print(
        json.dumps(
            {
                "status": "PASSED",
                "classification": payload["classification"],
                "raw_text_emitted": False,
                "local_paths_emitted": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe deterministic in-memory augmentation producer fill rate."
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("verify-inputs", "benchmark", "summarize"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if args.stage == "verify-inputs":
        verify_inputs(config_path)
    elif args.stage == "benchmark":
        stage_benchmark(config_path)
    else:
        stage_summarize(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
