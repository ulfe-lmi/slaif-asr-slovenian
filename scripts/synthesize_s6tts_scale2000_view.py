#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.s6tts_tts import (
    CONFIG_PATH,
    EXPECTED_ROWS,
    load_s6_config,
    load_scale2000_rows,
    sample_rows,
    s6_paths,
    smoke_rows,
    summarize_local_view,
    synthesize_one,
    synthesize_worker_status,
    validate_public_payload,
    write_manifest_pair,
    write_public_evidence,
)
from slaif_asr.tts import atomic_write_json


def render_stage(rows, paths, *, stage: str, overwrite: bool = False, workers: int = 1) -> tuple[list[dict], list[dict]]:
    rendered = []
    failures = []
    started = time.perf_counter()
    total = len(rows)

    def emit(idx: int) -> None:
        elapsed = time.perf_counter() - started
        rate = idx / elapsed if elapsed else 0.0
        print(
            json.dumps(
                {
                    "stage": stage,
                    "processed": idx,
                    "total": total,
                    "workers": workers,
                    "percent": round(100.0 * idx / total, 3) if total else 100.0,
                    "elapsed_seconds": round(elapsed, 3),
                    "rows_per_second": round(rate, 3),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    if workers <= 1:
        for idx, row in enumerate(rows, start=1):
            result = synthesize_worker_status((paths, row, overwrite))
            if result["ok"]:
                rendered.append(result["row"])
            else:
                failures.append(result["failure"])
            if idx == 1 or idx == total or idx % 100 == 0:
                emit(idx)
        return rendered, failures

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        iterator = pool.map(synthesize_worker_status, [(paths, row, overwrite) for row in rows], chunksize=8)
        for idx, result in enumerate(iterator, start=1):
            if result["ok"]:
                rendered.append(result["row"])
            else:
                failures.append(result["failure"])
            if idx == 1 or idx == total or idx % 100 == 0:
                emit(idx)
    return rendered, failures


def write_failures(paths, failures: list[dict]) -> None:
    path = paths.run_root / "failures.local.jsonl"
    if not failures:
        path.unlink(missing_ok=True)
        return
    from slaif_asr.tts import write_jsonl

    write_jsonl(path, failures)


def stage_paths(paths, name: str):
    root = paths.run_root / name
    return paths.__class__(
        source_dir=paths.source_dir,
        build_dir=paths.build_dir,
        cli_path=paths.cli_path,
        runtime_ini=paths.runtime_ini,
        run_root=root,
        audio_manifest=root / "audio-manifest.local.jsonl",
        provenance_manifest=root / "provenance.local.jsonl",
        validation=root / "audio-validation.local.json",
        summary=root / "summary.local.json",
        logs_dir=root / "logs",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize and summarize the scale-2000 S6TTS clean view.")
    parser.add_argument("--stage", choices=["smoke", "sample", "full", "summarize"], required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--runs-root", type=Path, default=None)
    args = parser.parse_args()
    if args.runs_root is not None:
        os.environ["SLAIF_ASR_RUNS_ROOT"] = str(args.runs_root.expanduser().resolve())
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    config = load_s6_config(CONFIG_PATH)
    paths = s6_paths(config)
    if not paths.cli_path.exists():
        raise FileNotFoundError(paths.cli_path)
    if not paths.runtime_ini.exists():
        raise FileNotFoundError(paths.runtime_ini)

    if args.stage == "smoke":
        smoke = stage_paths(paths, "smoke")
        rows, failures = render_stage(smoke_rows(), smoke, stage="s6tts-smoke", overwrite=True, workers=min(args.workers, len(smoke_rows())))
        write_failures(smoke, failures)
        write_manifest_pair(smoke, rows)
        summary = summarize_local_view_for_stage(config, smoke, expected=len(smoke_rows()), status="S6TTS_SMOKE_PASSED" if not failures else "S6TTS_SMOKE_FAILED")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    scale_rows = load_scale2000_rows(config)
    if args.stage == "sample":
        sample = stage_paths(paths, "sample")
        selected = sample_rows(scale_rows, args.sample_count)
        rows, failures = render_stage(selected, sample, stage="s6tts-sample", overwrite=True, workers=min(args.workers, len(selected)))
        write_failures(sample, failures)
        write_manifest_pair(sample, rows)
        summary = summarize_local_view_for_stage(config, sample, expected=len(selected), status="S6TTS_SAMPLE_PASSED" if not failures else "S6TTS_SAMPLE_FAILED")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    if args.stage == "full":
        rows, failures = render_stage(scale_rows, paths, stage="s6tts-full", overwrite=False, workers=args.workers)
        write_failures(paths, failures)
        write_manifest_pair(paths, rows)
        summary = summarize_local_view(config, paths)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    summary = summarize_local_view(config, paths)
    certificate = Path("docs/data-certificates/sl-corpus-v4-s6tts-clean-view-v1.json")
    report_json = Path("docs/data-reports/0020-s6tts-vintage-clean-view-admission.json")
    report_md = Path("docs/data-reports/0020-s6tts-vintage-clean-view-admission.md")
    write_public_evidence(summary, report_json=report_json, report_md=report_md, certificate_json=certificate)
    print(json.dumps({"status": summary["status"], "certificate": str(certificate), "report": str(report_json)}, sort_keys=True))
    return 0


def summarize_local_view_for_stage(config: dict, paths, *, expected: int, status: str) -> dict:
    rows = []
    with paths.audio_manifest.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    failures_path = paths.run_root / "failures.local.jsonl"
    failures = []
    if failures_path.exists():
        with failures_path.open("r", encoding="utf-8") as fp:
            failures = [json.loads(line) for line in fp if line.strip()]
    summary = {
        "schema_version": "1.0",
        "status": status,
        "corpus_id": config["inputs"]["corpus_id"],
        "view_id": "s6tts-stage-check",
        "rows": len(rows),
        "expected_rows": expected,
        "synthesis_failure_count": len(failures),
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 2,
        "duplicate_path_count": len(rows) - len({row["audio_relative_path"] for row in rows}),
        "duplicate_audio_hash_count": len(rows) - len({row["audio_sha256"] for row in rows}),
    }
    validate_public_payload(summary)
    atomic_write_json(paths.summary, summary)
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
