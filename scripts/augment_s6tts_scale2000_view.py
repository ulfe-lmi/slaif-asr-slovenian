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

from slaif_asr.s6tts_augmentation import (
    load_clean_manifest,
    load_profiles,
    load_s6_augmentation_config,
    planned_tasks,
    render_one,
    s6_aug_paths,
    summarize_augmented_view,
    summarize_stage,
    write_augmented_manifests,
    write_public_evidence,
)
from slaif_asr.tts import write_jsonl


def render_one_status(task):
    try:
        return {"ok": True, "row": render_one(task)}
    except Exception as exc:
        paths = task[0]
        clean_row = task[2]
        profile = task[4]
        return {
            "ok": False,
            "failure": {
                "safe_key": clean_row.get("safe_key"),
                "row_index": clean_row.get("row_index"),
                "text_hash": clean_row.get("text_hash"),
                "profile_id": profile.get("profile_id"),
                "reason": exc.__class__.__name__,
                "message": str(exc).replace(str(paths.run_root), "<RUN_ROOT>"),
            },
        }


def stage_paths(paths, name: str):
    root = paths.run_root / name
    return paths.__class__(
        run_root=root,
        audio_manifest=root / "audio-manifest.local.jsonl",
        provenance_manifest=root / "provenance.local.jsonl",
        validation=root / "audio-validation.local.json",
        summary=root / "summary.local.json",
    )


def write_failures(paths, failures: list[dict]) -> None:
    path = paths.run_root / "failures.local.jsonl"
    if not failures:
        path.unlink(missing_ok=True)
        return
    write_jsonl(path, failures)


def render_stage(clean_rows, profiles, paths, *, stage: str, overwrite: bool, workers: int) -> tuple[list[dict], list[dict]]:
    total = len(clean_rows) * len(profiles)
    rendered: list[dict] = []
    failures: list[dict] = []
    started = time.perf_counter()

    def emit(processed: int) -> None:
        elapsed = time.perf_counter() - started
        rate = processed / elapsed if elapsed else 0.0
        eta = (total - processed) / rate if rate else 0.0
        print(
            json.dumps(
                {
                    "stage": stage,
                    "processed": processed,
                    "total": total,
                    "workers": workers,
                    "percent": round(100.0 * processed / total, 3) if total else 100.0,
                    "elapsed_seconds": round(elapsed, 3),
                    "eta_seconds": round(eta, 3),
                    "rows_per_second": round(rate, 3),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    task_iter = (
        (paths, clean_index, clean_row, profile_index, profile, overwrite)
        for clean_index, clean_row, profile_index, profile in planned_tasks(clean_rows, profiles)
    )
    if workers <= 1:
        for processed, task in enumerate(task_iter, start=1):
            try:
                rendered.append(render_one(task))
            except Exception as exc:
                clean_row = task[2]
                profile = task[4]
                failures.append(
                    {
                        "safe_key": clean_row.get("safe_key"),
                        "row_index": clean_row.get("row_index"),
                        "text_hash": clean_row.get("text_hash"),
                        "profile_id": profile.get("profile_id"),
                        "reason": exc.__class__.__name__,
                        "message": str(exc).replace(str(paths.run_root), "<RUN_ROOT>"),
                    }
                )
            if processed == 1 or processed == total or processed % 500 == 0:
                emit(processed)
        return rendered, failures

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for processed, result in enumerate(pool.map(render_one_status, task_iter, chunksize=16), start=1):
            if result["ok"]:
                rendered.append(result["row"])
            else:
                failures.append(result["failure"])
            if processed == 1 or processed == total or processed % 500 == 0:
                emit(processed)
    rendered.sort(key=lambda row: (int(row["row_index"]), int(row["profile_index"])))
    failures.sort(key=lambda row: (int(row.get("row_index") or -1), str(row.get("profile_id", ""))))
    return rendered, failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Build S6TTS transcript-preserving augmented views for scale-2000.")
    parser.add_argument("--stage", choices=["smoke", "sample", "full", "summarize"], required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--workers", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.runs_root is not None:
        os.environ["SLAIF_ASR_RUNS_ROOT"] = str(args.runs_root.expanduser().resolve())
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    config = load_s6_augmentation_config()
    profiles = load_profiles()
    clean_rows = load_clean_manifest(config)
    paths = s6_aug_paths(config)

    if args.stage == "smoke":
        smoke_paths = stage_paths(paths, "smoke")
        selected = clean_rows[:2]
        rows, failures = render_stage(selected, profiles, smoke_paths, stage="s6tts-augmentation-smoke", overwrite=True, workers=min(args.workers, len(selected) * len(profiles)))
        write_failures(smoke_paths, failures)
        write_augmented_manifests(smoke_paths, rows)
        summary = summarize_stage(smoke_paths, expected=len(selected) * len(profiles), status="S6TTS_AUGMENTATION_SMOKE_PASSED" if not failures else "S6TTS_AUGMENTATION_SMOKE_FAILED")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    if args.stage == "sample":
        sample_paths = stage_paths(paths, "sample")
        selected = clean_rows[: args.sample_count]
        rows, failures = render_stage(selected, profiles, sample_paths, stage="s6tts-augmentation-sample", overwrite=True, workers=min(args.workers, len(selected) * len(profiles)))
        write_failures(sample_paths, failures)
        write_augmented_manifests(sample_paths, rows)
        summary = summarize_stage(sample_paths, expected=len(selected) * len(profiles), status="S6TTS_AUGMENTATION_SAMPLE_PASSED" if not failures else "S6TTS_AUGMENTATION_SAMPLE_FAILED")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0

    if args.stage == "full":
        rows, failures = render_stage(clean_rows, profiles, paths, stage="s6tts-augmentation-full", overwrite=args.overwrite, workers=args.workers)
        write_failures(paths, failures)
        write_augmented_manifests(paths, rows)
        summary = summarize_augmented_view(config)
        print(json.dumps({"status": summary["status"], "actual_augmented_files": summary["actual_augmented_files"], "failures": summary["synthesis_or_augmentation_failure_count"]}, sort_keys=True))
        return 0

    summary = summarize_augmented_view(config)
    write_public_evidence(summary)
    print(json.dumps({"status": summary["status"], "certificate": "docs/data-certificates/sl-corpus-v4-s6tts-augmented-view-v1.json", "report": "docs/data-reports/0021-s6tts-vintage-augmentation-admission.json"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
