#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.artur_controller_dev import build_controller_dev_partition


def git_head() -> str:
    completed = subprocess.run(["git", "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the privacy-safe ARTUR controller-dev partition certificate.")
    parser.add_argument("--real-gates-config", type=Path, default=Path("configs/evaluation/real_gates.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/evaluation-gates"))
    parser.add_argument("--certificate", type=Path, default=Path("docs/data-certificates/artur-controller-dev-v1.json"))
    parser.add_argument("--artur-metadata", type=Path, default=Path("docs/evaluation-gates/artur-j-public-gate-v1.metadata.json"))
    parser.add_argument("--fleurs-metadata", type=Path, default=Path("docs/evaluation-gates/fleurs-sl-si-test-full-v2.metadata.json"))
    parser.add_argument(
        "--protected-index",
        type=Path,
        action="append",
        default=[Path("runs/data-quality/protected/artur-j.hash-index.json"), Path("runs/data-quality/protected/fleurs-v2.hash-index.json")],
    )
    parser.add_argument("--required-count", type=int, default=256)
    parser.add_argument("--max-segments-per-recording", type=int, default=12)
    args = parser.parse_args()

    try:
        certificate = build_controller_dev_partition(
            real_gates_config=args.real_gates_config,
            output_root=args.output_root,
            certificate_path=args.certificate,
            artur_metadata_path=args.artur_metadata,
            fleurs_metadata_path=args.fleurs_metadata,
            protected_index_paths=args.protected_index,
            repository_commit=git_head(),
            required_count=args.required_count,
            max_segments_per_recording=args.max_segments_per_recording,
        )
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "partition_id": certificate["partition_id"],
                "row_count": certificate["row_count"],
                "manifest_sha256": certificate["manifest_sha256"],
                "audio_duration_seconds": certificate["audio_duration_seconds"]["total"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
