#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.acoustic_quality import (
    build_audio_certificate_and_reports,
    run_full_synthesis,
    run_worker_benchmark,
    verify_piper_runtime,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize corpus-v2 accepted text with external Piper.")
    parser.add_argument(
        "--stage",
        choices=("verify", "benchmark-workers", "synthesize", "summarize"),
        required=True,
    )
    parser.add_argument(
        "--corpus-role",
        choices=("synthetic_candidate", "synthetic_holdout"),
        default="synthetic_candidate",
        help="Corpus partition role to synthesize or summarize.",
    )
    args = parser.parse_args()

    if args.stage == "verify":
        payload = verify_piper_runtime(args.corpus_role)
    elif args.stage == "benchmark-workers":
        payload = run_worker_benchmark(args.corpus_role)
    elif args.stage == "synthesize":
        payload = run_full_synthesis(args.corpus_role)
    else:
        payload = build_audio_certificate_and_reports(args.corpus_role)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
