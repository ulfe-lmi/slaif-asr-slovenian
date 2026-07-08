#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.artur_controller_dev import watcher_contract_valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the future ARTUR controller-dev second-GPU watcher contract.")
    parser.add_argument("--training-gpu", required=True)
    parser.add_argument("--evaluation-gpu", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        watcher_contract_valid(args.training_gpu, args.evaluation_gpu, args.checkpoint_dir, args.metrics_dir)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    if not args.check_only:
        print("watch mode is intentionally not implemented in this governance PR", file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {
                "status": "contract_valid",
                "training_gpu": args.training_gpu,
                "evaluation_gpu": args.evaluation_gpu,
                "atomic_marker": "checkpoint_dir/round-N.complete",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
