#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.artur_controller_dev import (
    checkpoint_availability,
    read_json,
    synthetic_loss_rows,
    write_curve_reports,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the ARTUR controller-dev curve report from available local checkpoints.")
    parser.add_argument("--certificate", type=Path, default=Path("docs/data-certificates/artur-controller-dev-v1.json"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("runs/experiments/scale2000-decoder-joint-rnnt-v1/scale2000_augmented_decoder_joint_rnnt/checkpoints"),
    )
    parser.add_argument("--experiment-0017", type=Path, default=Path("docs/experiments/0017-scale2000-decoder-joint-rnnt-directional.json"))
    parser.add_argument("--json-report", type=Path, default=Path("docs/experiments/0018-artur-controller-dev-real-validation-curve.json"))
    parser.add_argument("--md-report", type=Path, default=Path("docs/experiments/0018-artur-controller-dev-real-validation-curve.md"))
    args = parser.parse_args()

    try:
        certificate = read_json(args.certificate)
        checkpoints = checkpoint_availability(args.checkpoint_root)
        report = write_curve_reports(
            certificate=certificate,
            checkpoint_rows=checkpoints,
            synthetic_rows=synthetic_loss_rows(args.experiment_0017),
            json_path=args.json_report,
            md_path=args.md_report,
        )
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "partition_id": report["partition_id"],
                "checkpoint_count_available": sum(1 for row in report["checkpoint_availability"] if row["available"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
