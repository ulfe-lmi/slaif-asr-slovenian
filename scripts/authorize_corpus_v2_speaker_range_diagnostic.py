#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.batched_streaming import file_sha256
from slaif_asr.speaker_range_augmentation import (
    DIAGNOSTIC_STATUS,
    SPEAKER_RANGE_CERTIFICATE_PATH,
    build_speaker_range_certificate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize the corpus-v2 speaker-range diagnostic.")
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--selected-certificate", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--augmentation-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--require-status", required=True)
    args = parser.parse_args()

    certificate = build_speaker_range_certificate(
        experiment_config_path=args.experiment_config,
        selected_certificate_path=args.selected_certificate,
        baseline_report_path=args.baseline_report,
        augmentation_config_path=args.augmentation_config,
        work_order_id=args.work_order_id,
    )
    status = certificate.get("status")
    result = {
        "status": status,
        "certificate": str(SPEAKER_RANGE_CERTIFICATE_PATH.relative_to(Path.cwd())),
        "certificate_sha256": file_sha256(SPEAKER_RANGE_CERTIFICATE_PATH),
        "work_order_id": certificate.get("work_order_id"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if status != args.require_status:
        return 1
    if status != DIAGNOSTIC_STATUS:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
