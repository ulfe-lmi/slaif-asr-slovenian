#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.corpus_v2_training import (
    DIAGNOSTIC_STATUS,
    build_diagnostic_certificate,
    file_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize the corpus-v2 prompt-column diagnostic as DIAGNOSTIC_ONLY.")
    parser.add_argument("--selected-certificate", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--work-order-id", required=True)
    parser.add_argument("--require-status", required=True)
    args = parser.parse_args()

    certificate = build_diagnostic_certificate(
        args.experiment_config,
        selected_certificate_path=args.selected_certificate,
        work_order_id=args.work_order_id,
    )
    status = str(certificate.get("status", ""))
    result = {
        "status": status,
        "certificate": "docs/data-certificates/sl-corpus-v2-prompt-column-diagnostic-v1.json",
        "certificate_sha256": file_sha256(Path("docs/data-certificates/sl-corpus-v2-prompt-column-diagnostic-v1.json")),
        "work_order_id": certificate.get("work_order_id"),
        "experiment_config_sha256": certificate.get("experiment_config_sha256"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.require_status != DIAGNOSTIC_STATUS:
        raise SystemExit(f"this script can only require {DIAGNOSTIC_STATUS}")
    return 0 if status == args.require_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
