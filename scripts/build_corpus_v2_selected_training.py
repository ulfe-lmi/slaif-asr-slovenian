#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.corpus_v2_selection import SELECTED_CERTIFICATE_STATUS, build_selected_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a corpus-v2 selected-training manifest from candidate scoring output.")
    parser.add_argument("--target-hard", type=int, default=120)
    parser.add_argument("--target-control", type=int, default=40)
    parser.add_argument("--require-status", default=None)
    args = parser.parse_args()
    payload, return_code = build_selected_training(
        target_hard=args.target_hard,
        target_control=args.target_control,
        require_status=args.require_status or SELECTED_CERTIFICATE_STATUS,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
