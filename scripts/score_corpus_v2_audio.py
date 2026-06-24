#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.corpus_v2_scoring import (
    SCORING_AUTHORIZED,
    score_corpus_role,
    summarize_scoring,
    verify_all_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score corpus-v2 synthetic audio with the untouched Nemotron base.")
    parser.add_argument("--stage", choices=("verify", "score", "summarize"), required=True)
    parser.add_argument("--corpus-role", choices=("synthetic_candidate", "synthetic_holdout"))
    parser.add_argument("--require-authorization", default=None)
    args = parser.parse_args()

    if args.stage == "verify":
        payload = verify_all_inputs(require_authorization=args.require_authorization or SCORING_AUTHORIZED, check_gpu=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.stage == "score":
        if not args.corpus_role:
            parser.error("--corpus-role is required for --stage score")
        payload = score_corpus_role(args.corpus_role)
        print(json.dumps({"status": payload["status"], "corpus_role": args.corpus_role, "rows": payload["prediction_count"]}, indent=2, sort_keys=True))
        return 0
    if args.stage == "summarize":
        payload = summarize_scoring()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    parser.error(f"unsupported stage: {args.stage}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
