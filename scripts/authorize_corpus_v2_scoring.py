#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.acoustic_quality import build_scoring_authorization


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize corpus-v2 ASR scoring after text and audio admission.")
    parser.add_argument("--require-status", choices=("SCORING_AUTHORIZED", "SCORING_BLOCKED"), default=None)
    args = parser.parse_args()
    payload, return_code = build_scoring_authorization(require_status=args.require_status)
    certificate = payload["certificate"]
    summary = {
        "candidate_audio_manifest_sha256": certificate["candidate_source"]["audio_manifest_sha256"],
        "holdout_audio_manifest_sha256": certificate["synthetic_holdout"]["audio_manifest_sha256"],
        "status": certificate["status"],
        "text_overlap_counts": certificate["cross_partition_overlap_counts"]["text_fingerprint"],
        "audio_overlap_counts": certificate["cross_partition_overlap_counts"]["audio"],
        "protected_overlap_counts": certificate["protected_gate_overlap_counts"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
