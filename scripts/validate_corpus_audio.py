#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.acoustic_quality import validate_audio_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate corpus-v2 synthetic audio before acoustic admission.")
    parser.add_argument("--require-status", choices=("AUDIO_ACCEPTED", "AUDIO_REJECTED"), default=None)
    args = parser.parse_args()

    payload, return_code = validate_audio_manifest(require_status=args.require_status)
    summary = {
        "audio_manifest_sha256": payload["audio_manifest_sha256"],
        "failures_by_reason": payload["failures_by_reason"],
        "final_audio_status": payload["status"],
        "row_count": payload["row_count"],
        "validated_audio_count": payload["validated_audio_count"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
