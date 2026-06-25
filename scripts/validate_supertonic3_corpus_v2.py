#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.supertonic3_tts import load_supertonic_config, summarize_supertonic_audio, validate_supertonic_audio


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize Supertonic 3 corpus-v2 audio.")
    parser.add_argument("--config", type=Path, default=Path("configs/tts/supertonic3_sl_multivoice_v1.json"))
    parser.add_argument("--stage", required=True, choices=["validate", "summarize"])
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config = load_supertonic_config(args.config)
    if args.stage == "validate":
        result = validate_supertonic_audio(config, progress_interval_seconds=args.progress_interval_seconds)
    else:
        result = summarize_supertonic_audio(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.stage == "validate" and result.get("status") != "AUDIO_ACCEPTED":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
