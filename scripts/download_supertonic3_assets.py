#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.supertonic3_tts import download_assets, load_supertonic_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned Supertonic 3 model assets.")
    parser.add_argument("--config", type=Path, default=Path("configs/tts/supertonic3_sl_multivoice_v1.json"))
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    config = load_supertonic_config(args.config)
    result = download_assets(config, args.revision)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
