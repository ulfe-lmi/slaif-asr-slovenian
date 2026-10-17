#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_surface08_scale8000_otf_parametric_voiceaug_v1 import (
    DEFAULT_CONFIG,
    stage_audit,
)
from slaif_asr.config import REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local-only ParametricVoiceAug v1 waveform audit"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--examples-per-family", type=int, default=5)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    stage_audit(config_path, examples_per_family=args.examples_per_family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
