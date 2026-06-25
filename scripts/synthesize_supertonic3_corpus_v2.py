#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.supertonic3_tts import (
    assert_supertonic_runtime_environment,
    convert_native_audio,
    load_supertonic_config,
    package_wheel_hashes,
    runtime_versions,
    synthesize_partition,
    verify_assets,
    verify_input_identities,
    maybe_reexec_with_supertonic_cuda_libraries,
)


def run_stage(args: argparse.Namespace) -> dict[str, object]:
    config = load_supertonic_config(args.config)
    maybe_reexec_with_supertonic_cuda_libraries(config)
    if args.stage == "verify":
        assert_supertonic_runtime_environment(config)
        return {
            "stage": args.stage,
            "status": "PASSED",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "inputs": verify_input_identities(config),
            "assets": verify_assets(config),
            "runtime": runtime_versions(config),
            "wheel_hashes": package_wheel_hashes(config),
        }
    if args.stage == "synthesize-training":
        return synthesize_partition(config, partition="training", progress_interval_seconds=args.progress_interval_seconds)
    if args.stage == "synthesize-holdout":
        return synthesize_partition(config, partition="holdout", progress_interval_seconds=args.progress_interval_seconds)
    if args.stage == "convert":
        return convert_native_audio(config, progress_interval_seconds=args.progress_interval_seconds)
    raise ValueError(f"unsupported stage: {args.stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize corpus-v2 Supertonic 3 audio.")
    parser.add_argument("--config", type=Path, default=Path("configs/tts/supertonic3_sl_multivoice_v1.json"))
    parser.add_argument(
        "--stage",
        required=True,
        choices=["verify", "synthesize-training", "synthesize-holdout", "convert"],
    )
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    result = run_stage(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
