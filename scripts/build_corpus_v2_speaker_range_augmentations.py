#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.speaker_range_augmentation import (
    DEFAULT_AUGMENTATION_CONFIG,
    DEFAULT_EXPERIMENT_CONFIG,
    augmentation_paths,
    build_augmentations,
    load_augmentation_config,
    load_experiment_config,
    summarize_local_augmentation,
    validate_augmentations,
    verify_baseline_report,
    verify_data_identities,
    verify_speaker_range_certificate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build corpus-v2 speaker-range audio variants.")
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--augmentation-config", type=Path, default=DEFAULT_AUGMENTATION_CONFIG)
    parser.add_argument("--stage", required=True, choices=["verify", "generate", "validate", "summarize"])
    args = parser.parse_args()

    experiment_config = load_experiment_config(args.experiment_config)
    augmentation_config = load_augmentation_config(args.augmentation_config)
    if args.stage == "verify":
        payload = {
            "status": "PASSED",
            "certificate": verify_speaker_range_certificate(args.experiment_config, require_head=True),
            "baseline": verify_baseline_report(),
            "data": verify_data_identities(experiment_config),
            "augmentation_paths": {
                "run_root": str(augmentation_paths(augmentation_config).run_root),
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    verify_speaker_range_certificate(args.experiment_config, require_head=True)
    if args.stage == "generate":
        payload = build_augmentations(experiment_config, augmentation_config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.stage == "validate":
        payload = validate_augmentations(experiment_config, augmentation_config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.stage == "summarize":
        payload = summarize_local_augmentation(experiment_config, augmentation_config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
