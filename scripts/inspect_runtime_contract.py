#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import load_runtime_config, repo_path
from slaif_asr.contract import build_runtime_contract, write_contract


def main() -> int:
    cfg = load_runtime_config()
    parser = argparse.ArgumentParser(description="Load the pinned Nemotron checkpoint and write its runtime contract.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_path("local_artifacts.checkpoint_dir") / cfg["base_model"]["filename"],
    )
    parser.add_argument("--output", type=Path, default=repo_path("local_artifacts.runtime_contract_dir") / "contract.json")
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(args.checkpoint), map_location="cpu")
    contract = build_runtime_contract(model, checkpoint_path=str(args.checkpoint))
    write_contract(contract, args.output)
    print(f"Wrote runtime contract: {args.output}")
    if contract.prompt_indices.get("sl-SI") is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
