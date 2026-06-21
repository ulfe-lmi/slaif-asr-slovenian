#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import load_runtime_config, repo_path
from slaif_asr.tokenizer_audit import audit_tokenizer, write_audit_report


def main() -> int:
    cfg = load_runtime_config()
    parser = argparse.ArgumentParser(description="Audit Slovenian text round trips through the pinned ASR tokenizer.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=repo_path("local_artifacts.checkpoint_dir") / cfg["base_model"]["filename"],
    )
    parser.add_argument("--output", type=Path, default=repo_path("local_artifacts.tokenizer_audit_dir") / "sl-si.json")
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(restore_path=str(args.checkpoint), map_location="cpu")
    report = audit_tokenizer(model.tokenizer)
    write_audit_report(report, args.output)
    print(f"Wrote tokenizer audit: {args.output}")
    return 0 if report.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
