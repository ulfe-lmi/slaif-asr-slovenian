#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.corpus_v2_review import run_review_admission


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest native-speaker review decisions for the corpus-v2 GaMS reservoir.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/generation/slovenian_corpus_v2_candidate_reservoir.json",
        help="Corpus-v2 generation configuration.",
    )
    parser.add_argument(
        "--data-quality-config",
        type=Path,
        default=REPO_ROOT / "configs/data_quality/training_text_v1.json",
        help="Text-stage admission policy configuration.",
    )
    parser.add_argument(
        "--retired-registry",
        type=Path,
        default=REPO_ROOT / "configs/data_quality/retired_corpora.json",
        help="Retired corpus registry.",
    )
    parser.add_argument(
        "--require-status",
        default="TEXT_ACCEPTED",
        help="Required final status for zero exit. Defaults to TEXT_ACCEPTED.",
    )
    args = parser.parse_args()

    try:
        report, return_code = run_review_admission(
            generation_config_path=args.config,
            data_quality_config_path=args.data_quality_config,
            retired_registry_path=args.retired_registry,
            require_status=args.require_status,
        )
    except Exception as exc:
        print(f"review admission failed before report completion: {exc}", file=sys.stderr)
        return 2

    summary = {
        "accepted_count": report["review"]["accepted_count"],
        "corpus_id": report["corpus_id"],
        "decision_reasons": report["validator"]["decision_reasons"],
        "final_text_status": report["validator"]["status"],
        "review_sheet_sha256": report["review"]["review_sheet_sha256"],
        "review_total": report["review"]["total_review_rows"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
