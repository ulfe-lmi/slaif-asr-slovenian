#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.corpus_v2_generation import (
    filter_records,
    generated_all_path,
    load_config,
    load_jsonl,
    load_protected_indexes,
    pre_review_path,
    rejected_path,
    run_dir,
    write_public_reports,
    write_rejections,
    write_review_outputs,
)
from slaif_asr.data_quality import atomic_write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the local corpus-v2 native-speaker review pack.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not generated_all_path(config).exists():
        raise FileNotFoundError(f"missing generated corpus: {generated_all_path(config)}")
    generated = load_jsonl(generated_all_path(config))
    previous_rejections = load_jsonl(rejected_path(config)) if rejected_path(config).exists() else []
    protected_indexes = load_protected_indexes(config)
    retained, rejected, _summary = filter_records(
        generated,
        config=config,
        existing_rejections=previous_rejections,
        protected_indexes=protected_indexes,
    )
    atomic_write_jsonl(pre_review_path(config), retained)
    write_rejections(rejected_path(config), rejected)
    write_review_outputs(retained, config)
    payload = write_public_reports(config)
    summary = {
        "corpus_id": config["corpus_id"],
        "retained_for_review": len(retained),
        "minimum_structurally_admissible_rows": config["minimum_structurally_admissible_rows"],
        "status": payload["status"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if len(retained) >= int(config["minimum_structurally_admissible_rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
