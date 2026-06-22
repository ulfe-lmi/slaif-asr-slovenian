#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.active_curriculum import (
    build_round2_failure_brief,
    deterministic_active_selection,
    deterministic_controls,
    score_candidate,
)
from slaif_asr.prompt_experiment import atomic_write_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Select active prompt-column curriculum examples from scored candidates.")
    parser.add_argument("--scores-jsonl", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--hard-count", type=int, default=48)
    parser.add_argument("--control-count", type=int, default=16)
    parser.add_argument("--control-seed", type=int, default=1234)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--brief", type=Path)
    args = parser.parse_args()

    scored = []
    with args.scores_jsonl.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            scored.append(
                score_candidate(
                    candidate_id=row["candidate_id"],
                    reference=row["reference"],
                    hypothesis=row.get("hypothesis", ""),
                    phenomena=tuple(row.get("phenomena", [])),
                )
            )
    hard = deterministic_active_selection(scored, hard_count=args.hard_count)
    controls = deterministic_controls(scored, exclude_ids={item.candidate_id for item in hard}, count=args.control_count, seed=args.control_seed)
    payload = {
        "schema_version": "1.0",
        "round_id": args.round_id,
        "hard_examples": [item.candidate_id for item in hard],
        "general_controls": [item.candidate_id for item in controls],
    }
    atomic_write_text(args.selection, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    if args.brief is not None:
        atomic_write_text(args.brief, json.dumps(build_round2_failure_brief(scored), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
