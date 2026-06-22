#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.active_curriculum import decide_promotion, promotion_decision_json
from slaif_asr.metrics import corpus_metric_summary
from slaif_asr.prompt_experiment import atomic_write_text


def load_pairs(path: Path, *, hypothesis_field: str) -> list[tuple[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((row["reference"], row.get(hypothesis_field, "")))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate active prompt-column promotion gates from JSONL predictions.")
    parser.add_argument("--synthetic-jsonl", type=Path, required=True)
    parser.add_argument("--real-jsonl", type=Path, required=True)
    parser.add_argument("--integrity-passed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    synthetic_base = load_pairs(args.synthetic_jsonl, hypothesis_field="base_hypothesis")
    synthetic_challenger = load_pairs(args.synthetic_jsonl, hypothesis_field="challenger_hypothesis")
    real_base = load_pairs(args.real_jsonl, hypothesis_field="base_hypothesis")
    real_challenger = load_pairs(args.real_jsonl, hypothesis_field="challenger_hypothesis")
    decision = decide_promotion(
        integrity_passed=args.integrity_passed,
        synthetic_base_rows=synthetic_base,
        synthetic_challenger_rows=synthetic_challenger,
        real_base_rows=real_base,
        real_challenger_rows=real_challenger,
    )
    payload = {
        "schema_version": "1.0",
        "synthetic_base": asdict(corpus_metric_summary(synthetic_base)),
        "synthetic_challenger": asdict(corpus_metric_summary(synthetic_challenger)),
        "real_base": asdict(corpus_metric_summary(real_base)),
        "real_challenger": asdict(corpus_metric_summary(real_challenger)),
        "promotion": promotion_decision_json(decision),
    }
    atomic_write_text(args.output, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if decision.promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
