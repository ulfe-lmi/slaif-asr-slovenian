#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from slaif_asr.data_quality import (
    EMITTABLE_STATUSES,
    assert_privacy_safe_report,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json_sha256,
    load_json,
    validate_corpus,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_partition(values: list[str]) -> dict[str, Path]:
    partitions: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"partition must use role=path syntax: {value!r}")
        role, path_text = value.split("=", 1)
        role = role.strip()
        if not role:
            raise ValueError("partition role must not be empty")
        if role in partitions:
            raise ValueError(f"duplicate partition role: {role}")
        partitions[role] = Path(path_text)
    if not partitions:
        raise ValueError("at least one --partition is required")
    return partitions


def git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed text-stage training-corpus admission validator.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--partition", action="append", default=[], help="Repeated role=path partition input.")
    parser.add_argument("--linguistic-review", type=Path, required=False)
    parser.add_argument("--protected-index", type=Path, action="append", default=[])
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--local-review-output", type=Path, required=True)
    parser.add_argument("--require-status", choices=sorted(EMITTABLE_STATUSES), required=True)
    parser.add_argument(
        "--retired-registry",
        type=Path,
        default=REPO_ROOT / "configs/data_quality/retired_corpora.json",
    )
    args = parser.parse_args()

    try:
        partitions = parse_partition(args.partition)
        config = load_json(args.config)
        retired_registry = load_json(args.retired_registry)
        report, local_review_rows = validate_corpus(
            corpus_id=args.corpus_id,
            config=config,
            config_sha256=canonical_json_sha256(config),
            retired_registry=retired_registry,
            partitions=partitions,
            linguistic_review_path=args.linguistic_review,
            protected_index_paths=list(args.protected_index),
            repository_revision=git_revision(),
        )
        assert_privacy_safe_report(report)
        atomic_write_json(args.output_report, report)
        atomic_write_jsonl(args.local_review_output, local_review_rows)
    except Exception as exc:
        print(f"training-corpus validation failed before report completion: {exc}", file=sys.stderr)
        return 2

    status = str(report["final_text_status"])
    summary = {
        "corpus_id": args.corpus_id,
        "final_text_status": status,
        "decision_reasons": report.get("decision_reasons", []),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if status != args.require_status:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
