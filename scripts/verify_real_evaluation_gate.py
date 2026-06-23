#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.real_eval import (
    ensure_no_references_or_paths,
    sha256_file,
    validate_fleurs_v2_manifest_rows,
    validate_gate_manifest,
)


def load_metadata(path: Path) -> dict[str, object]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be a JSON object")
    ensure_no_references_or_paths(metadata)
    return metadata


def verify_gate(manifest: Path, metadata_path: Path, expected_gate_id: str, expected_rows: int) -> dict[str, object]:
    metadata = load_metadata(metadata_path)
    rows = validate_gate_manifest(manifest)
    manifest_sha256 = sha256_file(manifest)
    declared_manifest_sha256 = metadata.get("manifest_sha256")

    if metadata.get("gate_id") != expected_gate_id:
        raise ValueError(f"metadata gate_id mismatch: {metadata.get('gate_id')} != {expected_gate_id}")
    if len(rows) != expected_rows:
        raise ValueError(f"manifest row count mismatch: {len(rows)} != {expected_rows}")
    if metadata.get("rows") != expected_rows:
        raise ValueError(f"metadata row count mismatch: {metadata.get('rows')} != {expected_rows}")
    if declared_manifest_sha256 != manifest_sha256:
        raise ValueError(f"manifest SHA256 mismatch: {manifest_sha256} != {declared_manifest_sha256}")

    selected = metadata.get("selected")
    if not isinstance(selected, list):
        raise ValueError("metadata selected must be a list")
    if len(selected) != expected_rows:
        raise ValueError(f"metadata selected count mismatch: {len(selected)} != {expected_rows}")

    sample_ids = {str(row["sample_id"]) for row in rows}
    audio_paths = {Path(row["audio_filepath"]).expanduser().resolve(strict=False).as_posix() for row in rows}
    source_row_indexes: set[int] | None = None
    if expected_gate_id == "fleurs-sl-si-test-full-v2":
        validate_fleurs_v2_manifest_rows(rows, expected_rows=expected_rows)
        source_row_indexes = {int(row["source_row_index"]) for row in rows}
        metadata_indexes = [int(item["source_row_index"]) for item in selected if isinstance(item, dict)]
        if sorted(metadata_indexes) != list(range(expected_rows)):
            raise ValueError("metadata source_row_index values do not cover the expected FLEURS v2 range")

    return {
        "rows": len(rows),
        "unique_sample_ids": len(sample_ids),
        "unique_audio_paths": len(audio_paths),
        "unique_source_row_indexes": len(source_row_indexes) if source_row_indexes is not None else "not_applicable",
        "manifest_sha256": manifest_sha256,
        "metadata_declared_manifest_sha256": declared_manifest_sha256,
        "audio_validation_count": len(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an ignored real evaluation gate manifest against committed metadata.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expected-gate-id", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()

    try:
        summary = verify_gate(args.manifest, args.metadata, args.expected_gate_id, args.expected_rows)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
