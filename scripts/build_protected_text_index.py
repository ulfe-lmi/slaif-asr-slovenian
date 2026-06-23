#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slaif_asr.data_quality import (
    atomic_write_json,
    build_protected_index_payload,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a privacy-safe hash-only protected text index.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        payload = build_protected_index_payload(args.manifest, args.metadata)
        atomic_write_json(args.output, payload)
    except Exception as exc:
        print(f"protected-index build failed: {exc}", file=sys.stderr)
        return 1

    summary = {
        "gate_id": payload["gate_id"],
        "manifest_sha256": payload["manifest_sha256"],
        "row_count": payload["row_count"],
        "surface_hashes": len(payload["surface_hashes"]),
        "number_masked_hashes": len(payload["number_masked_hashes"]),
        "output_sha256": sha256_file(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
