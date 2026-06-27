#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_json, atomic_write_text
from slaif_asr.scale8000_corpus import (
    build_dual_gpu_generation_plan,
    load_scale8000_generation_config,
    safe_public_status_report,
    scale8000_multiplier_table,
    storage_preflight,
    verify_inherited_scale2000_rows,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/generation/gams_corpus_v5_scale8000_v1.json"


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": " ".join(command), "exit_code": completed.returncode, "output_tail": completed.stdout[-4000:]}


def stage_verify(config_path: Path) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    inherited = verify_inherited_scale2000_rows(config)
    payload = {
        "status": "PASSED",
        "corpus_id": config["corpus_id"],
        "inherited_rows": len(inherited),
        "multiplier_table": scale8000_multiplier_table(),
        "dual_gpu_plan": build_dual_gpu_generation_plan(config),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def stage_preflight(config_path: Path) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    payload = storage_preflight(config)
    payload["status"] = "PASSED" if payload["sufficient"] else "ENVIRONMENT_BLOCKED"
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _markdown_report(payload: dict[str, Any]) -> str:
    preflight = payload.get("storage_preflight", {})
    plan = payload["scale8000_plan"]
    lines = [
        "# Scale-8000 Dual-GPU Generation",
        "",
        f"Status: `{payload['status']}`",
        "",
        "This report is privacy-safe planning and preflight evidence. It contains no raw generated text, audio paths, hypotheses, model artifacts, or monitor CSV data.",
        "",
        "## Corpus",
        "",
        f"- Corpus ID: `{payload['corpus_id']}`",
        f"- Parent scale-2000 corpus: `{payload['parent_scale2000']['corpus_id']}`",
        f"- Parent scale-2000 SHA256: `{payload['parent_scale2000']['sha256']}`",
        f"- Semantic rows planned: `{plan['semantic_rows']}`",
        f"- Clean files/views planned: `{plan['clean_files']}`",
        f"- Augmented files/views planned: `{plan['augmented_files']}`",
        f"- Total views/exposures planned: `{plan['total_views']}`",
        "",
        "## Inclusion",
        "",
        f"- Policy: `{payload['inclusion_policy']['type']}`",
        f"- Evidence: {payload['inclusion_policy']['description']}",
        "",
        "## Dual-GPU Plan",
        "",
    ]
    for worker, worker_payload in payload["dual_gpu_plan"]["workers"].items():
        lines.append(
            f"- `{worker}`: physical GPU `{worker_payload['physical_gpu']}`, "
            f"`CUDA_VISIBLE_DEVICES={worker_payload['cuda_visible_devices']}`, "
            f"tasks `{worker_payload['task_count']}`, requested rows `{worker_payload['requested_rows']}`"
        )
    lines.extend(
        [
            "",
            "## Storage Preflight",
            "",
            f"- Available bytes: `{preflight.get('available_bytes', 'unknown')}`",
            f"- Projected new bytes: `{preflight.get('projected_new_bytes', 'unknown')}`",
            f"- Required free bytes with safety margin: `{preflight.get('required_free_bytes', 'unknown')}`",
            f"- Sufficient: `{preflight.get('sufficient', 'unknown')}`",
            "",
            "## Decision",
            "",
            "Generation must not begin while storage preflight is insufficient. This is an environment blocker, not a corpus acceptance decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def stage_write_report(config_path: Path, canonical_results_path: Path | None = None) -> dict[str, Any]:
    config = load_scale8000_generation_config(config_path)
    canonical = {}
    if canonical_results_path and canonical_results_path.exists():
        canonical = json.loads(canonical_results_path.read_text(encoding="utf-8"))
    preflight = storage_preflight(config)
    payload = safe_public_status_report(config, canonical_results=canonical, preflight=preflight)
    report_paths = config["public_reports"]
    json_path = REPO_ROOT / report_paths["planning_report_json"]
    md_path = REPO_ROOT / report_paths["planning_report_markdown"]
    atomic_write_json(json_path, payload)
    atomic_write_text(md_path, _markdown_report(payload))
    print(json.dumps({"status": payload["status"], "json": str(json_path), "markdown": str(md_path)}, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["verify", "preflight", "write-report"], required=True)
    parser.add_argument("--canonical-results", type=Path)
    args = parser.parse_args()
    if args.stage == "verify":
        stage_verify(args.config)
    elif args.stage == "preflight":
        stage_preflight(args.config)
    elif args.stage == "write-report":
        stage_write_report(args.config, args.canonical_results)
    else:  # pragma: no cover
        raise AssertionError(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
