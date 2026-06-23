#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import load_runtime_config
from slaif_asr.gpu_policy import require_single_visible_cuda
from slaif_asr.metrics import empty_status, raw_cer, raw_wer, recognition_change
from slaif_asr.prompt_experiment import load_json, repository_path, write_json
from slaif_asr.tts import sha256_file


def newest_context_dir(output_dir: Path) -> Path:
    run_dirs = sorted((path for path in output_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
    if not run_dirs:
        raise FileNotFoundError(f"no inference run directory under {output_dir}")
    context_dirs = sorted(run_dirs[-1].glob("context_56_3"))
    if not context_dirs:
        raise FileNotFoundError(f"no context_56_3 under {run_dirs[-1]}")
    return context_dirs[-1]


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def read_predictions(context_dir: Path) -> list[dict[str, Any]]:
    candidates = sorted(context_dir.glob("streaming_out_*.json"))
    if not candidates:
        raise FileNotFoundError(f"no NeMo streaming output in {context_dir}")
    with candidates[0].open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def run_inference(*, checkpoint: Path, manifest: Path, output_dir: Path) -> tuple[Path, float]:
    command = [
        sys.executable,
        "scripts/run_streaming_inference.py",
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(checkpoint),
        "--context",
        "[56,3]",
        "--batch-size",
        "1",
        "--cuda",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
    wall_time = time.perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command.log").write_text(" ".join(command) + "\n\n" + completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)
    return newest_context_dir(output_dir), wall_time


def evaluate_split(split_name: str, manifest: Path, base_context: Path, adapted_context: Path) -> dict[str, Any]:
    rows = read_manifest(manifest)
    base_predictions = read_predictions(base_context)
    adapted_predictions = read_predictions(adapted_context)
    if len(rows) != len(base_predictions) or len(rows) != len(adapted_predictions):
        raise ValueError(f"{split_name}: prediction count does not match manifest count")
    items = []
    for row, base_row, adapted_row in zip(rows, base_predictions, adapted_predictions, strict=True):
        reference = row["text"]
        base_hypothesis = str(base_row.get("pred_text", ""))
        adapted_hypothesis = str(adapted_row.get("pred_text", ""))
        items.append(
            {
                "sample_id": row.get("sample_id"),
                "reference": reference,
                "base_hypothesis": base_hypothesis,
                "adapted_hypothesis": adapted_hypothesis,
                "base_empty_status": empty_status(base_hypothesis),
                "adapted_empty_status": empty_status(adapted_hypothesis),
                "base_raw_wer": raw_wer(reference, base_hypothesis).percent,
                "adapted_raw_wer": raw_wer(reference, adapted_hypothesis).percent,
                "base_raw_cer": raw_cer(reference, base_hypothesis).percent,
                "adapted_raw_cer": raw_cer(reference, adapted_hypothesis).percent,
                "pipeline_status": "PASSED",
                "recognition_status": recognition_change(reference, base_hypothesis, adapted_hypothesis),
            }
        )
    aggregate = {
        "base_wer": round(sum(item["base_raw_wer"] for item in items) / len(items), 3),
        "adapted_wer": round(sum(item["adapted_raw_wer"] for item in items) / len(items), 3),
        "base_cer": round(sum(item["base_raw_cer"] for item in items) / len(items), 3),
        "adapted_cer": round(sum(item["adapted_raw_cer"] for item in items) / len(items), 3),
        "base_empty_count": sum(1 for item in items if item["base_empty_status"] == "EMPTY_HYPOTHESIS"),
        "adapted_empty_count": sum(1 for item in items if item["adapted_empty_status"] == "EMPTY_HYPOTHESIS"),
    }
    return {"split": split_name, "items": items, "aggregate": aggregate}


def conclusion(training_summary: dict[str, Any], synthetic_training: dict[str, Any]) -> str:
    if not training_summary.get("integrity_passed"):
        return "EXPERIMENT_INVALID"
    phase_a_status = training_summary["phase_a"][-1]["classification"]
    if phase_a_status != "Supported":
        return "PROMPT_COLUMN_NOT_SUPPORTED" if phase_a_status == "Not supported" else "PROMPT_COLUMN_PARTIALLY_SUPPORTED"
    base = synthetic_training["aggregate"]["base_wer"]
    adapted = synthetic_training["aggregate"]["adapted_wer"]
    absolute = base - adapted
    relative = 0.0 if base == 0 else absolute / base * 100.0
    empty_ok = synthetic_training["aggregate"]["adapted_empty_count"] <= synthetic_training["aggregate"]["base_empty_count"]
    if empty_ok and (absolute >= 20.0 or relative >= 25.0):
        return "PROMPT_COLUMN_SUPPORTED"
    return "PROMPT_COLUMN_PARTIALLY_SUPPORTED"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate base and adapted prompt-column checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/prompt_column_micro_overfit.json"))
    args = parser.parse_args()
    require_single_visible_cuda()
    config = load_json(args.config)
    runtime_cfg = load_runtime_config()
    run_dir = repository_path(config["paths"]["run_dir"])
    training_summary = load_json(run_dir / "training-summary.json")
    base_checkpoint = repository_path(config["paths"]["checkpoint"])
    adapted_checkpoint = Path(training_summary["merged_checkpoint"]).resolve()
    if sha256_file(base_checkpoint) != runtime_cfg["base_model"]["sha256"]:
        raise RuntimeError("base checkpoint SHA256 mismatch")
    manifests = {
        "synthetic_training": run_dir / "manifests" / "synthetic_training.jsonl",
        "synthetic_holdout": run_dir / "manifests" / "synthetic_holdout.jsonl",
        "real_public_smoke": run_dir / "manifests" / "real_public_smoke.jsonl",
    }
    results = {}
    wall_times = {}
    for split, manifest in manifests.items():
        base_context, base_time = run_inference(
            checkpoint=base_checkpoint,
            manifest=manifest,
            output_dir=run_dir / "eval" / split / "base",
        )
        adapted_context, adapted_time = run_inference(
            checkpoint=adapted_checkpoint,
            manifest=manifest,
            output_dir=run_dir / "eval" / split / "adapted",
        )
        results[split] = evaluate_split(split, manifest, base_context, adapted_context)
        wall_times[split] = {"base": round(base_time, 3), "adapted": round(adapted_time, 3)}
    payload = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "base_checkpoint": str(base_checkpoint),
        "adapted_checkpoint": str(adapted_checkpoint),
        "wall_times_seconds": wall_times,
        "results": results,
        "scientific_conclusion": conclusion(training_summary, results["synthetic_training"]),
        "quality_claim_made": False,
    }
    write_json(run_dir / "evaluation-summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
