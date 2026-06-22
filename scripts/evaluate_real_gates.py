#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import load_runtime_config, repo_path
from slaif_asr.inference import resolve_existing_path
from slaif_asr.real_eval import atomic_write_json, atomic_write_jsonl, sha256_file, summarize_predictions, validate_gate_manifest


def run_nvidia_smi(path: Path) -> None:
    completed = subprocess.run(["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(completed.stdout, encoding="utf-8")


def parse_nemo_jsonl(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append({"reference": str(item.get("text", "")), "hypothesis": str(item.get("pred_text", ""))})
    return rows


def newest_streaming_output(context_dir: Path) -> Path:
    candidates = sorted(context_dir.glob("streaming_out_*.json"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"missing NeMo streaming output under {context_dir}")
    return candidates[-1]


def evaluate_gate(args: argparse.Namespace, manifest: Path, gate_id: str) -> dict[str, object]:
    cfg = load_runtime_config()
    checkpoint = resolve_existing_path(args.checkpoint, "checkpoint")
    nemo_root = args.nemo_root.expanduser().resolve()
    script = resolve_existing_path(
        nemo_root / "examples" / "asr" / "asr_cache_aware_streaming" / "speech_to_text_cache_aware_streaming_infer.py",
        "NeMo streaming script",
    )
    manifest_rows = validate_gate_manifest(manifest)
    run_root = args.output_root / gate_id / time.strftime("%Y%m%d-%H%M%S")
    context_dir = run_root / "context_56_3"
    context_dir.mkdir(parents=True, exist_ok=True)
    run_nvidia_smi(run_root / "nvidia-smi-before.txt")
    command = [
        sys.executable,
        str(script),
        f"model_path={checkpoint}",
        f"batch_size={args.batch_size}",
        "target_lang=sl-SI",
        "strip_lang_tags=true",
        "att_context_size=[56,3]",
        f"output_path={context_dir}",
        "cuda=0",
        f"dataset_manifest={manifest.resolve()}",
    ]
    env = os.environ.copy()
    env.setdefault("NEMO_ROOT", str(nemo_root))
    start = time.perf_counter()
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)
    wall_time = time.perf_counter() - start
    run_nvidia_smi(run_root / "nvidia-smi-after.txt")
    (run_root / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    (run_root / "inference.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"{gate_id}: inference failed with exit {completed.returncode}; see {run_root / 'inference.log'}")
    output_path = newest_streaming_output(context_dir)
    prediction_rows = parse_nemo_jsonl(output_path)
    if len(prediction_rows) != len(manifest_rows):
        raise RuntimeError(f"{gate_id}: prediction count {len(prediction_rows)} != manifest count {len(manifest_rows)}")
    per_sample = []
    for manifest_row, prediction in zip(manifest_rows, prediction_rows, strict=True):
        per_sample.append(
            {
                "sample_id": manifest_row["sample_id"],
                "reference": prediction["reference"],
                "hypothesis": prediction["hypothesis"],
                "pipeline_status": "PASSED",
                "empty_hypothesis": not prediction["hypothesis"].strip(),
            }
        )
    metrics = summarize_predictions(per_sample)
    per_sample_path = run_root / "per-sample.local.jsonl"
    atomic_write_jsonl(per_sample_path, per_sample)
    audio_duration = round(sum(float(row["duration"]) for row in manifest_rows), 6)
    summary = {
        "gate_id": gate_id,
        "manifest_sha256": sha256_file(manifest),
        "model_repository": cfg["base_model"]["repository"],
        "model_revision": cfg["base_model"]["revision"],
        "checkpoint_sha256": cfg["base_model"]["sha256"],
        "att_context_size": [56, 3],
        "target_lang": "sl-SI",
        "batch_size": args.batch_size,
        "wall_time_seconds": round(wall_time, 3),
        "audio_duration_seconds": audio_duration,
        "real_time_factor": round(wall_time / audio_duration, 6) if audio_duration else None,
        "rows": len(per_sample),
        "metrics": metrics,
        "nemo_output": str(output_path),
        "per_sample_local": str(per_sample_path),
        "exit_status": completed.returncode,
    }
    atomic_write_json(run_root / "summary.local.json", summary)
    return summary


def main() -> int:
    cfg = load_runtime_config()
    checkpoint_default = repo_path("local_artifacts.checkpoint_dir") / cfg["base_model"]["filename"]
    parser = argparse.ArgumentParser(description="Evaluate untouched Nemotron on immutable real gates.")
    parser.add_argument("--fleurs-manifest", type=Path)
    parser.add_argument("--artur-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_default)
    parser.add_argument("--nemo-root", type=Path, default=repo_path("nemo.source_tree"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/evaluation-baselines/real-gates"))
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if not args.fleurs_manifest and not args.artur_manifest:
        parser.error("at least one manifest is required")
    summaries = []
    if args.fleurs_manifest:
        summaries.append(evaluate_gate(args, args.fleurs_manifest, "fleurs-sl-si-test-full-v1"))
    if args.artur_manifest:
        summaries.append(evaluate_gate(args, args.artur_manifest, "artur-j-public-gate-v1"))
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
