#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import load_runtime_config, repo_path, streaming_contexts


def parse_context(value: str) -> tuple[int, int]:
    normalized = value.strip().strip("[]")
    parts = [part.strip() for part in normalized.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("context must look like [56,13]")
    context = (int(parts[0]), int(parts[1]))
    if context not in streaming_contexts():
        raise argparse.ArgumentTypeError(f"unsupported context {context}")
    return context


def main() -> int:
    cfg = load_runtime_config()
    checkpoint_default = repo_path("local_artifacts.checkpoint_dir") / cfg["base_model"]["filename"]
    parser = argparse.ArgumentParser(description="Run pinned cache-aware streaming inference with target_lang=sl-SI.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-file", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=checkpoint_default)
    parser.add_argument("--context", type=parse_context, default=(56, 13))
    parser.add_argument("--all-contexts", action="store_true", help="Run all supported context settings.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nemo-root", type=Path, default=Path(".external/NeMo"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("local_artifacts.inference_output_dir"))
    parser.add_argument("--cuda", type=int, default=None)
    args = parser.parse_args()

    script = args.nemo_root / "examples" / "asr" / "asr_cache_aware_streaming" / "speech_to_text_cache_aware_streaming_infer.py"
    if not script.exists():
        raise SystemExit(f"Missing NeMo streaming script: {script}. Run scripts/setup_runtime_env.sh first.")
    if not args.checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {args.checkpoint}. Run scripts/download_nemotron_checkpoint.py first.")

    contexts = streaming_contexts() if args.all_contexts else [args.context]
    run_root = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest
    if args.audio_file is not None:
        manifest = run_root / "single-audio-input.jsonl"
        with manifest.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({"audio_filepath": str(args.audio_file)}, ensure_ascii=False) + "\n")

    for context in contexts:
        output_path = run_root / f"context_{context[0]}_{context[1]}"
        command = [
            "python3",
            str(script),
            f"model_path={args.checkpoint}",
            f"batch_size={args.batch_size}",
            "target_lang=sl-SI",
            "strip_lang_tags=true",
            f"att_context_size=[{context[0]},{context[1]}]",
            f"output_path={output_path}",
        ]
        if args.cuda is not None:
            command.append(f"cuda={args.cuda}")
        command.append(f"dataset_manifest={manifest}")
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
