#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.config import load_runtime_config, repo_path, streaming_contexts
from slaif_asr.inference import resolve_existing_path, run_context


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
    parser.add_argument("--nemo-root", type=Path, default=repo_path("nemo.source_tree"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("local_artifacts.inference_output_dir"))
    parser.add_argument("--cuda", type=int, default=None)
    args = parser.parse_args()

    script = resolve_existing_path(
        args.nemo_root / "examples" / "asr" / "asr_cache_aware_streaming" / "speech_to_text_cache_aware_streaming_infer.py",
        "NeMo streaming script",
    )
    checkpoint = resolve_existing_path(args.checkpoint, "checkpoint")
    output_dir = args.output_dir.expanduser().resolve()
    nemo_root = args.nemo_root.expanduser().resolve()
    if args.manifest is not None:
        source_manifest = resolve_existing_path(args.manifest, "manifest")
    else:
        source_manifest = None
    if args.audio_file is not None:
        audio_file = resolve_existing_path(args.audio_file, "audio file")
    else:
        audio_file = None

    contexts = streaming_contexts() if args.all_contexts else [args.context]
    run_root = output_dir / time.strftime("%Y%m%d-%H%M%S")
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest
    if audio_file is not None:
        manifest = run_root / "single-audio-input.jsonl"
        with manifest.open("w", encoding="utf-8") as fp:
            fp.write(json.dumps({"audio_filepath": str(audio_file)}, ensure_ascii=False) + "\n")

    for context in contexts:
        context_dir = run_root / f"context_{context[0]}_{context[1]}"
        command = [
            sys.executable,
            str(script),
            f"model_path={checkpoint}",
            f"batch_size={args.batch_size}",
            "target_lang=sl-SI",
            "strip_lang_tags=true",
            f"att_context_size=[{context[0]},{context[1]}]",
            f"output_path={context_dir}",
        ]
        if args.cuda is not None:
            command.append(f"cuda={args.cuda}")
        command.append(f"dataset_manifest={manifest}")
        env = os.environ.copy()
        env.setdefault("NEMO_ROOT", str(nemo_root))
        result = run_context(
            command=command,
            context=context,
            context_dir=context_dir,
            checkpoint_sha256=cfg["base_model"]["sha256"],
            cuda_index=args.cuda,
            env=env,
        )
        print(f"context=[{context[0]},{context[1]}] exit_status={result.exit_status}")
        print(f"log={result.log_path}")
        print(f"result_json={result.result_path}")
        print(f"wall_time_seconds={result.wall_time_seconds:.3f}")
        print(f"transcript={result.transcript}")
        if result.exit_status != 0:
            return result.exit_status
        if not result.result_path.exists():
            print(f"Missing result JSON: {result.result_path}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
