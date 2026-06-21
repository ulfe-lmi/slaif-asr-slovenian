#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.tts import load_candidates, load_tts_config, render_candidates, repo_resolve


def verify_one_visible_gpu(piper_python: Path) -> str:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    code = (
        "import onnxruntime as ort\n"
        "providers = ort.get_available_providers()\n"
        "print(providers)\n"
        "assert 'CUDAExecutionProvider' in providers\n"
    )
    completed = subprocess.run(
        [str(piper_python), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or "CUDAExecutionProvider unavailable")
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if nvidia.returncode != 0:
        raise RuntimeError(nvidia.stderr.strip() or "nvidia-smi failed")
    first_line = nvidia.stdout.splitlines()[0] if nvidia.stdout.splitlines() else ""
    if "2080 Ti" not in first_line:
        raise RuntimeError(f"physical GPU 0 is not an RTX 2080 Ti: {first_line}")
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render validated Slovenian smoke candidates with Piper on GPU 0.")
    parser.add_argument("--candidates", type=Path, default=Path("configs/tts/piper_smoke_candidates.jsonl"))
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_tts_config()
    piper_python = repo_resolve(cfg["engine"]["environment"]) / "bin" / "python"
    provider_report = verify_one_visible_gpu(piper_python)
    candidates = load_candidates(args.candidates)
    result = render_candidates(candidates=candidates, config=cfg, output_root=args.output_root)
    result["provider_report"] = provider_report
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
