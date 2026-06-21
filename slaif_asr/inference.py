from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slaif_asr.config import load_runtime_config


TRANSCRIPT_PATTERN = re.compile(r"Final streaming transcriptions:\s*(\[[^\n]*\])")


@dataclass(frozen=True)
class InferenceRunResult:
    context: tuple[int, int]
    result_path: Path
    log_path: Path
    transcript: str
    wall_time_seconds: float
    exit_status: int


def resolve_existing_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def chunk_seconds(context: tuple[int, int]) -> float | None:
    for item in load_runtime_config()["streaming_contexts"]:
        if tuple(item["att_context_size"]) == context:
            return float(item["chunk_seconds"])
    return None


def parse_final_transcript(log_text: str) -> str:
    matches = TRANSCRIPT_PATTERN.findall(log_text)
    if not matches:
        return ""
    try:
        value = ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return ""
    if isinstance(value, list) and value:
        return str(value[0])
    return ""


def gpu_name(cuda_index: int | None) -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = 0 if cuda_index is None else cuda_index
        if index < torch.cuda.device_count():
            return torch.cuda.get_device_name(index)
    except Exception:
        return None
    return None


def build_result_record(
    *,
    context: tuple[int, int],
    transcript: str,
    wall_time_seconds: float,
    exit_status: int,
    checkpoint_sha256: str,
    gpu: str | None,
) -> dict[str, Any]:
    cfg = load_runtime_config()
    return {
        "model_repository": cfg["base_model"]["repository"],
        "model_revision": cfg["base_model"]["revision"],
        "checkpoint_sha256": checkpoint_sha256,
        "target_lang": cfg["prompt"]["target_lang"],
        "att_context_size": [context[0], context[1]],
        "chunk_seconds": chunk_seconds(context),
        "transcript": transcript,
        "reference_text": None,
        "wer": None,
        "wall_time_seconds": round(wall_time_seconds, 3),
        "gpu": gpu,
        "exit_status": exit_status,
    }


def run_context(
    *,
    command: list[str],
    context: tuple[int, int],
    context_dir: Path,
    checkpoint_sha256: str,
    cuda_index: int | None,
    env: dict[str, str] | None = None,
) -> InferenceRunResult:
    context_dir.mkdir(parents=True, exist_ok=True)
    log_path = context_dir / "inference.log"
    result_path = context_dir / "result.json"
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env if env is not None else os.environ.copy(),
        check=False,
    )
    wall_time = time.perf_counter() - start
    log_path.write_text(completed.stdout, encoding="utf-8")
    transcript = parse_final_transcript(completed.stdout)
    if completed.returncode != 0:
        return InferenceRunResult(context, result_path, log_path, transcript, wall_time, completed.returncode)
    record = build_result_record(
        context=context,
        transcript=transcript,
        wall_time_seconds=wall_time,
        exit_status=completed.returncode,
        checkpoint_sha256=checkpoint_sha256,
        gpu=gpu_name(cuda_index),
    )
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return InferenceRunResult(context, result_path, log_path, transcript, wall_time, completed.returncode)
