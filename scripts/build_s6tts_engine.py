#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slaif_asr.s6tts_tts import CONFIG_PATH, PINNED_REVISION, load_s6_config, s6_paths, smoke_rows, synthesize_one
from slaif_asr.tts import atomic_write_json


def run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(json.dumps({"stage": "command", "cmd": command, "cwd": str(cwd) if cwd else None}), flush=True)
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout)
    return completed


def tool(name: str) -> str:
    system_path = Path("/usr/bin") / name
    if system_path.exists():
        return str(system_path)
    resolved = shutil.which(name)
    if resolved is None:
        raise FileNotFoundError(name)
    return resolved


def main() -> int:
    config = load_s6_config(CONFIG_PATH)
    paths = s6_paths(config)
    paths.source_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (paths.source_dir / ".git").exists():
        run(["git", "clone", "https://github.com/ulfe-lmi/s6tts.git", str(paths.source_dir)])
    else:
        run(["git", "fetch", "origin"], cwd=paths.source_dir)
    run(["git", "checkout", PINNED_REVISION], cwd=paths.source_dir)
    revision = run(["git", "rev-parse", "HEAD"], cwd=paths.source_dir).stdout.strip()
    if revision != PINNED_REVISION:
        raise RuntimeError(f"S6TTS revision mismatch: {revision}")
    started = time.perf_counter()
    cmake = tool("cmake")
    ctest_bin = tool("ctest")
    run([cmake, "-S", ".", "-B", "build", "-DS6TTS_WITH_ALSA=OFF", "-DS6TTS_BUILD_TESTS=ON"], cwd=paths.source_dir)
    run([cmake, "--build", "build"], cwd=paths.source_dir)
    ctest = run([ctest_bin, "--test-dir", "build", "--output-on-failure"], cwd=paths.source_dir)
    if not paths.cli_path.exists():
        raise FileNotFoundError(paths.cli_path)
    rows = smoke_rows()
    smoke_root = paths.run_root / "build-smoke"
    smoke_paths = paths.__class__(
        source_dir=paths.source_dir,
        build_dir=paths.build_dir,
        cli_path=paths.cli_path,
        runtime_ini=paths.runtime_ini,
        run_root=smoke_root,
        audio_manifest=smoke_root / "audio-manifest.local.jsonl",
        provenance_manifest=smoke_root / "provenance.local.jsonl",
        validation=smoke_root / "audio-validation.local.json",
        summary=smoke_root / "summary.local.json",
        logs_dir=smoke_root / "logs",
    )
    rendered = [synthesize_one(smoke_paths, row, overwrite=True) for row in rows]
    summary = {
        "schema_version": "1.0",
        "status": "passed",
        "s6tts_revision": revision,
        "cli_path": str(paths.cli_path),
        "ctest_output": ctest.stdout.strip().splitlines()[-10:],
        "smoke_files": len(rendered),
        "wall_time_seconds": round(time.perf_counter() - started, 3),
    }
    atomic_write_json(paths.run_root / "build-summary.local.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cli_path"}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
