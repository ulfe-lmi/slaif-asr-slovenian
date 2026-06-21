from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG_PATH = REPO_ROOT / "configs" / "runtime" / "nemotron_3_5_asr.json"


@lru_cache(maxsize=1)
def load_runtime_config() -> dict[str, Any]:
    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def repo_path(config_key: str) -> Path:
    current: Any = load_runtime_config()
    for part in config_key.split("."):
        current = current[part]
    return REPO_ROOT / current


def streaming_contexts() -> list[tuple[int, int]]:
    return [tuple(item["att_context_size"]) for item in load_runtime_config()["streaming_contexts"]]
