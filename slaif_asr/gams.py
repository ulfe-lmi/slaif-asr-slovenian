from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMA_VERSION = "1.0"
CANDIDATE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,79}$")
SLOVENIAN_HINT_PATTERN = re.compile(r"[čšžČŠŽ]|\b(je|in|sem|si|smo|ste|lahko|prosim|danes)\b", re.IGNORECASE)
MARKDOWN_PATTERN = re.compile(r"(^|\n)\s*(```|[-*]\s+|#{1,6}\s+)")


@dataclass(frozen=True)
class GamsCandidate:
    candidate_id: str
    spoken_text: str
    target_text: str
    language: str
    phenomena: tuple[str, ...]
    source_error_clusters: tuple[str, ...]
    generation_seed: int


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def protected_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(normalize_text(text).casefold().encode("utf-8")).hexdigest()


def parse_strict_json_candidates(text: str) -> list[dict[str, Any]]:
    if MARKDOWN_PATTERN.search(text):
        raise ValueError("GaMS output must be strict JSON without Markdown")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: line {exc.lineno} column {exc.colno}") from exc
    if isinstance(payload, dict) and "candidates" in payload:
        payload = payload["candidates"]
    if not isinstance(payload, list):
        raise ValueError("GaMS output must be a JSON list or an object with candidates")
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("every GaMS candidate must be a JSON object")
    return payload


def validate_gams_candidate(row: dict[str, Any], *, protected_hashes: set[str] | None = None) -> GamsCandidate:
    candidate_id = str(row.get("candidate_id", ""))
    if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate_id: {candidate_id!r}")
    if row.get("language") != "sl-SI":
        raise ValueError(f"{candidate_id}: language must be sl-SI")
    spoken_text = normalize_text(str(row.get("spoken_text", "")))
    target_text = normalize_text(str(row.get("target_text", "")))
    if not spoken_text or not target_text:
        raise ValueError(f"{candidate_id}: spoken_text and target_text are required")
    if spoken_text != row.get("spoken_text") or target_text != row.get("target_text"):
        raise ValueError(f"{candidate_id}: text must be UTF-8 NFC and whitespace-normalized")
    if spoken_text != target_text:
        raise ValueError(f"{candidate_id}: spoken_text must equal target_text")
    if len(spoken_text) > 240:
        raise ValueError(f"{candidate_id}: text exceeds bounded length")
    if not SLOVENIAN_HINT_PATTERN.search(spoken_text):
        raise ValueError(f"{candidate_id}: text does not pass the minimal Slovenian language check")
    phenomena = row.get("phenomena")
    if not isinstance(phenomena, list) or not all(isinstance(item, str) and item for item in phenomena):
        raise ValueError(f"{candidate_id}: phenomena must be a non-empty string list")
    clusters = row.get("source_error_clusters", [])
    if not isinstance(clusters, list) or not all(isinstance(item, str) for item in clusters):
        raise ValueError(f"{candidate_id}: source_error_clusters must be a string list")
    seed = row.get("generation_seed")
    if not isinstance(seed, int):
        raise ValueError(f"{candidate_id}: generation_seed must be an integer")
    if protected_hashes and protected_hash(spoken_text) in protected_hashes:
        raise ValueError(f"{candidate_id}: overlaps a protected evaluation text hash")
    return GamsCandidate(
        candidate_id=candidate_id,
        spoken_text=spoken_text,
        target_text=target_text,
        language="sl-SI",
        phenomena=tuple(phenomena),
        source_error_clusters=tuple(clusters),
        generation_seed=seed,
    )


def validate_candidate_batch(
    rows: list[dict[str, Any]],
    *,
    protected_hashes: set[str] | None = None,
    forbidden_texts: set[str] | None = None,
) -> tuple[list[GamsCandidate], list[str]]:
    valid: list[GamsCandidate] = []
    rejected: list[str] = []
    seen_ids: set[str] = set()
    seen_texts = {normalize_text(text).casefold() for text in (forbidden_texts or set())}
    for index, row in enumerate(rows, start=1):
        try:
            candidate = validate_gams_candidate(row, protected_hashes=protected_hashes)
            text_key = candidate.spoken_text.casefold()
            if candidate.candidate_id in seen_ids:
                raise ValueError(f"duplicate candidate_id {candidate.candidate_id}")
            if text_key in seen_texts:
                raise ValueError(f"{candidate.candidate_id}: duplicate or protected text")
            if is_near_duplicate(text_key, seen_texts):
                raise ValueError(f"{candidate.candidate_id}: near-duplicate text")
        except Exception as exc:
            rejected.append(f"row {index}: {exc}")
            continue
        seen_ids.add(candidate.candidate_id)
        seen_texts.add(text_key)
        valid.append(candidate)
    return valid, rejected


def token_set(text: str) -> set[str]:
    return {item for item in re.split(r"\W+", text.casefold()) if item}


def is_near_duplicate(text: str, existing_texts: set[str], *, threshold: float = 0.9) -> bool:
    tokens = token_set(text)
    if not tokens:
        return False
    for existing in existing_texts:
        other = token_set(existing)
        if not other:
            continue
        score = len(tokens.intersection(other)) / len(tokens.union(other))
        if score >= threshold:
            return True
    return False


def load_generation_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)
    validate_generation_config(config)
    return config


def validate_generation_config(config: dict[str, Any]) -> None:
    primary = config.get("primary_model", {})
    fallback = config.get("fallback_model", {})
    if primary.get("repository") != "cjvt/GaMS3-12B-Instruct":
        raise ValueError("primary GaMS repository must be pinned to cjvt/GaMS3-12B-Instruct")
    if fallback.get("repository") != "cjvt/GaMS-9B-Instruct":
        raise ValueError("fallback GaMS repository must be pinned to cjvt/GaMS-9B-Instruct")
    for label, model in (("primary", primary), ("fallback", fallback)):
        revision = model.get("revision")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{label} GaMS revision must be a 40-character commit SHA")
    quant = config.get("quantization", {})
    if quant.get("load_in_4bit") is not True or quant.get("quant_type") != "nf4":
        raise ValueError("GaMS generator must use 4-bit NF4 quantization")
    if quant.get("double_quantization") is not True or quant.get("compute_dtype") != "float16":
        raise ValueError("GaMS generator must use double quantization with FP16 compute")
    if config.get("device_policy", {}).get("cpu_offload") is not False:
        raise ValueError("GaMS CPU offload is forbidden")
    if config.get("device_policy", {}).get("visible_gpu_count") != 1:
        raise ValueError("GaMS must run with one visible GPU")
