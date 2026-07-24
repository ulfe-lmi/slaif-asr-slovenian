from __future__ import annotations

import json
from typing import Any


CANDIDATE_IDS = ("base", "pr36_round20", "surface06_round05", "surface07_round13")
REAL_GATE_SPLITS = ("fleurs_v2", "artur_j")
METRIC_KEYS = (
    ("fleurs_v2", "wer", 0.50),
    ("fleurs_v2", "cer", 0.25),
    ("artur_j", "wer", 0.50),
    ("artur_j", "cer", 0.25),
)
SURFACE07_SHA256 = "349d06dd517b6e99b71a74f15a04d6020afe56223ef946014d3bdca1440706b0"
PUBLIC_FORBIDDEN_KEYS = {
    "audio_filepath",
    "command",
    "hypothesis",
    "hypotheses",
    "local_manifest",
    "local_path",
    "manifest_path",
    "prediction",
    "predictions",
    "reference",
    "references",
    "sample_id",
    "sample_ids",
    "text",
}
PUBLIC_FORBIDDEN_MARKERS = (
    "/data/",
    "/home/",
    "/mnt/",
    "/synology/",
    "artur-controller-dev",
    "predictions.local",
)


def validate_canonical_config(config: dict[str, Any], policy: dict[str, Any]) -> None:
    expected_policy = {
        "canonical": True,
        "promotion_eligible": False,
        "batch_size": 1,
        "duration_bucketing": False,
        "att_context_size": [56, 3],
        "target_lang": "sl-SI",
        "normalization": "sl-asr-normalization-v1",
        "precision": "fp32",
        "tf32": False,
        "visible_cuda_devices": 1,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"canonical policy {key} must be {expected!r}")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    for key in ("promotion_eligible", "training_eligible", "checkpoint_accepted", "model_published"):
        if config.get(key) is not False:
            raise ValueError(f"{key} must be false")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or tuple(row.get("candidate_id") for row in candidates) != CANDIDATE_IDS:
        raise ValueError("canonical candidate order or identity mismatch")
    surface07 = candidates[-1]
    if surface07.get("checkpoint_sha256") != SURFACE07_SHA256 or surface07.get("required") is not True:
        raise ValueError("Surface07 round 13 identity must be required and exact")
    gates = config.get("gates", {})
    if set(gates) != set(REAL_GATE_SPLITS):
        raise ValueError("canonical evaluation must contain exactly FLEURS-v2 and ARTUR-J")
    serialized = json.dumps(config, sort_keys=True)
    if "artur-controller-dev" in serialized:
        raise ValueError("ARTUR controller-dev is forbidden in canonical evaluation")


def metric_row(split_summary: dict[str, Any]) -> dict[str, Any]:
    normalized = split_summary["metrics"]["normalized"]
    raw = split_summary["metrics"]["raw"]
    return {
        "wer": round(float(normalized["corpus_wer"]), 3),
        "cer": round(float(normalized["corpus_cer"]), 3),
        "empty": int(raw["empty_hypothesis_count"]),
    }


def classify_canonical(metrics: dict[str, dict[str, dict[str, Any]]]) -> str:
    surface07 = metrics.get("surface07_round13")
    if surface07 is None:
        return "CANONICAL_BLOCKED_SURFACE07_CHECKPOINT_UNAVAILABLE"
    base = metrics.get("base")
    if base is None:
        return "EXPERIMENT_INVALID"

    improves_base = all(
        float(surface07[split][metric]) < float(base[split][metric])
        for split, metric, _tolerance in METRIC_KEYS
    )
    empty_safe = all(int(surface07[split]["empty"]) <= int(base[split]["empty"]) for split in REAL_GATE_SPLITS)
    if not improves_base or not empty_safe:
        return "CANONICAL_REJECTS_SURFACE07_DIRECTIONAL_GAIN"

    challengers = [
        metrics[candidate_id]
        for candidate_id in ("pr36_round20", "surface06_round05")
        if candidate_id in metrics
    ]
    if not challengers:
        return "CANONICAL_SURFACE07_CONFIRMED_NEW_BEST"

    surface07_best_count = 0
    within_tolerance = True
    for split, metric, tolerance in METRIC_KEYS:
        value = float(surface07[split][metric])
        best_prior = min(float(challenger[split][metric]) for challenger in challengers)
        if value < best_prior:
            surface07_best_count += 1
        if value > best_prior + tolerance:
            within_tolerance = False
    if surface07_best_count >= 3 and within_tolerance:
        return "CANONICAL_SURFACE07_CONFIRMED_NEW_BEST"

    for challenger in challengers:
        prior_wins = sum(
            float(challenger[split][metric]) < float(surface07[split][metric])
            for split, metric, _tolerance in METRIC_KEYS
        )
        if prior_wins >= 3:
            return "CANONICAL_SURFACE06_OR_PRIOR_STRONGER"
    return "CANONICAL_SURFACE07_CONFIRMED_BUT_MIXED"


def assert_public_report_safe(payload: Any) -> None:
    def walk(value: Any, key: str = "") -> None:
        if key in PUBLIC_FORBIDDEN_KEYS:
            raise ValueError(f"public report contains forbidden key: {key}")
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if any(marker in serialized for marker in PUBLIC_FORBIDDEN_MARKERS):
        raise ValueError("public report contains a forbidden local or controller-dev marker")
