from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.artur_controller_dev import select_earliest_within_tolerance
from slaif_asr.config import REPO_ROOT
from slaif_asr.emission_rnnt_finetune import (
    BASE_DIRECTIONAL_METRICS,
    EXPECTED_ALL_VIEWS_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    EXPECTED_TEXT_SHA256,
)


SURFACE_ID = "SURFACE_04_DECODER_JOINT_PLUS_LAST_ENCODER_BLOCK"
FINAL_ENCODER_BLOCK_PREFIX = "encoder.layers.23."
ALLOWED_TRAINABLE_PREFIXES = ("decoder.", "joint.", FINAL_ENCODER_BLOCK_PREFIX)
FORBIDDEN_PUBLIC_KEYS = {
    "audio_filepath",
    "checkpoint_path",
    "hypothesis",
    "hypotheses",
    "local_path",
    "reference",
    "references",
    "text",
}
FORBIDDEN_PUBLIC_MARKERS = ("/home/", "/tmp/", "/data-nvme/", ".wav", ".nemo", ".ckpt")

PR36_METRICS = {
    "piper_synthetic_holdout": {"wer": 34.317, "cer": 13.765, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 14.752, "cer": 4.682, "empty": 0},
    "fleurs_v2": {"wer": 46.195, "cer": 15.604, "empty": 0},
    "artur_j": {"wer": 56.793, "cer": 20.177, "empty": 0},
}


@dataclass(frozen=True)
class SurfaceSummary:
    surface_id: str
    final_encoder_block: str
    trainable_parameter_count: int
    decoder_parameter_count: int
    joint_parameter_count: int
    final_encoder_block_parameter_count: int
    frozen_parameter_count: int
    trainable_prefixes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path = "configs/experiments/fixed-scale2000-surface04-last-encoder-block.json") -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any], *, adr_text: str | None = None) -> None:
    if config.get("work_order_id") != "0037":
        raise ValueError("work_order_id must be 0037")
    if config.get("status") != "DIAGNOSTIC_ONLY" or config.get("accepted_parent") != "none":
        raise ValueError("surface sweep must remain DIAGNOSTIC_ONLY with accepted_parent none")
    if adr_text is None:
        adr_path = REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md"
        if not adr_path.exists():
            raise ValueError("ADR 0009 is required")
        adr_text = adr_path.read_text(encoding="utf-8")
    if "Authorize a fixed-data trainable-surface sweep" not in adr_text or SURFACE_ID not in adr_text:
        raise ValueError("ADR 0009 does not authorize SURFACE_04")

    data = config.get("data", {})
    expected_data = {
        "corpus_id": "sl-corpus-v4-gams-16000-training-v1",
        "semantic_rows": 16000,
        "exposure_records": 320000,
        "fixed_text_sha256": EXPECTED_TEXT_SHA256,
        "all_views_sha256": EXPECTED_ALL_VIEWS_SHA256,
        "exposure_schedule_sha256": EXPECTED_SCHEDULE_SHA256,
    }
    for key, expected in expected_data.items():
        if data.get(key) != expected:
            raise ValueError(f"data.{key} must be {expected!r}")
    serialized_data = json.dumps(data, sort_keys=True).lower()
    if "s6tts" in serialized_data or "scale8000" in serialized_data or "scale-8000" in serialized_data:
        raise ValueError("fixed-data sweep forbids S6TTS and scale-8000 data")

    surface = config.get("trainable_surface", {})
    if surface.get("surface_id") != SURFACE_ID:
        raise ValueError("Phase 1 permits only SURFACE_04")
    if surface.get("encoder_layer_count") != 24 or surface.get("final_encoder_layer_index") != 23:
        raise ValueError("SURFACE_04 must select only final encoder layer 23 of 24")
    if tuple(surface.get("allowed_prefixes", ())) != ALLOWED_TRAINABLE_PREFIXES:
        raise ValueError("trainable prefixes must be decoder, joint, and encoder.layers.23 only")
    if surface.get("full_encoder_allowed") is not False:
        raise ValueError("full encoder training is prohibited")
    if surface.get("text_only_objective_allowed") is not False or surface.get("temporary_lm_head_allowed") is not False:
        raise ValueError("text-only training and temporary LM heads are prohibited")

    training = config.get("training", {})
    expected_training = {
        "objective": "audio_conditioned_rnnt",
        "effective_batch_size": 8,
        "round_size_exposures": 16000,
        "steps_per_round": 2000,
        "max_rounds": 20,
        "max_exposures": 320000,
        "max_optimizer_steps": 40000,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "scheduler": "none",
        "precision": "fp32",
        "tf32": False,
        "seed": 1234,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if training.get("physical_microbatch_candidates") != [4, 2, 1]:
        raise ValueError("physical microbatch candidates must be [4, 2, 1]")
    lrs = training.get("learning_rates", {})
    if lrs != {"decoder": 0.0005, "joint": 0.0005, "final_encoder_block": 0.00002}:
        raise ValueError("learning-rate groups do not match Phase 1")
    if not float(lrs["final_encoder_block"]) < float(lrs["decoder"]):
        raise ValueError("encoder learning rate must be lower than decoder/joint")

    control = config.get("controller_dev", {})
    if control.get("partition_id") != "artur-controller-dev-v1" or control.get("batch_size") != 1:
        raise ValueError("ARTUR controller-dev batch-1 policy is required")
    if control.get("duration_bucketing") is not False or control.get("immutable_gate_selection_allowed") is not False:
        raise ValueError("controller policy or immutable-gate isolation drifted")


def configure_surface04_trainable(model: Any) -> SurfaceSummary:
    for name, _module in model.named_modules():
        lowered = name.lower()
        if "lm_head" in lowered or "lm_adapter" in lowered or "decoder_lm" in lowered:
            raise RuntimeError(f"forbidden text-only module present: {name}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    counts = {"decoder": 0, "joint": 0, "final_encoder_block": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            parameter.requires_grad_(True)
            counts["decoder"] += parameter.numel()
        elif name.startswith("joint."):
            parameter.requires_grad_(True)
            counts["joint"] += parameter.numel()
        elif name.startswith(FINAL_ENCODER_BLOCK_PREFIX):
            parameter.requires_grad_(True)
            counts["final_encoder_block"] += parameter.numel()
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"required SURFACE_04 parameter group missing: {counts}")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(ALLOWED_TRAINABLE_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"unauthorized trainable parameters: {unexpected[:10]}")
    frozen = sum(parameter.numel() for _name, parameter in model.named_parameters() if not parameter.requires_grad)
    total = sum(counts.values())
    return SurfaceSummary(
        surface_id=SURFACE_ID,
        final_encoder_block="encoder.layers.23",
        trainable_parameter_count=total,
        decoder_parameter_count=counts["decoder"],
        joint_parameter_count=counts["joint"],
        final_encoder_block_parameter_count=counts["final_encoder_block"],
        frozen_parameter_count=frozen,
        trainable_prefixes=ALLOWED_TRAINABLE_PREFIXES,
    )


def set_surface04_training_mode(model: Any) -> None:
    # Match PR #36 training-mode semantics; requires_grad is the only changed
    # surface control. Evaluation probes switch the entire model to eval mode.
    model.train()


def optimizer_parameter_groups(model: Any, learning_rates: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = {"decoder": [], "joint": [], "final_encoder_block": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder."):
            groups["decoder"].append(parameter)
        elif name.startswith("joint."):
            groups["joint"].append(parameter)
        elif name.startswith(FINAL_ENCODER_BLOCK_PREFIX):
            groups["final_encoder_block"].append(parameter)
        else:
            raise RuntimeError(f"unauthorized optimizer parameter: {name}")
    if any(not values for values in groups.values()):
        raise RuntimeError("optimizer is missing a SURFACE_04 parameter group")
    return [
        {"name": name, "params": groups[name], "lr": float(learning_rates[name])}
        for name in ("decoder", "joint", "final_encoder_block")
    ]


def verify_optimizer_scope(optimizer: Any, model: Any, learning_rates: dict[str, float]) -> None:
    expected = {id(parameter) for _name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer parameters do not exactly match SURFACE_04")
    by_name = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
    if by_name != {name: float(value) for name, value in learning_rates.items()}:
        raise RuntimeError("optimizer learning-rate groups drifted")


def changed_tensor_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed: list[str] = []
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        if left.shape != right.shape or not bool((left == right).all()):
            changed.append(name)
    unauthorized = [name for name in changed if not name.startswith(ALLOWED_TRAINABLE_PREFIXES)]
    return {
        "changed_tensor_count": len(changed),
        "authorized_changed_tensor_count": len(changed) - len(unauthorized),
        "unauthorized_changed_tensors": unauthorized,
        "missing_tensors": missing,
        "unexpected_tensors": extra,
        "lower_encoder_unchanged": not any(
            name.startswith("encoder.") and not name.startswith(FINAL_ENCODER_BLOCK_PREFIX) for name in changed
        ),
        "preprocessor_unchanged": not any(name.startswith("preprocessor.") for name in changed),
        "prompt_path_unchanged": not any("prompt" in name.lower() for name in changed),
        "only_surface04_changed": not missing and not extra and not unauthorized,
    }


def microbatch_plan(physical_microbatch: int) -> dict[str, int]:
    if physical_microbatch not in (4, 2, 1):
        raise ValueError("physical microbatch must be one of 4, 2, 1")
    return {
        "physical_microbatch": physical_microbatch,
        "gradient_accumulation_steps": 8 // physical_microbatch,
        "effective_batch_size": 8,
    }


def select_microbatch(outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for candidate in (4, 2, 1):
        if outcomes.get(candidate, {}).get("status") == "PASSED":
            return {"status": "PASSED", **microbatch_plan(candidate)}
    return {
        "status": "BLOCKED_SURFACE04_OOM",
        "physical_microbatch": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": 8,
    }


def should_stop_controller_curve(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    post = [row for row in rows if int(row["round"]) > 0]
    if len(post) < 3:
        return {"stop": False, "reason": "minimum_rounds_not_reached"}
    if any(int(row["round"]) > 1 and int(row.get("empty", 0)) > 0 for row in post):
        return {"stop": True, "reason": "empty_hypotheses_reappeared"}

    best_wer = float("inf")
    best_round = 0
    for row in post:
        wer = float(row["wer"])
        if wer < best_wer:
            best_wer = wer
            best_round = int(row["round"])
    last_round = int(post[-1]["round"])
    regressions = [float(row["wer"]) > best_wer + 5.0 for row in post[-2:]]
    if len(regressions) == 2 and all(regressions):
        return {"stop": True, "reason": "two_consecutive_catastrophic_wer_regressions", "best_round": best_round}
    if last_round - best_round >= 3:
        return {"stop": True, "reason": "three_rounds_without_new_raw_best", "best_round": best_round}
    return {"stop": False, "reason": "new_best_patience_active", "best_round": best_round}


def mark_controller_selection(rows: Sequence[dict[str, Any]], *, base_empty_count: int) -> dict[str, Any]:
    selected = select_earliest_within_tolerance(rows, base_empty_count=base_empty_count)
    available = [row for row in rows if row.get("available") and row.get("wer") is not None and row.get("cer") is not None]
    if not available:
        return {"rows": [dict(row) for row in rows], "selected_round": None, "best_raw_wer_round": None}

    best = min(available, key=lambda row: (float(row["wer"]), float(row["cer"]), int(row["round"])))
    best_wer = float(best["wer"])
    best_cer = float(best["cer"])
    marked = []
    for source in rows:
        row = dict(source)
        row["eligible"] = bool(
            row.get("available")
            and row.get("wer") is not None
            and row.get("cer") is not None
            and float(row["wer"]) <= best_wer + 0.50
            and float(row["cer"]) <= best_cer + 0.25
            and int(row.get("empty", 0)) <= base_empty_count
        )
        row["selected_by_rule"] = bool(selected and int(row["round"]) == int(selected["round"]))
        marked.append(row)
    return {
        "rows": marked,
        "selected_round": int(selected["round"]) if selected else None,
        "best_raw_wer_round": int(best["round"]),
    }


def component_or_not_recorded(metrics: dict[str, Any], key: str) -> int | float | str:
    value = metrics.get(key)
    return "NOT_RECORDED" if value is None else value


def bind_post_selection_metrics(selected_round: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"selected_round": int(selected_round), "directional_metrics": metrics}


def classify_surface04(
    metrics: dict[str, dict[str, Any]],
    *,
    parameter_integrity: bool = True,
    selected_round: int | None = None,
) -> str:
    if not parameter_integrity:
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    if any(int(metrics[split].get("empty", 0)) > 0 for split in real_splits):
        return "SURFACE04_SYNTHETIC_OR_REAL_REGRESSION"
    synthetic_safe = all(
        float(metrics[split][metric]) < float(BASE_DIRECTIONAL_METRICS[split][metric])
        for split in ("piper_synthetic_holdout", "supertonic_heldout_voice_holdout")
        for metric in ("wer", "cer")
    )
    base_better = all(
        float(metrics[split][metric]) < float(BASE_DIRECTIONAL_METRICS[split][metric])
        for split in real_splits
        for metric in ("wer", "cer")
    )
    beats_pr36 = all(
        float(metrics[split][metric]) < float(PR36_METRICS[split][metric])
        for split in real_splits
        for metric in ("wer", "cer")
    )
    if beats_pr36 and synthetic_safe:
        return "SURFACE04_BEATS_PR36_DIRECTIONAL"
    within = all(
        abs(float(metrics[split][metric]) - float(PR36_METRICS[split][metric]))
        <= (0.50 if metric == "wer" else 0.25)
        for split in real_splits
        for metric in ("wer", "cer")
    )
    improves_one = any(
        float(metrics[split][metric]) < float(PR36_METRICS[split][metric])
        for split in real_splits
        for metric in ("wer", "cer")
    )
    if within and improves_one and synthetic_safe:
        return "SURFACE04_MATCHES_PR36_WITH_ACCEPTABLE_TRADEOFF"
    if base_better:
        return "SURFACE04_BEATS_BASE_BUT_NOT_PR36"
    if selected_round is not None and selected_round > 0:
        return "SURFACE04_ARTUR_DEV_GOOD_BUT_GATE_DIRECTIONAL_REGRESSES"
    return "SURFACE04_SYNTHETIC_OR_REAL_REGRESSION"


def assert_public_report_safe(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in FORBIDDEN_PUBLIC_KEYS:
                    raise ValueError(f"public report contains forbidden key: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    for marker in FORBIDDEN_PUBLIC_MARKERS:
        if marker in serialized:
            raise ValueError(f"public report contains forbidden marker: {marker}")


def trainable_names(model: Any) -> Iterable[str]:
    return (name for name, parameter in model.named_parameters() if parameter.requires_grad)
