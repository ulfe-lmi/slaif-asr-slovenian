from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import atomic_write_jsonl, load_json, load_jsonl, sha256_file
from slaif_asr.gams_retry_controller import AttemptTask, RetryLimits
from slaif_asr.scale200_corpus import (
    TRAINING_VIEWS,
    build_prompt,
    generation_seed,
    load_augmentation_config,
    load_generation_config,
    prompt_cell_by_id,
    stable_sha256,
)


SCALE2000_TEXT_VERSION = "gams-corpus-v4-16000-v1"
COMBINED_CORPUS_ID = "sl-corpus-v4-gams-16000-training-v1"
ADDITION_CORPUS_ID = "sl-corpus-v4-gams-14400-addition-v1"
ANCHOR_CORPUS_ID = "sl-corpus-v3-gams-1600-training-v1"
ANCHOR_TEXT_SHA256 = "9a23df00734193eca0a52bf9b3dae385ff6087d0282529f3f4cb1a28bbf6dccf"
EXPERIMENT_0013_SHA256 = "5128a7d63cb15bb243ad7e54e853de42178e88897c3e9d8d17dcf3d33346f1e1"
SCALE200_BURDEN = 2.868


@dataclass(frozen=True)
class ProtectedConfigFingerprint:
    path: str
    git_blob_sha: str
    byte_sha256: str

    def to_json(self) -> dict[str, str]:
        return {"path": self.path, "git_blob_sha": self.git_blob_sha, "byte_sha256": self.byte_sha256}


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def git_blob_sha(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    completed = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, check=True)
    return completed.stdout.strip()


def protected_config_fingerprints(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    fingerprints = []
    for path_text in paths:
        path = repo_path(path_text)
        fingerprints.append(
            ProtectedConfigFingerprint(
                path=path.relative_to(REPO_ROOT).as_posix(),
                git_blob_sha=git_blob_sha(path),
                byte_sha256=sha256_file(path),
            ).to_json()
        )
    return fingerprints


def load_scale2000_generation_config(path: Path) -> dict[str, Any]:
    config = load_json(repo_path(path))
    validate_scale2000_generation_config(config)
    return config


def load_scale2000_experiment_config(path: Path) -> dict[str, Any]:
    config = load_json(repo_path(path))
    validate_scale2000_experiment_config(config)
    return config


def validate_scale2000_generation_config(config: dict[str, Any]) -> None:
    if config.get("corpus_id") != COMBINED_CORPUS_ID:
        raise ValueError("unexpected combined corpus_id")
    if config.get("new_addition_corpus_id") != ADDITION_CORPUS_ID:
        raise ValueError("unexpected new-addition corpus_id")
    if config.get("partition_role") != "selected_training":
        raise ValueError("partition_role must be selected_training")
    if int(config.get("final_rows", 0)) != 16000:
        raise ValueError("final_rows must be 16000")
    if int(config.get("new_rows", 0)) != 14400:
        raise ValueError("new_rows must be 14400")
    if int(config.get("combined_rows_per_cell", 0)) != 400:
        raise ValueError("combined_rows_per_cell must be 400")
    if int(config.get("inherited_rows_per_cell", 0)) != 40:
        raise ValueError("inherited_rows_per_cell must be 40")
    if int(config.get("new_rows_per_cell", 0)) != 360:
        raise ValueError("new_rows_per_cell must be 360")
    if int(config.get("shards_per_cell", 0)) != 9:
        raise ValueError("shards_per_cell must be 9")
    if int(config.get("requested_rows_per_shard", 0)) != 60:
        raise ValueError("requested_rows_per_shard must be 60")
    if int(config.get("initial_requested_rows", 0)) != 21600:
        raise ValueError("initial_requested_rows must be 21600")
    if int(config.get("maximum_requested_rows", 0)) != 86400:
        raise ValueError("maximum_requested_rows must be 86400")
    model = config.get("model", {})
    if model.get("repository") != "cjvt/GaMS3-12B-Instruct":
        raise ValueError("GaMS repository mismatch")
    if model.get("revision") != "1d0b27af5748784482600d24779409e7e1dc9adc":
        raise ValueError("GaMS revision mismatch")
    quant = config.get("quantization", {})
    if quant.get("load_in_4bit") is not True or quant.get("quant_type") != "nf4":
        raise ValueError("GaMS must use 4-bit NF4")
    if quant.get("double_quantization") is not True or quant.get("compute_dtype") != "bfloat16":
        raise ValueError("GaMS must use double quantization and BF16")
    generation = config.get("generation", {})
    if generation.get("prompt_batch_size") != 8 or generation.get("oom_fallback_batch_size") != 4:
        raise ValueError("prompt batch policy must be 8 with OOM fallback to 4")
    retry = config.get("retry_policy", {})
    expected = {
        "max_verification_rounds": 8,
        "max_attempts_per_shard": 12,
        "max_attempts_per_cell": 48,
        "max_total_attempts": 1440,
        "max_requested_rows": 86400,
        "max_refill_attempts_per_deficient_cell_per_round": 5,
    }
    for key, value in expected.items():
        if int(retry.get(key, -1)) != value:
            raise ValueError(f"retry_policy.{key} must be {value}")
    anchor = config.get("inherited_corpus", {})
    if anchor.get("corpus_id") != ANCHOR_CORPUS_ID or anchor.get("sha256") != ANCHOR_TEXT_SHA256:
        raise ValueError("inherited corpus identity mismatch")
    cells = config.get("prompt_cells", [])
    if len(cells) != 40:
        raise ValueError("exactly forty prompt cells are required")
    seen = set()
    for cell in cells:
        cell_id = str(cell.get("cell_id", ""))
        if cell_id in seen:
            raise ValueError(f"duplicate cell {cell_id}")
        seen.add(cell_id)
        if int(cell.get("requested_rows", 0)) != 60:
            raise ValueError(f"{cell_id}: requested_rows must remain 60")
        for key in ("domain", "register", "length_target", "phenomena", "source_family_id", "prompt_revision"):
            if key not in cell:
                raise ValueError(f"{cell_id}: missing {key}")


def validate_scale2000_experiment_config(config: dict[str, Any]) -> None:
    if config.get("work_order_id") != "0026":
        raise ValueError("experiment config must belong to work order 0026")
    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("status must be DIAGNOSTIC_ONLY")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    training = config.get("training", {})
    required = {
        "arm": "gams16000_nine_voice_augmented_joint_adapter_dim32",
        "batch_size": 8,
        "exposure_rounds": 20,
        "semantic_rows": 16000,
        "sample_exposures": 320000,
        "optimizer_steps": 40000,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "scheduler": "none",
        "gradient_accumulation": "none",
        "gradient_clipping": "none",
        "seed": 1234,
        "precision": "fp32",
        "tf32": False,
        "early_stopping": False,
    }
    for key, expected in required.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    evaluation = config.get("evaluation", {})
    if evaluation.get("policy") != "configs/evaluation/a100-directional-batch32-v1.json":
        raise ValueError("evaluation policy must remain a100 directional batch32")
    if evaluation.get("batch_size") != 32 or evaluation.get("canonical") is not False:
        raise ValueError("evaluation must be noncanonical batch-32")


def retry_limits_from_config(config: dict[str, Any]) -> RetryLimits:
    retry = config["retry_policy"]
    return RetryLimits(
        max_verification_rounds=int(retry["max_verification_rounds"]),
        max_attempts_per_shard=int(retry["max_attempts_per_shard"]),
        max_attempts_per_cell=int(retry["max_attempts_per_cell"]),
        max_total_attempts=int(retry["max_total_attempts"]),
        max_requested_rows=int(retry["max_requested_rows"]),
        requested_rows_per_attempt=int(config["requested_rows_per_shard"]),
        max_refill_attempts_per_cell_per_round=int(retry["max_refill_attempts_per_deficient_cell_per_round"]),
    )


def verify_prompt_cells_match_anchor(config: dict[str, Any], anchor_config_path: Path = Path("configs/generation/gams_corpus_v3_1600_v1.json")) -> None:
    anchor = load_generation_config(anchor_config_path)
    anchor_cells = prompt_cell_by_id(anchor)
    cells = prompt_cell_by_id(config)
    if set(cells) != set(anchor_cells):
        raise ValueError("scale-2000 prompt cells must match scale-200 cell IDs")
    immutable_keys = ("domain", "register", "length_target", "phenomena", "source_family_id")
    for cell_id, cell in cells.items():
        for key in immutable_keys:
            if cell.get(key) != anchor_cells[cell_id].get(key):
                raise ValueError(f"{cell_id}: prompt-cell {key} changed")


def inherited_text_path(config: dict[str, Any]) -> Path:
    return repo_path(config["inherited_corpus"]["text"])


def run_dir(config: dict[str, Any]) -> Path:
    return repo_path(config["run_directory"])


def fixed_combined_text_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "fixed-combined-training-text.local.jsonl"


def new_addition_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "new-addition.local.jsonl"


def generation_state_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "generation-state.local.json"


def retry_history_path(config: dict[str, Any]) -> Path:
    return run_dir(config) / "retry-history.local.jsonl"


def verify_inherited_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = inherited_text_path(config)
    expected = config["inherited_corpus"]
    if sha256_file(path) != expected["sha256"]:
        raise RuntimeError("inherited scale-200 corpus SHA256 mismatch")
    rows = load_jsonl(path)
    if len(rows) != int(expected["rows"]):
        raise RuntimeError("inherited scale-200 row count mismatch")
    counts = counts_by_cell(rows)
    bad = {cell: count for cell, count in counts.items() if count != int(config["inherited_rows_per_cell"])}
    if len(counts) != 40 or bad:
        raise RuntimeError(f"inherited per-cell counts are not exactly 40: {bad}")
    return rows


def counts_by_cell(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("generation", {}).get("prompt_cell", "unknown")) for row in rows)
    return dict(sorted(counts.items()))


def new_candidate_id(cell_id: str, shard_id: str, attempt_index: int, output_ordinal: int) -> str:
    return f"gamsv4-{cell_id}-{shard_id}-a{attempt_index:02d}-o{output_ordinal:03d}"


def build_new_record(
    *,
    config: dict[str, Any],
    cell: dict[str, Any],
    task: AttemptTask,
    output_ordinal: int,
    text: str,
) -> dict[str, Any]:
    cid = new_candidate_id(task.cell_id, task.shard_id, task.attempt_index, output_ordinal)
    return {
        "schema_version": "2.0",
        "candidate_id": cid,
        "language": "sl-SI",
        "spoken_text": text,
        "target_text": text,
        "partition_role": config["partition_role"],
        "source_type": "generated_text",
        "source_id": f"source-{cid}",
        "source_family_id": cell["source_family_id"],
        "template_family_id": None,
        "utterance_family_id": cid,
        "phenomena": list(cell["phenomena"]),
        "domain": cell["domain"],
        "license": config["model"]["license"],
        "generation": {
            "system": "project-generated",
            "method": "gams-local-text-proposal",
            "corpus_id": config["new_addition_corpus_id"],
            "combined_corpus_id": config["corpus_id"],
            "model_repository": config["model"]["repository"],
            "model_revision": config["model"]["revision"],
            "prompt_revision": cell["prompt_revision"],
            "corpus_prompt_revision": config["prompt_revision"],
            "seed": task.seed,
            "prompt_cell": task.cell_id,
            "generation_shard": task.shard_id,
            "generation_attempt": task.attempt_id,
            "verification_round": task.verification_round,
            "extraction_mode": "line",
            "quantization_policy": config["quantization"]["policy"],
        },
        "entities": [],
        "minimal_pair": None,
    }


def build_task_prompt(config: dict[str, Any], task: AttemptTask) -> str:
    cell = prompt_cell_by_id(config)[task.cell_id]
    prompt = build_prompt(cell, requested_rows=task.requested_rows, avoid_openings=task.diversity_guidance)
    if task.shard_id in prompt or task.attempt_id in prompt:
        raise ValueError("shard or attempt identifier leaked into prompt")
    return prompt


def select_new_rows(new_rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in new_rows:
        rows_by_cell[str(row.get("generation", {}).get("prompt_cell", "unknown"))].append(row)
    needed = int(config["new_rows_per_cell"])
    surplus_required = int(config.get("new_surplus_per_cell", 40))
    shortfalls: dict[str, int] = {}
    surplus_shortfalls: dict[str, int] = {}
    for cell_id in sorted(prompt_cell_by_id(config)):
        available = len(rows_by_cell.get(cell_id, []))
        if available < needed:
            shortfalls[cell_id] = needed - available
        if available < needed + surplus_required:
            surplus_shortfalls[cell_id] = needed + surplus_required - available
    if shortfalls:
        raise RuntimeError(f"new-row selection shortfall: {shortfalls}")
    if surplus_shortfalls:
        raise RuntimeError(f"new-row surplus shortfall: {surplus_shortfalls}")
    selected: list[dict[str, Any]] = []
    for cell_id in sorted(prompt_cell_by_id(config)):
        cell_rows = sorted(rows_by_cell[cell_id], key=lambda row: stable_sha256(str(row["candidate_id"])))
        selected.extend(cell_rows[:needed])
    summary = {
        "selected_new_rows": len(selected),
        "new_rows_per_cell": {cell_id: needed for cell_id in sorted(prompt_cell_by_id(config))},
        "admissible_new_per_cell": {cell_id: len(rows_by_cell.get(cell_id, [])) for cell_id in sorted(prompt_cell_by_id(config))},
        "selector": "sha256-candidate-id-ascending-v1",
    }
    return selected, summary


def build_combined_rows(inherited_rows: Sequence[dict[str, Any]], new_rows: Sequence[dict[str, Any]], *, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inherited_counts = counts_by_cell(inherited_rows)
    if any(count != int(config["inherited_rows_per_cell"]) for count in inherited_counts.values()) or len(inherited_counts) != 40:
        raise RuntimeError("inherited rows must remain exactly 40 per cell")
    selected_new, selection_summary = select_new_rows(new_rows, config=config)
    combined = [*inherited_rows, *selected_new]
    combined_counts = counts_by_cell(combined)
    expected = int(config["combined_rows_per_cell"])
    bad = {cell: count for cell, count in combined_counts.items() if count != expected}
    if len(combined) != int(config["final_rows"]) or len(combined_counts) != 40 or bad:
        raise RuntimeError(f"combined corpus count mismatch: rows={len(combined)} bad={bad}")
    combined = sorted(
        combined,
        key=lambda row: (
            str(row["generation"]["prompt_cell"]),
            0 if str(row["candidate_id"]).startswith("gamsv3-") else 1,
            stable_sha256(str(row["candidate_id"])),
        ),
    )
    summary = {
        "inherited_rows": len(inherited_rows),
        "new_rows": len(selected_new),
        "combined_rows": len(combined),
        "per_cell_inherited": inherited_counts,
        "per_cell_new": selection_summary["new_rows_per_cell"],
        "per_cell_combined": combined_counts,
        "selection": selection_summary,
    }
    return combined, summary


def write_combined_rows(config: dict[str, Any], rows: Sequence[dict[str, Any]], new_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fixed_combined_text_path(config).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(fixed_combined_text_path(config), rows)
    atomic_write_jsonl(new_addition_path(config), new_rows)
    return {
        "combined_sha256": sha256_file(fixed_combined_text_path(config)),
        "new_addition_sha256": sha256_file(new_addition_path(config)),
        "combined_rows": len(rows),
        "new_rows": len(new_rows),
    }


def scale2000_multiplier_table() -> dict[str, Any]:
    return {
        "reference_semantic_items": 160,
        "scale200_semantic_items": 1600,
        "scale2000_semantic_items": 16000,
        "semantic_text_multiplier_vs_scale200": 10,
        "semantic_text_multiplier_vs_reference": 100,
        "clean_voice_realizations_per_text": 9,
        "augmentation_views_per_text": 11,
        "views_per_text": 20,
        "clean_files": 144000,
        "augmented_files": 176000,
        "total_view_records": 320000,
        "optimizer_steps": 40000,
        "exposure_multiplier_vs_scale200": 10,
        "exposure_multiplier_vs_reference": 2000,
        "interpretation": "2000x refers to deterministic exposure count, not independent linguistic information.",
    }


def build_scale2000_exposure_schedule(
    text_rows: Sequence[dict[str, Any]],
    augmentation_config: dict[str, Any],
    *,
    batch_size: int = 8,
    seed: int = 1234,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(text_rows) != 16000:
        raise ValueError(f"expected 16000 semantic rows, got {len(text_rows)}")
    clean_voices = list(augmentation_config["clean_voices"])
    if tuple(clean_voices) != TRAINING_VIEWS:
        raise ValueError("clean voices must remain unchanged")
    profiles = list(augmentation_config["augmentation_profiles"])
    if len(profiles) != 11:
        raise ValueError("eleven augmentation profiles are required")
    ordered = sorted(text_rows, key=lambda row: stable_sha256(str(row["candidate_id"])))
    schedule: list[dict[str, Any]] = []
    for round_index, voice in enumerate(clean_voices, start=1):
        for position, row in enumerate(ordered):
            schedule.append(
                {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": row["candidate_id"],
                    "view_type": "clean",
                    "voice": voice,
                    "profile_id": "clean",
                    "spec_augment": False,
                    "batch_order_seed": stable_sha256(f"{seed}:{round_index}:{position}"),
                }
            )
    for profile_index, profile in enumerate(profiles):
        round_index = int(profile["view_round"])
        profile_id = str(profile["profile_id"])
        for position, row in enumerate(ordered):
            voice = clean_voices[(position + profile_index) % len(clean_voices)]
            schedule.append(
                {
                    "round": round_index,
                    "semantic_position": position,
                    "semantic_key": row["candidate_id"],
                    "view_type": "augmented",
                    "voice": voice,
                    "profile_id": profile_id,
                    "spec_augment": ((position + profile_index) % 2) == 0,
                    "batch_order_seed": stable_sha256(f"{seed}:{round_index}:{position}"),
                }
            )
    summary = validate_scale2000_exposure_schedule(schedule, augmentation_config, batch_size=batch_size)
    return schedule, summary


def validate_scale2000_exposure_schedule(
    schedule: Sequence[dict[str, Any]],
    augmentation_config: dict[str, Any],
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    if len(schedule) != 320000:
        raise ValueError("scale-2000 schedule must contain exactly 320000 exposures")
    if len(schedule) % batch_size != 0:
        raise ValueError("schedule must divide evenly into optimizer steps")
    clean_voices = list(augmentation_config["clean_voices"])
    profiles = [str(row["profile_id"]) for row in augmentation_config["augmentation_profiles"]]
    semantic_by_round: dict[int, set[str]] = defaultdict(set)
    voice_counts = Counter(str(row["voice"]) for row in schedule)
    profile_counts = Counter(str(row["profile_id"]) for row in schedule if row["view_type"] == "augmented")
    for row in schedule:
        round_index = int(row["round"])
        key = str(row["semantic_key"])
        if key in semantic_by_round[round_index]:
            raise ValueError(f"duplicate semantic item in round {round_index}")
        semantic_by_round[round_index].add(key)
    issues: list[str] = []
    for round_index in range(1, 21):
        if len(semantic_by_round[round_index]) != 16000:
            issues.append(f"round_{round_index}_semantic_count")
    for voice in clean_voices:
        if voice_counts[voice] == 0:
            issues.append(f"voice_{voice}_missing")
    for profile_id in profiles:
        if profile_counts[profile_id] != 16000:
            issues.append(f"profile_{profile_id}_count")
    for held_out in ("supertonic-M5", "supertonic-F5", "M5", "F5"):
        if voice_counts.get(held_out, 0):
            issues.append(f"heldout_voice_{held_out}_leakage")
    if issues:
        raise ValueError(f"invalid scale-2000 exposure schedule: {issues}")
    return {
        "status": "PASSED",
        "exposures": len(schedule),
        "rounds": 20,
        "optimizer_steps": len(schedule) // batch_size,
        "batch_size": batch_size,
        "clean_voice_counts": {voice: voice_counts[voice] for voice in clean_voices},
        "augmentation_profile_counts": {profile_id: profile_counts[profile_id] for profile_id in profiles},
        "heldout_voice_exposures": {voice: voice_counts.get(voice, 0) for voice in ("supertonic-M5", "supertonic-F5")},
    }


def burden(metrics: dict[str, dict[str, float]], base: dict[str, dict[str, float]]) -> float:
    value = 0.0
    for split in ("fleurs_v2", "artur_j"):
        value += max(0.0, float(metrics[split]["wer"]) - float(base[split]["wer"]))
        value += max(0.0, float(metrics[split]["cer"]) - float(base[split]["cer"]))
    return round(value, 6)


def classify_scale2000(
    *,
    base_metrics: dict[str, dict[str, float]],
    scale200_metrics: dict[str, dict[str, float]],
    scale2000_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    scale2000_burden = burden(scale2000_metrics, base_metrics)
    piper_gain = (
        float(scale2000_metrics["piper_synthetic_holdout"]["wer"]) < float(base_metrics["piper_synthetic_holdout"]["wer"])
        or float(scale2000_metrics["piper_synthetic_holdout"]["cer"]) < float(base_metrics["piper_synthetic_holdout"]["cer"])
    )
    supertonic_gain = (
        float(scale2000_metrics["supertonic_heldout_voice_holdout"]["wer"]) < float(base_metrics["supertonic_heldout_voice_holdout"]["wer"])
        or float(scale2000_metrics["supertonic_heldout_voice_holdout"]["cer"]) < float(base_metrics["supertonic_heldout_voice_holdout"]["cer"])
    )
    real_non_regression = (
        float(scale2000_metrics["fleurs_v2"]["wer"]) - float(base_metrics["fleurs_v2"]["wer"]) <= 1.0
        and float(scale2000_metrics["fleurs_v2"]["cer"]) - float(base_metrics["fleurs_v2"]["cer"]) <= 1.5
        and float(scale2000_metrics["artur_j"]["wer"]) - float(base_metrics["artur_j"]["wer"]) <= 1.0
        and float(scale2000_metrics["artur_j"]["cer"]) - float(base_metrics["artur_j"]["cer"]) <= 1.5
        and int(scale2000_metrics["fleurs_v2"].get("empty", 0)) <= int(base_metrics["fleurs_v2"].get("empty", 0))
        and int(scale2000_metrics["artur_j"].get("empty", 0)) <= int(base_metrics["artur_j"].get("empty", 0))
    )
    real_improvement = (
        float(scale2000_metrics["fleurs_v2"]["wer"]) <= float(base_metrics["fleurs_v2"]["wer"]) - 1.0
        or float(scale2000_metrics["fleurs_v2"]["cer"]) <= float(base_metrics["fleurs_v2"]["cer"]) - 1.5
        or float(scale2000_metrics["artur_j"]["wer"]) <= float(base_metrics["artur_j"]["wer"]) - 1.0
        or float(scale2000_metrics["artur_j"]["cer"]) <= float(base_metrics["artur_j"]["cer"]) - 1.5
    )
    no_metric_more_than_half_worse = True
    no_metric_more_than_one_worse_without_compensation = True
    real_compensation = real_improvement
    for split in ("fleurs_v2", "artur_j"):
        for metric in ("wer", "cer"):
            delta = float(scale2000_metrics[split][metric]) - float(scale200_metrics[split][metric])
            if delta > 0.5:
                no_metric_more_than_half_worse = False
            if delta > 1.0 and not real_compensation:
                no_metric_more_than_one_worse_without_compensation = False
    if piper_gain and supertonic_gain and real_non_regression and real_improvement:
        classification = "SCALE2000_TEXT_REAL_GAIN_DIRECTIONAL"
    elif (
        piper_gain
        and supertonic_gain
        and scale2000_burden <= 2.0076
        and scale2000_burden <= SCALE200_BURDEN * 0.7
        and no_metric_more_than_half_worse
    ):
        classification = "SCALE2000_TEXT_IMPROVES_SCALE200"
    elif piper_gain and supertonic_gain and 2.0076 <= scale2000_burden <= 3.7284 and not real_non_regression:
        classification = "SCALE2000_TEXT_PLATEAUS"
    elif (not piper_gain) or (not supertonic_gain) or scale2000_burden > 3.7284 or not no_metric_more_than_one_worse_without_compensation:
        classification = "SCALE2000_TEXT_DEGRADES"
    else:
        classification = "EXPERIMENT_INVALID"
    return {
        "classification": classification,
        "accepted_parent": "none",
        "piper_holdout_gain": piper_gain,
        "supertonic_holdout_gain": supertonic_gain,
        "real_non_regression": real_non_regression,
        "real_improvement": real_improvement,
        "scale200_burden": SCALE200_BURDEN,
        "scale2000_burden": scale2000_burden,
        "burden_change": round(scale2000_burden - SCALE200_BURDEN, 6),
        "no_metric_more_than_half_point_worse_than_scale200": no_metric_more_than_half_worse,
    }


def verify_scale200_report(path: str | Path, expected_sha256: str = EXPERIMENT_0013_SHA256) -> dict[str, Any]:
    actual = sha256_file(repo_path(path))
    if actual != expected_sha256:
        raise RuntimeError(f"Experiment 0013 report SHA mismatch: {actual}")
    payload = load_json(repo_path(path))
    decision = payload["directional_evaluation"]["decision"]
    if round(float(decision["scale200_burden"]), 3) != SCALE200_BURDEN:
        raise RuntimeError("Experiment 0013 scale-200 burden mismatch")
    return {"sha256": actual, "classification": decision["classification"], "scale200_burden": decision["scale200_burden"]}

