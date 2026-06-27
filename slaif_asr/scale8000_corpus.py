from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from slaif_asr.config import REPO_ROOT
from slaif_asr.data_quality import load_json, load_jsonl, sha256_file
from slaif_asr.gams_retry_controller import AttemptTask
from slaif_asr.scale200_corpus import TRAINING_VIEWS, load_augmentation_config, stable_sha256
from slaif_asr.scale2000_corpus import counts_by_cell as scale2000_counts_by_cell


SCALE8000_TEXT_VERSION = "gams-corpus-v5-scale8000-v1"
SCALE8000_CORPUS_ID = "sl-corpus-v5-scale8000-training-v1"
SCALE8000_ADDITION_CORPUS_ID = "sl-corpus-v5-scale8000-addition-v1"
INHERITED_SCALE2000_CORPUS_ID = "sl-corpus-v4-gams-16000-training-v1"
INHERITED_SCALE2000_SHA256 = "dd38cf0ac0e36abc14559b379319bed0b27c2929e1342b6fc9bbeb0eed7efe14"


@dataclass(frozen=True)
class Scale8000Plan:
    semantic_rows: int
    inherited_rows: int
    new_rows: int
    clean_files: int
    augmented_files: int
    total_views: int
    optimizer_steps_at_batch8: int
    exposure_multiplier_vs_reference: int

    def to_json(self) -> dict[str, int]:
        return {
            "semantic_rows": self.semantic_rows,
            "inherited_rows": self.inherited_rows,
            "new_rows": self.new_rows,
            "clean_files": self.clean_files,
            "augmented_files": self.augmented_files,
            "total_views": self.total_views,
            "optimizer_steps_at_batch8": self.optimizer_steps_at_batch8,
            "exposure_multiplier_vs_reference": self.exposure_multiplier_vs_reference,
        }


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def load_scale8000_generation_config(path: str | Path = "configs/generation/gams_corpus_v5_scale8000_v1.json") -> dict[str, Any]:
    config = load_json(repo_path(path))
    validate_scale8000_generation_config(config)
    return config


def load_scale8000_experiment_config(path: str | Path = "configs/experiments/scale8000_dual_gpu_generation_v1.json") -> dict[str, Any]:
    config = load_json(repo_path(path))
    validate_scale8000_experiment_config(config)
    return config


def validate_scale8000_generation_config(config: dict[str, Any]) -> None:
    expected = {
        "corpus_id": SCALE8000_CORPUS_ID,
        "new_addition_corpus_id": SCALE8000_ADDITION_CORPUS_ID,
        "partition_role": "selected_training",
        "final_rows": 64000,
        "new_rows": 48000,
        "combined_rows_per_cell": 1600,
        "inherited_rows_per_cell": 400,
        "new_rows_per_cell": 1200,
        "new_surplus_per_cell": 120,
        "shards_per_cell": 30,
        "requested_rows_per_shard": 60,
        "initial_requested_rows": 72000,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    model = config.get("model", {})
    if model.get("repository") != "cjvt/GaMS3-12B-Instruct":
        raise ValueError("GaMS repository mismatch")
    if model.get("revision") != "1d0b27af5748784482600d24779409e7e1dc9adc":
        raise ValueError("GaMS revision mismatch")
    retry = config.get("retry_policy", {})
    if retry.get("retry_until_valid") is not True:
        raise ValueError("scale-8000 retry policy must run until valid")
    for key in ("max_verification_rounds", "max_attempts_per_shard", "max_attempts_per_cell", "max_total_attempts", "max_requested_rows"):
        if retry.get(key) is not None:
            raise ValueError(f"{key} must be null under the approved unlimited retry policy")
    if int(retry.get("requested_rows_per_attempt", 0)) != 60:
        raise ValueError("retry attempts must continue to request 60 rows")
    inherited = config.get("inherited_corpus", {})
    if inherited.get("corpus_id") != INHERITED_SCALE2000_CORPUS_ID:
        raise ValueError("inherited corpus ID mismatch")
    if inherited.get("sha256") != INHERITED_SCALE2000_SHA256:
        raise ValueError("inherited corpus SHA mismatch")
    if inherited.get("rows") != 16000 or inherited.get("rows_per_cell") != 400:
        raise ValueError("inherited scale-2000 count mismatch")
    gpu = config.get("device_policy", {})
    if gpu.get("authorized_physical_gpus") != [0, 1]:
        raise ValueError("scale-8000 must authorize GPU0 and GPU1")
    if gpu.get("single_visible_gpu_per_worker") is not True:
        raise ValueError("workers must use one visible logical CUDA device")


def validate_scale8000_experiment_config(config: dict[str, Any]) -> None:
    if config.get("work_order_id") != "0027":
        raise ValueError("work_order_id must be 0027")
    if config.get("status") != "DIAGNOSTIC_ONLY":
        raise ValueError("status must remain DIAGNOSTIC_ONLY")
    if config.get("accepted_parent") != "none":
        raise ValueError("accepted_parent must remain none")
    target = config.get("target_scale8000", {})
    plan = scale8000_plan()
    for key, value in plan.to_json().items():
        config_key = "total_views" if key == "total_views" else key
        if target.get(config_key) != value:
            raise ValueError(f"target_scale8000.{config_key} must be {value}")
    if config.get("dual_gpu_policy", {}).get("authorized_physical_gpus") != [0, 1]:
        raise ValueError("dual GPU policy must name GPU0 and GPU1")


def prompt_cells(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = load_json(repo_path(config["prompt_cells_source"]))
    cells = list(source.get("prompt_cells", []))
    if len(cells) != 40:
        raise ValueError("scale-8000 must reuse forty prompt cells")
    return cells


def scale8000_plan() -> Scale8000Plan:
    semantic_rows = 64000
    clean = semantic_rows * 9
    augmented = semantic_rows * 11
    views = clean + augmented
    return Scale8000Plan(
        semantic_rows=semantic_rows,
        inherited_rows=16000,
        new_rows=48000,
        clean_files=clean,
        augmented_files=augmented,
        total_views=views,
        optimizer_steps_at_batch8=views // 8,
        exposure_multiplier_vs_reference=views // 160,
    )


def scale8000_multiplier_table() -> dict[str, Any]:
    plan = scale8000_plan()
    return {
        **plan.to_json(),
        "reference_semantic_items": 160,
        "scale2000_semantic_items": 16000,
        "semantic_text_multiplier_vs_scale2000": 4,
        "semantic_text_multiplier_vs_reference": 400,
        "clean_voice_realizations_per_text": 9,
        "augmentation_views_per_text": 11,
        "views_per_text": 20,
        "interpretation": "8000x refers to deterministic exposure count, not independent linguistic information.",
    }


def verify_inherited_scale2000_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    inherited = config["inherited_corpus"]
    path = repo_path(inherited["text"])
    if sha256_file(path) != inherited["sha256"]:
        raise RuntimeError("inherited scale-2000 text SHA mismatch")
    rows = load_jsonl(path)
    if len(rows) != int(inherited["rows"]):
        raise RuntimeError("inherited scale-2000 row-count mismatch")
    counts = scale2000_counts_by_cell(rows)
    if len(counts) != 40 or any(count != int(inherited["rows_per_cell"]) for count in counts.values()):
        raise RuntimeError("inherited scale-2000 per-cell counts changed")
    return rows


def counts_by_cell(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(row.get("generation", {}).get("prompt_cell", "unknown")) for row in rows))


def build_new_record(config: dict[str, Any], cell: dict[str, Any], task: AttemptTask, output_ordinal: int, text: str) -> dict[str, Any]:
    candidate_id = f"gamsv5-{task.cell_id}-{task.shard_id}-a{task.attempt_index:02d}-o{output_ordinal:03d}-{stable_sha256(f'{task.attempt_id}:{output_ordinal}')[:12]}"
    return {
        "schema_version": "2.0",
        "candidate_id": candidate_id,
        "language": "sl-SI",
        "partition_role": config["partition_role"],
        "source_type": config["source_type"],
        "spoken_text": text,
        "target_text": text,
        "source_id": candidate_id,
        "source_family_id": f"scale8000-{cell['source_family_id']}",
        "template_family_id": None,
        "minimal_pair": None,
        "utterance_family_id": candidate_id,
        "domain": cell.get("domain"),
        "phenomena": cell.get("phenomena", []),
        "generation": {
            "model": config["model"]["repository"],
            "revision": config["model"]["revision"],
            "prompt_revision": config["prompt_revision"],
            "prompt_cell": cell["cell_id"],
            "generation_shard": task.shard_id,
            "attempt_index": task.attempt_index,
            "attempt_id": task.attempt_id,
            "seed": task.seed,
        },
    }


def verify_scale2000_prefix(combined_rows: Sequence[dict[str, Any]], inherited_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(combined_rows) < len(inherited_rows):
        raise ValueError("combined corpus is shorter than inherited scale-2000")
    prefix = list(combined_rows[: len(inherited_rows)])
    mismatches = []
    for index, (left, right) in enumerate(zip(prefix, inherited_rows)):
        if left != right:
            mismatches.append(index)
            if len(mismatches) >= 5:
                break
    if mismatches:
        raise ValueError(f"scale-2000 prefix mutated at positions {mismatches}")
    ids = [str(row["candidate_id"]) for row in combined_rows]
    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        raise ValueError("combined corpus contains duplicate semantic IDs")
    return {
        "status": "PASSED",
        "inherited_rows": len(inherited_rows),
        "combined_rows": len(combined_rows),
        "prefix_preserved": True,
        "duplicate_semantic_ids": 0,
    }


def build_scale8000_exposure_schedule(
    text_rows: Sequence[dict[str, Any]],
    augmentation_config: dict[str, Any],
    *,
    batch_size: int = 8,
    seed: int = 1234,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(text_rows) != 64000:
        raise ValueError(f"expected 64000 semantic rows, got {len(text_rows)}")
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
    summary = validate_scale8000_exposure_schedule(schedule, augmentation_config, batch_size=batch_size)
    return schedule, summary


def validate_scale8000_exposure_schedule(schedule: Sequence[dict[str, Any]], augmentation_config: dict[str, Any], *, batch_size: int = 8) -> dict[str, Any]:
    if len(schedule) != 1280000:
        raise ValueError("scale-8000 schedule must contain exactly 1280000 exposures")
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
        if len(semantic_by_round[round_index]) != 64000:
            issues.append(f"round_{round_index}_semantic_count")
    for voice in clean_voices:
        if voice_counts[voice] == 0:
            issues.append(f"voice_{voice}_missing")
    for profile_id in profiles:
        if profile_counts[profile_id] != 64000:
            issues.append(f"profile_{profile_id}_count")
    for held_out in ("supertonic-M5", "supertonic-F5", "M5", "F5"):
        if voice_counts.get(held_out, 0):
            issues.append(f"heldout_voice_{held_out}_leakage")
    if issues:
        raise ValueError(f"invalid scale-8000 exposure schedule: {issues}")
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


def build_dual_gpu_generation_plan(config: dict[str, Any]) -> dict[str, Any]:
    cells = sorted(str(cell["cell_id"]) for cell in prompt_cells(config))
    shards_per_cell = int(config["shards_per_cell"])
    requested = int(config["requested_rows_per_shard"])
    tasks = []
    for cell_id in cells:
        for shard_index in range(1, shards_per_cell + 1):
            tasks.append({"cell_id": cell_id, "shard_id": f"shard{shard_index:02d}", "requested_rows": requested})
    tasks = sorted(tasks, key=lambda task: (task["cell_id"], task["shard_id"]))
    workers = {
        "gpu0": {"physical_gpu": 0, "cuda_visible_devices": "0", "logical_device": "cuda:0", "tasks": []},
        "gpu1": {"physical_gpu": 1, "cuda_visible_devices": "1", "logical_device": "cuda:0", "tasks": []},
    }
    for index, task in enumerate(tasks):
        worker = "gpu0" if index % 2 == 0 else "gpu1"
        workers[worker]["tasks"].append(task)
    worker_summaries = {}
    for worker, payload in workers.items():
        worker_summaries[worker] = {
            "physical_gpu": payload["physical_gpu"],
            "cuda_visible_devices": payload["cuda_visible_devices"],
            "logical_device": payload["logical_device"],
            "task_count": len(payload["tasks"]),
            "requested_rows": sum(int(task["requested_rows"]) for task in payload["tasks"]),
        }
    return {
        "status": "PLANNED",
        "total_tasks": len(tasks),
        "total_requested_rows": sum(int(task["requested_rows"]) for task in tasks),
        "workers": worker_summaries,
    }


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    completed = subprocess.run(["du", "-sb", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return int(completed.stdout.split()[0])


def estimate_scale8000_storage(
    *,
    inherited_scale2000_bytes: int,
    available_bytes: int,
    safety_margin_fraction: float = 0.25,
) -> dict[str, Any]:
    if inherited_scale2000_bytes <= 0:
        raise ValueError("inherited_scale2000_bytes must be positive")
    new_bytes = inherited_scale2000_bytes * 3
    required_free = int(new_bytes * (1.0 + safety_margin_fraction))
    return {
        "inherited_scale2000_bytes": inherited_scale2000_bytes,
        "projected_new_bytes": new_bytes,
        "safety_margin_fraction": safety_margin_fraction,
        "required_free_bytes": required_free,
        "available_bytes": available_bytes,
        "sufficient": available_bytes >= required_free,
    }


def storage_preflight(config: dict[str, Any]) -> dict[str, Any]:
    inherited_dir = repo_path("runs/data-quality/sl-corpus-v4-gams-16000-training-v1")
    inherited_bytes = directory_size_bytes(inherited_dir)
    usage = shutil.disk_usage(REPO_ROOT)
    return estimate_scale8000_storage(
        inherited_scale2000_bytes=inherited_bytes,
        available_bytes=usage.free,
        safety_margin_fraction=0.25,
    )


def safe_public_status_report(config: dict[str, Any], *, canonical_results: dict[str, Any] | None = None, preflight: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = scale8000_plan().to_json()
    gpu_plan = build_dual_gpu_generation_plan(config)
    status = "ENVIRONMENT_BLOCKED" if preflight is not None and not preflight.get("sufficient", False) else "PLANNED"
    return {
        "schema_version": "1.0",
        "report_id": "0015-scale8000-dual-gpu-generation",
        "status": status,
        "corpus_id": config["corpus_id"],
        "new_addition_corpus_id": config["new_addition_corpus_id"],
        "accepted_parent": "none",
        "scale8000_plan": plan,
        "multiplier_table": scale8000_multiplier_table(),
        "inclusion_policy": config["inclusion_policy"],
        "parent_scale2000": {
            "corpus_id": config["inherited_corpus"]["corpus_id"],
            "sha256": config["inherited_corpus"]["sha256"],
            "rows": config["inherited_corpus"]["rows"],
            "audio_certificate": config["inherited_corpus"]["audio_certificate"],
            "experiment_report": config["inherited_corpus"]["experiment_report"],
        },
        "dual_gpu_plan": gpu_plan,
        "canonical_pass": canonical_results or {},
        "storage_preflight": preflight or {},
        "safety": {
            "raw_audio_committed": False,
            "raw_generated_text_committed": False,
            "model_or_checkpoint_committed": False,
            "training_eligible_issued": False,
            "model_publication": False,
        },
        "limitations": [
            "Scale-8000 is synthetic dataset construction evidence, not checkpoint acceptance.",
            "The 8000x figure is exposure count, not independent linguistic information.",
            "Generation must not start when storage preflight is insufficient.",
        ],
    }
