from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from slaif_asr.config import REPO_ROOT
from slaif_asr.otf_augmentation_dataset import sha256_file
from slaif_asr.parametric_voice_augmentation import (
    ALGORITHM_VERSION,
    AUGMENTATION_FAMILY,
    AUGMENTATION_KEY,
    DEPENDENCIES,
    MAX_ROUNDS,
    PHASES,
    load_policy,
    verify_dependencies,
)
from slaif_asr.scale8000_otf_surface08 import (
    ALGORITHM_VERSION as STANDARD_ALGORITHM_VERSION,
)
from slaif_asr.scale8000_otf_surface08 import (
    AUGMENTATION_KEY as STANDARD_AUGMENTATION_KEY,
)
from slaif_asr.scale8000_otf_surface08 import validate_config as validate_standard_config


EXPERIMENT_ID = "surface08-scale8000-otf-parametric-voiceaug-v1"
WORK_ORDER_ID = "0047"
POLICY_CONFIG_SHA256 = "2f914c5ca780bab73c14af82861c18186cfaee54d464c844bfccfdde40dacee7"
STANDARD_OTF_EXPERIMENT = "0031-surface08-scale8000-otf-augmented"
STRONGAUG_EXPERIMENT = "0032-surface08-scale8000-otf-strongaug-v1"


def validate_config(config: dict[str, Any]) -> None:
    normalized = copy.deepcopy(config)
    normalized["experiment_id"] = "surface08-scale8000-otf-augmented-v1"
    normalized["work_order_id"] = "0045"
    normalized_augmentation = normalized["otf_augmentation"]
    normalized_augmentation["augmentation_key"] = STANDARD_AUGMENTATION_KEY
    normalized_augmentation["algorithm_version"] = STANDARD_ALGORITHM_VERSION
    normalized_training = normalized["training"]
    normalized_training.update(
        {
            "primary_exposure_budget": 160_000,
            "max_exposures": 320_000,
            "max_rounds_primary": 10,
            "max_rounds": 20,
            "max_optimizer_steps": 40_000,
        }
    )
    validate_standard_config(normalized)

    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected ParametricVoiceAug experiment identity")
    if config.get("work_order_id") != WORK_ORDER_ID:
        raise ValueError(f"work_order_id must be {WORK_ORDER_ID}")
    augmentation = config["otf_augmentation"]
    expected_augmentation = {
        "augmentation_family": AUGMENTATION_FAMILY,
        "augmentation_key": AUGMENTATION_KEY,
        "algorithm_version": ALGORITHM_VERSION,
        "policy_config_sha256": POLICY_CONFIG_SHA256,
        "mixture_schedule": PHASES,
        "max_operations_per_sample": 3,
        "workers": 3,
        "prefetch_microbatches": 16,
        "disk_output": False,
        "transcript_preserved": True,
        "language_agnostic": True,
        "target_identity_models": False,
        "dependencies": list(DEPENDENCIES),
    }
    for key, expected in expected_augmentation.items():
        if augmentation.get(key) != expected:
            raise ValueError(f"otf_augmentation.{key} must be {expected!r}")
    if "strongaug" in json.dumps(augmentation, sort_keys=True).lower():
        raise ValueError("StrongAug cannot be the ParametricVoiceAug default policy")
    policy_path = REPO_ROOT / str(augmentation["policy_config"])
    if sha256_file(policy_path) != POLICY_CONFIG_SHA256:
        raise ValueError("ParametricVoiceAug policy config SHA256 mismatch")
    load_policy(policy_path)

    training = config["training"]
    expected_training = {
        "physical_microbatch": 2,
        "gradient_accumulation_steps": 4,
        "effective_batch_size": 8,
        "round_size_exposures": 16_000,
        "steps_per_round": 2_000,
        "primary_exposure_budget": 144_000,
        "max_exposures": 144_000,
        "max_rounds_primary": MAX_ROUNDS,
        "max_rounds": MAX_ROUNDS,
        "max_optimizer_steps": 18_000,
    }
    for key, expected in expected_training.items():
        if training.get(key) != expected:
            raise ValueError(f"training.{key} must be {expected!r}")
    if config["model"].get("initialization") != "untouched_base":
        raise ValueError("ParametricVoiceAug must start from the untouched base")
    if config["data"].get("training_source") != "scale8000_clean_otf_only":
        raise ValueError(
            "ParametricVoiceAug must use only scale-8000 clean OTF training"
        )
    if config.get("standard_otf_comparator", {}).get("experiment") != STANDARD_OTF_EXPERIMENT:
        raise ValueError("PR #51 standard OTF comparator identity is required")
    if config.get("strongaug_negative_comparator", {}).get("experiment") != STRONGAUG_EXPERIMENT:
        raise ValueError("PR #52 StrongAug negative comparator identity is required")


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ParametricVoiceAug experiment config must be a JSON object")
    validate_config(payload)
    return payload


def load_policy_for_config(config: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / str(config["otf_augmentation"]["policy_config"])
    if sha256_file(path) != POLICY_CONFIG_SHA256:
        raise RuntimeError("ParametricVoiceAug policy config SHA256 mismatch")
    policy = load_policy(path)
    verify_dependencies()
    return policy
