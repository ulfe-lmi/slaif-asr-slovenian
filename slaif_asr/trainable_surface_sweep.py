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
SURFACE05_ID = "SURFACE_05_DECODER_JOINT_PLUS_LAST_TWO_ENCODER_BLOCKS"
SURFACE05_ENCODER_BLOCKS = ("encoder.layers.22", "encoder.layers.23")
SURFACE05_ENCODER_BLOCK_PREFIXES = tuple(f"{name}." for name in SURFACE05_ENCODER_BLOCKS)
SURFACE05_ALLOWED_TRAINABLE_PREFIXES = ("decoder.", "joint.", *SURFACE05_ENCODER_BLOCK_PREFIXES)
SURFACE06_ID = "SURFACE_06_DECODER_JOINT_PLUS_LAST_FOUR_ENCODER_BLOCKS"
SURFACE06_ENCODER_BLOCKS = (
    "encoder.layers.20",
    "encoder.layers.21",
    "encoder.layers.22",
    "encoder.layers.23",
)
SURFACE06_ENCODER_BLOCK_PREFIXES = tuple(f"{name}." for name in SURFACE06_ENCODER_BLOCKS)
SURFACE06_ALLOWED_TRAINABLE_PREFIXES = ("decoder.", "joint.", *SURFACE06_ENCODER_BLOCK_PREFIXES)
SURFACE07_ID = "SURFACE_07_TOP_ENCODER_PLUS_PROMPT_ACOUSTIC_FUSION"
SURFACE07_ENCODER_BLOCKS = SURFACE06_ENCODER_BLOCKS
SURFACE07_ENCODER_BLOCK_PREFIXES = SURFACE06_ENCODER_BLOCK_PREFIXES
SURFACE07_FUSION_BRIDGE_MODULE = "prompt_kernel"
SURFACE07_FUSION_BRIDGE_PREFIX = f"{SURFACE07_FUSION_BRIDGE_MODULE}."
SURFACE07_ALLOWED_TRAINABLE_PREFIXES = (
    "decoder.",
    "joint.",
    *SURFACE07_ENCODER_BLOCK_PREFIXES,
    SURFACE07_FUSION_BRIDGE_PREFIX,
)
SURFACE08_ID = "SURFACE_08_FULL_ENCODER"
SURFACE08_ENCODER_LAYER_PREFIX = "encoder.layers."
SURFACE08_FUSION_BRIDGE_MODULE = SURFACE07_FUSION_BRIDGE_MODULE
SURFACE08_FUSION_BRIDGE_PREFIX = SURFACE07_FUSION_BRIDGE_PREFIX
SURFACE08_ALLOWED_TRAINABLE_PREFIXES = (
    "decoder.",
    "joint.",
    SURFACE08_ENCODER_LAYER_PREFIX,
    SURFACE08_FUSION_BRIDGE_PREFIX,
)
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

SURFACE04_METRICS = {
    "piper_synthetic_holdout": {"wer": 41.460, "cer": 14.522, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 16.071, "cer": 4.962, "empty": 0},
    "fleurs_v2": {"wer": 46.292, "cer": 14.792, "empty": 0},
    "artur_j": {"wer": 55.920, "cer": 18.535, "empty": 0},
}

SURFACE05_METRICS = {
    "piper_synthetic_holdout": {"wer": 39.130, "cer": 13.485, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 13.509, "cer": 4.177, "empty": 0},
    "fleurs_v2": {"wer": 46.564, "cer": 14.950, "empty": 0},
    "artur_j": {"wer": 53.473, "cer": 17.473, "empty": 0},
}

BEST_REAL_GATE_ENVELOPE = {
    "fleurs_v2": {
        "wer": {"value": 46.195, "source": "PR #36"},
        "cer": {"value": 14.792, "source": "Surface04"},
    },
    "artur_j": {
        "wer": {"value": 55.920, "source": "Surface04"},
        "cer": {"value": 18.535, "source": "Surface04"},
    },
}

SURFACE06_BEST_REAL_GATE_ENVELOPE = {
    "fleurs_v2": {
        "wer": {"value": 46.195, "source": "PR #36"},
        "cer": {"value": 14.792, "source": "Surface04"},
    },
    "artur_j": {
        "wer": {"value": 53.473, "source": "Surface05"},
        "cer": {"value": 17.473, "source": "Surface05"},
    },
}

SURFACE06_METRICS = {
    "piper_synthetic_holdout": {"wer": 34.161, "cer": 9.952, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 9.783, "cer": 2.761, "empty": 0},
    "fleurs_v2": {"wer": 44.506, "cer": 13.528, "empty": 0},
    "artur_j": {"wer": 50.590, "cer": 15.803, "empty": 0},
}

SURFACE07_BEST_REAL_GATE_ENVELOPE = {
    "fleurs_v2": {
        "wer": {"value": 44.506, "source": "Surface06"},
        "cer": {"value": 13.528, "source": "Surface06"},
    },
    "artur_j": {
        "wer": {"value": 50.590, "source": "Surface06"},
        "cer": {"value": 15.803, "source": "Surface06"},
    },
}

SURFACE08_BEST_REAL_GATE_ENVELOPE = {
    "fleurs_v2": {
        "wer": {"value": 42.084, "source": "Surface07"},
        "cer": {"value": 12.985, "source": "Surface07"},
    },
    "artur_j": {
        "wer": {"value": 47.357, "source": "Surface07"},
        "cer": {"value": 14.805, "source": "Surface07"},
    },
}

SURFACE07_METRICS = {
    "piper_synthetic_holdout": {"wer": 23.137, "cer": 7.429, "empty": 0},
    "supertonic_heldout_voice_holdout": {"wer": 7.842, "cer": 2.145, "empty": 0},
    "fleurs_v2": {"wer": 42.084, "cer": 12.985, "empty": 0},
    "artur_j": {"wer": 47.357, "cer": 14.805, "empty": 0},
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


@dataclass(frozen=True)
class Surface05Summary:
    surface_id: str
    final_encoder_blocks: tuple[str, str]
    trainable_parameter_count: int
    decoder_parameter_count: int
    joint_parameter_count: int
    final_two_encoder_blocks_parameter_count: int
    frozen_parameter_count: int
    trainable_prefixes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Surface06Summary:
    surface_id: str
    final_encoder_blocks: tuple[str, str, str, str]
    trainable_parameter_count: int
    decoder_parameter_count: int
    joint_parameter_count: int
    final_four_encoder_blocks_parameter_count: int
    frozen_parameter_count: int
    trainable_prefixes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Surface07Summary:
    surface_id: str
    final_encoder_blocks: tuple[str, str, str, str]
    fusion_bridge_module: str
    trainable_parameter_count: int
    decoder_parameter_count: int
    joint_parameter_count: int
    final_four_encoder_blocks_parameter_count: int
    fusion_bridge_parameter_count: int
    frozen_parameter_count: int
    trainable_prefixes: tuple[str, ...]
    fusion_discovery: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Surface08Summary:
    surface_id: str
    encoder_layers: tuple[str, ...]
    fusion_bridge_module: str
    trainable_parameter_count: int
    decoder_parameter_count: int
    joint_parameter_count: int
    encoder_all_layers_parameter_count: int
    fusion_bridge_parameter_count: int
    frozen_parameter_count: int
    trainable_prefixes: tuple[str, ...]
    fusion_discovery: dict[str, Any]

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


def load_surface05_config(
    path: str | Path = "configs/experiments/fixed-scale2000-surface05-last-two-encoder-blocks.json",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_surface05_config(config)
    return config


def validate_surface05_config(config: dict[str, Any], *, adr_text: str | None = None) -> None:
    if config.get("work_order_id") != "0038":
        raise ValueError("work_order_id must be 0038")
    if config.get("status") != "DIAGNOSTIC_ONLY" or config.get("accepted_parent") != "none":
        raise ValueError("surface sweep must remain DIAGNOSTIC_ONLY with accepted_parent none")
    if adr_text is None:
        adr_path = REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md"
        if not adr_path.exists():
            raise ValueError("ADR 0009 is required")
        adr_text = adr_path.read_text(encoding="utf-8")
    required_adr_markers = ("Phase 2 / Work Order 0038", SURFACE05_ID)
    if any(marker not in adr_text for marker in required_adr_markers):
        raise ValueError("ADR 0009 Phase 2 does not authorize SURFACE_05")

    model = config.get("model", {})
    if model.get("checkpoint_sha256") != "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74":
        raise ValueError("Surface05 must start from the untouched base checkpoint")
    if model.get("initialization") != "untouched_base":
        raise ValueError("Surface05 initialization must be untouched_base")

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
    forbidden_data_markers = ("s6tts", "scale8000", "scale-8000", "database-extension", "database_extension")
    if any(marker in serialized_data for marker in forbidden_data_markers):
        raise ValueError("fixed-data sweep forbids S6TTS, scale-8000, and database-extension data")

    surface = config.get("trainable_surface", {})
    if surface.get("surface_id") != SURFACE05_ID:
        raise ValueError("Work Order 0038 permits only SURFACE_05")
    if surface.get("encoder_layer_count") != 24:
        raise ValueError("SURFACE_05 requires the pinned 24-layer encoder")
    if surface.get("final_encoder_layer_indices") != [22, 23]:
        raise ValueError("SURFACE_05 must select final encoder layers 22 and 23")
    if tuple(surface.get("final_encoder_block_modules", ())) != SURFACE05_ENCODER_BLOCKS:
        raise ValueError("SURFACE_05 final encoder block identities drifted")
    if tuple(surface.get("allowed_prefixes", ())) != SURFACE05_ALLOWED_TRAINABLE_PREFIXES:
        raise ValueError("trainable prefixes must be decoder, joint, and final two encoder blocks only")
    if surface.get("full_encoder_allowed") is not False or surface.get("surface06_allowed") is not False:
        raise ValueError("full encoder and Surface06 training are prohibited")
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
    expected_lrs = {"decoder": 0.0005, "joint": 0.0005, "final_two_encoder_blocks": 0.00002}
    if lrs != expected_lrs:
        raise ValueError("learning-rate groups do not match Surface05")
    if not float(lrs["final_two_encoder_blocks"]) < float(lrs["decoder"]):
        raise ValueError("encoder learning rate must be lower than decoder/joint")

    control = config.get("controller_dev", {})
    if control.get("partition_id") != "artur-controller-dev-v1" or control.get("batch_size") != 1:
        raise ValueError("ARTUR controller-dev batch-1 policy is required")
    if control.get("duration_bucketing") is not False or control.get("immutable_gate_selection_allowed") is not False:
        raise ValueError("controller policy or immutable-gate isolation drifted")


def load_surface06_config(
    path: str | Path = "configs/experiments/fixed-scale2000-surface06-last-four-encoder-blocks.json",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_surface06_config(config)
    return config


def validate_surface06_config(config: dict[str, Any], *, adr_text: str | None = None) -> None:
    if config.get("work_order_id") != "0039":
        raise ValueError("work_order_id must be 0039")
    if config.get("status") != "DIAGNOSTIC_ONLY" or config.get("accepted_parent") != "none":
        raise ValueError("surface sweep must remain DIAGNOSTIC_ONLY with accepted_parent none")
    if adr_text is None:
        adr_path = REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md"
        if not adr_path.exists():
            raise ValueError("ADR 0009 is required")
        adr_text = adr_path.read_text(encoding="utf-8")
    required_adr_markers = ("Phase 3 / Work Order 0039", SURFACE06_ID)
    if any(marker not in adr_text for marker in required_adr_markers):
        raise ValueError("ADR 0009 Phase 3 does not authorize SURFACE_06")

    model = config.get("model", {})
    if model.get("checkpoint_sha256") != "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74":
        raise ValueError("Surface06 must start from the untouched base checkpoint")
    if model.get("initialization") != "untouched_base":
        raise ValueError("Surface06 initialization must be untouched_base")

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
    forbidden_data_markers = ("s6tts", "scale8000", "scale-8000", "database-extension", "database_extension")
    if any(marker in serialized_data for marker in forbidden_data_markers):
        raise ValueError("fixed-data sweep forbids S6TTS, scale-8000, and database-extension data")

    surface = config.get("trainable_surface", {})
    if surface.get("surface_id") != SURFACE06_ID:
        raise ValueError("Work Order 0039 permits only SURFACE_06")
    if surface.get("encoder_layer_count") != 24:
        raise ValueError("SURFACE_06 requires the pinned 24-layer encoder")
    if surface.get("final_encoder_layer_indices") != [20, 21, 22, 23]:
        raise ValueError("SURFACE_06 must select final encoder layers 20 through 23")
    if tuple(surface.get("final_encoder_block_modules", ())) != SURFACE06_ENCODER_BLOCKS:
        raise ValueError("SURFACE_06 final encoder block identities drifted")
    if tuple(surface.get("allowed_prefixes", ())) != SURFACE06_ALLOWED_TRAINABLE_PREFIXES:
        raise ValueError("trainable prefixes must be decoder, joint, and final four encoder blocks only")
    if surface.get("full_encoder_allowed") is not False or surface.get("surface07_allowed") is not False:
        raise ValueError("full encoder and Surface07 training are prohibited")
    if surface.get("prompt_acoustic_fusion_allowed") is not False:
        raise ValueError("prompt/acoustic fusion changes are prohibited")
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
    expected_lrs = {"decoder": 0.0005, "joint": 0.0005, "final_four_encoder_blocks": 0.00001}
    if lrs != expected_lrs:
        raise ValueError("learning-rate groups do not match Surface06")
    if not float(lrs["final_four_encoder_blocks"]) < float(lrs["decoder"]):
        raise ValueError("encoder learning rate must be lower than decoder/joint")

    control = config.get("controller_dev", {})
    if control.get("partition_id") != "artur-controller-dev-v1" or control.get("batch_size") != 1:
        raise ValueError("ARTUR controller-dev batch-1 policy is required")
    if control.get("duration_bucketing") is not False or control.get("immutable_gate_selection_allowed") is not False:
        raise ValueError("controller policy or immutable-gate isolation drifted")


def load_surface07_config(
    path: str | Path = "configs/experiments/fixed-scale2000-surface07-topencoder-fusion.json",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_surface07_config(config)
    return config


def validate_surface07_config(config: dict[str, Any], *, adr_text: str | None = None) -> None:
    if config.get("work_order_id") != "0040":
        raise ValueError("work_order_id must be 0040")
    if config.get("status") != "DIAGNOSTIC_ONLY" or config.get("accepted_parent") != "none":
        raise ValueError("surface sweep must remain DIAGNOSTIC_ONLY with accepted_parent none")
    if adr_text is None:
        adr_path = REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md"
        if not adr_path.exists():
            raise ValueError("ADR 0009 is required")
        adr_text = adr_path.read_text(encoding="utf-8")
    required_adr_markers = ("Phase 4 / Work Order 0040", SURFACE07_ID)
    if any(marker not in adr_text for marker in required_adr_markers):
        raise ValueError("ADR 0009 Phase 4 does not authorize SURFACE_07")

    model = config.get("model", {})
    if model.get("checkpoint_sha256") != "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74":
        raise ValueError("Surface07 must start from the untouched base checkpoint")
    if model.get("initialization") != "untouched_base":
        raise ValueError("Surface07 initialization must be untouched_base")

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
    forbidden_data_markers = ("s6tts", "scale8000", "scale-8000", "database-extension", "database_extension")
    if any(marker in serialized_data for marker in forbidden_data_markers):
        raise ValueError("fixed-data sweep forbids S6TTS, scale-8000, and database-extension data")

    surface = config.get("trainable_surface", {})
    if surface.get("surface_id") != SURFACE07_ID:
        raise ValueError("Work Order 0040 permits only SURFACE_07")
    if surface.get("encoder_layer_count") != 24:
        raise ValueError("SURFACE_07 requires the pinned 24-layer encoder")
    if surface.get("final_encoder_layer_indices") != [20, 21, 22, 23]:
        raise ValueError("SURFACE_07 must select final encoder layers 20 through 23")
    if tuple(surface.get("final_encoder_block_modules", ())) != SURFACE07_ENCODER_BLOCKS:
        raise ValueError("SURFACE_07 final encoder block identities drifted")
    if surface.get("fusion_bridge_module") != SURFACE07_FUSION_BRIDGE_MODULE:
        raise ValueError("SURFACE_07 requires the proven prompt_kernel fusion bridge")
    if surface.get("fusion_bridge_parameter_prefix") != SURFACE07_FUSION_BRIDGE_PREFIX:
        raise ValueError("SURFACE_07 fusion bridge prefix drifted")
    if tuple(surface.get("allowed_prefixes", ())) != SURFACE07_ALLOWED_TRAINABLE_PREFIXES:
        raise ValueError("trainable prefixes must be decoder, joint, final four encoder blocks, and prompt_kernel")
    required_false = (
        "full_encoder_allowed",
        "surface08_allowed",
        "prompt_labels_tables_embeddings_allowed",
        "language_id_mapping_changes_allowed",
        "target_lang_machinery_changes_allowed",
        "non_selected_prompt_fusion_changes_allowed",
        "text_only_objective_allowed",
        "temporary_lm_head_allowed",
    )
    if any(surface.get(key) is not False for key in required_false):
        raise ValueError("Surface07 protected-surface policy drifted")
    if surface.get("prompt_acoustic_fusion_allowed") is not True:
        raise ValueError("Surface07 must explicitly authorize one prompt/acoustic fusion bridge")

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
    expected_lrs = {
        "decoder": 0.0005,
        "joint": 0.0005,
        "final_four_encoder_blocks": 0.00001,
        "fusion_bridge": 0.00005,
    }
    if lrs != expected_lrs:
        raise ValueError("learning-rate groups do not match Surface07")
    if not float(lrs["final_four_encoder_blocks"]) < float(lrs["decoder"]):
        raise ValueError("encoder learning rate must be lower than decoder/joint")
    if not float(lrs["fusion_bridge"]) < float(lrs["decoder"]):
        raise ValueError("fusion bridge learning rate must be lower than decoder/joint")

    control = config.get("controller_dev", {})
    if control.get("partition_id") != "artur-controller-dev-v1" or control.get("batch_size") != 1:
        raise ValueError("ARTUR controller-dev batch-1 policy is required")
    if control.get("duration_bucketing") is not False or control.get("immutable_gate_selection_allowed") is not False:
        raise ValueError("controller policy or immutable-gate isolation drifted")


def load_surface08_config(
    path: str | Path = "configs/experiments/fixed-scale2000-surface08-full-encoder.json",
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_surface08_config(config)
    return config


def validate_surface08_config(config: dict[str, Any], *, adr_text: str | None = None) -> None:
    if config.get("work_order_id") != "0043":
        raise ValueError("work_order_id must be 0043")
    if config.get("status") != "DIAGNOSTIC_ONLY" or config.get("accepted_parent") != "none":
        raise ValueError("surface sweep must remain DIAGNOSTIC_ONLY with accepted_parent none")
    if adr_text is None:
        adr_path = REPO_ROOT / "docs/adr/0009-fixed-scale2000-surface-sweep.md"
        if not adr_path.exists():
            raise ValueError("ADR 0009 is required")
        adr_text = adr_path.read_text(encoding="utf-8")
    required_adr_markers = ("Phase 5 / Work Order 0043", SURFACE08_ID)
    if any(marker not in adr_text for marker in required_adr_markers):
        raise ValueError("ADR 0009 Phase 5 does not authorize SURFACE_08")

    model = config.get("model", {})
    if model.get("checkpoint_sha256") != "210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74":
        raise ValueError("Surface08 must start from the untouched base checkpoint")
    if model.get("initialization") != "untouched_base":
        raise ValueError("Surface08 initialization must be untouched_base")

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
    forbidden_data_markers = (
        "s6tts",
        "scale8000",
        "scale-8000",
        "database-extension",
        "database_extension",
        "real_speech",
    )
    if any(marker in serialized_data for marker in forbidden_data_markers):
        raise ValueError("Surface08 forbids S6TTS, scale-8000, database-extension, and real training data")

    surface = config.get("trainable_surface", {})
    if surface.get("surface_id") != SURFACE08_ID:
        raise ValueError("Work Order 0043 permits only SURFACE_08")
    if surface.get("encoder_layer_count") != 24:
        raise ValueError("SURFACE_08 requires the pinned 24-layer encoder")
    if surface.get("encoder_layer_indices") != list(range(24)):
        raise ValueError("SURFACE_08 must select encoder layers 0 through 23")
    expected_modules = [f"encoder.layers.{index}" for index in range(24)]
    if surface.get("encoder_layer_modules") != expected_modules:
        raise ValueError("SURFACE_08 encoder layer identities drifted")
    if surface.get("fusion_bridge_module") != SURFACE08_FUSION_BRIDGE_MODULE:
        raise ValueError("SURFACE_08 requires the proven prompt_kernel fusion bridge")
    if surface.get("fusion_bridge_parameter_prefix") != SURFACE08_FUSION_BRIDGE_PREFIX:
        raise ValueError("SURFACE_08 fusion bridge prefix drifted")
    if tuple(surface.get("allowed_prefixes", ())) != SURFACE08_ALLOWED_TRAINABLE_PREFIXES:
        raise ValueError("trainable prefixes must be decoder, joint, all encoder layers, and prompt_kernel")
    required_false = (
        "surface09_allowed",
        "full_model_allowed",
        "preprocessor_training_allowed",
        "frontend_subsampling_training_allowed",
        "prompt_labels_tables_embeddings_allowed",
        "prompt_identity_mapping_changes_allowed",
        "language_id_mapping_changes_allowed",
        "target_lang_machinery_changes_allowed",
        "non_selected_prompt_fusion_changes_allowed",
        "text_only_objective_allowed",
        "temporary_lm_head_allowed",
    )
    if any(surface.get(key) is not False for key in required_false):
        raise ValueError("Surface08 protected-surface policy drifted")
    if surface.get("full_encoder_allowed") is not True:
        raise ValueError("Surface08 must explicitly authorize all encoder layers")
    if surface.get("prompt_acoustic_fusion_allowed") is not True:
        raise ValueError("Surface08 must explicitly authorize the proven prompt_kernel bridge")

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
    if training.get("physical_microbatch_candidates") != [8, 4, 2, 1]:
        raise ValueError("physical microbatch candidates must be [8, 4, 2, 1]")
    lrs = training.get("learning_rates", {})
    expected_lrs = {
        "decoder": 0.0005,
        "joint": 0.0005,
        "encoder_all_layers": 0.000005,
        "fusion_bridge": 0.00005,
    }
    if lrs != expected_lrs:
        raise ValueError("learning-rate groups do not match Surface08")
    if not float(lrs["encoder_all_layers"]) < float(lrs["fusion_bridge"]) < float(lrs["decoder"]):
        raise ValueError("Surface08 encoder and fusion learning rates must remain below decoder/joint")

    control = config.get("controller_dev", {})
    if control.get("partition_id") != "artur-controller-dev-v1" or control.get("batch_size") != 1:
        raise ValueError("ARTUR controller-dev batch-1 policy is required")
    if control.get("duration_bucketing") is not False or control.get("immutable_gate_selection_allowed") is not False:
        raise ValueError("controller policy or immutable-gate isolation drifted")
    expected_guards = {
        "surface07_selected_wer": 43.443,
        "surface07_regression_delta": 5.0,
        "surface07_regression_consecutive_rounds": 2,
        "synthetic_real_divergence_delta": 8.0,
    }
    for key, expected in expected_guards.items():
        if control.get(key) != expected:
            raise ValueError(f"controller_dev.{key} must be {expected!r}")


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


def _encoder_layer_indices(model: Any) -> list[int]:
    indices: set[int] = set()
    prefix = "encoder.layers."
    for name, _parameter in model.named_parameters():
        if not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        token = remainder.split(".", 1)[0]
        if token.isdigit():
            indices.add(int(token))
    return sorted(indices)


def configure_surface05_trainable(model: Any) -> Surface05Summary:
    for name, _module in model.named_modules():
        lowered = name.lower()
        if "lm_head" in lowered or "lm_adapter" in lowered or "decoder_lm" in lowered:
            raise RuntimeError(f"forbidden text-only module present: {name}")

    layer_indices = _encoder_layer_indices(model)
    if layer_indices != list(range(24)):
        raise RuntimeError(
            "EXPERIMENT_INVALID_ENCODER_SURFACE_UNRESOLVED: "
            f"expected contiguous encoder layers 0..23, found {layer_indices}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    counts = {"decoder": 0, "joint": 0, "final_two_encoder_blocks": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            parameter.requires_grad_(True)
            counts["decoder"] += parameter.numel()
        elif name.startswith("joint."):
            parameter.requires_grad_(True)
            counts["joint"] += parameter.numel()
        elif name.startswith(SURFACE05_ENCODER_BLOCK_PREFIXES):
            parameter.requires_grad_(True)
            counts["final_two_encoder_blocks"] += parameter.numel()
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"required SURFACE_05 parameter group missing: {counts}")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(SURFACE05_ALLOWED_TRAINABLE_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"unauthorized trainable parameters: {unexpected[:10]}")
    frozen = sum(parameter.numel() for _name, parameter in model.named_parameters() if not parameter.requires_grad)
    return Surface05Summary(
        surface_id=SURFACE05_ID,
        final_encoder_blocks=SURFACE05_ENCODER_BLOCKS,
        trainable_parameter_count=sum(counts.values()),
        decoder_parameter_count=counts["decoder"],
        joint_parameter_count=counts["joint"],
        final_two_encoder_blocks_parameter_count=counts["final_two_encoder_blocks"],
        frozen_parameter_count=frozen,
        trainable_prefixes=SURFACE05_ALLOWED_TRAINABLE_PREFIXES,
    )


def configure_surface06_trainable(model: Any) -> Surface06Summary:
    for name, _module in model.named_modules():
        lowered = name.lower()
        if "lm_head" in lowered or "lm_adapter" in lowered or "decoder_lm" in lowered:
            raise RuntimeError(f"forbidden text-only module present: {name}")

    layer_indices = _encoder_layer_indices(model)
    if layer_indices != list(range(24)):
        raise RuntimeError(
            "EXPERIMENT_INVALID_ENCODER_SURFACE_UNRESOLVED: "
            f"expected contiguous encoder layers 0..23, found {layer_indices}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    counts = {"decoder": 0, "joint": 0, "final_four_encoder_blocks": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            parameter.requires_grad_(True)
            counts["decoder"] += parameter.numel()
        elif name.startswith("joint."):
            parameter.requires_grad_(True)
            counts["joint"] += parameter.numel()
        elif name.startswith(SURFACE06_ENCODER_BLOCK_PREFIXES):
            parameter.requires_grad_(True)
            counts["final_four_encoder_blocks"] += parameter.numel()
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"required SURFACE_06 parameter group missing: {counts}")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(SURFACE06_ALLOWED_TRAINABLE_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"unauthorized trainable parameters: {unexpected[:10]}")
    frozen = sum(parameter.numel() for _name, parameter in model.named_parameters() if not parameter.requires_grad)
    return Surface06Summary(
        surface_id=SURFACE06_ID,
        final_encoder_blocks=SURFACE06_ENCODER_BLOCKS,
        trainable_parameter_count=sum(counts.values()),
        decoder_parameter_count=counts["decoder"],
        joint_parameter_count=counts["joint"],
        final_four_encoder_blocks_parameter_count=counts["final_four_encoder_blocks"],
        frozen_parameter_count=frozen,
        trainable_prefixes=SURFACE06_ALLOWED_TRAINABLE_PREFIXES,
    )


def discover_surface07_fusion_bridge(model: Any) -> dict[str, Any]:
    keywords = (
        "prompt",
        "lang",
        "language",
        "fusion",
        "concat",
        "project",
        "projection",
        "adapter",
        "conditioning",
    )
    modules = list(model.named_modules())
    module_names = {name for name, _module in modules}
    parameters = list(model.named_parameters())
    parameter_names = [name for name, _parameter in parameters]
    candidates: list[dict[str, Any]] = []
    for name, module in modules:
        if not name or not any(keyword in name.lower() for keyword in keywords):
            continue
        prefix = f"{name}."
        recursive_names = [parameter_name for parameter_name in parameter_names if parameter_name.startswith(prefix)]
        direct_names = [
            parameter_name
            for parameter_name in recursive_names
            if "." not in parameter_name[len(prefix) :]
        ]
        by_name = dict(parameters)
        candidates.append(
            {
                "module": name,
                "module_type": f"{type(module).__module__}.{type(module).__name__}",
                "included": name == SURFACE07_FUSION_BRIDGE_MODULE,
                "direct_parameters": sum(by_name[item].numel() for item in direct_names),
                "recursive_parameters": sum(by_name[item].numel() for item in recursive_names),
                "reason": (
                    "post-concat acoustic/prompt projection selected as the sole fusion bridge"
                    if name == SURFACE07_FUSION_BRIDGE_MODULE
                    else "nested bridge component or non-selected candidate"
                ),
            }
        )

    if SURFACE07_FUSION_BRIDGE_MODULE not in module_names:
        return {
            "status": "BLOCKED_FUSION_BRIDGE_UNRESOLVED",
            "reason": "model does not expose prompt_kernel",
            "candidate_modules": candidates,
        }

    bridge_names = sorted(
        name for name in parameter_names if name.startswith(SURFACE07_FUSION_BRIDGE_PREFIX)
    )
    expected_bridge_names = [
        "prompt_kernel.0.bias",
        "prompt_kernel.0.weight",
        "prompt_kernel.2.bias",
        "prompt_kernel.2.weight",
    ]
    if bridge_names != expected_bridge_names:
        return {
            "status": "BLOCKED_FUSION_BRIDGE_UNRESOLVED",
            "reason": f"prompt_kernel parameter structure drifted: {bridge_names}",
            "candidate_modules": candidates,
        }

    protected_markers = ("prompt", "fusion", "conditioning", "language_id", "target_lang")
    protected_outside_bridge = sorted(
        name
        for name in parameter_names
        if any(marker in name.lower() for marker in protected_markers)
        and not name.startswith(SURFACE07_FUSION_BRIDGE_PREFIX)
    )
    nested_forbidden = sorted(
        name
        for name in module_names
        if name.startswith(SURFACE07_FUSION_BRIDGE_PREFIX)
        and any(marker in name.lower() for marker in ("embedding", "table", "adapter", "tokenizer"))
    )
    if protected_outside_bridge or nested_forbidden:
        return {
            "status": "BLOCKED_FUSION_BRIDGE_UNRESOLVED",
            "reason": "prompt identity parameters or forbidden nested modules overlap the candidate bridge",
            "protected_parameters_outside_bridge": protected_outside_bridge,
            "forbidden_nested_modules": nested_forbidden,
            "candidate_modules": candidates,
        }

    by_name = dict(parameters)
    return {
        "status": "PASSED",
        "module_name": SURFACE07_FUSION_BRIDGE_MODULE,
        "parameter_prefix": SURFACE07_FUSION_BRIDGE_PREFIX,
        "parameter_names": bridge_names,
        "parameter_count": sum(by_name[name].numel() for name in bridge_names),
        "prompt_identity_storage": "one_hot_config_not_parameter",
        "protected_parameters_outside_bridge": [],
        "forbidden_nested_modules": [],
        "candidate_modules": candidates,
    }


def configure_surface07_trainable(model: Any) -> Surface07Summary:
    for name, _module in model.named_modules():
        lowered = name.lower()
        if "lm_head" in lowered or "lm_adapter" in lowered or "decoder_lm" in lowered:
            raise RuntimeError(f"forbidden text-only module present: {name}")

    layer_indices = _encoder_layer_indices(model)
    if layer_indices != list(range(24)):
        raise RuntimeError(
            "EXPERIMENT_INVALID_ENCODER_SURFACE_UNRESOLVED: "
            f"expected contiguous encoder layers 0..23, found {layer_indices}"
        )

    discovery = discover_surface07_fusion_bridge(model)
    if discovery["status"] != "PASSED":
        raise RuntimeError(f"BLOCKED_FUSION_BRIDGE_UNRESOLVED: {discovery['reason']}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    counts = {
        "decoder": 0,
        "joint": 0,
        "final_four_encoder_blocks": 0,
        "fusion_bridge": 0,
    }
    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            parameter.requires_grad_(True)
            counts["decoder"] += parameter.numel()
        elif name.startswith("joint."):
            parameter.requires_grad_(True)
            counts["joint"] += parameter.numel()
        elif name.startswith(SURFACE07_ENCODER_BLOCK_PREFIXES):
            parameter.requires_grad_(True)
            counts["final_four_encoder_blocks"] += parameter.numel()
        elif name.startswith(SURFACE07_FUSION_BRIDGE_PREFIX):
            parameter.requires_grad_(True)
            counts["fusion_bridge"] += parameter.numel()
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"required SURFACE_07 parameter group missing: {counts}")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(SURFACE07_ALLOWED_TRAINABLE_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"unauthorized trainable parameters: {unexpected[:10]}")
    frozen = sum(parameter.numel() for _name, parameter in model.named_parameters() if not parameter.requires_grad)
    return Surface07Summary(
        surface_id=SURFACE07_ID,
        final_encoder_blocks=SURFACE07_ENCODER_BLOCKS,
        fusion_bridge_module=SURFACE07_FUSION_BRIDGE_MODULE,
        trainable_parameter_count=sum(counts.values()),
        decoder_parameter_count=counts["decoder"],
        joint_parameter_count=counts["joint"],
        final_four_encoder_blocks_parameter_count=counts["final_four_encoder_blocks"],
        fusion_bridge_parameter_count=counts["fusion_bridge"],
        frozen_parameter_count=frozen,
        trainable_prefixes=SURFACE07_ALLOWED_TRAINABLE_PREFIXES,
        fusion_discovery=discovery,
    )


def configure_surface08_trainable(model: Any) -> Surface08Summary:
    for name, _module in model.named_modules():
        lowered = name.lower()
        if "lm_head" in lowered or "lm_adapter" in lowered or "decoder_lm" in lowered:
            raise RuntimeError(f"forbidden text-only module present: {name}")

    layer_indices = _encoder_layer_indices(model)
    if layer_indices != list(range(24)):
        raise RuntimeError(
            "EXPERIMENT_INVALID_ENCODER_SURFACE_UNRESOLVED: "
            f"expected contiguous encoder layers 0..23, found {layer_indices}"
        )
    discovery = discover_surface07_fusion_bridge(model)
    if discovery["status"] != "PASSED":
        raise RuntimeError(f"BLOCKED_PROMPT_KERNEL_UNRESOLVED: {discovery['reason']}")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    counts = {
        "decoder": 0,
        "joint": 0,
        "encoder_all_layers": 0,
        "fusion_bridge": 0,
    }
    for name, parameter in model.named_parameters():
        if name.startswith("decoder."):
            parameter.requires_grad_(True)
            counts["decoder"] += parameter.numel()
        elif name.startswith("joint."):
            parameter.requires_grad_(True)
            counts["joint"] += parameter.numel()
        elif name.startswith(SURFACE08_ENCODER_LAYER_PREFIX):
            parameter.requires_grad_(True)
            counts["encoder_all_layers"] += parameter.numel()
        elif name.startswith(SURFACE08_FUSION_BRIDGE_PREFIX):
            parameter.requires_grad_(True)
            counts["fusion_bridge"] += parameter.numel()
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"required SURFACE_08 parameter group missing: {counts}")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(SURFACE08_ALLOWED_TRAINABLE_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"unauthorized trainable parameters: {unexpected[:10]}")
    frozen = sum(parameter.numel() for _name, parameter in model.named_parameters() if not parameter.requires_grad)
    return Surface08Summary(
        surface_id=SURFACE08_ID,
        encoder_layers=tuple(f"encoder.layers.{index}" for index in range(24)),
        fusion_bridge_module=SURFACE08_FUSION_BRIDGE_MODULE,
        trainable_parameter_count=sum(counts.values()),
        decoder_parameter_count=counts["decoder"],
        joint_parameter_count=counts["joint"],
        encoder_all_layers_parameter_count=counts["encoder_all_layers"],
        fusion_bridge_parameter_count=counts["fusion_bridge"],
        frozen_parameter_count=frozen,
        trainable_prefixes=SURFACE08_ALLOWED_TRAINABLE_PREFIXES,
        fusion_discovery=discovery,
    )


def set_surface04_training_mode(model: Any) -> None:
    # Match PR #36 training-mode semantics; requires_grad is the only changed
    # surface control. Evaluation probes switch the entire model to eval mode.
    model.train()


def set_surface05_training_mode(model: Any) -> None:
    # Keep the Surface04/PR #36 model-mode semantics; only requires_grad scope changes.
    model.train()


def set_surface06_training_mode(model: Any) -> None:
    # Keep prior surface-sweep model-mode semantics; only requires_grad scope changes.
    model.train()


def set_surface07_training_mode(model: Any) -> None:
    # Preserve Surface06 model-mode semantics; only the fusion bridge scope changes.
    model.train()


def set_surface08_training_mode(model: Any) -> None:
    # Preserve prior sweep model-mode semantics; only requires_grad scope changes.
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


def surface05_optimizer_parameter_groups(model: Any, learning_rates: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = {"decoder": [], "joint": [], "final_two_encoder_blocks": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder."):
            groups["decoder"].append(parameter)
        elif name.startswith("joint."):
            groups["joint"].append(parameter)
        elif name.startswith(SURFACE05_ENCODER_BLOCK_PREFIXES):
            groups["final_two_encoder_blocks"].append(parameter)
        else:
            raise RuntimeError(f"unauthorized optimizer parameter: {name}")
    if any(not values for values in groups.values()):
        raise RuntimeError("optimizer is missing a SURFACE_05 parameter group")
    return [
        {"name": name, "params": groups[name], "lr": float(learning_rates[name])}
        for name in ("decoder", "joint", "final_two_encoder_blocks")
    ]


def surface06_optimizer_parameter_groups(model: Any, learning_rates: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = {"decoder": [], "joint": [], "final_four_encoder_blocks": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder."):
            groups["decoder"].append(parameter)
        elif name.startswith("joint."):
            groups["joint"].append(parameter)
        elif name.startswith(SURFACE06_ENCODER_BLOCK_PREFIXES):
            groups["final_four_encoder_blocks"].append(parameter)
        else:
            raise RuntimeError(f"unauthorized optimizer parameter: {name}")
    if any(not values for values in groups.values()):
        raise RuntimeError("optimizer is missing a SURFACE_06 parameter group")
    return [
        {"name": name, "params": groups[name], "lr": float(learning_rates[name])}
        for name in ("decoder", "joint", "final_four_encoder_blocks")
    ]


def surface07_optimizer_parameter_groups(model: Any, learning_rates: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = {
        "decoder": [],
        "joint": [],
        "final_four_encoder_blocks": [],
        "fusion_bridge": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder."):
            groups["decoder"].append(parameter)
        elif name.startswith("joint."):
            groups["joint"].append(parameter)
        elif name.startswith(SURFACE07_ENCODER_BLOCK_PREFIXES):
            groups["final_four_encoder_blocks"].append(parameter)
        elif name.startswith(SURFACE07_FUSION_BRIDGE_PREFIX):
            groups["fusion_bridge"].append(parameter)
        else:
            raise RuntimeError(f"unauthorized optimizer parameter: {name}")
    if any(not values for values in groups.values()):
        raise RuntimeError("optimizer is missing a SURFACE_07 parameter group")
    return [
        {"name": name, "params": groups[name], "lr": float(learning_rates[name])}
        for name in ("decoder", "joint", "final_four_encoder_blocks", "fusion_bridge")
    ]


def surface08_optimizer_parameter_groups(model: Any, learning_rates: dict[str, float]) -> list[dict[str, Any]]:
    groups: dict[str, list[Any]] = {
        "decoder": [],
        "joint": [],
        "encoder_all_layers": [],
        "fusion_bridge": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("decoder."):
            groups["decoder"].append(parameter)
        elif name.startswith("joint."):
            groups["joint"].append(parameter)
        elif name.startswith(SURFACE08_ENCODER_LAYER_PREFIX):
            groups["encoder_all_layers"].append(parameter)
        elif name.startswith(SURFACE08_FUSION_BRIDGE_PREFIX):
            groups["fusion_bridge"].append(parameter)
        else:
            raise RuntimeError(f"unauthorized optimizer parameter: {name}")
    if any(not values for values in groups.values()):
        raise RuntimeError("optimizer is missing a SURFACE_08 parameter group")
    return [
        {"name": name, "params": groups[name], "lr": float(learning_rates[name])}
        for name in ("decoder", "joint", "encoder_all_layers", "fusion_bridge")
    ]


def verify_optimizer_scope(optimizer: Any, model: Any, learning_rates: dict[str, float]) -> None:
    expected = {id(parameter) for _name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer parameters do not exactly match SURFACE_04")
    by_name = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
    if by_name != {name: float(value) for name, value in learning_rates.items()}:
        raise RuntimeError("optimizer learning-rate groups drifted")


def verify_surface05_optimizer_scope(optimizer: Any, model: Any, learning_rates: dict[str, float]) -> None:
    expected = {id(parameter) for _name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer parameters do not exactly match SURFACE_05")
    by_name = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
    if by_name != {name: float(value) for name, value in learning_rates.items()}:
        raise RuntimeError("optimizer learning-rate groups drifted")


def verify_surface06_optimizer_scope(optimizer: Any, model: Any, learning_rates: dict[str, float]) -> None:
    expected = {id(parameter) for _name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer parameters do not exactly match SURFACE_06")
    by_name = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
    if by_name != {name: float(value) for name, value in learning_rates.items()}:
        raise RuntimeError("optimizer learning-rate groups drifted")


def verify_surface07_optimizer_scope(optimizer: Any, model: Any, learning_rates: dict[str, float]) -> None:
    expected = {id(parameter) for _name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer parameters do not exactly match SURFACE_07")
    by_name = {str(group.get("name")): float(group["lr"]) for group in optimizer.param_groups}
    if by_name != {name: float(value) for name, value in learning_rates.items()}:
        raise RuntimeError("optimizer learning-rate groups drifted")


def verify_surface08_optimizer_scope(optimizer: Any, model: Any, learning_rates: dict[str, float]) -> None:
    expected = {id(parameter) for _name, parameter in model.named_parameters() if parameter.requires_grad}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected:
        raise RuntimeError("optimizer parameters do not exactly match SURFACE_08")
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


def surface05_changed_tensor_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed: list[str] = []
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        if left.shape != right.shape or not bool((left == right).all()):
            changed.append(name)
    unauthorized = [name for name in changed if not name.startswith(SURFACE05_ALLOWED_TRAINABLE_PREFIXES)]
    return {
        "changed_tensor_count": len(changed),
        "authorized_changed_tensor_count": len(changed) - len(unauthorized),
        "unauthorized_changed_tensors": unauthorized,
        "missing_tensors": missing,
        "unexpected_tensors": extra,
        "lower_encoder_unchanged": not any(
            name.startswith("encoder.") and not name.startswith(SURFACE05_ENCODER_BLOCK_PREFIXES)
            for name in changed
        ),
        "preprocessor_unchanged": not any(name.startswith("preprocessor.") for name in changed),
        "prompt_path_unchanged": not any("prompt" in name.lower() for name in changed),
        "only_surface05_changed": not missing and not extra and not unauthorized,
    }


def surface06_changed_tensor_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed: list[str] = []
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        if left.shape != right.shape or not bool((left == right).all()):
            changed.append(name)
    unauthorized = [name for name in changed if not name.startswith(SURFACE06_ALLOWED_TRAINABLE_PREFIXES)]
    return {
        "changed_tensor_count": len(changed),
        "authorized_changed_tensor_count": len(changed) - len(unauthorized),
        "unauthorized_changed_tensors": unauthorized,
        "missing_tensors": missing,
        "unexpected_tensors": extra,
        "lower_encoder_unchanged": not any(
            name.startswith("encoder.") and not name.startswith(SURFACE06_ENCODER_BLOCK_PREFIXES)
            for name in changed
        ),
        "preprocessor_unchanged": not any(name.startswith("preprocessor.") for name in changed),
        "prompt_path_unchanged": not any("prompt" in name.lower() for name in changed),
        "only_surface06_changed": not missing and not extra and not unauthorized,
    }


def surface07_changed_tensor_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed: list[str] = []
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        if left.shape != right.shape or not bool((left == right).all()):
            changed.append(name)
    unauthorized = [name for name in changed if not name.startswith(SURFACE07_ALLOWED_TRAINABLE_PREFIXES)]
    non_selected_prompt_or_fusion = [
        name
        for name in changed
        if any(marker in name.lower() for marker in ("prompt", "fusion", "conditioning", "language_id", "target_lang"))
        and not name.startswith(SURFACE07_FUSION_BRIDGE_PREFIX)
    ]
    return {
        "changed_tensor_count": len(changed),
        "authorized_changed_tensor_count": len(changed) - len(unauthorized),
        "unauthorized_changed_tensors": unauthorized,
        "missing_tensors": missing,
        "unexpected_tensors": extra,
        "lower_encoder_unchanged": not any(
            name.startswith("encoder.") and not name.startswith(SURFACE07_ENCODER_BLOCK_PREFIXES)
            for name in changed
        ),
        "preprocessor_unchanged": not any(name.startswith("preprocessor.") for name in changed),
        "fusion_bridge_changed": any(
            name.startswith(SURFACE07_FUSION_BRIDGE_PREFIX) for name in changed
        ),
        "non_selected_prompt_or_fusion_unchanged": not non_selected_prompt_or_fusion,
        "prompt_labels_tables_embeddings_unchanged": not non_selected_prompt_or_fusion,
        "only_surface07_changed": not missing and not extra and not unauthorized,
    }


def surface08_changed_tensor_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    changed: list[str] = []
    for name in sorted(set(before) & set(after)):
        left, right = before[name], after[name]
        if left.shape != right.shape or not bool((left == right).all()):
            changed.append(name)
    unauthorized = [name for name in changed if not name.startswith(SURFACE08_ALLOWED_TRAINABLE_PREFIXES)]
    non_selected_prompt_or_fusion = [
        name
        for name in changed
        if any(marker in name.lower() for marker in ("prompt", "fusion", "conditioning", "language_id", "target_lang"))
        and not name.startswith(SURFACE08_FUSION_BRIDGE_PREFIX)
    ]
    frontend_changed = [
        name
        for name in changed
        if name.startswith("preprocessor.")
        or name.startswith("encoder.pre_encode.")
        or (name.startswith("encoder.") and not name.startswith(SURFACE08_ENCODER_LAYER_PREFIX))
    ]
    return {
        "changed_tensor_count": len(changed),
        "authorized_changed_tensor_count": len(changed) - len(unauthorized),
        "unauthorized_changed_tensors": unauthorized,
        "missing_tensors": missing,
        "unexpected_tensors": extra,
        "encoder_all_layers_changed": any(
            name.startswith(SURFACE08_ENCODER_LAYER_PREFIX) for name in changed
        ),
        "preprocessor_unchanged": not any(name.startswith("preprocessor.") for name in changed),
        "subsampling_frontend_unchanged": not frontend_changed,
        "fusion_bridge_changed": any(
            name.startswith(SURFACE08_FUSION_BRIDGE_PREFIX) for name in changed
        ),
        "non_selected_prompt_or_fusion_unchanged": not non_selected_prompt_or_fusion,
        "prompt_identity_unchanged": not non_selected_prompt_or_fusion,
        "only_surface08_changed": not missing and not extra and not unauthorized,
    }


def microbatch_plan(physical_microbatch: int) -> dict[str, int]:
    if physical_microbatch not in (8, 4, 2, 1):
        raise ValueError("physical microbatch must be one of 8, 4, 2, 1")
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


def select_surface05_microbatch(outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for candidate in (4, 2, 1):
        if outcomes.get(candidate, {}).get("status") == "PASSED":
            return {"status": "PASSED", **microbatch_plan(candidate)}
    return {
        "status": "BLOCKED_SURFACE05_OOM",
        "physical_microbatch": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": 8,
    }


def select_surface06_microbatch(outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for candidate in (4, 2, 1):
        if outcomes.get(candidate, {}).get("status") == "PASSED":
            return {"status": "PASSED", **microbatch_plan(candidate)}
    return {
        "status": "BLOCKED_SURFACE06_OOM",
        "physical_microbatch": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": 8,
    }


def select_surface07_microbatch(outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for candidate in (4, 2, 1):
        if outcomes.get(candidate, {}).get("status") == "PASSED":
            return {"status": "PASSED", **microbatch_plan(candidate)}
    return {
        "status": "BLOCKED_SURFACE07_OOM",
        "physical_microbatch": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": 8,
    }


def select_surface08_microbatch(outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    for candidate in (8, 4, 2, 1):
        if outcomes.get(candidate, {}).get("status") == "PASSED":
            return {"status": "PASSED", **microbatch_plan(candidate)}
    return {
        "status": "BLOCKED_SURFACE08_OOM",
        "physical_microbatch": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": 8,
    }


def apply_observed_training_ooms(
    outcomes: dict[int, dict[str, Any]],
    failures: Sequence[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    adjusted = {int(candidate): dict(outcome) for candidate, outcome in outcomes.items()}
    for failure in failures:
        if failure.get("status") != "FAILED_TRAINING_OOM":
            continue
        physical = int(failure["physical_microbatch"])
        prior = adjusted.get(physical, {})
        adjusted[physical] = {
            "status": "FAILED",
            "error_type": "ObservedTrainingOOM",
            "error": "full-schedule training OOM overrides the bounded memory probe",
            "probe_status_before_override": prior.get("status", "NOT_RUN"),
            "observed_optimizer_step": int(failure["optimizer_step"]),
        }
    return adjusted


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


def should_stop_surface08_controller_curve(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    post = [row for row in rows if int(row["round"]) > 0]
    if not post:
        return {"stop": False, "reason": "minimum_rounds_not_reached"}
    latest = post[-1]
    if int(latest["round"]) > 1 and int(latest.get("empty", 0)) > 0:
        return {"stop": True, "reason": "surface08_empty_hypotheses_reappeared"}

    if len(post) >= 2 and all(float(row["wer"]) > 43.443 + 5.0 for row in post[-2:]):
        return {
            "stop": True,
            "reason": "surface08_two_consecutive_rounds_worse_than_surface07_by_5",
        }

    base = next((row for row in rows if int(row["round"]) == 0), None)
    best_wer = min(float(row["wer"]) for row in rows if row.get("wer") is not None)
    synthetic_improved = bool(
        base
        and latest.get("synthetic_anchor_probe_loss") is not None
        and latest.get("synthetic_scale_probe_loss") is not None
        and base.get("synthetic_anchor_probe_loss") is not None
        and base.get("synthetic_scale_probe_loss") is not None
        and float(latest["synthetic_anchor_probe_loss"]) < float(base["synthetic_anchor_probe_loss"])
        and float(latest["synthetic_scale_probe_loss"]) < float(base["synthetic_scale_probe_loss"])
    )
    if synthetic_improved and float(latest["wer"]) > best_wer + 8.0:
        return {
            "stop": True,
            "reason": "surface08_synthetic_real_divergence_over_8",
        }
    return should_stop_controller_curve(rows)


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
    within_non_regression_tolerance = all(
        float(metrics[split][metric])
        <= float(PR36_METRICS[split][metric]) + (0.50 if metric == "wer" else 0.25)
        for split in real_splits
        for metric in ("wer", "cer")
    )
    improves_one = any(
        float(metrics[split][metric]) < float(PR36_METRICS[split][metric])
        for split in real_splits
        for metric in ("wer", "cer")
    )
    if within_non_regression_tolerance and improves_one and synthetic_safe:
        return "SURFACE04_MATCHES_PR36_WITH_ACCEPTABLE_TRADEOFF"
    if base_better:
        return "SURFACE04_BEATS_BASE_BUT_NOT_PR36"
    if selected_round is not None and selected_round > 0:
        return "SURFACE04_ARTUR_DEV_GOOD_BUT_GATE_DIRECTIONAL_REGRESSES"
    return "SURFACE04_SYNTHETIC_OR_REAL_REGRESSION"


def surface05_envelope_comparison(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("fleurs_v2", "artur_j"):
        for metric in ("wer", "cer"):
            prior = BEST_REAL_GATE_ENVELOPE[split][metric]
            value = float(metrics[split][metric])
            tolerance = 0.50 if metric == "wer" else 0.25
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "best_prior_value": float(prior["value"]),
                    "prior_source": str(prior["source"]),
                    "surface05_value": value,
                    "within_tolerance": value <= float(prior["value"]) + tolerance,
                    "improved": value < float(prior["value"]),
                }
            )
    return rows


def classify_surface05(
    metrics: dict[str, dict[str, Any]],
    *,
    parameter_integrity: bool = True,
    selected_round: int | None = None,
) -> str:
    if not parameter_integrity:
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    if any(int(metrics[split].get("empty", 0)) > 0 for split in real_splits):
        return "SURFACE05_SYNTHETIC_OR_REAL_REGRESSION"

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
    envelope = surface05_envelope_comparison(metrics)
    within_envelope = all(row["within_tolerance"] for row in envelope)
    improved_envelope_count = sum(bool(row["improved"]) for row in envelope)

    if synthetic_safe and within_envelope and improved_envelope_count >= 3:
        return "SURFACE05_NEW_BEST_DIRECTIONAL_CANDIDATE"

    improves_prior_challenger = any(
        float(metrics[split][metric]) < float(comparator[split][metric])
        for comparator in (PR36_METRICS, SURFACE04_METRICS)
        for split in real_splits
        for metric in ("wer", "cer")
    )
    if synthetic_safe and within_envelope and improves_prior_challenger:
        return "SURFACE05_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF"
    if synthetic_safe and base_better:
        return "SURFACE05_BEATS_BASE_BUT_NOT_PRIOR_CHALLENGERS"
    if selected_round is not None and selected_round > 0:
        return "SURFACE05_ARTUR_DEV_GOOD_BUT_GATE_DIRECTIONAL_REGRESSES"
    return "SURFACE05_SYNTHETIC_OR_REAL_REGRESSION"


def surface06_envelope_comparison(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("fleurs_v2", "artur_j"):
        for metric in ("wer", "cer"):
            prior = SURFACE06_BEST_REAL_GATE_ENVELOPE[split][metric]
            value = float(metrics[split][metric])
            tolerance = 0.50 if metric == "wer" else 0.25
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "best_prior_value": float(prior["value"]),
                    "prior_source": str(prior["source"]),
                    "surface06_value": value,
                    "within_tolerance": value <= float(prior["value"]) + tolerance,
                    "improved": value < float(prior["value"]),
                }
            )
    return rows


def classify_surface06(
    metrics: dict[str, dict[str, Any]],
    *,
    parameter_integrity: bool = True,
    selected_round: int | None = None,
) -> str:
    if not parameter_integrity:
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    if any(int(metrics[split].get("empty", 0)) > 0 for split in real_splits):
        return "SURFACE06_SYNTHETIC_OR_REAL_REGRESSION"

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
    envelope = surface06_envelope_comparison(metrics)
    within_envelope = all(row["within_tolerance"] for row in envelope)
    improved_envelope_count = sum(bool(row["improved"]) for row in envelope)

    if synthetic_safe and within_envelope and improved_envelope_count >= 3:
        return "SURFACE06_NEW_BEST_DIRECTIONAL_CANDIDATE"

    improves_prior_challenger = any(
        float(metrics[split][metric]) < float(comparator[split][metric])
        for comparator in (PR36_METRICS, SURFACE04_METRICS, SURFACE05_METRICS)
        for split in real_splits
        for metric in ("wer", "cer")
    )
    if synthetic_safe and within_envelope and improves_prior_challenger:
        return "SURFACE06_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF"

    fleurs_within = all(
        row["within_tolerance"] for row in envelope if row["split"] == "fleurs_v2"
    )
    artur_improves = any(
        row["improved"] for row in envelope if row["split"] == "artur_j"
    )
    if selected_round is not None and selected_round > 0 and artur_improves and not fleurs_within:
        return "SURFACE06_ARTUR_DEV_GOOD_BUT_FLEURS_REGRESSES"
    if synthetic_safe and base_better:
        return "SURFACE06_BEATS_BASE_BUT_NOT_PRIOR_CHALLENGERS"
    return "SURFACE06_SYNTHETIC_OR_REAL_REGRESSION"


def surface07_envelope_comparison(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("fleurs_v2", "artur_j"):
        for metric in ("wer", "cer"):
            prior = SURFACE07_BEST_REAL_GATE_ENVELOPE[split][metric]
            value = float(metrics[split][metric])
            tolerance = 0.50 if metric == "wer" else 0.25
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "best_prior_value": float(prior["value"]),
                    "prior_source": str(prior["source"]),
                    "surface07_value": value,
                    "within_tolerance": value <= float(prior["value"]) + tolerance,
                    "improved": value < float(prior["value"]),
                }
            )
    return rows


def classify_surface07(
    metrics: dict[str, dict[str, Any]],
    *,
    parameter_integrity: bool = True,
    fusion_bridge_proven: bool = True,
    selected_round: int | None = None,
) -> str:
    if not fusion_bridge_proven:
        return "BLOCKED_FUSION_BRIDGE_UNRESOLVED"
    if not parameter_integrity:
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    if any(int(metrics[split].get("empty", 0)) > 0 for split in real_splits):
        return "SURFACE07_SYNTHETIC_OR_REAL_REGRESSION"

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
    envelope = surface07_envelope_comparison(metrics)
    within_envelope = all(row["within_tolerance"] for row in envelope)
    improved_envelope_count = sum(bool(row["improved"]) for row in envelope)

    if synthetic_safe and within_envelope and improved_envelope_count >= 3:
        return "SURFACE07_NEW_BEST_DIRECTIONAL_CANDIDATE"
    if synthetic_safe and within_envelope and improved_envelope_count >= 1:
        return "SURFACE07_MATCHES_BEST_WITH_ACCEPTABLE_TRADEOFF"

    fleurs_within = all(
        row["within_tolerance"] for row in envelope if row["split"] == "fleurs_v2"
    )
    artur_improves = any(
        row["improved"] for row in envelope if row["split"] == "artur_j"
    )
    if selected_round is not None and selected_round > 0 and artur_improves and not fleurs_within:
        return "SURFACE07_FUSION_GOOD_BUT_FLEURS_REGRESSES"
    if synthetic_safe and base_better:
        return "SURFACE07_BEATS_BASE_BUT_NOT_PRIOR_CHALLENGERS"
    return "SURFACE07_SYNTHETIC_OR_REAL_REGRESSION"


def surface08_envelope_comparison(metrics: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("fleurs_v2", "artur_j"):
        for metric in ("wer", "cer"):
            prior = SURFACE08_BEST_REAL_GATE_ENVELOPE[split][metric]
            value = float(metrics[split][metric])
            tolerance = 0.50 if metric == "wer" else 0.25
            rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "best_prior_value": float(prior["value"]),
                    "prior_source": str(prior["source"]),
                    "surface08_value": value,
                    "within_tolerance": value <= float(prior["value"]) + tolerance,
                    "improved": value < float(prior["value"]),
                }
            )
    return rows


def classify_surface08(
    metrics: dict[str, dict[str, Any]],
    *,
    parameter_integrity: bool = True,
    selected_round: int | None = None,
) -> str:
    if not parameter_integrity:
        return "EXPERIMENT_INVALID"
    real_splits = ("fleurs_v2", "artur_j")
    if any(int(metrics[split].get("empty", 0)) > 0 for split in real_splits):
        return "SURFACE08_SYNTHETIC_OVERFIT_OR_REAL_REGRESSION"

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
    envelope = surface08_envelope_comparison(metrics)
    within_envelope = all(row["within_tolerance"] for row in envelope)
    improved_envelope_count = sum(bool(row["improved"]) for row in envelope)

    if synthetic_safe and within_envelope and improved_envelope_count >= 3:
        return "SURFACE08_NEW_BEST_DIRECTIONAL_CANDIDATE"
    if synthetic_safe and within_envelope and improved_envelope_count >= 1:
        return "SURFACE08_MATCHES_SURFACE07_WITH_ACCEPTABLE_TRADEOFF"

    fleurs_within = all(
        row["within_tolerance"] for row in envelope if row["split"] == "fleurs_v2"
    )
    if selected_round is not None and selected_round > 0 and not fleurs_within:
        return "SURFACE08_ARTUR_DEV_GOOD_BUT_FLEURS_REGRESSES"
    if synthetic_safe and base_better:
        return "SURFACE08_BEATS_BASE_BUT_NOT_SURFACE07"
    return "SURFACE08_SYNTHETIC_OVERFIT_OR_REAL_REGRESSION"


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
