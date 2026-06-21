from __future__ import annotations

import importlib.metadata
import json
import platform
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slaif_asr.config import load_runtime_config


PROMPT_PATTERN = re.compile(r"(prompt|lang|language)", re.IGNORECASE)


@dataclass(frozen=True)
class RuntimeContract:
    loaded_class: str
    total_parameters: int | None
    encoder_layer_count: int | None
    encoder_width: int | None
    tokenizer_vocabulary_size: int | None
    sample_rate: int | None
    prompt_indices: dict[str, int | None]
    prompt_kernel_structure: list[dict[str, Any]]
    checkpoint_detected_streaming_contexts: list[list[int]]
    configured_supported_streaming_contexts: list[list[int]]
    default_streaming_context: list[int] | None
    checkpoint: dict[str, Any]
    environment: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_contract(contract: RuntimeContract, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(contract.to_json(), encoding="utf-8")


def build_runtime_contract(model: Any, checkpoint_path: str | None = None) -> RuntimeContract:
    cfg = load_runtime_config()
    prompt_indices = {
        "sl-SI": find_prompt_index(model, "sl-SI"),
        "sl": find_prompt_index(model, "sl"),
    }
    detected_contexts = extract_streaming_contexts(model)
    configured_contexts = configured_streaming_contexts()
    return RuntimeContract(
        loaded_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
        total_parameters=count_parameters(model),
        encoder_layer_count=encoder_layer_count(model),
        encoder_width=encoder_width(model),
        tokenizer_vocabulary_size=tokenizer_vocabulary_size(model),
        sample_rate=sample_rate(model),
        prompt_indices=prompt_indices,
        prompt_kernel_structure=prompt_kernel_structure(model),
        checkpoint_detected_streaming_contexts=detected_contexts,
        configured_supported_streaming_contexts=configured_contexts,
        default_streaming_context=detected_contexts[0] if detected_contexts else [56, 3],
        checkpoint={
            "repository": cfg["base_model"]["repository"],
            "revision": cfg["base_model"]["revision"],
            "filename": cfg["base_model"]["filename"],
            "sha256": cfg["base_model"]["sha256"],
            "hf_lfs_etag": cfg["base_model"].get("hf_lfs_etag"),
            "byte_size": cfg["base_model"].get("byte_size"),
            "local_path": checkpoint_path,
        },
        environment=environment_details(),
        notes=contract_notes(model, prompt_indices),
    )


def count_parameters(model: Any) -> int | None:
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return None
    return sum(int(param.numel()) for param in parameters())


def encoder_layer_count(model: Any) -> int | None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return None
    for name in ("layers", "encoder", "conformer_layers", "transformer_layers"):
        value = getattr(encoder, name, None)
        if value is not None and hasattr(value, "__len__"):
            return len(value)
    cfg_value = nested_get(config_container(model), ("encoder", "num_layers"))
    return int(cfg_value) if cfg_value is not None else None


def encoder_width(model: Any) -> int | None:
    cfg = config_container(model)
    for path in (
        ("encoder", "d_model"),
        ("encoder", "feat_out"),
        ("encoder", "hidden_size"),
        ("model", "encoder", "d_model"),
    ):
        value = nested_get(cfg, path)
        if value is not None:
            return int(value)
    encoder = getattr(model, "encoder", None)
    for name in ("d_model", "feat_out", "hidden_size"):
        value = getattr(encoder, name, None)
        if isinstance(value, int):
            return value
    return None


def tokenizer_vocabulary_size(model: Any) -> int | None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    value = getattr(tokenizer, "vocab_size", None)
    if value is not None:
        return int(value)
    vocabulary = getattr(tokenizer, "vocab", None)
    if vocabulary is not None and hasattr(vocabulary, "__len__"):
        return len(vocabulary)
    inner = getattr(tokenizer, "tokenizer", None)
    get_vocab = getattr(inner, "get_vocab", None)
    if get_vocab is not None:
        return len(get_vocab())
    return None


def sample_rate(model: Any) -> int | None:
    cfg = config_container(model)
    for path in (
        ("preprocessor", "sample_rate"),
        ("model", "sample_rate"),
        ("sample_rate",),
    ):
        value = nested_get(cfg, path)
        if value is not None:
            return int(value)
    return None


def extract_streaming_contexts(model: Any) -> list[list[int]]:
    encoder = getattr(model, "encoder", None)
    values: list[Any] = []
    for source in (encoder, getattr(encoder, "streaming_cfg", None), config_container(model).get("encoder", {})):
        if source is None:
            continue
        if isinstance(source, Mapping):
            value = source.get("att_context_size")
        else:
            value = getattr(source, "att_context_size", None)
        if value is not None:
            values = list(value)
            break
    contexts = normalize_contexts(values)
    if contexts:
        return contexts
    return []


def configured_streaming_contexts() -> list[list[int]]:
    return [item["att_context_size"] for item in load_runtime_config()["streaming_contexts"]]


def normalize_contexts(values: list[Any]) -> list[list[int]]:
    if not values:
        return []
    if len(values) == 2 and all(isinstance(item, int) for item in values):
        return [[int(values[0]), int(values[1])]]
    contexts: list[list[int]] = []
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            contexts.append([int(item[0]), int(item[1])])
    return contexts


def find_prompt_index(model: Any, prompt: str) -> int | None:
    for mapping in prompt_mappings(model):
        if prompt in mapping:
            value = mapping[prompt]
            if isinstance(value, int):
                return value
            if hasattr(value, "item"):
                return int(value.item())
    return None


def prompt_mappings(model: Any) -> list[Mapping[str, Any]]:
    mappings: list[Mapping[str, Any]] = []
    queue = [model, getattr(model, "decoder", None), getattr(model, "joint", None), getattr(model, "encoder", None)]
    for obj in queue:
        if obj is None:
            continue
        for name in ("prompt_dictionary", "prompt_dict", "lang_dict", "language_id_map", "language_ids"):
            value = getattr(obj, name, None)
            if isinstance(value, Mapping):
                mappings.append(value)
    mappings.extend(find_prompt_mappings(config_container(model)))
    return mappings


def find_prompt_mappings(value: Any) -> list[Mapping[str, Any]]:
    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if any(key in value for key in ("sl-SI", "sl", "auto")):
            found.append(value)
        for child in value.values():
            found.extend(find_prompt_mappings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_prompt_mappings(child))
    return found


def prompt_kernel_structure(model: Any) -> list[dict[str, Any]]:
    named_parameters = getattr(model, "named_parameters", None)
    if named_parameters is None:
        return []
    structure = []
    for name, param in named_parameters():
        if PROMPT_PATTERN.search(name):
            shape = list(getattr(param, "shape", []))
            structure.append({"name": name, "shape": shape, "parameters": int(param.numel())})
    return structure


def config_container(model: Any) -> dict[str, Any]:
    cfg = getattr(model, "cfg", {})
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    return cfg if isinstance(cfg, dict) else {}


def nested_get(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def environment_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "nemo_revision": load_runtime_config()["nemo"]["revision"],
    }
    for package in ("torch", "nemo_toolkit", "nemo"):
        try:
            details[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            details[package] = None
    try:
        import torch

        details["cuda_available"] = bool(torch.cuda.is_available())
        details["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            details["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        details["torch_runtime_error"] = str(exc)
    return details


def contract_notes(model: Any, prompt_indices: dict[str, int | None]) -> list[str]:
    notes = []
    if not hasattr(model, "set_inference_prompt"):
        notes.append("Loaded model does not expose set_inference_prompt.")
    if prompt_indices.get("sl-SI") is None:
        notes.append("sl-SI prompt index was not found by local introspection.")
    return notes
