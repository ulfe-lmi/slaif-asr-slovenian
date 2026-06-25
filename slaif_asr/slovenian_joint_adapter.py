from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from slaif_asr.batched_streaming import file_sha256
from slaif_asr.config import REPO_ROOT


ADAPTER_NAME = "sl-si-joint-adapter-v1"
TARGET_LANG = "sl-SI"
ADAPTER_TYPE = "nemo.collections.common.parts.adapter_modules.LinearAdapter"
STRATEGY_TYPE = "nemo.core.classes.mixins.adapter_mixin_strategies.ResidualAddAdapterStrategy"


@dataclass(frozen=True)
class JointAdapterSpec:
    name: str
    bottleneck_dim: int
    activation: str
    norm_position: str
    dropout: float
    stochastic_depth: float
    l2_lambda: float
    target_lang: str


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_adapter_spec(path: str | Path) -> JointAdapterSpec:
    payload = json.loads(repo_path(path).read_text(encoding="utf-8"))
    if payload.get("adapter_id") != ADAPTER_NAME:
        raise ValueError("adapter config has unexpected adapter_id")
    if payload.get("target_module") != "model.joint":
        raise ValueError("adapter must target model.joint")
    if payload.get("adapter_type") != ADAPTER_TYPE:
        raise ValueError("adapter config must use NeMo LinearAdapter")
    if payload.get("strategy_type") != STRATEGY_TYPE:
        raise ValueError("adapter config must use ResidualAddAdapterStrategy")
    if payload.get("target_language") != TARGET_LANG:
        raise ValueError("adapter target language must be sl-SI")
    if float(payload.get("dropout")) != 0.0:
        raise ValueError("joint adapter dropout must be zero")
    if int(payload.get("bottleneck_dimension")) <= 0:
        raise ValueError("joint adapter bottleneck must be positive")
    return JointAdapterSpec(
        name=str(payload["adapter_id"]),
        bottleneck_dim=int(payload["bottleneck_dimension"]),
        activation=str(payload["activation"]),
        norm_position=str(payload["normalization_position"]),
        dropout=float(payload["dropout"]),
        stochastic_depth=float(payload["stochastic_depth"]),
        l2_lambda=float(payload["l2_auxiliary_penalty"]),
        target_lang=str(payload["target_language"]),
    )


def joint_hidden_dim(model: Any) -> int:
    joint = model.joint
    value = getattr(joint, "joint_hidden", None)
    if value is None:
        joint_net = getattr(joint, "joint_net", None)
        if joint_net is not None:
            for module in joint_net:
                if hasattr(module, "in_features"):
                    value = int(module.in_features)
                    break
    if value is None:
        raise ValueError("could not derive model.joint.joint_hidden")
    return int(value)


def expected_trainable_count(joint_hidden: int, bottleneck_dim: int) -> int:
    return 2 * int(joint_hidden) * int(bottleneck_dim) + 2 * int(joint_hidden)


def adapter_config_for_nemo(spec: JointAdapterSpec, *, joint_hidden: int) -> Any:
    try:
        from nemo.collections.common.parts.adapter_modules import LinearAdapterConfig
        from nemo.core.classes.mixins.adapter_mixin_strategies import ResidualAddAdapterStrategyConfig
    except ModuleNotFoundError:
        return {
            "_target_": ADAPTER_TYPE,
            "in_features": int(joint_hidden),
            "dim": spec.bottleneck_dim,
            "activation": spec.activation,
            "norm_position": spec.norm_position,
            "dropout": spec.dropout,
            "adapter_strategy": {
                "_target_": STRATEGY_TYPE,
                "stochastic_depth": spec.stochastic_depth,
                "l2_lambda": spec.l2_lambda,
            },
        }
    return LinearAdapterConfig(
        in_features=int(joint_hidden),
        dim=spec.bottleneck_dim,
        activation=spec.activation,
        norm_position=spec.norm_position,
        dropout=spec.dropout,
        adapter_strategy=ResidualAddAdapterStrategyConfig(
            stochastic_depth=spec.stochastic_depth,
            l2_lambda=spec.l2_lambda,
        ),
    )


def add_joint_adapter(model: Any, spec: JointAdapterSpec) -> dict[str, Any]:
    hidden = joint_hidden_dim(model)
    if hasattr(model.joint, "get_adapter_module") and model.joint.get_adapter_module(spec.name) is None:
        if hasattr(model, "add_adapter"):
            model.add_adapter(f"joint:{spec.name}", adapter_config_for_nemo(spec, joint_hidden=hidden))
        else:
            model.joint.add_adapter(spec.name, adapter_config_for_nemo(spec, joint_hidden=hidden))
    elif not hasattr(model.joint, "get_adapter_module"):
        model.joint.add_adapter(spec.name, adapter_config_for_nemo(spec, joint_hidden=hidden))
    module = adapter_module(model, spec.name)
    try:
        joint_device = next(model.joint.parameters()).device
        module.to(joint_device)
    except StopIteration:
        pass
    disable_joint_adapters(model)
    return {
        "adapter_name": spec.name,
        "joint_hidden": hidden,
        "bottleneck_dim": spec.bottleneck_dim,
        "expected_trainable_parameters": expected_trainable_count(hidden, spec.bottleneck_dim),
        "enabled_adapters": enabled_joint_adapters(model),
    }


def disable_joint_adapters(model: Any) -> None:
    if hasattr(model.joint, "set_enabled_adapters"):
        if hasattr(model.joint, "is_adapter_available") and not model.joint.is_adapter_available():
            return
        model.joint.set_enabled_adapters(enabled=False)


def enable_for_target_language(model: Any, target_lang: str, *, adapter_name: str = ADAPTER_NAME) -> list[str]:
    if not target_lang:
        raise ValueError("target language is required for adapter gating")
    if target_lang != TARGET_LANG:
        disable_joint_adapters(model)
        return []
    disable_joint_adapters(model)
    model.joint.set_enabled_adapters(name=adapter_name, enabled=True)
    enabled = enabled_joint_adapters(model)
    if enabled != [adapter_name]:
        raise RuntimeError(f"expected exactly {adapter_name} enabled, found {enabled}")
    return enabled


def enabled_joint_adapters(model: Any) -> list[str]:
    if hasattr(model.joint, "get_enabled_adapters"):
        return list(model.joint.get_enabled_adapters())
    return []


def adapter_module(model: Any, adapter_name: str = ADAPTER_NAME) -> Any:
    module = model.joint.get_adapter_module(adapter_name)
    if module is None:
        raise RuntimeError(f"adapter {adapter_name} is not installed")
    return module


def freeze_all_pretrained(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def trainable_adapter_named_parameters(model: Any, adapter_name: str = ADAPTER_NAME) -> list[tuple[str, Any]]:
    prefix = f"joint.adapter_layer.{adapter_name}."
    return [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefix)]


def configure_only_adapter_trainable(model: Any, adapter_name: str = ADAPTER_NAME) -> dict[str, Any]:
    freeze_all_pretrained(model)
    params = trainable_adapter_named_parameters(model, adapter_name)
    if not params:
        raise RuntimeError(f"no parameters found under joint adapter {adapter_name}")
    for _name, parameter in params:
        parameter.requires_grad_(True)
    unexpected = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith(f"joint.adapter_layer.{adapter_name}.")]
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected}")
    return {
        "adapter_name": adapter_name,
        "trainable_parameters": sum(parameter.numel() for _name, parameter in params),
        "trainable_names": [name for name, _parameter in params],
    }


def adapter_parameters(model: Any, adapter_name: str = ADAPTER_NAME) -> list[Any]:
    return [parameter for _name, parameter in trainable_adapter_named_parameters(model, adapter_name)]


def optimizer_parameter_ids(model: Any, adapter_name: str = ADAPTER_NAME) -> set[int]:
    return {id(parameter) for parameter in adapter_parameters(model, adapter_name)}


def verify_optimizer_scope(optimizer: Any, model: Any, adapter_name: str = ADAPTER_NAME) -> None:
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = optimizer_parameter_ids(model, adapter_name)
    if actual != expected:
        raise RuntimeError("optimizer contains parameters outside the named joint adapter")


def state_dict_cpu(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def split_adapter_and_base_state(state: dict[str, Any], adapter_name: str = ADAPTER_NAME) -> tuple[dict[str, Any], dict[str, Any]]:
    needle = f".adapter_layer.{adapter_name}."
    adapter = {name: value for name, value in state.items() if needle in name}
    base = {name: value for name, value in state.items() if needle not in name}
    return base, adapter


def compare_base_state(before: dict[str, Any], after: dict[str, Any], adapter_name: str = ADAPTER_NAME) -> dict[str, Any]:
    base_before, _ = split_adapter_and_base_state(before, adapter_name)
    base_after, _ = split_adapter_and_base_state(after, adapter_name)
    missing = sorted(set(base_before) - set(base_after))
    unexpected = sorted(set(base_after) - set(base_before))
    changed = []
    for name in sorted(set(base_before) & set(base_after)):
        left = base_before[name]
        right = base_after[name]
        if left.shape != right.shape or not bool((left == right).all()):
            changed.append(name)
    return {
        "base_tensors_identical": not missing and not unexpected and not changed,
        "changed_pretrained_tensors": changed,
        "missing_pretrained_tensors": missing,
        "unexpected_pretrained_tensors": unexpected,
    }


def compare_adapter_state(initial: dict[str, Any], trained: dict[str, Any], adapter_name: str = ADAPTER_NAME) -> dict[str, Any]:
    _base_initial, adapter_initial = split_adapter_and_base_state(initial, adapter_name)
    _base_trained, adapter_trained = split_adapter_and_base_state(trained, adapter_name)
    changed = []
    unchanged = []
    for name in sorted(set(adapter_initial) & set(adapter_trained)):
        if adapter_initial[name].shape == adapter_trained[name].shape and bool((adapter_initial[name] == adapter_trained[name]).all()):
            unchanged.append(name)
        else:
            changed.append(name)
    return {
        "adapter_tensor_count": len(adapter_trained),
        "adapter_tensors_changed": changed,
        "adapter_tensors_unchanged": unchanged,
        "only_adapter_tensors_may_change": True,
    }


def save_adapter_artifact(path: Path, *, model: Any, spec: JointAdapterSpec, metadata: dict[str, Any]) -> str:
    import torch

    module = adapter_module(model, spec.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "1.0",
            "adapter_name": spec.name,
            "adapter_config": metadata.get("adapter_config", {}),
            "state_dict": {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )
    return file_sha256(path)


def load_adapter_artifact(path: Path, *, model: Any, spec: JointAdapterSpec) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if payload.get("adapter_name") != spec.name:
        raise RuntimeError("adapter artifact name mismatch")
    add_joint_adapter(model, spec)
    module = adapter_module(model, spec.name)
    module.load_state_dict(payload["state_dict"])
    disable_joint_adapters(model)
    return payload
