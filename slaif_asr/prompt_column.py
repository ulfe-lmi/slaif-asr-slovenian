from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PromptColumnSelection:
    prompt_name: str
    prompt_index: int
    encoder_width: int
    num_prompts: int
    selected_column: int
    first_linear_name: str
    first_linear_shape: tuple[int, int]
    effective_trainable_parameters: int


@dataclass(frozen=True)
class IntegrityReport:
    selected_prompt: str
    prompt_index: int
    selected_column: int
    effective_trainable_parameters: int
    changed_tensors: list[str]
    unexpected_changed_tensors: list[str]
    unexpected_changed_elements: int
    selected_column_changed: bool
    other_columns_bitwise_identical: bool
    bias_bitwise_identical: bool
    tensor_shapes_match: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


class PromptColumnDelta(nn.Module):
    """Additive delta equivalent to changing one input column of a frozen Linear."""

    def __init__(self, linear: nn.Linear, selected_column: int):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("PromptColumnDelta requires an nn.Linear module")
        if linear.weight.ndim != 2:
            raise ValueError("linear weight must be rank 2")
        if selected_column < 0 or selected_column >= linear.in_features:
            raise ValueError(f"selected_column {selected_column} is outside {linear.in_features} input features")
        self.linear = linear
        self.selected_column = int(selected_column)
        self.delta = nn.Parameter(torch.zeros(linear.out_features, dtype=linear.weight.dtype, device=linear.weight.device))
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(inputs)
        activation = inputs[..., self.selected_column].unsqueeze(-1)
        return base_output + activation * self.delta

    @property
    def effective_trainable_parameters(self) -> int:
        return int(self.delta.numel())


def freeze_all_original_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def prompt_dictionary(model: Any) -> dict[str, int]:
    cfg = getattr(model, "cfg", None)
    defaults = getattr(cfg, "model_defaults", None)
    mapping = getattr(defaults, "prompt_dictionary", None)
    if mapping is None and isinstance(defaults, dict):
        mapping = defaults.get("prompt_dictionary")
    if mapping is None:
        raise ValueError("model configuration does not expose model_defaults.prompt_dictionary")
    return {str(key): int(value) for key, value in dict(mapping).items()}


def num_prompts(model: Any) -> int:
    cfg = getattr(model, "cfg", None)
    defaults = getattr(cfg, "model_defaults", None)
    value = getattr(defaults, "num_prompts", None)
    if value is None and isinstance(defaults, dict):
        value = defaults.get("num_prompts")
    if value is None:
        value = getattr(model, "num_prompts", None)
    if value is None:
        raise ValueError("model does not expose num_prompts")
    return int(value)


def encoder_width(model: Any) -> int:
    cfg = getattr(model, "cfg", None)
    encoder_cfg = getattr(cfg, "encoder", None)
    for name in ("d_model", "feat_out", "hidden_size"):
        value = getattr(encoder_cfg, name, None)
        if value is not None:
            return int(value)
    encoder = getattr(model, "encoder", None)
    for name in ("d_model", "feat_out", "hidden_size"):
        value = getattr(encoder, name, None)
        if isinstance(value, int):
            return value
    raise ValueError("model does not expose encoder width")


def first_prompt_linear(model: Any) -> tuple[str, nn.Linear]:
    kernel = getattr(model, "prompt_kernel", None)
    if kernel is None:
        raise ValueError("model does not expose prompt_kernel")
    for name, module in kernel.named_modules():
        if isinstance(module, nn.Linear):
            full_name = "prompt_kernel" if not name else f"prompt_kernel.{name}"
            return full_name, module
    raise ValueError("prompt_kernel does not contain an nn.Linear")


def derive_prompt_column_selection(model: Any, prompt_name: str = "sl-SI") -> PromptColumnSelection:
    prompts = prompt_dictionary(model)
    if prompt_name not in prompts:
        raise ValueError(f"prompt {prompt_name!r} is not present in model prompt dictionary")
    prompt_index = int(prompts[prompt_name])
    width = encoder_width(model)
    prompts_count = num_prompts(model)
    selected_column = width + prompt_index
    linear_name, linear = first_prompt_linear(model)
    if linear.in_features != width + prompts_count:
        raise ValueError(
            f"unexpected prompt first-linear input shape: {linear.in_features} != {width} + {prompts_count}"
        )
    if selected_column >= linear.in_features:
        raise ValueError(f"selected prompt column {selected_column} is outside first-linear shape")
    return PromptColumnSelection(
        prompt_name=prompt_name,
        prompt_index=prompt_index,
        encoder_width=width,
        num_prompts=prompts_count,
        selected_column=selected_column,
        first_linear_name=linear_name,
        first_linear_shape=(int(linear.out_features), int(linear.in_features)),
        effective_trainable_parameters=int(linear.out_features),
    )


def replace_module(root: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() and isinstance(parent, nn.Sequential) else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit() and isinstance(parent, nn.Sequential):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def install_prompt_delta(model: nn.Module, prompt_name: str = "sl-SI") -> tuple[PromptColumnSelection, PromptColumnDelta]:
    freeze_all_original_parameters(model)
    selection = derive_prompt_column_selection(model, prompt_name)
    _, linear = first_prompt_linear(model)
    wrapper = PromptColumnDelta(linear, selection.selected_column)
    replace_module(model, selection.first_linear_name, wrapper)
    wrapper.delta.requires_grad = True
    return selection, wrapper


def trainable_delta_parameters(wrapper: PromptColumnDelta, *, weight_decay: float) -> list[nn.Parameter]:
    if weight_decay != 0:
        raise ValueError("prompt-column proof requires weight_decay=0")
    if not wrapper.delta.requires_grad:
        raise ValueError("delta parameter is not trainable")
    return [wrapper.delta]


def merge_prompt_delta(model: nn.Module, selection: PromptColumnSelection) -> nn.Linear:
    module = dict(model.named_modules()).get(selection.first_linear_name)
    if not isinstance(module, PromptColumnDelta):
        raise ValueError(f"{selection.first_linear_name} is not a PromptColumnDelta wrapper")
    with torch.no_grad():
        module.linear.weight[:, selection.selected_column].add_(module.delta.to(module.linear.weight.dtype))
    merged_linear = module.linear
    replace_module(model, selection.first_linear_name, merged_linear)
    return merged_linear


def compare_prompt_column_state_dicts(
    base_state: dict[str, torch.Tensor],
    adapted_state: dict[str, torch.Tensor],
    *,
    first_linear_weight_name: str,
    first_linear_bias_name: str | None,
    selected_column: int,
    selected_prompt: str,
    prompt_index: int,
    effective_trainable_parameters: int,
) -> IntegrityReport:
    changed_tensors: list[str] = []
    unexpected_changed_tensors: list[str] = []
    unexpected_changed_elements = 0
    tensor_shapes_match = set(base_state) == set(adapted_state)
    if not tensor_shapes_match:
        unexpected_changed_tensors.extend(sorted(set(base_state).symmetric_difference(adapted_state)))

    selected_column_changed = False
    other_columns_bitwise_identical = True
    bias_bitwise_identical = True

    for name in sorted(set(base_state).intersection(adapted_state)):
        base = base_state[name]
        adapted = adapted_state[name]
        if tuple(base.shape) != tuple(adapted.shape):
            tensor_shapes_match = False
            unexpected_changed_tensors.append(name)
            unexpected_changed_elements += max(base.numel(), adapted.numel())
            continue
        if torch.equal(base, adapted):
            continue
        changed_tensors.append(name)
        if name != first_linear_weight_name:
            unexpected_changed_tensors.append(name)
            unexpected_changed_elements += int(base.numel())
            if first_linear_bias_name is not None and name == first_linear_bias_name:
                bias_bitwise_identical = False
            continue
        if base.ndim != 2 or selected_column >= base.shape[1]:
            unexpected_changed_tensors.append(name)
            unexpected_changed_elements += int(base.numel())
            continue
        selected_column_changed = not torch.equal(base[:, selected_column], adapted[:, selected_column])
        left_same = torch.equal(base[:, :selected_column], adapted[:, :selected_column])
        right_same = torch.equal(base[:, selected_column + 1 :], adapted[:, selected_column + 1 :])
        other_columns_bitwise_identical = left_same and right_same
        if not other_columns_bitwise_identical:
            unexpected_changed_tensors.append(name)
            changed_mask = base != adapted
            changed_mask[:, selected_column] = False
            unexpected_changed_elements += int(changed_mask.sum().item())

    return IntegrityReport(
        selected_prompt=selected_prompt,
        prompt_index=prompt_index,
        selected_column=selected_column,
        effective_trainable_parameters=effective_trainable_parameters,
        changed_tensors=changed_tensors,
        unexpected_changed_tensors=sorted(set(unexpected_changed_tensors)),
        unexpected_changed_elements=unexpected_changed_elements,
        selected_column_changed=selected_column_changed,
        other_columns_bitwise_identical=other_columns_bitwise_identical,
        bias_bitwise_identical=bias_bitwise_identical,
        tensor_shapes_match=tensor_shapes_match,
    )


def write_integrity_report(report: IntegrityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
