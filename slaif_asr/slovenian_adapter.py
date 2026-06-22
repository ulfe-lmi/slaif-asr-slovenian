from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from slaif_asr.prompt_column import derive_prompt_column_selection, freeze_all_original_parameters


@dataclass(frozen=True)
class ResidualAdapterSelection:
    prompt_name: str
    prompt_index: int
    encoder_width: int
    num_prompts: int
    selected_column: int
    prompt_kernel_output_width: int
    rank: int
    trainable_parameters: int
    base_prompt_kernel_class: str


@dataclass(frozen=True)
class ResidualIntegrityReport:
    selected_prompt: str
    prompt_index: int
    selected_column: int
    rank: int
    trainable_parameters: int
    base_tensors_identical: bool
    prompt_kernel_identical: bool
    encoder_identical: bool
    decoder_joint_identical: bool
    tokenizer_config_identical: bool
    changed_adapter_tensors: list[str]
    unexpected_base_tensors: list[str]
    missing_base_tensors: list[str]
    unexpected_changed_base_tensors: list[str]
    unexpected_changed_elements: int
    prompt_dictionary_unchanged: bool
    step_zero_equivalent: bool
    non_sl_residual_zero: bool
    optimizer_parameter_count: int

    def passed(self) -> bool:
        return (
            self.base_tensors_identical
            and self.prompt_kernel_identical
            and self.encoder_identical
            and self.decoder_joint_identical
            and self.tokenizer_config_identical
            and not self.unexpected_base_tensors
            and not self.missing_base_tensors
            and not self.unexpected_changed_base_tensors
            and self.unexpected_changed_elements == 0
            and self.prompt_dictionary_unchanged
            and self.step_zero_equivalent
            and self.non_sl_residual_zero
            and self.optimizer_parameter_count == self.trainable_parameters
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


class SlovenianResidualAdapter(nn.Module):
    """Frozen prompt kernel plus a Slovenian-only residual bottleneck."""

    def __init__(self, base_prompt_kernel: nn.Module, *, selected_column: int, output_width: int, rank: int):
        super().__init__()
        if rank <= 0:
            raise ValueError("adapter rank must be positive")
        self.base_prompt_kernel = base_prompt_kernel
        self.selected_column = int(selected_column)
        self.output_width = int(output_width)
        self.rank = int(rank)
        for parameter in self.base_prompt_kernel.parameters():
            parameter.requires_grad = False
        self.down = nn.Linear(self.output_width, self.rank, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.rank, self.output_width, bias=False)
        nn.init.zeros_(self.up.weight)
        first_parameter = next(self.base_prompt_kernel.parameters(), None)
        if first_parameter is not None:
            self.down.to(device=first_parameter.device, dtype=first_parameter.dtype)
            self.up.to(device=first_parameter.device, dtype=first_parameter.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_prompt_kernel(inputs).detach()
        sl_active = inputs[..., self.selected_column].detach().unsqueeze(-1)
        residual = self.up(self.activation(self.down(base_output)))
        return base_output + sl_active * residual

    @property
    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.adapter_parameters()))

    def adapter_parameters(self) -> list[nn.Parameter]:
        return [self.down.weight, self.up.weight]

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "down.weight": self.down.weight.detach().cpu().clone(),
            "up.weight": self.up.weight.detach().cpu().clone(),
        }


def _module_output_width(module: nn.Module, input_features: int, *, device: torch.device, dtype: torch.dtype) -> int:
    with torch.no_grad():
        probe = torch.zeros(1, 1, input_features, device=device, dtype=dtype)
        output = module(probe)
    if output.ndim < 1:
        raise ValueError("prompt kernel output must have at least one dimension")
    return int(output.shape[-1])


def derive_residual_adapter_selection(model: Any, *, rank: int, prompt_name: str = "sl-SI") -> ResidualAdapterSelection:
    column = derive_prompt_column_selection(model, prompt_name)
    prompt_kernel = getattr(model, "prompt_kernel", None)
    if not isinstance(prompt_kernel, nn.Module):
        raise ValueError("model does not expose an nn.Module prompt_kernel")
    first_parameter = next(prompt_kernel.parameters(), None)
    if first_parameter is None:
        raise ValueError("prompt_kernel has no parameters")
    output_width = _module_output_width(
        prompt_kernel,
        column.first_linear_shape[1],
        device=first_parameter.device,
        dtype=first_parameter.dtype,
    )
    trainable = output_width * rank * 2
    return ResidualAdapterSelection(
        prompt_name=column.prompt_name,
        prompt_index=column.prompt_index,
        encoder_width=column.encoder_width,
        num_prompts=column.num_prompts,
        selected_column=column.selected_column,
        prompt_kernel_output_width=output_width,
        rank=int(rank),
        trainable_parameters=int(trainable),
        base_prompt_kernel_class=f"{prompt_kernel.__class__.__module__}.{prompt_kernel.__class__.__name__}",
    )


def install_slovenian_residual_adapter(
    model: nn.Module,
    *,
    rank: int,
    prompt_name: str = "sl-SI",
) -> tuple[ResidualAdapterSelection, SlovenianResidualAdapter]:
    freeze_all_original_parameters(model)
    selection = derive_residual_adapter_selection(model, rank=rank, prompt_name=prompt_name)
    base_prompt_kernel = getattr(model, "prompt_kernel")
    adapter = SlovenianResidualAdapter(
        base_prompt_kernel,
        selected_column=selection.selected_column,
        output_width=selection.prompt_kernel_output_width,
        rank=rank,
    )
    setattr(model, "prompt_kernel", adapter)
    for parameter in adapter.adapter_parameters():
        parameter.requires_grad = True
    return selection, adapter


def trainable_adapter_parameters(adapter: SlovenianResidualAdapter, *, weight_decay: float) -> list[nn.Parameter]:
    if weight_decay != 0:
        raise ValueError("residual-adapter proof requires weight_decay=0")
    params = adapter.adapter_parameters()
    if any(not parameter.requires_grad for parameter in params):
        raise ValueError("all adapter parameters must require gradients")
    return params


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape)).encode("utf-8"))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def original_state_dict_from_wrapped_model(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    prompt_kernel = getattr(model, "prompt_kernel", None)
    wrapped = isinstance(prompt_kernel, SlovenianResidualAdapter)
    for name, tensor in model.state_dict().items():
        if wrapped and name.startswith("prompt_kernel.base_prompt_kernel."):
            original_name = "prompt_kernel." + name.removeprefix("prompt_kernel.base_prompt_kernel.")
            state[original_name] = tensor.detach().cpu()
        elif wrapped and (
            name.startswith("prompt_kernel.down.")
            or name.startswith("prompt_kernel.up.")
            or name.startswith("prompt_kernel.activation.")
        ):
            continue
        else:
            state[name] = tensor.detach().cpu()
    return state


def state_hashes(state: dict[str, torch.Tensor]) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in state.items()}


def adapter_state_hashes(adapter: SlovenianResidualAdapter) -> dict[str, str]:
    return {name: tensor_sha256(tensor) for name, tensor in adapter.adapter_state_dict().items()}


def changed_adapter_tensors(initial: dict[str, str], final: dict[str, str]) -> list[str]:
    changed = []
    for name in sorted(set(initial) | set(final)):
        if initial.get(name) != final.get(name):
            changed.append(name)
    return changed


def compare_base_hashes(base: dict[str, str], current: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
    missing = sorted(set(base) - set(current))
    unexpected = sorted(set(current) - set(base))
    changed = sorted(name for name in set(base) & set(current) if base[name] != current[name])
    return unexpected, missing, changed


def count_changed_elements(base_state: dict[str, torch.Tensor], current_state: dict[str, torch.Tensor], names: list[str]) -> int:
    total = 0
    for name in names:
        if name not in base_state or name not in current_state:
            continue
        base = base_state[name]
        current = current_state[name]
        if tuple(base.shape) != tuple(current.shape):
            total += max(base.numel(), current.numel())
        else:
            total += int((base != current).sum().item())
    return total


def category_integrity(changed: list[str], prefix: str) -> bool:
    return not any(name.startswith(prefix) for name in changed)


def write_adapter_artifact(
    path: Path,
    *,
    adapter: SlovenianResidualAdapter,
    selection: ResidualAdapterSelection,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "1.0",
            "selection": asdict(selection),
            "adapter_state_dict": adapter.adapter_state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_adapter_artifact(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def write_residual_integrity_report(report: ResidualIntegrityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_json(), encoding="utf-8")
