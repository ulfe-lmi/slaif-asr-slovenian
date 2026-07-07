from __future__ import annotations

import json
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


ADAPTER_NAME = "sl-si-decoder-lm-adapter-v1"


@dataclass(frozen=True)
class DecoderLMAdapterSpec:
    name: str = ADAPTER_NAME
    bottleneck_dim: int = 128
    dropout: float = 0.0
    activation: str = "silu"


class DecoderLMAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.bottleneck_dim = int(bottleneck_dim)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.down = nn.Linear(self.hidden_size, self.bottleneck_dim)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(float(dropout))
        self.up = nn.Linear(self.bottleneck_dim, self.hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        delta = self.up(self.dropout(self.activation(self.down(self.norm(x)))))
        return x + delta


class TemporaryLMHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(hidden_size), int(vocab_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


def load_decoder_lm_adapter_spec(path: str | Path) -> DecoderLMAdapterSpec:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    adapter = payload.get("adapter", payload)
    if adapter.get("name") != ADAPTER_NAME:
        raise ValueError("decoder LM adapter name mismatch")
    if adapter.get("type") != "bottleneck_residual_mlp":
        raise ValueError("decoder LM adapter type mismatch")
    if int(adapter.get("bottleneck_dimension")) != 128:
        raise ValueError("decoder LM adapter bottleneck must be 128")
    if float(adapter.get("dropout")) != 0.0:
        raise ValueError("decoder LM adapter dropout must be zero")
    return DecoderLMAdapterSpec(
        name=str(adapter["name"]),
        bottleneck_dim=int(adapter["bottleneck_dimension"]),
        dropout=float(adapter["dropout"]),
        activation=str(adapter.get("activation", "silu")),
    )


def decoder_hidden_size(model: Any) -> int:
    decoder = model.decoder
    for name in ("pred_hidden", "hidden_size", "joint_hidden"):
        value = getattr(decoder, name, None)
        if value is not None:
            return int(value)
    for module in decoder.modules():
        if isinstance(module, nn.Embedding):
            return int(module.embedding_dim)
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    raise ValueError("could not derive decoder hidden size")


def install_decoder_lm_adapter(model: Any, spec: DecoderLMAdapterSpec | None = None) -> dict[str, Any]:
    spec = spec or DecoderLMAdapterSpec()
    decoder = model.decoder
    hidden = decoder_hidden_size(model)
    if not hasattr(decoder, "decoder_lm_adapter"):
        decoder.add_module("decoder_lm_adapter", DecoderLMAdapter(hidden, spec.bottleneck_dim, spec.dropout))
    adapter = decoder.decoder_lm_adapter
    adapter.enabled = False
    if not hasattr(decoder, "_slaif_decoder_lm_original_predict"):
        decoder._slaif_decoder_lm_original_predict = decoder.predict

        def wrapped_predict(self, *args: Any, **kwargs: Any):
            output, state = self._slaif_decoder_lm_original_predict(*args, **kwargs)
            return self.decoder_lm_adapter(output), state

        decoder.predict = types.MethodType(wrapped_predict, decoder)
    return {
        "adapter_name": spec.name,
        "hidden_size": hidden,
        "bottleneck_dim": spec.bottleneck_dim,
        "dropout": spec.dropout,
        "trainable_adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "enabled": bool(adapter.enabled),
    }


def enable_decoder_lm_adapter(model: Any) -> list[str]:
    adapter = model.decoder.decoder_lm_adapter
    adapter.enabled = True
    return [ADAPTER_NAME]


def disable_decoder_lm_adapter(model: Any) -> None:
    if hasattr(model.decoder, "decoder_lm_adapter"):
        model.decoder.decoder_lm_adapter.enabled = False


def enabled_decoder_lm_adapters(model: Any) -> list[str]:
    if hasattr(model.decoder, "decoder_lm_adapter") and bool(model.decoder.decoder_lm_adapter.enabled):
        return [ADAPTER_NAME]
    return []


def freeze_pretrained_for_text_only(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def decoder_lm_adapter_named_parameters(model: Any) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("decoder.decoder_lm_adapter.")
    ]


def configure_text_only_trainable(model: Any, lm_head: nn.Module) -> dict[str, Any]:
    freeze_pretrained_for_text_only(model)
    adapter_params = decoder_lm_adapter_named_parameters(model)
    if not adapter_params:
        raise RuntimeError("decoder LM adapter is not installed")
    for _name, parameter in adapter_params:
        parameter.requires_grad_(True)
    for parameter in lm_head.parameters():
        parameter.requires_grad_(True)
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("decoder.decoder_lm_adapter.")
    ]
    if unexpected:
        raise RuntimeError(f"unexpected trainable pretrained parameters: {unexpected}")
    return {
        "adapter_trainable_parameters": sum(parameter.numel() for _name, parameter in adapter_params),
        "lm_head_trainable_parameters": sum(parameter.numel() for parameter in lm_head.parameters()),
        "trainable_names": [name for name, _parameter in adapter_params] + ["temporary_lm_head.*"],
    }


def text_only_optimizer_parameters(model: Any, lm_head: nn.Module) -> list[nn.Parameter]:
    return [parameter for _name, parameter in decoder_lm_adapter_named_parameters(model)] + list(lm_head.parameters())


def verify_text_only_optimizer_scope(optimizer: Any, model: Any, lm_head: nn.Module) -> None:
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in text_only_optimizer_parameters(model, lm_head)}
    if actual != expected:
        raise RuntimeError("optimizer contains parameters outside decoder LM adapter and temporary LM head")


def pretrained_parameters_with_grad(model: Any) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if not name.startswith("decoder.decoder_lm_adapter.") and parameter.grad is not None
    ]


def state_dict_cpu(model: Any) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def compare_pretrained_state(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, Any]:
    changed: list[str] = []
    missing = sorted(set(before) - set(after))
    unexpected = sorted(set(after) - set(before))
    for name in sorted(set(before) & set(after)):
        if name.startswith("decoder.decoder_lm_adapter."):
            continue
        left = before[name]
        right = after[name]
        if left.shape != right.shape or not bool(torch.equal(left, right)):
            changed.append(name)
    return {
        "pretrained_tensors_unchanged": not missing and not unexpected and not changed,
        "changed_pretrained_tensors": changed,
        "missing_tensors": missing,
        "unexpected_tensors": unexpected,
    }


def adapter_state(model: Any) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name.startswith("decoder.decoder_lm_adapter.")}


def save_text_only_artifact(path: Path, *, model: Any, lm_head: nn.Module, metadata: dict[str, Any]) -> str:
    from slaif_asr.batched_streaming import file_sha256

    payload = {
        "adapter_name": ADAPTER_NAME,
        "adapter_state": adapter_state(model),
        "lm_head_state": {name: tensor.detach().cpu().clone() for name, tensor in lm_head.state_dict().items()},
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return file_sha256(path)


def load_text_only_artifact(path: Path, *, model: Any, lm_head: nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("adapter_name") != ADAPTER_NAME:
        raise RuntimeError("decoder LM adapter artifact name mismatch")
    install_decoder_lm_adapter(model)
    model.load_state_dict(payload["adapter_state"], strict=False)
    lm_head.load_state_dict(payload["lm_head_state"])
    disable_decoder_lm_adapter(model)
    return dict(payload.get("metadata", {}))

