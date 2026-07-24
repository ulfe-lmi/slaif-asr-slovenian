from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


RTX_3090_NAME = "NVIDIA GeForce RTX 3090"
RTX_3090_MIN_VRAM_MIB = 22 * 1024


@dataclass(frozen=True)
class SingleGpuInfo:
    cuda_visible_devices: str
    physical_selector: str
    logical_device: str
    device_name: str
    capability: tuple[int, int]
    total_vram_mib: int
    pytorch_cuda: str | None
    visible_device_count: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["capability"] = list(self.capability)
        return payload


def _single_selector(value: str | None) -> str:
    if value is None or not value.strip():
        raise RuntimeError("CUDA_VISIBLE_DEVICES must select exactly one physical GPU")
    selector = value.strip()
    if "," in selector:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly one physical GPU")
    return selector


def is_approved_development_gpu(device_name: str, total_vram_mib: int) -> bool:
    if "A100" in device_name or "2080 Ti" in device_name:
        return True
    return device_name == RTX_3090_NAME and total_vram_mib >= RTX_3090_MIN_VRAM_MIB


def require_single_visible_cuda(*, allowed_name_fragments: Sequence[str] | None = None) -> SingleGpuInfo:
    import os

    import torch

    selector = _single_selector(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; CPU fallback is forbidden")
    visible_count = torch.cuda.device_count()
    if visible_count != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, saw {visible_count}")
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    total_vram_mib = int(props.total_memory // 1024 // 1024)
    if name == RTX_3090_NAME and total_vram_mib < RTX_3090_MIN_VRAM_MIB:
        raise RuntimeError(
            f"{RTX_3090_NAME} requires at least {RTX_3090_MIN_VRAM_MIB} MiB VRAM, "
            f"saw {total_vram_mib} MiB"
        )
    if allowed_name_fragments is None:
        if not is_approved_development_gpu(name, total_vram_mib):
            raise RuntimeError(f"unsupported development GPU: {name}")
    elif not any(fragment in name for fragment in allowed_name_fragments):
        allowed = ", ".join(allowed_name_fragments)
        raise RuntimeError(f"expected one of [{allowed}], saw {name}")
    return SingleGpuInfo(
        cuda_visible_devices=selector,
        physical_selector=selector,
        logical_device="cuda:0",
        device_name=name,
        capability=tuple(torch.cuda.get_device_capability(0)),
        total_vram_mib=total_vram_mib,
        pytorch_cuda=torch.version.cuda,
        visible_device_count=visible_count,
    )
