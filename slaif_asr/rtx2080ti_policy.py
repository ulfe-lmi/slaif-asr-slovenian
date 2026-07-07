from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class GpuInventoryRow:
    index: int
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_percent: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Rtx2080TiDevice:
    cuda_visible_devices: str
    physical_selector: str
    logical_device: str
    device_name: str
    total_vram_mib: int
    free_vram_mib: int
    pytorch_cuda: str | None
    visible_device_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_nvidia_smi_csv(text: str) -> list[GpuInventoryRow]:
    rows: list[GpuInventoryRow] = []
    reader = csv.reader(line for line in text.splitlines() if line.strip())
    for raw in reader:
        if len(raw) != 5:
            raise ValueError(f"expected five nvidia-smi CSV fields, got {len(raw)}")
        rows.append(
            GpuInventoryRow(
                index=int(raw[0].strip()),
                name=raw[1].strip(),
                memory_total_mib=int(raw[2].strip()),
                memory_used_mib=int(raw[3].strip()),
                utilization_percent=int(raw[4].strip()),
            )
        )
    return rows


def nvidia_smi_inventory() -> list[GpuInventoryRow]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return parse_nvidia_smi_csv(completed.stdout)


def require_single_visible_rtx2080ti() -> Rtx2080TiDevice:
    import torch

    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector is None or not selector.strip() or "," in selector:
        raise RuntimeError("training requires exactly one visible physical RTX 2080 Ti")
    selector = selector.strip()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU training is forbidden")
    visible = torch.cuda.device_count()
    if visible != 1:
        raise RuntimeError(f"expected one visible CUDA device, saw {visible}")
    name = torch.cuda.get_device_name(0)
    if "RTX 2080 Ti" not in name:
        raise RuntimeError(f"expected NVIDIA GeForce RTX 2080 Ti, saw {name}")
    if "A100" in name:
        raise RuntimeError("A100 is rejected for this work order")
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return Rtx2080TiDevice(
        cuda_visible_devices=selector,
        physical_selector=selector,
        logical_device="cuda:0",
        device_name=name,
        total_vram_mib=int(total_bytes // 1024 // 1024),
        free_vram_mib=int(free_bytes // 1024 // 1024),
        pytorch_cuda=torch.version.cuda,
        visible_device_count=visible,
    )


def select_microbatch(candidates: Sequence[int], outcomes: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if list(candidates) != [8, 4, 2, 1]:
        raise ValueError("microbatch candidates must be [8, 4, 2, 1]")
    for candidate in candidates:
        outcome = outcomes.get(candidate, {})
        if outcome.get("status") == "PASSED":
            if 8 % candidate != 0:
                raise ValueError("physical microbatch must divide effective batch size 8")
            return {
                "physical_microbatch": candidate,
                "gradient_accumulation_steps": 8 // candidate,
                "effective_batch_size": 8,
                "status": "PASSED",
            }
    if outcomes.get(1, {}).get("status") == "FAILED":
        return {
            "status": "ENVIRONMENT_BLOCKED",
            "reason": "physical microbatch 1 failed",
            "physical_microbatch": None,
            "gradient_accumulation_steps": None,
            "effective_batch_size": 8,
        }
    return {
        "status": "ENVIRONMENT_BLOCKED",
        "reason": "no passing physical microbatch",
        "physical_microbatch": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": 8,
    }
