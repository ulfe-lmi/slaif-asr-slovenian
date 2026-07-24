from __future__ import annotations

import os
import types
import unittest
from unittest import mock

from slaif_asr.gpu_policy import RTX_3090_MIN_VRAM_MIB, require_single_visible_cuda


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool = True,
        count: int = 1,
        name: str = "NVIDIA A100-SXM4-80GB",
        total_vram_mib: int = 80 * 1024,
    ) -> None:
        self._available = available
        self._count = count
        self._name = name
        self._total_vram_mib = total_vram_mib

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        return self._name

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return (8, 0)

    def get_device_properties(self, index: int):
        return types.SimpleNamespace(total_memory=self._total_vram_mib * 1024 * 1024)


class GpuPolicyTests(unittest.TestCase):
    def patch_torch(self, cuda: FakeCuda):
        fake_torch = types.SimpleNamespace(cuda=cuda, version=types.SimpleNamespace(cuda="12.6"))
        return mock.patch.dict("sys.modules", {"torch": fake_torch})

    def test_physical_gpu_1_maps_to_logical_cuda_0(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), self.patch_torch(FakeCuda()):
            info = require_single_visible_cuda()
        self.assertEqual(info.physical_selector, "1")
        self.assertEqual(info.logical_device, "cuda:0")
        self.assertIn("A100", info.device_name)

    def test_a100_is_accepted(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), self.patch_torch(
            FakeCuda(name="NVIDIA A100-SXM4-80GB")
        ):
            self.assertEqual(require_single_visible_cuda().visible_device_count, 1)

    def test_2080_ti_is_accepted(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), self.patch_torch(
            FakeCuda(name="NVIDIA GeForce RTX 2080 Ti")
        ):
            self.assertIn("2080 Ti", require_single_visible_cuda().device_name)

    def test_rtx_3090_with_at_least_22_gib_is_accepted(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), self.patch_torch(
            FakeCuda(name="NVIDIA GeForce RTX 3090", total_vram_mib=24 * 1024)
        ):
            info = require_single_visible_cuda()
        self.assertEqual(info.device_name, "NVIDIA GeForce RTX 3090")
        self.assertEqual(info.visible_device_count, 1)

    def test_rtx_3090_below_22_gib_is_rejected(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), self.patch_torch(
            FakeCuda(name="NVIDIA GeForce RTX 3090", total_vram_mib=RTX_3090_MIN_VRAM_MIB - 1)
        ):
            with self.assertRaisesRegex(RuntimeError, "requires at least"):
                require_single_visible_cuda()

    def test_rtx_3090_ti_is_not_implicitly_accepted(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), self.patch_torch(
            FakeCuda(name="NVIDIA GeForce RTX 3090 Ti", total_vram_mib=24 * 1024)
        ):
            with self.assertRaisesRegex(RuntimeError, "unsupported development GPU"):
                require_single_visible_cuda()

    def test_multiple_visible_gpus_fail(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), self.patch_torch(FakeCuda(count=2)):
            with self.assertRaises(RuntimeError):
                require_single_visible_cuda()

    def test_no_visible_gpu_fails(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), self.patch_torch(FakeCuda()):
            with self.assertRaises(RuntimeError):
                require_single_visible_cuda()

    def test_cpu_fallback_fails(self) -> None:
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), self.patch_torch(FakeCuda(available=False)):
            with self.assertRaises(RuntimeError):
                require_single_visible_cuda()


if __name__ == "__main__":
    unittest.main()
