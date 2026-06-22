#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
requirements="${repo_root}/requirements/nemotron-training-cu126.txt"
venv_python="${repo_root}/.venv/bin/python"

if [[ ! -x "${venv_python}" ]]; then
  echo "Missing repository-local .venv. Run scripts/setup_runtime_env.sh first." >&2
  exit 1
fi

"${venv_python}" -m pip install -r "${requirements}"
# shellcheck source=/dev/null
source "${repo_root}/scripts/slaif_nemotron_training_env.sh"
"${venv_python}" -m pip check
"${venv_python}" - <<'PY'
import importlib.metadata
import os
import torch
import numba
import llvmlite

visible = os.environ.get("CUDA_VISIBLE_DEVICES")
assert visible and "," not in visible, visible
assert torch.__version__ == "2.7.1+cu126", torch.__version__
assert torch.version.cuda == "12.6", torch.version.cuda
assert numba.__version__ == "0.61.2", numba.__version__
assert llvmlite.__version__ == "0.44.0", llvmlite.__version__
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
assert any(name in torch.cuda.get_device_name(0) for name in ("2080 Ti", "A100"))
print(f"Python={os.sys.version.split()[0]}")
print(f"PyTorch={torch.__version__}")
print(f"CUDA runtime={torch.version.cuda}")
print(f"Numba={numba.__version__}")
print(f"llvmlite={llvmlite.__version__}")
print(f"NVCC wheel={importlib.metadata.version('nvidia-cuda-nvcc-cu12')}")
print(f"CUDA device={torch.cuda.get_device_name(0)}")
PY
"${venv_python}" - <<'PY'
import torch

torch.empty(1, device="cuda")
torch.cuda.reset_peak_memory_stats(0)
log_probs = torch.randn(1, 2, 2, 3, device="cuda", requires_grad=True).log_softmax(-1)
targets = torch.tensor([[1]], dtype=torch.int32, device="cuda")
input_lengths = torch.tensor([2], dtype=torch.int32, device="cuda")
target_lengths = torch.tensor([1], dtype=torch.int32, device="cuda")
try:
    from nemo.collections.asr.losses.rnnt import RNNTLoss

    loss_fn = RNNTLoss(num_classes=2, reduction="mean_batch")
    loss = loss_fn(log_probs=log_probs, targets=targets, input_lengths=input_lengths, target_lengths=target_lengths)
except Exception as exc:
    raise RuntimeError(f"minimal RNNT CUDA loss smoke failed: {exc}") from exc
loss.backward()
print(f"RNNT CUDA loss smoke PASSED loss={float(loss.detach().cpu()):.6f}")
print(f"Peak VRAM MiB={torch.cuda.max_memory_allocated(0) / 1024 / 1024:.1f}")
PY
