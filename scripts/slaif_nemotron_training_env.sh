#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${SLAIF_VENV:-${repo_root}/.venv}"
cuda_home="${venv_dir}/lib/python3.12/site-packages/nvidia/cuda_nvcc"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  echo "Missing repository-local .venv. Run scripts/setup_runtime_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${cuda_home}/nvvm" ]]; then
  echo "Missing CUDA 12.6 NVCC/NVVM wheel under ${cuda_home}." >&2
  echo "Install requirements/nemotron-training-cu126.txt into .venv." >&2
  return 1 2>/dev/null || exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export NUMBA_CUDA_USE_NVIDIA_BINDING="${NUMBA_CUDA_USE_NVIDIA_BINDING:-1}"
export CUDA_HOME="${cuda_home}"
export CUDA_PATH="${cuda_home}"
export PATH="${cuda_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${cuda_home}/nvvm/lib64:${LD_LIBRARY_PATH:-}"
