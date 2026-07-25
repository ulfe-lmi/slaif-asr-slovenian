#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SLAIF_VALIDATION_GPU="${SLAIF_VALIDATION_GPU:-${CUDA_VISIBLE_DEVICES}}"
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1

python_bin="${SLAIF_PYTHON:-.venv/bin/python}"
runner="scripts/run_fixed_scale2000_surface08_full_encoder.py"

"$python_bin" -u "$runner" --stage verify-inputs
"$python_bin" -u "$runner" --stage probe-hardware
"$python_bin" -u "$runner" --stage probe-surface
"$python_bin" -u "$runner" --stage probe-microbatch
"$python_bin" -u "$runner" --stage train
"$python_bin" -u "$runner" --stage evaluate-directional
"$python_bin" -u "$runner" --stage summarize
