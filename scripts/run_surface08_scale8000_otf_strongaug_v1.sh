#!/usr/bin/env bash
set -euo pipefail

: "${SLAIF_ASR_RUNS_ROOT:?set SLAIF_ASR_RUNS_ROOT to ignored checkpoint storage}"
: "${SLAIF_ASR_SCALE8000_RUNS_ROOT:?set SLAIF_ASR_SCALE8000_RUNS_ROOT}"
: "${SLAIF_ASR_AUX_RUNS_ROOT:?set SLAIF_ASR_AUX_RUNS_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1

exec .venv/bin/python -u scripts/run_surface08_scale8000_otf_strongaug_v1.py "$@"
