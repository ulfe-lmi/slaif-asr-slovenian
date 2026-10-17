#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to exactly one authorized GPU}"
: "${SLAIF_ASR_RUNS_ROOT:?set SLAIF_ASR_RUNS_ROOT to ignored experiment storage}"
: "${SLAIF_ASR_SCALE8000_RUNS_ROOT:?set SLAIF_ASR_SCALE8000_RUNS_ROOT}"
: "${SLAIF_ASR_AUX_RUNS_ROOT:?set SLAIF_ASR_AUX_RUNS_ROOT}"

export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1

exec .venv/bin/python -u scripts/run_surface08_scale8000_otf_parametric_voiceaug_v1.py "$@"
