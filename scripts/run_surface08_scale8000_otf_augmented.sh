#!/usr/bin/env bash
set -euo pipefail

: "${CUDA_VISIBLE_DEVICES:?set exactly one authorized RTX 3090 selector}"
: "${SLAIF_ASR_RUNS_ROOT:?set the ignored experiment/checkpoint runs root}"
: "${SLAIF_ASR_SCALE8000_RUNS_ROOT:?set the local scale-8000 clean-data root}"
: "${SLAIF_ASR_AUX_RUNS_ROOT:?set the local controller/probe/evaluation runs root}"

stage="${1:?usage: $0 STAGE [additional runner arguments]}"
shift

export NVIDIA_TF32_OVERRIDE=0
export PYTHONUNBUFFERED=1

exec .venv/bin/python -u scripts/run_surface08_scale8000_otf_augmented.py \
  --stage "${stage}" \
  "$@"
