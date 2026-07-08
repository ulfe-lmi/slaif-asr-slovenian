#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/.venv/bin/python" -u "${ROOT}/scripts/run_scale2000_decoder_joint_rnnt_artur_earlystop.py" "$@"
