#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nemo_revision="8044a3924bfcfe8ef71d792bb73bf274fe853575"
nemo_root="${NEMO_ROOT:-${repo_root}/.external/NeMo}"
venv_dir="${SLAIF_VENV:-${repo_root}/.venv}"
runtime_requirements="${repo_root}/requirements/runtime-cu126.txt"
runtime_constraints="${repo_root}/requirements/runtime-cu126-constraints.txt"
recreate=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_runtime_env.sh [--recreate]

Creates or updates the repository-local .venv runtime.

Options:
  --recreate  Remove the existing .venv before installing.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      recreate=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 127
  fi
}

require_command git
require_command python3

python_version="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "${python_version}" != "3.12" ]]; then
  echo "Python 3.12 is required for the pinned M1 runtime; found ${python_version}." >&2
  exit 1
fi

if [[ ! -f "${runtime_requirements}" || ! -f "${runtime_constraints}" ]]; then
  echo "Missing runtime requirements or constraints file." >&2
  exit 1
fi

if [[ "${recreate}" -eq 1 ]]; then
  rm -rf "${venv_dir}"
fi

mkdir -p "$(dirname "${nemo_root}")"

if [[ ! -d "${nemo_root}/.git" ]]; then
  git clone https://github.com/NVIDIA-NeMo/NeMo.git "${nemo_root}"
fi

git -C "${nemo_root}" fetch --tags origin
git -C "${nemo_root}" checkout "${nemo_revision}"

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  python3 -m venv "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install -r "${runtime_requirements}"
"${venv_dir}/bin/python" -m pip install -e "${repo_root}" -c "${runtime_constraints}"
"${venv_dir}/bin/python" -m pip install -e "${nemo_root}[asr]" -c "${runtime_constraints}"
"${venv_dir}/bin/python" -m pip check

"${venv_dir}/bin/python" - <<'PY'
import importlib.metadata
import torch

print(f"Python runtime OK")
print(f"PyTorch={torch.__version__}")
print(f"PyTorch CUDA runtime={torch.version.cuda}")
print(f"CUDA available={torch.cuda.is_available()}")
print(f"CUDA device count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device 0={torch.cuda.get_device_name(0)}")
print(f"NeMo toolkit={importlib.metadata.version('nemo_toolkit')}")
import nemo.collections.asr  # noqa: F401
print("NeMo ASR import OK")
PY

cat <<EOF
Runtime setup completed.
NEMO_ROOT=${nemo_root}
SLAIF_VENV=${venv_dir}
PyTorch pin=torch==2.7.1+cu126
NeMo revision=${nemo_revision}
Activate with:
source .venv/bin/activate
EOF
