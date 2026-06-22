#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${SLAIF_GAMS_VENV:-${repo_root}/.venv-gams}"
requirements="${repo_root}/requirements/gams-generator.txt"
recreate=0

usage() {
  cat <<'EOF'
Usage: scripts/setup_gams_env.sh [--recreate]

Creates or updates the repository-local .venv-gams generator environment.
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

if ! command -v python3 >/dev/null 2>&1; then
  echo "Missing python3." >&2
  exit 127
fi

python_version="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "${python_version}" != "3.12" ]]; then
  echo "Python 3.12 is required for .venv-gams; found ${python_version}." >&2
  exit 1
fi

if [[ "${recreate}" -eq 1 ]]; then
  rm -rf "${venv_dir}"
fi

if [[ ! -x "${venv_dir}/bin/python" ]]; then
  python3 -m venv "${venv_dir}"
fi

"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install -r "${requirements}"
"${venv_dir}/bin/python" -m pip check

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to exactly one physical GPU before verifying GaMS." >&2
  exit 1
fi

"${venv_dir}/bin/python" - <<'PY'
import torch
import transformers
import accelerate
import bitsandbytes

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
device_name = torch.cuda.get_device_name(0)
assert any(name in device_name for name in ("2080 Ti", "A100")), device_name
print(f"PyTorch={torch.__version__}")
print(f"Transformers={transformers.__version__}")
print(f"Accelerate={accelerate.__version__}")
print(f"bitsandbytes={bitsandbytes.__version__}")
print(f"CUDA device={device_name}")
PY

cat <<EOF
GaMS generator environment completed.
SLAIF_GAMS_VENV=${venv_dir}
Use:
source .venv-gams/bin/activate
EOF
