#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-piper"
PIPER_DIR="${ROOT_DIR}/.external/piper1-gpl"
PIPER_REPO="https://github.com/OHF-Voice/piper1-gpl.git"
PIPER_REVISION="b4bdd9ebeaea68cbc7a9c4ac907afcb13e7378b6"
REQUIREMENTS="${ROOT_DIR}/requirements/piper-tts-gpu.txt"
RECREATE=0

for arg in "$@"; do
  case "${arg}" in
    --recreate)
      RECREATE=1
      ;;
    -h|--help)
      echo "Usage: scripts/setup_piper_tts_env.sh [--recreate]"
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

need_command python3
need_command git

if [[ "${RECREATE}" -eq 1 ]]; then
  rm -rf "${VENV_DIR}" "${PIPER_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -r "${REQUIREMENTS}"

mkdir -p "$(dirname "${PIPER_DIR}")"
if [[ ! -d "${PIPER_DIR}/.git" ]]; then
  git clone "${PIPER_REPO}" "${PIPER_DIR}"
fi
git -C "${PIPER_DIR}" fetch --tags --prune origin
git -C "${PIPER_DIR}" checkout --detach "${PIPER_REVISION}"
git -C "${PIPER_DIR}" reset --hard "${PIPER_REVISION}"
test "$(git -C "${PIPER_DIR}" rev-parse HEAD)" = "${PIPER_REVISION}"

"${VENV_DIR}/bin/python" -m pip install --no-deps "${PIPER_DIR}"

"${VENV_DIR}/bin/python" - <<'PY'
from __future__ import annotations

import importlib.metadata
from pathlib import Path

dist = importlib.metadata.distribution("piper-tts")
metadata_path = Path(dist._path) / "METADATA"  # type: ignore[attr-defined]
text = metadata_path.read_text(encoding="utf-8")
text = text.replace("Requires-Dist: onnxruntime<2,>=1", "Requires-Dist: onnxruntime-gpu==1.22.0")
text = text.replace("Requires-Dist: onnxruntime>=1,<2", "Requires-Dist: onnxruntime-gpu==1.22.0")
metadata_path.write_text(text, encoding="utf-8")
PY

if "${VENV_DIR}/bin/python" -m pip show onnxruntime >/dev/null 2>&1; then
  echo "The CPU onnxruntime package is installed; refusing mixed runtime packages." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m pip check

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *","* ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to exactly one physical GPU before verifying Piper." >&2
  exit 1
fi

"${VENV_DIR}/bin/python" - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)
assert "CUDAExecutionProvider" in providers
PY

"${VENV_DIR}/bin/python" -m piper --help >/dev/null
echo "Piper environment ready: ${VENV_DIR}"
