#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv-supertonic"

"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$ROOT/requirements/supertonic3.txt"
"$VENV/bin/python" -m pip install --force-reinstall onnxruntime-gpu==1.23.2
mkdir -p "$ROOT/runs/data-quality"
"$VENV/bin/python" -m pip freeze > "$ROOT/runs/data-quality/supertonic3-env.freeze.local.txt"
