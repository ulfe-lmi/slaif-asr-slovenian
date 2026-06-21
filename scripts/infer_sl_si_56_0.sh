#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${repo_root}/.venv/bin/python" "${repo_root}/scripts/run_streaming_inference.py" --context "[56,0]" "$@"
