#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nemo_revision="8044a3924bfcfe8ef71d792bb73bf274fe853575"
nemo_root="${NEMO_ROOT:-${repo_root}/.external/NeMo}"
venv_dir="${SLAIF_VENV:-${repo_root}/.venv}"

mkdir -p "$(dirname "${nemo_root}")"

if [[ ! -d "${nemo_root}/.git" ]]; then
  git clone https://github.com/NVIDIA-NeMo/NeMo.git "${nemo_root}"
fi

git -C "${nemo_root}" fetch --tags origin
git -C "${nemo_root}" checkout "${nemo_revision}"

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip setuptools wheel
"${venv_dir}/bin/python" -m pip install -e "${repo_root}"
"${venv_dir}/bin/python" -m pip install -e "${nemo_root}[asr]"

cat <<EOF
Runtime setup completed.
NEMO_ROOT=${nemo_root}
SLAIF_VENV=${venv_dir}
NeMo revision=${nemo_revision}
EOF
