#!/usr/bin/env bash
set -Eeuo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$project_dir"
export REPEAT_SFT_ROOT=${REPEAT_SFT_ROOT:-/data/repeat_sft}
python_bin=${VLLM_PYTHON_BIN:-"$REPEAT_SFT_ROOT/venvs/vllm-0.11.0/bin/python"}
if [[ ! -x "$python_bin" ]]; then
    echo "Missing vLLM environment: $python_bin; follow VLLM_EVALUATION.md" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" -B -u "$project_dir/transfer_evaluation.py" "$@"
