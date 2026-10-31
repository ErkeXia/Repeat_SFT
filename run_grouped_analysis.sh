#!/usr/bin/env bash
# VM: nohup bash run_grouped_analysis.sh runs/random-lora-s42 grouped-8192-v1 > /data/repeat_sft/logs/grouped-8192.log 2>&1 < /dev/null &
set -Eeuo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$project_dir"
if [[ -n ${PYTHON_BIN:-} ]]; then
    python_bin=$PYTHON_BIN
elif [[ -x "$project_dir/.venv/bin/python" ]]; then
    python_bin="$project_dir/.venv/bin/python"
else
    python_bin=python3
fi
export REPEAT_SFT_ROOT=${REPEAT_SFT_ROOT:-/data/repeat_sft}
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "$python_bin" -B -u "$project_dir/grouped_evaluation.py" "$@"
