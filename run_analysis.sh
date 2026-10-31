#!/usr/bin/env bash
# VM: nohup bash run_analysis.sh runs/random-lora-s42 full-v1 > /data/repeat_sft/logs/full-v1.log 2>&1 < /dev/null &
set -Eeuo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash run_analysis.sh [RUN] [JOB_NAME] [EPOCHS]
  RUN       Training run; default runs/random-lora-s42 (relative to REPEAT_SFT_ROOT).
  JOB_NAME  Output/resume name; default full-v1. Reuse it to continue this job.
  EPOCHS    Comma-separated checkpoints; default the run's configured milestones.
Examples:
  bash run_analysis.sh runs/random-lora-s42 full-v1
  bash run_analysis.sh runs/random-lora-s42 early-v1 0,1,2,4
Environment: REPEAT_SFT_ROOT (default /data/repeat_sft), PYTHON_BIN (optional).
Runs loss evaluation -> generation -> grading -> analysis. No training.
HELP
    exit 0
fi
[[ $# -le 3 ]] || { echo 'Too many arguments; use --help' >&2; exit 2; }

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
launcher_path="$project_dir/$(basename -- "${BASH_SOURCE[0]}")"
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

run_arg=${1:-runs/random-lora-s42}
job_name=${2:-full-v1}
epoch_arg=${3:-}

# Freeze the inputs and Python source once. Repeated launches keep the same source,
# so a later git pull does not invalidate partially completed evaluation identities.
job_dir=$("$python_bin" - "$run_arg" "$job_name" "$epoch_arg" "$launcher_path" <<'PY'
from pathlib import Path
import re
import shutil
import sys
from recipe.common import (checkpoint_path, file_hash, fresh_dir, now, read_json,
                           snapshot_code, write_json)
from recipe.paths import artifact_path, configure_runtime
from recipe.resume import verified_code, verify_run_inputs

configure_runtime()
run = artifact_path(sys.argv[1])
name, requested = sys.argv[2:4]
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
    raise ValueError('JOB_NAME must start with a letter/digit and contain only letters, digits, _, - or .')
metadata = verify_run_inputs(run)
job = run / 'postprocess' / name
manifest_path = job / 'pipeline.json'
if manifest_path.exists():
    manifest = read_json(manifest_path)
    if manifest['run_sha256'] != file_hash(run / 'run.json') or Path(manifest['run']) != run:
        raise ValueError('This postprocessing job belongs to different run inputs')
    epochs = manifest['epochs']
    if requested and sorted(set(map(int, requested.split(',')))) != epochs:
        raise ValueError('Epochs differ from this job; use a new JOB_NAME')
    verified_code(job / 'code', manifest['code_sha256'])
else:
    epochs = sorted(set(map(int, requested.split(',')))) if requested else sorted(set(metadata['config']['checkpoint_epochs']))
    if not epochs or any(e < 0 for e in epochs):
        raise ValueError('Expected nonnegative checkpoint epochs')
for epoch in epochs:
    checkpoint = checkpoint_path(run, epoch)
    marker = read_json(checkpoint / 'complete.json')
    if marker['config_sha256'] != metadata['config_sha256'] or marker['step'] != epoch * metadata['config']['expected_train_size']:
        raise ValueError(f'Checkpoint does not match this run: {checkpoint}')
if not manifest_path.exists():
    fresh_dir(job)
    code_hash = snapshot_code(job / 'code')
    shutil.copy2(sys.argv[4], job / 'launcher.sh')
    write_json(manifest_path, {'created_at': now(), 'run': str(run), 'epochs': epochs,
               'run_sha256': file_hash(run / 'run.json'), 'code_sha256': code_hash,
               'launcher_sha256': file_hash(job / 'launcher.sh')})
print(job)
PY
)
run_dir=$(dirname -- "$(dirname -- "$job_dir")")
code_dir="$job_dir/code"
epoch_csv=$("$python_bin" - "$job_dir/pipeline.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding='utf-8') as f:
    print(','.join(map(str, json.load(f)['epochs'])))
PY
)
IFS=, read -r -a epoch_list <<< "$epoch_csv"
mkdir -p -- "$job_dir/state"

# Prevent two launchers from measuring/generating the same job concurrently.
# flock releases this lock even if the process is killed; no stale lock cleanup.
command -v flock >/dev/null || { echo 'flock is required (normally provided by util-linux)' >&2; exit 1; }
exec 9>"$job_dir/pipeline.lock"
flock -n 9 || { echo "This job is already running: $job_dir" >&2; exit 1; }

child_pid=''
stage='startup'
stop_job() {
    local exit_code=$1
    trap '' TERM INT
    if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
    fi
    echo "Stopped during $stage. Repeat the same command to continue."
    exit "$exit_code"
}
trap 'stop_job 143' TERM
trap 'stop_job 130' INT
trap 'result=$?; if (( result != 0 )); then echo "Pipeline stopped (exit=$result), stage=$stage; output=$job_dir" >&2; fi' EXIT

run_stage() {
    stage=$1
    shift
    if [[ -f "$job_dir/state/$stage.done" ]]; then
        echo "[$(date -u +%FT%TZ)] SKIP $stage (completed)"
        return
    fi
    echo "[$(date -u +%FT%TZ)] START $stage"
    "$@" &
    child_pid=$!
    local result=0
    wait "$child_pid" || result=$?
    child_pid=''
    if (( result != 0 )); then
        return "$result"
    fi
    date -u +%FT%TZ > "$job_dir/state/$stage.done"
    echo "[$(date -u +%FT%TZ)] DONE $stage"
}

echo "PID=$$"
echo "Run: $run_dir"
echo "Job: $job_dir"
echo "Epochs: $epoch_csv"
echo "Python: $python_bin"
echo 'Phase 1/4: train/holdout teacher-forcing loss and token accuracy'
for epoch in "${epoch_list[@]}"; do
    run_stage "loss-$epoch" "$python_bin" -B -u "$code_dir/evaluate_loss.py" \
        --run "$run_dir" --epochs "$epoch" --out "$job_dir/teacher_forcing"
done

echo 'Phase 2/4: fixed MATH500/GPQA generation'
for epoch in "${epoch_list[@]}"; do
    run_stage "generation-$epoch" "$python_bin" -B -u "$code_dir/generate.py" \
        --run "$run_dir" --epochs "$epoch" --out "$job_dir/generation" \
        --config "$code_dir/configs/generation.json"
done

echo 'Phase 3/4: grade saved responses'
if [[ ! -f "$job_dir/state/grading.done" ]]; then
    grades_dir="$job_dir/grades"
    if [[ -e "$grades_dir" ]]; then grades_dir="$job_dir/grades-retry-$(date -u +%Y%m%dT%H%M%S)-$$"; fi
    printf '%s\n' "$grades_dir" > "$job_dir/grades_path.txt"
fi
grades_dir=$(<"$job_dir/grades_path.txt")
run_stage grading "$python_bin" -B -u "$code_dir/grade.py" \
    --responses "$job_dir/generation" --out "$grades_dir"

echo 'Phase 4/4: saturation, transfer, behavior and per-question evidence'
if [[ ! -f "$job_dir/state/analysis.done" ]]; then
    analysis_dir="$job_dir/analysis"
    if [[ -e "$analysis_dir" ]]; then analysis_dir="$job_dir/analysis-retry-$(date -u +%Y%m%dT%H%M%S)-$$"; fi
    printf '%s\n' "$analysis_dir" > "$job_dir/analysis_path.txt"
fi
analysis_dir=$(<"$job_dir/analysis_path.txt")
run_stage analysis "$python_bin" -B -u "$code_dir/analyze_run.py" \
    --run "$run_dir" --epochs "$epoch_csv" --teacher-forcing "$job_dir/teacher_forcing" \
    --grades "$grades_dir" --out "$analysis_dir"
stage=complete
echo "[$(date -u +%FT%TZ)] ALL DONE"
echo "Report: $analysis_dir/report.html"
