#!/usr/bin/env bash
# Five topics x ten questions: prioritize selected epochs, then fill the gaps.
set -Eeuo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec bash "$project_dir/dataset/run_curated_benchmarks.sh" \
  --run runs/topic-balanced-50x16-s42/train \
  --panel-dir artifacts/transfer-s2026 \
  --epochs 1,2,3,4,8,12,16,5,6,7,9,10,11,13,14,15 \
  --epoch-order listed \
  --job-name four-benchmarks-priority-vllm-v1 \
  "$@"
