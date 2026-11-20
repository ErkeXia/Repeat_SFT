#!/usr/bin/env bash
# High-five experiment defaults; use the existing epoch-first four-task pipeline.
set -Eeuo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
exec bash "$project_dir/dataset/run_curated_benchmarks.sh" \
  --run runs/curated-high5-64-s42/train \
  --panel-dir artifacts/transfer-s2026 \
  --epochs 64,48,32,16 \
  --job-name four-benchmarks-e16plus-vllm-v1 \
  "$@"
