#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6}"
export TP_SIZE="${TP_SIZE:-6}"

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.935}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

exec "${SCRIPT_DIR}/serve-glm51.sh" \
  "$@"
