#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/data/cache/huggingface/hub/models--local-inference-lab--GLM-5.3-Flash-NVFP4/snapshots/3959f8a063b77cfdb22ab2e085a1f76fd38b195b}"
export DSPARK_MODEL="${DSPARK_MODEL:-/data/models/GLM-5.3-Flash-DSpark-step1896}"
export SPECULATOR="${SPECULATOR:-dspark}"
export DSPARK_DEPTH_MODE="${DSPARK_DEPTH_MODE:-adaptive}"
export DEVICE_IDS="${DEVICE_IDS:-0,1,2,3}"
export TP_SIZE="${TP_SIZE:-4}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES-}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-0}"
export VLLM_MXFP8_LM_HEAD="${VLLM_MXFP8_LM_HEAD:-1}"

exec bash "${SCRIPT_DIR}/serve-glm53-flash-nvfp4.sh" \
  --per-request-spec-decode-metrics summary "$@"
