#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TP_SIZE=3
export CONTAINER_NAME="${CONTAINER_NAME:-vllm_ds41_flash_tp3}"
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6442450944}"
export DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE="${DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE:-2}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
engram_default='{"table_memory":"disk","disk_resident_scales":false}'
image_limit_default='{"image":0}'
export ENGRAM_CONFIG="${ENGRAM_CONFIG:-${engram_default}}"
export LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-${image_limit_default}}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export CONTAINER_MEMORY_GB="${CONTAINER_MEMORY_GB:-118}"
export CONTAINER_MEMORY_SWAP_GB="${CONTAINER_MEMORY_SWAP_GB:-122}"
export B12X_WEIGHTS_COMPILE_WORKERS="${B12X_WEIGHTS_COMPILE_WORKERS:-8}"
export B12X_STATE_COMPILE_WORKERS="${B12X_STATE_COMPILE_WORKERS:-4}"
export B12X_BIND_COMPILE_WORKERS="${B12X_BIND_COMPILE_WORKERS:-2}"

exec "${SCRIPT_DIR}/serve-ds41-flash-dspark-tp4-rdma.sh" "$@"
