#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4.1-Flash}"
export MODEL_REVISION="${MODEL_REVISION:-fb2764a5cf321eaa5070ca8f9e892818f477c16d}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm_ds41_flash_tp4}"
export IMAGE_NAME="${IMAGE_NAME:-vllm-node-eugr-20260712-io-uring:latest}"
export TOKENIZER_MODE="${TOKENIZER_MODE:-deepseek_v41}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
engram_default='{"table_memory":"disk","disk_resident_scales":true}'
image_limit_default='{"image":2}'
export ENGRAM_CONFIG="${ENGRAM_CONFIG:-${engram_default}}"
export LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-${image_limit_default}}"
export SECCOMP_PROFILE="${SECCOMP_PROFILE:-${SCRIPT_DIR}/seccomp/spark-io-uring.json}"

exec "${SCRIPT_DIR}/serve-ds4-flash-dspark-tp4-rdma.sh" "$@"
