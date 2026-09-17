#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}"
export MODEL_REVISION="${MODEL_REVISION:-6821d6ad3681a4b137b066b76094fa82ebd0a380}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-Vision-Exp}"
export CONTAINER_NAME="${CONTAINER_NAME:-vllm_ds4_flash_vision_tp2}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"

exec "${SCRIPT_DIR}/serve-ds4-flash-dspark-tp2-rdma.sh" "$@"
