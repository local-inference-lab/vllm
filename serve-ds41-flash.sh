#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4.1-Flash setup: TP4 on GPUs 0-3, SSD Engram, DSpark.
# Keep the four GPUs otherwise idle; the default memory budget is set below.
# Preview without loading the model: DRY_RUN=1 ./serve-ds41-flash.sh
# HOST, PORT, MODEL_PATH and the capacity variables below may be overridden.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python interpreter not found or not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export CUTE_DSL_ARCH=sm_120a
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
# Use the qualified TP4 collective policy, not earlier TP8 tuning overrides.
unset VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE
# Callable-worker tracing used this flag; a serving endpoint must not inherit it.
unset VLLM_ALLOW_INSECURE_SERIALIZATION

speculative_config='{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":4,"attention_backend":"B12X_MLA_SPARSE_DSV41","draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":true}'
command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}" --port "${PORT}"
  --dtype bfloat16
  --tensor-parallel-size 4
  --pipeline-parallel-size 1
  --enable-expert-parallel
  --load-format safetensors
  --safetensors-load-strategy lazy
  --block-size 256
  --enable-prefix-caching
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --generation-config vllm
  --limit-mm-per-prompt '{"image":2}'
  --engram-config '{"cpu_offload":false,"table_memory":"disk"}'
  --linear-backend b12x
  --moe-backend b12x
  --speculative-config "${speculative_config}"
  --jit-monitor-mode error
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  "$@"
)

cd "${SCRIPT_DIR}"
printf 'Launching %s: TP4, GPUs %s, SSD Engram, DSpark (5 draft tokens)\n' \
  "${SERVED_MODEL_NAME}" "${CUDA_VISIBLE_DEVICES}" >&2
printf 'Endpoint: http://%s:%s/v1  |  GPU memory budget: %s\n' \
  "${HOST}" "${PORT}" "${GPU_MEMORY_UTILIZATION}" >&2
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
