#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/opt/cuda}}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${SCRIPT_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export LLM_WORKER_MULTIPROC_METHOD="${LLM_WORKER_MULTIPROC_METHOD:-spawn}"

export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export B12X_MOE_FORCE_A8="${B12X_MOE_FORCE_A8:-0}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-0}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"

MODEL_PATH="${MODEL_PATH:-poolside/Laguna-S-2.1-NVFP4}"
MODEL_REVISION="${MODEL_REVISION_OVERRIDE:-RC1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-laguna-s-2.1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASHINFER}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-FLASHINFER}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${MODEL_PATH}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP_SIZE}" \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --load-format "${LOAD_FORMAT}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --moe-backend "${MOE_BACKEND}" \
  --linear-backend "${LINEAR_BACKEND}" \
  --speculative-config "{\"model\":\"poolside/Laguna-S-2.1-DFlash-NVFP4\",\"num_speculative_tokens\":7,\"method\":\"dflash\",\"attention_backend\":\"${DRAFT_ATTENTION_BACKEND}\"}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --reasoning-parser poolside_v1 \
  --tool-call-parser poolside_v1 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --override-generation-config '{"temperature":0.7,"top_p":0.95}' \
  "$@"
