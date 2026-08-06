#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export B12X_MOE_FORCE_A8="${B12X_MOE_FORCE_A8:-0}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-0}"

MODEL_PATH="${MODEL_PATH:-nvidia/Qwen3.5-397B-A17B-NVFP4-V2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-397B-A17B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_ATTN}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --load-format "${LOAD_FORMAT}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --moe-backend "${MOE_BACKEND}" \
  --linear-backend "${LINEAR_BACKEND}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
