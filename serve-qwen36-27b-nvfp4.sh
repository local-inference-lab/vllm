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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export LLM_WORKER_MULTIPROC_METHOD="${LLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"

export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"

model_path=${MODEL_PATH:-/data/models/Qwen3.6-27B-NVFP4}
served_model_name=${SERVED_MODEL_NAME:-Qwen3.6-27B-NVFP4}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp_size=${TP_SIZE:-1}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_seqs=${MAX_NUM_SEQS:-8}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
load_format=${LOAD_FORMAT:-fastsafetensors}
linear_backend=${LINEAR_BACKEND:-b12x}
attention_backend=${ATTENTION_BACKEND:-flashinfer}

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${model_path}" \
  --served-model-name "${served_model_name}" \
  --host "${host}" \
  --port "${port}" \
  --trust-remote-code \
  --tensor-parallel-size "${tp_size}" \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --load-format "${load_format}" \
  --quantization modelopt_mixed \
  --linear-backend "${linear_backend}" \
  --attention-backend "${attention_backend}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max-cudagraph-capture-size "${max_cudagraph_capture_size}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --enable-flashinfer-autotune \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  "$@"
