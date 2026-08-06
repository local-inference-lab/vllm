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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

MODEL_PATH="${MODEL_PATH:-RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-128}"

MAX_IMAGES="${MAX_IMAGES:-4}"
MAX_VIDEOS="${MAX_VIDEOS:-1}"
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"min_pixels\":784,\"max_pixels\":1003520,\"fps\":1}}"

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
LINEAR_BACKEND="${LINEAR_BACKEND:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"

optional_args=()
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  optional_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
if [[ -n "${LINEAR_BACKEND}" ]]; then
  optional_args+=(--linear-backend "${LINEAR_BACKEND}")
fi
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  optional_args+=(--attention-backend "${ATTENTION_BACKEND}")
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --limit-mm-per-prompt.image "${MAX_IMAGES}" \
  --limit-mm-per-prompt.video "${MAX_VIDEOS}" \
  --mm-processor-kwargs "${MM_PROCESSOR_KWARGS}" \
  "${optional_args[@]}" \
  "$@"
