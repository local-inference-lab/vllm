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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7,8,9}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=16


export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${GLM51_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"

export B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO:-1}"
export B12X_W4A16_TC_DECODE="${B12X_W4A16_TC_DECODE:-1}"

json_bool() {
  local name="$1"
  local value="$2"

  case "${value,,}" in
    1|true|yes|on)
      echo true
      ;;
    0|false|no|off)
      echo false
      ;;
    *)
      echo "ERROR: ${name} must be one of 1/0, true/false, yes/no, on/off; got '${value}'" >&2
      exit 1
      ;;
  esac
}

MODEL="moonshotai/Kimi-K2.6"
MTP_MODEL="${MTP_MODEL:-${MODEL}}"
SERVED_MODEL_NAME="Kimi-K2.6"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DCP_SIZE="${DCP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_CUDAGRAPH_CAPTURE_SIZE=128

MOE_BACKEND="${MOE_BACKEND:-b12x}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"

DEFAULT_HF_OVERRIDES='{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
HF_OVERRIDES="${HF_OVERRIDES:-$DEFAULT_HF_OVERRIDES}"

SPEC_ARGS=()
if [[ "${GLM51_DISABLE_MTP:-0}" != "1" ]]; then
  NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
  SPEC_CONFIG='{"model":"lightseekorg/kimi-k2.6-eagle3-mla","method":"eagle3","num_speculative_tokens":3}'
  SPEC_ARGS+=(--speculative-config "${SPEC_CONFIG}")
fi

PROFILER_ARGS=()

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend "${DCP_COMM_BACKEND:-ag_rs}" \
  --dcp-kv-cache-interleave-size "${DCP_KV_CACHE_INTERLEAVE_SIZE:-1}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  -cc.pass_config.fuse_allreduce_rms=True \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --mm-encoder-tp-mode data \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --tool-call-parser kimi_k2 \
  --enable-auto-tool-choice \
  --reasoning-parser kimi_k2 \
  --speculative-config $SPEC_CONFIG \
  --hf-overrides "${HF_OVERRIDES}" \
  "$@"
