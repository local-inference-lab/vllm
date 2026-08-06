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

expand_user_path() {
  local path="$1"
  case "${path}" in
    "~")
      printf '%s\n' "${HOME}"
      ;;
    "~/"*)
      printf '%s\n' "${HOME}/${path#"~/"}"
      ;;
    *)
      printf '%s\n' "${path}"
      ;;
  esac
}

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

MODEL="${MODEL:-${HOME}/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.5-Pro-FP4-DFlash/snapshots/b754e6c86008bdb5cc901308dda5a38173ec7276}"
MODEL="$(expand_user_path "${MODEL}")"
MODEL="${MODEL%/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Mimo-2.5-Pro}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-256}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-TRITON_ATTN}"
DRAFT_ATTENTION_BACKEND="${DRAFT_ATTENTION_BACKEND:-TRITON_ATTN}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\",\"custom_ops\":[\"all\"]}}"
ENABLE_SPECULATION="${ENABLE_SPECULATION:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
DRAFT_MODEL="${DRAFT_MODEL:-${MODEL}/dflash}"
DRAFT_MODEL="$(expand_user_path "${DRAFT_MODEL}")"
DRAFT_MODEL="${DRAFT_MODEL%/}"
SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-{\"model\":\"${DRAFT_MODEL}\",\"method\":\"dflash\",\"num_speculative_tokens\":${NUM_SPECULATIVE_TOKENS},\"attention_backend\":\"${DRAFT_ATTENTION_BACKEND}\"}}"
MIMO25_PROFILE="${MIMO25_PROFILE:-${VLLM_ENABLE_TORCH_PROFILER:-0}}"

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${GLM51_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"


unset \
  NCCL_GRAPH_FILE \
  NCCL_GRAPH_DUMP_FILE \
  VLLM_B12X_MLA_EXTEND_MAX_CHUNKS \
  VLLM_ENABLE_PCIE_ALLREDUCE \
  VLLM_PCIE_ALLREDUCE_BACKEND \
  VLLM_CPP_AR_1STAGE_NCCL_CUTOFF \
  VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS

SPECULATIVE_ARGS=()
case "${ENABLE_SPECULATION,,}" in
  0 | false | no | off | "")
    ;;
  *)
    SPECULATIVE_ARGS=(--speculative-config "${SPECULATIVE_CONFIG}")
    ;;
esac

PROFILER_ARGS=()
case "${MIMO25_PROFILE,,}" in
  0 | false | no | off | "")
    ;;
  1 | true | yes | on | torch)
    MIMO25_PROFILE_DIR="${MIMO25_PROFILE_DIR:-/tmp/vllm-profile/mimo25-$(date +%Y%m%d-%H%M%S)}"
    MIMO25_TORCH_PROFILER_WITH_STACK_JSON="$(json_bool MIMO25_TORCH_PROFILER_WITH_STACK "${MIMO25_TORCH_PROFILER_WITH_STACK:-1}")"
    MIMO25_TORCH_PROFILER_RECORD_SHAPES_JSON="$(json_bool MIMO25_TORCH_PROFILER_RECORD_SHAPES "${MIMO25_TORCH_PROFILER_RECORD_SHAPES:-0}")"
    MIMO25_TORCH_PROFILER_WITH_MEMORY_JSON="$(json_bool MIMO25_TORCH_PROFILER_WITH_MEMORY "${MIMO25_TORCH_PROFILER_WITH_MEMORY:-0}")"
    MIMO25_TORCH_PROFILER_WITH_FLOPS_JSON="$(json_bool MIMO25_TORCH_PROFILER_WITH_FLOPS "${MIMO25_TORCH_PROFILER_WITH_FLOPS:-0}")"
    MIMO25_TORCH_PROFILER_USE_GZIP_JSON="$(json_bool MIMO25_TORCH_PROFILER_USE_GZIP "${MIMO25_TORCH_PROFILER_USE_GZIP:-1}")"
    MIMO25_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON="$(json_bool MIMO25_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL "${MIMO25_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL:-0}")"
    MIMO25_PROFILE_IGNORE_FRONTEND_JSON="$(json_bool MIMO25_PROFILE_IGNORE_FRONTEND "${MIMO25_PROFILE_IGNORE_FRONTEND:-1}")"

    if [[ "${MIMO25_PROFILE_DIR}" != *"://"* ]]; then
      mkdir -p "${MIMO25_PROFILE_DIR}"
    fi
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
    PROFILER_ARGS+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${MIMO25_PROFILE_DIR}"
      --profiler-config.torch_profiler_with_stack="${MIMO25_TORCH_PROFILER_WITH_STACK_JSON}"
      --profiler-config.torch_profiler_record_shapes="${MIMO25_TORCH_PROFILER_RECORD_SHAPES_JSON}"
      --profiler-config.torch_profiler_with_memory="${MIMO25_TORCH_PROFILER_WITH_MEMORY_JSON}"
      --profiler-config.torch_profiler_with_flops="${MIMO25_TORCH_PROFILER_WITH_FLOPS_JSON}"
      --profiler-config.torch_profiler_use_gzip="${MIMO25_TORCH_PROFILER_USE_GZIP_JSON}"
      --profiler-config.torch_profiler_dump_cuda_time_total="${MIMO25_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON}"
      --profiler-config.ignore_frontend="${MIMO25_PROFILE_IGNORE_FRONTEND_JSON}"
      --profiler-config.delay_iterations="${MIMO25_PROFILE_DELAY_ITERATIONS:-0}"
      --profiler-config.max_iterations="${MIMO25_PROFILE_MAX_ITERATIONS:-4}"
      --profiler-config.warmup_iterations="${MIMO25_PROFILE_WARMUP_ITERATIONS:-0}"
      --profiler-config.active_iterations="${MIMO25_PROFILE_ACTIVE_ITERATIONS:-5}"
      --profiler-config.wait_iterations="${MIMO25_PROFILE_WAIT_ITERATIONS:-0}"
    )
    echo "Torch profiling enabled. Traces will be written under: ${MIMO25_PROFILE_DIR}" >&2
    ;;
  cuda | nsys | nsight)
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    PROFILER_ARGS+=(--profiler-config.profiler=cuda)
    echo "CUDA profiler enabled. Use nsys with --capture-range=cudaProfilerApi and drive /start_profile + /stop_profile." >&2
    ;;
  *)
    echo "ERROR: MIMO25_PROFILE must be one of 1/0, true/false, torch, cuda, nsys, or nsight; got '${MIMO25_PROFILE}'" >&2
    exit 1
    ;;
esac

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --block-size "${BLOCK_SIZE}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --kernel-config.moe_backend "${MOE_BACKEND}" \
  --kernel-config.linear_backend "${LINEAR_BACKEND}" \
  --reasoning-parser mimo \
  --enable-auto-tool-choice \
  --tool-call-parser mimo \
  "${SPECULATIVE_ARGS[@]}" \
  "${PROFILER_ARGS[@]}" \
  --compilation-config "${COMPILATION_CONFIG}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  "$@"
