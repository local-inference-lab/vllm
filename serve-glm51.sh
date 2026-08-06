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
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL,LL128,Simple
export OMP_NUM_THREADS=16
export LLM_WORKER_MULTIPROC_METHOD=spawn


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

MODEL="${MODEL:-/models/GLM-5.1-NVFP4-MTP-NVFP4}"
MTP_MODEL="${MTP_MODEL:-${MODEL}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DCP_SIZE="${DCP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_CUDAGRAPH_CAPTURE_SIZE=16

MOE_BACKEND="${MOE_BACKEND:-b12x}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GLM51_PROFILE="${GLM51_PROFILE:-0}"

DEFAULT_HF_OVERRIDES='{"index_topk_pattern":"FFSFSSSFSSFFFSSSFFFSFSSSSSSFFSFFSFFSSFFFFFFSFFFFFSFFSSSSSSFSFFFSFSSSFSFFSFFSSS"}'
HF_OVERRIDES="${HF_OVERRIDES:-$DEFAULT_HF_OVERRIDES}"

SPEC_ARGS=()
if [[ "${GLM51_DISABLE_MTP:-0}" != "1" ]]; then
  NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
  SPEC_CONFIG="${SPEC_CONFIG:-$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s","draft_sample_method":"probabilistic"}' "${MTP_MODEL}" "${NUM_SPECULATIVE_TOKENS}" "${MOE_BACKEND}")}"
  SPEC_ARGS+=(--speculative-config "${SPEC_CONFIG}")
fi

PROFILER_ARGS=()
case "${GLM51_PROFILE,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on|torch)
    GLM51_PROFILE_DIR="${GLM51_PROFILE_DIR:-/tmp/vllm-profile/glm51-$(date +%Y%m%d-%H%M%S)}"
    GLM51_TORCH_PROFILER_WITH_STACK_JSON="$(json_bool GLM51_TORCH_PROFILER_WITH_STACK "${GLM51_TORCH_PROFILER_WITH_STACK:-0}")"
    GLM51_TORCH_PROFILER_RECORD_SHAPES_JSON="$(json_bool GLM51_TORCH_PROFILER_RECORD_SHAPES "${GLM51_TORCH_PROFILER_RECORD_SHAPES:-0}")"
    GLM51_TORCH_PROFILER_WITH_MEMORY_JSON="$(json_bool GLM51_TORCH_PROFILER_WITH_MEMORY "${GLM51_TORCH_PROFILER_WITH_MEMORY:-0}")"
    GLM51_TORCH_PROFILER_WITH_FLOPS_JSON="$(json_bool GLM51_TORCH_PROFILER_WITH_FLOPS "${GLM51_TORCH_PROFILER_WITH_FLOPS:-0}")"
    GLM51_TORCH_PROFILER_USE_GZIP_JSON="$(json_bool GLM51_TORCH_PROFILER_USE_GZIP "${GLM51_TORCH_PROFILER_USE_GZIP:-1}")"
    GLM51_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON="$(json_bool GLM51_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL "${GLM51_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL:-0}")"
    GLM51_PROFILE_IGNORE_FRONTEND_JSON="$(json_bool GLM51_PROFILE_IGNORE_FRONTEND "${GLM51_PROFILE_IGNORE_FRONTEND:-1}")"

    if [[ "${GLM51_PROFILE_DIR}" != *"://"* ]]; then
      mkdir -p "${GLM51_PROFILE_DIR}"
    fi
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
    PROFILER_ARGS+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${GLM51_PROFILE_DIR}"
      --profiler-config.torch_profiler_with_stack="${GLM51_TORCH_PROFILER_WITH_STACK_JSON}"
      --profiler-config.torch_profiler_record_shapes="${GLM51_TORCH_PROFILER_RECORD_SHAPES_JSON}"
      --profiler-config.torch_profiler_with_memory="${GLM51_TORCH_PROFILER_WITH_MEMORY_JSON}"
      --profiler-config.torch_profiler_with_flops="${GLM51_TORCH_PROFILER_WITH_FLOPS_JSON}"
      --profiler-config.torch_profiler_use_gzip="${GLM51_TORCH_PROFILER_USE_GZIP_JSON}"
      --profiler-config.torch_profiler_dump_cuda_time_total="${GLM51_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON}"
      --profiler-config.ignore_frontend="${GLM51_PROFILE_IGNORE_FRONTEND_JSON}"
      --profiler-config.delay_iterations="${GLM51_PROFILE_DELAY_ITERATIONS:-0}"
      --profiler-config.max_iterations="${GLM51_PROFILE_MAX_ITERATIONS:-4}"
      --profiler-config.warmup_iterations="${GLM51_PROFILE_WARMUP_ITERATIONS:-0}"
      --profiler-config.active_iterations="${GLM51_PROFILE_ACTIVE_ITERATIONS:-5}"
      --profiler-config.wait_iterations="${GLM51_PROFILE_WAIT_ITERATIONS:-0}"
    )
    echo "Torch profiling enabled. Traces will be written under: ${GLM51_PROFILE_DIR}" >&2
    ;;
  cuda|nsys|nsight)
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    PROFILER_ARGS+=(--profiler-config.profiler=cuda)
    echo "CUDA profiler enabled. Use nsys with --capture-range=cudaProfilerApi and drive /start_profile + /stop_profile." >&2
    ;;
  *)
    echo "ERROR: GLM51_PROFILE must be one of 1/0, true/false, torch, cuda, nsys, or nsight; got '${GLM51_PROFILE}'" >&2
    exit 1
    ;;
esac

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
  -cc.pass_config.fuse_allreduce_rms=True \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --quantization modelopt_fp4 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --moe-backend "${MOE_BACKEND}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  "${SPEC_ARGS[@]}" \
  "${PROFILER_ARGS[@]}" \
  --hf-overrides "${HF_OVERRIDES}" \
  "$@"
