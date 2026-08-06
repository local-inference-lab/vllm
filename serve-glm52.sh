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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
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
export VLLM_B12X_ABSORB_BMM="${VLLM_B12X_ABSORB_BMM:-0}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${GLM51_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-84KB}"

export B12X_MOE_FORCE_A8="${B12X_MOE_FORCE_A8:-0}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-0}"

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

VLLM_EXTRA_ARGS=()
while (($#)); do
  case "$1" in
    --causal-cascade)
      GLM52_CAUSAL_CASCADE=1
      shift
      ;;
    --no-causal-cascade)
      GLM52_CAUSAL_CASCADE=0
      shift
      ;;
    --no-mtp)
      GLM51_DISABLE_MTP=1
      shift
      ;;
    --dspark)
      GLM52_DSPARK=1
      shift
      ;;
    --no-dspark)
      GLM52_DSPARK=0
      shift
      ;;
    *)
      VLLM_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${VLLM_EXTRA_ARGS[@]}"

MODEL="${MODEL:-lukealonso/GLM-5.2-NVFP4}"
MTP_MODEL="${MTP_MODEL:-${MODEL}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DCP_SIZE="${DCP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"

MOE_BACKEND="${MOE_BACKEND:-b12x}"
MOE_SPEC_BACKEND="${MOE_SPEC_BACKEND:-b12x}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X_MLA_SPARSE}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GLM51_PROFILE="${GLM51_PROFILE:-0}"
GLM52_CAUSAL_CASCADE="${GLM52_CAUSAL_CASCADE:-0}"
GLM52_DSPARK="${GLM52_DSPARK:-0}"
DSPARK_MODEL="${DSPARK_MODEL:-RedHatAI/GLM-5.2-speculator.dspark}"
ADAPTIVE_SPEC_WINDOW="${GLM52_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW:-}"

if [[ -n "${ADAPTIVE_SPEC_WINDOW}" \
  && ! "${ADAPTIVE_SPEC_WINDOW}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GLM52_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW must be a positive" \
    "integer; got '${ADAPTIVE_SPEC_WINDOW}'" >&2
  exit 1
fi

DEFAULT_CAUSAL_CASCADE_MODEL="/data/datasets/glmflash/runs/train/mtp-v10-clean-p1r256-dualstream2-markov-tv-20260701-161620-prefixgrad01-bufferfix-fromstep5000/checkpoints/latest"
CAUSAL_CASCADE_MODEL="${CAUSAL_CASCADE_MODEL:-${DEFAULT_CAUSAL_CASCADE_MODEL}}"
CAUSAL_CASCADE_BLOCK_SIZE="${CAUSAL_CASCADE_BLOCK_SIZE:-9}"
# Slot 0 is the verifier bonus/known token; slots 1 through 8 are drafts.
export CAUSAL_CASCADE_FIRST_DRAFT_SLOT="${CAUSAL_CASCADE_FIRST_DRAFT_SLOT:-1}"
export CAUSAL_CASCADE_USE_CAPTURE_POSITIONS="${CAUSAL_CASCADE_USE_CAPTURE_POSITIONS:-0}"
CAUSAL_CASCADE_DRAFT_SAMPLE_METHOD="${CAUSAL_CASCADE_DRAFT_SAMPLE_METHOD:-greedy}"
CAUSAL_CASCADE_DEFAULT_NUM_SPECULATIVE_TOKENS="$((CAUSAL_CASCADE_BLOCK_SIZE - CAUSAL_CASCADE_FIRST_DRAFT_SLOT))"
CAUSAL_CASCADE_NUM_SPECULATIVE_TOKENS="${CAUSAL_CASCADE_NUM_SPECULATIVE_TOKENS:-${CAUSAL_CASCADE_DEFAULT_NUM_SPECULATIVE_TOKENS}}"
export CAUSAL_CASCADE_MIN_CONTEXT_TOKENS="${CAUSAL_CASCADE_MIN_CONTEXT_TOKENS:-128}"
CAUSAL_CASCADE_ENFORCE_EAGER="${CAUSAL_CASCADE_ENFORCE_EAGER:-0}"

GLM52_INDEX_TOPK_PATTERN="FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
DEFAULT_HF_OVERRIDES="$(printf '{"index_topk_pattern":"%s"}' "${GLM52_INDEX_TOPK_PATTERN}")"
HF_OVERRIDES="${HF_OVERRIDES:-$DEFAULT_HF_OVERRIDES}"

SPEC_ARGS=()
CAUSAL_CASCADE_ARGS=()
GLM52_CAUSAL_CASCADE_JSON="$(json_bool GLM52_CAUSAL_CASCADE "${GLM52_CAUSAL_CASCADE}")"
GLM52_DSPARK_JSON="$(json_bool GLM52_DSPARK "${GLM52_DSPARK}")"
if [[ "${GLM52_DSPARK_JSON}" == "true" && "${GLM52_CAUSAL_CASCADE_JSON}" == "true" ]]; then
  echo "ERROR: --dspark and --causal-cascade are mutually exclusive" >&2
  exit 1
fi
if [[ "${GLM52_DSPARK_JSON}" == "true" ]]; then
  if [[ -n "${ADAPTIVE_SPEC_WINDOW}" && -n "${SPEC_CONFIG:-}" ]]; then
    echo "ERROR: GLM52_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW cannot be combined" \
      "with an explicit SPEC_CONFIG" >&2
    exit 1
  fi

  # The speculators-format translation (which would read speculative_tokens
  # from the checkpoint) only runs when the speculator is the served model,
  # not when it is referenced from --speculative-config, so the depth must be
  # explicit here. 7 matches RedHatAI/GLM-5.2-speculator.dspark (block_size 8
  # = 1 bonus anchor + 7 drafts).
  NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
  DSPARK_NUM_SPEC_CONFIG="$(
    printf ',"num_speculative_tokens":%s' "${NUM_SPECULATIVE_TOKENS}"
  )"
  ADAPTIVE_SPEC_CONFIG=""
  if [[ -n "${ADAPTIVE_SPEC_WINDOW}" ]]; then
    ADAPTIVE_SPEC_CONFIG="$(
      printf ',"adaptive_speculative_tokens_window":%s' \
        "${ADAPTIVE_SPEC_WINDOW}"
    )"
    echo "Adaptive speculative depth enabled with a ${ADAPTIVE_SPEC_WINDOW}-step" \
      "window." >&2
  fi
  # The dense DSpark draft must not inherit the target's fp8_ds_mla KV
  # layout; bf16 draft KV keeps FLASH_ATTN eligible for its 5 layers.
  DSPARK_DRAFT_KV_CACHE_DTYPE="${DSPARK_DRAFT_KV_CACHE_DTYPE:-auto}"
  DEFAULT_SPEC_CONFIG="$(
    printf '{"model":"%s","method":"dspark","draft_kv_cache_dtype":"%s"%s%s}' \
      "${DSPARK_MODEL}" \
      "${DSPARK_DRAFT_KV_CACHE_DTYPE}" \
      "${DSPARK_NUM_SPEC_CONFIG}" \
      "${ADAPTIVE_SPEC_CONFIG}"
  )"
  SPEC_CONFIG="${SPEC_CONFIG:-${DEFAULT_SPEC_CONFIG}}"
  SPEC_ARGS+=(--speculative-config "${SPEC_CONFIG}")

  echo "DSpark speculative decoding enabled with ${DSPARK_MODEL}." >&2
elif [[ "${GLM52_CAUSAL_CASCADE_JSON}" == "true" ]]; then
  if [[ -n "${ADAPTIVE_SPEC_WINDOW}" ]]; then
    echo "ERROR: GLM52_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW requires the MTP" \
      "speculative path; disable GLM52_CAUSAL_CASCADE" >&2
    exit 1
  fi

  NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-${CAUSAL_CASCADE_NUM_SPECULATIVE_TOKENS}}"
  SPEC_CONFIG="${SPEC_CONFIG:-$(printf '{"model":"%s","method":"causal_cascade","num_speculative_tokens":%s,"draft_sample_method":"%s"}' "${CAUSAL_CASCADE_MODEL}" "${NUM_SPECULATIVE_TOKENS}" "${CAUSAL_CASCADE_DRAFT_SAMPLE_METHOD}")}"
  SPEC_ARGS+=(--speculative-config "${SPEC_CONFIG}")

  echo "CausalCascade native inference enabled." >&2

  CAUSAL_CASCADE_ENFORCE_EAGER_JSON="$(json_bool CAUSAL_CASCADE_ENFORCE_EAGER "${CAUSAL_CASCADE_ENFORCE_EAGER}")"
  if [[ "${CAUSAL_CASCADE_ENFORCE_EAGER_JSON}" == "true" ]]; then
    CAUSAL_CASCADE_ARGS+=(--enforce-eager)
  fi
elif [[ "${GLM51_DISABLE_MTP:-0}" != "1" ]]; then
  if [[ -n "${ADAPTIVE_SPEC_WINDOW}" && -n "${SPEC_CONFIG:-}" ]]; then
    echo "ERROR: GLM52_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW cannot be combined" \
      "with an explicit SPEC_CONFIG" >&2
    exit 1
  fi

  NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
  ADAPTIVE_SPEC_CONFIG=""
  if [[ -n "${ADAPTIVE_SPEC_WINDOW}" ]]; then
    ADAPTIVE_SPEC_CONFIG="$(
      printf ',"adaptive_speculative_tokens_window":%s' \
        "${ADAPTIVE_SPEC_WINDOW}"
    )"
    echo "Adaptive speculative depth enabled with a ${ADAPTIVE_SPEC_WINDOW}-step" \
      "window and maximum depth ${NUM_SPECULATIVE_TOKENS}." >&2
  fi
  DEFAULT_SPEC_CONFIG="$(
    printf \
      '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s","draft_sample_method":"probabilistic"%s}' \
      "${MTP_MODEL}" \
      "${NUM_SPECULATIVE_TOKENS}" \
      "${MOE_SPEC_BACKEND}" \
      "${ADAPTIVE_SPEC_CONFIG}"
  )"
  SPEC_CONFIG="${SPEC_CONFIG:-${DEFAULT_SPEC_CONFIG}}"
  SPEC_ARGS+=(--speculative-config "${SPEC_CONFIG}")
elif [[ -n "${ADAPTIVE_SPEC_WINDOW}" ]]; then
  echo "ERROR: GLM52_ADAPTIVE_SPECULATIVE_TOKENS_WINDOW requires MTP; remove" \
    "--no-mtp or unset GLM51_DISABLE_MTP" >&2
  exit 1
fi

PROFILER_ARGS=()
case "${GLM51_PROFILE,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on|torch)
    GLM51_PROFILE_DIR="${GLM51_PROFILE_DIR:-/tmp/vllm-profile/glm51-$(date +%Y%m%d-%H%M%S)}"
    GLM51_TORCH_PROFILER_WITH_STACK_JSON="$(json_bool GLM51_TORCH_PROFILER_WITH_STACK "${GLM51_TORCH_PROFILER_WITH_STACK:-1}")"
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
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend "${DCP_COMM_BACKEND:-a2a}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --load-format fastsafetensors \
  --async-scheduling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64]}' \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --long-prefill-token-threshold 2048 \
  --moe-backend "${MOE_BACKEND}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --generation-config "${GENERATION_CONFIG}" \
  --hf-overrides "${HF_OVERRIDES}" \
  "${SPEC_ARGS[@]}" \
  "${PROFILER_ARGS[@]}" \
  "${CAUSAL_CASCADE_ARGS[@]}" \
  "$@"
