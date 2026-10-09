#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4.1-Flash setup: TP4 on GPUs 0-3, configurable Engram storage,
# DSpark, and mixed prefill/decode scheduling.
# Keep the visible GPUs otherwise idle; the default memory budget is set below.
# Preview without loading the model: DRY_RUN=1 ./serve-ds41-flash.sh
# HOST, PORT, MODEL_PATH and the capacity variables below may be overridden.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-64}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
ENGRAM_TABLE_MEMORY="${ENGRAM_TABLE_MEMORY:-disk}"
ENGRAM_DISK_RESIDENT_SCALES="${ENGRAM_DISK_RESIDENT_SCALES:-0}"
ENGRAM_PROJECTION_TP="${ENGRAM_PROJECTION_TP:-0}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
SCHEDULER_RESERVE_FULL_ISL="${SCHEDULER_RESERVE_FULL_ISL:-0}"
PREFILL_COMPUTE_SHARE="${PREFILL_COMPUTE_SHARE:-auto}"
PREFILL_COMPUTE_HALF_LIFE="${PREFILL_COMPUTE_HALF_LIFE:-}"
MAX_PARALLEL_PREFILLS="${MAX_PARALLEL_PREFILLS:-auto}"
PREFILL_POLICY="${PREFILL_POLICY:-}"
DECODE_REFILL_TARGET="${DECODE_REFILL_TARGET:-auto}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES:-0}"
TORCH_PROFILE_WITH_MEMORY="${TORCH_PROFILE_WITH_MEMORY:-0}"
TORCH_PROFILE_WITH_STACK="${TORCH_PROFILE_WITH_STACK:-1}"
TORCH_PROFILE_WITH_FLOPS="${TORCH_PROFILE_WITH_FLOPS:-0}"
TORCH_PROFILE_USE_GZIP="${TORCH_PROFILE_USE_GZIP:-1}"
TORCH_PROFILE_DEFAULT_DIR=/tmp/vllm-ds4-decode
TORCH_PROFILE_MAX_ITERATIONS=4
TP_SIZE="${TP_SIZE:-4}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
DSPARK_ADAPTIVE_VERIFICATION="${DSPARK_ADAPTIVE_VERIFICATION:-1}"
DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE="${DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE:-1.0}"

bool_value() {
  local name=$1 value=${2,,}
  case "${value}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *)
      echo "${name} must be 1/0, true/false, yes/no, or on/off; got '${2}'" >&2
      exit 2
      ;;
  esac
}

require_positive_int() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
}

require_positive_number() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
    || ! awk -v value="${value}" 'BEGIN { exit !((value + 0) > 0) }'; then
    echo "${name} must be a positive number; got '${value}'" >&2
    exit 2
  fi
}

usage() {
  printf '%s\n' \
    "Usage: $0 [launcher options] [vLLM options]" \
    "" \
    "Launcher options:" \
    "  --torch-profile [DIR]         Configure a triggered four-step CPU+CUDA capture." \
    "                                DIR defaults to /tmp/vllm-ds4-decode." \
    "  --torch-profile-record-shapes Record tensor shapes." \
    "  --torch-profile-with-memory   Record tensor memory activity." \
    "  --torch-profile-with-flops    Estimate supported operator FLOPs." \
    "  --torch-profile-no-stack      Disable Python stack capture." \
    "  --torch-profile-no-gzip       Write uncompressed trace files." \
    "  -h, --help                    Show this help." \
    "" \
    "Profiling starts only when triggered; enabling it does not start a capture." \
    "All other arguments are forwarded to vLLM. Equivalent environment" \
    "variables use the TORCH_PROFILE_* names declared at the top of the script." \
    "Mixed serving defaults to adaptive prefill compute sharing, decode-aware" \
    "parallel prefill selection, chunked prefill, and async scheduling." \
    "DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE controls verification trimming" \
    "aggressiveness (default: 1.0; larger values trim more)."
}

vllm_args=()
while (($#)); do
  case "$1" in
    --torch-profile)
      if (($# >= 2)) && [[ "$2" != -* ]]; then
        TORCH_PROFILE_DIR=$2
        shift 2
      else
        TORCH_PROFILE_DIR=${TORCH_PROFILE_DIR:-${TORCH_PROFILE_DEFAULT_DIR}}
        shift
      fi
      ;;
    --torch-profile=*)
      TORCH_PROFILE_DIR=${1#*=}
      if [[ -z "${TORCH_PROFILE_DIR}" ]]; then
        echo "--torch-profile requires a non-empty output directory" >&2
        exit 2
      fi
      shift
      ;;
    --torch-profile-record-shapes)
      TORCH_PROFILE_RECORD_SHAPES=1
      shift
      ;;
    --torch-profile-with-memory)
      TORCH_PROFILE_WITH_MEMORY=1
      shift
      ;;
    --torch-profile-with-flops)
      TORCH_PROFILE_WITH_FLOPS=1
      shift
      ;;
    --torch-profile-no-stack)
      TORCH_PROFILE_WITH_STACK=0
      shift
      ;;
    --torch-profile-no-gzip)
      TORCH_PROFILE_USE_GZIP=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      vllm_args+=("$@")
      break
      ;;
    *)
      vllm_args+=("$1")
      shift
      ;;
  esac
done

TORCH_PROFILE_RECORD_SHAPES=$(bool_value \
  TORCH_PROFILE_RECORD_SHAPES "${TORCH_PROFILE_RECORD_SHAPES}")
TORCH_PROFILE_WITH_MEMORY=$(bool_value \
  TORCH_PROFILE_WITH_MEMORY "${TORCH_PROFILE_WITH_MEMORY}")
TORCH_PROFILE_WITH_STACK=$(bool_value \
  TORCH_PROFILE_WITH_STACK "${TORCH_PROFILE_WITH_STACK}")
TORCH_PROFILE_WITH_FLOPS=$(bool_value \
  TORCH_PROFILE_WITH_FLOPS "${TORCH_PROFILE_WITH_FLOPS}")
TORCH_PROFILE_USE_GZIP=$(bool_value \
  TORCH_PROFILE_USE_GZIP "${TORCH_PROFILE_USE_GZIP}")
PREFIX_CACHING=$(bool_value PREFIX_CACHING "${PREFIX_CACHING}")
ENABLE_CHUNKED_PREFILL=$(bool_value \
  ENABLE_CHUNKED_PREFILL "${ENABLE_CHUNKED_PREFILL}")
ASYNC_SCHEDULING=$(bool_value ASYNC_SCHEDULING "${ASYNC_SCHEDULING}")
SCHEDULER_RESERVE_FULL_ISL=$(bool_value \
  SCHEDULER_RESERVE_FULL_ISL "${SCHEDULER_RESERVE_FULL_ISL}")
DSPARK_ADAPTIVE_VERIFICATION=$(bool_value \
  DSPARK_ADAPTIVE_VERIFICATION "${DSPARK_ADAPTIVE_VERIFICATION}")
ENGRAM_DISK_RESIDENT_SCALES=$(bool_value \
  ENGRAM_DISK_RESIDENT_SCALES "${ENGRAM_DISK_RESIDENT_SCALES}")
ENGRAM_PROJECTION_TP=$(bool_value \
  ENGRAM_PROJECTION_TP "${ENGRAM_PROJECTION_TP}")

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python interpreter not found or not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

require_positive_int TP_SIZE "${TP_SIZE}"
require_positive_int MAX_NUM_SEQS "${MAX_NUM_SEQS}"
require_positive_int MAX_NUM_BATCHED_TOKENS "${MAX_NUM_BATCHED_TOKENS}"
require_positive_int BLOCK_SIZE "${BLOCK_SIZE}"
require_positive_int MAX_CUDAGRAPH_CAPTURE_SIZE \
  "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
require_positive_int NUM_SPECULATIVE_TOKENS "${NUM_SPECULATIVE_TOKENS}"
require_positive_number DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE \
  "${DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE}"
if [[ "${MAX_MODEL_LEN}" != "auto" && "${MAX_MODEL_LEN}" != "-1" ]]; then
  require_positive_int MAX_MODEL_LEN "${MAX_MODEL_LEN}"
fi
if [[ ! "${GPU_MEMORY_UTILIZATION}" =~ ^(0([.][0-9]+)?|1([.]0*)?)$ ]] \
  || ! awk -v value="${GPU_MEMORY_UTILIZATION}" \
    'BEGIN { exit !((value + 0) > 0 && (value + 0) <= 1) }'; then
  echo "GPU_MEMORY_UTILIZATION must be in (0, 1]; got '${GPU_MEMORY_UTILIZATION}'" >&2
  exit 2
fi
case "${ENGRAM_TABLE_MEMORY}" in
  device|ram|disk) ;;
  *)
    echo "ENGRAM_TABLE_MEMORY must be device, ram, or disk; got '${ENGRAM_TABLE_MEMORY}'" >&2
    exit 2
    ;;
esac

if [[ "${MAX_PARALLEL_PREFILLS}" != "auto" ]]; then
  require_positive_int MAX_PARALLEL_PREFILLS "${MAX_PARALLEL_PREFILLS}"
  if ((MAX_PARALLEL_PREFILLS > MAX_NUM_SEQS)); then
    echo "MAX_PARALLEL_PREFILLS cannot exceed MAX_NUM_SEQS" >&2
    exit 2
  fi
fi
if [[ -z "${PREFILL_POLICY}" ]]; then
  if [[ "${MAX_PARALLEL_PREFILLS}" == "1" ]]; then
    PREFILL_POLICY=round-robin
  else
    PREFILL_POLICY=decode-aware
  fi
fi
case "${PREFILL_POLICY}" in
  round-robin|decode-aware) ;;
  *)
    echo "PREFILL_POLICY must be round-robin or decode-aware" >&2
    exit 2
    ;;
esac
if [[ "${MAX_PARALLEL_PREFILLS}" == "1" \
  && "${PREFILL_POLICY}" != "round-robin" ]]; then
  echo "PREFILL_POLICY=decode-aware requires MAX_PARALLEL_PREFILLS greater than one" >&2
  exit 2
fi
if [[ "${DECODE_REFILL_TARGET}" != "auto" ]]; then
  require_positive_int DECODE_REFILL_TARGET "${DECODE_REFILL_TARGET}"
  if ((DECODE_REFILL_TARGET > MAX_NUM_SEQS)); then
    echo "DECODE_REFILL_TARGET cannot exceed MAX_NUM_SEQS" >&2
    exit 2
  fi
  if [[ "${PREFILL_POLICY}" != "decode-aware" ]]; then
    echo "DECODE_REFILL_TARGET requires PREFILL_POLICY=decode-aware" >&2
    exit 2
  fi
fi
if [[ "${MAX_PARALLEL_PREFILLS}" != "1" \
  && "${ENABLE_CHUNKED_PREFILL}" != "1" ]]; then
  echo "Parallel prefill interleaving requires ENABLE_CHUNKED_PREFILL=1" >&2
  exit 2
fi

fairness_args=()
case "${PREFILL_COMPUTE_SHARE}" in
  off|none|disabled)
    if [[ -n "${PREFILL_COMPUTE_HALF_LIFE}" ]]; then
      echo "PREFILL_COMPUTE_HALF_LIFE requires PREFILL_COMPUTE_SHARE=auto" >&2
      exit 2
    fi
    ;;
  auto)
    PREFILL_COMPUTE_HALF_LIFE="${PREFILL_COMPUTE_HALF_LIFE:-smooth}"
    fairness_args=(
      --prefill-compute-share auto
      --prefill-compute-half-life "${PREFILL_COMPUTE_HALF_LIFE}"
    )
    ;;
  *)
    if [[ ! "${PREFILL_COMPUTE_SHARE}" =~ ^(0([.][0-9]+)|[.][0-9]+)$ ]] \
      || ! awk -v value="${PREFILL_COMPUTE_SHARE}" \
        'BEGIN { exit !((value + 0) > 0 && (value + 0) < 1) }'; then
      echo "PREFILL_COMPUTE_SHARE must be auto, off, or a value in (0, 1)" >&2
      exit 2
    fi
    if [[ -n "${PREFILL_COMPUTE_HALF_LIFE}" ]]; then
      echo "PREFILL_COMPUTE_HALF_LIFE requires PREFILL_COMPUTE_SHARE=auto" >&2
      exit 2
    fi
    fairness_args=(--prefill-compute-share "${PREFILL_COMPUTE_SHARE}")
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
if ((${#visible_devices[@]} < TP_SIZE)); then
  echo "TP_SIZE=${TP_SIZE} requires at least ${TP_SIZE} CUDA_VISIBLE_DEVICES; got '${CUDA_VISIBLE_DEVICES}'" >&2
  exit 2
fi

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_STANDALONE_COMPILE="${VLLM_USE_STANDALONE_COMPILE:-1}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}"
export VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD="${VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD:-1024}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-96KB}"
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-96KB}"
unset VLLM_ALLOW_INSECURE_SERIALIZATION

profiler_args=()
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  if [[ "${TORCH_PROFILE_DIR}" != /* ]]; then
    TORCH_PROFILE_DIR="${SCRIPT_DIR}/${TORCH_PROFILE_DIR}"
  fi
  mkdir -p -- "${TORCH_PROFILE_DIR}"
  profiler_config="$(
    "${PYTHON_BIN}" - \
      "${TORCH_PROFILE_DIR}" \
      "${TORCH_PROFILE_RECORD_SHAPES}" \
      "${TORCH_PROFILE_WITH_MEMORY}" \
      "${TORCH_PROFILE_WITH_STACK}" \
      "${TORCH_PROFILE_WITH_FLOPS}" \
      "${TORCH_PROFILE_USE_GZIP}" \
      "${TORCH_PROFILE_MAX_ITERATIONS}" <<'PY'
import json
import sys

(
    output_dir,
    record_shapes,
    with_memory,
    with_stack,
    with_flops,
    use_gzip,
    max_iterations,
) = sys.argv[1:]
print(
    json.dumps(
        {
            "profiler": "torch",
            "torch_profiler_dir": output_dir,
            "torch_profiler_record_shapes": record_shapes == "1",
            "torch_profiler_with_memory": with_memory == "1",
            "torch_profiler_with_stack": with_stack == "1",
            "torch_profiler_with_flops": with_flops == "1",
            "torch_profiler_use_gzip": use_gzip == "1",
            "ignore_frontend": True,
            "delay_iterations": 0,
            "max_iterations": int(max_iterations),
        }
    )
)
PY
  )"
  profiler_args=(--profiler-config "${profiler_config}")
fi

scheduler_args=(
  --max-parallel-prefills "${MAX_PARALLEL_PREFILLS}"
  --prefill-policy "${PREFILL_POLICY}"
  --decode-refill-target "${DECODE_REFILL_TARGET}"
  "${fairness_args[@]}"
)
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  scheduler_args+=(--enable-chunked-prefill)
else
  scheduler_args+=(--no-enable-chunked-prefill)
fi
if [[ "${ASYNC_SCHEDULING}" == "1" ]]; then
  scheduler_args+=(--async-scheduling)
else
  scheduler_args+=(--no-async-scheduling)
fi
if [[ "${SCHEDULER_RESERVE_FULL_ISL}" == "1" ]]; then
  scheduler_args+=(--scheduler-reserve-full-isl)
else
  scheduler_args+=(--no-scheduler-reserve-full-isl)
fi

prefix_cache_args=(--enable-prefix-caching)
if [[ "${PREFIX_CACHING}" == "0" ]]; then
  prefix_cache_args=(--no-enable-prefix-caching)
fi

adaptive_verification=false
if [[ "${DSPARK_ADAPTIVE_VERIFICATION}" == "1" ]]; then
  adaptive_verification=true
fi
disk_resident_scales=false
if [[ "${ENGRAM_DISK_RESIDENT_SCALES}" == "1" ]]; then
  disk_resident_scales=true
fi
projection_tp=false
if [[ "${ENGRAM_PROJECTION_TP}" == "1" ]]; then
  projection_tp=true
fi
speculative_config=$(printf \
  '{"method":"dspark","num_speculative_tokens":%s,"draft_tensor_parallel_size":%s,"attention_backend":"B12X","draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":%s,"adaptive_verification_cost_scale":%s}' \
  "${NUM_SPECULATIVE_TOKENS}" "${TP_SIZE}" "${adaptive_verification}" \
  "${DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE}")
engram_config=$(printf \
  '{"cpu_offload":false,"table_memory":"%s","disk_resident_scales":%s,"projection_tp":%s}' \
  "${ENGRAM_TABLE_MEMORY}" "${disk_resident_scales}" \
  "${projection_tp}")
compilation_config='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}" --port "${PORT}"
  --dtype bfloat16
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1
  --load-format "${LOAD_FORMAT}"
  --safetensors-load-strategy lazy
  --block-size "${BLOCK_SIZE}"
  "${prefix_cache_args[@]}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  --compilation-config "${compilation_config}"
  "${scheduler_args[@]}"
  --generation-config vllm
  --engram-config "${engram_config}"
  --attention-backend B12X
  --linear-backend b12x
  --moe-backend b12x
  --speculative-config "${speculative_config}"
  --jit-monitor-mode error
  --tokenizer-mode deepseek_v41
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-request-id-headers
  --default-chat-template-kwargs.thinking=true
  --default-chat-template-kwargs.reasoning_effort=high
  "${profiler_args[@]}"
  "${vllm_args[@]}"
)

cd "${SCRIPT_DIR}"
printf 'Launching %s: TP%s, GPUs %s, %s Engram, DSpark (%s draft tokens)\n' \
  "${SERVED_MODEL_NAME}" "${TP_SIZE}" "${CUDA_VISIBLE_DEVICES}" \
  "${ENGRAM_TABLE_MEMORY}" "${NUM_SPECULATIVE_TOKENS}" >&2
printf 'Engram offload: resident_scales=%s projection_tp=%s\n' \
  "${ENGRAM_DISK_RESIDENT_SCALES}" \
  "${ENGRAM_PROJECTION_TP}" >&2
printf 'DSpark adaptive verification cost scale: %s\n' \
  "${DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE}" >&2
printf 'Scheduling: chunked=%s async=%s reserve_full_isl=%s compute_share=%s parallel_prefills=%s policy=%s decode_refill=%s\n' \
  "${ENABLE_CHUNKED_PREFILL}" "${ASYNC_SCHEDULING}" \
  "${SCHEDULER_RESERVE_FULL_ISL}" "${PREFILL_COMPUTE_SHARE}" \
  "${MAX_PARALLEL_PREFILLS}" "${PREFILL_POLICY}" \
  "${DECODE_REFILL_TARGET}" >&2
printf 'Execution: native V4.1 KV, B12X mixed prefill/decode attention, %s graphs (max %s)\n' \
  "FULL_AND_PIECEWISE" "${MAX_CUDAGRAPH_CAPTURE_SIZE}" >&2
printf 'Endpoint: http://%s:%s/v1  |  GPU memory budget: %s\n' \
  "${HOST}" "${PORT}" "${GPU_MEMORY_UTILIZATION}" >&2
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  printf 'Torch CPU+CUDA profiling configured; traces: %s\n' \
    "${TORCH_PROFILE_DIR}" >&2
  printf 'Trigger using the vllm-take-capture skill; auto-stop: %s engine steps.\n' \
    "${TORCH_PROFILE_MAX_ITERATIONS}" >&2
fi
if [[ "$(bool_value DRY_RUN "${DRY_RUN:-0}")" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
