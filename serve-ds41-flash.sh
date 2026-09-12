#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4.1-Flash setup: TP4 on GPUs 0-3, configurable Engram storage, DSpark.
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
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
ENGRAM_TABLE_MEMORY="${ENGRAM_TABLE_MEMORY:-disk}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES:-0}"
TORCH_PROFILE_WITH_MEMORY="${TORCH_PROFILE_WITH_MEMORY:-0}"
TORCH_PROFILE_WITH_STACK="${TORCH_PROFILE_WITH_STACK:-1}"
TORCH_PROFILE_WITH_FLOPS="${TORCH_PROFILE_WITH_FLOPS:-0}"
TORCH_PROFILE_USE_GZIP="${TORCH_PROFILE_USE_GZIP:-1}"
TORCH_PROFILE_DEFAULT_DIR=/tmp/vllm-ds4-decode
TORCH_PROFILE_MAX_ITERATIONS=4
TP_SIZE="${TP_SIZE:-4}"

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
    "variables use the TORCH_PROFILE_* names declared at the top of the script."
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

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf 'Python interpreter not found or not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export CUTE_DSL_ARCH=sm_120a
export OMP_NUM_THREADS=8

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
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="96KB"
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="96KB"

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

speculative_config="{\"method\":\"dspark\",\"num_speculative_tokens\":7,\"draft_tensor_parallel_size\":${TP_SIZE},\"attention_backend\":\"B12X\",\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"standard\",\"enable_adaptive_verification\":true}"
command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}" --port "${PORT}"
  --dtype bfloat16
  --tensor-parallel-size "${TP_SIZE}"
  --load-format "${LOAD_FORMAT}"
  --safetensors-load-strategy lazy
  --block-size 256
  --enable-prefix-caching
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max_cudagraph_capture_size 128
  --generation-config vllm
  --limit-mm-per-prompt '{"image":2}'
  --engram-config "{\"cpu_offload\":false,\"table_memory\":\"${ENGRAM_TABLE_MEMORY}\"}"
  --linear-backend b12x
  --moe-backend b12x
  --speculative-config "${speculative_config}"
  --jit-monitor-mode error
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  "${profiler_args[@]}"
  "${vllm_args[@]}"
)

cd "${SCRIPT_DIR}"
printf 'Launching %s: TP4, GPUs %s, %s Engram, DSpark (7 draft tokens)\n' \
  "${SERVED_MODEL_NAME}" "${CUDA_VISIBLE_DEVICES}" "${ENGRAM_TABLE_MEMORY}" >&2
printf 'Endpoint: http://%s:%s/v1  |  GPU memory budget: %s\n' \
  "${HOST}" "${PORT}" "${GPU_MEMORY_UTILIZATION}" >&2
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  printf 'Torch CPU+CUDA profiling configured; traces: %s\n' \
    "${TORCH_PROFILE_DIR}" >&2
  printf 'Trigger using the vllm-take-capture skill; auto-stop: %s engine steps.\n' \
    "${TORCH_PROFILE_MAX_ITERATIONS}" >&2
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
