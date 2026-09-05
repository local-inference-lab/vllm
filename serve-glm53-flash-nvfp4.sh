#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

MODEL_PATH="${MODEL_PATH:-/data/models/GLM-5.3-Flash-4p67}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"
LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
DEFAULT_DEVICE_IDS=8,9
DEVICE_IDS="${DEVICE_IDS:-${DEFAULT_DEVICE_IDS}}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
if [[ -z "${MAX_NUM_BATCHED_TOKENS+x}" ]]; then
  if [[ "${TP_SIZE}" == 2 ]]; then
    MAX_NUM_BATCHED_TOKENS=2048
  else
    MAX_NUM_BATCHED_TOKENS=4096
  fi
fi
if [[ -z "${KV_CACHE_MEMORY_BYTES+x}" ]]; then
  if [[ "${TP_SIZE}" == 2 ]]; then
    KV_CACHE_MEMORY_BYTES=2G
  else
    KV_CACHE_MEMORY_BYTES=
  fi
fi
SPECULATOR="${SPECULATOR:-mtp}"
DFLASH2_MODEL="${DFLASH2_MODEL:-incoai/GLM-5.3-Flash-DFlash2}"
ADAPTIVE_SPECULATIVE_TOKENS="${ADAPTIVE_SPECULATIVE_TOKENS:-0}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES:-0}"
TORCH_PROFILE_WITH_MEMORY="${TORCH_PROFILE_WITH_MEMORY:-0}"
TORCH_PROFILE_WITH_STACK="${TORCH_PROFILE_WITH_STACK:-1}"
TORCH_PROFILE_WITH_FLOPS="${TORCH_PROFILE_WITH_FLOPS:-0}"
TORCH_PROFILE_USE_GZIP="${TORCH_PROFILE_USE_GZIP:-1}"
TORCH_PROFILE_DEFAULT_DIR=/tmp/vllm-ds4-decode
TORCH_PROFILE_MAX_ITERATIONS=4

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
    "  --torch-profile [DIR]         Enable a four-step Torch CPU+CUDA capture." \
    "                                DIR defaults to /tmp/vllm-ds4-decode." \
    "  --torch-profile-record-shapes Record tensor shapes." \
    "  --torch-profile-with-memory   Record tensor memory activity." \
    "  --torch-profile-with-flops    Estimate supported operator FLOPs." \
    "  --torch-profile-no-stack      Disable Python stack capture." \
    "  --torch-profile-no-gzip       Write uncompressed trace files." \
    "  -h, --help                    Show this help." \
    "" \
    "Launcher environment:" \
    "  ADAPTIVE_SPECULATIVE_TOKENS=1 Enable adaptive MTP draft depth." \
    "  ADAPTIVE_SPECULATIVE_TOKENS_INITIAL=N" \
    "                                Initial adaptive depth (default: 3)." \
    "  ADAPTIVE_SPECULATIVE_TOKENS_WINDOW=N" \
    "                                Verification steps per update (default: 32)." \
    "  VLLM_MXFP8_LM_HEAD=1          Runtime MXFP8 verifier head (default: 1)." \
    "  VLLM_LM_HEAD_A16=1           BF16 head activations (default: 1)." \
    "  VLLM_MTP_NVFP4_LM_HEAD=1     Separate NVFP4 MTP head (default: 1)." \
    "  VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=1" \
    "                                Reuse KDA decode metadata (default: 1)." \
    "" \
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

case "${SPECULATOR}" in
  mtp)
    default_num_speculative_tokens=5
    ;;
  dflash2)
    # The checkpoint is trained for an eight-token block: one verified token
    # plus seven draft tokens.
    default_num_speculative_tokens=7
    ;;
  *)
    echo "SPECULATOR must be mtp or dflash2; got '${SPECULATOR}'" >&2
    exit 2
    ;;
esac

NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-${default_num_speculative_tokens}}"

if [[ ! "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a non-negative integer; got '${NUM_SPECULATIVE_TOKENS}'" >&2
  exit 2
fi

ADAPTIVE_SPECULATIVE_TOKENS=$(bool_value \
  ADAPTIVE_SPECULATIVE_TOKENS "${ADAPTIVE_SPECULATIVE_TOKENS}")
ADAPTIVE_SPECULATIVE_TOKENS_WINDOW="${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW:-32}"
if [[ -z "${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL+x}" ]]; then
  if ((NUM_SPECULATIVE_TOKENS < 3)); then
    ADAPTIVE_SPECULATIVE_TOKENS_INITIAL=${NUM_SPECULATIVE_TOKENS}
  else
    ADAPTIVE_SPECULATIVE_TOKENS_INITIAL=3
  fi
fi
if ((ADAPTIVE_SPECULATIVE_TOKENS)); then
  if [[ "${SPECULATOR}" != mtp ]] || ((NUM_SPECULATIVE_TOKENS == 0)); then
    echo "ADAPTIVE_SPECULATIVE_TOKENS requires MTP with a positive speculative depth" >&2
    exit 2
  fi
  if [[ ! "${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ADAPTIVE_SPECULATIVE_TOKENS_WINDOW must be a positive integer; got '${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW}'" >&2
    exit 2
  fi
  if [[ ! "${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ADAPTIVE_SPECULATIVE_TOKENS_INITIAL must be a positive integer; got '${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL}'" >&2
    exit 2
  fi
  if ((ADAPTIVE_SPECULATIVE_TOKENS_INITIAL > NUM_SPECULATIVE_TOKENS)); then
    echo "ADAPTIVE_SPECULATIVE_TOKENS_INITIAL must not exceed NUM_SPECULATIVE_TOKENS" >&2
    exit 2
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
HUMMING_NVRTC_LIB_DIR="$(
  "${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13/lib")'
)"
if [[ ! -f "${HUMMING_NVRTC_LIB_DIR}/libnvrtc-builtins.so.13.0" ]]; then
  echo "Humming CUDA 13 NVRTC builtins not found: ${HUMMING_NVRTC_LIB_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is already set; this launcher selects physical GPUs with --device-ids." >&2
  echo "Run it from an unmasked shell so DEVICE_IDS=${DEVICE_IDS} remains physical." >&2
  exit 2
fi
if [[ ! "${DEVICE_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "DEVICE_IDS must be a comma-separated list of physical GPU indices; got '${DEVICE_IDS}'" >&2
  exit 2
fi

IFS=, read -r -a device_id_list <<< "${DEVICE_IDS}"
if ((${#device_id_list[@]} != TP_SIZE)); then
  echo "TP_SIZE=${TP_SIZE} requires ${TP_SIZE} DEVICE_IDS; got '${DEVICE_IDS}'" >&2
  exit 2
fi
if [[ "${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" != PCI_BUS_ID ]]; then
  echo "CUDA_DEVICE_ORDER must be PCI_BUS_ID when using physical --device-ids" >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${HUMMING_NVRTC_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PLUGINS=
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
if [[ "${TP_SIZE}" == 2 ]]; then
  export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-67108864}"
  export INSTANTTENSOR_IO_DEPTH="${INSTANTTENSOR_IO_DEPTH:-3}"
  export INSTANTTENSOR_CONCURRENCY="${INSTANTTENSOR_CONCURRENCY:-1}"
  export INSTANTTENSOR_CHUNK_SIZE="${INSTANTTENSOR_CHUNK_SIZE:-8388608}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-1}"
export VLLM_MXFP8_LM_HEAD="${VLLM_MXFP8_LM_HEAD:-1}"
export VLLM_LM_HEAD_A16="${VLLM_LM_HEAD_A16:-1}"
export VLLM_MTP_NVFP4_LM_HEAD="${VLLM_MTP_NVFP4_LM_HEAD:-1}"
export VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH="${VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH:-1}"

speculative_args=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  case "${SPECULATOR}" in
    mtp)
      adaptive_speculative_config=
      if ((ADAPTIVE_SPECULATIVE_TOKENS)); then
        printf -v adaptive_speculative_config \
          ',"adaptive_speculative_tokens_window":%s,"adaptive_speculative_tokens_initial":%s' \
          "${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW}" \
          "${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL}"
      fi
      printf -v speculative_config \
        '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"humming","attention_backend":"B12X"%s}' \
        "${NUM_SPECULATIVE_TOKENS}" "${adaptive_speculative_config}"
      ;;
    dflash2)
      printf -v speculative_config \
        '{"method":"dflash","model":"%s","num_speculative_tokens":%s,"kv_cache_dtype":"auto"}' \
        "${DFLASH2_MODEL}" "${NUM_SPECULATIVE_TOKENS}"
      ;;
  esac
  speculative_args=(--speculative-config "${speculative_config}")
fi

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

kv_cache_args=()
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  kv_cache_args=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
fi

command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --device-ids "${DEVICE_IDS}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --attention-backend B12X
  --block-size 256
  --moe-backend b12x
  --linear-backend "${LINEAR_BACKEND}"
  --no-enable-flashinfer-autotune
  --load-format "${LOAD_FORMAT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  "${kv_cache_args[@]}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  "${speculative_args[@]}"
  "${profiler_args[@]}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
  "${vllm_args[@]}"
)

cd "${SCRIPT_DIR}"
printf 'Launching %s as %s directly on devices %s\n' \
  "${MODEL_PATH}" "${SERVED_MODEL_NAME}" "${DEVICE_IDS}" >&2
printf 'Serving NVFP4 routed experts through B12X W4A16 (BF16 activations)\n' >&2
printf 'Linear backend: %s\n' "${LINEAR_BACKEND}" >&2
if [[ "${LOAD_FORMAT}" == instanttensor && "${TP_SIZE}" == 2 ]]; then
  printf 'InstantTensor staging: %s-byte GPU ceiling, depth %s, concurrency %s\n' \
    "${INSTANTTENSOR_BUFFER_SIZE}" "${INSTANTTENSOR_IO_DEPTH}" \
    "${INSTANTTENSOR_CONCURRENCY}" >&2
fi
printf 'Speculator: %s (%s draft tokens)\n' \
  "${SPECULATOR}" "${NUM_SPECULATIVE_TOKENS}" >&2
if [[ "${SPECULATOR}" == mtp ]] \
    && ((NUM_SPECULATIVE_TOKENS > 0)) \
    && ((ADAPTIVE_SPECULATIVE_TOKENS)); then
  printf 'Adaptive MTP depth: initial %s, maximum %s, window %s verification steps\n' \
    "${ADAPTIVE_SPECULATIVE_TOKENS_INITIAL}" \
    "${NUM_SPECULATIVE_TOKENS}" \
    "${ADAPTIVE_SPECULATIVE_TOKENS_WINDOW}" >&2
fi
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  printf 'K/V cache: %s per GPU; max batched tokens: %s\n' \
    "${KV_CACHE_MEMORY_BYTES}" "${MAX_NUM_BATCHED_TOKENS}" >&2
else
  printf 'K/V cache: auto at %.2f GPU utilization; max batched tokens: %s\n' \
    "${GPU_MEMORY_UTILIZATION}" "${MAX_NUM_BATCHED_TOKENS}" >&2
fi
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  printf 'Torch CPU+CUDA profiling enabled; traces: %s\n' \
    "${TORCH_PROFILE_DIR}" >&2
  printf 'Trigger with b12x vllm-take-capture; auto-stop: %s engine steps.\n' \
    "${TORCH_PROFILE_MAX_ITERATIONS}" >&2
fi
exec "${command[@]}"
