#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

MODEL_PATH="${MODEL_PATH:-/data/models/qwen3.8-flash-next-mixed/qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next-4p89bpw}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-2}"
if [[ "${TP_SIZE}" == 1 ]]; then
  DEFAULT_DEVICE_IDS=8
  default_ple_cpu_offload=1
else
  DEFAULT_DEVICE_IDS=8,9
  default_ple_cpu_offload=0
fi
DEVICE_IDS="${DEVICE_IDS:-${DEFAULT_DEVICE_IDS}}"
VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-${default_ple_cpu_offload}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
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
    "Environment modes:" \
    "  TP_SIZE=1                    Use one GPU and mapped-host n-gram tables." \
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
VLLM_PLE_CPU_OFFLOAD=$(bool_value \
  VLLM_PLE_CPU_OFFLOAD "${VLLM_PLE_CPU_OFFLOAD}")

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
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
if [[ "${VLLM_SSM_CONV_STATE_LAYOUT:-DS}" != DS ]]; then
  echo "VLLM_SSM_CONV_STATE_LAYOUT must be DS for Qwen3.8-Flash-Next" >&2
  exit 2
fi
if [[ "${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" != PCI_BUS_ID ]]; then
  echo "CUDA_DEVICE_ORDER must be PCI_BUS_ID when using physical --device-ids" >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_PLUGINS=
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_PLE_CPU_OFFLOAD
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf -v speculative_config \
  '{"method":"mtp","num_speculative_tokens":%s}' \
  "${NUM_SPECULATIVE_TOKENS}"

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

command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --device-ids "${DEVICE_IDS}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size 1
  --mm-encoder-tp-mode data
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype bfloat16
  --quantization modelopt_mixed
  --block-size 16
  --load-format "${LOAD_FORMAT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --speculative-config "${speculative_config}"
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  "${profiler_args[@]}"
  "${vllm_args[@]}"
)

cd "${SCRIPT_DIR}"
printf 'Launching %s as %s on devices %s\n' \
  "${MODEL_PATH}" "${SERVED_MODEL_NAME}" "${DEVICE_IDS}" >&2
if [[ "${VLLM_PLE_CPU_OFFLOAD}" == 1 ]]; then
  printf 'PLE n-gram tables: CUDA-mapped host DRAM\n' >&2
fi
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  printf 'Torch CPU+CUDA profiling enabled; traces: %s\n' \
    "${TORCH_PROFILE_DIR}" >&2
  printf 'Trigger with b12x vllm-take-capture; auto-stop: %s engine steps.\n' \
    "${TORCH_PROFILE_MAX_ITERATIONS}" >&2
fi
exec "${command[@]}"
