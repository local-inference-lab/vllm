#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/models/qwen3.8-flash-next-mixed/qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next-4p89bpw}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1610612736}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
LOG_FILE="${LOG_FILE-${SCRIPT_DIR}/runlogs/qwen38-flash-next-tp1.log}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
TORCH_PROFILE_DEFAULT_DIR=/tmp/vllm-ds4-decode
TORCH_PROFILE_MAX_ITERATIONS=4
B12X_POLICY_MODE="${B12X_POLICY_MODE:-auto}"
COMPILATION_CONFIG="${VLLM_QWEN38_COMPILATION_CONFIG:-}"
if [[ -z "${COMPILATION_CONFIG}" ]]; then
  COMPILATION_CONFIG='{"pass_config":{"fuse_act_quant":true}}'
fi

usage() {
  printf '%s\n' \
    "Usage: $0 [launcher options] [vLLM options]" \
    "" \
    "Launcher options:" \
    "  --b12x-policy-mode MODE" \
    "                         auto, heuristic-only, or preplanned-only." \
    "  --torch-profile [DIR]  Configure triggered CPU+CUDA Torch profiles." \
    "                         DIR defaults to /tmp/vllm-ds4-decode." \
    "                         Each trigger auto-stops after four engine steps." \
    "  -h, --help             Show this help." \
    "" \
    "All other arguments are forwarded to vLLM."
}

vllm_args=()
while (($#)); do
  case "$1" in
    --b12x-policy-mode)
      if (($# < 2)); then
        echo "--b12x-policy-mode requires a value" >&2
        exit 2
      fi
      B12X_POLICY_MODE=$2
      shift 2
      ;;
    --b12x-policy-mode=*)
      B12X_POLICY_MODE=${1#*=}
      shift
      ;;
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

case "${B12X_POLICY_MODE}" in
  auto|heuristic-only|preplanned-only) ;;
  *)
    echo "Invalid B12X policy mode: ${B12X_POLICY_MODE}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/usr/local/cuda}}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_PLUGINS=
export VLLM_SSM_CONV_STATE_LAYOUT="${VLLM_SSM_CONV_STATE_LAYOUT:-DS}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_MXFP8_LM_HEAD="${VLLM_MXFP8_LM_HEAD:-1}"
export VLLM_LM_HEAD_A16="${VLLM_LM_HEAD_A16:-1}"
export VLLM_MTP_NVFP4_LM_HEAD="${VLLM_MTP_NVFP4_LM_HEAD:-1}"
export VLLM_QWEN3_8_FLASH_NEXT_OVERLAP="${VLLM_QWEN3_8_FLASH_NEXT_OVERLAP:-1}"
export VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT="${VLLM_QWEN3_8_FLASH_NEXT_MTP_COMPACT:-1}"
export VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH="${VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH:-1}"
export B12X_POLICY_MODE
export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
export INSTANTTENSOR_BUFFER_SIZE="${INSTANTTENSOR_BUFFER_SIZE:-1342177280}"
export INSTANTTENSOR_CONCURRENCY="${INSTANTTENSOR_CONCURRENCY:-1}"
export INSTANTTENSOR_IO_DEPTH="${INSTANTTENSOR_IO_DEPTH:-3}"

speculative_config=$(printf \
  '{"method":"mtp","num_speculative_tokens":%s}' \
  "${NUM_SPECULATIVE_TOKENS}")

profiler_args=()
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  if [[ "${TORCH_PROFILE_DIR}" != /* ]]; then
    TORCH_PROFILE_DIR="${SCRIPT_DIR}/${TORCH_PROFILE_DIR}"
  fi
  profiler_config=$(
    "${PYTHON_BIN}" - \
      "${TORCH_PROFILE_DIR}" \
      "${TORCH_PROFILE_MAX_ITERATIONS}" <<'PY'
import json
import sys

profile_dir, max_iterations = sys.argv[1:]
print(
    json.dumps(
        {
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
            "torch_profiler_record_shapes": False,
            "torch_profiler_with_memory": False,
            "torch_profiler_with_stack": False,
            "torch_profiler_with_flops": False,
            "torch_profiler_use_gzip": True,
            "ignore_frontend": True,
            "delay_iterations": 0,
            "max_iterations": int(max_iterations),
        }
    )
)
PY
  )
  profiler_args=(--profiler-config "${profiler_config}")
fi

command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size 1
  --pipeline-parallel-size 1
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 16
  --load-format instanttensor
  --model-loader-extra-config '{"instanttensor_copy":false}'
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --speculative-config "${speculative_config}"
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --mm-encoder-tp-mode data
  --mm-processor-cache-gb 0
  --limit-mm-per-prompt '{"image":1}'
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --compilation-config "${COMPILATION_CONFIG}"
  "${profiler_args[@]}"
)
command+=("${vllm_args[@]}")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

mapfile -t existing_server_pids < <(
  pgrep -f -- "vllm.entrypoints.cli.main serve ${MODEL_PATH}" || true
)
if ((${#existing_server_pids[@]})); then
  printf 'A vLLM server for this model already exists (PID %s).\n' \
    "${existing_server_pids[0]}" >&2
  echo "Stop it before starting another copy." >&2
  exit 1
fi

if command -v ss >/dev/null \
  && ss -H -ltn "sport = :${PORT}" | rg -q .; then
  echo "Port ${PORT} is already in use; stop the existing server first." >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  mkdir -p -- "${TORCH_PROFILE_DIR}"
  printf 'Triggered CPU+CUDA profiling enabled; traces: %s\n' \
    "${TORCH_PROFILE_DIR}" >&2
  printf 'Each /start_profile trigger auto-stops after %s engine steps.\n' \
    "${TORCH_PROFILE_MAX_ITERATIONS}" >&2
fi
if [[ -z "${LOG_FILE}" ]]; then
  exec "${command[@]}"
fi

mkdir -p -- "$(dirname -- "${LOG_FILE}")"
printf 'Logging to %s\n' "${LOG_FILE}" >&2
"${command[@]}" 2>&1 | tee -a -- "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
