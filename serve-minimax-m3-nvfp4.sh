#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_PATH="/models/MiniMax-M3-NVFP4"
SERVED_MODEL_NAME="MiniMax-M3-NVFP4"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

model_path = Path(os.environ["MODEL_PATH"])
config_path = model_path / "config.json"
if not config_path.is_file():
    raise SystemExit(f"ERROR: missing model config: {config_path}")

config = json.loads(config_path.read_text())
hq_path = model_path / "hf_quant_config.json"
hf_quant_config = json.loads(hq_path.read_text()) if hq_path.is_file() else {}
quant_config = config.get("quantization_config") or {}
quant_algo = str(
    quant_config.get("quant_algo") or hf_quant_config.get("quant_algo") or ""
).upper()
quant_method = str(
    quant_config.get("quant_method") or hf_quant_config.get("quant_method") or ""
).lower()
config_text = json.dumps(
    {"config": config, "hf_quant_config": hf_quant_config}
).upper()

if quant_algo != "NVFP4" or quant_method != "modelopt":
    raise SystemExit(
        "ERROR: this diagnostic launcher requires the ModelOpt NVFP4 "
        f"checkpoint, got quant_algo={quant_algo!r}, "
        f"quant_method={quant_method!r}."
    )
if "MXFP8" in config_text or "MXFP8" in str(model_path).upper():
    raise SystemExit("ERROR: MXFP8 checkpoint/config detected; use NVFP4 here.")

text_config = config.get("text_config") or {}
print(
    f"Verified non-MXFP8 model config: {model_path} "
    f"(quant_algo={quant_algo}, quant_method={quant_method}, "
    f"text_model_type={text_config.get('model_type')}, "
    f"hidden_size={text_config.get('hidden_size')}, "
    f"layers={text_config.get('num_hidden_layers')})",
    flush=True,
)
PY

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SAFETENSORS_FAST_GPU=1
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE="${VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-0}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_USE_B12X_MINIMAX_M3_MSA="${VLLM_USE_B12X_MINIMAX_M3_MSA:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export VLLM_USE_B12X_SPARSE_INDEXER="${VLLM_USE_B12X_SPARSE_INDEXER:-1}"
export B12X_DYNAMIC_DETERMINISTIC_OUTPUT="${B12X_DYNAMIC_DETERMINISTIC_OUTPUT:-0}"
export B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START="${B12X_LOG_CUTE_COMPILES_AFTER_ENGINE_START:-1}"

case "${VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    export VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=1
    ;;
  *)
    echo "ERROR: this MiniMax M3 launcher requires full graph compilation." >&2
    echo "Do not set VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE=0 for this run." >&2
    exit 1
    ;;
esac

case "${VLLM_USE_BREAKABLE_CUDAGRAPH}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF|"")
    export VLLM_USE_BREAKABLE_CUDAGRAPH=0
    ;;
  *)
    echo "ERROR: this MiniMax M3 launcher is for full, unbroken graphs." >&2
    echo "Do not set VLLM_USE_BREAKABLE_CUDAGRAPH=1 for this run." >&2
    exit 1
    ;;
esac

case "${VLLM_USE_AOT_COMPILE}" in
  1|true|True|TRUE|yes|Yes|YES|on|On|ON)
    export VLLM_USE_AOT_COMPILE=1
    ;;
  *)
    echo "ERROR: this MiniMax M3 launcher requires AOT compile." >&2
    echo "Do not set VLLM_USE_AOT_COMPILE=0 for this run." >&2
    exit 1
    ;;
esac

case "${VLLM_USE_B12X_FP8_GEMM}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF|"")
    export VLLM_USE_B12X_FP8_GEMM=0
    ;;
  *)
    echo "ERROR: this NVFP4 diagnostic launcher keeps B12X FP8 GEMM disabled." >&2
    echo "Unset VLLM_USE_B12X_FP8_GEMM or set it to 0 for this run." >&2
    exit 1
    ;;
esac

EXTRA_ARGS=()
while (($#)); do
  arg="$1"
  case "${arg}" in
    --host)
      if (($# < 2)); then
        echo "ERROR: --host requires a value." >&2
        exit 1
      fi
      HOST="$2"
      shift 2
      ;;
    --host=*)
      HOST="${arg#--host=}"
      shift
      ;;
    --port)
      if (($# < 2)); then
        echo "ERROR: --port requires a value." >&2
        exit 1
      fi
      PORT="$2"
      shift 2
      ;;
    --port=*)
      PORT="${arg#--port=}"
      shift
      ;;
    --served-model-name|--served-model-name=*|--tokenizer|--tokenizer=*|\
    --hf-config-path|--hf-config-path=*|--quantization|--quantization=*|\
    --attention-backend|--attention-backend=*|--kv-cache-dtype|\
    --kv-cache-dtype=*|--moe-backend|--moe-backend=*|--block-size|\
    --block-size=*|-cc.mode|-cc.mode=*|-cc.cudagraph_mode|\
    -cc.cudagraph_mode=*|mxfp8|modelopt_mxfp8|*MiniMax-M3-MXFP8*)
      echo "ERROR: this launcher is pinned to ${MODEL_PATH} as ${SERVED_MODEL_NAME}." >&2
      echo "Do not override model identity, quantization, backend, or graph settings here." >&2
      exit 1
      ;;
    *)
      EXTRA_ARGS+=("${arg}")
      shift
      ;;
  esac
done

M3_PROFILE="${M3_PROFILE:-torch}"
PROFILER_ARGS=()
case "${M3_PROFILE,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on|torch)
    M3_PROFILE_DIR="/tmp/vllm-profile/minimax-m3-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "${M3_PROFILE_DIR}"
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
    PROFILER_ARGS+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${M3_PROFILE_DIR}"
      --profiler-config.torch_profiler_with_stack=true
      --profiler-config.torch_profiler_record_shapes=false
      --profiler-config.torch_profiler_with_memory=false
      --profiler-config.torch_profiler_with_flops=false
      --profiler-config.torch_profiler_use_gzip=true
      --profiler-config.torch_profiler_dump_cuda_time_total=false
      --profiler-config.ignore_frontend=true
      --profiler-config.delay_iterations=0
      --profiler-config.max_iterations=4
      --profiler-config.warmup_iterations=0
      --profiler-config.active_iterations=5
      --profiler-config.wait_iterations=0
    )
    echo "Torch profiling enabled. Traces will be written under: ${M3_PROFILE_DIR}" >&2
    ;;
  cuda|nsys|nsight)
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    PROFILER_ARGS+=(--profiler-config.profiler=cuda)
    echo "CUDA profiler enabled. Use nsys with --capture-range=cudaProfilerApi and drive /start_profile + /stop_profile." >&2
    ;;
  *)
    echo "ERROR: M3_PROFILE must be one of off, torch, cuda, nsys, or nsight; got '${M3_PROFILE}'" >&2
    exit 1
    ;;
esac

cd "${SCRIPT_DIR}"
echo "Launching ${MODEL_PATH} as ${SERVED_MODEL_NAME}" >&2
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.98 \
  --max-num-batched-tokens 2048 \
  --max-model-len 256000 \
  --max-num-seqs 4 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend B12X_ATTN \
  --moe-backend b12x \
  -cc.mode=VLLM_COMPILE \
  -cc.cudagraph_mode=FULL_AND_PIECEWISE \
  --block-size 128 \
  --load-format fastsafetensors \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  "${PROFILER_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
