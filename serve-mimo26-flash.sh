#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_REVISION="${MODEL_REVISION:-5711b268169967567844e1e560e8a3966da959b1}"
hf_cache="${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-${HOME}/.cache}/huggingface}/hub}"
MODEL_PATH="${MODEL_PATH:-${hf_cache}/models--XiaomiMiMo--MiMo-V2.6-Flash-RL/snapshots/${MODEL_REVISION}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-XiaomiMiMo/MiMo-V2.6-Flash-RL}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE_IDS="${DEVICE_IDS:-8,9,10,11}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SCHEDULED_TOKENS="${MAX_NUM_SCHEDULED_TOKENS:-2048}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
DFLASH_MODEL_PATH="${DFLASH_MODEL_PATH:-${MODEL_PATH}/dflash}"
DFLASH_KV_CACHE_DTYPE="${DFLASH_KV_CACHE_DTYPE:-fp8}"
LOAD_FORMAT="${LOAD_FORMAT:-b12x}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
for model_file in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json; do
  if [[ ! -f "${MODEL_PATH}/${model_file}" ]]; then
    echo "Model file not found: ${MODEL_PATH}/${model_file}" >&2
    echo "Complete the HF snapshot with:" >&2
    printf '  uv run --no-project --python %q --with huggingface_hub hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --revision %q\n' \
      "${PYTHON_BIN}" "${MODEL_REVISION}" >&2
    exit 1
  fi
done
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is already set; this launcher selects physical GPUs with --device-ids." >&2
  echo "Run it from an unmasked shell and select GPUs with DEVICE_IDS." >&2
  exit 2
fi
if [[ ! "${DEVICE_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "DEVICE_IDS must be a comma-separated list of physical GPU indices; got '${DEVICE_IDS}'" >&2
  exit 2
fi
IFS=, read -r -a device_id_list <<< "${DEVICE_IDS}"
TP_SIZE="${TP_SIZE:-${#device_id_list[@]}}"
if [[ ! "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer; got '${TP_SIZE}'" >&2
  exit 2
fi
if ((${#device_id_list[@]} != TP_SIZE)); then
  echo "TP_SIZE=${TP_SIZE} requires ${TP_SIZE} DEVICE_IDS; got '${DEVICE_IDS}'" >&2
  exit 2
fi
if [[ "${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" != PCI_BUS_ID ]]; then
  echo "CUDA_DEVICE_ORDER must be PCI_BUS_ID when using physical --device-ids" >&2
  exit 2
fi
if [[ ! "${MAX_MODEL_LEN}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_MODEL_LEN must be a positive integer; got '${MAX_MODEL_LEN}'" >&2
  exit 2
fi
if [[ ! "${NUM_SPECULATIVE_TOKENS}" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a non-negative integer; got '${NUM_SPECULATIVE_TOKENS}'" >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"
if [[ "${LOAD_FORMAT}" == b12x ]]; then
  export VLLM_PLUGINS="${VLLM_PLUGINS:+${VLLM_PLUGINS},}b12x_loader"
fi
export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/opt/cuda}}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE:-2MB}"
export VLLM_PCIE_DMA_MIN_BYTES="${VLLM_PCIE_DMA_MIN_BYTES:-off}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-BLHNC}"
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-1}"
export B12X_W4A16_FP32_TOPK_WEIGHTS="${B12X_W4A16_FP32_TOPK_WEIGHTS:-1}"
export B12X_DENSE_SPLITK_TURBO="${B12X_DENSE_SPLITK_TURBO:-0}"
export VLLM_B12X_BF16_GEMV="${VLLM_B12X_BF16_GEMV:-1}"
export VLLM_MIMO_L2_PREFETCH="${VLLM_MIMO_L2_PREFETCH:-1}"
export VLLM_DFLASH_VOCAB_PARALLEL_DRAFT="${VLLM_DFLASH_VOCAB_PARALLEL_DRAFT:-1}"
export VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS="${VLLM_SCHEDULER_UNCAP_PREFILL_ONLY_STEPS:-1}"
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN="${VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN:-32768}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

speculative_args=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  if [[ ! -f "${DFLASH_MODEL_PATH}/config.json" ]]; then
    echo "DFlash config not found: ${DFLASH_MODEL_PATH}/config.json" >&2
    exit 1
  fi
  speculative_config="$("${PYTHON_BIN}" -c \
    'import json, sys; print(json.dumps({"method": "dflash", "model": sys.argv[1], "num_speculative_tokens": int(sys.argv[2]), "draft_tensor_parallel_size": int(sys.argv[3]), "max_model_len": int(sys.argv[4]), "enable_adaptive_verification": True, "draft_sample_method": "probabilistic", "rejection_sample_method": "standard", "attention_backend": "B12X", "kv_cache_dtype": sys.argv[5]}))' \
    "${DFLASH_MODEL_PATH}" "${NUM_SPECULATIVE_TOKENS}" "${TP_SIZE}" "${MAX_MODEL_LEN}" "${DFLASH_KV_CACHE_DTYPE}")"
  speculative_args=(--speculative-config "${speculative_config}")
fi

command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --device-ids "${DEVICE_IDS}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size 1
  --mm-encoder-tp-mode weights
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --block-size 64
  --attention-backend B12X
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --load-format "${LOAD_FORMAT}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-num-scheduled-tokens "${MAX_NUM_SCHEDULED_TOKENS}"
  --prefill-compute-share 0.8
  --async-scheduling
  --no-scheduler-reserve-full-isl
  --max-cudagraph-capture-size 256
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --hf-overrides '{"architectures":["MiMoV2OmniForCausalLM"]}'
  --reasoning-parser mimo
  --tool-call-parser mimo
  --enable-auto-tool-choice
  --generation-config vllm
  --override-generation-config '{"temperature":1.0,"top_p":0.95}'
  "${speculative_args[@]}"
  "$@"
)

cd "${SCRIPT_DIR}"
printf 'Launching %s as %s on devices %s (TP=%s)\n' \
  "${MODEL_PATH}" "${SERVED_MODEL_NAME}" "${DEVICE_IDS}" "${TP_SIZE}" >&2
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
