#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

MODEL="${MODEL:-/data/models/GLM-5.3-NVFP4}"
MTP_MODEL="${MTP_MODEL:-${MODEL}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-GLM-5.3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-auto}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-2048}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"],\"cudagraph_capture_sizes\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,24,28,32,36,40,44,48,52,56,60,64]}}"
GLM53_INDEX_TOPK_PATTERN="FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
DEFAULT_HF_OVERRIDES="$(printf \
  '{\"index_topk_pattern\":\"%s\"}' "${GLM53_INDEX_TOPK_PATTERN}")"
HF_OVERRIDES="${HF_OVERRIDES:-${DEFAULT_HF_OVERRIDES}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi
if [[ ! -f "${MODEL}/config.json" ]]; then
  echo "Model config not found: ${MODEL}/config.json" >&2
  exit 1
fi
if [[ ! "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "TP_SIZE must be a positive integer." >&2
  exit 2
fi
if [[ ! "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a non-negative integer." >&2
  exit 2
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
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
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE="${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-84KB}"
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-1}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

speculative_args=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  printf -v speculative_config \
    '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"b12x","attention_backend":"B12X","draft_sample_method":"probabilistic"}' \
    "${MTP_MODEL}" "${NUM_SPECULATIVE_TOKENS}"
  speculative_args=(--speculative-config "${SPEC_CONFIG:-${speculative_config}}")
fi

cd "${SCRIPT_DIR}"
exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size "${DCP_SIZE}" \
  --dcp-comm-backend "${DCP_COMM_BACKEND:-a2a}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --compilation-config "${COMPILATION_CONFIG}" \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --linear-backend "${LINEAR_BACKEND}" \
  --moe-backend "${MOE_BACKEND}" \
  --load-format "${LOAD_FORMAT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --long-prefill-token-threshold "${LONG_PREFILL_TOKEN_THRESHOLD}" \
  "${speculative_args[@]}" \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --reasoning-parser glm45 \
  --generation-config "${GENERATION_CONFIG}" \
  --hf-overrides "${HF_OVERRIDES}" \
  "$@"
