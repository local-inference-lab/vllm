#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4.1-Flash on one GB300 (SM103) with b12x expert residency.
# Routed experts that do not fit in HBM are served from coherent Grace memory;
# "expert residency" is a working name. Text only: the b12x vision attention
# has no SM103 kernel. Preview without loading the model: DRY_RUN=1 ./serve-ds41-flash-gb300.sh
# HOST, PORT, MODEL_PATH and the capacity variables below may be overridden.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/fb2764a5cf321eaa5070ca8f9e892818f477c16d}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4.1-Flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# Each MoE layer owns a private workspace sized by this capacity (about
# 0.39 GiB per layer per 1024 tokens), which competes with hot experts for HBM.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-64}"
# The KV cache is reserved before experts are placed in HBM. V4.1's compressed
# cache is small: 8 GiB holds several million tokens at block size 256.
KV_CACHE_GB="${KV_CACHE_GB:-8}"
# HBM left free for activations, CUDA graphs and other prepared operators.
HBM_RESERVE_GB="${HBM_RESERVE_GB:-12}"
# GB300 HBM appears as a CPU-less NUMA node. Bind host allocations and the page
# cache to the Grace node so reading the checkpoint cannot occupy HBM that the
# expert placement counts on. Set HOST_NUMA_NODE= (empty) to disable.
HOST_NUMA_NODE="${HOST_NUMA_NODE-0}"
# Optional b12x placement profile; without one, placement is budget-balanced.
RESIDENCY_PROFILE="${RESIDENCY_PROFILE:-}"
ENGRAM_TABLE_MEMORY="${ENGRAM_TABLE_MEMORY:-disk}"
# DSpark draft experts stay entirely in HBM.
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_103a}"
export VLLM_B12X_SM103=1
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset VLLM_ALLOW_INSECURE_SERIALIZATION

kv_cache_bytes=$(awk -v gb="${KV_CACHE_GB}" 'BEGIN { printf "%d", gb * 1073741824 }')
residency_config=$(printf '{"hbm_reserve_gb":%s' "${HBM_RESERVE_GB}")
if [[ -n "${RESIDENCY_PROFILE}" ]]; then
  residency_config+=$(printf ',"profile_path":"%s"' "${RESIDENCY_PROFILE}")
fi
residency_config+='}'
engram_config=$(printf '{"cpu_offload":false,"table_memory":"%s"}' "${ENGRAM_TABLE_MEMORY}")
compilation_config='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'

speculative_args=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  speculative_args=(--speculative-config "$(printf \
    '{"method":"dspark","num_speculative_tokens":%s,"attention_backend":"B12X","draft_sample_method":"greedy","rejection_sample_method":"standard"}' \
    "${NUM_SPECULATIVE_TOKENS}")")
fi

numa_prefix=()
if [[ -n "${HOST_NUMA_NODE}" ]] && command -v numactl >/dev/null; then
  numa_prefix=(numactl --membind="${HOST_NUMA_NODE}")
fi

command=(
  "${numa_prefix[@]}"
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}" --port "${PORT}"
  --dtype bfloat16
  --tensor-parallel-size 1
  --language-model-only
  --load-format safetensors
  --block-size "${BLOCK_SIZE}"
  --kv-cache-memory-bytes "${kv_cache_bytes}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  --compilation-config "${compilation_config}"
  --generation-config vllm
  --engram-config "${engram_config}"
  --expert-residency-config "${residency_config}"
  --attention-backend B12X
  --linear-backend b12x
  --moe-backend b12x
  "${speculative_args[@]}"
  --tokenizer-mode deepseek_v41
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41
  --enable-auto-tool-choice
  "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
