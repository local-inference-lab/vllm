#!/usr/bin/env bash
# Kimi-K3 uniform NF3-refit (kquant artifact) bring-up: TP=12, eager, MXFP8
# online overlay on attention + shared experts, all else BF16, router
# replicated (TP-sharded router is a planned follow-up).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_PROTO=LL,LL128,Simple
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=8
export OMP_NUM_THREADS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_B12X_MOE=1
export VLLM_USE_V2_MODEL_RUNNER=1
# PCIe oneshot allreduce supports world sizes (2,4,6,8,10); at 12 use NCCL.
export VLLM_ENABLE_PCIE_ALLREDUCE=0
export KDA_DISABLE_AUTOTUNE=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.1
export INSTANTTENSOR_BACKEND=AIO

OVERLAY='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj","re:.*conv1d","re:.*\\.\\d*[02468]\\.self_attn\\.(g_proj|f_a_proj|f_b_proj)","re:.*\\.b_proj","re:.*lm_head","re:.*attn_res"]}'

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  /models/Kimi-K3-NF3R-Uniform-3p25-serve \
  --served-model-name kimi-k3-nf3r \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "${PORT:-8011}" \
  --max-model-len "${MAX_MODEL_LEN:-128}" \
  --tensor-parallel-size 6 \
  --pipeline-parallel-size 2 \
  --enforce-eager \
  --load-format instanttensor \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.985}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-128}" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" \
  --quantization-config "${OVERLAY}" \
  "$@"
