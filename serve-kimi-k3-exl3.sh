#!/usr/bin/env bash
# Kimi-K3 keep+EXL3-3.0 (kquant Phase B) at TP=12, eager bring-up.
# Uses the packaged checkpoint's serialized MXFP8 non-expert weights; no
# online quantization overlay is involved.
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
export VLLM_ENABLE_PCIE_ALLREDUCE=0
export KDA_DISABLE_AUTOTUNE=1
# Refactored b12x defaults K3 SiTU MXFP4 experts to w4a8_mx at TP12 (local
# I=256 divides 128); the hybrid kept-tier launches are built for w4a16.
# Revisit W4A8 during the perf pass.
export B12X_MOE_FORCE_A16=1
# Route serialized MXFP8 dense linears through sparkinfer B12X GEMM (flashinfer
# mm_mxfp8 does not exist in this build) - from Martin's hard-won constraints.
export VLLM_USE_B12X_FP8_GEMM=1
# First-request decode-shape CuTe compiles can exceed the 300s default;
# they disk-cache after the first run.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=134217728
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# EXL3 skeleton replicates suh/svh sign vectors per rank (~1.4 GiB), leaving
# little free memory at load start; staging buffers are freed after load.
export INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.6
export INSTANTTENSOR_BACKEND=AIO

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${MODEL_DIR:-/models/Kimi-K3-EXL3-3p14-serve}" \
  --served-model-name kimi-k3-exl3 \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port "${PORT:-8011}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --tensor-parallel-size 12 \
  --enforce-eager \
  --load-format "${LOAD_FORMAT:-instanttensor}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.985}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  \
  "$@"
