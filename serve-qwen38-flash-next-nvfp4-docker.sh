#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:jovian-judgement-community-20260901-r12}"
MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD}"

exec docker run --rm \
  --network host \
  --ipc host \
  --gpus '"device=0,1"' \
  --mount "type=bind,src=${MODEL_PATH},dst=/model,readonly" \
  --mount type=bind,src=/cache,dst=/cache \
  --workdir /opt/glm53-flash/vllm \
  --env VLLM_SSM_CONV_STATE_LAYOUT=DS \
  --env CUTE_DSL_ARCH=sm_120a \
  --env NCCL_IB_DISABLE=1 \
  --env VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  --env VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --env SAFETENSORS_FAST_GPU=1 \
  --entrypoint /opt/venv/bin/python \
  "${IMAGE}" \
  -m vllm.entrypoints.cli.main serve /model \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --block-size 16 \
  --load-format b12x \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --gdn-decode-kernel b12x \
  --linear-backend b12x \
  --moe-backend b12x \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  "$@"
