#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-voipmonitor/vllm:jovian-judgement-community-20260901-r12}"
MODEL_PATH="${MODEL_PATH:-/data/models/qwen3.8-flash-next-mixed/qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1}"

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
  --served-model-name qwen3.8-flash-next-4p89bpw \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --mm-encoder-tp-mode data \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --quantization modelopt_mixed \
  --block-size 16 \
  --load-format fastsafetensors \
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
