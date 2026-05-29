#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=7,8
export NCCL_IB_DISABLE=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_MHC=0
export B12X_W4A16_TC_DECODE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1

profiler_args=()
if [[ "${VLLM_ENABLE_TORCH_PROFILER:-0}" == "1" ]]; then
  profile_dir="${VLLM_TORCH_PROFILER_DIR:-/tmp/vllm-ds4-decode}"
  profile_delay_iterations="${VLLM_TORCH_PROFILER_DELAY_ITERATIONS:-0}"
  profile_max_iterations="${VLLM_TORCH_PROFILER_MAX_ITERATIONS:-4}"
  profile_with_stack="${VLLM_TORCH_PROFILER_WITH_STACK:-false}"
  profile_record_shapes="${VLLM_TORCH_PROFILER_RECORD_SHAPES:-false}"
  profile_with_memory="${VLLM_TORCH_PROFILER_WITH_MEMORY:-false}"
  profile_use_gzip="${VLLM_TORCH_PROFILER_USE_GZIP:-true}"

  profiler_config=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_with_stack":%s,"torch_profiler_record_shapes":%s,"torch_profiler_with_memory":%s,"torch_profiler_use_gzip":%s,"ignore_frontend":true,"delay_iterations":%s,"max_iterations":%s}' \
    "${profile_dir}" \
    "${profile_with_stack}" \
    "${profile_record_shapes}" \
    "${profile_with_memory}" \
    "${profile_use_gzip}" \
    "${profile_delay_iterations}" \
    "${profile_max_iterations}")
  profiler_args+=(--profiler-config "${profiler_config}")
  echo "Torch profiler enabled: dir=${profile_dir} delay_iterations=${profile_delay_iterations} max_iterations=${profile_max_iterations}"
fi

exec .venv/bin/python -m vllm.entrypoints.cli.main serve \
  deepseek-ai/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 8000 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 65536 \  # just for testing perf/debugging
  --load-format fastsafetensors \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 2 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --max_cudagraph_capture_size 2048 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x"}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  "${profiler_args[@]}" \
  "$@"
