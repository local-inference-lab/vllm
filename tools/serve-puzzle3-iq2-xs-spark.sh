#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-${repo_dir}/.venv/bin/python}
model_path=${MODEL_PATH:?Set MODEL_PATH to the Puzzle 3 checkpoint directory}

export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_121a}
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname -- "${python_bin}"):/usr/local/cuda/bin:${PATH}"
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${repo_dir}/.runtime/vllm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${repo_dir}/.runtime/triton}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${repo_dir}/.runtime}

exec "${python_bin}" -m vllm.entrypoints.cli.main serve "${model_path}" \
  --served-model-name "${SERVED_MODEL_NAME:-puzzle3-iq2-xs}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --load-format b12x --moe-backend b12x --linear-backend b12x \
  --attention-backend B12X --block-size 128 \
  --reasoning-parser nemotron_v3 \
  --mamba-backend triton --mamba-ssm-cache-dtype float32 \
  --gpu-memory-utilization 0.65 --kv-cache-memory-bytes 4294967296 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 512 \
  --enable-chunked-prefill --enable-prefix-caching \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/vllm-ds4-decode","max_iterations":4,"ignore_frontend":true}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16]}' \
  "$@"
