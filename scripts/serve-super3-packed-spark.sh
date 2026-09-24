#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-${repo_dir}/.venv/bin/python}
model_revision=${MODEL_REVISION:-d996c54af4ed4825931769bbe3dc69bb318fc37b}
hub_cache=${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}
cached_model=${hub_cache}/models--nvidia--NVIDIA_Super3_5_VL_IQ2XXS-Packed/snapshots/${model_revision}
model_path=${MODEL_PATH:-${cached_model}}
if [[ -z ${MODEL_PATH:-} && ! -d ${cached_model} ]]; then
  model_path=nvidia/NVIDIA_Super3_5_VL_IQ2XXS-Packed
fi

# GB10 settings validated with text, images, prefix caching, and CUDA graphs.
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_121a}
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export PATH="$(dirname -- "${python_bin}"):/usr/local/cuda/bin:${PATH}"
# Reuse the writable caches populated during validation.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vllm-super3-cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/b12x-iq2-triton-cache}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${VLLM_CACHE_ROOT}}

exec "${python_bin}" -m vllm.entrypoints.cli.main serve "${model_path}" \
  --revision "${model_revision}" --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME:-super3-packed}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" \
  --tensor-parallel-size 1 --dtype bfloat16 \
  --load-format b12x --moe-backend b12x --linear-backend b12x \
  --attention-backend B12X --block-size 128 \
  --reasoning-parser nemotron_v3 \
  --mamba-backend triton --mamba-ssm-cache-dtype float32 \
  --gpu-memory-utilization 0.65 \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES:-2147483648}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-256}" \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[1,2,4,8,16]}' \
  "$@"
