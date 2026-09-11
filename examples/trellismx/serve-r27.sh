#!/usr/bin/env bash
set -euo pipefail

# TrellisMX r27 DCP4 serving defaults matching the September 9 selected reference.
export VLLM_TRELLISMX_CHECKPOINT=${VLLM_TRELLISMX_CHECKPOINT:-/checkpoint}
carrier=${MODEL_ROOT:-/model}
port=${PORT:-8000}
if ! [[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] || ((port > 65535)); then
  echo 'Choose a valid production PORT' >&2
  exit 2
fi
test -f "$carrier/config.json"
test -f "$VLLM_TRELLISMX_CHECKPOINT/trellismx-manifest.json"
if [[ -n ${GLM53_P8_NATIVE:-} || -n ${GLM53_P8_PSEUDOQUANT:-} ]]; then
  echo 'Remove legacy P8 sitecustomize activation variables' >&2
  exit 2
fi

# Resolve the DCP4 contract before delegating backend selection, cache geometry,
# scheduler budget, PCIe policy, and graph mode to r27's qualified launcher.
tp=${TP:-4}
dcp=${DCP:-4}
if [[ "$tp" != 4 || "$dcp" != 4 ]]; then
  echo "This launcher requires TP=4 and DCP=4; got TP=$tp DCP=$dcp" >&2
  exit 2
fi
if ((dcp != 4)); then
  echo 'This candidate launcher requires DCP=4; use the DCP1 review recipe otherwise' >&2
  exit 2
fi

export TP="$tp"
export DCP="$dcp"
export CACHE_MODE=${CACHE_MODE:-vram}
export SPECULATOR=${SPECULATOR:-mtp}
export NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-3}
export KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-nvfp4_ds_mla}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-1000000}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-24}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4096}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.97}
export CP_KV_CACHE_INTERLEAVE_SIZE=${CP_KV_CACHE_INTERLEAVE_SIZE:-4}
export DCP_CKV_GATHER=${DCP_CKV_GATHER:-auto}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm53-flash-trellismx-p8-k45}
export PORT="$port"
export MODEL_ROOT="$carrier"

# r27's launcher owns these values so this overlay cannot silently regress its
# qualified B12X, KDA, scheduler, cache, or graph selections.
export NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8}
export NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-8}
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-131072}
export VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE=${VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE:-86016}
export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=${VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD:-4096}

exec /usr/local/bin/serve-glm53-flash.sh "$carrier" "$@"
