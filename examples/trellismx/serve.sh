#!/usr/bin/env bash
set -euo pipefail
export VLLM_TRELLISMX_CHECKPOINT=${VLLM_TRELLISMX_CHECKPOINT:-/checkpoint}
carrier=${MODEL_ROOT:-/model}
port=${PORT:-8033}
length=${MAX_MODEL_LEN:-1000000}
if ! [[ "$length" =~ ^[1-9][0-9]{0,6}$ ]] || ((length > 1048576)); then
  echo 'MAX_MODEL_LEN must be in 1..1048576' >&2; exit 2;
fi
if ! [[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] || ((port > 65535 || port == 8000)); then
  echo 'Choose a valid non-production PORT (not 8000)' >&2; exit 2;
fi
test -f "$carrier/config.json"
test -f "$VLLM_TRELLISMX_CHECKPOINT/trellismx-manifest.json"
# The integration is explicit in ModelOpt, not a sitecustomize monkey patch.
if [[ -n ${GLM53_P8_NATIVE:-} || -n ${GLM53_P8_PSEUDOQUANT:-} ]]; then
  echo 'Remove legacy P8 sitecustomize activation variables' >&2; exit 2
fi
export VLLM_ENABLE_PCIE_ALLREDUCE=1 VLLM_PCIE_ALLREDUCE_BACKEND=b12x
exec vllm serve "$carrier" \
  --served-model-name GLM-5.3-Flash-TrellisMX-MXFP8 \
  --host 0.0.0.0 --port "$port" --language-model-only \
  --tensor-parallel-size 4 --decode-context-parallel-size 1 \
  --dtype bfloat16 --attention-backend B12X_MLA_SPARSE \
  --kv-cache-dtype nvfp4_ds_mla --max-model-len "$length" \
  --max-num-batched-tokens 8192 --max-num-seqs 32 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.975}" \
  --enable-chunked-prefill --enable-prefix-caching \
  --generation-config "$carrier" --reasoning-parser glm45 \
  --quantization modelopt "$@"
