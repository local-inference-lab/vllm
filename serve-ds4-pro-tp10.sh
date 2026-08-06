#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export CUTE_DSL_ARCH=sm_120a
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export VLLM_USE_AOT_COMPILE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_MEGA_AOT_ARTIFACT=${VLLM_USE_MEGA_AOT_ARTIFACT:-1}
export VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_WO_PROJECTION=1
export VLLM_USE_B12X_MHC=${VLLM_USE_B12X_MHC:-1}
export VLLM_USE_B12X_FP8_GEMM=1
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export B12X_MLA_SM120_UNIFIED=1

export B12X_DENSE_SPLITK_TURBO=1
export B12X_W4A16_TC_DECODE=1

if [[ -z "${HF_HOME:-}" && -L "${HOME}/.cache/huggingface" && ! -e "${HOME}/.cache/huggingface" ]]; then
  if [[ -d /data && -w /data ]]; then
    export HF_HOME="${VLLM_HF_HOME:-/data/vllm-huggingface}"
  else
    export HF_HOME="${VLLM_HF_HOME:-${HOME}/.cache/vllm-huggingface}"
  fi
  mkdir -p "${HF_HOME}"
fi
allocator_conf="${PYTORCH_CUDA_ALLOC_CONF:-}"
if [[ "${allocator_conf}" != *"expandable_segments:"* ]]; then
  allocator_conf="${allocator_conf:+${allocator_conf},}expandable_segments:True"
fi
export PYTORCH_CUDA_ALLOC_CONF="${allocator_conf}"
model_path="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Pro}"
served_model_name="${SERVED_MODEL_NAME:-DeepSeek-V4-Pro}"
tp_size="${TP_SIZE:-10}"
dcp_size="${DCP_SIZE:-1}"
dcp_comm_backend="${DCP_COMM_BACKEND:-ag_rs}"
port="${PORT:-8000}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.94}"
max_model_len="${MAX_MODEL_LEN:-295000}"
max_num_seqs="${MAX_NUM_SEQS:-6}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-2048}"
max_cudagraph_capture_size="${MAX_CUDAGRAPH_CAPTURE_SIZE:-2048}"
cudagraph_mode="${CUDAGRAPH_MODE:-FULL_DECODE_ONLY}"
kv_cache_memory_bytes="${KV_CACHE_MEMORY_BYTES:-3000000000}"
load_format="${LOAD_FORMAT:-instanttensor}"
enable_flashinfer_autotune="${ENABLE_FLASHINFER_AUTOTUNE:-1}"

profiler_args=()
if [[ "${VLLM_ENABLE_TORCH_PROFILER:-0}" == "1" ]]; then
  profile_dir="${VLLM_TORCH_PROFILER_DIR:-/tmp/vllm-ds4-pro-tp10}"
  profile_delay_iterations="${VLLM_TORCH_PROFILER_DELAY_ITERATIONS:-0}"
  profile_max_iterations="${VLLM_TORCH_PROFILER_MAX_ITERATIONS:-4}"
  profile_with_stack="${VLLM_TORCH_PROFILER_WITH_STACK:-true}"
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

spec_args=()
if [[ "${VLLM_ENABLE_MTP:-1}" == "1" ]]; then
  spec_args=('--speculative-config' '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":true}')
fi

autotune_args=()
if [[ "${enable_flashinfer_autotune}" == "1" ]]; then
  autotune_args+=(--enable-flashinfer-autotune)
else
  autotune_args+=(--no-enable-flashinfer-autotune)
fi

kv_cache_args=()
if [[ "${kv_cache_memory_bytes}" != "auto" && "${kv_cache_memory_bytes}" != "0" ]]; then
  kv_cache_args+=(--kv-cache-memory-bytes "${kv_cache_memory_bytes}")
fi

dcp_args=(--decode-context-parallel-size "${dcp_size}")
if (( dcp_size > 1 )); then
  dcp_args+=(--dcp-comm-backend "${dcp_comm_backend}")
fi

exec .venv/bin/python -m vllm.entrypoints.cli.main serve \
  "${model_path}" \
  --served-model-name "${served_model_name}" \
  --host 0.0.0.0 \
  --port "${port}" \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --load-format "${load_format}" \
  --tensor-parallel-size "${tp_size}" \
  "${dcp_args[@]}" \
  --moe-backend b12x \
  --linear-backend b12x \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  "${kv_cache_args[@]}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max_cudagraph_capture_size "${max_cudagraph_capture_size}" \
  --attention-backend B12X_MLA_SPARSE \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"'"${cudagraph_mode}"'","custom_ops":["all"]}' \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  "${autotune_args[@]}" \
  "${profiler_args[@]}" \
  "${spec_args[@]}" \
  "$@"
