#!/usr/bin/env bash
# Correctness-validated Kimi-K3 EXL3-3p09 launcher.
#
# Defaults reproduce the full 93-layer TP12 run that generated the same next
# token as the streamed PyTorch reference. The checkpoint contains serialized
# MXFP8 non-expert weights, so InstantTensor only copies prepared tensors; no
# online weight conversion is involved.
set -euo pipefail

K3_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
K3_PYTHON_BIN="${K3_PYTHON_BIN:-${K3_SCRIPT_DIR}/.venv/bin/python}"
K3_DEFAULT_GPU_UUIDS="GPU-ac6fcbb2-ae5f-231d-cc3e-e843c305baff,GPU-d3f30e71-0df9-2fcc-add2-977c3893288f,GPU-673fde42-acea-c0a9-efb4-04ddc5a5952a,GPU-901b8a05-1c0c-61f6-260b-6a949135ae8f,GPU-a0816187-68b2-b679-587f-0e56bac804f5,GPU-9c204557-77b4-7ffb-c9f2-effcb51d054a,GPU-48d28d14-08f3-f3d1-cb99-20c3fa5eca41,GPU-cfe1f792-1907-1f21-64b7-fdeeb9056425,GPU-f0121aa7-a898-82be-f537-a099d50ef7d8,GPU-afd4b1ad-8a64-7057-4bde-241822724c7f,GPU-4e6952e1-d0fc-03ec-320c-5f76db1275ce,GPU-c7dc46e0-30bb-08e8-2ebb-f164ec57ce31"

json_bool() {
  local name="$1"
  local value="$2"

  case "${value,,}" in
    1|true|yes|on)
      echo true
      ;;
    0|false|no|off)
      echo false
      ;;
    *)
      echo "ERROR: ${name} must be one of 1/0, true/false, yes/no, on/off; got '${value}'" >&2
      exit 1
      ;;
  esac
}

export PYTHONPATH="${K3_SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${K3_DEFAULT_GPU_UUIDS}}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-2097152}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE="${VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE:-134217728}"

# These select the production B12X dense MXFP8 and normal W4A16 MoE kernels.
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-1}"
export KDA_DISABLE_AUTOTUNE="${KDA_DISABLE_AUTOTUNE:-1}"

export INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-AIO}"
export INSTANTTENSOR_MAX_FREE_MEM_USAGE="${INSTANTTENSOR_MAX_FREE_MEM_USAGE:-0.6}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"

# Native dense K3 MLA decode on SM120. Override with TRITON_MLA for an A/B run.
K3_ATTENTION_BACKEND="${K3_ATTENTION_BACKEND:-B12X_MLA}"
K3_MODEL_DIR="${K3_MODEL_DIR:-/models/Kimi-K3-EXL3-3p09-serve}"

# Opt-in Kimi K3 DSpark drafting. The target uses InstantTensor while the
# smaller draft uses fastsafetensors.
K3_DSPARK="${K3_DSPARK:-0}"
K3_DSPARK_ARGS=()
case "${K3_DSPARK,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on)
    K3_DSPARK_MODEL="${K3_DSPARK_MODEL:-/models/Inferact-Kimi-K3-DSpark-MXFP8}"
    K3_DSPARK_TOKENS="${K3_DSPARK_TOKENS:-7}"
    K3_DSPARK_TP_SIZE="${K3_DSPARK_TP_SIZE:-12}"
    K3_DSPARK_ATTENTION_BACKEND="${K3_DSPARK_ATTENTION_BACKEND:-TRITON_MLA}"
    K3_DSPARK_KV_CACHE_DTYPE="${K3_DSPARK_KV_CACHE_DTYPE:-fp8}"
    K3_DSPARK_LOAD_FORMAT="${K3_DSPARK_LOAD_FORMAT:-safetensors}"
    K3_DSPARK_CONFIG="$(
      printf \
        '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_tensor_parallel_size":%s,"attention_backend":"%s","kv_cache_dtype":"%s","draft_load_config":{"load_format":"%s"}}' \
        "${K3_DSPARK_MODEL}" \
        "${K3_DSPARK_TOKENS}" \
        "${K3_DSPARK_TP_SIZE}" \
        "${K3_DSPARK_ATTENTION_BACKEND}" \
        "${K3_DSPARK_KV_CACHE_DTYPE}" \
        "${K3_DSPARK_LOAD_FORMAT}"
    )"
    K3_DSPARK_ARGS+=(--speculative-config "${K3_DSPARK_CONFIG}")
    echo \
      "DSpark enabled with draft: ${K3_DSPARK_MODEL} (${K3_DSPARK_LOAD_FORMAT}, ${K3_DSPARK_ATTENTION_BACKEND}, ${K3_DSPARK_KV_CACHE_DTYPE} KV)" \
      >&2
    ;;
  *)
    echo "ERROR: K3_DSPARK must be one of 1/0, true/false, yes/no, or on/off; got '${K3_DSPARK}'" >&2
    exit 1
    ;;
esac

# Opt-in route-aware calibration from the resident interim EXL3 model. The
# official MXFP4 checkpoint remains the offline encoder's weight source; it is
# intentionally not loaded by this TP12 capture process.
K3_KQUANT_CAPTURE_DIR="${K3_KQUANT_CAPTURE_DIR:-}"
K3_KQUANT_ARGS=(--enable-prefix-caching)
K3_COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
if [[ -n "${K3_KQUANT_CAPTURE_DIR}" ]]; then
  if ((${#K3_DSPARK_ARGS[@]})); then
    echo "ERROR: KQuant calibration does not support a draft model" >&2
    exit 1
  fi
  export SPARKINFER_W4A16_SMALL_M_DIRECT=0
  export VLLM_KQUANT_CAPTURE_DIR="${K3_KQUANT_CAPTURE_DIR}"
  export VLLM_KQUANT_CAPTURE_RUN_ID="${VLLM_KQUANT_CAPTURE_RUN_ID:-$(basename -- "${K3_KQUANT_CAPTURE_DIR}")-$(date +%Y%m%d-%H%M%S)}"
  export VLLM_KQUANT_CORPUS="${K3_KQUANT_CORPUS:-unspecified}"
  export VLLM_KQUANT_SOURCE="interim_exl3_3p09_hybrid"
  export VLLM_KQUANT_TEACHER_CHECKPOINT="${K3_MODEL_DIR}"
  export VLLM_KQUANT_MOMENT_SAMPLE_RATE="${VLLM_KQUANT_MOMENT_SAMPLE_RATE:-16}"
  export VLLM_KQUANT_INPUT_HESSIAN_SAMPLE_RATE="${VLLM_KQUANT_INPUT_HESSIAN_SAMPLE_RATE:-512}"
  export VLLM_KQUANT_MID_HESSIAN_SAMPLE_RATE="${VLLM_KQUANT_MID_HESSIAN_SAMPLE_RATE:-8192}"
  export VLLM_KQUANT_VALIDATION_MODULUS="${VLLM_KQUANT_VALIDATION_MODULUS:-16}"
  export VLLM_KQUANT_SAMPLE_CAPACITY="${VLLM_KQUANT_SAMPLE_CAPACITY:-64}"
  export VLLM_KQUANT_SAMPLE_SAVE_EVERY="${VLLM_KQUANT_SAMPLE_SAVE_EVERY:-32}"
  export VLLM_KQUANT_SAMPLE_FLUSH_BYTES="${VLLM_KQUANT_SAMPLE_FLUSH_BYTES:-268435456}"
  export VLLM_KQUANT_STATS_SAVE_EVERY="${VLLM_KQUANT_STATS_SAVE_EVERY:-128}"
  export VLLM_KQUANT_FINALIZE_FILE="${VLLM_KQUANT_FINALIZE_FILE:-${K3_KQUANT_CAPTURE_DIR}.finalize}"
  K3_KQUANT_ARGS=(--no-enable-prefix-caching)
  # Keep CUDA graph replay but avoid placing capture-only extension launches
  # behind Dynamo during calibration.
  K3_COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1]}'
  echo "KQuant calibration enabled: ${K3_KQUANT_CAPTURE_DIR} (run ${VLLM_KQUANT_CAPTURE_RUN_ID})" >&2
  echo "Finalize with: touch ${VLLM_KQUANT_FINALIZE_FILE}; then send one final request" >&2
fi

# Opt-in full-vocabulary prefill-logit capture for the pinned KLD quality
# suite. This is mutually exclusive with calibration and speculative decode,
# uses small prefill chunks, and leaves the capture hook inactive in every
# ordinary serving process.
K3_KLD_CAPTURE_DIR="${K3_KLD_CAPTURE_DIR:-}"
if [[ -n "${K3_KLD_CAPTURE_DIR}" ]]; then
  if [[ -n "${K3_KQUANT_CAPTURE_DIR}" ]]; then
    echo "ERROR: KLD capture and KQuant calibration are mutually exclusive" >&2
    exit 1
  fi
  if ((${#K3_DSPARK_ARGS[@]})); then
    echo "ERROR: KLD capture does not support a draft model" >&2
    exit 1
  fi
  export VLLM_KLD_CAPTURE_DIR="${K3_KLD_CAPTURE_DIR}"
  K3_KQUANT_ARGS=(--no-enable-prefix-caching)
  K3_COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1]}'
  K3_MAX_NUM_BATCHED_TOKENS="${K3_MAX_NUM_BATCHED_TOKENS:-256}"
  echo "KLD prompt-logit capture enabled: ${K3_KLD_CAPTURE_DIR}" >&2
fi

K3_PROFILE="${K3_PROFILE:-0}"
K3_PROFILER_ARGS=()
case "${K3_PROFILE,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on|torch)
    K3_PROFILE_DIR="${K3_PROFILE_DIR:-/tmp/vllm-profile/kimi-k3-$(date +%Y%m%d-%H%M%S)}"
    K3_TORCH_PROFILER_WITH_STACK_JSON="$(
      json_bool K3_TORCH_PROFILER_WITH_STACK \
        "${K3_TORCH_PROFILER_WITH_STACK:-1}"
    )"
    K3_TORCH_PROFILER_RECORD_SHAPES_JSON="$(
      json_bool K3_TORCH_PROFILER_RECORD_SHAPES \
        "${K3_TORCH_PROFILER_RECORD_SHAPES:-0}"
    )"
    K3_TORCH_PROFILER_WITH_MEMORY_JSON="$(
      json_bool K3_TORCH_PROFILER_WITH_MEMORY \
        "${K3_TORCH_PROFILER_WITH_MEMORY:-0}"
    )"
    K3_TORCH_PROFILER_WITH_FLOPS_JSON="$(
      json_bool K3_TORCH_PROFILER_WITH_FLOPS \
        "${K3_TORCH_PROFILER_WITH_FLOPS:-0}"
    )"
    K3_TORCH_PROFILER_USE_GZIP_JSON="$(
      json_bool K3_TORCH_PROFILER_USE_GZIP \
        "${K3_TORCH_PROFILER_USE_GZIP:-1}"
    )"
    K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON="$(
      json_bool K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL \
        "${K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL:-0}"
    )"
    K3_PROFILE_IGNORE_FRONTEND_JSON="$(
      json_bool K3_PROFILE_IGNORE_FRONTEND \
        "${K3_PROFILE_IGNORE_FRONTEND:-1}"
    )"

    if [[ "${K3_PROFILE_DIR}" != *"://"* ]]; then
      mkdir -p "${K3_PROFILE_DIR}"
    fi
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"
    K3_PROFILER_ARGS+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${K3_PROFILE_DIR}"
      --profiler-config.torch_profiler_with_stack="${K3_TORCH_PROFILER_WITH_STACK_JSON}"
      --profiler-config.torch_profiler_record_shapes="${K3_TORCH_PROFILER_RECORD_SHAPES_JSON}"
      --profiler-config.torch_profiler_with_memory="${K3_TORCH_PROFILER_WITH_MEMORY_JSON}"
      --profiler-config.torch_profiler_with_flops="${K3_TORCH_PROFILER_WITH_FLOPS_JSON}"
      --profiler-config.torch_profiler_use_gzip="${K3_TORCH_PROFILER_USE_GZIP_JSON}"
      --profiler-config.torch_profiler_dump_cuda_time_total="${K3_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL_JSON}"
      --profiler-config.ignore_frontend="${K3_PROFILE_IGNORE_FRONTEND_JSON}"
      --profiler-config.delay_iterations="${K3_PROFILE_DELAY_ITERATIONS:-0}"
      --profiler-config.max_iterations="${K3_PROFILE_MAX_ITERATIONS:-4}"
      --profiler-config.warmup_iterations="${K3_PROFILE_WARMUP_ITERATIONS:-0}"
      --profiler-config.active_iterations="${K3_PROFILE_ACTIVE_ITERATIONS:-5}"
      --profiler-config.wait_iterations="${K3_PROFILE_WAIT_ITERATIONS:-0}"
    )
    echo "Torch profiling enabled. Traces will be written under: ${K3_PROFILE_DIR}" >&2
    ;;
  cuda|nsys|nsight)
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
    K3_PROFILER_ARGS+=(--profiler-config.profiler=cuda)
    echo "CUDA profiler enabled. Use nsys with --capture-range=cudaProfilerApi and drive /start_profile + /stop_profile." >&2
    ;;
  *)
    echo "ERROR: K3_PROFILE must be one of 1/0, true/false, torch, cuda, nsys, or nsight; got '${K3_PROFILE}'" >&2
    exit 1
    ;;
esac

# K3's GDN layers support full CUDA graphs for decode; prefill stays eager.
# FP8 MLA plus the KDA state currently resolves to a 944-token hybrid cache
# block. Keep the scheduler budget at the next power of two so one entire cache
# block always fits in a step. CUDA-graph profiling reserves about 0.11% of an
# RTX PRO 6000; 0.9711 preserves the effective KV budget of the old 0.9700
# setting while still accounting for captured graphs.
exec "${K3_PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${K3_MODEL_DIR}" \
  --served-model-name "${K3_SERVED_MODEL_NAME:-kimi-k3-exl3}" \
  --trust-remote-code \
  --host "${K3_HOST:-0.0.0.0}" \
  --port "${K3_PORT:-8000}" \
  --tensor-parallel-size 12 \
  --load-format instanttensor \
  --linear-backend b12x \
  --attention-backend "${K3_ATTENTION_BACKEND}" \
  --compilation-config "${K3_COMPILATION_CONFIG}" \
  "${K3_KQUANT_ARGS[@]}" \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3 \
  --max-model-len auto \
  --kv-cache-dtype fp8 \
  --block-size "${K3_BLOCK_SIZE:-128}" \
  --gpu-memory-utilization "${K3_GPU_MEMORY_UTILIZATION:-0.9711}" \
  --max-num-batched-tokens "${K3_MAX_NUM_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${K3_MAX_NUM_SEQS:-1}" \
  "${K3_DSPARK_ARGS[@]}" \
  "${K3_PROFILER_ARGS[@]}" \
  "$@"
