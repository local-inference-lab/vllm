#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  echo "Create the venv with: uv venv --python 3.12" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"

export CUDA_HOME="${CUDA_HOME:-${CUDA_PATH:-/opt/cuda}}"
if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${SCRIPT_DIR}/.venv/bin:${PATH}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_120a}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAFETENSORS_FAST_GPU="${SAFETENSORS_FAST_GPU:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export LLM_WORKER_MULTIPROC_METHOD="${LLM_WORKER_MULTIPROC_METHOD:-spawn}"

export VLLM_USE_AOT_COMPILE="${VLLM_USE_AOT_COMPILE:-1}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export VLLM_USE_MEGA_AOT_ARTIFACT="${VLLM_USE_MEGA_AOT_ARTIFACT:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_USE_B12X_FP8_GEMM="${VLLM_USE_B12X_FP8_GEMM:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${VLLM_ENABLE_PCIE_ALLREDUCE:-1}"
export VLLM_PCIE_ALLREDUCE_BACKEND="${VLLM_PCIE_ALLREDUCE_BACKEND:-b12x}"
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE="${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}"
export B12X_MOE_FORCE_A8="${B12X_MOE_FORCE_A8:-0}"
export B12X_MOE_FORCE_A16="${B12X_MOE_FORCE_A16:-0}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-SYS}"
export NCCL_PROTO="${NCCL_PROTO:-LL,LL128,Simple}"

model_path=${MODEL_PATH:-nvidia/Qwen3.6-35B-A3B-NVFP4}
model_revision=${MODEL_REVISION_OVERRIDE:-491c2f1ea524c639598bf8fa787a93fed5a6fbce}
served_model_name=${SERVED_MODEL_NAME:-Qwen3.6-35B-A3B-NVFP4}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp_size=${TP_SIZE:-1}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_seqs=${MAX_NUM_SEQS:-4}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
load_format=${LOAD_FORMAT:-fastsafetensors}
moe_backend=${MOE_BACKEND:-b12x}
linear_backend=${LINEAR_BACKEND:-b12x}
attention_backend=${ATTENTION_BACKEND:-B12X_ATTN}

enable_mtp=0
case "${VLLM_ENABLE_MTP:-1}" in
  0|false|no|off|"") ;;
  1|true|yes|on) enable_mtp=1 ;;
  *)
    echo "VLLM_ENABLE_MTP must be a boolean; got ${VLLM_ENABLE_MTP}" >&2
    exit 2
    ;;
esac

enable_dflash=0
case "${VLLM_ENABLE_DFLASH:-0}" in
  0|false|no|off|"") ;;
  1|true|yes|on) enable_dflash=1 ;;
  *)
    echo "VLLM_ENABLE_DFLASH must be a boolean; got ${VLLM_ENABLE_DFLASH}" >&2
    exit 2
    ;;
esac

if ((enable_mtp + enable_dflash > 1)); then
  echo "VLLM_ENABLE_MTP and VLLM_ENABLE_DFLASH are mutually exclusive" >&2
  exit 2
fi

speculative_args=()
if ((enable_mtp)); then
  num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-3}
  mtp_moe_backend=${QWEN36_MTP_MOE_BACKEND:-flashinfer_cutlass}
  printf -v speculative_config \
    '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s"}' \
    "${num_speculative_tokens}" "${mtp_moe_backend}"
  speculative_args+=(--speculative-config "${speculative_config}")
elif ((enable_dflash)); then
  num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-7}
  dflash_model=${QWEN36_DFLASH_MODEL_PATH:-z-lab/Qwen3.6-35B-A3B-DFlash}
  dflash_revision=${QWEN36_DFLASH_MODEL_REVISION_OVERRIDE:-f181eece646affea2c38b2765f1aaa01a9734ccd}
  dflash_attention_backend=${QWEN36_DFLASH_ATTENTION_BACKEND:-TRITON_ATTN}
  printf -v speculative_config \
    '{"model":"%s","revision":"%s","method":"dflash","num_speculative_tokens":%s,"attention_backend":"%s"}' \
    "${dflash_model}" "${dflash_revision}" "${num_speculative_tokens}" \
    "${dflash_attention_backend}"
  speculative_args+=(--speculative-config "${speculative_config}")
fi

if [[ -n "${num_speculative_tokens:-}" && \
      ! "${num_speculative_tokens}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a positive integer" >&2
  exit 2
fi

profiler_args=()
case "${VLLM_PROFILE:-0}" in
  0|false|no|off) ;;
  1|true|yes|on|torch)
    profile_dir=${VLLM_TORCH_PROFILER_DIR:-/tmp/vllm-profile/qwen36-$(date +%Y%m%d-%H%M%S)}
    [[ "${profile_dir}" == *"://"* ]] || mkdir -p "${profile_dir}"
    profiler_args+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${profile_dir}"
      --profiler-config.torch_profiler_with_stack=true
      --profiler-config.torch_profiler_record_shapes=false
      --profiler-config.torch_profiler_use_gzip=true
      --profiler-config.ignore_frontend=true
      --profiler-config.max_iterations="${VLLM_TORCH_PROFILER_MAX_ITERATIONS:-4}"
      --profiler-config.active_iterations="${VLLM_TORCH_PROFILER_ACTIVE_ITERATIONS:-5}"
    )
    ;;
  *)
    echo "VLLM_PROFILE must be off or torch; got ${VLLM_PROFILE}" >&2
    exit 2
    ;;
esac

unset VLLM_ENABLE_MTP VLLM_ENABLE_DFLASH VLLM_PROFILE

exec "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve \
  "${model_path}" \
  --revision "${model_revision}" \
  --served-model-name "${served_model_name}" \
  --host "${host}" \
  --port "${port}" \
  --trust-remote-code \
  --tensor-parallel-size "${tp_size}" \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --load-format "${load_format}" \
  --quantization modelopt_fp4 \
  --moe-backend "${moe_backend}" \
  --linear-backend "${linear_backend}" \
  --attention-backend "${attention_backend}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max-cudagraph-capture-size "${max_cudagraph_capture_size}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --skip-mm-profiling \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  "${speculative_args[@]}" \
  "${profiler_args[@]}" \
  "$@"
