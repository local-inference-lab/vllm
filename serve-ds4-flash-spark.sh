#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
source tools/spark/versions.env

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${PWD}/.venv/bin:${PATH}"
export TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-"${CUDA_HOME}/bin/ptxas"}
export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_121a}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export HF_HOME=${SPARK_HF_HOME:-"${HOME}/.cache/vllm-huggingface"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

export DG_JIT_USE_NVRTC=0
export USE_CUDNN=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_AOT_COMPILE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_USE_MEGA_AOT_ARTIFACT=${VLLM_USE_MEGA_AOT_ARTIFACT:-1}
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}
export VLLM_USE_FLASHINFER_SAMPLER=1
export VLLM_USE_B12X_WO_PROJECTION=1
export VLLM_USE_B12X_MHC=1
export VLLM_USE_B12X_FP8_GEMM=1
export VLLM_USE_B12X_MOE=1
export VLLM_USE_B12X_SPARSE_INDEXER=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export B12X_MLA_SM120_UNIFIED=1
export B12X_DENSE_SPLITK_TURBO=1
export B12X_W4A16_TC_DECODE=1
export B12X_MOE_FORCE_A8=1

json_bool() {
  local name=$1
  local value=$2
  case "${value,,}" in
    1|true|yes|on) printf 'true\n' ;;
    0|false|no|off) printf 'false\n' ;;
    *)
      printf '%s must be a boolean; got %s\n' "${name}" "${value}" >&2
      exit 2
      ;;
  esac
}

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f1,roceP2p1s0f1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f1np1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp1s0f1np1}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-enp1s0f1np1}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp1s0f1np1}
export NCCL_IGNORE_CPU_AFFINITY=${NCCL_IGNORE_CPU_AFFINITY:-1}

nccl_dir="${PWD}/.spark-artifacts/nccl/${NCCL_REF}/lib"
if [[ -e "${nccl_dir}/libnccl.so.2" ]]; then
  export VLLM_NCCL_SO_PATH="${nccl_dir}/libnccl.so.2"
  export LD_LIBRARY_PATH="${nccl_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LD_PRELOAD="${nccl_dir}/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

enable_mtp=${VLLM_ENABLE_MTP:-0}
enable_dspark=${VLLM_ENABLE_DSPARK:-0}
if [[ "${enable_dspark}" == 1 ]]; then
  model_path=${MODEL_PATH:-${DSPARK_MODEL_ID}}
  model_revision=${MODEL_REVISION_OVERRIDE:-${DSPARK_MODEL_REVISION}}
else
  model_path=${MODEL_PATH:-${MODEL_ID}}
  model_revision=${MODEL_REVISION_OVERRIDE:-${MODEL_REVISION}}
fi
served_model_name=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}
nnodes=${NNODES:-2}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-192.168.177.11}
master_port=${MASTER_PORT:-29501}
tp_size=${TP_SIZE:-${nnodes}}
dcp_size=${DCP_SIZE:-1}
dcp_comm_backend=${DCP_COMM_BACKEND:-a2a}
port=${PORT:-8000}
if [[ ${GPU_MEMORY_UTILIZATION+x} ]]; then
  gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
elif [[ "${enable_dspark}" == 1 ]]; then
  gpu_memory_utilization=${SPARK_DSPARK_GPU_MEMORY_UTILIZATION:-0.80}
else
  gpu_memory_utilization=0.80
fi
if [[ ${KV_CACHE_MEMORY_BYTES+x} ]]; then
  kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}
elif [[ "${enable_dspark}" == 1 ]]; then
  kv_cache_memory_bytes=${SPARK_DSPARK_KV_CACHE_MEMORY_BYTES:-}
else
  kv_cache_memory_bytes=${SPARK_KV_CACHE_MEMORY_BYTES:-}
fi
max_model_len=${MAX_MODEL_LEN:-500000}
max_num_seqs=${MAX_NUM_SEQS:-4}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
if [[ ${NUM_SPECULATIVE_TOKENS+x} ]]; then
  num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}
elif [[ "${enable_dspark}" == 1 ]]; then
  num_speculative_tokens=5
else
  num_speculative_tokens=2
fi
if [[ ${MAX_CUDAGRAPH_CAPTURE_SIZE+x} ]]; then
  max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE}
elif [[ "${enable_dspark}" == 1 ]]; then
  max_cudagraph_capture_size=$((max_num_seqs * (num_speculative_tokens + 1)))
else
  max_cudagraph_capture_size=256
fi
load_format=${LOAD_FORMAT:-instanttensor}
enable_flashinfer_autotune=${ENABLE_FLASHINFER_AUTOTUNE:-1}
enable_prefix_caching=${ENABLE_PREFIX_CACHING:-1}
dspark_draft_attention_backend=${DSPARK_DRAFT_ATTENTION_BACKEND:-B12X_MLA_SPARSE}
# DSpark confidence verification gating (all inert when empty). SPS curve is
# either the string "auto" or a raw JSON breakpoint list like [[8,900],[32,600]].
dspark_sps_curve=${DSPARK_SPS_CURVE:-}
dspark_confidence_threshold=${DSPARK_CONFIDENCE_THRESHOLD:-}
dspark_budget_frac=${DSPARK_BUDGET_FRAC:-}
dspark_confidence_temperature=${DSPARK_CONFIDENCE_TEMPERATURE:-}
dspark_sps_overhead_ms=${DSPARK_SPS_OVERHEAD_MS:-}
unset VLLM_ENABLE_MTP VLLM_ENABLE_DSPARK

export VLLM_USE_B12X_DCP_A2A=${VLLM_USE_B12X_DCP_A2A:-0}
export VLLM_DCP_A2A_MAX_TOKENS=${VLLM_DCP_A2A_MAX_TOKENS:-64}
export VLLM_DCP_A2A_LARGE_BACKEND=${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}

if [[ "${enable_mtp}" != 0 && "${enable_mtp}" != 1 ]]; then
  printf 'VLLM_ENABLE_MTP must be 0 or 1\n' >&2
  exit 2
fi
if [[ "${enable_dspark}" != 0 && "${enable_dspark}" != 1 ]]; then
  printf 'VLLM_ENABLE_DSPARK must be 0 or 1\n' >&2
  exit 2
fi
if [[ "${enable_mtp}" == 1 && "${enable_dspark}" == 1 ]]; then
  printf 'VLLM_ENABLE_MTP and VLLM_ENABLE_DSPARK are mutually exclusive\n' >&2
  exit 2
fi
if [[ "${enable_prefix_caching}" != 0 && "${enable_prefix_caching}" != 1 ]]; then
  printf 'ENABLE_PREFIX_CACHING must be 0 or 1\n' >&2
  exit 2
fi
if [[ -n "${kv_cache_memory_bytes}" && \
      ! "${kv_cache_memory_bytes}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'KV_CACHE_MEMORY_BYTES must be a positive integer\n' >&2
  exit 2
fi
if [[ ! "${num_speculative_tokens}" =~ ^[0-9]+$ ]]; then
  printf 'NUM_SPECULATIVE_TOKENS must be a non-negative integer\n' >&2
  exit 2
fi
if [[ "${enable_dspark}" == 1 ]]; then
  case "${dspark_draft_attention_backend}" in
    auto|B12X_MLA_SPARSE|FLASHINFER_MLA_SPARSE_DSV4|FLASHMLA_SPARSE_DSV4) ;;
    *)
      printf '%s\n' \
        'DSPARK_DRAFT_ATTENTION_BACKEND must be auto, B12X_MLA_SPARSE,' \
        'FLASHINFER_MLA_SPARSE_DSV4, or FLASHMLA_SPARSE_DSV4' >&2
      exit 2
      ;;
  esac
fi
if [[ ! "${dcp_size}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'DCP_SIZE must be a positive integer\n' >&2
  exit 2
fi
if [[ "${enable_dspark}" == 1 && "${dcp_size}" != 1 ]]; then
  printf 'DeepSeek V4 Flash DSpark requires DCP_SIZE=1\n' >&2
  exit 2
fi
if [[ "${dcp_comm_backend}" != a2a && "${dcp_comm_backend}" != ag_rs ]]; then
  printf 'DCP_COMM_BACKEND must be a2a or ag_rs\n' >&2
  exit 2
fi
if [[ "${VLLM_USE_B12X_DCP_A2A}" != 0 && \
      "${VLLM_USE_B12X_DCP_A2A}" != 1 ]]; then
  printf 'VLLM_USE_B12X_DCP_A2A must be 0 or 1\n' >&2
  exit 2
fi
if [[ ! "${VLLM_DCP_A2A_MAX_TOKENS}" =~ ^[0-9]+$ ]]; then
  printf 'VLLM_DCP_A2A_MAX_TOKENS must be a non-negative integer\n' >&2
  exit 2
fi
if [[ "${VLLM_DCP_A2A_LARGE_BACKEND}" != a2a && \
      "${VLLM_DCP_A2A_LARGE_BACKEND}" != ag_rs ]]; then
  printf 'VLLM_DCP_A2A_LARGE_BACKEND must be a2a or ag_rs\n' >&2
  exit 2
fi

headless_args=()
if [[ "${HEADLESS:-0}" == 1 || "${node_rank}" -ne 0 ]]; then
  headless_args+=(--headless)
fi

autotune_args=(--no-enable-flashinfer-autotune)
if [[ "${enable_flashinfer_autotune}" == 1 ]]; then
  autotune_args=(--enable-flashinfer-autotune)
fi

prefix_caching_args=(--no-enable-prefix-caching)
if [[ "${enable_prefix_caching}" == 1 ]]; then
  prefix_caching_args=(--enable-prefix-caching)
fi

memory_args=(--gpu-memory-utilization "${gpu_memory_utilization}")
if [[ -n "${kv_cache_memory_bytes}" ]]; then
  memory_args+=(--kv-cache-memory-bytes "${kv_cache_memory_bytes}")
fi

speculative_args=()
if [[ "${enable_dspark}" == 1 ]] && ((num_speculative_tokens > 0)); then
  draft_attention_json=
  if [[ "${dspark_draft_attention_backend}" != auto ]]; then
    draft_attention_json=$(printf \
      ',"attention_backend":"%s"' \
      "${dspark_draft_attention_backend}")
  fi
  gating_json=
  if [[ -n "${dspark_sps_curve}" ]]; then
    if [[ "${dspark_sps_curve}" == auto ]]; then
      gating_json+=',"dspark_sps_curve":"auto"'
    else
      gating_json+=",\"dspark_sps_curve\":${dspark_sps_curve}"
    fi
  fi
  if [[ -n "${dspark_confidence_threshold}" ]]; then
    gating_json+=",\"dspark_confidence_threshold\":${dspark_confidence_threshold}"
  fi
  if [[ -n "${dspark_budget_frac}" ]]; then
    gating_json+=",\"dspark_budget_frac\":${dspark_budget_frac}"
  fi
  if [[ -n "${dspark_confidence_temperature}" ]]; then
    gating_json+=",\"dspark_confidence_temperature\":${dspark_confidence_temperature}"
  fi
  if [[ -n "${dspark_sps_overhead_ms}" ]]; then
    gating_json+=",\"dspark_sps_overhead_ms\":${dspark_sps_overhead_ms}"
  fi
  speculative_config=$(printf \
    '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"probabilistic"%s%s}' \
    "${num_speculative_tokens}" "${draft_attention_json}" "${gating_json}")
  speculative_args+=(
    --speculative-config
    "${speculative_config}"
  )
elif [[ "${enable_mtp}" == 1 ]] && ((num_speculative_tokens > 0)); then
  speculative_config=$(printf \
    '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"probabilistic","moe_backend":"b12x"}' \
    "${num_speculative_tokens}")
  speculative_args+=(
    --speculative-config
    "${speculative_config}"
  )
fi

profiler_args=()
profile_mode=${VLLM_PROFILE:-${VLLM_ENABLE_TORCH_PROFILER:-0}}
case "${profile_mode,,}" in
  0|false|no|off|"")
    ;;
  1|true|yes|on|torch)
    profile_dir=${VLLM_TORCH_PROFILER_DIR:-/tmp/vllm-profile/ds4-$(date +%Y%m%d-%H%M%S)}
    if [[ "${profile_dir}" != *"://"* ]]; then
      mkdir -p "${profile_dir}"
    fi
    profile_with_stack=$(json_bool VLLM_TORCH_PROFILER_WITH_STACK \
      "${VLLM_TORCH_PROFILER_WITH_STACK:-1}")
    profile_record_shapes=$(json_bool VLLM_TORCH_PROFILER_RECORD_SHAPES \
      "${VLLM_TORCH_PROFILER_RECORD_SHAPES:-0}")
    profile_with_memory=$(json_bool VLLM_TORCH_PROFILER_WITH_MEMORY \
      "${VLLM_TORCH_PROFILER_WITH_MEMORY:-0}")
    profile_with_flops=$(json_bool VLLM_TORCH_PROFILER_WITH_FLOPS \
      "${VLLM_TORCH_PROFILER_WITH_FLOPS:-0}")
    profile_use_gzip=$(json_bool VLLM_TORCH_PROFILER_USE_GZIP \
      "${VLLM_TORCH_PROFILER_USE_GZIP:-1}")
    profile_dump_cuda_time=$(json_bool \
      VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL \
      "${VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL:-0}")
    profile_ignore_frontend=$(json_bool VLLM_PROFILE_IGNORE_FRONTEND \
      "${VLLM_PROFILE_IGNORE_FRONTEND:-1}")
    profiler_args+=(
      --profiler-config.profiler=torch
      --profiler-config.torch_profiler_dir="${profile_dir}"
      --profiler-config.torch_profiler_with_stack="${profile_with_stack}"
      --profiler-config.torch_profiler_record_shapes="${profile_record_shapes}"
      --profiler-config.torch_profiler_with_memory="${profile_with_memory}"
      --profiler-config.torch_profiler_with_flops="${profile_with_flops}"
      --profiler-config.torch_profiler_use_gzip="${profile_use_gzip}"
      --profiler-config.torch_profiler_dump_cuda_time_total="${profile_dump_cuda_time}"
      --profiler-config.ignore_frontend="${profile_ignore_frontend}"
      --profiler-config.delay_iterations="${VLLM_TORCH_PROFILER_DELAY_ITERATIONS:-0}"
      --profiler-config.max_iterations="${VLLM_TORCH_PROFILER_MAX_ITERATIONS:-4}"
      --profiler-config.warmup_iterations="${VLLM_TORCH_PROFILER_WARMUP_ITERATIONS:-0}"
      --profiler-config.active_iterations="${VLLM_TORCH_PROFILER_ACTIVE_ITERATIONS:-5}"
      --profiler-config.wait_iterations="${VLLM_TORCH_PROFILER_WAIT_ITERATIONS:-0}"
    )
    printf 'Torch profiling enabled; traces: %s\n' "${profile_dir}" >&2
    ;;
  *)
    printf 'VLLM_PROFILE must be off or torch; got %s\n' "${profile_mode}" >&2
    exit 2
    ;;
esac
unset VLLM_PROFILE VLLM_ENABLE_TORCH_PROFILER VLLM_TORCH_PROFILER_DIR
unset VLLM_TORCH_PROFILER_WITH_STACK VLLM_TORCH_PROFILER_RECORD_SHAPES
unset VLLM_TORCH_PROFILER_WITH_MEMORY VLLM_TORCH_PROFILER_WITH_FLOPS
unset VLLM_TORCH_PROFILER_USE_GZIP
unset VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL
unset VLLM_PROFILE_IGNORE_FRONTEND VLLM_TORCH_PROFILER_DELAY_ITERATIONS
unset VLLM_TORCH_PROFILER_MAX_ITERATIONS
unset VLLM_TORCH_PROFILER_WARMUP_ITERATIONS
unset VLLM_TORCH_PROFILER_ACTIVE_ITERATIONS
unset VLLM_TORCH_PROFILER_WAIT_ITERATIONS

compilation_extra=
if [[ "${enable_dspark}" == 1 ]]; then
  # Dense verify-graph buckets: every depth 1..decode_query_len must be a
  # distinctly captured (and profiled) batch size, else capacity choices pad
  # up a coarse pow2 grid and mid-depth pruning is mispriced.
  dense_sizes=$(python3 - "$((num_speculative_tokens + 1))" "${max_cudagraph_capture_size}" <<'PYEOF'
import sys
depth, cap = int(sys.argv[1]), int(sys.argv[2])
sizes = sorted(set(list(range(1, min(depth, cap) + 1)) + list(range(depth, cap + 1, 4)) + [cap]))
print(",".join(str(x) for x in sizes))
PYEOF
)
  compilation_extra=$(printf ',"cudagraph_capture_sizes":[%s]' "${dense_sizes}")
fi

exec .venv/bin/python -m vllm.entrypoints.cli.main serve \
  "${model_path}" \
  --revision "${model_revision}" \
  --served-model-name "${served_model_name}" \
  --host 0.0.0.0 \
  --port "${port}" \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --nnodes "${nnodes}" \
  --node-rank "${node_rank}" \
  --master-addr "${master_addr}" \
  --master-port "${master_port}" \
  --tensor-parallel-size "${tp_size}" \
  --decode-context-parallel-size "${dcp_size}" \
  --dcp-comm-backend "${dcp_comm_backend}" \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --load-format "${load_format}" \
  --moe-backend b12x \
  --linear-backend b12x \
  "${memory_args[@]}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs "${max_num_seqs}" \
  --max-num-batched-tokens "${max_num_batched_tokens}" \
  --max-cudagraph-capture-size "${max_cudagraph_capture_size}" \
  --attention-backend "${ATTN_BACKEND:-B12X_MLA_SPARSE}" \
  --async-scheduling \
  --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill \
  "${prefix_caching_args[@]}" \
  --compilation-config \
    "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"custom_ops\":[\"all\"]${compilation_extra}}" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config \
    '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high \
  "${speculative_args[@]}" \
  "${autotune_args[@]}" \
  "${profiler_args[@]}" \
  "${headless_args[@]}" \
  "$@"
