#!/usr/bin/env bash
set -euo pipefail

# DeepSeek-V4-Flash / DSpark launcher for the SM120 PCIe stack. The public
# interface is environment-only so the same command can be used from Compose,
# docker run, and benchmark automation.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=${PYTHON_BIN:-"${SCRIPT_DIR}/.venv/bin/python"}
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS

bool_value() {
  local name=$1 value=${2,,}
  case "${value}" in
    1|true|yes|on) printf '1\n' ;;
    0|false|no|off) printf '0\n' ;;
    *)
      echo "${name} must be 1/0, true/false, yes/no, or on/off; got '${2}'" >&2
      exit 2
      ;;
  esac
}

require_positive_int() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${name} must be a positive integer; got '${value}'" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name=$1 value=$2
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "${name} must be a non-negative integer; got '${value}'" >&2
    exit 2
  fi
}

mode=${MODE:-${SPEC_MODE:-dspark}}
case "${mode}" in
  off|mtp0|standard-mtp0) mode=mtp0 ;;
  mtp2|standard-mtp2) mode=mtp2 ;;
  mtp3|standard-mtp3) mode=mtp3 ;;
  dspark-off|dspark-mtp0) mode=dspark-mtp0 ;;
  dspark) ;;
  *)
    echo "MODE must be mtp0, mtp2, mtp3, dspark-off, dspark-mtp0, or dspark; got '${mode}'" >&2
    exit 2
    ;;
esac

backend=${BACKEND:-b12x-a8}
case "${backend}" in
  b12x) backend=b12x-a16 ;;
  b12x-a16|b12x-a8|b12x-a8-dglin|lucifer-default|lucifer-cutlass) ;;
  *)
    echo "BACKEND must be b12x-a16, b12x-a8, b12x-a8-dglin," \
      "lucifer-default, or lucifer-cutlass; got '${backend}'" >&2
    exit 2
    ;;
esac

standard_model=${STANDARD_MODEL:-deepseek-ai/DeepSeek-V4-Flash}
dspark_model=${DSPARK_MODEL:-deepseek-ai/DeepSeek-V4-Flash-0731}
standard_model_revision=${STANDARD_MODEL_REVISION:-60d8d70770c6776ff598c94bb586a859a38244f1}
dspark_model_revision=${DSPARK_MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}
if [[ "${mode}" == "dspark" || "${mode}" == "dspark-mtp0" ]]; then
  model=${MODEL_PATH:-${MODEL:-${dspark_model}}}
  spec_model=${SPEC_MODEL_PATH:-${model}}
  served_model_name=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}
  default_model_revision=${dspark_model_revision}
else
  model=${MODEL_PATH:-${MODEL:-${standard_model}}}
  spec_model=
  served_model_name=${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}
  default_model_revision=${standard_model_revision}
fi

model_revision=${MODEL_REVISION:-}
if [[ -z "${model_revision}" ]]; then
  if [[ "${model}" == "${standard_model}" || "${model}" == "${dspark_model}" ]]; then
    model_revision=${default_model_revision}
  fi
fi
revision_args=()
if [[ -n "${model_revision}" ]]; then
  revision_args=(--revision "${model_revision}")
fi

host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp_size=${TP_SIZE:-${TP:-4}}
dcp_size=${DCP_SIZE:-${DCP:-1}}
if [[ "${mode}" == "dspark" || "${mode}" == "dspark-mtp0" ]]; then
  max_num_seqs=${MAX_NUM_SEQS:-16}
  max_model_len=${MAX_MODEL_LEN:-131072}
else
  max_num_seqs=${MAX_NUM_SEQS:-64}
  max_model_len=${MAX_MODEL_LEN:-262144}
fi
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-}
block_size=${BLOCK_SIZE:-256}
load_format=${LOAD_FORMAT:-instanttensor}
kv_offloading_size=${KV_OFFLOADING_SIZE:-}
native_l2_path=${NATIVE_L2_PATH:-}
native_l2_size=${NATIVE_L2_GB:-}
prefix_cache=$(bool_value PREFIX_CACHE "${PREFIX_CACHE:-1}")
enable_flashinfer_autotune=$(bool_value ENABLE_FLASHINFER_AUTOTUNE "${ENABLE_FLASHINFER_AUTOTUNE:-1}")
draft_sample_method=${DRAFT_SAMPLE_METHOD:-probabilistic}
rejection_sample_method=${REJECTION_SAMPLE_METHOD:-standard}

require_positive_int TP_SIZE "${tp_size}"
require_positive_int DCP_SIZE "${dcp_size}"
require_positive_int MAX_NUM_SEQS "${max_num_seqs}"
require_positive_int MAX_NUM_BATCHED_TOKENS "${max_num_batched_tokens}"
require_positive_int BLOCK_SIZE "${block_size}"
if [[ -n "${kv_offloading_size}" \
  && ! "${kv_offloading_size}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "KV_OFFLOADING_SIZE must be a non-negative GiB value; got '${kv_offloading_size}'" >&2
  exit 2
fi
native_l2_enabled=0
if [[ -n "${native_l2_path}" || -n "${native_l2_size}" ]]; then
  if [[ -z "${native_l2_path}" || -z "${native_l2_size}" ]]; then
    echo "NATIVE_L2_PATH and NATIVE_L2_GB must be set together" >&2
    exit 2
  fi
  if [[ "${native_l2_path}" != /* ]]; then
    echo "NATIVE_L2_PATH must be an absolute container path" >&2
    exit 2
  fi
  if [[ ! "${native_l2_size}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ \
    || "${native_l2_size}" =~ ^0*([.]0*)?$ ]]; then
    echo "NATIVE_L2_GB must be a positive GiB value; got '${native_l2_size}'" >&2
    exit 2
  fi
  if [[ -z "${kv_offloading_size}" \
    || "${kv_offloading_size}" =~ ^0*([.]0*)?$ ]]; then
    echo "NATIVE_L2 requires a positive KV_OFFLOADING_SIZE for its L1 tier" >&2
    exit 2
  fi
  native_l2_enabled=1
fi
if [[ "${mode}" == "dspark" && "${dcp_size}" != "1" ]]; then
  echo "DSpark non-causal attention currently requires DCP_SIZE=1" >&2
  exit 2
fi

case "${draft_sample_method}" in
  probabilistic|greedy) ;;
  *)
    echo "DRAFT_SAMPLE_METHOD must be probabilistic or greedy" >&2
    exit 2
    ;;
esac
case "${rejection_sample_method}" in
  standard|block) ;;
  *)
    echo "REJECTION_SAMPLE_METHOD must be standard or block" >&2
    exit 2
    ;;
esac

export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_120a}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_PROTO=${NCCL_PROTO:-LL,LL128,Simple}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export LLM_WORKER_MULTIPROC_METHOD=${LLM_WORKER_MULTIPROC_METHOD:-spawn}
export SAFETENSORS_FAST_GPU=${SAFETENSORS_FAST_GPU:-1}
export INSTANTTENSOR_BACKEND=${INSTANTTENSOR_BACKEND:-BUFFERED}

export VLLM_USE_AOT_COMPILE=${VLLM_USE_AOT_COMPILE:-1}
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}
export VLLM_USE_MEGA_AOT_ARTIFACT=${VLLM_USE_MEGA_AOT_ARTIFACT:-1}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-1}
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-1}
export VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=${VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD:-1024}

allreduce_mode=${ALLREDUCE_MODE:-auto}
if [[ "${allreduce_mode}" == "auto" ]]; then
  allreduce_mode=b12x
fi
allreduce_args=()
case "${allreduce_mode}" in
  b12x)
    export VLLM_ENABLE_PCIE_ALLREDUCE=1
    export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
    export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-64KB}
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    ;;
  nccl)
    export VLLM_ENABLE_PCIE_ALLREDUCE=0
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
    allreduce_args=(--disable-custom-all-reduce)
    ;;
  *)
    echo "ALLREDUCE_MODE must be auto, b12x, or nccl; got '${allreduce_mode}'" >&2
    exit 2
    ;;
esac

b12x_backend=0
backend_args=()
case "${backend}" in
  b12x-a16|b12x-a8|b12x-a8-dglin)
    b12x_backend=1
    backend_args=(--attention-backend B12X --moe-backend b12x)
    if [[ "${backend}" != "b12x-a8-dglin" ]]; then
      backend_args+=(--linear-backend b12x)
    fi
    if [[ "${backend}" == "b12x-a16" ]]; then
      export VLLM_B12X_MOE_FP4_FORCE_A16=1
    else
      export VLLM_B12X_MOE_FP4_FORCE_A16=0
    fi
    ;;
  lucifer-default)
    backend_args=(--attention-backend FLASHINFER_MLA_SPARSE_DSV4)
    ;;
  lucifer-cutlass)
    backend_args=(
      --attention-backend FLASHINFER_MLA_SPARSE_DSV4
      --moe-backend flashinfer_cutlass
    )
    ;;
esac

spec_args=()
spec_tokens=0
graph_multiplier=4
dspark_depth_mode=disabled
if [[ "${mode}" == "mtp2" || "${mode}" == "mtp3" ]]; then
  if [[ "${mode}" == "mtp2" ]]; then spec_tokens=2; else spec_tokens=3; fi
  mtp_moe_json=
  if (( b12x_backend )); then
    mtp_moe_json=',"moe_backend":"b12x"'
  fi
  spec_json=$(printf \
    '{"method":"mtp","num_speculative_tokens":%s,"draft_sample_method":"%s","rejection_sample_method":"%s"%s}' \
    "${spec_tokens}" "${draft_sample_method}" "${rejection_sample_method}" \
    "${mtp_moe_json}")
  spec_args=(--speculative-config "${spec_json}")
  graph_multiplier=8
elif [[ "${mode}" == "dspark" ]]; then
  spec_tokens=${DSPARK_TOKENS:-${NUM_SPECULATIVE_TOKENS:-7}}
  require_positive_int DSPARK_TOKENS "${spec_tokens}"
  # Target verification schedules at most one sampled token plus K drafts per
  # request. Capturing beyond that physical row count only consumes graph
  # memory (and can OOM at high concurrency) because those shapes are
  # unreachable for DSpark.
  graph_multiplier=$((spec_tokens + 1))
  draft_attention_backend=${DSPARK_DRAFT_ATTENTION_BACKEND:-auto}
  draft_attention_json=
  if [[ "${draft_attention_backend}" != "auto" ]]; then
    case "${draft_attention_backend}" in
      B12X|FLASHINFER_MLA_SPARSE_DSV4|FLASHMLA_SPARSE_DSV4) ;;
      *)
        echo "DSPARK_DRAFT_ATTENTION_BACKEND must be auto, B12X," \
          "FLASHINFER_MLA_SPARSE_DSV4, or FLASHMLA_SPARSE_DSV4" >&2
        exit 2
        ;;
    esac
    draft_attention_json=$(printf \
      ',"attention_backend":"%s"' "${draft_attention_backend}")
  fi
  dspark_depth_mode=${DSPARK_DEPTH_MODE:-fixed}
  case "${dspark_depth_mode}" in
    fixed)
      adaptive_verification_json=
      ;;
    adaptive|dynamic)
      adaptive_verification_json=',"enable_adaptive_verification":true'
      ;;
    *)
      echo "DSPARK_DEPTH_MODE must be fixed, adaptive, or dynamic" >&2
      exit 2
      ;;
  esac
  spec_json=$(printf \
    '{"model":"%s","method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"%s","rejection_sample_method":"%s"%s%s}' \
    "${spec_model}" "${spec_tokens}" "${draft_sample_method}" \
    "${rejection_sample_method}" \
    "${draft_attention_json}" \
    "${adaptive_verification_json}")
  spec_args=(--speculative-config "${spec_json}")
fi

# v9 used graph 256 for MTP-off and 512 for MTP modes at cc64. Keep that MTP
# contract; DSpark uses its exact (K + 1) physical verifier width.
graph_cap=${MAX_CUDAGRAPH_CAPTURE_SIZE:-${GRAPH:-}}
if [[ -z "${graph_cap}" || "${graph_cap}" == "auto" ]]; then
  graph_cap=$((max_num_seqs * graph_multiplier))
  if (( graph_cap < 6 )); then graph_cap=6; fi
fi
require_positive_int MAX_CUDAGRAPH_CAPTURE_SIZE "${graph_cap}"

sp_async_tp=$(bool_value SP_ASYNC_TP "${SP_ASYNC_TP:-0}")
compilation_config='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
export VLLM_USE_V2_MODEL_RUNNER=1
if [[ "${sp_async_tp}" == "1" ]]; then
  if [[ "${mode}" == "dspark" ]]; then
    echo "SP_ASYNC_TP=1 is not supported by the V2 DSpark runner" >&2
    exit 2
  fi
  sp_min_tokens=${SP_MIN_TOKEN_NUM:-512}
  require_positive_int SP_MIN_TOKEN_NUM "${sp_min_tokens}"
  compilation_config=$(printf \
    '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"use_inductor_graph_partition":true,"pass_config":{"enable_sp":true,"fuse_gemm_comms":true,"sp_min_token_num":%s}}' \
    "${sp_min_tokens}")
  export VLLM_USE_V2_MODEL_RUNNER=0
  export VLLM_SYMM_MEM_PCIE_SAFE_BARRIER=1
fi

if [[ -z "${gpu_memory_utilization}" ]]; then
  # The 0731 DSpark draft head is larger than the historical MTP head. Its
  # B12X TP2 profile needs 0.975 to retain the default 131k serving limit after
  # attention and FULL-graph allocations are accounted for. At 0.97 the r15
  # stack exposed 7.11 GiB of KV storage while this profile needs 7.37 GiB.
  if [[ "${mode}" == "dspark" ]]; then
    if [[ "${backend}" == "lucifer-default" ]]; then
      # The default DeepGEMM MoE path retains more model/runtime memory than
      # FlashInfer CUTLASS. At 0.9465 only 7.35 GiB remained for the 7.89 GiB
      # DSpark KV requirement. 0.953 profiles 8.03 GiB (266609 tokens) and
      # leaves enough transient headroom for a sustained 128k prefill.
      gpu_memory_utilization=0.953
    elif [[ "${backend}" == "lucifer-cutlass" ]] && (( tp_size >= 4 )); then
      # FlashInfer CUTLASS allocates a transient MoE workspace on the first
      # real prefill. At TP4, 0.9465 left only 777 MiB free for a 764 MiB
      # workspace and could OOM after an otherwise successful warmup. TP4+
      # still has ample KV capacity at 0.94; TP2 needs the 0.9465 Lucifer
      # default below to preserve the advertised 262k serving limit.
      gpu_memory_utilization=0.94
    elif [[ "${backend}" == lucifer-* ]]; then
      gpu_memory_utilization=0.9465
    else
      gpu_memory_utilization=0.975
    fi
  elif [[ "${backend}" == lucifer-* \
    && ( "${mode}" == "mtp2" || "${mode}" == "mtp3" ) ]]; then
    # Lucifer MTP FULL graphs leave about 7.48 GiB at 0.91, just below the
    # 7.55 GiB needed to preserve the documented 262k serving limit.
    gpu_memory_utilization=0.912
  else
    gpu_memory_utilization=0.91
  fi
fi

prefix_cache_retention_interval=${PREFIX_CACHE_RETENTION_INTERVAL:-4096}
require_nonnegative_int \
  PREFIX_CACHE_RETENTION_INTERVAL "${prefix_cache_retention_interval}"
prefix_args=(
  --enable-prefix-caching
  --prefix-cache-retention-interval "${prefix_cache_retention_interval}"
)
if [[ "${prefix_cache}" == "0" ]]; then
  prefix_args=(--no-enable-prefix-caching)
fi
autotune_args=(--enable-flashinfer-autotune)
if [[ "${enable_flashinfer_autotune}" == "0" ]]; then
  autotune_args=(--no-enable-flashinfer-autotune)
fi

offloading_args=()
if [[ -n "${kv_offloading_size}" \
  && ! "${kv_offloading_size}" =~ ^0*([.]0*)?$ ]]; then
  # This is the total host-cache capacity across all TP ranks, matching the
  # vLLM CLI contract. LMCache remains a separate deployment mode. Native KV
  # offload pins/registers GPU cache allocations, so PyTorch's remappable VMM
  # segments must be disabled while preserving any other allocator settings.
  allocator_config=${PYTORCH_CUDA_ALLOC_CONF:-}
  if [[ -z "${allocator_config}" ]]; then
    allocator_config=expandable_segments:False
  elif [[ "${allocator_config}" =~ (^|,)expandable_segments:True(,|$) ]]; then
    allocator_config=${allocator_config//expandable_segments:True/expandable_segments:False}
  fi
  export PYTORCH_CUDA_ALLOC_CONF=${allocator_config}
  offloading_args=(
    --kv-offloading-size "${kv_offloading_size}"
    --kv-offloading-backend native
  )
  if [[ "${native_l2_enabled}" == "1" ]]; then
    export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
    native_l2_config=$("${PYTHON_BIN}" - \
      "${native_l2_path}" "${native_l2_size}" <<'PY'
import json
import sys

root_dir, max_size_gb = sys.argv[1:]
config = {
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [
            {
                "type": "fs",
                "root_dir": root_dir,
                "n_read_threads": 32,
                "n_write_threads": 16,
                "gc_max_size_gb": float(max_size_gb),
            }
        ],
    },
}
print(json.dumps(config, separators=(",", ":")))
PY
    )
    offloading_args+=(--kv-transfer-config "${native_l2_config}")
  fi
fi

capture_args=()
capture_sizes=${CUDAGRAPH_CAPTURE_SIZES:-default}
if [[ "${capture_sizes}" == "auto" ]]; then
  sizes=(1)
  n=2
  while (( n < graph_cap )); do sizes+=("${n}"); n=$((n * 2)); done
  if (( max_num_seqs <= graph_cap )); then
    sizes+=("${max_num_seqs}")
  fi
  sizes+=("${graph_cap}")
  mapfile -t sizes < <(printf '%s\n' "${sizes[@]}" | sort -n -u)
  capture_args=(--cudagraph-capture-sizes "${sizes[@]}")
elif [[ "${capture_sizes}" != "default" && "${capture_sizes}" != "none" ]]; then
  read -r -a sizes <<< "${capture_sizes//,/ }"
  capture_args=(--cudagraph-capture-sizes "${sizes[@]}")
fi

cache_root=${XDG_CACHE_HOME:-/cache}
export XDG_CACHE_HOME=${cache_root}
vllm_cache_dir=${cache_root}/vllm
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-${cache_root}/tilelang}
export TILELANG_TMP_DIR=${TILELANG_TMP_DIR:-${cache_root}/tilelang/tmp}
export TVM_CACHE_DIR=${TVM_CACHE_DIR:-${cache_root}/tvm}
export TVM_FFI_CACHE_DIR=${TVM_FFI_CACHE_DIR:-${cache_root}/jit/tvm-ffi}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${cache_root}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torchinductor}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${cache_root}/jit/torch_extensions}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-${cache_root}/flashinfer}
mkdir -p \
  "${vllm_cache_dir}" "${TILELANG_CACHE_DIR}" "${TILELANG_TMP_DIR}" \
  "${TVM_CACHE_DIR}" "${TVM_FFI_CACHE_DIR}" "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
  "${FLASHINFER_WORKSPACE_BASE}"

command=(
  "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${model}"
  "${revision_args[@]}"
  --served-model-name "${served_model_name}"
  --host "${host}"
  --port "${port}"
  --trust-remote-code
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --block-size "${block_size}"
  --load-format "${load_format}"
  --tensor-parallel-size "${tp_size}"
  --decode-context-parallel-size "${dcp_size}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --max-model-len "${max_model_len}"
  --max-num-seqs "${max_num_seqs}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --max-cudagraph-capture-size "${graph_cap}"
  --compilation-config "${compilation_config}"
  --async-scheduling
  --no-scheduler-reserve-full-isl
  --enable-chunked-prefill
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --reasoning-parser deepseek_v4
  --enable-auto-tool-choice
  --enable-prompt-tokens-details
  --enable-force-include-usage
  --enable-request-id-headers
  --default-chat-template-kwargs.thinking=true
  --default-chat-template-kwargs.reasoning_effort=high
  "${autotune_args[@]}"
  "${prefix_args[@]}"
  "${capture_args[@]}"
  "${offloading_args[@]}"
  "${spec_args[@]}"
  "${backend_args[@]}"
  "${allreduce_args[@]}"
)

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # EXTRA_VLLM_ARGS is intentionally an escape hatch for temporary experiments.
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_VLLM_ARGS} )
  command+=("${extra_args[@]}")
fi
command+=("$@")

printf 'DS4 launch: mode=%s depth=%s backend=%s allreduce=%s tp=%s dcp=%s max_seqs=%s graph=%s load_format=%s instanttensor_backend=%s native_l2=%s allocator=%s model=%s\n' \
  "${mode}" "${dspark_depth_mode}" \
  "${backend}" "${allreduce_mode}" \
  "${tp_size}" "${dcp_size}" "${max_num_seqs}" \
  "${graph_cap}" "${load_format}" "${INSTANTTENSOR_BACKEND}" \
  "${native_l2_enabled}" \
  "${PYTORCH_CUDA_ALLOC_CONF:-<unset>}" "${model}" >&2
printf 'Command:' >&2
printf ' %q' "${command[@]}" >&2
printf '\n' >&2

if [[ "$(bool_value DRY_RUN "${DRY_RUN:-0}")" == "1" ]]; then
  exit 0
fi
exec "${command[@]}"
