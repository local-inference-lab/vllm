#!/usr/bin/env bash
# shellcheck disable=SC2029
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
B12X_ROOT="${B12X_ROOT:-/home/luke/projects/b12x}"
SPARK_ROOT="${SPARK_ROOT:-/home/luke/projects/spark-vllm-docker}"
CLUSTER_LAUNCHER="${CLUSTER_LAUNCHER:-${SPARK_ROOT}/launch-cluster.sh}"
HEAD_IP="${HEAD_IP:-192.168.42.223}"
WORKER_IP="${WORKER_IP:-192.168.42.110}"
ETH_IF="${ETH_IF:-enP7s7}"
IB_IF="${IB_IF:-rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1}"
HEAD_IB_IF="${HEAD_IB_IF:-rocep1s0f1,roceP2p1s0f1}"
WORKER_IB_IF="${WORKER_IB_IF:-rocep1s0f0,roceP2p1s0f0}"
NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-1}"
MASTER_PORT="${MASTER_PORT:-29655}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_ds4_flash_dspark_tp2}"
IMAGE_NAME="${IMAGE_NAME:-vllm-node-eugr-20260712:latest}"
CONTAINER_MEMORY_GB="${CONTAINER_MEMORY_GB:-108}"
CONTAINER_MEMORY_SWAP_GB="${CONTAINER_MEMORY_SWAP_GB:-112}"
PYTHON_BIN="${PYTHON_BIN:-${VLLM_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ROOT}/.venv/bin/vllm}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/vllm-huggingface}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
MODEL_REVISION="${MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-500000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-10737418240}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
DSPARK_DRAFT_ATTENTION_BACKEND="${DSPARK_DRAFT_ATTENTION_BACKEND:-auto}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
B12X_POLICY_MODE="${B12X_POLICY_MODE:-auto}"
# All-reduce transport between the two ranks: rocenante (b12x.comm.roce one-shot
# RDMA, VLLM_ENABLE_ROCE_ALLREDUCE=1) or nccl.
ALLREDUCE="${ALLREDUCE:-rocenante}"
ROCE_ALLREDUCE_MAX_SIZE="${ROCE_ALLREDUCE_MAX_SIZE:-2MB}"
ROCE_ALLGATHER_MAX_SIZE="${ROCE_ALLGATHER_MAX_SIZE:-16MB}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

sync_code=0
check_only=0
detach=0
vllm_args=()

usage() {
  cat <<EOF
Usage: $0 [launcher options] [-- vLLM options]

Launch DeepSeek-V4-Flash with DSpark speculative decoding and TP=2 across
tachyon and luxon through the Spark cluster launcher (one native vLLM rank per
node, management LAN for bootstrap, both RoCE rails on the direct ConnectX-7
link). ALLREDUCE=rocenante (default) routes the TP all-reduces and the logits
all-gather to the b12x one-shot RoCE collectives; ALLREDUCE=nccl keeps NCCL.

Launcher options:
  --sync-code   Mirror local vllm/ and b12x/ runtime packages to the worker.
  --check       Validate both nodes and Spark networking without launching.
  --detach      Run the head rank in the background; use docker logs to follow it.
  --no-spec     Plain decode: no DSpark drafter (same as NUM_SPECULATIVE_TOKENS=0).
  --spec N      DSpark with N speculative tokens (same as NUM_SPECULATIVE_TOKENS=N).
  -h, --help    Show this help.

Environment overrides include ALLREDUCE, HEAD_IP, WORKER_IP, MODEL_ID,
MODEL_REVISION, HF_CACHE, MAX_MODEL_LEN, MAX_NUM_SEQS, NUM_SPECULATIVE_TOKENS,
KV_CACHE_MEMORY_BYTES, GPU_MEMORY_UTILIZATION, B12X_ROOT, B12X_POLICY_MODE,
ROCE_ALLREDUCE_MAX_SIZE, ROCE_ALLGATHER_MAX_SIZE, IMAGE_NAME and
CONTAINER_MEMORY_GB.
EOF
}

while (($#)); do
  case "$1" in
    --sync-code) sync_code=1; shift ;;
    --check) check_only=1; shift ;;
    --detach) detach=1; shift ;;
    --no-spec) NUM_SPECULATIVE_TOKENS=0; shift ;;
    --spec)
      if (($# < 2)); then
        echo "--spec requires a token count" >&2
        exit 2
      fi
      NUM_SPECULATIVE_TOKENS=$2
      shift 2
      ;;
    --spec=*) NUM_SPECULATIVE_TOKENS=${1#*=}; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; vllm_args=("$@"); break ;;
    *)
      echo "Unknown launcher option: $1" >&2
      echo "Put additional vLLM arguments after --." >&2
      exit 2
      ;;
  esac
done

case "${ALLREDUCE}" in
  rocenante|nccl) ;;
  *)
    echo "ALLREDUCE must be rocenante or nccl; got '${ALLREDUCE}'" >&2
    exit 2
    ;;
esac
case "${B12X_POLICY_MODE}" in
  auto|heuristic-only|preplanned-only) ;;
  *)
    echo "Invalid B12X policy mode: ${B12X_POLICY_MODE}" >&2
    exit 2
    ;;
esac
case "${NCCL_DEBUG}" in
  VERSION|WARN|INFO|TRACE) ;;
  *)
    echo "Invalid NCCL_DEBUG level: ${NCCL_DEBUG}" >&2
    exit 2
    ;;
esac
if [[ ! "${NUM_SPECULATIVE_TOKENS}" =~ ^[0-9]+$ ]]; then
  echo "NUM_SPECULATIVE_TOKENS must be a non-negative integer." >&2
  exit 2
fi
case "${DSPARK_DRAFT_ATTENTION_BACKEND}" in
  auto|B12X_MLA_SPARSE|FLASHINFER_MLA_SPARSE_DSV4|FLASHMLA_SPARSE_DSV4) ;;
  *)
    echo "DSPARK_DRAFT_ATTENTION_BACKEND must be auto, B12X_MLA_SPARSE," \
      "FLASHINFER_MLA_SPARSE_DSV4, or FLASHMLA_SPARSE_DSV4" >&2
    exit 2
    ;;
esac

snapshot="${HF_CACHE}/hub/models--${MODEL_ID//\//--}/snapshots/${MODEL_REVISION}"
for path in "${VLLM_ROOT}" "${B12X_ROOT}" "${HF_CACHE}" "${CLUSTER_LAUNCHER}"; do
  if [[ "${path}" == *[[:space:]]* ]]; then
    echo "Spark bind-mount paths cannot contain whitespace: ${path}" >&2
    exit 2
  fi
done
if [[ ! -x "${CLUSTER_LAUNCHER}" ]]; then
  echo "Spark cluster launcher is not executable: ${CLUSTER_LAUNCHER}" >&2
  exit 1
fi
for path in "${PYTHON_BIN}" "${VLLM_BIN}"; do
  if [[ ! -x "${path}" ]]; then
    echo "Not executable: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${snapshot}/config.json" ]]; then
  echo "Local model snapshot not found: ${snapshot}/config.json" >&2
  exit 1
fi
if [[ ! -f "${VLLM_ROOT}/vllm/__init__.py" ]]; then
  echo "Local vLLM source tree not found under ${VLLM_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${B12X_ROOT}/b12x/comm/roce/roce_oneshot.py" ]]; then
  echo "b12x under ${B12X_ROOT} lacks b12x.comm.roce (RoCEnante)" >&2
  exit 1
fi

ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o StrictHostKeyChecking=no
)
if ! ssh "${ssh_opts[@]}" "${WORKER_IP}" true; then
  echo "Passwordless SSH to worker ${WORKER_IP} failed." >&2
  exit 1
fi

if ((sync_code)); then
  echo "Mirroring vLLM runtime source to ${WORKER_IP}..."
  rsync -a --delete \
    --exclude='__pycache__/' \
    --exclude='*.py[co]' \
    "${VLLM_ROOT}/vllm/" \
    "${WORKER_IP}:${VLLM_ROOT}/vllm/"
  echo "Mirroring b12x runtime source to ${WORKER_IP}..."
  rsync -a --delete \
    --exclude='__pycache__/' \
    --exclude='*.py[co]' \
    "${B12X_ROOT}/b12x/" \
    "${WORKER_IP}:${B12X_ROOT}/b12x/"
fi

remote_files=(
  "${PYTHON_BIN}"
  "${VLLM_BIN}"
  "${VLLM_ROOT}/vllm/__init__.py"
  "${B12X_ROOT}/b12x/comm/roce/roce_oneshot.py"
  "${snapshot}/config.json"
)
for path in "${remote_files[@]}"; do
  printf -v remote_path '%q' "${path}"
  if ! ssh "${ssh_opts[@]}" "${WORKER_IP}" "test -e ${remote_path}"; then
    echo "Required worker path is missing: ${WORKER_IP}:${path}" >&2
    echo "Rerun with --sync-code, or copy the model snapshot." >&2
    exit 1
  fi
done

runtime_digest() {
  LC_ALL=C find \
    "${VLLM_ROOT}/vllm" \
    "${B12X_ROOT}/b12x" \
    \( -type f -o -type l \) \
    ! -path '*/__pycache__/*' \
    ! -name '*.py[co]' \
    -print0 \
    | sort -z \
    | xargs -0 -r sha256sum \
    | sha256sum \
    | cut -d' ' -f1
}
printf -v remote_vllm '%q' "${VLLM_ROOT}/vllm"
printf -v remote_b12x '%q' "${B12X_ROOT}/b12x"
remote_digest_command="LC_ALL=C find ${remote_vllm} ${remote_b12x} \
  \\( -type f -o -type l \\) \
  ! -path '*/__pycache__/*' ! -name '*.py[co]' -print0 \
  | sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1"
local_digest="$(runtime_digest)"
worker_digest="$(
  ssh "${ssh_opts[@]}" "${WORKER_IP}" "${remote_digest_command}"
)"
if [[ "${local_digest}" != "${worker_digest}" ]]; then
  echo "vLLM/b12x runtime source differs on ${WORKER_IP}." >&2
  echo "Rerun with --sync-code so both TP ranks execute identical code." >&2
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image is missing locally: ${IMAGE_NAME}" >&2
  exit 1
fi
if ! ssh "${ssh_opts[@]}" "${WORKER_IP}" \
  "docker image inspect ${IMAGE_NAME} >/dev/null 2>&1"; then
  echo "Docker image is missing on ${WORKER_IP}: ${IMAGE_NAME}" >&2
  exit 1
fi

mount_args="-v ${VLLM_ROOT}:${VLLM_ROOT}"
mount_args+=" -v ${B12X_ROOT}:${B12X_ROOT}"
mount_args+=" -v ${HF_CACHE}:${HF_CACHE}:ro"
if [[ -n "${VLLM_SPARK_EXTRA_DOCKER_ARGS:-}" ]]; then
  mount_args+=" ${VLLM_SPARK_EXTRA_DOCKER_ARGS}"
fi
export VLLM_SPARK_EXTRA_DOCKER_ARGS="${mount_args}"

cluster_args=(
  --nodes "${HEAD_IP},${WORKER_IP}"
  -t "${IMAGE_NAME}"
  --name "${CONTAINER_NAME}"
  --eth-if "${ETH_IF}"
  --ib-if "${IB_IF}"
  --node-ib-if "${HEAD_IP}=${HEAD_IB_IF}"
  --node-ib-if "${WORKER_IP}=${WORKER_IB_IF}"
  --master-port "${MASTER_PORT}"
  --nccl-debug "${NCCL_DEBUG}"
  --no-ray
  --non-privileged
  --mem-limit-gb "${CONTAINER_MEMORY_GB}"
  --mem-swap-limit-gb "${CONTAINER_MEMORY_SWAP_GB}"
  --env "PYTHONPATH=${VLLM_ROOT}:${B12X_ROOT}"
  --env "CUDA_HOME=/usr/local/cuda"
  --env "TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas"
  --env "CUDA_VISIBLE_DEVICES=0"
  --env "CUTE_DSL_ARCH=sm_121a"
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  --env "SAFETENSORS_FAST_GPU=1"
  --env "OMP_NUM_THREADS=16"
  --env "VLLM_WORKER_MULTIPROC_METHOD=spawn"
  --env "HF_HOME=${HF_CACHE}"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "VLLM_PLUGINS="
  --env "DG_JIT_USE_NVRTC=0"
  --env "USE_CUDNN=1"
  --env "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  --env "VLLM_USE_AOT_COMPILE=1"
  --env "VLLM_USE_BREAKABLE_CUDAGRAPH=0"
  --env "VLLM_USE_MEGA_AOT_ARTIFACT=1"
  --env "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1"
  --env "VLLM_USE_FLASHINFER_SAMPLER=1"
  --env "VLLM_USE_V2_MODEL_RUNNER=1"
  --env "VLLM_USE_B12X_WO_PROJECTION=1"
  --env "VLLM_USE_B12X_MHC=1"
  --env "VLLM_USE_B12X_FP8_GEMM=1"
  --env "VLLM_USE_B12X_MOE=1"
  --env "VLLM_USE_B12X_SPARSE_INDEXER=1"
  --env "B12X_MLA_SM120_UNIFIED=1"
  --env "B12X_DENSE_SPLITK_TURBO=1"
  --env "B12X_W4A16_TC_DECODE=1"
  --env "B12X_MOE_FORCE_A8=1"
  --env "B12X_POLICY_MODE=${B12X_POLICY_MODE}"
  --env "VLLM_ENABLE_PCIE_ALLREDUCE=0"
  --env "NCCL_NET_PLUGIN=none"
  --env "NCCL_IB_GID_INDEX=3"
  --env "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS}"
  --env "NCCL_IB_SUBNET_AWARE_ROUTING=1"
)
if [[ "${ALLREDUCE}" == rocenante ]]; then
  # b12x.comm.roce: B12X_ROCE_HCA falls back to the per-node NCCL_IB_HCA the
  # launcher sets, the GID index to NCCL_IB_GID_INDEX; the proxy .so is built
  # once into the mounted vLLM cache.
  cluster_args+=(
    --env "VLLM_ENABLE_ROCE_ALLREDUCE=1"
    --env "VLLM_ROCE_ALLREDUCE_MAX_SIZE=${ROCE_ALLREDUCE_MAX_SIZE}"
    --env "VLLM_ROCE_ALLGATHER_MAX_SIZE=${ROCE_ALLGATHER_MAX_SIZE}"
    --env "B12X_ROCE_CACHE_DIR=/root/.cache/vllm/b12x-roce"
  )
else
  cluster_args+=(--env "VLLM_ENABLE_ROCE_ALLREDUCE=0")
fi

if ((check_only)); then
  exec "${CLUSTER_LAUNCHER}" "${cluster_args[@]}" --check-config
fi
if ((detach)); then
  cluster_args+=(-d)
fi

max_cudagraph_capture_size=$((MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1)))
cudagraph_sizes="$(
  "${PYTHON_BIN}" - "$((NUM_SPECULATIVE_TOKENS + 1))" "${max_cudagraph_capture_size}" <<'PY'
import sys
depth, cap = int(sys.argv[1]), int(sys.argv[2])
sizes = sorted(set(list(range(1, min(depth, cap) + 1)) + list(range(depth, cap + 1, 4)) + [cap]))
print(",".join(str(x) for x in sizes))
PY
)"
compilation_config=$(printf \
  '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":[%s]}' \
  "${cudagraph_sizes}")

speculative_args=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  draft_attention_json=
  if [[ "${DSPARK_DRAFT_ATTENTION_BACKEND}" != auto ]]; then
    draft_attention_json=$(printf ',"attention_backend":"%s"' \
      "${DSPARK_DRAFT_ATTENTION_BACKEND}")
  fi
  speculative_config=$(printf \
    '{"method":"dspark","num_speculative_tokens":%s,"draft_sample_method":"probabilistic"%s}' \
    "${NUM_SPECULATIVE_TOKENS}" "${draft_attention_json}")
  speculative_args=(--speculative-config "${speculative_config}")
fi

vllm_command=(
  "${VLLM_BIN}" serve "${MODEL_ID}"
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size 2
  --decode-context-parallel-size 1
  --kv-cache-dtype fp8
  --block-size 256
  --load-format fastsafetensors
  --moe-backend b12x
  --linear-backend b12x
  --attention-backend B12X
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-cudagraph-capture-size "${max_cudagraph_capture_size}"
  --async-scheduling
  --no-scheduler-reserve-full-isl
  --enable-chunked-prefill
  --enable-prefix-caching
  --enable-flashinfer-autotune
  --compilation-config "${compilation_config}"
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --reasoning-config
  '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
  --default-chat-template-kwargs.thinking=true
  --default-chat-template-kwargs.reasoning_effort=high
)
vllm_command+=("${speculative_args[@]}")
vllm_command+=("${vllm_args[@]}")

if ((NUM_SPECULATIVE_TOKENS > 0)); then
  spec_summary="DSpark, ${NUM_SPECULATIVE_TOKENS} speculative tokens"
else
  spec_summary="plain decode, no drafter"
fi
cat <<BANNER
Launching ${SERVED_MODEL_NAME} TP=2 on ${HEAD_IP} + ${WORKER_IP}
  all-reduce:      ${ALLREDUCE}
  speculation:     ${spec_summary}
  max seqs:        ${MAX_NUM_SEQS} (cudagraph capture up to ${max_cudagraph_capture_size})
  context / KV:    ${MAX_MODEL_LEN} tokens, ${KV_CACHE_MEMORY_BYTES} bytes
  b12x policy:     ${B12X_POLICY_MODE} (B12X_ROOT=${B12X_ROOT})
BANNER

exec "${CLUSTER_LAUNCHER}" "${cluster_args[@]}" exec "${vllm_command[@]}"
