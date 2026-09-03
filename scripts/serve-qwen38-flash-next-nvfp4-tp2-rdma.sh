#!/usr/bin/env bash
# shellcheck disable=SC2029

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
B12X_ROOT="${B12X_ROOT:-/home/luke/projects/b12x}"
SPARK_ROOT="${SPARK_ROOT:-/home/luke/projects/spark-vllm-docker}"
CLUSTER_LAUNCHER="${CLUSTER_LAUNCHER:-${SPARK_ROOT}/launch-cluster.sh}"

HEAD_IP="${HEAD_IP:-192.168.177.11}"
WORKER_IP="${WORKER_IP:-192.168.177.12}"
ETH_IF="${ETH_IF:-enp1s0f1np1}"
IB_IF="${IB_IF:-rocep1s0f1,roceP2p1s0f1}"
MASTER_PORT="${MASTER_PORT:-29638}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_qwen38_flash_next_tp2}"
IMAGE_NAME="${IMAGE_NAME:-vllm-node-eugr-20260712:latest}"
CONTAINER_MEMORY_GB="${CONTAINER_MEMORY_GB:-108}"
CONTAINER_MEMORY_SWAP_GB="${CONTAINER_MEMORY_SWAP_GB:-112}"

PYTHON_BIN="${PYTHON_BIN:-${VLLM_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ROOT}/.venv/bin/vllm}"
DEFAULT_MODEL_PATH=/data/models/qwen3.8-flash-next-mixed
DEFAULT_MODEL_PATH+="/qwen3.8-flash-next-180b-nvfp4-ple-mxfp8-attn-shared_vv1"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.8-flash-next-4p89bpw}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1610612736}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
B12X_POLICY_MODE="${B12X_POLICY_MODE:-auto}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
COMPILATION_CONFIG="${VLLM_QWEN38_COMPILATION_CONFIG:-}"
if [[ -z "${COMPILATION_CONFIG}" ]]; then
  COMPILATION_CONFIG='{"pass_config":{"fuse_act_quant":true}}'
fi

sync_code=0
sync_model=0
check_only=0
detach=0
vllm_args=()

usage() {
  cat <<EOF
Usage: $0 [launcher options] [-- vLLM options]

Launch Qwen 3.8 Flash Next with TP=2 across tachyon and luxon. The Spark
cluster launcher starts one native vLLM rank per node and configures NCCL RDMA
over both ConnectX-7 RoCE interfaces. No external scheduler is used.

Launcher options:
  --sync-code   Mirror local vllm/ and b12x/ to the worker, removing stale files
                inside those two worker package directories.
  --sync-model  Rsync the roughly 99 GiB MODEL_PATH to the worker first.
  --check       Validate both nodes and Spark networking without launching.
  --detach      Run the head rank in the background; use docker logs to follow it.
  -h, --help    Show this help.

Environment overrides include MODEL_PATH, MAX_MODEL_LEN, KV_CACHE_MEMORY_BYTES,
HEAD_IP, WORKER_IP, ETH_IF, IB_IF, IMAGE_NAME, and CONTAINER_MEMORY_GB.
EOF
}

while (($#)); do
  case "$1" in
    --sync-code)
      sync_code=1
      shift
      ;;
    --sync-model)
      sync_model=1
      shift
      ;;
    --check)
      check_only=1
      shift
      ;;
    --detach)
      detach=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      vllm_args=("$@")
      break
      ;;
    *)
      echo "Unknown launcher option: $1" >&2
      echo "Put additional vLLM arguments after --." >&2
      exit 2
      ;;
  esac
done

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

for path in \
  "${VLLM_ROOT}" \
  "${B12X_ROOT}" \
  "${MODEL_PATH}" \
  "${CLUSTER_LAUNCHER}"; do
  if [[ "${path}" == *[[:space:]]* ]]; then
    echo "Spark bind-mount paths cannot contain whitespace: ${path}" >&2
    exit 2
  fi
done

if [[ ! -x "${CLUSTER_LAUNCHER}" ]]; then
  echo "Spark cluster launcher is not executable: ${CLUSTER_LAUNCHER}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -x "${VLLM_BIN}" ]]; then
  echo "vLLM CLI is not executable: ${VLLM_BIN}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Local model config not found: ${MODEL_PATH}/config.json" >&2
  exit 1
fi
if [[ ! -f "${VLLM_ROOT}/vllm/__init__.py" ]]; then
  echo "Local vLLM source tree not found under ${VLLM_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${B12X_ROOT}/b12x/__init__.py" ]]; then
  echo "Local b12x source tree not found under ${B12X_ROOT}" >&2
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

if ((sync_model)); then
  printf -v remote_model_parent '%q' "$(dirname -- "${MODEL_PATH}")"
  remote_prepare_model="mkdir -p -- ${remote_model_parent} 2>/dev/null"
  remote_prepare_model+=" || sudo -n install -d"
  remote_prepare_model+=" -o \$(id -un) -g \$(id -gn)"
  remote_prepare_model+=" -- ${remote_model_parent}"
  ssh "${ssh_opts[@]}" "${WORKER_IP}" \
    "${remote_prepare_model}"
  echo "Rsyncing the model to ${WORKER_IP}:${MODEL_PATH}..."
  rsync -a --partial --info=progress2 \
    "${MODEL_PATH}/" \
    "${WORKER_IP}:${MODEL_PATH}/"
fi

remote_files=(
  "${PYTHON_BIN}"
  "${VLLM_BIN}"
  "${VLLM_ROOT}/vllm/__init__.py"
  "${B12X_ROOT}/b12x/__init__.py"
  "${MODEL_PATH}/config.json"
)
for path in "${remote_files[@]}"; do
  printf -v remote_path '%q' "${path}"
  if ! ssh "${ssh_opts[@]}" "${WORKER_IP}" "test -e ${remote_path}"; then
    echo "Required worker path is missing: ${WORKER_IP}:${path}" >&2
    if [[ "${path}" == "${MODEL_PATH}/config.json" ]]; then
      echo "Rerun with --sync-model to transfer the model first." >&2
    else
      echo "Rerun with --sync-code after preparing the worker venv." >&2
    fi
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
mount_args+=" -v ${MODEL_PATH}:${MODEL_PATH}:ro"
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
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "VLLM_PLUGINS="
  --env "VLLM_SSM_CONV_STATE_LAYOUT=DS"
  --env "VLLM_USE_AOT_COMPILE=1"
  --env "VLLM_USE_MEGA_AOT_ARTIFACT=1"
  --env "VLLM_USE_V2_MODEL_RUNNER=1"
  --env "VLLM_ENABLE_PCIE_ALLREDUCE=0"
  --env "B12X_POLICY_MODE=${B12X_POLICY_MODE}"
  --env "INSTANTTENSOR_BACKEND=BUFFERED"
  --env "INSTANTTENSOR_BUFFER_SIZE=1342177280"
  --env "INSTANTTENSOR_CONCURRENCY=1"
  --env "INSTANTTENSOR_IO_DEPTH=3"
  --env "NCCL_NET_PLUGIN=none"
  --env "NCCL_IB_MERGE_NICS=0"
  --env "NCCL_IB_SUBNET_AWARE_ROUTING=1"
)

if ((check_only)); then
  exec "${CLUSTER_LAUNCHER}" "${cluster_args[@]}" --check-config
fi
if ((detach)); then
  cluster_args+=(-d)
fi

speculative_config=$(printf \
  '{"method":"mtp","num_speculative_tokens":%s}' \
  "${NUM_SPECULATIVE_TOKENS}")

vllm_command=(
  "${VLLM_BIN}" serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --disable-custom-all-reduce
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 16
  --load-format instanttensor
  --model-loader-extra-config '{"instanttensor_copy":false}'
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --speculative-config "${speculative_config}"
  --gdn-decode-kernel b12x
  --linear-backend b12x
  --moe-backend b12x
  --no-enable-flashinfer-autotune
  --mm-encoder-tp-mode data
  --mm-processor-cache-gb 0
  --limit-mm-per-prompt '{"image":1}'
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --compilation-config "${COMPILATION_CONFIG}"
)
vllm_command+=("${vllm_args[@]}")

exec "${CLUSTER_LAUNCHER}" "${cluster_args[@]}" exec "${vllm_command[@]}"
