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
MASTER_PORT="${MASTER_PORT:-29653}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_glm53_flash_mtp_tp2}"
IMAGE_NAME="${IMAGE_NAME:-vllm-node-eugr-20260712:latest}"
CONTAINER_MEMORY_GB="${CONTAINER_MEMORY_GB:-108}"
CONTAINER_MEMORY_SWAP_GB="${CONTAINER_MEMORY_SWAP_GB:-112}"

PYTHON_BIN="${PYTHON_BIN:-${VLLM_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ROOT}/.venv/bin/vllm}"
MODEL_PATH="${MODEL_PATH:-/data/models/GLM-5.3-Flash-4p67}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.3-Flash}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-10G}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-5}"
MTP_MOE_BACKEND="${MTP_MOE_BACKEND:-humming}"
MTP_ATTENTION_BACKEND="${MTP_ATTENTION_BACKEND:-B12X}"
KDA_PREFILL_BACKEND="${KDA_PREFILL_BACKEND:-b12x}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-B12X}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
LINEAR_BACKEND="${LINEAR_BACKEND:-b12x}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
B12X_POLICY_MODE="${B12X_POLICY_MODE:-auto}"
NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES:-0}"
TORCH_PROFILE_WITH_MEMORY="${TORCH_PROFILE_WITH_MEMORY:-0}"
TORCH_PROFILE_WITH_STACK="${TORCH_PROFILE_WITH_STACK:-0}"
TORCH_PROFILE_WITH_FLOPS="${TORCH_PROFILE_WITH_FLOPS:-0}"
TORCH_PROFILE_USE_GZIP="${TORCH_PROFILE_USE_GZIP:-1}"
TORCH_PROFILE_DEFAULT_DIR="${VLLM_ROOT}/.profiles/glm53-tp2-mtp/torch"
TORCH_PROFILE_MAX_ITERATIONS=4

sync_code=0
sync_model=0
check_only=0
detach=0
vllm_args=()

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

usage() {
  cat <<EOF
Usage: $0 [launcher options] [-- vLLM options]

Launch GLM-5.3-Flash with native MTP and TP=2 across tachyon and luxon. The
Spark cluster launcher starts one native vLLM rank per node, uses the
management LAN for bootstrap, and combines the two RoCE rails on their direct
ConnectX-7 link in NCCL. No external scheduler is used.

Launcher options:
  --sync-code   Mirror local vllm/ and b12x/ runtime packages to luxon.
  --sync-model  Rsync the target model, including its MTP checkpoint, to luxon.
  --check       Validate both nodes and Spark networking without launching.
  --detach      Run the head rank in the background; use docker logs to follow it.
  --torch-profile [DIR]
                Configure a triggered four-step Torch CPU+CUDA capture.
  --torch-profile-record-shapes
                Record tensor shapes in the capture.
  --torch-profile-with-memory
                Record tensor memory activity in the capture.
  --torch-profile-with-flops
                Estimate supported operator FLOPs in the capture.
  --torch-profile-with-stack
                Record Python stacks; substantially increases unified-memory use.
  --torch-profile-no-stack
                Disable Python stack capture.
  --torch-profile-no-gzip
                Write uncompressed trace files.
  -h, --help    Show this help.

Environment overrides include MODEL_PATH, MAX_MODEL_LEN, MTP_MOE_BACKEND,
KDA_PREFILL_BACKEND, ATTENTION_BACKEND, MOE_BACKEND, LINEAR_BACKEND,
KV_CACHE_MEMORY_BYTES, HEAD_IP, WORKER_IP, ETH_IF, IB_IF, IMAGE_NAME,
CONTAINER_MEMORY_GB, and NUM_SPECULATIVE_TOKENS.
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
    --torch-profile)
      if (($# >= 2)) && [[ "$2" != -* ]]; then
        TORCH_PROFILE_DIR=$2
        shift 2
      else
        TORCH_PROFILE_DIR=${TORCH_PROFILE_DIR:-${TORCH_PROFILE_DEFAULT_DIR}}
        shift
      fi
      ;;
    --torch-profile=*)
      TORCH_PROFILE_DIR=${1#*=}
      if [[ -z "${TORCH_PROFILE_DIR}" ]]; then
        echo "--torch-profile requires a non-empty output directory" >&2
        exit 2
      fi
      shift
      ;;
    --torch-profile-record-shapes)
      TORCH_PROFILE_RECORD_SHAPES=1
      shift
      ;;
    --torch-profile-with-memory)
      TORCH_PROFILE_WITH_MEMORY=1
      shift
      ;;
    --torch-profile-with-flops)
      TORCH_PROFILE_WITH_FLOPS=1
      shift
      ;;
    --torch-profile-with-stack)
      TORCH_PROFILE_WITH_STACK=1
      shift
      ;;
    --torch-profile-no-stack)
      TORCH_PROFILE_WITH_STACK=0
      shift
      ;;
    --torch-profile-no-gzip)
      TORCH_PROFILE_USE_GZIP=0
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

TORCH_PROFILE_RECORD_SHAPES=$(bool_value \
  TORCH_PROFILE_RECORD_SHAPES "${TORCH_PROFILE_RECORD_SHAPES}")
TORCH_PROFILE_WITH_MEMORY=$(bool_value \
  TORCH_PROFILE_WITH_MEMORY "${TORCH_PROFILE_WITH_MEMORY}")
TORCH_PROFILE_WITH_STACK=$(bool_value \
  TORCH_PROFILE_WITH_STACK "${TORCH_PROFILE_WITH_STACK}")
TORCH_PROFILE_WITH_FLOPS=$(bool_value \
  TORCH_PROFILE_WITH_FLOPS "${TORCH_PROFILE_WITH_FLOPS}")
TORCH_PROFILE_USE_GZIP=$(bool_value \
  TORCH_PROFILE_USE_GZIP "${TORCH_PROFILE_USE_GZIP}")

case "${B12X_POLICY_MODE}" in
  auto|heuristic-only|preplanned-only) ;;
  *)
    echo "Invalid B12X policy mode: ${B12X_POLICY_MODE}" >&2
    exit 2
    ;;
esac

case "${KDA_PREFILL_BACKEND}" in
  auto|triton|flashkda|b12x) ;;
  *)
    echo "Invalid KDA prefill backend: ${KDA_PREFILL_BACKEND}" >&2
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

profiler_args=()
if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  if [[ "${TORCH_PROFILE_DIR}" != /* ]]; then
    TORCH_PROFILE_DIR="${VLLM_ROOT}/${TORCH_PROFILE_DIR}"
  fi
  mkdir -p -- "${TORCH_PROFILE_DIR}"
  profiler_config="$(
    "${PYTHON_BIN}" - \
      "${TORCH_PROFILE_DIR}" \
      "${TORCH_PROFILE_RECORD_SHAPES}" \
      "${TORCH_PROFILE_WITH_MEMORY}" \
      "${TORCH_PROFILE_WITH_STACK}" \
      "${TORCH_PROFILE_WITH_FLOPS}" \
      "${TORCH_PROFILE_USE_GZIP}" \
      "${TORCH_PROFILE_MAX_ITERATIONS}" <<'PY'
import json
import sys

(
    output_dir,
    record_shapes,
    with_memory,
    with_stack,
    with_flops,
    use_gzip,
    max_iterations,
) = sys.argv[1:]
print(
    json.dumps(
        {
            "profiler": "torch",
            "torch_profiler_dir": output_dir,
            "torch_profiler_record_shapes": record_shapes == "1",
            "torch_profiler_with_memory": with_memory == "1",
            "torch_profiler_with_stack": with_stack == "1",
            "torch_profiler_with_flops": with_flops == "1",
            "torch_profiler_use_gzip": use_gzip == "1",
            "torch_profiler_dump_cuda_time_total": False,
            "ignore_frontend": True,
            "delay_iterations": 0,
            "max_iterations": int(max_iterations),
        }
    )
)
PY
  )"
  profiler_args=(--profiler-config "${profiler_config}")
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

prepare_remote_dir() {
  local path=$1 quoted_path remote_command
  printf -v quoted_path '%q' "${path}"
  remote_command="mkdir -p -- ${quoted_path} 2>/dev/null"
  remote_command+=" || sudo -n install -d"
  remote_command+=" -o \$(id -un) -g \$(id -gn) -- ${quoted_path}"
  ssh "${ssh_opts[@]}" "${WORKER_IP}" "${remote_command}"
}

if [[ -n "${TORCH_PROFILE_DIR}" ]]; then
  prepare_remote_dir "${TORCH_PROFILE_DIR}"
fi

if ((sync_model)); then
  prepare_remote_dir "${MODEL_PATH}"
  echo "Rsyncing the target model to ${WORKER_IP}:${MODEL_PATH}..."
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
    echo "Rerun with --sync-code or --sync-model as appropriate." >&2
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
  --env "INSTANTTENSOR_BUFFER_SIZE=67108864"
  --env "INSTANTTENSOR_CHUNK_SIZE=8388608"
  --env "INSTANTTENSOR_CONCURRENCY=1"
  --env "INSTANTTENSOR_IO_DEPTH=3"
  --env "NCCL_NET_PLUGIN=none"
  --env "NCCL_IB_GID_INDEX=3"
  --env "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS}"
  --env "NCCL_IB_SUBNET_AWARE_ROUTING=1"
)

if ((check_only)); then
  exec "${CLUSTER_LAUNCHER}" "${cluster_args[@]}" --check-config
fi
if ((detach)); then
  cluster_args+=(-d)
fi

speculative_args=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  speculative_config=$(printf \
    '{"method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s","attention_backend":"%s"}' \
    "${NUM_SPECULATIVE_TOKENS}" "${MTP_MOE_BACKEND}" "${MTP_ATTENTION_BACKEND}")
  speculative_args=(--speculative-config "${speculative_config}")
fi

vllm_command=(
  "${VLLM_BIN}" serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1
  --disable-custom-all-reduce
  --mamba-cache-mode align
  --enable-prefix-caching
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --attention-backend "${ATTENTION_BACKEND}"
  --block-size 256
  --moe-backend "${MOE_BACKEND}"
  --linear-backend "${LINEAR_BACKEND}"
  --no-enable-flashinfer-autotune
  --load-format instanttensor
  --model-loader-extra-config \
    '{"instanttensor_copy":false,"instanttensor_distributed":false}'
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  "${speculative_args[@]}"
  --kda-prefill-backend "${KDA_PREFILL_BACKEND}"
  --reasoning-parser glm45
  --tool-call-parser glm47
  --enable-auto-tool-choice
)
vllm_command+=("${profiler_args[@]}")
vllm_command+=("${vllm_args[@]}")

exec "${CLUSTER_LAUNCHER}" "${cluster_args[@]}" exec "${vllm_command[@]}"
