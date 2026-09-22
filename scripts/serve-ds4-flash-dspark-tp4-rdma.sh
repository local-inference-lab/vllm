#!/usr/bin/env bash
# shellcheck disable=SC2029
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${VLLM_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
B12X_ROOT="${B12X_ROOT:-/home/luke/projects/b12x}"
TP_SIZE="${TP_SIZE:-4}"
B12X_COMPILE_CACHE_DIR="${B12X_COMPILE_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/b12x/compile/vllm-tp${TP_SIZE}}"
NCCL_ROOT="${NCCL_ROOT:-/home/luke/projects/nccl-2.30.7}"
NCCL_LIB="${NCCL_LIB:-${NCCL_ROOT}/build/lib/libnccl.so.2.30.7}"
SPARK_ROOT="${SPARK_ROOT:-/home/luke/projects/spark-vllm-docker}"
CLUSTER_LAUNCHER="${CLUSTER_LAUNCHER:-${SPARK_ROOT}/launch-cluster.sh}"
HEAD_IP="${HEAD_IP:-192.168.42.223}"
LUXON_IP="${LUXON_IP:-192.168.42.110}"
GRAVITON_IP="${GRAVITON_IP:-192.168.42.55}"
CHRONITON_IP="${CHRONITON_IP:-192.168.42.78}"
WORKER_IPS=("${LUXON_IP}" "${GRAVITON_IP}")
case "${TP_SIZE}" in
  3) ;;
  4) WORKER_IPS+=("${CHRONITON_IP}") ;;
  *) echo "TP_SIZE must be 3 or 4; got '${TP_SIZE}'" >&2; exit 2 ;;
esac
NODE_IPS="${HEAD_IP}$(printf ',%s' "${WORKER_IPS[@]}")"
ETH_IF="${ETH_IF:-enP7s7}"
IB_IF="${IB_IF:-rocep1s0f0,roceP2p1s0f0}"
NCCL_IB_HCA="${NCCL_IB_HCA:-${IB_IF}}"
NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-1}"
ALLREDUCE="${ALLREDUCE:-rocenante}"
ROCE_ALLREDUCE_MAX_SIZE="${ROCE_ALLREDUCE_MAX_SIZE:-2MB}"
ROCE_ALLGATHER_MAX_SIZE="${ROCE_ALLGATHER_MAX_SIZE:-16MB}"
MASTER_PORT="${MASTER_PORT:-29656}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_ds4_flash_dspark_tp${TP_SIZE}}"
IMAGE_NAME="${IMAGE_NAME:-vllm-node-eugr-20260712:latest}"
CONTAINER_MEMORY_GB="${CONTAINER_MEMORY_GB:-108}"
CONTAINER_MEMORY_SWAP_GB="${CONTAINER_MEMORY_SWAP_GB:-112}"
PYTHON_BIN="${PYTHON_BIN:-${VLLM_ROOT}/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-${VLLM_ROOT}/.venv/bin/vllm}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/vllm-huggingface}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash-0731}"
MODEL_REVISION="${MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}"
TOKENIZER_MODE="${TOKENIZER_MODE:-deepseek_v4}"
ENGRAM_CONFIG="${ENGRAM_CONFIG:-}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
SECCOMP_PROFILE="${SECCOMP_PROFILE:-}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-500000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-10737418240}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-7}"
DSPARK_DRAFT_ATTENTION_BACKEND="${DSPARK_DRAFT_ATTENTION_BACKEND:-auto}"
DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-probabilistic}"
DSPARK_ADAPTIVE_VERIFICATION="${DSPARK_ADAPTIVE_VERIFICATION:-0}"
DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE="${DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE:-1.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

sync_code=0
check_only=0
detach=0
vllm_args=()

usage() {
  cat <<EOF
Usage: $0 [launcher options] [-- vLLM options]

Launch DeepSeek-V4-Flash with DSpark speculative decoding and TP=4 across
tachyon, luxon, graviton, and chroniton through the Spark cluster launcher.
TP_SIZE=3 uses tachyon, luxon, and graviton for models supporting TP3.
The launcher runs one native vLLM rank per node, uses the management LAN for
bootstrap, and uses both RoCE interfaces through the ConnectX-7 switch.
ALLREDUCE=rocenante (default) uses b12x collectives; ALLREDUCE=nccl uses NCCL.

Launcher options:
  --sync-code   Mirror local vllm/ and b12x/ runtime packages to all workers.
  --check       Validate the selected nodes and Spark networking without launching.
  --detach      Run the head rank in the background; use docker logs to follow it.
  --no-spec     Plain decode: no DSpark drafter (same as NUM_SPECULATIVE_TOKENS=0).
  --spec N      DSpark with N speculative tokens (same as NUM_SPECULATIVE_TOKENS=N).
  -h, --help    Show this help.

Environment overrides include TP_SIZE, ALLREDUCE, ROCE_ALLREDUCE_MAX_SIZE,
ROCE_ALLGATHER_MAX_SIZE, HEAD_IP, LUXON_IP, GRAVITON_IP, CHRONITON_IP,
MODEL_ID, MODEL_REVISION, HF_CACHE, MAX_MODEL_LEN, MAX_NUM_SEQS,
NUM_SPECULATIVE_TOKENS, KV_CACHE_MEMORY_BYTES, GPU_MEMORY_UTILIZATION,
B12X_ROOT, B12X_COMPILE_CACHE_DIR, NCCL_ROOT, IMAGE_NAME, CONTAINER_MEMORY_GB,
TOKENIZER_MODE, ENGRAM_CONFIG, LIMIT_MM_PER_PROMPT, SECCOMP_PROFILE,
DRAFT_SAMPLE_METHOD, DSPARK_ADAPTIVE_VERIFICATION and
DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE.
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
  auto|B12X|FLASHINFER_MLA_SPARSE_DSV4|FLASHMLA_SPARSE_DSV4) ;;
  *)
    echo "DSPARK_DRAFT_ATTENTION_BACKEND must be auto, B12X," \
      "FLASHINFER_MLA_SPARSE_DSV4, or FLASHMLA_SPARSE_DSV4" >&2
    exit 2
    ;;
esac

snapshot="${HF_CACHE}/hub/models--${MODEL_ID//\//--}/snapshots/${MODEL_REVISION}"
for path in \
  "${VLLM_ROOT}" \
  "${B12X_ROOT}" \
  "${B12X_COMPILE_CACHE_DIR}" \
  "${NCCL_ROOT}" \
  "${HF_CACHE}" \
  "${SECCOMP_PROFILE}" \
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
if [[ ! -f "${B12X_ROOT}/b12x/__init__.py" ]]; then
  echo "Local b12x source tree not found under ${B12X_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${NCCL_LIB}" ]]; then
  echo "Patched NCCL library not found: ${NCCL_LIB}" >&2
  exit 1
fi

ssh_opts=(
  -o BatchMode=yes
  -o ConnectTimeout=5
  -o StrictHostKeyChecking=no
)
for worker_ip in "${WORKER_IPS[@]}"; do
  if ! ssh "${ssh_opts[@]}" "${worker_ip}" true; then
    echo "Passwordless SSH to worker ${worker_ip} failed." >&2
    exit 1
  fi
done

if ((sync_code)); then
  for worker_ip in "${WORKER_IPS[@]}"; do
    echo "Mirroring vLLM runtime source to ${worker_ip}..."
    rsync -a --delete \
      --exclude='__pycache__/' \
      --exclude='*.py[co]' \
      "${VLLM_ROOT}/vllm/" \
      "${worker_ip}:${VLLM_ROOT}/vllm/"
    echo "Mirroring b12x runtime source to ${worker_ip}..."
    rsync -a --delete \
      --exclude='__pycache__/' \
      --exclude='*.py[co]' \
      "${B12X_ROOT}/b12x/" \
      "${worker_ip}:${B12X_ROOT}/b12x/"
  done
fi

remote_files=(
  "${PYTHON_BIN}"
  "${VLLM_BIN}"
  "${VLLM_ROOT}/vllm/__init__.py"
  "${B12X_ROOT}/b12x/__init__.py"
  "${NCCL_LIB}"
  "${snapshot}/config.json"
)
for worker_ip in "${WORKER_IPS[@]}"; do
  for path in "${remote_files[@]}"; do
    printf -v remote_path '%q' "${path}"
    if ! ssh "${ssh_opts[@]}" "${worker_ip}" "test -e ${remote_path}"; then
      echo "Required worker path is missing: ${worker_ip}:${path}" >&2
      echo "Rerun with --sync-code, or copy the model snapshot." >&2
      exit 1
    fi
  done
done

loader_check='from importlib.metadata import entry_points; raise SystemExit(not any(ep.name == "b12x_loader" for ep in entry_points(group="vllm.general_plugins")))'
if ! "${PYTHON_BIN}" -c "${loader_check}"; then
  echo "Install b12x in ${PYTHON_BIN}'s environment to register its loader." >&2
  exit 1
fi
printf -v remote_loader_check '%q -c %q' "${PYTHON_BIN}" "${loader_check}"
for worker_ip in "${WORKER_IPS[@]}"; do
  if ! ssh "${ssh_opts[@]}" "${worker_ip}" "${remote_loader_check}"; then
    echo "b12x loader registration is missing on ${worker_ip}; install b12x in ${PYTHON_BIN}'s environment." >&2
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
for worker_ip in "${WORKER_IPS[@]}"; do
  worker_digest="$(
    ssh "${ssh_opts[@]}" "${worker_ip}" "${remote_digest_command}"
  )"
  if [[ "${local_digest}" != "${worker_digest}" ]]; then
    echo "vLLM/b12x runtime source differs on ${worker_ip}." >&2
    echo "Rerun with --sync-code so all TP ranks execute identical code." >&2
    exit 1
  fi
done

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image is missing locally: ${IMAGE_NAME}" >&2
  exit 1
fi
for worker_ip in "${WORKER_IPS[@]}"; do
  if ! ssh "${ssh_opts[@]}" "${worker_ip}" \
    "docker image inspect ${IMAGE_NAME} >/dev/null 2>&1"; then
    echo "Docker image is missing on ${worker_ip}: ${IMAGE_NAME}" >&2
    exit 1
  fi
done

mkdir -p -- "${B12X_COMPILE_CACHE_DIR}"
printf -v remote_cache_dir '%q' "${B12X_COMPILE_CACHE_DIR}"
for worker_ip in "${WORKER_IPS[@]}"; do
  ssh "${ssh_opts[@]}" "${worker_ip}" "mkdir -p -- ${remote_cache_dir}"
done

mount_args="-v ${VLLM_ROOT}:${VLLM_ROOT}"
mount_args+=" -v ${B12X_ROOT}:${B12X_ROOT}"
mount_args+=" -v ${B12X_COMPILE_CACHE_DIR}:${B12X_COMPILE_CACHE_DIR}"
mount_args+=" -v ${NCCL_ROOT}:${NCCL_ROOT}:ro"
mount_args+=" -v ${HF_CACHE}:${HF_CACHE}:ro"
if [[ -n "${SECCOMP_PROFILE}" ]]; then
  cached_seccomp="${B12X_COMPILE_CACHE_DIR}/serving-seccomp.json"
  if [[ "${SECCOMP_PROFILE}" != "${cached_seccomp}" ]]; then
    cp -- "${SECCOMP_PROFILE}" "${cached_seccomp}"
  fi
  for worker_ip in "${WORKER_IPS[@]}"; do
    rsync -a "${cached_seccomp}" "${worker_ip}:${cached_seccomp}"
  done
  mount_args+=" --security-opt seccomp=${cached_seccomp}"
fi
if [[ -n "${VLLM_SPARK_EXTRA_DOCKER_ARGS:-}" ]]; then
  mount_args+=" ${VLLM_SPARK_EXTRA_DOCKER_ARGS}"
fi
export VLLM_SPARK_EXTRA_DOCKER_ARGS="${mount_args}"

cluster_args=(
  --nodes "${NODE_IPS}"
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
  --env "B12X_COMPILE_CACHE_DIR=${B12X_COMPILE_CACHE_DIR}"
  --env "B12X_WEIGHTS_COMPILE_WORKERS=${B12X_WEIGHTS_COMPILE_WORKERS:-${B12X_COMPILE_WORKERS:-16}}"
  --env "B12X_STATE_COMPILE_WORKERS=${B12X_STATE_COMPILE_WORKERS:-${B12X_COMPILE_WORKERS:-16}}"
  --env "B12X_BIND_COMPILE_WORKERS=${B12X_BIND_COMPILE_WORKERS:-${B12X_COMPILE_WORKERS:-4}}"
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  --env "SAFETENSORS_FAST_GPU=1"
  --env "OMP_NUM_THREADS=16"
  --env "VLLM_WORKER_MULTIPROC_METHOD=spawn"
  --env "HF_HOME=${HF_CACHE}"
  --env "HF_HUB_OFFLINE=1"
  --env "TRANSFORMERS_OFFLINE=1"
  --env "VLLM_PLUGINS=${VLLM_PLUGINS:-b12x_loader}"
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
  --env "VLLM_ENABLE_PCIE_ALLREDUCE=0"
  --env "NCCL_NET_PLUGIN=none"
  --env "LD_PRELOAD=${NCCL_LIB}"
  --env "VLLM_NCCL_SO_PATH=${NCCL_LIB}"
  --env "NCCL_NET=IB"
  --env "NCCL_IB_DISABLE=0"
  --env "NCCL_IB_HCA=${NCCL_IB_HCA}"
  --env "NCCL_IB_GID_INDEX=3"
  --env "NCCL_IB_TC=${NCCL_IB_TC:-106}"
  --env "B12X_ROCE_TRAFFIC_CLASS=${B12X_ROCE_TRAFFIC_CLASS:-${NCCL_IB_TC:-106}}"
  --env "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS}"
  --env "NCCL_NET_MERGE_POLICY=ALL"
  --env "NCCL_NET_MERGE_LEVEL=SYS"
  --env "NCCL_CUMEM_ENABLE=0"
  --env "NCCL_RUNTIME_CONNECT=1"
  --env "NCCL_P2P_LEVEL=SYS"
  --env "NCCL_IGNORE_CPU_AFFINITY=1"
)
for node_ip in "${HEAD_IP}" "${WORKER_IPS[@]}"; do
  cluster_args+=(--node-ib-if "${node_ip}=${IB_IF}")
done
if [[ -n "${NCCL_MAX_NCHANNELS:-}" ]]; then
  cluster_args+=(--env "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}")
fi

allreduce_args=()
if [[ "${ALLREDUCE}" == rocenante ]]; then
  cluster_args+=(
    --env "VLLM_ENABLE_ROCE_ALLREDUCE=1"
    --env "VLLM_ROCE_ALLREDUCE_MAX_SIZE=${ROCE_ALLREDUCE_MAX_SIZE}"
    --env "VLLM_ROCE_ALLGATHER_MAX_SIZE=${ROCE_ALLGATHER_MAX_SIZE}"
    --env "B12X_ROCE_CACHE_DIR=${B12X_COMPILE_CACHE_DIR}/roce"
  )
else
  cluster_args+=(--env "VLLM_ENABLE_ROCE_ALLREDUCE=0")
  allreduce_args+=(--disable-custom-all-reduce)
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
  speculative_config="$(
    "${PYTHON_BIN}" - "${NUM_SPECULATIVE_TOKENS}" "${TP_SIZE}" \
      "${DSPARK_DRAFT_ATTENTION_BACKEND}" "${DRAFT_SAMPLE_METHOD}" \
      "${DSPARK_ADAPTIVE_VERIFICATION}" \
      "${DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE}" <<'PY'
import json
import math
import sys

tokens, tp_size, attention, sampling, adaptive, scale = sys.argv[1:]
if sampling not in {"greedy", "probabilistic"}:
    raise SystemExit("DRAFT_SAMPLE_METHOD must be greedy or probabilistic")
booleans = {"1": True, "true": True, "yes": True, "on": True,
            "0": False, "false": False, "no": False, "off": False}
if adaptive.lower() not in booleans:
    raise SystemExit("DSPARK_ADAPTIVE_VERIFICATION must be a boolean")
adaptive = booleans[adaptive.lower()]
try:
    scale = float(scale)
except ValueError:
    raise SystemExit("DSPARK_ADAPTIVE_VERIFICATION_COST_SCALE must be positive")
if not math.isfinite(scale) or scale <= 0 or (not adaptive and scale != 1):
    raise SystemExit("Verification cost scale must be positive; changing it requires adaptive verification")
config = {
    "method": "dspark",
    "num_speculative_tokens": int(tokens),
    "draft_tensor_parallel_size": int(tp_size),
    "draft_sample_method": sampling,
    "rejection_sample_method": "standard",
    "enable_adaptive_verification": adaptive,
    "adaptive_verification_cost_scale": scale,
}
if attention != "auto":
    config["attention_backend"] = attention
print(json.dumps(config))
PY
  )"
  speculative_args=(--speculative-config "${speculative_config}")
fi

vllm_command=(
  "${VLLM_BIN}" serve "${MODEL_ID}"
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TP_SIZE}"
  --decode-context-parallel-size 1
  "${allreduce_args[@]}"
  --kv-cache-dtype fp8
  --block-size 256
  --load-format b12x
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
  --tokenizer-mode "${TOKENIZER_MODE}"
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --reasoning-config
  '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
  --default-chat-template-kwargs.thinking=true
  --default-chat-template-kwargs.reasoning_effort=high
)
vllm_command+=("${speculative_args[@]}")
if [[ -n "${ENGRAM_CONFIG}" ]]; then
  vllm_command+=(--engram-config "${ENGRAM_CONFIG}")
fi
if [[ -n "${LIMIT_MM_PER_PROMPT}" ]]; then
  vllm_command+=(--limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}")
fi
vllm_command+=("${vllm_args[@]}")

if ((NUM_SPECULATIVE_TOKENS > 0)); then
  spec_summary="DSpark, ${NUM_SPECULATIVE_TOKENS} speculative tokens"
else
  spec_summary="plain decode, no drafter"
fi
cat <<BANNER
Launching ${SERVED_MODEL_NAME} TP=${TP_SIZE} on ${NODE_IPS}
  all-reduce:      ${ALLREDUCE} over the switched RoCE fabric
  speculation:     ${spec_summary}
  max seqs:        ${MAX_NUM_SEQS} (cudagraph capture up to ${max_cudagraph_capture_size})
  context / KV:    ${MAX_MODEL_LEN} tokens, ${KV_CACHE_MEMORY_BYTES} bytes
  b12x source:     ${B12X_ROOT}
BANNER

exec "${CLUSTER_LAUNCHER}" "${cluster_args[@]}" exec "${vllm_command[@]}"
