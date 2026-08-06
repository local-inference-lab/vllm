#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

action=${1:-status}
if [[ $# -gt 0 ]]; then
  shift
fi
profile_requested=${SPARK_PROFILE:-0}
profile_base_dir=${SPARK_PROFILE_DIR:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      profile_requested=torch
      shift
      ;;
    --profile=*)
      profile_requested=${1#*=}
      shift
      ;;
    --profile-dir)
      [[ $# -ge 2 ]] || die "--profile-dir requires a path"
      profile_base_dir=$2
      profile_requested=torch
      shift 2
      ;;
    --profile-dir=*)
      profile_base_dir=${1#*=}
      profile_requested=torch
      shift
      ;;
    *)
      die "unknown cluster argument: $1"
      ;;
  esac
done

profile_mode=off
if [[ "${action}" == start ]]; then
  case "${profile_requested,,}" in
    0|false|no|off|"") profile_mode=off ;;
    1|true|yes|on|torch) profile_mode=torch ;;
    *) die "--profile supports only torch profiling" ;;
  esac
  if [[ -n "${profile_base_dir}" && \
        "${profile_base_dir}" != /* && \
        "${profile_base_dir}" != *"://"* ]]; then
    die "--profile-dir must be an absolute path or URI"
  fi
elif [[ -n "${profile_base_dir}" ]]; then
  die "--profile-dir is valid only with start"
fi

LOG_DIR=${SPARK_LOG_DIR:-"${STATE_ROOT}/logs"}
HEAD_PID_FILE="${STATE_ROOT}/head.pid"
WORKER_PID_FILE="${STATE_ROOT}/worker.pid"
HEAD_LOG="${LOG_DIR}/head.log"
WORKER_LOG="${LOG_DIR}/worker.log"
HEAD_UNIT=vllm-spark-head.service
WORKER_UNIT=vllm-spark-worker.service

usage() {
  cat <<'EOF'
Usage: tools/spark/cluster.sh start [--profile] [--profile-dir DIR]
       tools/spark/cluster.sh stop|status|logs|wait|preflight
       tools/spark/cluster.sh profile-start|profile-stop

Start luxon rank 1 first, followed by the tachyon API rank. Logs and PID files
live under ~/.local/state/vllm-spark. Start mirrors tachyon's live vLLM and
B12X source trees to luxon unless SPARK_SYNC_SOURCE_ON_START=0.
Profiling is armed at launch, then controlled through profile-start/profile-stop.
EOF
}

unit_active_local() {
  systemctl --user is-active --quiet "$1"
}

unit_active_remote() {
  ssh "${REMOTE_HOST}" \
    "systemctl --user is-active --quiet '$1'"
}

preflight() {
  [[ -x "${VENV_PYTHON}" ]] || die "local .venv is missing"
  [[ -f "${ARTIFACT_ROOT}/manifest.json" ]] || die "local manifest is missing"
  assert_editable_sources "${VENV_PYTHON}" "${VLLM_ROOT}" "${B12X_ROOT}"
  require_command systemctl
  require_command systemd-run
  require_command loginctl
  local local_linger remote_linger
  local_linger=$(loginctl show-user "$(id -un)" -p Linger --value)
  remote_linger=$(ssh "${REMOTE_HOST}" \
    'loginctl show-user "$(id -un)" -p Linger --value')
  [[ "${local_linger}" == yes ]] || \
    die "user lingering is disabled on tachyon; run sudo loginctl enable-linger $(id -un)"
  [[ "${remote_linger}" == yes ]] || \
    die "user lingering is disabled on luxon; run sudo loginctl enable-linger $(id -un)"
  local model_variant=regular
  if [[ "${VLLM_ENABLE_DSPARK:-0}" == 1 ]]; then
    model_variant=dspark
  fi
  "${SPARK_DIR}/model.sh" verify "${model_variant}"
  "${SPARK_DIR}/network.sh" verify
  local local_manifest remote_manifest
  local_manifest=$(sha256sum "${ARTIFACT_ROOT}/manifest.json" | cut -d ' ' -f 1)
  remote_manifest=$(ssh "${REMOTE_HOST}" \
    "sha256sum '${ARTIFACT_ROOT}/manifest.json' | cut -d ' ' -f 1")
  [[ "${local_manifest}" == "${remote_manifest}" ]] || \
    die "local and remote artifact manifests differ"
  ssh "${REMOTE_HOST}" \
    "test -x '${REMOTE_VLLM_ROOT}/.venv/bin/python' && test -d '${REMOTE_B12X_ROOT}' && command -v systemctl >/dev/null && command -v systemd-run >/dev/null && systemctl --user is-system-running >/dev/null"
  ssh "${REMOTE_HOST}" \
    "cd '${REMOTE_VLLM_ROOT}' && source tools/spark/lib.sh && \
      assert_editable_sources '${REMOTE_VLLM_ROOT}/.venv/bin/python' \
      '${REMOTE_VLLM_ROOT}' '${REMOTE_B12X_ROOT}'"
  log "cluster preflight passed"
}

start() {
  if unit_active_local "${HEAD_UNIT}" || unit_active_remote "${WORKER_UNIT}"; then
    die "a cluster process is already running"
  fi
  case "${SPARK_SYNC_SOURCE_ON_START:-1}" in
    1) "${SPARK_DIR}/sync.sh" --source-only ;;
    0) log "skipping source sync because SPARK_SYNC_SOURCE_ON_START=0" ;;
    *) die "SPARK_SYNC_SOURCE_ON_START must be 0 or 1" ;;
  esac
  preflight
  local enable_mtp=${VLLM_ENABLE_MTP:-0}
  local enable_dspark=${VLLM_ENABLE_DSPARK:-0}
  local gpu_memory_utilization
  if [[ ${GPU_MEMORY_UTILIZATION+x} ]]; then
    gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
  elif [[ "${enable_dspark}" == 1 ]]; then
    gpu_memory_utilization=${SPARK_DSPARK_GPU_MEMORY_UTILIZATION:-0.80}
  else
    gpu_memory_utilization=0.80
  fi
  local num_speculative_tokens
  if [[ ${NUM_SPECULATIVE_TOKENS+x} ]]; then
    num_speculative_tokens=${NUM_SPECULATIVE_TOKENS}
  elif [[ "${enable_dspark}" == 1 ]]; then
    num_speculative_tokens=5
  else
    num_speculative_tokens=2
  fi
  local dcp_size=${DCP_SIZE:-2}
  local dcp_comm_backend=${DCP_COMM_BACKEND:-a2a}
  local use_b12x_dcp_a2a=${VLLM_USE_B12X_DCP_A2A:-0}
  local dcp_a2a_max_tokens=${VLLM_DCP_A2A_MAX_TOKENS:-64}
  local dcp_a2a_large_backend=${VLLM_DCP_A2A_LARGE_BACKEND:-ag_rs}
  local enable_prefix_caching=${ENABLE_PREFIX_CACHING:-1}
  local local_profile_dir=
  local remote_profile_dir=
  if [[ "${profile_mode}" == torch ]]; then
    if [[ -z "${profile_base_dir}" ]]; then
      profile_base_dir="${STATE_ROOT}/profiles/ds4-$(date -u +%Y%m%d-%H%M%S)"
    fi
    local_profile_dir="${profile_base_dir%/}/rank-0"
    remote_profile_dir="${profile_base_dir%/}/rank-1"
    if [[ "${profile_base_dir}" != *"://"* ]]; then
      mkdir -p "${local_profile_dir}"
      ssh "${REMOTE_HOST}" "mkdir -p '${remote_profile_dir}'"
    fi
    log "Torch profiling armed"
    log "tachyon traces: ${local_profile_dir}"
    log "luxon traces: ${remote_profile_dir}"
  fi
  local profile_env=(
    VLLM_PROFILE="${profile_mode}"
    VLLM_TORCH_PROFILER_WITH_STACK="${VLLM_TORCH_PROFILER_WITH_STACK:-1}"
    VLLM_TORCH_PROFILER_RECORD_SHAPES="${VLLM_TORCH_PROFILER_RECORD_SHAPES:-0}"
    VLLM_TORCH_PROFILER_WITH_MEMORY="${VLLM_TORCH_PROFILER_WITH_MEMORY:-0}"
    VLLM_TORCH_PROFILER_WITH_FLOPS="${VLLM_TORCH_PROFILER_WITH_FLOPS:-0}"
    VLLM_TORCH_PROFILER_USE_GZIP="${VLLM_TORCH_PROFILER_USE_GZIP:-1}"
    VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL="${VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL:-0}"
    VLLM_PROFILE_IGNORE_FRONTEND="${VLLM_PROFILE_IGNORE_FRONTEND:-1}"
    VLLM_TORCH_PROFILER_DELAY_ITERATIONS="${VLLM_TORCH_PROFILER_DELAY_ITERATIONS:-0}"
    VLLM_TORCH_PROFILER_MAX_ITERATIONS="${VLLM_TORCH_PROFILER_MAX_ITERATIONS:-4}"
    VLLM_TORCH_PROFILER_WARMUP_ITERATIONS="${VLLM_TORCH_PROFILER_WARMUP_ITERATIONS:-0}"
    VLLM_TORCH_PROFILER_ACTIVE_ITERATIONS="${VLLM_TORCH_PROFILER_ACTIVE_ITERATIONS:-5}"
    VLLM_TORCH_PROFILER_WAIT_ITERATIONS="${VLLM_TORCH_PROFILER_WAIT_ITERATIONS:-0}"
  )
  local kv_cache_memory_bytes
  if [[ ${KV_CACHE_MEMORY_BYTES+x} ]]; then
    kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}
  elif [[ "${enable_dspark}" == 1 ]]; then
    kv_cache_memory_bytes=${SPARK_DSPARK_KV_CACHE_MEMORY_BYTES:-}
  else
    kv_cache_memory_bytes=${SPARK_KV_CACHE_MEMORY_BYTES:-}
  fi
  mkdir -p "${LOG_DIR}"
  rm -f "${HEAD_LOG}" "${HEAD_PID_FILE}"

  local remote_command
  printf -v remote_command '%q ' \
    systemd-run --user --quiet --collect --service-type=exec \
    --unit="${WORKER_UNIT}" \
    --working-directory="${REMOTE_VLLM_ROOT}" \
    --property="StandardOutput=append:${WORKER_LOG}" \
    --property="StandardError=append:${WORKER_LOG}" \
    /usr/bin/env NODE_RANK=1 NNODES=2 MASTER_ADDR="${HEAD_ADDR}" \
    MASTER_PORT=29501 HEADLESS=1 VLLM_HOST_IP="${WORKER_ADDR}" \
    "${profile_env[@]}" \
    VLLM_TORCH_PROFILER_DIR="${remote_profile_dir}" \
    VLLM_ENABLE_MTP="${enable_mtp}" \
    VLLM_ENABLE_DSPARK="${enable_dspark}" \
    GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
    NUM_SPECULATIVE_TOKENS="${num_speculative_tokens}" \
    DSPARK_SPS_CURVE="${DSPARK_SPS_CURVE:-}" \
    DSPARK_CONFIDENCE_THRESHOLD="${DSPARK_CONFIDENCE_THRESHOLD:-}" \
    DSPARK_BUDGET_FRAC="${DSPARK_BUDGET_FRAC:-}" \
    DSPARK_CONFIDENCE_TEMPERATURE="${DSPARK_CONFIDENCE_TEMPERATURE:-}" \
    DSPARK_SPS_OVERHEAD_MS="${DSPARK_SPS_OVERHEAD_MS:-}" \
    VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="${VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE:-0}" \
    VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH="${VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH:-0}" \
    VLLM_DSPARK_SPS_DEBUG="${VLLM_DSPARK_SPS_DEBUG:-0}" \
    B12X_MICRO_DYNAMIC_CUTOVER_PAIRS="${B12X_MICRO_DYNAMIC_CUTOVER_PAIRS:-64}" \
    B12X_W4A8_TINY_DECODE="${B12X_W4A8_TINY_DECODE:-1}" \
    DCP_SIZE="${dcp_size}" DCP_COMM_BACKEND="${dcp_comm_backend}" \
    VLLM_USE_B12X_DCP_A2A="${use_b12x_dcp_a2a}" \
    VLLM_DCP_A2A_MAX_TOKENS="${dcp_a2a_max_tokens}" \
    VLLM_DCP_A2A_LARGE_BACKEND="${dcp_a2a_large_backend}" \
    ENABLE_PREFIX_CACHING="${enable_prefix_caching}" \
    KV_CACHE_MEMORY_BYTES="${kv_cache_memory_bytes}" \
    "${REMOTE_VLLM_ROOT}/serve-ds4-flash-spark.sh"
  ssh "${REMOTE_HOST}" \
    "mkdir -p '${LOG_DIR}'; rm -f '${WORKER_LOG}' '${WORKER_PID_FILE}'; ${remote_command}"
  sleep 3
  unit_active_remote "${WORKER_UNIT}" || \
    die "luxon worker exited; inspect ${WORKER_LOG}"
  ssh "${REMOTE_HOST}" \
    "systemctl --user show -p MainPID --value '${WORKER_UNIT}' >'${WORKER_PID_FILE}'"

  systemd-run --user --quiet --collect --service-type=exec \
    --unit="${HEAD_UNIT}" \
    --working-directory="${VLLM_ROOT}" \
    --property="StandardOutput=append:${HEAD_LOG}" \
    --property="StandardError=append:${HEAD_LOG}" \
    /usr/bin/env NODE_RANK=0 NNODES=2 MASTER_ADDR="${HEAD_ADDR}" \
    MASTER_PORT=29501 HEADLESS=0 VLLM_HOST_IP="${HEAD_ADDR}" \
    "${profile_env[@]}" \
    VLLM_TORCH_PROFILER_DIR="${local_profile_dir}" \
    VLLM_ENABLE_MTP="${enable_mtp}" \
    VLLM_ENABLE_DSPARK="${enable_dspark}" \
    GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
    NUM_SPECULATIVE_TOKENS="${num_speculative_tokens}" \
    DSPARK_SPS_CURVE="${DSPARK_SPS_CURVE:-}" \
    DSPARK_CONFIDENCE_THRESHOLD="${DSPARK_CONFIDENCE_THRESHOLD:-}" \
    DSPARK_BUDGET_FRAC="${DSPARK_BUDGET_FRAC:-}" \
    DSPARK_CONFIDENCE_TEMPERATURE="${DSPARK_CONFIDENCE_TEMPERATURE:-}" \
    DSPARK_SPS_OVERHEAD_MS="${DSPARK_SPS_OVERHEAD_MS:-}" \
    VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE="${VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE:-0}" \
    VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH="${VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH:-0}" \
    VLLM_DSPARK_SPS_DEBUG="${VLLM_DSPARK_SPS_DEBUG:-0}" \
    B12X_MICRO_DYNAMIC_CUTOVER_PAIRS="${B12X_MICRO_DYNAMIC_CUTOVER_PAIRS:-64}" \
    B12X_W4A8_TINY_DECODE="${B12X_W4A8_TINY_DECODE:-1}" \
    DCP_SIZE="${dcp_size}" DCP_COMM_BACKEND="${dcp_comm_backend}" \
    VLLM_USE_B12X_DCP_A2A="${use_b12x_dcp_a2a}" \
    VLLM_DCP_A2A_MAX_TOKENS="${dcp_a2a_max_tokens}" \
    VLLM_DCP_A2A_LARGE_BACKEND="${dcp_a2a_large_backend}" \
    ENABLE_PREFIX_CACHING="${enable_prefix_caching}" \
    KV_CACHE_MEMORY_BYTES="${kv_cache_memory_bytes}" \
    "${VLLM_ROOT}/serve-ds4-flash-spark.sh"
  sleep 3
  if ! unit_active_local "${HEAD_UNIT}"; then
    ssh "${REMOTE_HOST}" \
      "systemctl --user stop '${WORKER_UNIT}'" 2>/dev/null || true
    die "tachyon head exited; inspect ${HEAD_LOG}"
  fi
  systemctl --user show -p MainPID --value "${HEAD_UNIT}" >"${HEAD_PID_FILE}"
  log "cluster ranks started; use '$0 wait' for API readiness"
}

stop() {
  systemctl --user stop "${HEAD_UNIT}" 2>/dev/null || true
  ssh "${REMOTE_HOST}" \
    "systemctl --user stop '${WORKER_UNIT}'" 2>/dev/null || true
  rm -f "${HEAD_PID_FILE}"
  ssh "${REMOTE_HOST}" "rm -f '${WORKER_PID_FILE}'"
  log "cluster stop requested"
}

status() {
  if unit_active_local "${HEAD_UNIT}"; then
    printf 'tachyon head: running (pid %s)\n' \
      "$(systemctl --user show -p MainPID --value "${HEAD_UNIT}")"
  else
    printf 'tachyon head: stopped\n'
  fi
  if unit_active_remote "${WORKER_UNIT}"; then
    printf 'luxon worker: running (pid %s)\n' \
      "$(ssh "${REMOTE_HOST}" \
        "systemctl --user show -p MainPID --value '${WORKER_UNIT}'")"
  else
    printf 'luxon worker: stopped\n'
  fi
}

logs() {
  printf '%s\n' '===== tachyon head ====='
  tail -n 120 "${HEAD_LOG}" 2>/dev/null || true
  printf '%s\n' '===== luxon worker ====='
  ssh "${REMOTE_HOST}" "tail -n 120 '${WORKER_LOG}' 2>/dev/null" || true
}

wait_ready() {
  for _ in {1..180}; do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
      log "OpenAI server is ready at http://127.0.0.1:8000"
      return
    fi
    unit_active_local "${HEAD_UNIT}" || die "tachyon head exited during load"
    unit_active_remote "${WORKER_UNIT}" || die "luxon worker exited during load"
    sleep 10
  done
  die "server did not become healthy within 30 minutes"
}

profile_request() {
  local endpoint=$1
  local timeout=$2
  local response
  response=$(curl -fsS --max-time "${timeout}" -X POST \
    "http://127.0.0.1:8000/${endpoint}")
  [[ -z "${response}" ]] || printf '%s\n' "${response}"
  log "${endpoint} completed"
}

case "${action}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  logs) logs ;;
  wait) wait_ready ;;
  preflight) preflight ;;
  profile-start) profile_request start_profile 60 ;;
  profile-stop) profile_request stop_profile "${SPARK_PROFILE_STOP_TIMEOUT:-1800}" ;;
  -h|--help) usage ;;
  *) usage; die "unknown cluster action: ${action}" ;;
esac
