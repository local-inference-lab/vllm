#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

action=${1:-verify}
variant=${2:-regular}
HF_HOME=${SPARK_HF_HOME:-"${HOME}/.cache/vllm-huggingface"}
case "${variant}" in
  regular)
    model_id=${MODEL_ID}
    model_revision=${MODEL_REVISION}
    model_shards=46
    ;;
  dspark)
    model_id=${DSPARK_MODEL_ID}
    model_revision=${DSPARK_MODEL_REVISION}
    model_shards=48
    ;;
  *)
    die "unknown model variant: ${variant}"
    ;;
esac
MODEL_CACHE_NAME="models--${model_id//\//--}"
MODEL_CACHE_ROOT="${HF_HOME}/hub/${MODEL_CACHE_NAME}"
MODEL_SNAPSHOT="${MODEL_CACHE_ROOT}/snapshots/${model_revision}"

usage() {
  cat <<'EOF'
Usage: tools/spark/model.sh download|verify|sync [regular|dspark]

Download a pinned DeepSeek V4 Flash checkpoint once and mirror its complete
Hugging Face cache repository to luxon over the primary direct link.
EOF
}

verify_local() {
  [[ -d "${MODEL_SNAPSHOT}" ]] || \
    die "model snapshot is missing: ${MODEL_SNAPSHOT}"
  local shards
  shards=$(find -L "${MODEL_SNAPSHOT}" -maxdepth 1 \
    -name "model-*-of-$(printf '%05d' "${model_shards}").safetensors" \
    -type f | wc -l)
  [[ "${shards}" -eq "${model_shards}" ]] || \
    die "found ${shards}/${model_shards} model shards"
  [[ -f "${MODEL_SNAPSHOT}/config.json" ]] || die "model config is missing"
  [[ -f "${MODEL_SNAPSHOT}/model.safetensors.index.json" ]] || \
    die "model index is missing"
  log "verified all ${model_shards} ${variant} model shards at revision ${model_revision}"
}

download() {
  [[ -x "${VLLM_ROOT}/.venv/bin/hf" ]] || \
    die "Hugging Face CLI is missing; bootstrap the environment first"
  mkdir -p "${HF_HOME}"
  HF_HOME="${HF_HOME}" "${VLLM_ROOT}/.venv/bin/hf" download \
    "${model_id}" --revision "${model_revision}"
  verify_local
}

sync_model() {
  verify_local
  local target="${REMOTE_HOST}"
  local ssh_command=(ssh)
  if ssh -o BatchMode=yes -o ConnectTimeout=5 \
    -o HostKeyAlias="${REMOTE_HOST}" "${USER}@${WORKER_ADDR}" true; then
    target="${USER}@${WORKER_ADDR}"
    ssh_command=(ssh -o "HostKeyAlias=${REMOTE_HOST}")
  else
    log "direct-link SSH unavailable; using the management hostname"
  fi
  ssh "${REMOTE_HOST}" "mkdir -p '${HF_HOME}/hub'"
  rsync -a --delete-delay --partial --partial-dir=.rsync-partial \
    --human-readable --info=progress2 \
    -e "${ssh_command[*]}" "${MODEL_CACHE_ROOT}/" \
    "${target}:${MODEL_CACHE_ROOT}/"
  ssh "${REMOTE_HOST}" \
    "test \"\$(find -L '${MODEL_SNAPSHOT}' -maxdepth 1 -name 'model-*-of-$(printf '%05d' "${model_shards}").safetensors' -type f | wc -l)\" -eq '${model_shards}'"
  log "model snapshot mirrored and verified on ${REMOTE_HOST}"
}

case "${action}" in
  download) download ;;
  verify) verify_local ;;
  sync) sync_model ;;
  -h|--help) usage ;;
  *) usage; die "unknown model action: ${action}" ;;
esac
