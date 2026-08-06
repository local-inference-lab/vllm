#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

mode=all
dry_run=0

usage() {
  cat <<'EOF'
Usage: tools/spark/sync.sh [--source-only|--artifacts-only|--all] [--dry-run]

Mirror the live vLLM/B12X source and/or built artifacts to luxon. The command
always treats tachyon's live source trees as authoritative.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-only)
      mode=source
      shift
      ;;
    --artifacts-only)
      mode=artifacts
      shift
      ;;
    --all)
      mode=all
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown sync argument: $1"
      ;;
  esac
done

require_command rsync
require_command ssh
if [[ "${mode}" == source ]]; then
  assert_runtime_inputs
else
  assert_inputs
fi
mkdir -p "${STATE_ROOT}"

rsync_flags=(-a --delete-delay --human-readable)
if [[ ${dry_run} -eq 1 ]]; then
  rsync_flags+=(--dry-run --itemize-changes)
fi
source_excludes=(
  --exclude .git/
  --exclude .venv/
  --exclude .deps/
  --exclude __pycache__/
  --exclude '*.py[co]'
  --exclude '*.ncu-rep'
  --exclude '*.nsys-rep'
  --exclude '/results.*.tsv'
  --exclude '/run.*.log'
  --exclude .pytest_cache/
  --exclude .ruff_cache/
  --exclude .mypy_cache/
  --exclude '*.egg-info/'
  --exclude build/
  --exclude dist/
  --exclude target/
  --exclude 'cmake-build-*/'
  --exclude '.ninja_*'
  --exclude compile_commands.json
  --exclude 'csrc/libtorch_stable/moe/marlin_moe_wna16/sm*_kernel_*.cu'
  --exclude csrc/libtorch_stable/moe/marlin_moe_wna16/kernel_selector.h
  --exclude 'csrc/libtorch_stable/quantization/marlin/sm*_kernel_*.cu'
  --exclude csrc/libtorch_stable/quantization/marlin/kernel_selector.h
  --exclude vllm/_version.py
  --exclude vllm/vllm-rs
  --exclude vllm/third_party/flashmla/flash_mla_interface.py
  --exclude vllm/third_party/deep_gemm/
  --exclude vllm/third_party/fmha_sm100/
  --exclude vllm/third_party/triton_kernels/
  --exclude vllm/vllm_flash_attn/cute/
  --exclude vllm/vllm_flash_attn/layers/
  --exclude vllm/vllm_flash_attn/ops/
  --exclude '*.so'
  --exclude '*.so.*'
)

sync_vllm_source() {
  log "syncing the editable vLLM source to ${REMOTE_HOST}"
  rsync "${rsync_flags[@]}" \
    "${source_excludes[@]}" \
    --exclude .spark-artifacts/ \
    --exclude .spark-build/ \
    "${VLLM_ROOT}/" "${REMOTE_HOST}:${REMOTE_VLLM_ROOT}/"
}

ensure_remote_b12x() {
  if ssh "${REMOTE_HOST}" "test -d '${REMOTE_B12X_ROOT}/.git'"; then
    return
  fi
  local origin
  origin=$(git -C "${B12X_ROOT}" remote get-url origin)
  ssh "${REMOTE_HOST}" \
    "mkdir -p '$(dirname "${REMOTE_B12X_ROOT}")' && git clone '${origin}' '${REMOTE_B12X_ROOT}'"
}

sync_b12x_source() {
  if [[ ${dry_run} -eq 1 ]] && \
    ! ssh "${REMOTE_HOST}" "test -d '${REMOTE_B12X_ROOT}/.git'"; then
    log "dry run: remote B12X checkout would be created"
    return
  fi
  ensure_remote_b12x
  log "syncing the live B12X working tree to ${REMOTE_HOST}"
  rsync "${rsync_flags[@]}" \
    "${source_excludes[@]}" \
    "${B12X_ROOT}/" "${REMOTE_HOST}:${REMOTE_B12X_ROOT}/"
}

sync_artifacts() {
  [[ -f "${ARTIFACT_ROOT}/manifest.json" ]] || \
    die "artifact manifest is missing; build before syncing artifacts"
  log "syncing wheels, NCCL, and provenance to ${REMOTE_HOST}"
  rsync "${rsync_flags[@]}" "${ARTIFACT_ROOT}/" \
    "${REMOTE_HOST}:${ARTIFACT_ROOT}/"

  local native_list
  native_list=$(mktemp)
  (
    cd "${VLLM_ROOT}"
    find vllm -type f \( -name '*.so' -o -name '*.so.*' -o -name vllm-rs \) \
      -print >"${native_list}"
  )
  if [[ -s "${native_list}" ]]; then
    rsync "${rsync_flags[@]}" --files-from="${native_list}" \
      "${VLLM_ROOT}/" "${REMOTE_HOST}:${REMOTE_VLLM_ROOT}/"
  fi
  rm -f "${native_list}"
}

case "${mode}" in
  source)
    sync_vllm_source
    sync_b12x_source
    ;;
  artifacts)
    sync_artifacts
    ;;
  all)
    sync_vllm_source
    sync_b12x_source
    sync_artifacts
    ;;
esac

log "sync complete"
