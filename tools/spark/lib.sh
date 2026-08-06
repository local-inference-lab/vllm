#!/usr/bin/env bash

set -euo pipefail

SPARK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VLLM_ROOT=$(cd "${SPARK_DIR}/../.." && pwd)
source "${SPARK_DIR}/versions.env"

B12X_ROOT=${B12X_ROOT:-"${VLLM_ROOT}/../b12x"}
FLASHINFER_ROOT=${FLASHINFER_ROOT:-"${VLLM_ROOT}/../flashinfer"}
SPARK_RECIPE_ROOT=${SPARK_RECIPE_ROOT:-"${VLLM_ROOT}/../spark-vllm-docker"}
ARTIFACT_ROOT=${SPARK_ARTIFACT_ROOT:-"${VLLM_ROOT}/.spark-artifacts"}
BUILD_ROOT=${SPARK_BUILD_ROOT:-"${VLLM_ROOT}/.spark-build"}
STATE_ROOT=${SPARK_STATE_ROOT:-"${HOME}/.local/state/vllm-spark"}
REMOTE_HOST=${SPARK_REMOTE_HOST:-luxon.lan}
REMOTE_VLLM_ROOT=${SPARK_REMOTE_VLLM_ROOT:-"${VLLM_ROOT}"}
REMOTE_B12X_ROOT=${SPARK_REMOTE_B12X_ROOT:-"${B12X_ROOT}"}
HEAD_ADDR=${SPARK_HEAD_ADDR:-192.168.177.11}
WORKER_ADDR=${SPARK_WORKER_ADDR:-192.168.177.12}
HEAD_ADDR_SECONDARY=${SPARK_HEAD_ADDR_SECONDARY:-192.168.178.11}
WORKER_ADDR_SECONDARY=${SPARK_WORKER_ADDR_SECONDARY:-192.168.178.12}
PRIMARY_IFACE=${SPARK_PRIMARY_IFACE:-enp1s0f1np1}
SECONDARY_IFACE=${SPARK_SECONDARY_IFACE:-enP2p1s0f1np1}
PRIMARY_HCA=${SPARK_PRIMARY_HCA:-rocep1s0f1}
SECONDARY_HCA=${SPARK_SECONDARY_HCA:-roceP2p1s0f1}
VENV_PYTHON="${VLLM_ROOT}/.venv/bin/python"

log() {
  printf '[spark] %s\n' "$*"
}

die() {
  printf '[spark] error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

find_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
  elif [[ -x "${HOME}/.local/bin/uv" ]]; then
    printf '%s\n' "${HOME}/.local/bin/uv"
  else
    return 1
  fi
}

ensure_uv() {
  if ! UV_BIN=$(find_uv); then
    require_command curl
    log "installing uv into ${HOME}/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN=$(find_uv) || die "uv installation did not produce an executable"
  fi
  export UV_BIN
}

uv_pip_check() {
  local output
  if output=$("${UV_BIN}" pip check --python "${VENV_PYTHON}" 2>&1); then
    printf '%s\n' "${output}"
    return
  fi
  if [[ "$(uname -m)" == aarch64 ]] && \
    [[ "${output}" == *'nvidia-cusparselt-cu13'* ]] && \
    [[ "${output}" == *'Found 1 incompatibility'* ]]; then
    local library="${VLLM_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0"
    if [[ -f "${library}" ]] && file "${library}" | grep -q 'ARM aarch64'; then
      log "accepting NVIDIA's SBSA-tagged cuSPARSELt wheel; its ELF is aarch64"
      return
    fi
  fi
  printf '%s\n' "${output}" >&2
  return 1
}

activate_cuda() {
  export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
  [[ -x "${CUDA_HOME}/bin/nvcc" ]] || \
    die "CUDA compiler not found at ${CUDA_HOME}/bin/nvcc"
  export PATH="${CUDA_HOME}/bin:${HOME}/.local/bin:${VLLM_ROOT}/.venv/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

git_ref() {
  git -C "$1" rev-parse HEAD
}

assert_git_checkout() {
  local repo=$1
  local label=$2
  git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    die "${label} checkout not found: ${repo}"
}

assert_ref() {
  local repo=$1
  local expected=$2
  local label=$3
  assert_git_checkout "${repo}" "${label}"
  local actual
  actual=$(git_ref "${repo}")
  [[ "${actual}" == "${expected}" ]] || \
    die "${label} is ${actual}; expected ${expected}"
}

assert_runtime_inputs() {
  assert_git_checkout "${VLLM_ROOT}" vLLM
  assert_git_checkout "${B12X_ROOT}" B12X
  assert_ref "${SPARK_RECIPE_ROOT}" "${SPARK_RECIPE_REF}" spark-vllm-docker

  [[ -z "$(git -C "${SPARK_RECIPE_ROOT}" status --porcelain)" ]] || \
    die "spark-vllm-docker checkout has unexpected changes"
}

assert_inputs() {
  assert_runtime_inputs
  assert_ref "${FLASHINFER_ROOT}" "${FLASHINFER_REF}" FlashInfer
  [[ -z "$(git -C "${FLASHINFER_ROOT}" status --porcelain)" ]] || \
    die "FlashInfer checkout has unexpected changes"
}

assert_editable_sources() {
  local python=$1
  local vllm_root=$2
  local b12x_root=$3
  [[ -x "${python}" ]] || die "Python environment not found: ${python}"
  "${python}" - "${vllm_root}" "${b12x_root}" <<'PY'
import sys
from pathlib import Path

import b12x
import vllm

for module, root in ((vllm, sys.argv[1]), (b12x, sys.argv[2])):
    module_path = Path(module.__file__).resolve()
    root_path = Path(root).resolve()
    if not module_path.is_relative_to(root_path):
        message = f"{module.__name__} resolves to {module_path}, not {root_path}"
        raise SystemExit(message)
PY
}

nccl_lib_dir() {
  printf '%s\n' "${ARTIFACT_ROOT}/nccl/${NCCL_REF}/lib"
}

nccl_so() {
  local lib_dir
  lib_dir=$(nccl_lib_dir)
  if [[ -e "${lib_dir}/libnccl.so.2" ]]; then
    printf '%s\n' "${lib_dir}/libnccl.so.2"
  elif [[ -e "${lib_dir}/libnccl.so" ]]; then
    printf '%s\n' "${lib_dir}/libnccl.so"
  else
    return 1
  fi
}

link_python_nccl() {
  local library site_packages python_nccl
  library=$(nccl_so) || die "custom NCCL library is missing"
  site_packages=$("${VENV_PYTHON}" -c \
    'import site; print(site.getsitepackages()[0])')
  python_nccl="${site_packages}/nvidia/nccl/lib/libnccl.so.2"
  [[ -e "${python_nccl}" || -L "${python_nccl}" ]] || \
    die "Python NCCL library is missing: ${python_nccl}"
  ln -sfn "${library}" "${python_nccl}"
}

single_wheel() {
  local pattern=$1
  local wheels=()
  mapfile -t wheels < <(compgen -G "${pattern}" || true)
  [[ ${#wheels[@]} -eq 1 ]] || return 1
  printf '%s\n' "${wheels[0]}"
}

install_artifacts() {
  [[ -x "${VENV_PYTHON}" ]] || die "create .venv before installing artifacts"
  ensure_uv

  local flashinfer_wheel cubin_wheel jit_wheel vllm_wheel vllm_version
  flashinfer_wheel=$(single_wheel \
    "${ARTIFACT_ROOT}/wheels/flashinfer_python-${FLASHINFER_VERSION}-*.whl") || \
    die "missing or ambiguous FlashInfer ${FLASHINFER_VERSION} wheel"
  cubin_wheel=$(single_wheel \
    "${ARTIFACT_ROOT}/wheels/flashinfer_cubin-${FLASHINFER_VERSION}-*.whl") || \
    die "missing or ambiguous FlashInfer cubin ${FLASHINFER_VERSION} wheel"
  jit_wheel=$(single_wheel \
    "${ARTIFACT_ROOT}/wheels/flashinfer_jit_cache-*.whl") || \
    die "missing or ambiguous FlashInfer JIT-cache wheel"
  vllm_wheel=$(single_wheel "${ARTIFACT_ROOT}/wheels/vllm-*.whl") || \
    die "missing or ambiguous vLLM wheel"
  vllm_version=$("${VENV_PYTHON}" - "${vllm_wheel}" <<'PY'
import email
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    metadata = [
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata) != 1:
        raise RuntimeError(f"expected one METADATA file, found {metadata}")
    print(email.message_from_bytes(wheel.read(metadata[0]))["Version"])
PY
  )

  "${UV_BIN}" pip install --python "${VENV_PYTHON}" \
    --torch-backend=cu130 "${flashinfer_wheel}"
  "${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps \
    --force-reinstall "${cubin_wheel}" "${jit_wheel}"
  "${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps \
    --force-reinstall "${vllm_wheel}"

  VLLM_VERSION_OVERRIDE="${vllm_version}" \
    VLLM_PRECOMPILED_WHEEL_LOCATION="${vllm_wheel}" \
    VLLM_USE_PRECOMPILED=1 \
    "${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps \
      --no-build-isolation --editable "${VLLM_ROOT}"
  "${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps \
    --no-build-isolation --editable "${B12X_ROOT}"
  link_python_nccl
}
