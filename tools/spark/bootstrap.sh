#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

host=local
recreate=0
install_only=0

usage() {
  cat <<'EOF'
Usage: tools/spark/bootstrap.sh [--host local|luxon|both] [--recreate]
                                [--install-only]

Create the pinned Python environment. If built artifacts are present, install
them and overlay the vLLM and B12X checkouts as editable packages.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      host=$2
      shift 2
      ;;
    --recreate)
      recreate=1
      shift
      ;;
    --install-only)
      install_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown bootstrap argument: $1"
      ;;
  esac
done

bootstrap_local() {
  assert_runtime_inputs
  activate_cuda
  ensure_uv

  if [[ ${install_only} -eq 0 ]]; then
    if [[ ${recreate} -eq 1 && -d "${VLLM_ROOT}/.venv" ]]; then
      log "removing the existing vLLM virtual environment"
      rm -rf "${VLLM_ROOT}/.venv"
    fi

    if [[ ! -x "${VENV_PYTHON}" ]]; then
      "${UV_BIN}" venv --python "${PYTHON_VERSION}" --seed \
        "${VLLM_ROOT}/.venv"
    fi

    if [[ -f "${ARTIFACT_ROOT}/environment.lock" ]]; then
      log "installing the resolved build environment lock"
      "${UV_BIN}" pip install --python "${VENV_PYTHON}" \
        --torch-backend=cu130 -r "${ARTIFACT_ROOT}/environment.lock"
    else
      local runtime_requirements
      runtime_requirements=$(mktemp)
      sed \
        -e "s|^-r common.txt|-r ${VLLM_ROOT}/requirements/common.txt|" \
        -e '/^flashinfer-python==/d' \
        -e '/^flashinfer-cubin==/d' \
        "${VLLM_ROOT}/requirements/cuda.txt" >"${runtime_requirements}"

      "${UV_BIN}" pip install --python "${VENV_PYTHON}" \
        --torch-backend=cu130 \
        -r "${VLLM_ROOT}/requirements/build/cuda.txt"
      "${UV_BIN}" pip install --python "${VENV_PYTHON}" \
        --torch-backend=cu130 -r "${runtime_requirements}"
      rm -f "${runtime_requirements}"

      "${UV_BIN}" pip install --python "${VENV_PYTHON}" \
        -r "${VLLM_ROOT}/requirements/lint.txt"
      "${UV_BIN}" pip install --python "${VENV_PYTHON}" \
        'instanttensor>=0.1.5'
    fi
    "${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps \
      --no-build-isolation --editable "${B12X_ROOT}"
  fi

  if compgen -G "${ARTIFACT_ROOT}/wheels/vllm-*.whl" >/dev/null; then
    install_artifacts
  elif [[ ${install_only} -eq 1 ]]; then
    die "--install-only requested but no vLLM wheel is available"
  else
    log "native artifacts are not built yet; leaving the base environment ready"
  fi

  uv_pip_check
  if [[ -x "${VLLM_ROOT}/.venv/bin/pre-commit" ]]; then
    "${VLLM_ROOT}/.venv/bin/pre-commit" install
  fi
  log "bootstrap complete on $(hostname)"
}

case "${host}" in
  local)
    bootstrap_local
    ;;
  luxon)
    remote_args=(--host local)
    [[ ${recreate} -eq 1 ]] && remote_args+=(--recreate)
    [[ ${install_only} -eq 1 ]] && remote_args+=(--install-only)
    printf -v remote_command ' %q' "${remote_args[@]}"
    ssh "${REMOTE_HOST}" \
      "cd '${REMOTE_VLLM_ROOT}' && tools/spark/bootstrap.sh${remote_command}"
    ;;
  both)
    bootstrap_local
    remote_args=(--host local)
    [[ ${recreate} -eq 1 ]] && remote_args+=(--recreate)
    [[ ${install_only} -eq 1 ]] && remote_args+=(--install-only)
    printf -v remote_command ' %q' "${remote_args[@]}"
    ssh "${REMOTE_HOST}" \
      "cd '${REMOTE_VLLM_ROOT}' && tools/spark/bootstrap.sh${remote_command}"
    ;;
  *)
    die "--host must be local, luxon, or both"
    ;;
esac
