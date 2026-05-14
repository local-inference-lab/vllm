#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
MAX_JOBS="${MAX_JOBS:-128}"
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.11.2.dev278+glm51kimi${DATE_TAG}}"
BASE_IMAGE="${BASE_IMAGE:-}"
B12X_GIT_SHA="${B12X_GIT_SHA:-}"
CONTAINER_NAME="${CONTAINER_NAME:-glm51-vllm-editable-build-test}"

VLLM_GIT_SHA="$(git -C "${ROOT_DIR}" rev-parse HEAD)"
VLLM_GIT_SHORT="$(git -C "${ROOT_DIR}" rev-parse --short=7 HEAD)"
IMAGE_TAG="${IMAGE_TAG:-voipmonitor/vllm:glm51-canonical-editable-vllm${VLLM_GIT_SHORT}-b12x${B12X_GIT_SHA}-${DATE_TAG}}"

if ! git -C "${ROOT_DIR}" diff --quiet || ! git -C "${ROOT_DIR}" diff --cached --quiet; then
  echo "Refusing to build from a dirty vLLM tree. Commit or stash changes first." >&2
  git -C "${ROOT_DIR}" status --short >&2
  exit 1
fi

if [[ -z "${BASE_IMAGE}" || -z "${B12X_GIT_SHA}" ]]; then
  cat >&2 <<'EOF'
BASE_IMAGE and B12X_GIT_SHA must be set explicitly.

This script only rebuilds/overlays editable vLLM inside an existing base image.
It does not install or update B12X. Build the canonical full image from
docker/Dockerfile.glm51-kimi-b12x013 first, then pass that image here together
with the exact B12X git commit contained in it.
EOF
  exit 1
fi

echo "Base image: ${BASE_IMAGE}"
echo "Output image: ${IMAGE_TAG}"
echo "vLLM git: ${VLLM_GIT_SHA}"
echo "B12X git label: ${B12X_GIT_SHA}"
echo "MAX_JOBS: ${MAX_JOBS}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER_NAME}" --entrypoint sleep "${BASE_IMAGE}" infinity >/dev/null

git -C "${ROOT_DIR}" archive --format=tar HEAD \
  | docker exec -i "${CONTAINER_NAME}" bash -lc 'rm -rf /opt/vllm && mkdir -p /opt/vllm && tar -xf - -C /opt/vllm'

docker exec "${CONTAINER_NAME}" bash -lc "
  set -euo pipefail
  cd /opt/vllm
  env \
    MAX_JOBS='${MAX_JOBS}' \
    CMAKE_BUILD_PARALLEL_LEVEL='${CMAKE_BUILD_PARALLEL_LEVEL}' \
    TORCH_CUDA_ARCH_LIST='${TORCH_CUDA_ARCH_LIST}' \
    VLLM_TARGET_DEVICE=cuda \
    VLLM_VERSION_OVERRIDE='${VLLM_VERSION_OVERRIDE}' \
    /opt/venv/bin/python -m pip install -e . --no-deps --no-build-isolation -v
"

docker exec "${CONTAINER_NAME}" bash -lc "
  set -euo pipefail
  PYTHONPATH=/opt/vllm /opt/venv/bin/python - <<'PY'
import b12x
import vllm
import vllm._C as vllm_c

assert vllm.__file__.startswith('/opt/vllm/'), vllm.__file__
assert vllm_c.__file__.startswith('/opt/vllm/'), vllm_c.__file__
print('vllm_version', vllm.__version__)
print('vllm_file', vllm.__file__)
print('vllm_C', vllm_c.__file__)
print('b12x_file', b12x.__file__)
PY
"

docker commit \
  --change 'ENV PYTHONPATH=/opt/vllm' \
  --change 'ENTRYPOINT ["/usr/local/bin/run-glm51-vllm"]' \
  --change "LABEL voipmonitor.vllm.git_sha=${VLLM_GIT_SHA}" \
  --change "LABEL voipmonitor.vllm.install=editable-rebuilt-sm120a-maxjobs${MAX_JOBS}" \
  --change "LABEL voipmonitor.b12x.git_sha=${B12X_GIT_SHA}" \
  "${CONTAINER_NAME}" "${IMAGE_TAG}" >/dev/null

echo "Built ${IMAGE_TAG}"
