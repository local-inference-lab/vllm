#!/usr/bin/env bash

set -euo pipefail
source "$(dirname "$0")/lib.sh"

target=all
jobs=${SPARK_BUILD_JOBS:-16}

usage() {
  cat <<'EOF'
Usage: tools/spark/build.sh [all|nccl|flashinfer|vllm|incremental]
                            [--jobs N]

Build distributable artifacts on tachyon. The incremental target installs
changed vLLM native targets directly into the editable source tree.
EOF
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  target=$1
  shift
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)
      jobs=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown build argument: $1"
      ;;
  esac
done

prepare() {
  assert_inputs
  activate_cuda
  ensure_uv
  [[ -x "${VENV_PYTHON}" ]] || \
    die "run tools/spark/bootstrap.sh before building"
  require_command git
  require_command make
  mkdir -p "${ARTIFACT_ROOT}/wheels" "${BUILD_ROOT}/src"
  export MAX_JOBS=${jobs}
  export CMAKE_BUILD_PARALLEL_LEVEL=${jobs}
  export CARGO_BUILD_JOBS=${jobs}
  export NINJAFLAGS="-j${jobs}"
  export MAKEFLAGS="-j${jobs}"
  export CMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc
  export TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
  export FLASHINFER_CUDA_ARCH_LIST=${CUDA_ARCH}
  export TRITON_PTXAS_PATH=${CUDA_HOME}/bin/ptxas
  export CUTE_DSL_ARCH
  export DG_JIT_USE_NVRTC=0
  export PROTOC_INCLUDE=/usr/include
  export USE_CUDNN=1
  export VLLM_REQUIRE_RUST_FRONTEND=1
  export CCACHE_NOHASHDIR=1
  export NVCC_THREADS=${SPARK_NVCC_THREADS:-2}
  if command -v ccache >/dev/null 2>&1; then
    export CCACHE_COMPRESS=1
    export CCACHE_MAXSIZE=${SPARK_CCACHE_MAXSIZE:-50G}
    ccache --max-size "${CCACHE_MAXSIZE}" >/dev/null
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
  fi
}

build_nccl() {
  local source_dir="${BUILD_ROOT}/src/nccl-${NCCL_REF}"
  local output_dir
  output_dir=$(nccl_lib_dir)
  if [[ ! -d "${source_dir}/.git" ]]; then
    log "cloning pinned NCCL ${NCCL_REF}"
    git clone --filter=blob:none https://github.com/NVIDIA/nccl.git \
      "${source_dir}"
    git -C "${source_dir}" checkout --detach "${NCCL_REF}"
  fi
  assert_ref "${source_dir}" "${NCCL_REF}" NCCL
  [[ -z "$(git -C "${source_dir}" status --porcelain)" ]] || \
    die "generated NCCL source tree is dirty"

  log "building NCCL for sm_121"
  make -C "${source_dir}" -j "${jobs}" src.build \
    NVCC_GENCODE="${NCCL_NVCC_GENCODE}"
  mkdir -p "${output_dir}"
  cp -a "${source_dir}/build/lib/." "${output_dir}/"
  nccl_so >/dev/null || die "NCCL build did not produce libnccl"
}

build_flashinfer() {
  local wheel_dir="${ARTIFACT_ROOT}/wheels"
  local build_source="${BUILD_ROOT}/src/flashinfer-${FLASHINFER_REF}"
  local cubin_source="${FLASHINFER_ROOT}/flashinfer-cubin/flashinfer_cubin/cubins"
  local cubin_build="${build_source}/flashinfer-cubin/flashinfer_cubin/cubins"
  mkdir -p "${build_source}" "${cubin_build}"
  rm -rf "${build_source}/build"
  rsync -a --delete \
    --exclude .git/ \
    --exclude build/ \
    --exclude '*.egg-info/' \
    --exclude flashinfer-cubin/flashinfer_cubin/cubins/ \
    "${FLASHINFER_ROOT}/" "${build_source}/"
  rsync -a "${cubin_source}/" "${cubin_build}/"
  patch -d "${build_source}" -p1 \
    <"${SPARK_RECIPE_ROOT}/flashinfer_cache.patch"
  patch -d "${build_source}" -p1 \
    <"${SPARK_DIR}/flashinfer-checksum-cache.patch"
  patch -d "${build_source}" -p1 \
    <"${SPARK_DIR}/flashinfer-manifest-cache.patch"
  rm -f "${wheel_dir}"/flashinfer_python-*.whl \
    "${wheel_dir}"/flashinfer_cubin-*.whl \
    "${wheel_dir}"/flashinfer_jit_cache-*.whl

  log "building FlashInfer ${FLASHINFER_REF}"
  BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 \
    "${VENV_PYTHON}" -m build --no-isolation --wheel \
      --outdir "${wheel_dir}" "${build_source}"
  "${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps \
    --force-reinstall nvidia-nvjitlink==13.0.88
  "${VENV_PYTHON}" -m build --no-isolation --wheel \
    --outdir "${wheel_dir}" "${build_source}/flashinfer-cubin"
  FLASHINFER_CUDA_ARCH_LIST=${CUDA_ARCH} \
    "${VENV_PYTHON}" -m build --no-isolation --wheel \
      --outdir "${wheel_dir}" "${build_source}/flashinfer-jit-cache"
  "${UV_BIN}" pip install --python "${VENV_PYTHON}" --no-deps \
    --force-reinstall nvidia-nvjitlink==13.0.88
}

build_vllm() {
  local wheel_dir="${ARTIFACT_ROOT}/wheels"
  rm -f "${wheel_dir}"/vllm-*.whl
  log "building the custom vLLM wheel for ${CUDA_ARCH}"
  env -u DEEPGEMM_SRC_DIR \
    "${VENV_PYTHON}" -m build --no-isolation --wheel \
      --outdir "${wheel_dir}" "${VLLM_ROOT}"
}

build_incremental() {
  local cmake_dir="${BUILD_ROOT}/vllm-cmake"
  log "configuring the persistent vLLM native build"
  env -u DEEPGEMM_SRC_DIR cmake -S "${VLLM_ROOT}" -B "${cmake_dir}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
    -DCMAKE_INSTALL_PREFIX="${VLLM_ROOT}" \
    -DVLLM_PYTHON_EXECUTABLE="${VENV_PYTHON}" \
    -DNVCC_THREADS="${SPARK_NVCC_THREADS:-2}"
  cmake --build "${cmake_dir}" --target install --parallel "${jobs}"
}

write_manifest() {
  ensure_uv
  "${UV_BIN}" pip freeze --python "${VENV_PYTHON}" \
    --exclude-editable \
    --exclude vllm \
    --exclude b12x \
    --exclude flashinfer-python \
    --exclude flashinfer-cubin \
    --exclude flashinfer-jit-cache \
    >"${ARTIFACT_ROOT}/environment.lock"
  NVCC="${CUDA_HOME}/bin/nvcc" "${VENV_PYTHON}" \
    "${SPARK_DIR}/manifest.py" \
      --output "${ARTIFACT_ROOT}/manifest.json" \
      --vllm-root "${VLLM_ROOT}" \
      --b12x-root "${B12X_ROOT}" \
      --flashinfer-root "${FLASHINFER_ROOT}" \
      --recipe-root "${SPARK_RECIPE_ROOT}" \
      --artifact-root "${ARTIFACT_ROOT}" \
      --uv "${UV_BIN}"
}

prepare
case "${target}" in
  all)
    build_nccl
    build_flashinfer
    build_vllm
    install_artifacts
    ;;
  nccl)
    build_nccl
    ;;
  flashinfer)
    build_flashinfer
    ;;
  vllm)
    build_vllm
    if compgen -G "${ARTIFACT_ROOT}/wheels/flashinfer_python-*.whl" \
      >/dev/null; then
      install_artifacts
    fi
    ;;
  incremental)
    build_incremental
    ;;
  *)
    die "build target must be all, nccl, flashinfer, vllm, or incremental"
    ;;
esac
write_manifest
log "${target} build complete"
