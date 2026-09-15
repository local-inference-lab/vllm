#!/usr/bin/env bash
# Build vLLM while preserving generated Marlin sources with the native cache.
set -euo pipefail

source_root=/src/vllm
generated_cache=/tmp/vllm-generated
dense_source=${source_root}/csrc/libtorch_stable/quantization/marlin
moe_source=${source_root}/csrc/libtorch_stable/moe/marlin_moe_wna16
dense_cache=${generated_cache}/dense
moe_cache=${generated_cache}/moe

mkdir -p "${dense_cache}" "${moe_cache}" /wheelhouse

restore_generated_sources() {
  cp -a "${dense_cache}/." "${dense_source}/"
  cp -a "${moe_cache}/." "${moe_source}/"
}

save_generated_sources() {
  find "${dense_source}" -maxdepth 1 -type f \
    \( -name '*kernel_*.cu' -o -name kernel_selector.h \) -print0 \
    | xargs -0 -r cp --preserve=timestamps --target-directory="${dense_cache}"
  find "${moe_source}" -maxdepth 1 -type f \
    \( -name '*kernel_*.cu' -o -name kernel_selector.h \) -print0 \
    | xargs -0 -r cp --preserve=timestamps --target-directory="${moe_cache}"
}

restore_generated_sources
trap save_generated_sources EXIT

# CMake records generator hashes in its persistent cache. A clean source tree
# lacks generated files, so discard only its configuration when the generated
# source cache is empty. Compiled objects remain available to the next graph.
if ! test -f "${dense_source}/kernel_selector.h" \
  || ! test -f "${moe_source}/kernel_selector.h"; then
  while IFS= read -r -d '' cmake_cache; do
    rm -f -- "${cmake_cache}"
  done < <(find "${source_root}/build" -type f -name CMakeCache.txt -print0)
fi

env -u PYTHONPATH \
  VLLM_TARGET_DEVICE=cuda \
  CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120a -DFETCHCONTENT_BASE_DIR=/tmp/vllm-fetchcontent -DVLLM_BUILD_CUTLASS_SCALED_MM_C2X=${VLLM_BUILD_CUTLASS_SCALED_MM_C2X:?}" \
  MAX_JOBS="${BUILD_JOBS:?}" \
  uv build \
    --wheel \
    --no-build-isolation \
    --python /build-venv/bin/python \
    --out-dir /wheelhouse \
    /src/vllm

test "$(find /wheelhouse -maxdepth 1 -name 'vllm-*.whl' | wc -l)" -eq 1
/build-venv/bin/python tools/jovian_wheel_release/normalize_wheel.py \
  --wheel /wheelhouse/vllm-*.whl \
  --torch-version 2.14.0a0+4fdf77b940.nv26.8.63802676 \
  --torchvision-version 0.29.0a0+0bc41e67.nv26.8.63802676 \
  --flashinfer-version 0.6.18 \
  --source-date-epoch "${SOURCE_DATE_EPOCH:?}"
