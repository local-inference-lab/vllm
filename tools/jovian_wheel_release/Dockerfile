# syntax=docker/dockerfile:1.7

ARG BUILDER_IMAGE
ARG FLASHINFER_ARTIFACT_IMAGE

FROM ${BUILDER_IMAGE} AS build-environment

ARG BUILD_JOBS=64
ENV PATH=/root/.cargo/bin:/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    TORCH_CUDA_ARCH_LIST=12.0a \
    FLASHINFER_CUDA_ARCH_LIST=12.0f \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NVCC_THREADS=4

RUN env -u PYTHONPATH /opt/venv/bin/python - <<'PY'
import torch

assert torch.__version__ == "2.13.0"
assert torch.version.cuda == "13.3"
assert torch._C._GLIBCXX_USE_CXX11_ABI
PY

FROM build-environment AS b12x-wheel

ARG B12X_SOURCE_DATE_EPOCH
COPY --from=b12x_source / /src/b12x
WORKDIR /src/b12x
RUN --mount=type=cache,id=jovian-wheel-pip,target=/root/.cache/pip,sharing=locked \
    mkdir -p /wheelhouse \
    && env -u PYTHONPATH \
      SOURCE_DATE_EPOCH="${B12X_SOURCE_DATE_EPOCH}" \
      /opt/venv/bin/python -m pip wheel \
      --no-build-isolation --no-deps --wheel-dir /wheelhouse . \
    && test "$(find /wheelhouse -maxdepth 1 -name 'b12x-*.whl' | wc -l)" -eq 1

FROM build-environment AS vllm-wheel

ARG VLLM_PACKAGE_VERSION
ARG VLLM_SOURCE_DATE_EPOCH
ARG BUILD_JOBS=64
COPY . /src/vllm
WORKDIR /src/vllm
RUN --mount=type=cache,id=jovian-wheel-pip,target=/root/.cache/pip,sharing=locked \
    env -u PYTHONPATH /opt/venv/bin/python -m pip install \
      'cmake>=3.26.1' ninja 'packaging>=24.2' \
      'setuptools>=77.0.3,<81.0.0' 'setuptools-scm>=8.0' \
      'setuptools-rust>=1.9.0' wheel jinja2
RUN --mount=type=cache,id=jovian-vllm-native-cu133-torch213-sm120,target=/src/vllm/build,sharing=locked \
    --mount=type=cache,id=jovian-vllm-fetchcontent-cu133,target=/tmp/vllm-fetchcontent,sharing=locked \
    --mount=type=cache,id=jovian-vllm-generated-cu133-torch213-sm120,target=/tmp/vllm-generated,sharing=locked \
    --mount=type=cache,id=jovian-vllm-cargo-registry,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,id=jovian-vllm-cargo-git,target=/root/.cargo/git,sharing=locked \
    --mount=type=cache,id=jovian-vllm-rust-target,target=/src/vllm/rust/target,sharing=locked \
    --mount=type=cache,id=jovian-wheel-pip,target=/root/.cache/pip,sharing=locked \
    VLLM_PACKAGE_VERSION="${VLLM_PACKAGE_VERSION}" \
    SOURCE_DATE_EPOCH="${VLLM_SOURCE_DATE_EPOCH}" \
    BUILD_JOBS="${BUILD_JOBS}" \
    tools/jovian_wheel_release/build_vllm_wheel.sh

FROM ${FLASHINFER_ARTIFACT_IMAGE} AS flashinfer-wheels

FROM scratch AS wheelhouse
COPY --from=vllm-wheel /wheelhouse/ /wheels/
COPY --from=b12x-wheel /wheelhouse/ /wheels/
COPY --from=flashinfer-wheels /opt/flashinfer-wheels/ /wheels/
