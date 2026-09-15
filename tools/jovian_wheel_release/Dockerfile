# syntax=docker/dockerfile:1.7

ARG UV_IMAGE
ARG RUST_IMAGE
ARG BUILDER_IMAGE

FROM ${UV_IMAGE} AS uv
FROM ${RUST_IMAGE} AS rust

FROM ${BUILDER_IMAGE} AS wheels

ARG BUILD_JOBS=64
ARG SOURCE_DATE_EPOCH
ARG UV_SHA256
ARG VLLM_BUILD_CUTLASS_SCALED_MM_C2X
ENV PATH=/build-venv/bin:/usr/local/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    TORCH_CUDA_ARCH_LIST=12.0a \
    CMAKE_CUDA_ARCHITECTURES=120a \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    MAX_JOBS=${BUILD_JOBS} \
    NVCC_THREADS=4 \
    VLLM_BUILD_CUTLASS_SCALED_MM_C2X=${VLLM_BUILD_CUTLASS_SCALED_MM_C2X} \
    TORCH_NIGHTLY=1

COPY --from=uv /uv /usr/local/bin/uv
COPY --from=rust /usr/local/cargo /usr/local/cargo
COPY --from=rust /usr/local/rustup /usr/local/rustup
COPY tools/jovian_wheel_release/build-requirements.lock /tmp/build-requirements.lock
RUN test "$(sha256sum /usr/local/bin/uv | awk '{print $1}')" = "${UV_SHA256}" \
    && test "$(rustc --version | awk '{print $2}')" = 1.95.0 \
    && uv venv --python /usr/bin/python --system-site-packages /build-venv \
    && uv pip install --python /build-venv/bin/python \
      --require-hashes --no-deps -r /tmp/build-requirements.lock

COPY . /src/vllm
WORKDIR /src/vllm
RUN /build-venv/bin/python - <<'PY'
import importlib.metadata
import torch

assert importlib.metadata.version("torch") == (
    "2.14.0a0+4fdf77b940.nv26.8.63802676"
)
assert torch.__version__ == "2.14.0a0+4fdf77b940.nv26.08"
assert torch.version.cuda == "13.4"
assert torch._C._GLIBCXX_USE_CXX11_ABI
PY

RUN --mount=type=cache,id=lil-vllm-native-cu134-torch214-sm120,target=/src/vllm/build,sharing=locked \
    --mount=type=cache,id=lil-vllm-fetchcontent-cu134,target=/tmp/vllm-fetchcontent,sharing=locked \
    --mount=type=cache,id=lil-vllm-generated-cu134-torch214-sm120,target=/tmp/vllm-generated,sharing=locked \
    --mount=type=cache,id=lil-vllm-cargo-registry,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,id=lil-vllm-cargo-git,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,id=lil-vllm-rust-target,target=/src/vllm/rust/target,sharing=locked \
    --mount=type=cache,id=lil-wheel-uv,target=/root/.cache/uv,sharing=locked \
    SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
    BUILD_JOBS="${BUILD_JOBS}" \
    tools/jovian_wheel_release/build_vllm_wheel.sh

FROM scratch AS export
COPY --from=wheels /wheelhouse/ /wheels/
