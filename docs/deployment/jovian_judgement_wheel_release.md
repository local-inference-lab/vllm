# Jovian Judgement application wheel releases

Status: **research-only**

The `Jovian Judgement wheel release` GitHub Actions workflow builds an
immutable application wheel bundle for each source pair resolved after a push to
`dev/jovian-judgement`. A release records the exact vLLM source tree, the B12X
`master` commit resolved at build time, and the qualified FlashInfer artifact.

The workflow does not rebuild FlashInfer. It extracts `flashinfer-python` and
`flashinfer-jit-cache` from the content-addressed OCI artifact declared in
`tools/jovian_wheel_release/runtime.lock`. That artifact is reused until its
commit or CUDA/PyTorch ABI declaration changes. The self-hosted BuildKit daemon
also retains the vLLM CMake, FetchContent, Cargo, and pip caches. A Python-only
vLLM commit therefore packages a wheel without recompiling unchanged CUDA and
Rust objects.

## Bundle contents

Each source-pair-addressed prerelease contains:

- a vLLM wheel compiled for SM120 against PyTorch 2.13.0 and CUDA 13.3;
- a B12X wheel built from the recorded `master` commit;
- qualified `flashinfer-python` and `flashinfer-jit-cache` wheels;
- SHA-256 checksums, a machine-readable source manifest, and an installer.

The installer creates a Python 3.12 venv with `uv`, verifies every bundled
wheel, installs the four packages without dependency resolution, and imports
the compiled vLLM extension. The created venv uses the immutable foundation's
Python packages through a `.pth` file; application wheels installed in the venv
retain precedence. Run the installer inside the runtime foundation declared in
the manifest:

```bash
tar --zstd -xf \
  jovian-judgement-wheels-<vllm-commit>-b12x-<b12x-commit>.tar.zst
./install.sh /opt/jovian-venv
```

The application bundle is not yet a portable CUDA runtime. It deliberately
uses `--system-site-packages` because the serving image contains a custom
PyTorch 2.13.0 build linked to the local-inference-lab NCCL 2.31.2 library.
It also requires the foundation Python paths declared in `runtime.lock`.
The following independently built components must become an immutable runtime
foundation before installation on an arbitrary CUDA host is supported:

- PyTorch 2.13.0 and TorchVision 0.28.0;
- NCCL 2.31.2 and its loader path contract;
- XGrammar 0.2.5;
- LMCache with recurrent checkpoint extensions;
- InstantTensor, DeepGEMM, ExLlamaV3, and the FlashKDA native extension.

Those components are not invalidated by an ordinary vLLM commit. Their wheels
or native archives belong in a separately versioned foundation release, keyed
by their source commits and CUDA/PyTorch ABI. A complete offline wheelhouse can
then combine one foundation release with one source-pair-addressed application
release and install it with `uv pip install --require-hashes --no-index`.
The release tag and archive name contain both source commits, so rerunning a
workflow never replaces an artifact that was built from a different B12X tree.

## Cache ownership

The self-hosted BuildKit daemon owns compilation caches for producing wheels.
Their names include the CUDA, PyTorch, and target-architecture identity. A
source-only commit reuses compatible CMake objects, fetched native
dependencies, generated Marlin sources, Cargo registries, Rust targets, and pip
downloads. Generated source files and CMake's generator fingerprints are cached
together so a clean checkout cannot reuse an object graph whose generated
inputs are absent. The patched PyTorch header is copied only when its content
changes, preserving the timestamp used by Ninja's dependency checks. Changing
the declared ABI requires new cache names rather than reusing incompatible
objects. Wheel timestamps use each source commit's authoritative commit time
instead of the runner clock.

The default `BUILD_JOBS=64` produces 16 concurrent `nvcc` processes because
each compiler process uses four CUDA frontend threads. A smaller runner may set
`BUILD_JOBS` explicitly; changing this scheduling limit does not invalidate
compiled objects.

An installed venv does not own runtime JIT caches. Deployments must mount a
persistent cache directory and direct `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`,
`TRITON_CACHE_DIR`, `CUTE_DSL_CACHE_DIR`, `B12X_CUTE_COMPILE_CACHE_DIR`,
`B12X_COMPILE_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `CUDA_CACHE_PATH` into
ABI-keyed subdirectories. Sharing those directories across venv replacements
prevents B12X, CuTe DSL, Triton, TorchInductor, and the CUDA driver from
recompiling unchanged runtime kernels. Concurrent processes may share only
cache implementations that provide their own locking; otherwise each server
must use a distinct writable leaf directory.

## Self-hosted runner communication

The workflow requires a repository-scoped runner carrying the
`lil-wheel-builder` label. The runner process opens outbound TLS connections to
GitHub on TCP port 443, polls for jobs, downloads the checked-out commit and
actions, streams logs, and uploads release assets. GitHub does not initiate a
connection to the server, so no inbound firewall rule or SSH exposure is
required.

The runner host must provide Docker BuildKit, `git`, `gh`, `jq`, `unzip`,
`zstd`, the NVIDIA Container Toolkit, a compatible NVIDIA driver, and the exact
`uv` binary recorded in `runtime.lock`. Compilation does not use a GPU. The
post-install import test requests the NVIDIA runtime so the container receives
`libcuda.so.1`; it does not run a model or allocate a serving workload. The
Docker daemon retains content-addressed base images and named BuildKit caches.
The runner must be scoped only to `local-inference-lab/vllm`; the workflow does
not run for pull requests and grants the job only `contents: write` permission.

A Docker socket grants effective root access to the host. Do not attach a
public-repository runner to the production Docker daemon. Run the service in a
dedicated virtual machine or with a dedicated rootless Docker daemon whose
storage and credentials are isolated from model-serving workloads.
