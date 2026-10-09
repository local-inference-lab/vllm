# Jovian Judgement CUDA 13.4 vLLM releases

Status: **research-only until GPU qualification in an assembled container**

The `Jovian Judgement CUDA 13.4 vLLM wheels` workflow publishes one immutable
vLLM wheel for each source commit on `dev/jovian-judgement`. The wheel targets
Python 3.12, CUDA 13.4.1, NVIDIA PyTorch 26.08, the C++11 ABI, and SM120a. Its
release manifest records the exact source tree, builder image, Rust toolchain,
CUDA architecture, package version, and wheel SHA-256 digest.

## Component boundary

The release contains only the vLLM wheel and its verification metadata. B12X,
FlashInfer, LMCache, InstantTensor, NCCL, and the NVIDIA PyTorch foundation are
independently versioned components. A Qwen3.8 container assembler rejects a set
of component manifests unless every application wheel declares the same Python,
CUDA, PyTorch, C++ ABI, and immutable builder-image contract.

The vLLM wheel intentionally excludes runtime dependencies that are not used by
the Qwen3.8 SM120 deployment profile. TileLang, Tokenspeed MLA, Humming, QuACK,
TorchCodec, PyNvVideoCodec, and fastsafetensors remain external packages. Image
input is supported by the assembled container; audio and video decoding are
unsupported.

Each branch commit publishes a prerelease named
`vllm-jovian-cu134-beta-<source-commit>`. A stable tag promotes the same files;
it never rebuilds or replaces wheel bytes. The verifier checks exact asset
membership, source identity, manifest contents, and archive checksums before
accepting either release.
Reusing a published beta compares every asset with a fresh build. Reusing or
creating a stable release compares every beta asset with the source beta and
validates the promotion record's source identity and manifest checksum.
Stable promotion records byte identity, not model-serving qualification.
Verification consumes the flat asset directory produced by GitHub Downloads;
the installable archive retains its separate nested wheel layout.

## Installation role

The component bundle is not a standalone Python or CUDA environment. The
supported deployment installs it into an isolated application venv layered over
the immutable NVIDIA PyTorch 26.08 image. The complete container adds the
source-addressed B12X, FlashInfer, LMCache, InstantTensor, and NCCL components,
applies hash-checked compatibility patches, and preloads the packaged NCCL
library before importing PyTorch.

Direct-host installation is research-only because NVIDIA PyTorch 26.08 is not
published as an ordinary public wheel and requires a verified native-library
closure. The component installer remains useful for inspecting a release in an
already qualified Python 3.12 environment:

```bash
tar --zstd -xf vllm-jovian-cu134-<source-commit>.tar.zst
./install.sh /path/to/qualified/venv
```

## Build and cache contract

The self-hosted runner uses a dedicated rootless Docker daemon and a persistent
BuildKit worker. Native jobs from all runtime repositories acquire the same host
file lock, so only one compiler workload can consume the shared CPU and memory
budget at a time. The worker is limited to 256 GiB of memory and logical CPUs
64 through 127; it does not mount the production Docker daemon.

Cache identities include the CUDA, PyTorch, compiler, and target-architecture
contract. BuildKit retains CMake and Ninja objects, fetched native dependencies,
generated Marlin sources, Cargo registries, Rust targets, and package downloads.
A Python-only source commit can therefore reuse compatible native objects.
Generated sources and their CMake fingerprints are retained together so an
object cannot outlive an input required to validate it.

The default `BUILD_JOBS=64` permits 16 concurrent NVCC processes because each
process uses four CUDA frontend threads. Lowering the scheduling limit does not
change artifact identity. Runtime JIT caches are deployment-owned and must use
persistent ABI-keyed directories.

## Runner communication and trust

The repository-scoped runner opens outbound TLS connections to GitHub on TCP
port 443. GitHub does not initiate a connection to the host. The workflow runs
only after a push to `dev/jovian-judgement` or a matching stable tag; pull
requests do not receive runner execution or release credentials.

The host must provide Docker BuildKit, Git, GitHub CLI, jq, unzip, zstd, the
NVIDIA Container Toolkit, and the exact uv binary declared by
`tools/jovian_wheel_release/runtime.lock`. Compilation does not require a GPU.
SM120 serving, CUDA graph capture, text generation, and image input are release
qualification gates of the assembled container rather than the component wheel.
