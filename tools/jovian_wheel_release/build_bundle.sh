#!/usr/bin/env bash
# Build a source-locked vLLM wheel for the CUDA 13.4 serving runtime.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tool_dir="${repo_root}/tools/jovian_wheel_release"
lock_path="${tool_dir}/runtime.lock"
output_dir=${1:-"${repo_root}/dist/jovian-wheel-release"}
build_jobs=${BUILD_JOBS:-64}

lock_value() {
  local key=$1
  awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {exit !found}' \
    "${lock_path}"
}

source_commit=$(git -C "${repo_root}" rev-parse HEAD)
source_tree=$(git -C "${repo_root}" rev-parse 'HEAD^{tree}')
source_date_epoch=$(git -C "${repo_root}" show -s --format=%ct HEAD)
repository=${GITHUB_REPOSITORY:-local-inference-lab/vllm}
release_tag=${VLLM_RELEASE_TAG:-"vllm-jovian-cu134-beta-${source_commit}"}
builder=$(lock_value buildx.builder)
test -z "$(git -C "${repo_root}" status --porcelain)"

mkdir -p "$(dirname "${output_dir}")"
if ! mkdir "${output_dir}"; then
  printf 'Output path already exists: %s\n' "${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}/raw" "${output_dir}/bundle/wheels"

docker buildx build \
  --builder "${builder}" \
  --file "${tool_dir}/Dockerfile" \
  --build-arg "BUILDER_IMAGE=$(lock_value builder.image)" \
  --build-arg "RUST_IMAGE=$(lock_value rust.image)" \
  --build-arg "UV_IMAGE=$(lock_value uv.image)" \
  --build-arg "UV_SHA256=$(lock_value uv.sha256)" \
  --build-arg "SOURCE_DATE_EPOCH=${source_date_epoch}" \
  --build-arg "BUILD_JOBS=${build_jobs}" \
  --build-arg "VLLM_BUILD_CUTLASS_SCALED_MM_C2X=$(lock_value build.cutlass-scaled-mm-c2x)" \
  --target export \
  --output "type=local,dest=${output_dir}/raw" \
  "${repo_root}"
cp -a "${output_dir}/raw/wheels/." "${output_dir}/bundle/wheels/"

wheel=$(find "${output_dir}/bundle/wheels" -maxdepth 1 -name 'vllm-*.whl' -print -quit)
test -n "${wheel}"
metadata=$(unzip -p "${wheel}" '*/METADATA')
test "$(awk -F': ' '$1 == "Name" {print $2; exit}' <<<"${metadata}")" = vllm
package_version=$(awk -F': ' '$1 == "Version" {print $2; exit}' <<<"${metadata}")
test -n "${package_version}"
case "${package_version}" in
  *"g${source_commit:0:7}"*) ;;
  *)
    printf 'Wheel version does not identify source commit %s: %s\n' \
      "${source_commit}" "${package_version}" >&2
    exit 1
    ;;
esac
grep -Fqx "Requires-Dist: torch==$(lock_value pytorch.version)" <<<"${metadata}"
grep -Fqx "Requires-Dist: torchvision==$(lock_value torchvision.version)" <<<"${metadata}"
grep -Fqx "Requires-Dist: flashinfer-python==$(lock_value flashinfer.requirement)" \
  <<<"${metadata}"
if grep -Fq 'Requires-Dist: torchaudio' <<<"${metadata}"; then
  printf 'The normalized wheel must not require the unsupported audio extra.\n' >&2
  exit 1
fi
for package in apache-tvm-ffi tilelang tokenspeed-mla humming-kernels \
  quack-kernels torchcodec PyNvVideoCodec; do
  if grep -Fiq "Requires-Dist: ${package}" <<<"${metadata}"; then
    printf 'The Qwen SM120 wheel must not require unused backend %s.\n' \
      "${package}" >&2
    exit 1
  fi
done

digest=$(sha256sum "${wheel}" | awk '{print $1}')
file=$(basename "${wheel}")
url="https://github.com/${repository}/releases/download/${release_tag}/${file}"
printf 'vllm @ %s --hash=sha256:%s\n' "${url}" "${digest}" \
  > "${output_dir}/bundle/requirements-github.txt"

jq -n \
  --arg status research-only \
  --arg repository "https://github.com/${repository}.git" \
  --arg commit "${source_commit}" \
  --arg tree "${source_tree}" \
  --arg package_version "${package_version}" \
  --arg release_tag "${release_tag}" \
  --arg file "${file}" \
  --arg sha256 "${digest}" \
  --arg url "${url}" \
  --arg builder_image "$(lock_value builder.image)" \
  --arg rust_image "$(lock_value rust.image)" \
  --arg uv_image "$(lock_value uv.image)" \
  --arg python "$(lock_value python.version)" \
  --arg cuda "$(lock_value cuda.version)" \
  --arg pytorch "$(lock_value pytorch.version)" \
  --arg pytorch_commit "$(lock_value pytorch.commit)" \
  --arg cuda_arch_list "$(lock_value cuda.arch-list)" \
  --arg cutlass_scaled_mm_c2x "$(lock_value build.cutlass-scaled-mm-c2x)" \
  '{schema: "local-inference-vllm-wheel-release/v2", status: $status,
    scope: "vLLM native and Python runtime for Qwen3.8 SM120 serving",
    source: {repository: $repository, commit: $commit, tree: $tree},
    package_version: $package_version, release_tag: $release_tag,
    runtime: {builder_image: $builder_image, rust_image: $rust_image,
      uv_image: $uv_image, python: $python, cuda: $cuda, pytorch: $pytorch,
      pytorch_commit: $pytorch_commit, cuda_arch_list: $cuda_arch_list,
      cutlass_scaled_mm_c2x: $cutlass_scaled_mm_c2x,
      unsupported_extras: ["audio", "video"],
      external_device_backends: ["tilelang", "tokenspeed-mla",
        "humming-kernels", "quack-kernels"]},
    packages: [{name: "vllm", version: $package_version, file: $file,
      sha256: $sha256, url: $url}]}' \
  > "${output_dir}/bundle/manifest.json"

cp "${lock_path}" "${tool_dir}/install.sh" "${output_dir}/bundle/"
chmod 0755 "${output_dir}/bundle/install.sh"
(
  cd "${output_dir}/bundle"
  find wheels -maxdepth 1 -name '*.whl' -print0 | sort -z | xargs -0 sha256sum
  sha256sum manifest.json requirements-github.txt runtime.lock install.sh
) > "${output_dir}/bundle/SHA256SUMS"

archive="${output_dir}/vllm-jovian-cu134-${source_commit}.tar.zst"
tar --sort=name --mtime="@${source_date_epoch}" --owner=0 --group=0 \
  --numeric-owner --zstd -C "${output_dir}/bundle" -cf "${archive}" .
(cd "${output_dir}" && sha256sum "$(basename "${archive}")") \
  > "${archive}.sha256"
cat > "${output_dir}/release-notes.md" <<EOF
Status: **research-only**

This release contains vLLM ${package_version} from source commit
\`${source_commit}\` for Python 3.12, CUDA 13.4.1, NVIDIA PyTorch 26.08, the
C++11 ABI, and SM120a. Foundation and third-party dependency wheels are not
included. The wheel metadata installs the Qwen3.8 SM120 execution profile;
TileLang, Tokenspeed, Humming, and QuACK remain separate backend packages.
Audio and video serving are unsupported because the foundation does not ship
matching TorchAudio or video-decoder packages.
EOF

printf '%s\n' "${output_dir}/bundle"
