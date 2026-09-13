#!/usr/bin/env bash
# Build and verify wheels for one vLLM commit and one resolved B12X commit.
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

builder_image=$(lock_value builder.image)
flashinfer_image=$(lock_value flashinfer.artifact.image)
b12x_repository=$(lock_value b12x.repository)
b12x_ref=$(lock_value b12x.ref)
expected_uv_version=$(lock_value uv.version)
expected_uv_sha256=$(lock_value uv.sha256)
uv_binary=${UV_BIN:-uv}

if ! uv_path=$(command -v "${uv_binary}"); then
  printf 'uv is required; set UV_BIN to its absolute path.\n' >&2
  exit 1
fi
test "$("${uv_path}" --version | awk '{print $2}')" = "${expected_uv_version}"
test "$(sha256sum "${uv_path}" | awk '{print $1}')" = "${expected_uv_sha256}"

vllm_commit=$(git -C "${repo_root}" rev-parse HEAD)
vllm_tree=$(git -C "${repo_root}" rev-parse 'HEAD^{tree}')
vllm_source_date_epoch=$(git -C "${repo_root}" show -s --format=%ct HEAD)
test -z "$(git -C "${repo_root}" status --porcelain --untracked-files=no)"
vllm_version="0.26.1rc0+jj.g${vllm_commit:0:12}"

work_dir=$(mktemp -d -p "${RUNNER_TEMP:-/tmp}" jovian-wheel-build.XXXXXX)
trap 'rm -rf "${work_dir}"' EXIT

git init --quiet "${work_dir}/b12x"
git -C "${work_dir}/b12x" remote add origin "${b12x_repository}"
git -C "${work_dir}/b12x" fetch --depth=1 --filter=blob:none origin "${b12x_ref}"
git -C "${work_dir}/b12x" checkout --detach FETCH_HEAD
b12x_commit=$(git -C "${work_dir}/b12x" rev-parse HEAD)
b12x_tree=$(git -C "${work_dir}/b12x" rev-parse 'HEAD^{tree}')
b12x_source_date_epoch=$(git -C "${work_dir}/b12x" show -s --format=%ct HEAD)

if test -e "${output_dir}"; then
  printf 'Output path already exists: %s\n' "${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}/raw"

DOCKER_BUILDKIT=1 docker buildx build \
  --file "${tool_dir}/Dockerfile" \
  --build-context "b12x_source=${work_dir}/b12x" \
  --build-arg "BUILDER_IMAGE=${builder_image}" \
  --build-arg "FLASHINFER_ARTIFACT_IMAGE=${flashinfer_image}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_version}" \
  --build-arg "VLLM_SOURCE_DATE_EPOCH=${vllm_source_date_epoch}" \
  --build-arg "B12X_SOURCE_DATE_EPOCH=${b12x_source_date_epoch}" \
  --build-arg "BUILD_JOBS=${build_jobs}" \
  --target wheelhouse \
  --output "type=local,dest=${output_dir}/raw" \
  "${repo_root}"

wheel_dir="${output_dir}/bundle/wheels"
mkdir -p "${wheel_dir}"
cp -a "${output_dir}/raw/wheels/." "${wheel_dir}/"

flashinfer_python_wheel=$(find "${wheel_dir}" -maxdepth 1 -type f \
  -name 'flashinfer_python-*.whl' -print -quit)
flashinfer_jit_wheel=$(find "${wheel_dir}" -maxdepth 1 -type f \
  -name 'flashinfer_jit_cache-*.whl' -print -quit)
test -n "${flashinfer_python_wheel}"
test -n "${flashinfer_jit_wheel}"
test "$(sha256sum "${flashinfer_python_wheel}" | awk '{print $1}')" = \
  "$(lock_value flashinfer.python.sha256)"
test "$(sha256sum "${flashinfer_jit_wheel}" | awk '{print $1}')" = \
  "$(lock_value flashinfer.jit-cache.sha256)"

cp "${lock_path}" "${output_dir}/bundle/runtime.lock"
cp "${tool_dir}/install.sh" "${tool_dir}/verify_install.py" \
  "${output_dir}/bundle/"
chmod 0755 "${output_dir}/bundle/install.sh"

wheel_metadata() {
  local wheel=$1 field=$2
  unzip -p "${wheel}" '*/METADATA' \
    | awk -F': ' -v field="${field}" '$1 == field {print $2; exit}'
}

requirements_path="${output_dir}/bundle/requirements-wheelhouse.txt"
: > "${requirements_path}"
packages_json='[]'
while IFS= read -r wheel; do
  name=$(wheel_metadata "${wheel}" Name)
  version=$(wheel_metadata "${wheel}" Version)
  digest=$(sha256sum "${wheel}" | awk '{print $1}')
  printf '%s==%s --hash=sha256:%s\n' "${name}" "${version}" "${digest}" \
    >> "${requirements_path}"
  packages_json=$(jq \
    --arg name "${name}" \
    --arg version "${version}" \
    --arg file "$(basename "${wheel}")" \
    --arg sha256 "${digest}" \
    '. + [{name: $name, version: $version, file: $file, sha256: $sha256}]' \
    <<<"${packages_json}")
done < <(find "${wheel_dir}" -maxdepth 1 -type f -name '*.whl' | sort)

test "$(jq length <<<"${packages_json}")" -eq 4

jq -n \
  --arg status research-only \
  --arg vllm_repository https://github.com/local-inference-lab/vllm.git \
  --arg vllm_commit "${vllm_commit}" \
  --arg vllm_tree "${vllm_tree}" \
  --arg b12x_repository "${b12x_repository}" \
  --arg b12x_commit "${b12x_commit}" \
  --arg b12x_tree "${b12x_tree}" \
  --arg builder_image "${builder_image}" \
  --arg foundation_python_path "$(lock_value foundation.python-path)" \
  --arg python_version "$(lock_value python.version)" \
  --arg cuda_version "$(lock_value cuda.version)" \
  --arg pytorch_version "$(lock_value pytorch.version)" \
  --arg pytorch_commit "$(lock_value pytorch.commit)" \
  --arg nccl_version "$(lock_value nccl.version)" \
  --arg nccl_commit "$(lock_value nccl.commit)" \
  --arg cutlass_dsl_version "$(lock_value cutlass-dsl.version)" \
  --arg flashinfer_image "${flashinfer_image}" \
  --arg flashinfer_commit "$(lock_value flashinfer.commit)" \
  --argjson packages "${packages_json}" \
  '{
    schema: "local-inference-jovian-wheel-bundle/v1",
    status: $status,
    scope: "Application wheels for the declared CUDA, PyTorch, and NCCL runtime foundation",
    source: {
      vllm: {repository: $vllm_repository, commit: $vllm_commit, tree: $vllm_tree},
      b12x: {repository: $b12x_repository, commit: $b12x_commit, tree: $b12x_tree},
      flashinfer: {
        artifact_image: $flashinfer_image,
        commit: $flashinfer_commit
      }
    },
    runtime: {
      builder_image: $builder_image,
      foundation_python_path: $foundation_python_path,
      python_version: $python_version,
      cuda_version: $cuda_version,
      pytorch_version: $pytorch_version,
      pytorch_commit: $pytorch_commit,
      nccl_version: $nccl_version,
      nccl_commit: $nccl_commit,
      cutlass_dsl_version: $cutlass_dsl_version,
      requires_compatible_system_packages: true
    },
    packages: $packages
  }' > "${output_dir}/bundle/manifest.json"

(
  cd "${output_dir}/bundle"
  find wheels -maxdepth 1 -type f -name '*.whl' -print0 \
    | sort -z \
    | xargs -0 sha256sum
  sha256sum manifest.json requirements-wheelhouse.txt runtime.lock \
    install.sh verify_install.py
) > "${output_dir}/bundle/SHA256SUMS"

docker run --rm \
  --gpus all \
  --entrypoint /bin/bash \
  -v "${output_dir}/bundle:/bundle:ro" \
  -v "${uv_path}:/usr/local/bin/uv:ro" \
  "${builder_image}" \
  -lc 'unset PYTHONPATH; /bundle/install.sh /tmp/jovian-wheel-venv'

archive_name="jovian-judgement-wheels-${vllm_commit}-b12x-${b12x_commit}.tar.zst"
tar --sort=name \
  --mtime="@${vllm_source_date_epoch}" \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --zstd \
  -C "${output_dir}/bundle" \
  -cf "${output_dir}/${archive_name}" .
sha256sum "${output_dir}/${archive_name}" > \
  "${output_dir}/${archive_name}.sha256"

jq -r '
  "Status: **research-only**\n\n" +
  "This release contains application wheels for the declared CUDA 13.3, " +
  "PyTorch 2.13.0, and NCCL 2.31.2 runtime foundation. It is not a portable " +
  "CUDA runtime.\n\n" +
  "- vLLM commit: `" + .source.vllm.commit + "`\n" +
  "- B12X commit: `" + .source.b12x.commit + "`\n" +
  "- FlashInfer commit: `" + .source.flashinfer.commit + "`\n\n" +
  "Extract the archive and run `./install.sh /path/to/venv` inside the " +
  "compatible runtime foundation. The installer verifies every bundled " +
  "wheel and imports the compiled vLLM extension from the created venv."
' "${output_dir}/bundle/manifest.json" > "${output_dir}/release-notes.md"

printf '%s\n' "${output_dir}/${archive_name}"
