#!/usr/bin/env bash
# Install the application wheel bundle into a CUDA 13.3 / PyTorch 2.13 runtime.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_path=${1:-.venv-jovian}
uv_binary=${UV_BIN:-uv}
if ! uv_path=$(command -v "${uv_binary}"); then
  printf 'uv is required; set UV_BIN to its absolute path.\n' >&2
  exit 1
fi

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)

"${uv_path}" venv --python 3.12 --system-site-packages "${venv_path}"
venv_site_packages=$("${venv_path}/bin/python" -c \
  'import site; print(site.getsitepackages()[0])')
foundation_python_path=$(awk -F= \
  '$1 == "foundation.python-path" {sub(/^[^=]*=/, ""); print; found=1} END {exit !found}' \
  "${bundle_dir}/runtime.lock")
: > "${venv_site_packages}/jovian-foundation.pth"
while IFS= read -r foundation_path; do
  test -d "${foundation_path}" || {
    printf 'Runtime foundation path does not exist: %s\n' \
      "${foundation_path}" >&2
    exit 1
  }
  printf '%s\n' "${foundation_path}" \
    >> "${venv_site_packages}/jovian-foundation.pth"
done < <(tr ':' '\n' <<<"${foundation_python_path}")

"${uv_path}" pip install \
  --python "${venv_path}/bin/python" \
  --require-hashes \
  --no-index \
  --find-links "${bundle_dir}/wheels" \
  --no-deps \
  -r "${bundle_dir}/requirements-wheelhouse.txt"

env -u PYTHONPATH "${venv_path}/bin/python" "${bundle_dir}/verify_install.py"
