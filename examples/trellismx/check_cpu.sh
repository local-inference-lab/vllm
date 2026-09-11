#!/usr/bin/env bash
# CPU-only source/loader tests using pinned dependencies; no GPU devices exposed.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
b12x_source=${B12X_SOURCE:?Set B12X_SOURCE to the companion B12X review checkout}
expected=ac8ef2ca23ab1f5bb45a94976b36a3ded0006bf1
test "$(git -C "$b12x_source" rev-parse HEAD)" = "$expected"
git -C "$b12x_source" diff --quiet HEAD -- b12x
image=verdictai/trellismx@sha256:609a5fc1cd7d994ba32d9c03626c414d315947eb9f13fab474a15bc8dfbe0129
docker run --rm --network none --entrypoint /opt/venv/bin/python \
  -e PYTHONPATH=/jj:/review \
  -v "$root:/jj:ro" -v "$b12x_source:/review:ro" -w /jj \
  "$image" -S -c '
import site
for path in ["/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages", "/opt/venv/lib/python3.12/site-packages"]:
    site.addsitedir(path)
import runpy
runpy.run_path("examples/trellismx/verify_runtime.py")
import pytest
raise SystemExit(pytest.main([
    "--confcutdir=tests/quantization", "-p", "no:cacheprovider",
    "tests/quantization/test_trellismx_manifest.py",
    "tests/quantization/test_trellismx_method.py", "-q",
]))'
