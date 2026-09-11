#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
b12x_source=${B12X_SOURCE:?Set B12X_SOURCE to the companion B12X checkout}
uv_bin=$(command -v uv)
test "$(git -C "$b12x_source" rev-parse HEAD)" = d564f6ca54c092497ec5ae7e07a272f55eec7dbe
if [[ -n $(git -C "$b12x_source" status --porcelain --untracked-files=all) ]]; then
  echo 'B12X_SOURCE must be clean, including untracked files' >&2
  exit 2
fi
docker run --rm --network none --entrypoint /bin/bash \
 -e PYTHONPATH=/jj:/review \
 -v "$root:/jj:ro" -v "$b12x_source:/review:ro" \
 -v "$uv_bin:/usr/local/bin/uv:ro" \
 verdictai/trellismx@sha256:1c8a10d2b21bd6ed5a7ca4a29bcc3900d29acc3ce42e1d722b9ebaa74357de3f -c '
uv venv /tmp/check/.venv --python /opt/venv/bin/python --system-site-packages
cd /jj
/tmp/check/.venv/bin/python -S -c '\''
import site
for path in ["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages","/opt/venv/lib/python3.12/site-packages"]: site.addsitedir(path)
import runpy
runpy.run_path("examples/trellismx/verify_runtime.py")
import pytest
raise SystemExit(pytest.main(["--confcutdir=tests/quantization","-p","no:cacheprovider","tests/quantization/test_trellismx_manifest.py","tests/quantization/test_trellismx_method.py","-q"]))
'\''
/tmp/check/.venv/bin/python -S -c '\''
import site
for path in ["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages","/opt/venv/lib/python3.12/site-packages"]: site.addsitedir(path)
import pytest
raise SystemExit(pytest.main(["--confcutdir=tests/distributed","-p","no:cacheprovider","tests/distributed/test_dcp_a2a.py","tests/distributed/test_dcp_direct_a2a_lse_reduce.py","-q","-rs"]))
'\''
cd /review
/tmp/check/.venv/bin/python -S -c '\''
import site
for path in ["/usr/local/lib/python3.12/dist-packages","/usr/lib/python3/dist-packages","/opt/venv/lib/python3.12/site-packages"]: site.addsitedir(path)
import pytest
raise SystemExit(pytest.main(["--confcutdir=tests/moe","-p","no:cacheprovider","tests/moe/test_trellismx_contract.py","-q"]))
'\''
'
