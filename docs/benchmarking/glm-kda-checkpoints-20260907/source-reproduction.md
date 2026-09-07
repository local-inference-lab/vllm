# Reconstructing the measured prefill source from public forks

The experiment records vLLM `abb715f132bdccb592a34b2596a3d3a8d757ffbc` as its
historical build-source identifier. That commit is not retrievable from the
public fork. Reconstruct the equivalent runtime from the two published feature
commits below; no private Git repository, deployment snapshot, or image is
needed to obtain this source.

| Component | Public fork branch | Frozen public commit |
| --- | --- | --- |
| vLLM continuation | `FujitsuPolycom/vllm:feat/gb10-continuation-prefill` | [`8b5a65e23c43695565802fd5a9408322ab300294`](https://github.com/FujitsuPolycom/vllm/commit/8b5a65e23c43695565802fd5a9408322ab300294) |
| vLLM mHC | `FujitsuPolycom/vllm:feat/gb10-mhc-token-sharding` | [`7a43439c18f6c1bde921762e9ed0d40c6334909c`](https://github.com/FujitsuPolycom/vllm/commit/7a43439c18f6c1bde921762e9ed0d40c6334909c) |
| B12X publication | `FujitsuPolycom/b12x:feat/gb10-kda-checkpoint-export` | [`b8044279b346095761d45f3a5db59afbfc2f4929`](https://github.com/FujitsuPolycom/b12x/commit/b8044279b346095761d45f3a5db59afbfc2f4929) |
| B12X tested implementation | Ancestor of the published B12X commit | [`70fe41974ef4b18f61caaa2579c81cdc05d1265f`](https://github.com/FujitsuPolycom/b12x/commit/70fe41974ef4b18f61caaa2579c81cdc05d1265f) |

These public commit endpoints and branch mappings were verified on
2026-09-07. Pin the commits rather than moving branch tips. The vLLM commits
share base `2a979314dc97b03173a0a76fc15664ec924db32b`; B12X's review base is
`06b4de7c723e6f166d65abf5909c5b7d0f8acc68`.

## Obtain and verify the source

Run the following Bash commands from a directory where `vllm-source` and
`b12x-source` do not exist. Git must support `merge-tree --write-tree`.
The merge is left uncommitted; it does not push or alter either public branch.

```bash
set -euo pipefail

CP_SOURCE=8b5a65e23c43695565802fd5a9408322ab300294
MHC_SOURCE=7a43439c18f6c1bde921762e9ed0d40c6334909c
B12X_PUBLIC=b8044279b346095761d45f3a5db59afbfc2f4929
B12X_TESTED=70fe41974ef4b18f61caaa2579c81cdc05d1265f

git clone --filter=blob:none --no-checkout \
  https://github.com/FujitsuPolycom/vllm.git vllm-source
git -C vllm-source fetch --no-tags origin "$CP_SOURCE" "$MHC_SOURCE"

SOURCE_TREE=$(git -C vllm-source merge-tree --write-tree \
  "$CP_SOURCE" "$MHC_SOURCE")
test "$SOURCE_TREE" = 3769f7f9754421b2f330031255e33a446af34b1a
test "$(git -C vllm-source rev-parse "$SOURCE_TREE:vllm")" = \
  5f6b014449c94344bde776d086b0b895fad683ec
test "$(git -C vllm-source rev-parse "$SOURCE_TREE:tests")" = \
  ee93b94ed702d40a8ff855987ff424f5b0210585

git -C vllm-source checkout --detach "$CP_SOURCE"
git -C vllm-source merge --no-commit --no-ff "$MHC_SOURCE"
test "$(git -C vllm-source write-tree)" = "$SOURCE_TREE"
git -C vllm-source diff --exit-code

git clone --filter=blob:none --no-checkout \
  https://github.com/FujitsuPolycom/b12x.git b12x-source
git -C b12x-source fetch --no-tags origin "$B12X_PUBLIC"
git -C b12x-source merge-base --is-ancestor "$B12X_TESTED" "$B12X_PUBLIC"
git -C b12x-source checkout --detach "$B12X_TESTED"
test "$(git -C b12x-source rev-parse HEAD:b12x)" = \
  be623cf8abea2e5dc6771232246edf7ac0a39712
test "$(git -C b12x-source rev-parse HEAD:tests)" = \
  4db2bd8301af46711aa408b11f7fd0c95629a7ee
git -C b12x-source diff --exit-code
```

The reconstructed vLLM `vllm/` and `tests/` tree hashes exactly match the
historical `abb715f1` build source. A local comparison also found no differences
outside `docs/`. The complete reconstructed tree differs because the public
commits include additional documentation and evidence.

B12X `b8044279` updates the module docstring in
`b12x/sequence/kda_prefill/__init__.py` as well as publishing evidence. Its
`b12x/` tree is therefore not byte-identical to the tested source, although the
checkpoint implementation is unchanged. Checking out its public `70fe4197`
ancestor, as above, recovers the exact tested package tree. Its tests are
identical at both commits.

`HEAD` in `vllm-source` still names the CP commit while the merged files are
staged. The authoritative source identity for this uncommitted reconstruction
is `SOURCE_TREE` and the verified subtree hashes, not `git rev-parse HEAD`.
Keep both source directories after an editable installation.

## Integrate into a Linux CUDA build

Source reconstruction and hash equality were verified. A new Linux/aarch64
clean build was **not** executed as part of this reconstruction check, and no
public base-image recipe has been qualified to reproduce image digest
`52b207e716a285c16e5e1b14ec2a41f6208b9450c617e7d1cde5a507ea879d7f`.
The following is the pinned repository's documented existing-PyTorch build
procedure, applied to the reconstructed source; a successful rebuild receives
its own binary and image identities.

Use a Linux/aarch64 CUDA development environment appropriate for GB10, with
Python3.12, a working CUDA13.0 PyTorch installation, CUDA development tools,
and the C++ compiler required by the pinned vLLM source. The measured runtime
used PyTorch `2.13.0+cu130`, Triton `3.7.1`, CUTLASS DSL `4.6.2`, and FlashInfer
`0.6.17`. Provisioning that platform-compatible PyTorch/toolchain installation
is a prerequisite, not a verified download step in this note. Follow the
[pinned CUDA build instructions](https://github.com/FujitsuPolycom/vllm/blob/8b5a65e23c43695565802fd5a9408322ab300294/docs/getting_started/installation/gpu.cuda.inc.md)
for the native build prerequisites.

From the parent directory containing both source checkouts, where the chosen
Python3.12 interpreter already has the required PyTorch available:

```bash
set -euo pipefail
cd vllm-source
uv venv --python 3.12 --system-site-packages .venv
uv run --no-project -- .venv/bin/python -c \
  'import torch; assert str(torch.__version__) == "2.13.0+cu130"; assert torch.version.cuda == "13.0"'

# This repository helper adjusts dependency declarations for the installed torch.
# It does not change the verified vllm/ or tests/ source trees.
uv run --no-project -- .venv/bin/python use_existing_torch.py
uv pip install --python .venv/bin/python -r requirements/build/cuda.txt
uv pip install --python .venv/bin/python --no-build-isolation --editable .
uv pip install --python .venv/bin/python --editable ../b12x-source
uv pip check --python .venv/bin/python

uv run --no-project -- .venv/bin/python - <<'PY'
from pathlib import Path
from importlib.metadata import version
import torch
import vllm
import vllm._C
import b12x
from b12x.sequence.kda_prefill import Caps

root = Path.cwd().resolve()
assert Path(vllm.__file__).resolve().is_relative_to(root / "vllm")
assert Path(b12x.__file__).resolve().is_relative_to(root.parent / "b12x-source" / "b12x")
assert "max_checkpoints" in Caps.__dataclass_fields__
assert str(torch.__version__) == "2.13.0+cu130"
assert torch.version.cuda == "13.0"
for package in ("triton", "nvidia-cutlass-dsl", "flashinfer-python"):
    print(package, version(package))
print("vllm:", vllm.__file__)
print("b12x:", b12x.__file__)
PY
```

Record the resolved dependency versions and native artifact hashes. If they
differ from the measured toolchain, describe the run as a new build rather
than an identical binary reproduction. Do not replace the source build with
an arbitrary upstream-main precompiled wheel and assume its native ABI matches
the maintained GLM source. B12X's CuTe kernels compile on first use; installing
the package or importing `Caps` does not qualify GPU execution.

Run the committed CPU and GB10 component checks before evaluating a rebuilt
serving stack. Source equality and import success do not establish native ABI,
GPU numerical, graph-replay, model-quality, or performance equivalence. The
model checkpoint/tokenizer pin and benchmark workload remain separate inputs;
this document reconstructs and integrates the source only.
