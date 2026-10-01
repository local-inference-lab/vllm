# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manifest validation and bounded file access for compressed FP4 checkpoints."""

import json
from contextlib import ExitStack
from pathlib import Path

from safetensors import safe_open


def read_csf_contract(root, *, schema, codec, families):
    """Check storage identity and the source tensor-to-shard inventory."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    contract = json.loads((root / "build-contract.json").read_text())
    for key, expected in (("schema", schema), ("codec", codec)):
        if manifest.get(key) != expected or contract.get(key) != expected:
            raise ValueError(f"CSF requires {key}={expected!r}")
    if (
        contract.get("family") not in families
        or manifest.get("family") != contract["family"]
    ):
        raise ValueError("CSF manifest and contract must name a supported model family")
    names = contract["source_names"]
    files = {item["file"] for item in manifest["shards"]}
    if set(names.values()) != files:
        raise ValueError("CSF manifest and source tensor inventory disagree")
    for name in files:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(
                "CSF shard names must be checkpoint-local safetensors files"
            )
    return contract


class CsfTensorReader(ExitStack):
    """Keep a layer's lazily opened shards alive while B12X consumes its tensors."""

    def __init__(self, root, source_names, codec):
        super().__init__()
        self.root = Path(root)
        self.source_names = source_names
        self.codec = codec
        self.handles = {}

    def _handle(self, name):
        filename = self.source_names[name]
        if filename not in self.handles:
            self.handles[filename] = self.enter_context(
                safe_open(
                    self.root / "tensors" / filename, framework="pt", device="cpu"
                )
            )
        return self.handles[filename]

    def matrix(self, weight, scale, *, global_scale=None, input_scale=None):
        """Expose a lazy packed weight view and one projection's CPU scale data."""
        from b12x.moe.checkpoints.csf import CsfMatrix

        scales = self._handle(scale)
        return CsfMatrix(
            weight=self._handle(weight).get_slice(weight),
            fixed=scales.get_tensor(f"{scale}.{self.codec}_csf_fixed"),
            exceptions=scales.get_tensor(f"{scale}.{self.codec}_csf_exceptions"),
            global_scale=(
                self._handle(global_scale).get_tensor(global_scale)
                if global_scale is not None
                else None
            ),
            input_scale=(
                self._handle(input_scale).get_tensor(input_scale)
                if input_scale is not None
                else None
            ),
        )
