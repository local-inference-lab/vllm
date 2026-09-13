# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate the ABI identity and imports of a Jovian Judgement wheel bundle."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path

import torch

bundle_dir = Path(__file__).resolve().parent
manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))

assert torch.__version__ == manifest["runtime"]["pytorch_version"]
assert torch.version.cuda == manifest["runtime"]["cuda_version"]
assert torch._C._GLIBCXX_USE_CXX11_ABI

for package in manifest["packages"]:
    assert importlib.metadata.version(package["name"]) == package["version"]

for module in ("b12x", "flashinfer", "vllm", "vllm._C_stable_libtorch"):
    importlib.import_module(module)

print("Jovian Judgement application wheel bundle: PASS")
