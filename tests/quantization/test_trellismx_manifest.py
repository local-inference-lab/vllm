# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlay contracts without importing torch, CUDA, or the vLLM engine.

The loader consumes a full rank inventory and returns a checked rank-local
sidecar. Tiny headers catch wrong rates, missing ranks and identity corruption
more cheaply than a serving run. These tests do not establish device closure.
"""

import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / "vllm/utils/trellismx.py"
spec = importlib.util.spec_from_file_location("trellismx_manifest_test", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.fixture
def overlay(tmp_path):
    (tmp_path / "design").mkdir()
    (tmp_path / "sidecars").mkdir()
    for index in range(3):
        (tmp_path / f"design/design-{index}.json").write_text("{}")
    design = hashlib.sha256(b"{}").hexdigest()
    (tmp_path / "design/transform.json").write_text(
        json.dumps(
            {
                "boundary": "coupled-h512-h128-suh-svh-v1",
                "sign_draw": 0,
                "activation": "silu-cap10",
            }
        )
    )
    records = []
    allocation = {str(layer): 5 if layer % 2 else 4 for layer in range(3, 45)}
    for layer in range(3, 45):
        for rank in range(4):
            path = f"sidecars/p8-layer-{layer:03d}-tp4-rank-{rank}.safetensors"
            metadata = {
                "schema": "glm53-p8-coupled-h512-h128-tp4-rank.v1",
                "layer": str(layer),
                "rank": str(rank),
                "world_size": "4",
                "bits": str(allocation[str(layer)]),
                "alphabet": "e4m3",
                "scale": "ue8m0-k32",
                "law": "procedural-mcg-alpha2",
                "source_design_sha256": design,
            }
            header = json.dumps({"__metadata__": metadata}).encode()
            payload = struct.pack("<Q", len(header)) + header
            (tmp_path / path).write_bytes(payload)
            records.append(
                {
                    "path": path,
                    "layer": layer,
                    "rank": rank,
                    "bits": allocation[str(layer)],
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source_design_sha256": design,
                }
            )
    manifest = {
        "carrier": {
            "repo_id": "local-inference-lab/GLM-5.3-Flash-NVFP4",
            "revision": "520de24eabf507659eaef7c70f14fd584527facc",
        },
        "schema": "trellismx.hf-overlay-release.v1",
        "allocation": allocation,
        "files": records,
    }
    (tmp_path / "trellismx-manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def rewrite(root, manifest):
    (root / "trellismx-manifest.json").write_text(json.dumps(manifest))
    module.load_overlay.cache_clear()


def test_loads_both_rates_and_correct_rank(overlay):
    root, _ = overlay
    checkpoint = module.load_overlay(str(root))
    assert len(checkpoint.records) == 168
    assert checkpoint.sidecar(3, 2).name == "p8-layer-003-tp4-rank-2.safetensors"
    assert checkpoint.sidecar(4, 0).name == "p8-layer-004-tp4-rank-0.safetensors"


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "rate", "design", "path", "size", "schema"]
)
def test_rejects_broken_inventory(overlay, corruption):
    root, manifest = overlay
    if corruption == "missing":
        manifest["files"].pop()
    elif corruption == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif corruption == "rate":
        manifest["files"][0]["bits"] = 3
    elif corruption == "design":
        manifest["files"][0]["source_design_sha256"] = "0" * 64
    elif corruption == "path":
        manifest["files"][0]["path"] = "../elsewhere"
    elif corruption == "size":
        manifest["files"][0]["bytes"] += 1
    else:
        manifest["schema"] = "unknown"
    rewrite(root, manifest)
    with pytest.raises(ValueError):
        module.load_overlay(str(root))


def test_rejects_weight_corruption_even_when_size_unchanged(overlay):
    root, manifest = overlay
    checkpoint = module.load_overlay(str(root))
    path = root / manifest["files"][0]["path"]
    payload = path.read_bytes().replace(b"e4m3", b"e2m1")
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="weight hash mismatch"):
        checkpoint.sidecar(3, 0)
    with pytest.raises(ValueError, match="alphabet"):
        checkpoint.sidecar(3, 0, verify=False)


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.45.mlp.experts",
        "model.layers.45.mtp_block.mlp.experts",
        "model.layers.2.mlp.experts",
        "model.layers.3.mlp.shared_experts",
        "model.layers.3.self_attn",
        "other.layers.3.mlp.experts",
    ],
)
def test_does_not_replace_mtp_dense_shared_or_attention(prefix):
    assert module.routed_layer(prefix) is None


@pytest.mark.parametrize(
    "prefix",
    [
        "model.layers.3.mlp.experts",
        "model.language_model.layers.3.mlp.experts",
        "language_model.model.layers.3.mlp.experts",
    ],
)
def test_accepts_glm_text_prefixes(prefix):
    assert module.routed_layer(prefix) == 3
