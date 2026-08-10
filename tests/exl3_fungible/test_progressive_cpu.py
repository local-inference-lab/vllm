# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the progressive weights stream + boot metadata synthesis."""
import json
import struct

import pytest

torch = pytest.importorskip("torch")

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        progressive as pg,
    )
except ImportError:  # standalone run against an env without built vllm._C
    import importlib.util
    import sys as _sys
    from pathlib import Path as _P

    _p = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible" / "progressive.py")
    _s = importlib.util.spec_from_file_location("fq_progressive_standalone", _p)
    pg = importlib.util.module_from_spec(_s)
    _sys.modules[_s.name] = pg
    _s.loader.exec_module(pg)

from test_fragments_cpu import (  # noqa: E402
    NUM_EXPERTS,
    PROJS,
    RANKS,
    _expert_tensors,
    make_manifest_dir,
    write_segment,
)


def _write_safetensors(path, tensors):
    """tensors: list of (name, dtype, shape, data-bytes)."""
    header, blobs, off = {}, [], 0
    for name, dtype, shape, data in tensors:
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def make_dense_source(tmp_path, moe_layers=(3,), source_k=3):
    """Mini rank-sliced checkpoint: one dense shard + MoE layer shards whose
    expert tensors are the K{source_k} encodings."""
    d = tmp_path / "dense-src"
    d.mkdir()
    _write_safetensors(d / "model-layer-000.safetensors", [
        ("model.layers.0.input_layernorm.weight", "BF16", (8,), bytes(16)),
        ("model.layers.0.mlp.weight", "BF16", (4, 4), bytes(32)),
    ])
    for layer in moe_layers:
        tensors = [
            (f"model.layers.{layer}.input_layernorm.weight", "BF16", (8,),
             bytes(range(16))),
            (f"model.layers.{layer}.mlp.gate.weight", "F32", (2, 2),
             struct.pack("<4f", 1, 2, 3, 4)),
        ]
        for expert in range(NUM_EXPERTS):
            tensors.extend(_expert_tensors(layer, expert, source_k, seed=0))
        _write_safetensors(d / f"model-layer-{layer:03d}.safetensors", tensors)
    (d / "config.json").write_text(json.dumps({
        "hybrid_tr3_tail": {
            "format": "exl3-trellis",
            "bits": float(source_k),
            "codebook": "mcg",
            "exllamav3_version": "0.0.43",
            "moe_layers": [min(moe_layers), max(moe_layers)],
            "experts_per_layer": NUM_EXPERTS,
            "tensor_schema": ("model.layers.{L}.mlp.experts.{E}.{proj}"
                              ".rank{r}.{trellis|suh|svh|mcg}"),
            "tp": RANKS,
        },
    }))
    return d


def make_policy(bits_by_layer):
    return {
        "schema": "fq-policy/2",
        "manifest": "test",
        "budget": {"mode": "fixed_cardinality",
                   "n_k4_per_layer": {L: sum(b == 4 for b in bits)
                                      for L, bits in bits_by_layer.items()}},
        "bits_per_expert": bits_by_layer,
        "pinned": {},
        "provenance": {"parent": "test"},
    }


def make_spec(tmp_path, policy, *, seg_dir=None, dense=None):
    seg_dir = seg_dir or make_manifest_dir(tmp_path, layers=(3,), ks=(3, 4))
    dense = dense or make_dense_source(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    return pg.ProgressiveSpec.from_env(
        dense,
        environ={},
        overrides={"manifest_dir": str(seg_dir), "policy": str(policy_path),
                   "dense_source": str(dense)},
    )


def stream(spec, tmp_path, tp_rank=None):
    resolver = spec.make_resolver(cache_dir=tmp_path / "cache", environ={})
    logs = []
    out = list(pg.progressive_weights_iterator(
        spec, resolver, tp_rank=tp_rank, log=logs.append))
    return dict(out), logs


def test_mixed_stream_matches_policy(tmp_path):
    bits = [4, 3, 3, 4]
    spec = make_spec(tmp_path, make_policy({"3": bits}))
    tensors, logs = stream(spec, tmp_path)

    # dense tensors pass through from the source shards
    assert "model.layers.0.mlp.weight" in tensors
    assert "model.layers.3.mlp.gate.weight" in tensors

    # every expert tensor is present and its trellis width follows the policy
    for expert, k in enumerate(bits):
        for proj in PROJS:
            for rank in range(RANKS):
                name = (f"model.layers.3.mlp.experts.{expert}.{proj}"
                        f".rank{rank}.trellis")
                assert tuple(tensors[name].shape) == (2, 2, 16 * k), name
    # name-set parity with the source shard
    n_expert_tensors = NUM_EXPERTS * len(PROJS) * RANKS * 4
    assert len(tensors) == 4 + n_expert_tensors
    assert any("tiers=((3, 2), (4, 2))" in line for line in logs)


def test_all_k3_stream_is_byte_identical_to_source(tmp_path):
    """The K3 segments are a repack of the source shard: an all-K3 policy
    must reproduce the source expert bytes exactly (fq_assemble's
    byte-identity property, now on the stream)."""
    spec = make_spec(tmp_path, make_policy({"3": [3] * NUM_EXPERTS}))
    tensors, _ = stream(spec, tmp_path)
    from test_fragments_cpu import _expert_tensors as et
    for expert in range(NUM_EXPERTS):
        for name, dtype, shape, data in et(3, expert, 3, seed=0):
            got = tensors[name]
            assert bytes(got.contiguous().view(torch.uint8).flatten()
                         .numpy().tobytes()) == data, name


def test_rank_filter(tmp_path):
    bits = [4, 3, 3, 4]
    spec = make_spec(tmp_path, make_policy({"3": bits}))
    tensors, _ = stream(spec, tmp_path, tp_rank=1)
    expert_names = [n for n in tensors if ".experts." in n]
    assert expert_names and all(".rank1." in n for n in expert_names)
    # dense tensors are never rank-filtered
    assert "model.layers.3.mlp.gate.weight" in tensors


def test_missing_segment_k_fails_loudly(tmp_path):
    seg_dir = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))  # no k4
    spec = make_spec(tmp_path, make_policy({"3": [4, 3, 3, 3]}),
                     seg_dir=seg_dir)
    resolver = spec.make_resolver(cache_dir=tmp_path / "cache", environ={})
    with pytest.raises(Exception, match="k4"):
        list(pg.progressive_weights_iterator(spec, resolver))


def test_policy_layer_coverage_enforced(tmp_path):
    seg_dir = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    manifest = json.loads((seg_dir / "fq-manifest.json").read_text())
    manifest["moe_layers"] = [3, 4]
    (seg_dir / "fq-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing MoE layers"):
        make_spec(tmp_path, make_policy({"3": [3] * NUM_EXPERTS}),
                  seg_dir=seg_dir)


# ------------------------------------------------------- boot metadata


def test_synthesize_hf_overrides_mixed(tmp_path):
    dense = make_dense_source(tmp_path)
    policy = make_policy({"3": [4, 3, 3, 4]})
    bitmap = tmp_path / "bitmap.json"
    pg.write_tier_bitmap(policy, bitmap)
    ov = pg.synthesize_hf_overrides(dense, policy, bitmap)

    tail = ov["hybrid_tr3_tail"]
    assert tail["bits"] == "mixed"
    assert tail["k_values"] == [3, 4]
    ref_file, field = tail["bits_per_expert"].rsplit(":", 1)
    assert field == "bits_per_expert"
    payload = json.loads(open(ref_file).read())
    assert payload["3"]["bits_per_expert"] == [4, 3, 3, 4]
    # untouched contract keys survive the patch
    assert tail["tensor_schema"].startswith("model.layers.{L}")
    assert tail["tp"] == RANKS

    stub = ov["quantization_config"]
    assert stub["quant_method"] == "exl3" and stub["bits"] == "mixed"


def test_synthesize_hf_overrides_uniform_and_existing_stub(tmp_path):
    dense = make_dense_source(tmp_path)
    cfg_path = dense / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["quantization_config"] = {"quant_method": "exl3", "bits": 3.0}
    cfg_path.write_text(json.dumps(cfg))

    policy = make_policy({"3": [3] * NUM_EXPERTS})
    ov = pg.synthesize_hf_overrides(dense, policy, tmp_path / "b.json",
                                    extra={"use_index_cache": True})
    tail = ov["hybrid_tr3_tail"]
    assert tail["bits"] == 3.0
    assert "k_values" not in tail and "bits_per_expert" not in tail
    assert "quantization_config" not in ov  # source already has one
    assert ov["use_index_cache"] is True


def test_spec_source_fallback_streams_missing_k(tmp_path):
    """Local dir has only K3; K4 arrives via the sources chain."""
    from test_fragments_cpu import FakeSource

    remote_root = tmp_path / "remote"
    write_segment(remote_root, 3, 4, seed=0)
    seg_dir = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    spec = make_spec(tmp_path, make_policy({"3": [4, 3, 3, 4]}),
                     seg_dir=seg_dir)
    resolver = spec.make_resolver(
        cache_dir=tmp_path / "cache", environ={},
        sources=[FakeSource(remote_root)])
    tensors = dict(pg.progressive_weights_iterator(spec, resolver))
    name = "model.layers.3.mlp.experts.0.gate_proj.rank0.trellis"
    assert tuple(tensors[name].shape) == (2, 2, 16 * 4)
    assert resolver.stats["fetched"] == 2  # experts 0 and 3 at K4
