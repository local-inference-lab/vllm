# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for FragmentResolver: local hit, source fallback, sha-mismatch
rejection, cache reuse, verification policy."""
import base64
import hashlib
import json
import struct

import pytest

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        fragments as fr,
    )
except ImportError:  # standalone run against an env without built vllm._C
    import importlib.util
    import sys as _sys
    from pathlib import Path as _P

    _p = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible" / "fragments.py")
    _s = importlib.util.spec_from_file_location("fq_fragments_standalone", _p)
    fr = importlib.util.module_from_spec(_s)
    _sys.modules[_s.name] = fr
    _s.loader.exec_module(fr)


# ------------------------------------------------------------- fixtures

NUM_EXPERTS = 4
RANKS = 2
PROJS = ("gate_proj", "up_proj", "down_proj")


def _expert_tensors(layer, expert, k, seed):
    """Tiny rank-sliced EXL3-shaped tensor set for one expert."""
    out = []
    for proj in PROJS:
        for rank in range(RANKS):
            base = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.rank{rank}"
            trellis = bytes(
                (seed + expert + rank + i) % 251 for i in range(2 * 2 * 16 * k * 2)
            )
            out.append((f"{base}.trellis", "I16", (2, 2, 16 * k), trellis))
            suh = bytes((seed + expert + i) % 249 for i in range(32 * 2))
            out.append((f"{base}.suh", "F16", (32,), suh))
            svh = bytes((seed + expert + 1 + i) % 247 for i in range(32 * 2))
            out.append((f"{base}.svh", "F16", (32,), svh))
            out.append((f"{base}.mcg", "I32", (1,), struct.pack("<i", 7)))
    return out


def write_segment(seg_dir, layer, k, *, seed=0, num_experts=NUM_EXPERTS,
                  with_attestation=True):
    """Emit layer segment + index-k{k}.json entry + attestation, fq_repack
    layout: per-expert contiguous body, expert-major tensor order."""
    header, blobs = {}, []
    per_expert, digests = {}, {}
    off = 0
    for expert in range(num_experts):
        start = off
        sha = hashlib.sha256()
        for name, dtype, shape, data in _expert_tensors(layer, expert, k, seed):
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [off, off + len(data)],
            }
            blobs.append(data)
            sha.update(data)
            off += len(data)
        per_expert[str(expert)] = [start, off]
        digests[str(expert)] = sha.hexdigest()

    hj = json.dumps({"__metadata__": {"k": str(k)}, **header},
                    separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    payload = struct.pack("<Q", len(hj)) + hj + b"".join(blobs)

    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_name = f"layer-{layer:03d}.k{k}.safetensors"
    (seg_dir / seg_name).write_bytes(payload)

    entry = {
        "file": seg_name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "body_offset": 8 + len(hj),
        "experts": per_expert,
    }
    index_path = seg_dir / f"index-k{k}.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[str(layer)] = entry
    index_path.write_text(json.dumps(index))

    if with_attestation:
        att = {"schema": "fq-attestation/1", "expert_sha256": digests}
        line = json.dumps({
            "payload": base64.b64encode(json.dumps(att).encode()).decode(),
            "signature": "dGVzdA==",
            "keyid": "00",
        })
        att_dir = seg_dir / "attestations"
        att_dir.mkdir(exist_ok=True)
        (att_dir / f"layer-{layer:03d}.k{k}.jsonl").write_text(line + "\n")
    return entry, digests


class FakeSource:
    """In-memory remote implementing the source protocol, with counters."""

    def __init__(self, root, *, corrupt_fragments=False, drop_attestations=False):
        self.name = "fake:src"
        self.root = root
        self.corrupt_fragments = corrupt_fragments
        self.drop_attestations = drop_attestations
        self.range_reads = 0
        self.json_reads = 0

    def read_json(self, relpath):
        self.json_reads += 1
        p = self.root / relpath
        return json.loads(p.read_text()) if p.exists() else None

    def read_text(self, relpath):
        if self.drop_attestations and relpath.startswith("attestations/"):
            return None
        p = self.root / relpath
        return p.read_text() if p.exists() else None

    def read_range(self, relpath, start, end):
        self.range_reads += 1
        data = (self.root / relpath).read_bytes()[start:end]
        if self.corrupt_fragments and not relpath.endswith(".jsonl"):
            # flip a body byte, keep length (headers are re-parsed, so only
            # corrupt when the caller asked for a body range)
            if start > 0:
                data = bytes([data[0] ^ 0xFF]) + data[1:]
        return data


def make_manifest_dir(tmp_path, layers=(3,), ks=(3,), **kw):
    d = tmp_path / "segments"
    d.mkdir(exist_ok=True)
    (d / "fq-manifest.json").write_text(json.dumps({
        "schema": "fq-manifest/1",
        "num_experts": NUM_EXPERTS,
        "moe_layers": [min(layers), max(layers)],
        "sources": [],
    }))
    for layer in layers:
        for k in ks:
            write_segment(d, layer, k, **kw)
    return d


def resolver(manifest_dir, tmp_path, **kw):
    kw.setdefault("cache_dir", tmp_path / "cache")
    kw.setdefault("environ", {})
    return fr.FragmentResolver(manifest_dir, **kw)


# ---------------------------------------------------------------- tests


def test_local_hit(tmp_path):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3, 4))
    r = resolver(d, tmp_path)
    frag = r.resolve(3, 2, 4)
    assert frag.origin == "local"
    entry = json.loads((d / "index-k4.json").read_text())["3"]
    lo, hi = entry["experts"]["2"]
    raw = (d / entry["file"]).read_bytes()
    assert bytes(frag.payload) == raw[entry["body_offset"] + lo:
                                      entry["body_offset"] + hi]
    assert r.stats["local"] == 1 and r.stats["fetched"] == 0


def test_local_tensors_shapes_and_rank_filter(tmp_path):
    torch = pytest.importorskip("torch")
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(4,))
    r = resolver(d, tmp_path)
    tensors = dict(r.expert_tensors(3, 1, 4))
    name = "model.layers.3.mlp.experts.1.gate_proj.rank0.trellis"
    assert tensors[name].dtype == torch.int16
    assert tuple(tensors[name].shape) == (2, 2, 16 * 4)
    assert len(tensors) == len(PROJS) * RANKS * 4  # trellis/suh/svh/mcg

    only_r1 = r.expert_tensors(3, 1, 4, name_filter=lambda n: ".rank1." in n)
    assert len(only_r1) == len(PROJS) * 4
    assert all(".rank1." in n for n, _ in only_r1)


def test_source_fallback_fetches_verifies_and_caches(tmp_path):
    remote_root = tmp_path / "remote"
    write_segment(remote_root, 3, 4, seed=9)
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))  # no local k4
    src = FakeSource(remote_root)
    r = resolver(d, tmp_path, sources=[src])
    frag = r.resolve(3, 0, 4)
    assert frag.origin == "fetched"
    assert r.stats["fetched"] == 1
    cached = r._cache_fragment_path(frag.sha256)  # noqa: SLF001
    assert cached.exists() and cached.read_bytes() == bytes(frag.payload)

    # local k3 still resolves locally, never touching the source
    reads = src.range_reads
    assert r.resolve(3, 0, 3).origin == "local"
    assert src.range_reads == reads


def test_fetched_sha_mismatch_rejected(tmp_path):
    remote_root = tmp_path / "remote"
    write_segment(remote_root, 3, 4, seed=9)
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    src = FakeSource(remote_root, corrupt_fragments=True)
    r = resolver(d, tmp_path, sources=[src])
    with pytest.raises(fr.FragmentVerificationError):
        r.resolve(3, 0, 4)
    assert r.stats["sha_mismatch"] == 1
    assert r.stats["fetched"] == 0


def test_cache_reuse_skips_refetch(tmp_path):
    remote_root = tmp_path / "remote"
    write_segment(remote_root, 3, 4, seed=9)
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    src1 = FakeSource(remote_root)
    r1 = resolver(d, tmp_path, sources=[src1])
    r1.resolve(3, 0, 4)

    src2 = FakeSource(remote_root)
    r2 = resolver(d, tmp_path, sources=[src2])
    frag = r2.resolve(3, 0, 4)
    assert frag.origin == "cache"
    # index + cached header may be consulted; the fragment body itself must
    # not be re-fetched
    assert src2.range_reads == 0
    assert r2.stats["fetched"] == 0 and r2.stats["cache"] == 1
    assert bytes(frag.payload) == bytes(r1.resolve(3, 0, 4).payload)


def test_unverified_fetch_refused_then_allowed(tmp_path):
    remote_root = tmp_path / "remote"
    write_segment(remote_root, 3, 4, seed=9, with_attestation=False)
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    src = FakeSource(remote_root, drop_attestations=True)
    r = resolver(d, tmp_path, sources=[src])
    with pytest.raises(fr.FragmentVerificationError, match="attestation"):
        r.resolve(3, 0, 4)

    r_off = resolver(d, tmp_path, sources=[FakeSource(remote_root,
                                                      drop_attestations=True)],
                     verify="off")
    assert r_off.resolve(3, 0, 4).origin == "fetched"


def test_verify_all_catches_local_corruption(tmp_path):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    seg = d / "layer-003.k3.safetensors"
    raw = bytearray(seg.read_bytes())
    entry = json.loads((d / "index-k3.json").read_text())["3"]
    raw[entry["body_offset"] + 1] ^= 0xFF  # corrupt expert 0's body
    seg.write_bytes(bytes(raw))
    r = resolver(d, tmp_path, verify="all")
    with pytest.raises(fr.FragmentVerificationError):
        r.resolve(3, 0, 3)
    # default policy trusts local reads
    r_default = resolver(d, tmp_path)
    assert r_default.resolve(3, 0, 3).origin == "local"


def test_unavailable_reports_chain(tmp_path):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = resolver(d, tmp_path)
    with pytest.raises(fr.FragmentUnavailableError, match="k5"):
        r.resolve(3, 0, 5)
