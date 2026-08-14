# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the operator-facing FragmentResolver capabilities:
configurable multi-repo sources (VLLM_FQ_SOURCES[_MODE]), attestation-based
trust filtering (VLLM_FQ_TRUST_PREDICATES / VLLM_FQ_TRUST_SIGNERS), the
lazy-encode fallback ladder (VLLM_FQ_K_FALLBACK + EncodeQueue), and the
structured decision lines + per-reason stats counters."""
import base64
import json
import logging
import sys
from types import SimpleNamespace

import pytest

import test_fragments_cpu as tf
from test_fragments_cpu import (  # noqa: F401
    NUM_EXPERTS,
    FakeSource,
    make_manifest_dir,
    resolver,
    write_segment,
)

fr = tf.fr

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        lazy_encode as le,
    )
except ImportError:  # standalone run against an env without built vllm._C
    import importlib.util
    from pathlib import Path as _P

    _p = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible" / "lazy_encode.py")
    _s = importlib.util.spec_from_file_location("fq_lazy_encode_standalone", _p)
    if _s.name in sys.modules:
        le = sys.modules[_s.name]
    else:
        le = importlib.util.module_from_spec(_s)
        sys.modules[_s.name] = le
        _s.loader.exec_module(le)

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: E402
    Ed25519PrivateKey,
)


# ------------------------------------------------------------- helpers


@pytest.fixture
def fq_log():
    """Records emitted by the fragments logger (vllm's logger hierarchy has
    propagate=False, so caplog's root handler never sees them)."""
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Handler(level=logging.DEBUG)
    old_level = fr.logger.level
    fr.logger.addHandler(handler)
    fr.logger.setLevel(logging.DEBUG)
    yield records
    fr.logger.removeHandler(handler)
    fr.logger.setLevel(old_level)


def _messages(records):
    return [r.getMessage() for r in records]


class NamedSource(FakeSource):
    def __init__(self, root, name, **kw):
        super().__init__(root, **kw)
        self.name = name


def make_key():
    key = Ed25519PrivateKey.generate()
    return key, key.public_key().public_bytes_raw().hex()


def att_line(key, pub, digests, predicate="repack-of"):
    """One signed fq-attestation/1 envelope line (fq_repack Signer format)."""
    payload = {
        "schema": "fq-attestation/1",
        "predicate": predicate,
        "expert_sha256": digests,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return json.dumps({
        "payload": base64.b64encode(raw).decode(),
        "signature": base64.b64encode(key.sign(raw)).decode(),
        "keyid": pub,
    })


def signed_remote(root, layer, k, *, seed, make_lines):
    """Remote segment root whose attestation lines come from make_lines."""
    _entry, digests = write_segment(
        root, layer, k, seed=seed, with_attestation=False
    )
    att_dir = root / "attestations"
    att_dir.mkdir(exist_ok=True)
    (att_dir / f"layer-{layer:03d}.k{k}.jsonl").write_text(
        "\n".join(make_lines(digests)) + "\n"
    )
    return digests


def trusted_manifest_dir(tmp_path, pub, ks=(3,)):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=ks)
    manifest = json.loads((d / "fq-manifest.json").read_text())
    manifest["signer_pubkey"] = pub
    (d / "fq-manifest.json").write_text(json.dumps(manifest))
    return d


def expected_payload(root, layer, k, expert):
    entry = json.loads((root / f"index-k{k}.json").read_text())[str(layer)]
    lo, hi = entry["experts"][str(expert)]
    raw = (root / entry["file"]).read_bytes()
    return raw[entry["body_offset"] + lo:entry["body_offset"] + hi]


# ------------------------------------------- feature 1: source config


def test_env_sources_mode_variants(tmp_path):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    manifest = json.loads((d / "fq-manifest.json").read_text())
    manifest["sources"] = ["org/manifest-src"]
    (d / "fq-manifest.json").write_text(json.dumps(manifest))
    factory = lambda spec: SimpleNamespace(name=spec, spec=spec)  # noqa: E731

    def specs(environ):
        r = resolver(d, tmp_path, environ=environ, source_factory=factory)
        return [s.spec for s in r.sources]

    env = {"VLLM_FQ_SOURCES": "repoA@r1,repoB"}
    assert specs(env) == ["repoA@r1", "repoB", "org/manifest-src"]  # prepend
    assert specs({**env, "VLLM_FQ_SOURCES_MODE": "prepend"}) == [
        "repoA@r1", "repoB", "org/manifest-src"]
    assert specs({**env, "VLLM_FQ_SOURCES_MODE": "append"}) == [
        "org/manifest-src", "repoA@r1", "repoB"]
    assert specs({**env, "VLLM_FQ_SOURCES_MODE": "replace"}) == [
        "repoA@r1", "repoB"]
    assert specs({}) == ["org/manifest-src"]  # no env: manifest chain
    # dedup: env entry already in the manifest chain appears once
    assert specs({"VLLM_FQ_SOURCES": "org/manifest-src,repoB"}) == [
        "org/manifest-src", "repoB"]
    with pytest.raises(ValueError, match="VLLM_FQ_SOURCES_MODE"):
        specs({**env, "VLLM_FQ_SOURCES_MODE": "sideways"})


def test_env_sources_resolve_order_behavioral(tmp_path):
    root_env, root_man = tmp_path / "remote-env", tmp_path / "remote-man"
    write_segment(root_env, 3, 4, seed=5)
    write_segment(root_man, 3, 4, seed=9)
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    manifest = json.loads((d / "fq-manifest.json").read_text())
    manifest["sources"] = ["man/src"]
    (d / "fq-manifest.json").write_text(json.dumps(manifest))
    roots = {"env/src": root_env, "man/src": root_man}
    factory = lambda spec: NamedSource(roots[spec], f"fake:{spec}")  # noqa: E731

    r = resolver(d, tmp_path, environ={"VLLM_FQ_SOURCES": "env/src"},
                 source_factory=factory)
    # local dirs always first: the k3 hit never consults any source
    assert r.resolve(3, 0, 3).origin == "local"
    frag = r.resolve(3, 0, 4)  # prepend: env source wins
    assert bytes(frag.payload) == expected_payload(root_env, 3, 4, 0)

    r2 = resolver(d, tmp_path / "c2",
                  environ={"VLLM_FQ_SOURCES": "env/src",
                           "VLLM_FQ_SOURCES_MODE": "append"},
                  source_factory=factory, cache_dir=tmp_path / "c2" / "cache")
    frag2 = r2.resolve(3, 0, 4)  # append: manifest source wins
    assert bytes(frag2.payload) == expected_payload(root_man, 3, 4, 0)


def test_explicit_sources_kwarg_bypasses_env(tmp_path):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    marker = SimpleNamespace(name="explicit")
    r = resolver(d, tmp_path, sources=[marker],
                 environ={"VLLM_FQ_SOURCES": "repoA"})
    assert r.sources == [marker]


# ------------------------------------------- feature 2: trust filtering


def test_predicate_filtering(tmp_path, fq_log):
    key, pub = make_key()
    remote = tmp_path / "remote"
    signed_remote(remote, 3, 4, seed=9,
                  make_lines=lambda digests: [att_line(key, pub, digests)])
    d = trusted_manifest_dir(tmp_path, pub)

    # default predicate list trusts repack-of
    r_ok = resolver(d, tmp_path, sources=[FakeSource(remote)])
    assert r_ok.trust_enabled
    assert r_ok.resolve(3, 0, 4).origin == "fetched"

    # restricted list rejects it
    r = resolver(d, tmp_path / "c2", cache_dir=tmp_path / "c2" / "cache",
                 sources=[FakeSource(remote)],
                 environ={"VLLM_FQ_TRUST_PREDICATES": "encode-of"})
    with pytest.raises(fr.FragmentUnavailableError):
        r.resolve(3, 0, 4)
    assert r.stats["reject_predicate"] == 1
    assert r.stats["fetched"] == 0
    assert any("REJECT predicate=repack-of not-trusted" in m
               for m in _messages(fq_log))


def test_signer_rejection_and_countersignature(tmp_path):
    key, pub = make_key()
    rogue_key, rogue_pub = make_key()
    d = trusted_manifest_dir(tmp_path, pub)

    remote_rogue = tmp_path / "remote-rogue"
    signed_remote(remote_rogue, 3, 4, seed=9, make_lines=lambda digests: [
        att_line(rogue_key, rogue_pub, digests)])
    r = resolver(d, tmp_path, sources=[FakeSource(remote_rogue)])
    with pytest.raises(fr.FragmentUnavailableError, match="signer not-trusted"):
        r.resolve(3, 0, 4)
    assert r.stats["reject_signer"] == 1

    # countersigned file: rogue line + allowed line -> ANY allowed accepts
    remote_counter = tmp_path / "remote-counter"
    signed_remote(remote_counter, 3, 4, seed=9, make_lines=lambda digests: [
        att_line(rogue_key, rogue_pub, digests),
        att_line(key, pub, digests)])
    r2 = resolver(d, tmp_path / "c2", cache_dir=tmp_path / "c2" / "cache",
                  sources=[FakeSource(remote_counter)])
    assert r2.resolve(3, 0, 4).origin == "fetched"
    assert r2.stats["reject_signer"] == 0

    # VLLM_FQ_TRUST_SIGNERS overrides the manifest anchor
    r3 = resolver(d, tmp_path / "c3", cache_dir=tmp_path / "c3" / "cache",
                  sources=[FakeSource(remote_rogue)],
                  environ={"VLLM_FQ_TRUST_SIGNERS": rogue_pub})
    assert r3.resolve(3, 0, 4).origin == "fetched"


def test_bad_signature_rejected(tmp_path):
    key, pub = make_key()
    d = trusted_manifest_dir(tmp_path, pub)
    remote = tmp_path / "remote"

    def forged(digests):
        line = json.loads(att_line(key, pub, digests))
        line["signature"] = base64.b64encode(b"\x00" * 64).decode()
        return [json.dumps(line)]

    signed_remote(remote, 3, 4, seed=9, make_lines=forged)
    r = resolver(d, tmp_path, sources=[FakeSource(remote)])
    with pytest.raises(fr.FragmentUnavailableError, match="bad-signature"):
        r.resolve(3, 0, 4)
    assert r.stats["reject_signature"] == 1
    assert r.stats["bytes_fetched"] == 0  # rejected before any body read


def test_trust_disabled_without_anchor_keeps_legacy_behavior(tmp_path):
    # no signer_pubkey in the manifest, no env: garbage signatures pass
    remote = tmp_path / "remote"
    write_segment(remote, 3, 4, seed=9)  # test-signature attestation
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = resolver(d, tmp_path, sources=[FakeSource(remote)])
    assert not r.trust_enabled
    assert r.resolve(3, 0, 4).origin == "fetched"


# ------------------------------------- feature 3: fallback + lazy encode


def test_fallback_substitution_surfaces_actual_k(tmp_path, fq_log):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))  # no k4 anywhere
    queue_path = tmp_path / "queue.jsonl"
    r = resolver(d, tmp_path, environ={
        "VLLM_FQ_K_FALLBACK": "3",
        "VLLM_FQ_ENCODE_QUEUE": str(queue_path)})
    frag = r.resolve(3, 2, 4)
    assert frag.k == 3 and frag.requested_k == 4 and frag.substituted
    assert frag.origin == "local"
    assert r.stats["fallback_substituted"] == 1
    assert r.stats["encode_queued"] == 1

    info = [rec for rec in fq_log if rec.levelno == logging.INFO]
    assert len(info) == 1
    msg = info[0].getMessage()
    assert msg.startswith("FQ resolve L3/e2 K4:")
    assert "FALLBACK K3" in msg and "ACCEPT (encode queued #1)" in msg

    entries = le.EncodeQueue(queue_path).entries()
    assert [(e["layer"], e["expert"], e["k"]) for e in entries] == [(3, 2, 4)]
    assert entries[0]["reason"] == "substituted-with-k3"

    # dedup: a second resolve substitutes again but does not re-enqueue
    frag2 = r.resolve(3, 2, 4)
    assert frag2.substituted and r.stats["encode_queued"] == 1
    assert any("(encode already queued)" in m for m in _messages(fq_log))


def test_unavailable_miss_is_queued_and_counted(tmp_path, fq_log):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    queue_path = tmp_path / "queue.jsonl"
    r = resolver(d, tmp_path,
                 environ={"VLLM_FQ_ENCODE_QUEUE": str(queue_path)})
    with pytest.raises(fr.FragmentUnavailableError, match="k5"):
        r.resolve(3, 0, 5)
    assert r.stats["unavailable"] == 1
    assert r.stats["encode_queued"] == 1
    entries = le.EncodeQueue(queue_path).entries()
    assert [(e["layer"], e["expert"], e["k"]) for e in entries] == [(3, 0, 5)]
    assert entries[0]["reason"] == "unavailable"
    warning = [rec for rec in fq_log if rec.levelno == logging.WARNING]
    assert any("UNAVAILABLE (encode queued #1)" in rec.getMessage()
               for rec in warning)


def test_progressive_stream_surfaces_substituted_k(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841
    import test_progressive_cpu as tpc

    seg_dir = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))  # no k4
    spec = tpc.make_spec(tmp_path, tpc.make_policy({"3": [4, 3, 3, 4]}),
                         seg_dir=seg_dir)
    resolver_ = spec.make_resolver(
        cache_dir=tmp_path / "cache",
        environ={"VLLM_FQ_K_FALLBACK": "3",
                 "VLLM_FQ_ENCODE_QUEUE": str(tmp_path / "queue.jsonl")})
    logs, actual = [], {}
    tensors = dict(tpc.pg.progressive_weights_iterator(
        spec, resolver_, log=logs.append, actual_bits_out=actual))

    # reality: every expert streamed at K3, and the metadata says so
    name = "model.layers.3.mlp.experts.0.gate_proj.rank0.trellis"
    assert tuple(tensors[name].shape) == (2, 2, 16 * 3)
    assert actual == {3: [3, 3, 3, 3]}
    layer_line = next(line for line in logs if "layer 3:" in line)
    assert "tiers=((3, 4),)" in layer_line
    assert f"bits_digest={tpc.pg._bits_digest([3, 3, 3, 3])}" in layer_line
    assert "substituted=e0:K4->K3,e3:K4->K3" in layer_line
    assert resolver_.stats["fallback_substituted"] == 2

    # the tier bitmap can be rewritten from reality
    bitmap = tpc.pg.write_tier_bitmap(
        spec.policy, tmp_path / "bitmap.json", actual_bits=actual)
    assert json.loads(bitmap.read_text())["3"]["bits_per_expert"] == [3, 3, 3, 3]

    queued = le.EncodeQueue(tmp_path / "queue.jsonl").entries()
    assert [(e["layer"], e["expert"], e["k"]) for e in queued] == [
        (3, 0, 4), (3, 3, 4)]


def test_queue_dedup_and_persistence(tmp_path):
    path = tmp_path / "q.jsonl"
    q = le.EncodeQueue(path)
    assert q.enqueue(3, 0, 4, "unavailable") == (1, True)
    assert q.enqueue(3, 0, 4, "again") == (1, False)  # dedup by key
    assert q.enqueue(3, 1, 4, "unavailable") == (2, True)
    assert len(path.read_text().splitlines()) == 2

    q2 = le.EncodeQueue(path)  # persistence across processes
    assert len(q2) == 2
    assert q2.enqueue(3, 0, 4, "later") == (1, False)
    assert q2.enqueue(5, 7, 4, "unavailable") == (3, True)


def _bf16_dir(tmp_path, layer_experts=((3, 0),)):
    d = tmp_path / "bf16"
    d.mkdir(exist_ok=True)
    weight_map = {}
    for layer, expert in layer_experts:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            weight_map[
                f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
            ] = "model-x.safetensors"
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    return d


def _capture_dir(tmp_path, layers=(3,)):
    d = tmp_path / "capture"
    d.mkdir(exist_ok=True)
    for layer in layers:
        sub = d / f"layer_{layer:03d}"
        sub.mkdir(exist_ok=True)
        (sub / "payload.bin").write_bytes(b"x")
    return d


def test_drain_dry_run_validates_without_gpu(tmp_path):
    q = le.EncodeQueue(tmp_path / "q.jsonl")
    q.enqueue(3, 0, 4, "substituted-with-k3")
    q.enqueue(7, 0, 4, "unavailable")  # no capture for layer 7
    bf16 = _bf16_dir(tmp_path, layer_experts=((3, 0), (7, 0)))
    capture = _capture_dir(tmp_path, layers=(3,))

    lines = []
    rc = le.drain(q, dry_run=True, bf16_dir=bf16, capture_dir=capture,
                  environ={}, out=lines.append)
    assert rc == 1  # one entry blocked
    ok = next(line for line in lines if line.startswith("#1"))
    assert "DRY-RUN OK" in ok and "bf16=ok(index)" in ok
    assert "--bits 4" in ok and "--layers 3" in ok  # default driver template
    blocked = next(line for line in lines if line.startswith("#2"))
    assert "BLOCKED" in blocked and "capture missing layer_007" in blocked
    # dry run never mutates the queue
    assert len(le.EncodeQueue(tmp_path / "q.jsonl")) == 2

    rc_ok = le.drain(q, dry_run=True, bf16_dir=bf16,
                     capture_dir=_capture_dir(tmp_path, layers=(3, 7)),
                     environ={}, out=lines.append)
    assert rc_ok == 0


def test_drain_execute_runs_template_and_dequeues(tmp_path):
    q = le.EncodeQueue(tmp_path / "q.jsonl")
    q.enqueue(3, 0, 4, "unavailable")
    bf16 = _bf16_dir(tmp_path)
    capture = _capture_dir(tmp_path)

    calls, lines = [], []

    def runner(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    rc = le.drain(
        q, dry_run=False, bf16_dir=bf16, capture_dir=capture, environ={},
        encoder_cmd=("encoder --bits {k} --layers {layer} --expert {expert} "
                     "--src {bf16_dir} --capture-dir {capture_dir}"),
        out=lines.append, runner=runner)
    assert rc == 0
    assert calls == [["encoder", "--bits", "4", "--layers", "3", "--expert",
                      "0", "--src", str(bf16), "--capture-dir", str(capture)]]
    assert any("DONE" in line for line in lines)
    assert len(le.EncodeQueue(tmp_path / "q.jsonl")) == 0  # drained


# --------------------------------------------- decision lines + stats


def test_decision_line_full_chain(tmp_path, fq_log):
    """The operator example: predicate reject, sha-mismatch reject, fallback
    accept with an encode queued — one INFO line carrying the whole chain."""
    key, pub = make_key()
    d = trusted_manifest_dir(tmp_path, pub)  # local k3 only

    remote_a = tmp_path / "remote-a"
    signed_remote(remote_a, 3, 4, seed=5, make_lines=lambda digests: [
        att_line(key, pub, digests, predicate="derived-from")])
    remote_b = tmp_path / "remote-b"
    signed_remote(remote_b, 3, 4, seed=6, make_lines=lambda digests: [
        att_line(key, pub, digests)])

    r = resolver(
        d, tmp_path,
        sources=[NamedSource(remote_a, "hf:repoA@ab12"),
                 NamedSource(remote_b, "hf:repoB@cd34",
                             corrupt_fragments=True)],
        environ={"VLLM_FQ_TRUST_PREDICATES": "repack-of,encode-of",
                 "VLLM_FQ_K_FALLBACK": "3",
                 "VLLM_FQ_ENCODE_QUEUE": str(tmp_path / "queue.jsonl")})
    frag = r.resolve(3, 0, 4)
    assert frag.k == 3 and frag.substituted

    info = [rec for rec in fq_log if rec.levelno == logging.INFO]
    assert len(info) == 1
    msg = info[0].getMessage()
    assert msg.startswith("FQ resolve L3/e0 K4: ")
    assert "local(1 dirs) MISS" in msg
    assert "hf:repoA@ab12 REJECT predicate=derived-from not-trusted" in msg
    assert "hf:repoB@cd34 REJECT sha-mismatch" in msg
    assert "FALLBACK K3 local ACCEPT (encode queued #1)" in msg

    assert r.stats["reject_predicate"] == 1
    assert r.stats["reject_sha_mismatch"] == 1
    assert r.stats["fallback_substituted"] == 1
    assert r.stats["encode_queued"] == 1
    assert r.stats["local"] == 1  # the K3 fallback hit


def test_plain_success_logs_debug_line(tmp_path, fq_log):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = resolver(d, tmp_path)
    assert r.resolve(3, 1, 3).origin == "local"
    debug = [rec for rec in fq_log if rec.levelno == logging.DEBUG]
    assert [rec.getMessage() for rec in debug] == [
        "FQ resolve L3/e1 K3: local ACCEPT"]
    assert not [rec for rec in fq_log if rec.levelno > logging.DEBUG]


def test_source_miss_counted(tmp_path):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    empty = tmp_path / "empty-remote"
    empty.mkdir()
    r = resolver(d, tmp_path, sources=[FakeSource(empty)])
    with pytest.raises(fr.FragmentUnavailableError):
        r.resolve(3, 0, 4)
    assert r.stats["source_miss"] == 1
    assert r.stats["unavailable"] == 1


def test_invalid_k_fallback_rejected(tmp_path):
    d = make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    with pytest.raises(ValueError, match="VLLM_FQ_K_FALLBACK"):
        resolver(d, tmp_path, environ={"VLLM_FQ_K_FALLBACK": "three"})
