# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for ``swap.ResolverFragmentSource`` — the loader-v2 seam.

The swap engine's :class:`FragmentSource` is what makes a swap's bytes
appear. ``LocalSegmentSource`` requires an on-disk fq-segment dir;
``ResolverFragmentSource`` adapts ``fragments.FragmentResolver`` so a swap
stages through boot's supply chain instead: local dirs -> content-addressed
cache -> the manifest / ``VLLM_FQ_SOURCES`` chain of HF repos and mirrors,
with the same attestation trust filtering and per-fragment sha256
verification (implementation/10 §2/§4).

Two properties beyond "it reads bytes":

* **Byte fidelity** — an engine staged through the resolver must produce a
  layer state that is byte-identical to the same swap staged from local
  segments, which T4 already pinned to a fresh-built layer on GPU.
* **Pending promotions** (07 §1) — when a source refuses (untrusted
  attestation, sha mismatch, nothing has the fragment, or the K-fallback
  ladder substituted a lower K because the K4 encode has not landed), the
  promotion must PEND: the pair drops out of the swap list, cardinality is
  preserved, the interval proceeds. Corruption and structural faults stay
  fatal — dropping is only for supply.

The real ``FragmentResolver`` is used wherever it can be (real segments,
real sha verification, real trust filtering through a fake remote source);
hand-written fake resolvers cover the paths a real one cannot be forced into
cheaply (substituted K, structural corruption).
"""
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_fragments_cpu as tf  # noqa: E402  (FakeSource + resolver helpers)
import toy_segments as toy  # noqa: E402

fq_swap = toy.load_tree_module("swap")  # never the rootfs' synced copy
fr = tf.fr

E = 8
T0_GLOBALS = [0, 2, 4, 5, 6]  # K3 tier, slot order
T1_GLOBALS = [3, 1, 7]        # K4 tier, slot order


# ----------------------------------------------------------------- fixtures


def _expert_digests(root: Path, layer: int, k: int) -> dict[str, str]:
    index = json.loads((root / f"index-k{k}.json").read_text())[str(layer)]
    raw = (root / index["file"]).read_bytes()
    body = index["body_offset"]
    return {expert: hashlib.sha256(raw[body + lo:body + hi]).hexdigest()
            for expert, (lo, hi) in index["experts"].items()}


def make_segments(root: Path, *, attest=True, layer=toy.LAYER_ID) -> Path:
    """Toy fq-segment dir upgraded to what FragmentResolver expects:
    file-level ``sha256`` in the index + an fq-attestation/1 line carrying
    the per-expert digests (fq_repack writes both; toy_segments predates
    the resolver and writes neither)."""
    ckpt = toy.make_toy_checkpoint(E)
    toy.write_toy_segments(root, ckpt, layer=layer)
    for k in (3, 4):
        index_path = root / f"index-k{k}.json"
        index = json.loads(index_path.read_text())
        entry = index[str(layer)]
        entry["sha256"] = hashlib.sha256(
            (root / entry["file"]).read_bytes()).hexdigest()
        index_path.write_text(json.dumps(index))
        if attest:
            att = {"schema": "fq-attestation/1", "predicate": "repack-of",
                   "expert_sha256": _expert_digests(root, layer, k)}
            att_dir = root / "attestations"
            att_dir.mkdir(exist_ok=True)
            (att_dir / f"layer-{layer:03d}.k{k}.jsonl").write_text(
                json.dumps({
                    "payload": tf.base64.b64encode(
                        json.dumps(att).encode()).decode(),
                    "signature": "dGVzdA==",
                    "keyid": "00",
                }) + "\n")
    return ckpt


def manifest_dir(path: Path, **manifest) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "fq-manifest.json").write_text(json.dumps({
        "schema": "fq-manifest/1", "num_experts": E, "sources": [],
        **manifest}))
    return path


@pytest.fixture
def segments(tmp_path):
    root = tmp_path / "segments"
    ckpt = make_segments(root)
    return root, ckpt


def make_engine(source, state, **kw):
    kw.setdefault("expected_mcg", toy.MCG)
    return fq_swap.SwapEngine(
        {toy.LAYER_ID: state}, source,
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        pin_memory=False, build_maps_fn=toy.build_maps_reference, **kw)


PLAN = fq_swap.SwapPlan([(toy.LAYER_ID, 1, 4)])  # e_out=1 K4->K3, e_in=4


# --------------------------------------------------------------- byte fidelity


def test_resolver_source_stages_bitwise_like_local_segments(segments, tmp_path):
    """The adapter is byte-faithful: same plan, same layer, same bytes as
    LocalSegmentSource (which T4 pinned bitwise to a fresh-built layer)."""
    root, ckpt = segments
    local_state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    make_engine(fq_swap.LocalSegmentSource(root), local_state).apply(
        PLAN, quiesce=nullcontext())

    resolver = tf.resolver(root, tmp_path)  # local segment dir = manifest dir
    source = fq_swap.ResolverFragmentSource(resolver)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    report = make_engine(source, state).apply(PLAN, quiesce=nullcontext())

    assert report.pairs == 1 and report.dropped == ()
    assert source.reads == 2 and resolver.stats["local"] == 2
    toy.assert_states_equal(state, local_state)
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, [0, 2, 1, 5, 6], [3, 4, 7]))


def test_resolver_source_stages_from_a_remote_with_verification(tmp_path):
    """No local segments at all: the swap's bytes come from a remote source
    through ranged reads and are sha-verified before they are staged —
    boot's verification, applied to a runtime swap."""
    remote = tmp_path / "remote"
    ckpt = make_segments(remote)
    resolver = tf.resolver(manifest_dir(tmp_path / "empty"), tmp_path,
                           sources=[tf.FakeSource(remote)])
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    make_engine(fq_swap.ResolverFragmentSource(resolver), state).apply(
        PLAN, quiesce=nullcontext())

    assert resolver.stats["fetched"] == 2 and resolver.stats["verified"] == 2
    assert resolver.stats["local"] == 0
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, [0, 2, 1, 5, 6], [3, 4, 7]))


def test_resolver_source_reuses_the_fragment_cache(tmp_path):
    """Fetched fragments stay in the content-addressed cache: the next
    process staging the same swap — the other TP rank, or this rank after a
    restart — reads zero remote bytes and re-verifies from disk."""
    remote = tmp_path / "remote"
    ckpt = make_segments(remote)
    first = tf.resolver(manifest_dir(tmp_path / "empty"), tmp_path,
                        sources=[tf.FakeSource(remote)])
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    make_engine(fq_swap.ResolverFragmentSource(first), state).apply(
        PLAN, quiesce=nullcontext())
    assert first.stats["fetched"] == 2

    src2 = tf.FakeSource(remote)  # fresh resolver + source, same cache dir
    second = tf.resolver(manifest_dir(tmp_path / "empty"), tmp_path,
                         sources=[src2])
    peer = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    make_engine(fq_swap.ResolverFragmentSource(second), peer).apply(
        PLAN, quiesce=nullcontext())
    assert second.stats["cache"] == 2 and second.stats["fetched"] == 0
    assert src2.range_reads == 0, "re-staged the swap over the network"
    toy.assert_states_equal(peer, state)


# ------------------------------------------------------- trust rejection path


def untrusted_resolver(tmp_path, *, drop_attestations=True):
    """Real resolver, real trust filtering (10 §4): a signer is configured,
    so a remote whose attestation is missing/unsigned-by-it is refused."""
    remote = tmp_path / "remote"
    ckpt = make_segments(remote, attest=not drop_attestations)
    md = manifest_dir(tmp_path / "empty", signer_pubkey="ab" * 32)
    resolver = tf.resolver(
        md, tmp_path,
        sources=[tf.FakeSource(remote, drop_attestations=drop_attestations)])
    assert resolver.trust_enabled
    return resolver, ckpt


def test_trust_rejection_raises_by_default(tmp_path):
    """Fail-closed default: an untrusted source is an error, not a silent
    no-op — an operator who did not ask for pending promotions sees it."""
    resolver, ckpt = untrusted_resolver(tmp_path)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(fq_swap.ResolverFragmentSource(resolver), state)
    with pytest.raises(fq_swap.FragmentUnavailable, match="resolver refused"):
        engine.stage(PLAN)
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS))


def test_trust_rejection_drops_the_promotion(tmp_path):
    """07 §1 pending promotion: with ``on_unavailable="drop"`` the refused
    pair leaves the swap list instead of crashing the interval; both experts
    keep their tier, so per-layer K4 cardinality is untouched (D1)."""
    resolver, ckpt = untrusted_resolver(tmp_path)
    source = fq_swap.ResolverFragmentSource(resolver)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(source, state)

    staged = engine.stage(PLAN, on_unavailable="drop")
    assert len(staged.plan) == 0 and staged.requested_plan == PLAN
    assert [d.swap for d in staged.dropped] == [(toy.LAYER_ID, 1, 4)]
    assert "not-trusted" in staged.dropped[0].reason or \
        "no-attestation" in staged.dropped[0].reason
    assert staged.slab_ops == [] and staged.map_ops == []
    assert source.unavailable >= 1
    assert resolver.stats["reject_no_attestation"] >= 1

    report = engine.apply(staged=staged, quiesce=nullcontext())
    assert report.pairs == 0 and len(report.dropped) == 1
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS))


def test_partial_supply_applies_the_rest_and_preserves_cardinality(tmp_path):
    """Two-pair plan, one pair unavailable: the supplied pair applies, the
    refused one pends, occupancy stays at capacity, and the policy document
    the caller persists must be derived from ``staged.plan``."""
    root = tmp_path / "segments"
    ckpt = make_segments(root)
    resolver = tf.resolver(root, tmp_path)
    supplied = fq_swap.ResolverFragmentSource(resolver)

    class HalfSupply:
        """Refuses everything about expert 7 (its K4 encode has not landed)."""

        def read_expert(self, *, layer, k, expert, rank, dest):
            if expert == 7:
                raise fr.FragmentUnavailableError(
                    f"L{layer}/e{expert} K{k}: no source has it")
            supplied.read_expert(layer=layer, k=k, expert=expert, rank=rank,
                                 dest=dest)

    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(HalfSupply(), state, max_pairs=2)
    plan = fq_swap.SwapPlan([(toy.LAYER_ID, 1, 4), (toy.LAYER_ID, 7, 6)])

    staged = engine.stage(plan, on_unavailable="drop")
    assert staged.plan.swaps == ((toy.LAYER_ID, 1, 4),)
    assert [(d.swap, d.expert, d.k) for d in staged.dropped] == [
        ((toy.LAYER_ID, 7, 6), 7, 3)]
    report = engine.apply(staged=staged, quiesce=nullcontext())
    assert report.pairs == 1 and report.generation == 1

    # Only the supplied pair moved; expert 7 is still K4, 6 still K3.
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, [0, 2, 1, 5, 6], [3, 4, 7]))
    assert sorted(state.tier1_globals) == [3, 4, 7]
    assert len(state.tier1_globals) == len(T1_GLOBALS)  # cardinality preserved

    bits = {e: 4 if e in state.tier1_globals else 3 for e in range(E)}
    assert bits[7] == 4 and bits[6] == 3 and bits[4] == 4 and bits[1] == 3


# ------------------------------------------- fallback ladder / fake resolvers


class FakeResolver:
    """Minimal resolver stand-in: ``resolve`` + ``materialize`` is the whole
    protocol ResolverFragmentSource needs."""

    def __init__(self, inner, *, mutate=None, substitute_k=None, error=None):
        self.inner = inner
        self.mutate = mutate
        self.substitute_k = substitute_k
        self.error = error

    def resolve(self, layer, expert, k):
        if self.error is not None:
            raise self.error
        want = self.substitute_k or k
        fragment = self.inner.resolve(layer, expert, want)
        fragment.requested_k = k
        return fragment

    def materialize(self, fragment, *, name_filter=None):
        pairs = self.inner.materialize(fragment, name_filter=name_filter)
        return self.mutate(pairs) if self.mutate else pairs


def test_substituted_k_pends_the_promotion(segments, tmp_path):
    """The K-fallback ladder answering K4 with a K3 fragment means the encode
    has not landed (07 §1) — and its rows would not even fit the K4 slab.
    The promotion pends; the resolver has already queued the encode."""
    root, ckpt = segments
    resolver = FakeResolver(tf.resolver(root, tmp_path), substitute_k=3)
    source = fq_swap.ResolverFragmentSource(resolver)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(source, state)

    staged = engine.stage(PLAN, on_unavailable="drop")
    assert len(staged.plan) == 0
    assert "substituted K3" in staged.dropped[0].reason
    assert staged.dropped[0].expert == 4 and staged.dropped[0].k == 4
    # The demotion side (K3, expert 1) was supplied fine — only the K4 read
    # pended, and the pair went with it.
    with pytest.raises(fq_swap.FragmentUnavailable, match="substituted"):
        engine.stage(PLAN)


def test_structural_faults_stay_fatal_even_when_dropping(segments, tmp_path):
    """Corruption is not supply: a fragment that resolves but is malformed
    must fail the interval loudly, with or without ``drop``."""
    root, ckpt = segments
    inner = tf.resolver(root, tmp_path)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)

    def drop_a_tensor(pairs):
        return [(n, t) for n, t in pairs if not n.endswith("gate_proj.rank0.suh")]

    engine = make_engine(fq_swap.ResolverFragmentSource(
        FakeResolver(inner, mutate=drop_a_tensor)), state)
    with pytest.raises(KeyError, match="missing"):
        engine.stage(PLAN, on_unavailable="drop")

    def wrong_shape(pairs):
        return [(n, torch.zeros(3, dtype=t.dtype) if n.endswith(
            "up_proj.rank0.suh") else t) for n, t in pairs]

    engine = make_engine(fq_swap.ResolverFragmentSource(
        FakeResolver(inner, mutate=wrong_shape)), state)
    with pytest.raises(ValueError, match="stage expects"):
        engine.stage(PLAN, on_unavailable="drop")

    # A programming error in the resolver is not a supply failure either.
    engine = make_engine(fq_swap.ResolverFragmentSource(
        FakeResolver(inner, error=RuntimeError("boom"))), state)
    with pytest.raises(RuntimeError, match="boom"):
        engine.stage(PLAN, on_unavailable="drop")
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS))


def test_foreign_mcg_from_a_source_is_still_refused(segments, tmp_path):
    """The layer-wide MCG codebook check (r33 pins the LUT ABI) applies to
    resolver-staged fragments too — a foreign encoding is corruption."""
    root, ckpt = segments
    resolver = tf.resolver(root, tmp_path)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(fq_swap.ResolverFragmentSource(resolver), state,
                         expected_mcg=None)
    engine.expected_mcg = 12345
    with pytest.raises(ValueError, match="mcg"):
        engine.stage(PLAN, on_unavailable="drop")


def test_is_unavailable_error_classification():
    """The droppable set is matched by class name across the MRO so swap.py
    keeps importing standalone (the GPU harness loads it by path)."""
    assert fq_swap.is_unavailable_error(fr.FragmentUnavailableError("x"))
    assert fq_swap.is_unavailable_error(fr.FragmentVerificationError("x"))
    assert fq_swap.is_unavailable_error(fq_swap.FragmentUnavailable("x"))

    class MySourceUnavailable(fr.FragmentUnavailableError):
        pass

    assert fq_swap.is_unavailable_error(MySourceUnavailable("x"))
    for exc in (ValueError("x"), KeyError("x"), RuntimeError("x"),
                MemoryError()):
        assert not fq_swap.is_unavailable_error(exc)


def test_stage_rejects_unknown_on_unavailable(segments, tmp_path):
    root, ckpt = segments
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(fq_swap.LocalSegmentSource(root), state)
    with pytest.raises(ValueError, match="raise\\|drop"):
        engine.stage(PLAN, on_unavailable="ignore")


if __name__ == "__main__":
    sys.exit(pytest.main([
        __file__, "-v", "-p", "no:cacheprovider",
        "--confcutdir", os.path.dirname(os.path.abspath(__file__)),
    ]))
