# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the "weight not available at the required K" contract.

The operator requirement these pin: *a weight that is not available at the
required K bpw must never crash the serve*. Preference order, in this file's
terms:

1. serve the requested K;
2. else the nearest available LOWER K, said loudly, encode queued
   (``FragmentResolver.resolve_best`` + the auto fallback ladder);
3. else keep the incumbent tier — the pair drops out of the swap batch
   (``SwapEngine.stage(on_unavailable="drop")``) or the whole apply is
   downgraded to "not applied" (``FungibleQuantState._maybe_apply``);
4. never raise into an engine loop.

Every failure mode is injected with fakes: no network, no GPU, no model.
Covered here: a resolver that has nothing, a resolver that has only a lower
K, a source that times out, a corrupt local segment dir, an unwritable
fragment cache, a mid-swap disappearance, an apply backend that throws, and
the lazy-encode queue end to end (miss -> queue file -> drain command),
including a torn/garbage queue file.
"""
import errno
import json
import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
# The gg rootfs carries a periodically-synced copy of the package in its
# site-packages, so a direct ``python test_missing_k_cpu.py`` would grade
# whatever was last copied there. Put the working tree first, exactly as the
# pytest-from-repo-root invocation does, so both ways test the same code.
sys.path.insert(0, str(_HERE.parents[2]))
sys.path.insert(0, str(_HERE.parent))
import test_fragments_cpu as tf  # noqa: E402
import toy_segments as toy  # noqa: E402

fr = tf.fr
fq_swap = toy.load_tree_module("swap")

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        lazy_encode as le,
    )
except ImportError:  # standalone run without a built vllm._C
    le = toy.load_tree_module("lazy_encode")


E = 8
T0_GLOBALS = [0, 2, 4, 5, 6]  # K3 tier, slot order
T1_GLOBALS = [3, 1, 7]        # K4 tier, slot order
PLAN = fq_swap.SwapPlan([(toy.LAYER_ID, 1, 4)])  # e_out=1 K4->K3, e_in=4 up


@pytest.fixture
def fq_log():
    """Records from the fragments logger (vllm loggers do not propagate)."""
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Handler(level=logging.DEBUG)
    old = fr.logger.level
    fr.logger.addHandler(handler)
    fr.logger.setLevel(logging.DEBUG)
    yield records
    fr.logger.removeHandler(handler)
    fr.logger.setLevel(old)


def _messages(records, level=None):
    return [r.getMessage() for r in records
            if level is None or r.levelno == level]


# ============================================================ the resolver
# ---------------------------------------------- 1. nothing can supply it


def test_resolve_best_returns_none_when_nothing_has_it(tmp_path, fq_log):
    """No local segment, no cache, no source, no other K: ``resolve_best``
    reports None instead of raising, and the miss is queued for encode."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=())  # empty segment dir
    queue_path = tmp_path / "queue.jsonl"
    r = tf.resolver(d, tmp_path,
                    environ={"VLLM_FQ_ENCODE_QUEUE": str(queue_path)})

    assert r.resolve_best(3, 0, 4) is None
    assert r.stats["unavailable"] == 1 and r.stats["encode_queued"] == 1
    assert any("UNAVAILABLE" in m for m in _messages(fq_log))
    assert any("keeping the incumbent tier" in m
               for m in _messages(fq_log, logging.ERROR))
    entries = le.EncodeQueue(queue_path).entries()
    assert [(e["layer"], e["expert"], e["k"]) for e in entries] == [(3, 0, 4)]

    # strict resolve() still raises — tooling keeps its fail-closed contract
    with pytest.raises(fr.FragmentUnavailableError):
        r.resolve(3, 0, 4)


def test_resolve_best_survives_a_resolver_bug(tmp_path):
    """Even an internal error is reported as "unavailable": nothing about a
    fragment may reach the engine loop as an exception."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = tf.resolver(d, tmp_path)

    def boom(*_a, **_kw):
        raise MemoryError("resolver bug")

    r._resolve_k = boom  # noqa: SLF001
    assert r.resolve_best(3, 0, 4) is None
    assert r.stats["resolve_error"] == 1


# ------------------------------------------- 2. only a lower K is present


def test_auto_ladder_substitutes_the_nearest_lower_k(tmp_path, fq_log):
    """Requested K5, only K2/K3 on disk: K3 (nearest lower) wins, the
    substitution is logged loudly and the K5 encode is queued."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(2, 3))
    queue_path = tmp_path / "queue.jsonl"
    r = tf.resolver(d, tmp_path,
                    environ={"VLLM_FQ_ENCODE_QUEUE": str(queue_path)})

    assert r.k_universe() == (2, 3)
    assert r.fallback_ladder(5) == (3, 2)  # nearest lower first

    frag = r.resolve_best(3, 1, 5)
    assert frag is not None
    assert frag.k == 3 and frag.requested_k == 5 and frag.substituted
    assert r.stats["fallback_substituted"] == 1
    loud = _messages(fq_log, logging.WARNING)
    assert any("FQ DEGRADED L3/e1: K5 unavailable, serving K3" in m
               for m in loud)
    assert [(e["layer"], e["expert"], e["k"])
            for e in le.EncodeQueue(queue_path).entries()] == [(3, 1, 5)]


def test_auto_ladder_never_climbs_unless_asked(tmp_path):
    """A higher K is not a safe substitute (more memory; on SM120 K5 does
    not serve as a mixed tier at all), so upward substitution is opt-in."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(4,))
    r = tf.resolver(d, tmp_path)
    assert r.fallback_ladder(3) == ()
    assert r.resolve_best(3, 0, 3) is None

    up = tf.resolver(d, tmp_path / "c2", cache_dir=tmp_path / "c2" / "cache",
                     environ={"VLLM_FQ_K_FALLBACK_UP": "1"})
    assert up.fallback_ladder(3) == (4,)
    frag = up.resolve_best(3, 0, 3)
    assert frag is not None and frag.k == 4 and frag.requested_k == 3


def test_fallback_off_disables_substitution(tmp_path):
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = tf.resolver(d, tmp_path, environ={"VLLM_FQ_K_FALLBACK": "off"})
    assert r.fallback_ladder(4) == ()
    assert r.resolve_best(3, 0, 4) is None


def test_explicit_ladder_still_wins(tmp_path):
    """An operator-listed ladder is honoured verbatim by both entry points
    (that is the pre-existing VLLM_FQ_K_FALLBACK contract)."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(2, 3))
    r = tf.resolver(d, tmp_path, environ={"VLLM_FQ_K_FALLBACK": "2"})
    assert r.fallback_ladder(4) == (2,)
    assert r.resolve(3, 0, 4).k == 2
    assert r.resolve_best(3, 0, 4).k == 2


def test_available_k_probes_without_reading_payloads(tmp_path):
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = tf.resolver(d, tmp_path)
    assert r.available_k(3, 0, 3) == 3
    assert r.available_k(3, 0, 4) == 3           # would be substituted
    assert r.available_k(3, 0, 2) is None        # nothing lower than 2
    assert r.available_k(9, 0, 3) is None        # unknown layer
    assert r.available_k(3, 99, 3) is None       # unknown expert


def test_project_bits_to_available_matches_what_boot_would_stream(tmp_path):
    """The boot-side projection: a tier bitmap built from this cannot ask
    for a K the resolver would have to substitute, which is what keeps a
    degraded boot from failing a slab shape check instead of crashing."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = tf.resolver(d, tmp_path)
    lines = []
    projected, subs, missing = fr.project_bits_to_available(
        r, {3: [4, 3, 5, 3]}, log=lines.append)

    assert projected == {3: [3, 3, 3, 3]}
    assert subs == [(3, 0, 4, 3), (3, 2, 5, 3)]
    assert missing == []
    assert any("K4 unavailable -> K3" in m for m in lines)

    # an expert nothing can supply is reported, not silently demoted
    _p, _s, missing2 = fr.project_bits_to_available(r, {9: [4]})
    assert missing2 == [(9, 0, 4)]


# --------------------------------------------- 3. a source that times out


class TimeoutSource:
    """A mirror that hangs: every access raises, like a real HF timeout."""

    def __init__(self, name="hf:slow/mirror", exc=None):
        self.name = name
        self.exc = exc or TimeoutError("timed out")
        self.calls = 0

    def _die(self, *_a, **_kw):
        self.calls += 1
        raise self.exc

    read_json = read_text = read_range = _die


def test_timing_out_source_degrades_to_the_local_lower_k(tmp_path, fq_log):
    """A dead mirror must not be an exception path: it is one REJECT segment
    in the decision chain, and the local K3 still serves the request."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    src = TimeoutSource()
    r = tf.resolver(d, tmp_path, sources=[src])

    frag = r.resolve_best(3, 0, 4)
    assert frag is not None and frag.k == 3 and frag.requested_k == 4
    assert src.calls > 0 and r.stats["source_error"] > 0
    assert any("REJECT error:TimeoutError" in m for m in _messages(fq_log))


def test_timing_out_source_with_nothing_local_returns_none(tmp_path):
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=())
    r = tf.resolver(d, tmp_path, sources=[TimeoutSource()])
    assert r.resolve_best(3, 0, 4) is None
    assert r.stats["unavailable"] == 1


def test_corrupt_cached_header_is_not_a_fatal_error(tmp_path):
    """Regression: the fragment-cache step's tensor-table lookup ran the
    segment-header read unguarded, so a corrupt (or half-written) cached
    header escaped ``resolve()`` as a raw JSONDecodeError — fatal at boot,
    and *not* classified as droppable by the swap engine either."""
    remote = tmp_path / "remote"
    tf.write_segment(remote, 3, 4, seed=9)
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))

    warm = tf.resolver(d, tmp_path, sources=[tf.FakeSource(remote)])
    assert warm.resolve(3, 0, 4).origin == "fetched"  # populates the cache
    headers = list((tmp_path / "cache" / "headers").glob("*.json"))
    assert headers
    for p in headers:
        p.write_text("{not json")  # torn write of a cached header

    cold = tf.resolver(d, tmp_path, sources=[tf.FakeSource(remote)])
    frag = cold.resolve(3, 0, 4)  # strict path must not raise either
    assert frag is not None and frag.k == 4
    # and the poisoned entry self-heals rather than pinning the segment as
    # unreadable for the rest of the process
    assert cold.stats["cache_error"] > 0
    assert json.loads(headers[0].read_text())  # rewritten, parseable again


def test_mirror_that_dies_mid_header_is_droppable_not_fatal(tmp_path):
    """A mirror that times out re-reading a segment header comes back as
    "unavailable" (droppable supply failure), never as a raw TimeoutError
    the swap engine would treat as structural and re-raise."""
    remote = tmp_path / "remote"
    tf.write_segment(remote, 3, 4, seed=9)
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=())

    class DyingAfterIndex(tf.FakeSource):
        """Serves the index (so the entry is found) but not the header."""

        def read_range(self, relpath, start, end):
            raise TimeoutError("mirror died mid-header")

    r = tf.resolver(d, tmp_path, sources=[DyingAfterIndex(remote)])
    assert r.resolve_best(3, 0, 4) is None
    with pytest.raises(fr.FragmentUnavailableError):
        r.resolve(3, 0, 4)
    assert fq_swap.is_unavailable_error(fr.FragmentUnavailableError("x"))


# ----------------------------------- 4. broken local dirs / unusable cache


def test_corrupt_local_index_is_a_miss_not_a_crash(tmp_path, fq_log):
    """A truncated index-k4.json used to escape as JSONDecodeError from
    every caller of resolve(). It must degrade that dir to a miss."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3, 4))
    (d / "index-k4.json").write_text('{"3": {"file": ')  # torn write

    r = tf.resolver(d, tmp_path)
    frag = r.resolve_best(3, 0, 4)
    assert frag is not None and frag.k == 3  # fell back to the intact K3
    assert r.stats["local_error"] == 1
    assert any("unusable for L3 K4" in m
               for m in _messages(fq_log, logging.WARNING))

    # the strict entry point must not leak the parse error either
    strict = tf.resolver(d, tmp_path / "c2", cache_dir=tmp_path / "c2" / "c",
                         environ={"VLLM_FQ_K_FALLBACK": "3"})
    assert strict.resolve(3, 0, 4).k == 3


def test_segment_index_skew_is_a_miss_not_a_crash(tmp_path):
    """index/segment body-offset skew is corruption of one dir, not of the
    deployment: other Ks and other sources must still be reachable."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3, 4))
    index = json.loads((d / "index-k4.json").read_text())
    index["3"]["body_offset"] = int(index["3"]["body_offset"]) + 8
    (d / "index-k4.json").write_text(json.dumps(index))

    r = tf.resolver(d, tmp_path, environ={"VLLM_FQ_K_FALLBACK": "3"})
    assert r.resolve(3, 0, 4).k == 3          # strict path: no ValueError
    assert r.resolve_best(3, 0, 4).k == 3
    assert r.stats["local_error"] == 1


def test_unwritable_cache_still_serves_the_fetched_fragment(tmp_path):
    """Disk full / read-only cache: the fragment was fetched and verified,
    so it must be served. It used to be lost to an OSError from the cache
    write, taking the boot or the swap with it."""
    remote = tmp_path / "remote"
    tf.write_segment(remote, 3, 4, seed=9)
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = tf.resolver(d, tmp_path, sources=[tf.FakeSource(remote)])

    def enospc(_path, _data):
        raise OSError(errno.ENOSPC, "No space left on device")

    r._atomic_write = enospc  # noqa: SLF001
    frag = r.resolve(3, 0, 4)  # strict path: the fetch must not be lost
    assert frag.k == 4 and frag.origin == "fetched"
    assert r.stats["cache_write_error"] > 0
    assert r.resolve_best(3, 0, 4).k == 4


# ================================================= the live swap (M4 path)


def make_engine(source, state, **kw):
    kw.setdefault("expected_mcg", toy.MCG)
    return fq_swap.SwapEngine(
        {toy.LAYER_ID: state}, source,
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        pin_memory=False, build_maps_fn=toy.build_maps_reference, **kw)


class VanishingResolver:
    """Wraps a real resolver; the named expert disappears at a chosen K.

    Models the mid-swap disappearance: the plan was decided while the K4
    fragment was reachable, and by staging time it is not."""

    def __init__(self, inner, *, gone=(), mode="none"):
        self.inner = inner
        self.gone = set(gone)          # {(expert, k)}
        self.mode = mode               # "none" | "raise"

    def resolve_best(self, layer, expert, k, *, chain_out=None):
        if (expert, k) in self.gone:
            if chain_out is not None:
                chain_out.append("local(1 dirs) MISS")
            return None
        return self.inner.resolve_best(layer, expert, k, chain_out=chain_out)

    def resolve(self, layer, expert, k):
        if (expert, k) in self.gone:
            raise fr.FragmentUnavailableError(
                f"L{layer}/e{expert} K{k}: vanished")
        return self.inner.resolve(layer, expert, k)

    def materialize(self, fragment, *, name_filter=None):
        return self.inner.materialize(fragment, name_filter=name_filter)


@pytest.fixture
def segments(tmp_path):
    import test_swap_resolver_cpu as tsr

    root = tmp_path / "segments"
    ckpt = tsr.make_segments(root)
    return root, ckpt


def test_midswap_disappearance_drops_the_pair_and_leaves_no_trace(
        segments, tmp_path):
    """The promotion target vanishes between decide and stage. With
    ``drop`` the pair pends: no slab op, no map op, no ordering change, and
    the layer is byte-identical to before. Fragment IO happens entirely
    before ``apply`` opens the quiesce window, so there is no torn state to
    roll back from in the first place."""
    root, ckpt = segments
    resolver = VanishingResolver(tf.resolver(root, tmp_path), gone={(4, 4)})
    source = fq_swap.ResolverFragmentSource(resolver)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    before = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(source, state)

    staged = engine.stage(PLAN, on_unavailable="drop")
    assert len(staged.plan) == 0 and staged.requested_plan == PLAN
    assert staged.slab_ops == [] and staged.rotation_ops == []
    assert staged.map_ops == [] and staged.staged_layers == []
    assert [(d.swap, d.expert, d.k) for d in staged.dropped] == [
        ((toy.LAYER_ID, 1, 4), 4, 4)]
    assert source.unavailable == 1

    report = engine.apply(staged=staged, quiesce=nullcontext())
    assert report.pairs == 0 and len(report.dropped) == 1
    assert state.tier0_globals == T0_GLOBALS
    assert state.tier1_globals == T1_GLOBALS
    toy.assert_states_equal(state, before)


def test_midswap_disappearance_under_fail_atomic_rolls_back_cleanly(
        segments, tmp_path):
    """``fail_atomic`` reads each destination slot twice (new + pre-swap
    content). A disappearance on the *undo* read must drop the pair just the
    same — a half-filled staging buffer is never referenced by an op list,
    and the next pair reuses (and fully overwrites) it."""
    root, ckpt = segments
    # (7, 4) is only read by the fail-atomic undo pass of the second pair
    resolver = VanishingResolver(tf.resolver(root, tmp_path), gone={(7, 4)})
    source = fq_swap.ResolverFragmentSource(resolver)
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(source, state, max_pairs=2)
    plan = fq_swap.SwapPlan([(toy.LAYER_ID, 7, 6), (toy.LAYER_ID, 1, 4)])

    staged = engine.stage(plan, fail_atomic=True, on_unavailable="drop")
    assert staged.plan.swaps == ((toy.LAYER_ID, 1, 4),)
    assert [d.swap for d in staged.dropped] == [(toy.LAYER_ID, 7, 6)]

    engine.apply(staged=staged, quiesce=nullcontext())
    # only the supplied pair moved; 7 keeps K4, 6 keeps K3 (cardinality D1)
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, [0, 2, 1, 5, 6], [3, 4, 7]))
    assert sorted(state.tier1_globals) == [3, 4, 7]


def test_on_unavailable_default_is_operator_configurable(
        segments, tmp_path, monkeypatch):
    """One env knob makes every staging path pend instead of fail, so a
    serve cannot lose an interval to a missing encode just because some
    call site forgot to pass ``on_unavailable="drop"``."""
    root, ckpt = segments
    resolver = VanishingResolver(tf.resolver(root, tmp_path), gone={(4, 4)})
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(fq_swap.ResolverFragmentSource(resolver), state)

    monkeypatch.delenv("VLLM_FQ_ON_UNAVAILABLE", raising=False)
    with pytest.raises(fq_swap.FragmentUnavailable):
        engine.stage(PLAN)  # fail-closed default preserved

    monkeypatch.setenv("VLLM_FQ_ON_UNAVAILABLE", "drop")
    staged = engine.stage(PLAN)
    assert len(staged.plan) == 0 and len(staged.dropped) == 1
    toy.assert_states_equal(
        state, toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS))

    monkeypatch.setenv("VLLM_FQ_ON_UNAVAILABLE", "sideways")
    with pytest.raises(ValueError, match="raise\\|drop"):
        engine.stage(PLAN)


def test_resolver_crash_during_staging_is_supply_not_structure(
        segments, tmp_path):
    """An OSError deep in the resolver (evicted cache, unreadable segment)
    reaches ``read_expert`` as None through ``resolve_best``, i.e. as a
    droppable supply failure rather than a fatal structural one."""
    root, ckpt = segments
    inner = tf.resolver(root, tmp_path)

    def boom(*_a, **_kw):
        raise OSError(errno.EIO, "I/O error")

    inner._resolve_k = boom  # noqa: SLF001
    state = toy.cpu_layer_state(fq_swap, ckpt, T0_GLOBALS, T1_GLOBALS)
    engine = make_engine(fq_swap.ResolverFragmentSource(inner), state)

    staged = engine.stage(PLAN, on_unavailable="drop")
    assert len(staged.plan) == 0 and len(staged.dropped) == 1
    assert inner.stats["resolve_error"] > 0


# ==================================================== the interval (M2 loop)


def _thrower(exc):
    def apply_fn(_doc, _swaps):
        raise exc
    return apply_fn


def _decision(state):
    return json.loads(
        next((state.store.root / "decisions").glob("*.json")).read_text())


def test_apply_backend_failure_does_not_abort_the_interval(tmp_path):
    """A swap engine that raises (missing fragment, unreachable mirror, an
    aborted stage) must not take the interval down: the incumbent tiering
    stays live, the decision is still explained and persisted, the proposal
    lands in history for the out-of-band path, and the next interval
    retries."""
    import test_loop_cpu as tlc

    state, routers = tlc.make_state(tmp_path, interval=4, apply_mode="reload")
    state.apply_fn = _thrower(
        fr.FragmentUnavailableError("L3/e4 K4: the encode has not landed"))
    tier_before = state.tier_of.copy()
    sha_before = state.policy_sha

    tlc.drive_hot_interval(state, routers)  # must not raise

    record = _decision(state)
    assert record["swaps"], "fixture must actually propose swaps"
    assert record["applied"] is False
    assert record["apply_failures"] == 1 and state.apply_failures == 1
    assert (state.tier_of == tier_before).all()
    assert state.policy_sha == sha_before
    assert list((state.store.root / "history").glob("*-proposed.json"))
    # current.json — the running policy — was never advanced
    current = json.loads((state.store.root / "current.json").read_text())
    assert current["bits_per_expert"]["3"][4] == 3


def test_apply_backend_failure_keeps_the_engine_stepping(tmp_path):
    """The outermost guarantee: ``step()`` never propagates, and repeated
    supply failures just accumulate on the counter."""
    import test_loop_cpu as tlc

    state, routers = tlc.make_state(tmp_path, interval=4, apply_mode="reload")
    state.apply_fn = _thrower(RuntimeError("swap engine on fire"))
    for _ in range(3):
        tlc.drive_hot_interval(state, routers)
    assert state.apply_failures == 3
    assert state._intervals_run == 3  # noqa: SLF001


# ================================================ the lazy-encode wiring


def test_miss_lands_in_the_queue_file_and_drains_to_a_sane_command(tmp_path):
    """End to end on CPU: a resolver miss appends to the persisted queue,
    a second process reads it back, and drain renders the encoder command
    for the layer that is actually missing."""
    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    queue_path = tmp_path / "encode-queue.jsonl"
    r = tf.resolver(d, tmp_path, environ={
        "VLLM_FQ_ENCODE_QUEUE": str(queue_path),
        "VLLM_FQ_K_FALLBACK": "off"})
    assert r.resolve_best(3, 2, 4) is None
    assert queue_path.exists()

    bf16 = tmp_path / "bf16"
    bf16.mkdir()
    (bf16 / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            f"model.layers.3.mlp.experts.2.{proj}.weight": "s.safetensors"
            for proj in ("gate_proj", "up_proj", "down_proj")}}))
    capture = tmp_path / "capture" / "layer_003"
    capture.mkdir(parents=True)
    (capture / "h.pt").write_bytes(b"x")

    lines = []
    rc = le.drain(le.EncodeQueue(queue_path), dry_run=True, bf16_dir=bf16,
                  capture_dir=capture.parent, out=lines.append)
    assert rc == 0
    assert len(lines) == 1
    assert "L3/e2 K4" in lines[0] and "DRY-RUN OK" in lines[0]
    assert "--bits 4" in lines[0] and "--layers 3" in lines[0]
    assert str(bf16) in lines[0] and str(capture.parent) in lines[0]


def test_drain_reports_missing_inputs_instead_of_encoding(tmp_path):
    queue = le.EncodeQueue(tmp_path / "q.jsonl")
    queue.enqueue(7, 1, 4, "unavailable")
    lines = []
    rc = le.drain(queue, dry_run=True, bf16_dir=None, capture_dir=None,
                  out=lines.append)
    assert rc == 1
    assert "BLOCKED" in lines[0] and "bf16-dir unset" in lines[0]


def test_corrupt_queue_file_does_not_crash_the_serve(tmp_path):
    """A queue written by plain appends can be found torn, truncated or
    byte-garbage. Loading must be total: keep what parses, count the rest,
    and let the resolver keep enqueueing."""
    path = tmp_path / "q.jsonl"
    good = json.dumps({"layer": 3, "expert": 2, "k": 4, "reason": "x"})
    path.write_bytes(
        good.encode() + b"\n"
        + b"not json at all\n"
        + b'{"layer": 5, "expert": 1\n'          # torn append, no newline yet
        + b'[1, 2, 3]\n'                          # valid json, wrong shape
        + b'{"layer": "oops", "expert": 1, "k": 4}\n'
        + b"\xff\xfe\x00binary garbage\n"
    )
    queue = le.EncodeQueue(path)
    assert [(e["layer"], e["expert"], e["k"]) for e in queue.entries()] == [
        (3, 2, 4)]
    assert queue.corrupt_lines == 5

    # still usable: appends land, dedup still holds
    assert queue.enqueue(3, 2, 4, "again") == (1, False)
    assert queue.enqueue(4, 0, 5, "unavailable") == (2, True)
    assert len(le.EncodeQueue(path).entries()) == 2


def test_unreadable_queue_degrades_to_empty(tmp_path):
    """The queue path is a directory (or unreadable): a serve must boot."""
    as_dir = tmp_path / "queue-is-a-dir"
    as_dir.mkdir()
    assert le.EncodeQueue(as_dir).entries() == []

    d = tf.make_manifest_dir(tmp_path, layers=(3,), ks=(3,))
    r = tf.resolver(d, tmp_path, environ={
        "VLLM_FQ_ENCODE_QUEUE": str(as_dir), "VLLM_FQ_K_FALLBACK": "off"})
    assert r.resolve_best(3, 0, 4) is None  # enqueue fails; resolve does not


def test_drain_survives_a_bad_encoder_template(tmp_path):
    queue = le.EncodeQueue(tmp_path / "q.jsonl")
    queue.enqueue(3, 2, 4, "unavailable")
    bf16 = tmp_path / "bf16"
    bf16.mkdir()
    (bf16 / "w.safetensors").write_bytes(b"x")
    capture = tmp_path / "capture" / "layer_003"
    capture.mkdir(parents=True)
    (capture / "h.pt").write_bytes(b"x")

    lines = []
    rc = le.drain(queue, dry_run=True, bf16_dir=bf16,
                  capture_dir=capture.parent, out=lines.append,
                  encoder_cmd="encode --bits {k} --who {nonexistent_field}")
    assert rc == 0
    assert "{nonexistent_field}" in lines[0] and "--bits 4" in lines[0]


def test_drain_survives_an_encoder_that_cannot_start(tmp_path):
    queue = le.EncodeQueue(tmp_path / "q.jsonl")
    queue.enqueue(3, 2, 4, "unavailable")
    bf16 = tmp_path / "bf16"
    bf16.mkdir()
    (bf16 / "w.safetensors").write_bytes(b"x")
    capture = tmp_path / "capture" / "layer_003"
    capture.mkdir(parents=True)
    (capture / "h.pt").write_bytes(b"x")

    def missing_binary(_argv):
        raise FileNotFoundError("no such encoder")

    lines = []
    rc = le.drain(queue, dry_run=False, bf16_dir=bf16,
                  capture_dir=capture.parent, out=lines.append,
                  runner=missing_binary)
    assert rc == 1
    assert any("FAILED FileNotFoundError" in m for m in lines)
    assert len(le.EncodeQueue(tmp_path / "q.jsonl").entries()) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([
        __file__, "-v", "-p", "no:cacheprovider",
        "--confcutdir", os.path.dirname(os.path.abspath(__file__)),
    ]))
