# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T5 (CPU half) — torn-update fault injection on the layer STATE.

The GPU half (``test_swap_t5_gpu.py``) aborts the commit protocol at every
step k and compares the kernel's forward output. This half aborts at exactly
the same points and compares a sha256 over *every byte the kernel would
read* — both slabs, all four combined rotation/suh/svh tables, both maps.
That is strictly stronger than comparing one forward (a forward only reads
the rows its routes touch) and it needs no GPU, so the property is defended
in the always-green CPU suite.

Abort points, k = 0..5 (02 §Commit protocol):

    0 quiesce   the engine pause itself fails — nothing must be written
    1 slabs     slab rows written
    2 rotations suh/svh/rotation rows written
    3 maps      map contents flipped   <-- the visibility flip
    4 memo      FusedMoEQuantConfig memo nulled
    5 persist   policy generation bumped + current.json committed

Property: every abort point yields either the fully-old or the fully-new
state — never a third one. Below the flip that requires the engine's
fail-atomic staging (``stage(fail_atomic=True)``), and this file also proves
the *converse*: without it, a pre-flip abort really does leave a torn layer
(so the guard is load-bearing, and the equalities above are not vacuous).
"""
import os
import sys
from contextlib import contextmanager, nullcontext

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_segments as toy  # noqa: E402

fq_swap = toy.load_tree_module("swap")  # never the rootfs' synced copy
fq_store = toy.load_tree_module("store")

E = 8
T0_GLOBALS = [0, 2, 4, 5, 6]  # K3 tier, slot order
T1_GLOBALS = [3, 1, 7]        # K4 tier, slot order
PLAN = ((toy.LAYER_ID, 1, 4),)  # e_out=1 (K4->K3), e_in=4 (K3->K4)
POST_T0, POST_T1 = [0, 2, 1, 5, 6], [3, 4, 7]

# k=0 is the pause failing; k=1..5 are COMMIT_STEPS.
ABORT_POINTS = ("quiesce", *fq_swap.COMMIT_STEPS)
PRE_FLIP = ("quiesce", "slabs", "rotations")
POST_FLIP = ("maps", "memo", "persist")


class Abort(RuntimeError):
    """The injected fault."""


@contextmanager
def failing_quiesce():
    raise Abort("quiesce")
    yield  # pragma: no cover


def abort_after(step):
    def hook(name):
        if name == step:
            raise Abort(name)
    return hook


@pytest.fixture(scope="module")
def toy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("fq-t5-cpu")
    ckpt = toy.make_toy_checkpoint(E)
    toy.write_toy_segments(root, ckpt)
    return root, ckpt


def make_engine(root, state, **kw):
    return fq_swap.SwapEngine(
        {toy.LAYER_ID: state}, fq_swap.LocalSegmentSource(root),
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        pin_memory=False, build_maps_fn=toy.build_maps_reference,
        expected_mcg=toy.MCG, **kw)


def fresh(ckpt, t0=None, t1=None):
    return toy.cpu_layer_state(fq_swap, ckpt, t0 or T0_GLOBALS, t1 or T1_GLOBALS)


def run_aborted(engine, at, *, fail_atomic=True, plan=PLAN, **apply_kw):
    """Stage ``plan``, abort the commit at ``at``, return the staged batch."""
    staged = engine.stage(fq_swap.SwapPlan(plan), fail_atomic=fail_atomic)
    quiesce = failing_quiesce() if at == "quiesce" else nullcontext()
    hook = None if at == "quiesce" else abort_after(at)
    with pytest.raises(Abort):
        engine.apply(staged=staged, quiesce=quiesce, step_hook=hook,
                     **apply_kw)
    return staged


# --------------------------------------------------------------------- T5


def test_t5_no_torn_state_is_observable(toy_root):
    """The property: over all six abort points the layer is only ever
    fully-old or fully-new — exactly two distinct byte states."""
    root, ckpt = toy_root
    pre = toy.state_fingerprint(fresh(ckpt))
    post = toy.state_fingerprint(fresh(ckpt, POST_T0, POST_T1))
    assert pre != post, "the swap is a no-op — the whole test would be vacuous"

    seen = {}
    for at in ABORT_POINTS:
        state = fresh(ckpt)
        engine = make_engine(root, state)
        staged = run_aborted(engine, at)
        seen[at] = toy.state_fingerprint(state)

        if at in PRE_FLIP:
            assert seen[at] == pre, f"abort at {at} left a torn layer"
            assert state.tier1_globals == T1_GLOBALS  # host view agrees
            assert engine.generation == 0
            assert staged.restored is (at != "quiesce")
        else:
            assert seen[at] == post, f"abort at {at} did not commit the swap"
            assert state.tier1_globals == POST_T1  # host view agrees
            assert engine.generation == 1
            assert staged.restored is False

    assert set(seen.values()) == {pre, post}, (
        "an abort point produced a THIRD state: "
        f"{ {k: v[:8] for k, v in seen.items()} }")
    assert [at for at, sha in seen.items() if sha == pre] == list(PRE_FLIP)
    assert [at for at, sha in seen.items() if sha == post] == list(POST_FLIP)


def test_t5_without_fail_atomic_the_torn_state_is_real(toy_root):
    """Converse (non-vacuity): with the same abort and no fail-atomic
    staging, the layer IS torn — a third state, distinct from both. This is
    what the restore buys, and it proves the harness can see tearing."""
    root, ckpt = toy_root
    pre = toy.state_fingerprint(fresh(ckpt))
    post = toy.state_fingerprint(fresh(ckpt, POST_T0, POST_T1))

    torn = {}
    for at in ("slabs", "rotations"):
        state = fresh(ckpt)
        engine = make_engine(root, state)
        staged = run_aborted(engine, at, fail_atomic=False)
        assert staged.undo_ops is None and staged.restored is False
        torn[at] = toy.state_fingerprint(state)
        assert torn[at] not in (pre, post), (
            f"abort at {at} without fail-atomic was harmless — the T5 "
            "restore assertions would be vacuous")

        # Roll forward: re-applying the same plan is the other legal repair.
        engine.apply(fq_swap.SwapPlan(PLAN), quiesce=nullcontext())
        assert toy.state_fingerprint(state) == post
    assert torn["slabs"] != torn["rotations"]  # tearing grows with each step


def test_t5_abort_then_inverse_plan_restores_pre_swap(toy_root):
    """The 02 §Rollback path after a *committed* abort: the flip landed, so
    the swap is real and the repair is the inverse plan (fragments re-read
    from the artifact pair) — pre-swap bytes come back exactly."""
    root, ckpt = toy_root
    pre = toy.state_fingerprint(fresh(ckpt))
    state = fresh(ckpt)
    engine = make_engine(root, state)

    run_aborted(engine, "memo")  # committed, but the memo hook blew up
    assert state.tier1_globals == POST_T1
    engine.apply(fq_swap.SwapPlan(PLAN).inverse(), quiesce=nullcontext())
    assert toy.state_fingerprint(state) == pre
    assert state.tier1_globals == T1_GLOBALS
    assert engine.generation == 2


def test_t5_restored_engine_is_still_usable(toy_root):
    """A pre-flip abort must not poison the engine: the very next apply of
    the same plan must land the full, correct swap."""
    root, ckpt = toy_root
    post = toy.state_fingerprint(fresh(ckpt, POST_T0, POST_T1))
    state = fresh(ckpt)
    engine = make_engine(root, state)

    run_aborted(engine, "rotations")
    report = engine.apply(fq_swap.SwapPlan(PLAN), quiesce=nullcontext())
    assert report.generation == 1 and report.pairs == 1
    assert toy.state_fingerprint(state) == post
    toy.assert_states_equal(state, fresh(ckpt, POST_T0, POST_T1))


def test_t5_multi_pair_abort_restores_every_pair(toy_root):
    """Two pairs in one layer: a mid-commit abort must restore BOTH, not
    just the one that was mid-write."""
    root, ckpt = toy_root
    pre = toy.state_fingerprint(fresh(ckpt))
    state = fresh(ckpt)
    engine = make_engine(root, state, max_pairs=2)
    plan = ((toy.LAYER_ID, 3, 0), (toy.LAYER_ID, 7, 6))

    run_aborted(engine, "rotations", plan=plan)
    assert toy.state_fingerprint(state) == pre
    assert state.tier0_globals == T0_GLOBALS
    assert state.tier1_globals == T1_GLOBALS

    engine.apply(fq_swap.SwapPlan(plan), quiesce=nullcontext())
    toy.assert_states_equal(state, fresh(ckpt, [3, 2, 4, 5, 7], [0, 1, 6]))


def test_t5_persist_failure_leaves_the_swap_committed(toy_root, tmp_path):
    """The realistic step-5 fault (disk full / read-only cache): the flip has
    landed, so the swap IS committed and the host view must say so; only the
    persisted policy is missing, which boot recovers by rehydrating the
    previous committed policy (T8)."""
    root, ckpt = toy_root
    post = toy.state_fingerprint(fresh(ckpt, POST_T0, POST_T1))
    state = fresh(ckpt)
    engine = make_engine(root, state)

    class ExplodingStore(fq_store.PolicyStore):
        def commit(self, doc, *, num_experts=None):
            raise OSError("No space left on device")

    store = ExplodingStore(tmp_path, "toy-manifest")
    doc = {
        "schema": "fq-policy/2",
        "manifest": "toy-manifest",
        "budget": {"n_k4_per_layer": {str(toy.LAYER_ID): 3}},
        "bits_per_expert": {str(toy.LAYER_ID): [3, 3, 3, 4, 4, 3, 3, 4]},
    }
    staged = engine.stage(fq_swap.SwapPlan(PLAN), fail_atomic=True)
    with pytest.raises(OSError, match="No space"):
        engine.apply(staged=staged, quiesce=nullcontext(), policy_store=store,
                     policy_doc=doc, policy_num_experts=E)

    assert toy.state_fingerprint(state) == post
    assert state.tier1_globals == POST_T1 and engine.generation == 1
    assert staged.restored is False
    assert store.load_current(num_experts=E) is None  # nothing torn on disk


def test_t5_fail_atomic_staging_costs_only_reads(toy_root):
    """Fail-atomic staging must not change what the commit writes: the same
    op lists, the same H2D byte count — only extra host-side reads."""
    root, ckpt = toy_root
    plain = make_engine(root, fresh(ckpt)).stage(fq_swap.SwapPlan(PLAN))
    atomic = make_engine(root, fresh(ckpt)).stage(fq_swap.SwapPlan(PLAN),
                                                  fail_atomic=True)
    assert plain.undo_ops is None and atomic.fail_atomic
    assert plain.bytes_h2d == atomic.bytes_h2d
    assert len(plain.slab_ops) == len(atomic.slab_ops) == 6
    assert len(plain.rotation_ops) == len(atomic.rotation_ops) == 8
    # The restore batch covers every destination the commit touches.
    assert len(atomic.undo_ops) == 6 + 8 + 2
    written = {t.data_ptr() for t, _ in atomic.slab_ops + atomic.rotation_ops
               + atomic.map_ops}
    assert {t.data_ptr() for t, _ in atomic.undo_ops} == written


if __name__ == "__main__":
    sys.exit(pytest.main([
        __file__, "-v", "-p", "no:cacheprovider",
        "--confcutdir", os.path.dirname(os.path.abspath(__file__)),
    ]))
