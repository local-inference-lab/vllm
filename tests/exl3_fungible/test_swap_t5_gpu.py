# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T5 — torn-update fault injection (03-testing-validation.md, GPU).

Deliberately abort the commit protocol after step k, for every k, and ask the
kernel what it sees. The property (02 §Commit protocol):

    every intermediate visible state is either fully-old or fully-new —
    no abort point may produce a third distinct output.

Abort points, k = 0..5:

    0 quiesce   the engine pause itself fails — nothing is written
    1 slabs     slab rows (w13 gate+up, w2) written
    2 rotations combined suh/svh/intermediate-rotation rows written
    3 maps      map contents flipped        <-- the visibility flip
    4 memo      FusedMoEQuantConfig memo nulled
    5 persist   generation bumped + policy committed

k < 3 must forward bitwise-equal to the PRE-swap output; k >= 3 bitwise-equal
to the POST-swap output (the fresh-built layer with the new membership, which
T4 pinned the successful apply to).

What makes k in {1, 2} true is not ordering alone: in the v1 row-write design
the two destination rows are both live, so writing them *is* the tear. The
engine's contract is that the write window is quiesced AND that an abort
inside it restores the pre-swap rows before the window is released
(``stage(fail_atomic=True)``). This file proves both halves — the restore
holds the property, and removing it produces a genuine third output, so the
equalities are not vacuous. It also replays a CUDA graph across an aborted
swap: a restore that the graph could not see would be worse than useless.

Run in the gg r33 env from a neutral cwd, on ONE free GPU::

    CUDA_VISIBLE_DEVICES=<free> .../runs/gg-env/gg-run.sh \
        python /home/mbelleau/src/gg-vllm/tests/exl3_fungible/test_swap_t5_gpu.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from contextlib import contextmanager, nullcontext

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_segments as toy  # noqa: E402

fq_swap = toy.load_tree_module("swap")  # never the rootfs' synced copy

try:
    from b12x.moe._shared.kernels.w4a16.host import max_packed_route_slots
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import (
        build_tiered_maps,
        combine_trellis_rotations,
        compile_mixed_trellis,
        make_mixed_trellis_buffers,
        run_mixed_trellis,
    )
    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_trellis256_moe_weights,
    )
    HAVE_B12X = True
except ImportError:
    HAVE_B12X = False


def _sm12x_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


pytestmark = pytest.mark.skipif(
    not (HAVE_B12X and _sm12x_available()),
    reason="requires b12x and an SM120/SM121 GPU")

# ------------------------------------------------------------------ geometry
# Same toy layer as T3/T4: E=16 = 12 K3 + 4 K4.
M, TOPK, MOE_BLOCK = 8, 4, 8
TIER0_GLOBALS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
TIER1_GLOBALS = [3, 7, 11, 15]
ROUTES16 = (
    (0, 3, 14, 5),
    (1, 11, 4, 8),
    (10, 12, 3, 14),
    (5, 0, 11, 1),
    (15, 8, 12, 10),
    (4, 7, 13, 0),
    (11, 14, 5, 12),
    (9, 10, 2, 6),
)

E_OUT, E_IN = 7, 6  # K4 slot 1 out, K3 slot 5 in — middle rows of both tiers
PLAN = ((toy.LAYER_ID, E_OUT, E_IN),)
POST_T0 = [0, 1, 2, 4, 5, 7, 8, 9, 10, 12, 13, 14]
POST_T1 = [3, 6, 11, 15]

ABORT_POINTS = ("quiesce", *fq_swap.COMMIT_STEPS)
PRE_FLIP = ("quiesce", "slabs", "rotations")
POST_FLIP = ("maps", "memo", "persist")

_NOTES: list[str] = []


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


# ------------------------------------------------------------------- harness


def _device():
    return torch.device("cuda", torch.cuda.current_device())


def _build_state(ckpt, t0_globals, t1_globals, device):
    """Fresh-built mixed layer from checkpoint tensors (exl3.py's assembly)."""
    prepared = []
    for globals_, bits in ((t0_globals, 3), (t1_globals, 4)):
        t = {k: v.to(device) for k, v in toy.assemble_membership_tensors(
            ckpt, globals_, bits).items()}
        prepared.append(prepare_trellis256_moe_weights(
            w13=t["w13"].contiguous(),
            w2=t["w2"].contiguous(),
            hidden_size=toy.HIDDEN,
            intermediate_size=toy.INTERMEDIATE,
            num_experts=len(globals_),
            activation="silu",
            fc1_tile_n=toy.TILE_CONFIG[1],
            fc2_tile_n=toy.TILE_CONFIG[3],
            params_dtype=torch.float16,
            w13_layout="trellis3_t256_proj",
            trellis_bits=bits,
            gate_suh=t["gate_suh"].contiguous(),
            up_suh=t["up_suh"].contiguous(),
            intermediate_rotations=t["intermediate"].contiguous(),
            down_svh=t["down_svh"].contiguous(),
            tile_config=toy.TILE_CONFIG,
        ))
    g2c, desc = build_tiered_maps(t0_globals, t1_globals, device=device)
    return fq_swap.MixedLayerState(
        tier0=prepared[0], tier1=prepared[1],
        rotations=combine_trellis_rotations(*prepared),
        global_to_combined=g2c, descriptor_map=desc,
        tier0_globals=list(t0_globals), tier1_globals=list(t1_globals),
        mcg=toy.MCG)


def _compile(device, cap0, cap1):
    props = torch.cuda.get_device_properties(device)
    route_slots = max_packed_route_slots(M * TOPK, MOE_BLOCK, cap0 + cap1)
    return compile_mixed_trellis(
        size_m=M,
        hidden_size=toy.HIDDEN,
        intermediate_size=toy.INTERMEDIATE,
        tier0_num_experts=cap0,
        tier1_num_experts=cap1,
        top_k=TOPK,
        max_m_blocks=(route_slots + MOE_BLOCK - 1) // MOE_BLOCK,
        sms=int(props.multi_processor_count),
        max_shared_mem=int(props.shared_memory_per_block_optin),
        force_tile_config=toy.TILE_CONFIG,
        moe_block_size=MOE_BLOCK,
        route_ids_dtype=torch.int32,
    ), int(props.multi_processor_count)


def _forward(state, launch, sms, probe, *, buffers=None):
    x, topk_weights, topk_ids = probe
    if buffers is None:
        buffers = make_mixed_trellis_buffers(launch, device=x.device, sms=sms)
    return run_mixed_trellis(
        x, state.tier0, state.tier1, topk_weights, topk_ids,
        state.global_to_combined, state.descriptor_map, state.rotations,
        launch, buffers)


def _out(state, launch, sms, probe):
    out = _forward(state, launch, sms, probe)
    torch.cuda.synchronize(probe[0].device)
    return out.clone()


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _probe(device):
    torch.manual_seed(20260810)
    x = (torch.randn((M, toy.HIDDEN), device=device) * 1.0e-3).to(torch.bfloat16)
    topk_ids = torch.tensor(ROUTES16, dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn((M, TOPK), dtype=torch.float32, device=device), dim=-1)
    return x, topk_weights, topk_ids


@contextmanager
def _quiesce(device):
    """Test stand-in for the engine pause: nothing in flight; sync on exit."""
    yield
    torch.cuda.synchronize(device)


@pytest.fixture(scope="module")
def toy_gpu(tmp_path_factory):
    root = tmp_path_factory.mktemp("fq-toy-segments-t5")
    ckpt = toy.make_toy_checkpoint(16)
    toy.write_toy_segments(root, ckpt)
    device = _device()
    launch, sms = _compile(device, len(TIER0_GLOBALS), len(TIER1_GLOBALS))
    probe = _probe(device)
    pre = _out(_build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device),
               launch, sms, probe)
    post = _out(_build_state(ckpt, POST_T0, POST_T1, device),
                launch, sms, probe)
    assert not torch.equal(pre, post), (
        "the swapped pair does not change the probe output — T5 would be "
        "vacuous; pick a routed expert pair")
    assert not torch.isnan(pre).any() and not torch.isnan(post).any()
    return root, ckpt, launch, sms, probe, pre, post


def _make_engine(root, state, **kw):
    return fq_swap.SwapEngine(
        {toy.LAYER_ID: state}, fq_swap.LocalSegmentSource(root),
        hidden_size=toy.HIDDEN, intermediate_size=toy.INTERMEDIATE,
        expected_mcg=toy.MCG, **kw)


def _run_aborted(engine, at, device, *, fail_atomic=True, plan=PLAN):
    staged = engine.stage(fq_swap.SwapPlan(plan), fail_atomic=fail_atomic)
    quiesce = failing_quiesce() if at == "quiesce" else _quiesce(device)
    hook = None if at == "quiesce" else abort_after(at)
    with pytest.raises(Abort):
        engine.apply(staged=staged, quiesce=quiesce, step_hook=hook)
    torch.cuda.synchronize(device)
    return staged


# ------------------------------------------------------------------------ T5


def test_t5_no_torn_state_is_observable(toy_gpu):
    """Abort after every step k; the forward must be PRE below the flip and
    POST at or above it, and those must be the only two outputs."""
    root, ckpt, launch, sms, probe, pre, post = toy_gpu
    device = _device()
    seen: dict[str, str] = {}

    for at in ABORT_POINTS:
        state = _build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device)
        engine = _make_engine(root, state)
        staged = _run_aborted(engine, at, device)
        out = _out(state, launch, sms, probe)
        seen[at] = _sha(out)

        if at in PRE_FLIP:
            assert torch.equal(out, pre), (
                f"T5 FAIL: abort at {at} left an observable torn state "
                "(forward differs from the pre-swap output)")
            assert state.tier1_globals == TIER1_GLOBALS
            assert engine.generation == 0
            assert staged.restored is (at != "quiesce")
        else:
            assert torch.equal(out, post), (
                f"T5 FAIL: abort at {at} is at/after the visibility flip but "
                "the forward is not the post-swap output")
            assert state.tier1_globals == POST_T1
            assert engine.generation == 1

    assert set(seen.values()) == {_sha(pre), _sha(post)}, (
        "an abort point produced a THIRD distinct output: "
        f"{ {k: v[:8] for k, v in seen.items()} }")
    _NOTES.append(
        "abort-point outputs: " + ", ".join(
            f"{at}={'PRE' if sha == _sha(pre) else 'POST'}"
            for at, sha in seen.items()))


def test_t5_without_fail_atomic_the_torn_state_is_real(toy_gpu):
    """Non-vacuity + why the restore is load-bearing: the same abort without
    fail-atomic staging yields a THIRD output (the displaced expert computing
    with its successor's rows). Rolling the plan forward repairs it."""
    root, ckpt, launch, sms, probe, pre, post = toy_gpu
    device = _device()
    thirds = {}
    for at in ("slabs", "rotations"):
        state = _build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device)
        engine = _make_engine(root, state)
        staged = _run_aborted(engine, at, device, fail_atomic=False)
        assert staged.undo_ops is None and staged.restored is False
        torn = _out(state, launch, sms, probe)
        assert not torch.equal(torn, pre) and not torch.equal(torn, post), (
            f"abort at {at} without fail-atomic was harmless — the T5 "
            "restore assertions would be vacuous")
        thirds[at] = _sha(torn)

        engine.apply(fq_swap.SwapPlan(PLAN), quiesce=_quiesce(device))
        assert torch.equal(_out(state, launch, sms, probe), post), (
            "rolling the plan forward did not repair the torn layer")
    _NOTES.append(f"torn-state outputs (no fail-atomic): "
                  f"{ {k: v[:8] for k, v in thirds.items()} }")


def test_t5_abort_then_inverse_plan_restores_pre_swap(toy_gpu):
    """02 §Rollback after a committed abort (the flip landed): the inverse
    plan, fragments re-read from the artifact pair, restores the pre-swap
    output bitwise."""
    root, ckpt, launch, sms, probe, pre, post = toy_gpu
    device = _device()
    state = _build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device)
    engine = _make_engine(root, state)

    _run_aborted(engine, "memo", device)
    assert torch.equal(_out(state, launch, sms, probe), post)
    engine.apply(fq_swap.SwapPlan(PLAN).inverse(), quiesce=_quiesce(device))
    assert torch.equal(_out(state, launch, sms, probe), pre), (
        "inverse plan after an aborted swap did not restore the pre-swap "
        "output bitwise")
    assert state.tier1_globals == TIER1_GLOBALS


def test_t5_restore_is_visible_to_a_captured_cuda_graph(toy_gpu):
    """The restore writes into the live slabs/tables/maps, so a CUDA graph
    captured before the aborted swap must replay the PRE-swap output — the
    same read-as-data property T3 proved for the flip itself."""
    root, ckpt, launch, sms, probe, pre, post = toy_gpu
    device = _device()
    state = _build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device)
    engine = _make_engine(root, state)

    buffers = make_mixed_trellis_buffers(launch, device=device, sms=sms)
    _forward(state, launch, sms, probe, buffers=buffers)  # warmup
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _forward(state, launch, sms, probe, buffers=buffers)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, pre), "capture drifted from eager"

    _run_aborted(engine, "rotations", device)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, pre), (
        "T5 FAIL: a graph replay after an aborted-and-restored swap saw a "
        "torn layer")

    engine.apply(fq_swap.SwapPlan(PLAN), quiesce=_quiesce(device))
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(captured, post), (
        "graph replay did not see the completed swap after the restore")


def test_t5_multi_pair_abort_restores_every_pair(toy_gpu):
    """Two pairs, one layer: the restore must cover both, and the completed
    2-pair swap must still match the fresh-built reference bitwise."""
    root, ckpt, launch, sms, probe, pre, _post = toy_gpu
    device = _device()
    state = _build_state(ckpt, TIER0_GLOBALS, TIER1_GLOBALS, device)
    engine = _make_engine(root, state, max_pairs=2)
    plan = ((toy.LAYER_ID, 7, 6), (toy.LAYER_ID, 15, 0))

    _run_aborted(engine, "rotations", device, plan=plan)
    assert torch.equal(_out(state, launch, sms, probe), pre)
    assert state.tier0_globals == TIER0_GLOBALS
    assert state.tier1_globals == TIER1_GLOBALS

    engine.apply(fq_swap.SwapPlan(plan), quiesce=_quiesce(device))
    reference = _build_state(ckpt, state.tier0_globals, state.tier1_globals,
                             device)
    assert torch.equal(_out(state, launch, sms, probe),
                       _out(reference, launch, sms, probe))


if __name__ == "__main__":
    code = pytest.main([
        __file__, "-v", "-s", "-p", "no:cacheprovider",
        "--confcutdir", os.path.dirname(os.path.abspath(__file__)),
    ])
    for line in _NOTES:
        print(f"[t5] {line}")
    sys.exit(int(code))
