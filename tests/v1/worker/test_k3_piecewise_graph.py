# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sequencing and storage rules of the Kimi-K3 piecewise prefill graphs.

The device-side capture needs a GPU; everything that decides *what* is
captured does not. These tests drive the session through a backend that
records begin/end calls instead of capturing, so the piece plan, the
boundary ordering, the static-buffer keying and the address guard are
checked without a device.
"""

import pytest
import torch

from vllm.v1.worker.gpu import k3_piecewise_graph as pg


class FakeGraph:
    def __init__(self, index: int) -> None:
        self.index = index
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


class FakeBackend:
    """Records the capture calls a real backend would make."""

    def __init__(self) -> None:
        self.pools: dict[int, str] = {}
        self.open: FakeGraph | None = None
        self.graphs: list[FakeGraph] = []
        self.events: list[str] = []

    def new_pool(self, half: int) -> str:
        pool = self.pools.setdefault(half, f"pool{half}")
        return pool

    def begin(self, pool):
        assert self.open is None, "a capture is already open"
        graph = FakeGraph(len(self.graphs))
        self.graphs.append(graph)
        self.open = graph
        self.events.append(f"begin:{pool}")
        return graph

    def end(self, graph) -> None:
        assert self.open is graph
        self.open = None
        self.events.append("end")


# The Kimi-K3 layer at nine ranks: hidden 7,168; router 104 local experts of
# 936 gathered; routed latent 400 local of 3,584 padded (the gather returns
# 3,600 before the padding is trimmed).
def layer_boundaries(rows: int):
    return pg.kimi_k3_layer_boundaries(
        rows=rows,
        hidden_size=7168,
        router_experts=104,
        latent_width=400,
        latent_width_padded=3584,
    )


def layer_buffers(rows: int):
    return pg.kimi_k3_boundary_buffers(
        rows=rows,
        hidden_size=7168,
        router_experts=104,
        router_experts_gathered=936,
        latent_width=400,
        latent_width_gathered=3600,
        latent_width_padded=3584,
    )


def run_layers(session: pg.PiecewiseSession, kinds):
    """Drive one forward: every layer reaches A1..A5 in order."""
    for layer_idx, kind in enumerate(kinds):
        with pg.layer_region(layer_idx, kind == pg.KDA_LAYER):
            for role in pg.BOUNDARY_ROLES:
                with pg.collective_boundary(role):
                    pg.record_collective(role, lambda: None)


@pytest.fixture(autouse=True)
def _clean():
    pg.reset()
    yield
    pg.reset()


# -- plan ------------------------------------------------------------------


def test_a_captured_layer_is_one_piece_per_collective():
    backend = FakeBackend()
    session = pg.PiecewiseSession(0, 2304, backend)
    with pg.half_region(session):
        run_layers(session, [pg.KDA_LAYER] * 3)
    # Three layers of five boundaries: a piece before each collective plus
    # the piece that closes the forward.
    assert len(session.plan.pieces) == 3 * 5 + 1
    assert session.plan.signature() == pg.BOUNDARY_ROLES * 3
    assert backend.open is None
    assert backend.events.count("begin:pool0") == len(session.plan.pieces)


def test_pieces_and_collectives_alternate():
    backend = FakeBackend()
    session = pg.PiecewiseSession(1, 2304, backend)
    with pg.half_region(session):
        run_layers(session, [pg.KDA_LAYER])
    kinds = [type(step).__name__ for step in session.plan.steps]
    assert kinds == [
        "RecordedPiece",
        "RecordedCollective",
    ] * 5 + ["RecordedPiece"]


def test_each_piece_names_the_collectives_it_sits_between():
    backend = FakeBackend()
    session = pg.PiecewiseSession(0, 2304, backend)
    with pg.half_region(session):
        run_layers(session, [pg.KDA_LAYER])
    pieces = session.plan.pieces
    assert [p.key.before for p in pieces] == [*pg.BOUNDARY_ROLES, None]
    assert [p.key.after for p in pieces] == [None, *pg.BOUNDARY_ROLES]
    assert [p.key.index for p in pieces] == list(range(len(pieces)))
    assert all(p.key.half == 0 and p.key.rows == 2304 for p in pieces)


def test_a_latent_attention_layer_runs_inside_no_piece():
    # Its chunked context pass has a context-dependent number of key gathers,
    # so it is not captured; the surrounding pieces close around it.
    backend = FakeBackend()
    session = pg.PiecewiseSession(0, 2304, backend)
    kinds = [pg.KDA_LAYER, pg.MLA_LAYER, pg.KDA_LAYER]
    with pg.half_region(session):
        run_layers(session, kinds)
    assert session.plan.signature() == pg.BOUNDARY_ROLES * 2
    assert [p.key.layer for p in session.plan.pieces] == [0] * 5 + [1] + [2] * 6
    assert backend.open is None
    specs = [pg.LayerSpec(kind, layer_boundaries(2304)) for kind in kinds]
    assert pg.expected_pieces(specs) == len(session.plan.pieces)


def test_the_two_halves_capture_into_separate_pools():
    backend = FakeBackend()
    for half in (0, 1):
        session = pg.PiecewiseSession(half, 2304, backend)
        with pg.half_region(session):
            run_layers(session, [pg.KDA_LAYER])
    assert backend.pools == {0: "pool0", 1: "pool1"}


def test_a_forward_that_misses_a_boundary_is_rejected():
    backend = FakeBackend()
    session = pg.PiecewiseSession(
        0, 2304, backend, expected_signature=pg.BOUNDARY_ROLES
    )
    with pytest.raises(pg.PlanMismatch):
        with pg.half_region(session):
            with pg.collective_boundary("A1"):
                pg.record_collective("A1", lambda: None)
    assert pg.disabled_reason() is not None
    assert backend.open is None


def test_a_failed_forward_leaves_no_open_capture_and_no_plan():
    backend = FakeBackend()
    session = pg.PiecewiseSession(0, 2304, backend)
    with pytest.raises(ZeroDivisionError):
        with pg.half_region(session):
            with pg.collective_boundary("A1"):
                pg.record_collective("A1", lambda: None)
            1 / 0
    assert backend.open is None
    assert session.plan.steps == []


# -- replay ----------------------------------------------------------------


def test_a_recorded_half_replays_instead_of_running_the_model():
    backend = FakeBackend()
    session = pg.PiecewiseSession(0, 2304, backend)
    with pg.half_region(session) as run:
        assert run is not None
        run_layers(session, [pg.KDA_LAYER] * 2)
    captured = len(backend.graphs)
    # A capture executes nothing, so the recording step replays once itself.
    assert all(g.replays == 1 for g in backend.graphs)
    collectives = []
    for step in session.plan.steps:
        if isinstance(step, pg.RecordedCollective):
            step.run_fn = lambda role=step.role: collectives.append(role)
    with pg.half_region(session) as run:
        assert run is None
    assert len(backend.graphs) == captured, "a replay captures nothing"
    assert all(g.replays == 2 for g in backend.graphs)
    assert tuple(collectives) == pg.BOUNDARY_ROLES * 2


def test_a_collective_result_gets_an_address_the_next_piece_can_read():
    backend = FakeBackend()
    static = pg.StaticBuffers(torch.device("cpu"))
    session = pg.PiecewiseSession(0, 4, backend, static=static)
    produced = torch.zeros(4, 8)
    gathered = torch.ones(4, 72)

    with pg.half_region(session):
        # In-place: the result is the input, whose address the preceding
        # piece fixed.
        same = pg.run_collective("ar", produced, lambda: produced)
        assert same is produced
        # Out-of-place: the result is copied into static storage, and that
        # buffer is what the model carries on.
        out = pg.run_collective("ag", torch.zeros(4, 8), lambda: gathered)
        assert out.data_ptr() != gathered.data_ptr()
        assert torch.equal(out, gathered)
    assert session.plan.signature() == ("ar[8]", "ag[8]")
    # Replaying the gather refreshes the same address.
    address = out.data_ptr()
    gathered.fill_(3)
    session.plan.collectives[-1].run()
    assert out.data_ptr() == address
    assert torch.equal(out, gathered)


def test_a_collective_outside_a_recording_is_just_the_collective():
    tensor = torch.zeros(2, 3)
    assert pg.run_collective("ar", tensor, lambda: tensor) is tensor


def test_each_thread_records_its_own_half():
    import threading

    backend = pg.PiecewiseSession(0, 4, FakeBackend())
    seen: dict[int, object] = {}

    def worker(half):
        session = pg.PiecewiseSession(half, 4, FakeBackend())
        with pg.half_region(session):
            seen[half] = pg.active_session()
            run_layers(session, [pg.KDA_LAYER])

    threads = [threading.Thread(target=worker, args=(h,)) for h in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert {half: s.half for half, s in seen.items()} == {0: 0, 1: 1}
    assert pg.active_session() is None
    del backend


# -- interleaving ----------------------------------------------------------


def test_the_halves_interleave_at_their_boundaries():
    plans = []
    for half in (0, 1):
        backend = FakeBackend()
        session = pg.PiecewiseSession(half, 2304, backend)
        with pg.half_region(session):
            run_layers(session, [pg.KDA_LAYER])
        plans.append(session.plan)
    order = pg.merged_steps(*plans)
    assert len(order) == len(plans[0].steps) + len(plans[1].steps)
    # A half's piece is followed by its own collective, then the other half.
    halves = [half for half, _ in order]
    assert halves[:8] == [0, 0, 1, 1, 0, 0, 1, 1]
    # Each half's steps keep their recorded order.
    for half, plan in enumerate(plans):
        assert [s for h, s in order if h == half] == plan.steps


def test_interleaving_a_shorter_half_drains_the_longer_one():
    backend = FakeBackend()
    short = pg.PiecewiseSession(0, 1152, backend)
    with pg.half_region(short):
        run_layers(short, [pg.KDA_LAYER])
    long = pg.PiecewiseSession(1, 2304, backend)
    with pg.half_region(long):
        run_layers(long, [pg.KDA_LAYER] * 3)
    order = pg.merged_steps(short.plan, long.plan)
    assert len(order) == len(short.plan.steps) + len(long.plan.steps)
    assert [s for h, s in order if h == 1] == long.plan.steps


def test_boundaries_are_inert_outside_a_recording():
    # The served path (flag off) reaches the same hook.
    calls = []
    with pg.collective_boundary("A1"):
        calls.append("collective ran")
    pg.record_collective("A1", lambda: None)
    assert calls == ["collective ran"]
    assert pg.active_session() is None


# -- static storage --------------------------------------------------------


def test_static_buffers_are_keyed_by_half_role_shape_and_dtype():
    static = pg.StaticBuffers(torch.device("cpu"))
    a1_half0 = static.get(0, "A1", (8, 4), torch.bfloat16)
    a1_half1 = static.get(1, "A1", (8, 4), torch.bfloat16)
    a5_half0 = static.get(0, "A5", (8, 4), torch.bfloat16)
    again = static.get(0, "A1", (8, 4), torch.bfloat16)
    assert again is a1_half0
    assert a1_half0 is not a1_half1, "the halves must not share a buffer"
    assert a1_half0 is not a5_half0, "one role must not alias another"
    assert len(static) == 3
    assert static.total_bytes == 3 * 8 * 4 * 2


def test_binding_copies_into_the_static_buffer_and_returns_it():
    static = pg.StaticBuffers(torch.device("cpu"))
    value = torch.arange(6, dtype=torch.bfloat16).view(2, 3)
    bound = static.bind(0, "positions", value)
    assert bound.data_ptr() != value.data_ptr()
    assert torch.equal(bound, value)
    value.fill_(7)
    again = static.bind(0, "positions", value)
    assert again.data_ptr() == bound.data_ptr(), "one address across chunks"
    assert torch.equal(again, value)


def test_a_bound_input_keeps_one_address_across_replays():
    backend = FakeBackend()
    static = pg.StaticBuffers(torch.device("cpu"))
    session = pg.PiecewiseSession(0, 4, backend, static=static)
    first = torch.zeros(4, dtype=torch.int32)
    with pg.half_region(session, {"positions": first}) as run:
        assert run is not None
        address = run["positions"].data_ptr()
        run_layers(session, [pg.KDA_LAYER])
    second = torch.ones(4, dtype=torch.int32)
    with pg.half_region(session, {"positions": second}):
        pass
    assert static.get(0, "positions", (4,), torch.int32).data_ptr() == address
    assert torch.equal(static.get(0, "positions", (4,), torch.int32), second)


def test_an_input_that_moved_between_capture_and_replay_is_rejected():
    backend = FakeBackend()
    session = pg.PiecewiseSession(0, 4, backend)  # no static storage
    with pg.half_region(session, {"positions": torch.zeros(4)}):
        run_layers(session, [pg.KDA_LAYER])
    with pytest.raises(pg.PlanMismatch):
        with pg.half_region(session, {"positions": torch.zeros(4)}):
            pass
    assert pg.disabled_reason() is not None


def test_the_guard_rejects_a_replay_that_drops_an_input():
    guard = pg.PointerGuard()
    tensor = torch.zeros(4)
    guard.snapshot("positions", tensor)
    guard.verify("positions", tensor)
    assert guard.names == ("positions",)
    with pytest.raises(pg.PlanMismatch):
        guard.verify_all({})
    with pytest.raises(pg.PlanMismatch):
        guard.verify("slot_mapping", tensor)


def test_every_per_chunk_tensor_a_replay_reads_is_named():
    names = [name for name, _ in pg.REPLAY_STATIC_REQUIREMENTS]
    assert len(set(names)) == len(names)
    assert {"positions", "slot_mapping", "block_table"} <= set(names)
    assert all(site for _, site in pg.REPLAY_STATIC_REQUIREMENTS)


# -- budget ----------------------------------------------------------------


def test_boundary_budget_of_the_served_shapes():
    whole = layer_buffers(4608)
    halves = layer_buffers(2304)
    # A half's boundary tensors are exactly half the whole chunk's, so two
    # halves of boundary storage cost what one unsplit chunk's would: the
    # split does not enlarge the boundary set, it only divides it in two.
    assert sum(b.nbytes() for b in halves) * 2 == sum(b.nbytes() for b in whole)
    assert pg.boundary_buffer_bytes(halves, halves=2, share_equal_shapes=False) == sum(
        b.nbytes() for b in whole
    )
    # Sharing equal shapes folds the attention-output and layer-output
    # all-reduces of a layer onto one buffer.
    shared = pg.boundary_buffer_bytes(halves, halves=2)
    assert shared < pg.boundary_buffer_bytes(halves, halves=2, share_equal_shapes=False)
    assert shared == 2 * (
        2304 * 7168 * 2  # A1 and A5 share
        + 2304 * 104 * 4  # router logits sent
        + 2304 * 936 * 4  # router logits gathered
        + 2304 * 400 * 2  # routed latent sent
        + 2304 * 3584 * 2  # routed latent all-reduced
        + 2304 * 3600 * 2  # routed latent gathered
    )
    assert shared == 155_123_712  # 147.9 MiB for both halves


def test_piece_count_of_the_served_layer_mix():
    # 69 linear-attention layers are captured, the 24 latent-attention
    # layers are not; the model's layer order does not change the count.
    layers = [
        pg.LayerSpec(pg.KDA_LAYER, layer_boundaries(2304)) for _ in range(69)
    ] + [pg.LayerSpec(pg.MLA_LAYER, layer_boundaries(2304)) for _ in range(24)]
    assert pg.expected_pieces(layers) == 69 * 5 + 24 + 1
    assert sum(1 for spec in layers if spec.capturable) == 69
