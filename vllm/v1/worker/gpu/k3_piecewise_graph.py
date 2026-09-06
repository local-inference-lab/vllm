# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a Kimi-K3 prefill half as CUDA graphs cut at its collectives.

A Kimi-K3 prefill layer issues about a hundred CUDA runtime calls and spends
most of its host frame in Python, Triton and CuTe DSL preparation rather than
in the driver: on the nine-rank deployment a layer whose device queue is empty
takes 7.3-8.2 ms of host time of which 0.8-1.0 ms is inside runtime calls
(research/prefill-campaign-20260906/evidence/r1/host_launch_stats.log). One
chunk of 4,608 rows costs about 12-13.5 ms of link-bound device time per
layer, so a single stream of layers is issue-limited only in bursts; running
two 2,304-row halves so that one half's collectives overlap the other half's
compute doubles the issue rate and makes the host the limit. Capturing the
compute between the collectives removes that limit: the captured pieces
replay with one launch each.

Structure. A layer's device work is cut at its collectives, which cannot go
inside a piece: the B12X ring all-reduce is itself a replayed CUDA graph and a
graph cannot launch a graph, and a captured collective would hold the compute
stream for the whole transfer, which is exactly the time the other half is
meant to use. The cut points of one layer are

    A1  attention output all-reduce   (kda.py, mla.py -> ring)
    A2  router gate logits all-gather (model.py padded column-parallel -> NCCL)
    A3  routed latent all-gather      (model.py padded column-parallel -> NCCL)
    A4  routed latent all-reduce      (moe_runner.py -> ring)
    A5  layer output all-reduce       (moe_runner.py -> ring)

so a layer is five pieces, and an MLA layer additionally leaves its chunked
context passes (DCP key gathers, whose count depends on the context length)
outside any piece.

Buffers. Every value that crosses a piece boundary must live at an address
fixed at capture time and outside the graphs' private memory pool, because a
pool recycles a piece's storage for the next piece captured into it. The
driver therefore owns one buffer per (half, boundary role, shape, dtype): a
piece ends by writing the collective's input into that buffer, the collective
runs on it, and the next piece reads it. The two halves need separate buffers
even though their shapes are equal: one half's collective is in flight on the
comm stream while the other half's compute stream is producing the input of
its own collective of the same shape, so a shared buffer would be overwritten
mid-transfer. The same argument applies to the ring's own replay entries,
which are keyed by ``(numel, dtype, granule)`` and would otherwise be shared
by the two halves (``B12X_PCIE_DMA_GRAPH_REPLAY_MAX_ENTRIES`` must hold one
entry per shape per half).

Status: research-only. The plan construction, the buffer budget, the key
derivation and the record/replay sequencing are exercised on CPU through
``GraphBackend`` (tests/v1/worker/test_k3_piecewise_graph.py); no GPU run has
captured or replayed a piece. ``VLLM_K3_PREFILL_PIECEWISE_GRAPH`` is off by
default and nothing in this module runs when it is off.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Boundary roles, in the order one layer reaches them. The role is part of a
# buffer's key, so a layer's five boundaries never share storage even when
# their shapes agree (A1 and A5 do).
BOUNDARY_ROLES = ("A1", "A2", "A3", "A4", "A5")

# Layer kinds. A linear-attention layer's device work between its collectives
# is a fixed sequence at a fixed row count; a latent-attention layer's
# chunked context pass is not (its chunk count follows the context length),
# so its attention region stays eager.
KDA_LAYER = "kda"
MLA_LAYER = "mla"


def enabled() -> bool:
    return os.getenv("VLLM_K3_PREFILL_PIECEWISE_GRAPH", "0") == "1"


def self_check_enabled() -> bool:
    """Replay each captured half a second time and compare the residual
    digest with the eager forward's (``residual_digest``). Bit-identical
    output is the requirement; a mismatch disables the piecewise path."""
    return os.getenv("VLLM_K3_PREFILL_PIECEWISE_GRAPH_CHECK", "0") == "1"


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PieceKey:
    """Identity of one captured piece.

    ``half`` separates the two micro-batches because they own separate
    boundary buffers; ``rows`` separates row counts because a graph bakes in
    every launch geometry; ``layer`` and ``index`` place the piece in the
    layer, and ``after``/``before`` name the collectives it sits between
    (``None`` at the model's first and last piece).
    """

    half: int
    rows: int
    layer: int
    index: int
    after: str | None
    before: str | None

    def __str__(self) -> str:
        return (
            f"half{self.half}/rows{self.rows}/layer{self.layer}"
            f"/piece{self.index}[{self.after or '^'}->{self.before or '$'}]"
        )


@dataclasses.dataclass(frozen=True)
class BoundarySpec:
    """One collective and the tensor that crosses into it."""

    role: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    in_place: bool

    @property
    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return n

    def nbytes(self) -> int:
        return self.numel * torch.empty((), dtype=self.dtype).element_size()


@dataclasses.dataclass(frozen=True)
class LayerSpec:
    """A layer's kind and the collectives it reaches, in order."""

    kind: str
    boundaries: tuple[BoundarySpec, ...]

    @property
    def capturable(self) -> bool:
        return self.kind == KDA_LAYER


def kimi_k3_layer_boundaries(
    rows: int,
    hidden_size: int,
    router_experts: int,
    latent_width: int,
    latent_width_padded: int,
    router_dtype: torch.dtype = torch.float32,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> tuple[BoundarySpec, ...]:
    """The five collectives of one Kimi-K3 layer at ``rows`` rows.

    ``router_experts`` and ``latent_width`` are the rank-local widths the two
    all-gathers send; their shape is the tensor that crosses into the
    collective.
    """
    return (
        BoundarySpec("A1", (rows, hidden_size), activation_dtype, True),
        BoundarySpec("A2", (rows, router_experts), router_dtype, False),
        BoundarySpec("A3", (rows, latent_width), activation_dtype, False),
        BoundarySpec("A4", (rows, latent_width_padded), activation_dtype, True),
        BoundarySpec("A5", (rows, hidden_size), activation_dtype, True),
    )


def kimi_k3_boundary_buffers(
    rows: int,
    hidden_size: int,
    router_experts: int,
    router_experts_gathered: int,
    latent_width: int,
    latent_width_gathered: int,
    latent_width_padded: int,
    router_dtype: torch.dtype = torch.float32,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> tuple[BoundarySpec, ...]:
    """Every tensor that crosses a boundary of one layer, both ends of the
    all-gathers included: a gather's result is read by the piece after it and
    therefore needs an address of its own."""
    sent = kimi_k3_layer_boundaries(
        rows,
        hidden_size,
        router_experts,
        latent_width,
        latent_width_padded,
        router_dtype,
        activation_dtype,
    )
    return sent + (
        BoundarySpec("A2out", (rows, router_experts_gathered), router_dtype, False),
        BoundarySpec("A3out", (rows, latent_width_gathered), activation_dtype, False),
    )


def expected_pieces(layers: Sequence[LayerSpec]) -> int:
    """Pieces a forward over ``layers`` produces.

    Mirrors the session's state machine: a piece closes before every
    collective of a captured layer, one closes when a layer that is not
    captured suspends the recording, and the forward closes the piece that is
    open at the end.
    """
    pieces = sum(
        len(layer.boundaries) if layer.capturable else 1 for layer in layers
    )
    return pieces + 1


def boundary_buffer_bytes(
    boundaries: Sequence[BoundarySpec], halves: int = 2, share_equal_shapes: bool = True
) -> int:
    """Bytes of boundary storage the driver retains.

    Boundary buffers are reused by every layer, so the budget is the distinct
    boundary set times the number of halves. ``share_equal_shapes`` folds
    boundaries of equal shape and dtype that are never live at the same time
    (a layer's attention-output and layer-output all-reduces) onto one
    buffer, which is what the ring's own replay entries do because they are
    keyed by element count.
    """
    if share_equal_shapes:
        sizes = {(b.shape, b.dtype): b.nbytes() for b in boundaries}
        per_half = sum(sizes.values())
    else:
        per_half = sum(b.nbytes() for b in boundaries)
    return per_half * halves


# --------------------------------------------------------------------------
# Static storage
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TensorRef:
    """What a replay requires of a tensor a captured piece reads."""

    name: str
    data_ptr: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def of(cls, name: str, tensor: torch.Tensor) -> TensorRef:
        return cls(name, tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)


class PointerGuard:
    """Addresses a capture baked in, checked again on every replay.

    A captured kernel reads the address its operand had at capture time. Any
    tensor a piece reads that the driver did not place in static storage --
    an attention-metadata tensor rebuilt per chunk, a workspace the caching
    allocator moved, a KV cache slot mapping -- makes the replay read stale
    memory silently. Recording the addresses and rejecting a replay whose
    inputs moved turns that into a fallback instead of a wrong answer.
    """

    def __init__(self) -> None:
        self._refs: dict[str, TensorRef] = {}

    def snapshot(self, name: str, tensor: torch.Tensor) -> None:
        self._refs[name] = TensorRef.of(name, tensor)

    def verify(self, name: str, tensor: torch.Tensor) -> None:
        expected = self._refs.get(name)
        if expected is None:
            raise PlanMismatch(f"{name} was not present when the plan was recorded")
        got = TensorRef.of(name, tensor)
        if got != expected:
            raise PlanMismatch(
                f"{name} moved between capture and replay: {expected} -> {got}"
            )

    def verify_all(self, tensors: dict[str, torch.Tensor]) -> None:
        missing = set(self._refs) - set(tensors)
        if missing:
            raise PlanMismatch(f"replay is missing {sorted(missing)}")
        for name, tensor in tensors.items():
            self.verify(name, tensor)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._refs))

    def __len__(self) -> int:
        return len(self._refs)


class StaticBuffers:
    """Storage a replay reads and writes, outside the graphs' memory pool.

    Keyed by ``(half, role, shape, dtype)``: the halves never share a buffer
    (see the module docstring) and a role never shares with another role of
    the same shape, so that a layer's attention-output and layer-output
    all-reduces cannot alias while one of them is in flight.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._buffers: dict[
            tuple[int, str, tuple[int, ...], torch.dtype], torch.Tensor
        ] = {}

    def get(
        self, half: int, role: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        key = (half, role, tuple(shape), dtype)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=dtype, device=self.device)
            self._buffers[key] = buffer
        return buffer

    def bind(self, half: int, role: str, tensor: torch.Tensor) -> torch.Tensor:
        """Copy ``tensor`` into its static buffer and return the buffer.

        The caller then uses the returned tensor, so a capture records the
        static address and a replay refreshes the same address.
        """
        buffer = self.get(half, role, tuple(tensor.shape), tensor.dtype)
        buffer.copy_(tensor)
        return buffer

    @property
    def total_bytes(self) -> int:
        return sum(b.numel() * b.element_size() for b in self._buffers.values())

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(sorted({key[1] for key in self._buffers}))

    def __len__(self) -> int:
        return len(self._buffers)


# Device tensors a captured prefill piece reads that the eager path rebuilds
# for every chunk. Each must be served from ``StaticBuffers`` (contents
# refreshed in place before the replay) or the replay reads the previous
# chunk's addresses. The list is the remaining work between this module and a
# replay that can serve traffic; ``PointerGuard`` rejects a replay in which
# any of them moved.
REPLAY_STATIC_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("input_ids", "vllm/v1/worker/gpu/model_runner.py prepare_inputs"),
    ("positions", "vllm/v1/worker/gpu/model_runner.py prepare_inputs"),
    ("query_start_loc", "vllm/v1/worker/gpu/input_batch.py"),
    ("seq_lens", "vllm/v1/worker/gpu/input_batch.py"),
    ("slot_mapping", "vllm/v1/worker/gpu/attn_utils.py build_slot_mappings_by_layer"),
    ("block_table", "vllm/v1/worker/gpu/attn_utils.py prepare_attn"),
    ("dcp_local_seq_lens", "vllm/v1/worker/gpu/cp_utils.py prepare_dcp_local_seq_lens"),
    ("mamba_state_indices", "vllm/models/kimi_k3/nvidia/kda_metadata.py"),
    ("mamba_conv_indices", "vllm/models/kimi_k3/nvidia/kda_metadata.py"),
    ("kda_checkpoint_offsets", "vllm/models/kimi_k3/nvidia/kda_metadata.py"),
)


# --------------------------------------------------------------------------
# Recorded plan
# --------------------------------------------------------------------------


@dataclasses.dataclass
class RecordedPiece:
    key: PieceKey
    graph: object

    def run(self) -> None:
        self.graph.replay()


@dataclasses.dataclass
class RecordedCollective:
    role: str
    layer: int
    run_fn: object

    def run(self) -> None:
        self.run_fn()


@dataclasses.dataclass
class RecordedPlan:
    """The half's device work as a flat list of pieces and collectives."""

    half: int
    rows: int
    steps: list[object] = dataclasses.field(default_factory=list)

    @property
    def pieces(self) -> list[RecordedPiece]:
        return [s for s in self.steps if isinstance(s, RecordedPiece)]

    @property
    def collectives(self) -> list[RecordedCollective]:
        return [s for s in self.steps if isinstance(s, RecordedCollective)]

    def signature(self) -> tuple[str, ...]:
        """Ordered roles of the collectives, for comparison against the plan
        the layer list predicts and against the other half's."""
        return tuple(c.role for c in self.collectives)

    def replay(self) -> None:
        for step in self.steps:
            step.run()


class PlanMismatch(RuntimeError):
    """The forward did not reach the boundaries the recorded plan holds."""


def merged_steps(
    first: RecordedPlan, second: RecordedPlan
) -> list[tuple[int, object]]:
    """Interleave two halves' recorded steps so each half's collective runs
    while the other half computes.

    One half's piece is issued, then its collective, then the other half's
    piece, then its collective, and so on. Because the collectives go to the
    comm stream and the pieces to the compute stream, a collective of one
    half covers the next piece of the other. Two recorded plans make the
    two-thread hand-off unnecessary: the interleaving is a host-side merge of
    two step lists in a fixed order, so both halves' collectives are issued
    in the same order on every rank.
    """
    order: list[tuple[int, object]] = []
    cursors = [0, 0]
    plans = (first, second)
    turn = 0
    while any(cursors[i] < len(plans[i].steps) for i in (0, 1)):
        plan = plans[turn]
        cursor = cursors[turn]
        if cursor < len(plan.steps):
            step = plan.steps[cursor]
            order.append((plan.half, step))
            cursors[turn] = cursor + 1
            # Stay on this half until its next collective has been issued, so
            # the hand-off happens at the boundary and not inside a piece.
            if isinstance(step, RecordedPiece) and cursor + 1 < len(plan.steps):
                order.append((plan.half, plan.steps[cursor + 1]))
                cursors[turn] = cursor + 2
        turn ^= 1
    return order


# --------------------------------------------------------------------------
# Capture backend
# --------------------------------------------------------------------------


class GraphBackend(Protocol):
    """The four graph operations the session needs, so the sequencing can be
    exercised without a device."""

    def new_pool(self, half: int) -> object: ...

    def begin(self, pool: object) -> object: ...

    def end(self, graph: object) -> None: ...


class CudaGraphBackend:
    """Capture on the calling (compute) stream, one memory pool per half.

    A pool is shared by every piece of one half and the pieces replay in the
    order they were captured, which is the condition under which a shared
    pool may recycle storage. The halves must not share a pool: their pieces
    interleave at replay in an order that differs from either half's capture
    order.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._pools: dict[int, object] = {}

    def new_pool(self, half: int) -> object:
        pool = self._pools.get(half)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            self._pools[half] = pool
        return pool

    def begin(self, pool: object) -> object:
        graph = torch.cuda.CUDAGraph()
        graph.capture_begin(pool=pool, capture_error_mode="thread_local")
        return graph

    def end(self, graph: object) -> None:
        graph.capture_end()


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------


class PiecewiseSession:
    """Records one half's pieces, then replays them.

    A session is per (half, rows). ``record`` runs the model normally with
    the pieces captured around every collective; ``replay`` runs the recorded
    steps and never enters the model's Python.
    """

    def __init__(
        self,
        half: int,
        rows: int,
        backend: GraphBackend,
        expected_signature: tuple[str, ...] | None = None,
        static: StaticBuffers | None = None,
    ) -> None:
        self.half = half
        self.rows = rows
        self.backend = backend
        self.expected_signature = expected_signature
        self.static = static
        self.guard = PointerGuard()
        # What the recording forward returned. A capture executes nothing, so
        # these tensors hold their values only after a replay; every later
        # step reads the same objects, which the replay refreshes in place.
        self.output: object = None
        self.plan = RecordedPlan(half=half, rows=rows)
        self._pool: object | None = None
        self._open: object | None = None
        self._layer = -1
        self._piece_index = 0
        self._last_role: str | None = None
        self._recording = False
        self._paused = 0

    # -- recording ---------------------------------------------------------

    @contextmanager
    def record(self) -> Iterator[PiecewiseSession]:
        if self._recording:
            raise PlanMismatch("nested recording")
        self.plan = RecordedPlan(half=self.half, rows=self.rows)
        self._pool = self.backend.new_pool(self.half)
        self._recording = True
        self._layer = -1
        self._piece_index = 0
        self._last_role = None
        self._paused = 0
        self._begin_piece()
        try:
            yield self
        except BaseException:
            self._abort()
            raise
        else:
            self._end_piece(before=None)
            self._recording = False
            self._check_signature()

    def _begin_piece(self) -> None:
        if self._open is not None:
            raise PlanMismatch("a piece is already open")
        self._open = self.backend.begin(self._pool)

    def _end_piece(self, before: str | None) -> None:
        if self._open is None:
            raise PlanMismatch("no piece is open")
        graph = self._open
        self._open = None
        self.backend.end(graph)
        key = PieceKey(
            half=self.half,
            rows=self.rows,
            layer=self._layer,
            index=self._piece_index,
            after=self._last_role,
            before=before,
        )
        self._piece_index += 1
        self.plan.steps.append(RecordedPiece(key=key, graph=graph))

    def _abort(self) -> None:
        if self._open is not None:
            try:
                self.backend.end(self._open)
            except Exception:
                logger.exception("piecewise capture could not be closed")
            self._open = None
        self._recording = False
        self.plan = RecordedPlan(half=self.half, rows=self.rows)

    def _check_signature(self) -> None:
        got = self.plan.signature()
        if self.expected_signature is not None and got != self.expected_signature:
            raise PlanMismatch(
                f"half {self.half} reached {got}, expected "
                f"{self.expected_signature}"
            )

    @property
    def recording(self) -> bool:
        return self._recording and self._paused == 0

    def enter_layer(self, layer: int, capturable: bool) -> None:
        """Called before a layer runs. A layer that is not capturable pauses
        the session: its device work stays eager and its collectives are not
        boundaries, so the piece that was open when it started continues
        after it."""
        self._layer = layer
        if self._recording and not capturable:
            self.pause()

    def leave_layer(self, layer: int, capturable: bool) -> None:
        if self._recording and not capturable:
            self.resume()

    def pause(self) -> None:
        if not self._recording:
            return
        if self._paused == 0:
            self._end_piece(before="pause")
        self._paused += 1

    def resume(self) -> None:
        if not self._recording:
            return
        if self._paused == 0:
            raise PlanMismatch("resume without pause")
        self._paused -= 1
        if self._paused == 0:
            self._last_role = "pause"
            self._begin_piece()

    @contextmanager
    def boundary(self, role: str) -> Iterator[None]:
        """Close the piece before the collective ``role`` and open the piece
        after it. The body runs the collective eagerly and must register it
        with ``collective`` so the replay reissues it."""
        if not self.recording:
            yield
            return
        self._end_piece(before=role)
        try:
            yield
        finally:
            self._last_role = role
            self._begin_piece()

    def collective(self, role: str, run_fn) -> None:
        if not self.recording:
            return
        self.plan.steps.append(
            RecordedCollective(role=role, layer=self._layer, run_fn=run_fn)
        )

    # -- inputs ------------------------------------------------------------

    def bind(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Serve ``tensor``'s value from this half's static storage.

        Returns the static buffer holding a copy, so a capture records the
        static address and a replay refreshes the same address. Without
        static storage the tensor is returned unchanged and its address is
        recorded for the guard, which then rejects the first replay whose
        input moved.
        """
        # The plan is empty until the recording pass has finished, so the
        # binding that precedes it snapshots and every later one verifies.
        recording = self._recording or not self.plan.steps
        buffer = tensor if self.static is None else self.static.bind(
            self.half, name, tensor
        )
        if recording:
            self.guard.snapshot(name, buffer)
        else:
            self.guard.verify(name, buffer)
        return buffer

    def watch(self, name: str, tensor: torch.Tensor) -> None:
        """Require ``tensor`` to keep the address it had when the plan was
        recorded, without copying it.

        For the per-chunk tensors a captured piece reads that the driver does
        not own -- attention metadata, slot mappings, block tables -- the
        eager path rebuilds them for every chunk. Watching them turns the
        stale-address failure into a rejected replay that names the tensor
        (see ``REPLAY_STATIC_REQUIREMENTS``).
        """
        if self._recording or not self.plan.steps:
            self.guard.snapshot(name, tensor)
        else:
            self.guard.verify(name, tensor)

    def result_buffer(self, role: str, tensor: torch.Tensor) -> torch.Tensor:
        """Static storage for a collective's result, holding ``tensor``'s
        value. Falls back to the tensor itself when the session has no static
        storage, which the guard then rejects on the first replay."""
        if self.static is None:
            return tensor
        return self.static.bind(self.half, f"{role}:out", tensor)

    # -- replay ------------------------------------------------------------

    def replay(self) -> None:
        if self._recording:
            raise PlanMismatch("cannot replay while recording")
        if not self.plan.steps:
            raise PlanMismatch("nothing recorded")
        self.plan.replay()

    def describe(self) -> str:
        return ", ".join(
            str(s.key) if isinstance(s, RecordedPiece) else f"<{s.role}>"
            for s in self.plan.steps
        )


# --------------------------------------------------------------------------
# Process-wide state
# --------------------------------------------------------------------------

_SESSIONS: dict[tuple[int, int], PiecewiseSession] = {}
_STATIC: StaticBuffers | None = None
# Per thread: the overlapped split runs its halves on two threads, each of
# which reaches the collective hook with its own session.
_LOCAL = threading.local()
_DISABLED_REASON: str | None = None


def active_session() -> PiecewiseSession | None:
    return getattr(_LOCAL, "session", None)


def _set_active(session: PiecewiseSession | None) -> None:
    _LOCAL.session = session


def disable(reason: str) -> None:
    """Stop using the piecewise path for the rest of the process and say why
    once. Every caller falls back to the eager forward."""
    global _DISABLED_REASON, _SESSIONS
    if _DISABLED_REASON is None:
        _DISABLED_REASON = reason
        logger.warning("Kimi-K3 piecewise prefill graphs disabled: %s", reason)
    _set_active(None)
    _SESSIONS = {}


def disabled_reason() -> str | None:
    return _DISABLED_REASON


def reset() -> None:
    """Drop every session and buffer (tests, and a shape change that makes
    the recorded plans unreachable)."""
    global _SESSIONS, _STATIC, _DISABLED_REASON
    _SESSIONS = {}
    _STATIC = None
    _set_active(None)
    _DISABLED_REASON = None


def static_buffers(device: torch.device) -> StaticBuffers:
    global _STATIC
    if _STATIC is None or _STATIC.device != device:
        _STATIC = StaticBuffers(device)
    return _STATIC


def session_for(half: int, rows: int, device: torch.device) -> PiecewiseSession | None:
    if not enabled() or _DISABLED_REASON is not None:
        return None
    key = (half, rows)
    session = _SESSIONS.get(key)
    if session is None:
        session = PiecewiseSession(
            half, rows, CudaGraphBackend(device), static=static_buffers(device)
        )
        _SESSIONS[key] = session
    return session


@contextmanager
def half_region(
    session: PiecewiseSession | None,
    inputs: dict[str, torch.Tensor] | None = None,
    watch: dict[str, torch.Tensor] | None = None,
) -> Iterator[dict[str, torch.Tensor] | None]:
    """Run one half through ``session``.

    ``inputs`` are the per-chunk device tensors the half's pieces read; each
    is served from static storage so that a capture and every later replay
    read one address. Yields the bound inputs when the caller must run the
    model (the recording pass, or the eager path when there is no session)
    and ``None`` when the half was replayed from the recorded plan.

    A capture executes nothing, so the recording forward leaves its outputs
    undefined; the region replays the plan once when the recording closes, so
    the recording step returns the same values a replayed step does and the
    caller need not distinguish them. The recording step therefore issues its
    collectives twice.
    """
    if session is None:
        yield inputs
        return
    previous = active_session()
    _set_active(session)
    try:
        for name, tensor in (watch or {}).items():
            session.watch(name, tensor)
        bound = {
            name: session.bind(name, tensor) for name, tensor in (inputs or {}).items()
        }
        if session.plan.steps:
            session.replay()
            yield None
        else:
            with session.record():
                yield bound
            session.replay()
    except PlanMismatch as exc:
        disable(str(exc))
        raise
    finally:
        _set_active(previous)


@contextmanager
def collective_boundary(role: str) -> Iterator[None]:
    """Boundary hook for the model's collective wrappers. A no-op unless a
    half is being recorded."""
    session = active_session()
    if session is None or not session.recording:
        yield
        return
    with session.boundary(role):
        yield


def record_collective(role: str, run_fn) -> None:
    session = active_session()
    if session is not None:
        session.collective(role, run_fn)


def collective_role(op: str, tensor: torch.Tensor) -> str:
    """Name a collective by its operation and the width it carries. A layer's
    five collectives have five distinct names, and the name is what the
    recorded plan is compared against on the next forward."""
    width = int(tensor.shape[-1]) if tensor.ndim else 0
    return f"{op}[{width}]"


def run_collective(op: str, tensor: torch.Tensor, run_fn):
    """Run one collective at a piece boundary.

    Outside a recording this is ``run_fn()``. Inside one it closes the piece
    before the collective, runs it, and gives its result an address the next
    piece can be captured against: an in-place collective already has one
    (its input, which the preceding piece produced at a fixed address in the
    graph pool), and one that returns a new tensor is copied into this half's
    static storage, which the replay refreshes.
    """
    session = active_session()
    if session is None or not session.recording:
        return run_fn()
    role = collective_role(op, tensor)
    with session.boundary(role):
        out = run_fn()
        if out.data_ptr() == tensor.data_ptr():
            session.collective(role, run_fn)
            result = out
        else:
            static = session.result_buffer(role, out)
            session.collective(role, lambda: static.copy_(run_fn()))
            result = static
    return result


@contextmanager
def layer_region(layer: int, capturable: bool) -> Iterator[None]:
    """Layer hook for the model's layer loop: a layer that is not capturable
    runs eagerly inside the surrounding piece boundary."""
    session = active_session()
    if session is None or not session._recording:
        yield
        return
    session.enter_layer(layer, capturable)
    try:
        yield
    finally:
        session.leave_layer(layer, capturable)
