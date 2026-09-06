# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def _ubatch_active() -> bool:
    """True when this thread runs one half of a split prefill (a
    ``UBatchContext`` is registered for it); other threads, e.g. background
    store threads, keep the plain collective path."""
    import threading

    from vllm.v1.worker import ubatching

    return threading.get_ident() in ubatching._THREAD_ID_TO_CONTEXT


class _UbatchTrace:
    """Per-thread record of the collectives issued under a split prefill
    (``VLLM_K3_UBATCH_TRACE=1``): sequence number, shape, and CUDA events
    around the comm-stream work so a watchdog can report which collective's
    device work never completed on this rank."""

    enabled = __import__("os").getenv("VLLM_K3_UBATCH_TRACE", "0") == "1"
    records: dict[int, list] = {}
    lock = __import__("threading").Lock()

    @classmethod
    def begin(cls, shape) -> tuple:
        if not cls.enabled:
            return ()
        import threading

        start = torch.cuda.Event()
        start.record(torch.cuda.current_stream())
        rec = [
            threading.get_ident(), tuple(shape), start, None, __import__("time").time()
        ]
        with cls.lock:
            cls.records.setdefault(threading.get_ident(), []).append(rec)
        return (rec,)

    @classmethod
    def end(cls, token) -> None:
        if not token:
            return
        end = torch.cuda.Event()
        end.record(torch.cuda.current_stream())
        token[0][3] = end

    @classmethod
    def report(cls) -> str:
        with cls.lock:
            lines = []
            for tid, recs in cls.records.items():
                pending = [
                    i for i, r in enumerate(recs) if r[3] is None or not r[3].query()
                ]
                started = [i for i, r in enumerate(recs) if not r[2].query()]
                last = recs[-1] if recs else None
                lines.append(
                    f"thread {tid}: issued={len(recs)} "
                    f"last={last[1] if last else None} "
                    f"first_incomplete={pending[0] if pending else None} "
                    f"first_not_started={started[0] if started else None}"
                )
            return "; ".join(lines) if lines else "no collectives recorded"

    @classmethod
    def reset(cls) -> None:
        with cls.lock:
            cls.records.clear()


def _ubatch_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce one half's tensor on the shared comm stream while the other
    half computes.

    The input is snapshotted on the compute stream first: MoE outputs live in
    runtime buffers shared by both halves, so the other half may overwrite
    the source before a comm-stream read of it would execute. The snapshot
    is then reduced on the comm stream, the CPU yields to the other half,
    and on resumption the compute stream waits for the collective.

    No ``record_stream``: it would keep every snapshot block out of the
    allocator until the lagging comm stream drains and exhaust device
    memory within a chunk. Ordering is guaranteed instead by the schedule:
    the snapshot stays referenced here while this thread is suspended (the
    other thread cannot obtain its block), it is released only after the
    compute stream has been made to wait for the collective, and the output
    is allocated on the comm stream, whose next allocation is ordered after
    the compute-side consumption by the next ``dbo_switch_to_comm_sync``.
    """
    from vllm.v1.worker.ubatching import (
        dbo_switch_to_comm_sync,
        dbo_yield_and_switch_from_comm_to_compute,
    )

    from vllm.v1.worker.ubatching import dbo_switch_to_compute_sync

    lockstep = _ubatch_lockstep()
    snapshot = input_.clone()
    if lockstep:
        torch.cuda.synchronize()
    dbo_switch_to_comm_sync()
    token = _UbatchTrace.begin(input_.shape)
    out = get_tp_group().all_reduce(snapshot)
    _UbatchTrace.end(token)
    if lockstep:
        torch.cuda.synchronize()
    if input_.shape[-1] >= _ubatch_yield_min_cols():
        dbo_yield_and_switch_from_comm_to_compute()
    else:
        # Narrow collectives (the latent all-reduce sits between the MoE and
        # a short up-projection) stay on the comm stream but hand the CPU
        # straight back: the other half's MoE is already queued on the
        # compute stream, so yielding here would only leave this half's
        # final all-reduce nothing long to hide behind.
        dbo_switch_to_compute_sync()
    if lockstep:
        torch.cuda.synchronize()
    del snapshot
    return out


def _ubatch_yield_min_cols() -> int:
    """Collectives narrower than this many columns do not yield (default
    7168: yield at the attention-output and final all-reduces, not at the
    3584-wide latent one)."""
    import os

    return int(os.getenv("VLLM_K3_UBATCH_YIELD_MIN_COLS", "7168") or 0)


def _ubatch_lockstep() -> bool:
    """Diagnostic: VLLM_K3_UBATCH_LOCKSTEP=1 (or mode file ``lockstep``)
    drains the device around every hand-off so the two halves never execute
    concurrently on the GPU; a result that stays wrong under lockstep points
    at shared host-side or persistent-buffer state rather than a stream
    race."""
    import os

    from vllm.v1.worker.gpu.k3_ubatch_prefill import runtime_mode

    mode = runtime_mode()
    if mode is not None:
        return mode == "lockstep"
    return os.getenv("VLLM_K3_UBATCH_LOCKSTEP", "0") == "1"


def _ubatch_no_yield() -> bool:
    """Diagnostic: VLLM_K3_UBATCH_NO_YIELD=1 keeps the ubatch threads and ids
    but runs every collective on the compute stream without a hand-off, so
    the halves execute one after the other (isolates the per-ubatch slot
    semantics from the interleaving)."""
    import os

    from vllm.v1.worker.gpu.k3_ubatch_prefill import runtime_mode

    mode = runtime_mode()
    if mode is not None:
        return mode == "noyield"
    return os.getenv("VLLM_K3_UBATCH_NO_YIELD", "0") == "1"


_PIECEWISE_MODULE: Any = None
_PIECEWISE_ON: bool | None = None


def _piecewise():
    """The Kimi-K3 piecewise prefill graph driver, or None when it is off.

    A collective is a boundary between two captured pieces (a graph cannot
    launch the ring's replay graph, and a captured collective would hold the
    compute stream for the whole transfer), so every collective the model
    reaches passes through the driver while a half is being recorded.

    Every forward reaches this five times per layer, decode included, so the
    disabled path is one global read: the driver's flag is set at boot and is
    read once.
    """
    global _PIECEWISE_MODULE, _PIECEWISE_ON
    if _PIECEWISE_ON is None:
        from vllm.v1.worker.gpu import k3_piecewise_graph

        _PIECEWISE_MODULE = k3_piecewise_graph
        _PIECEWISE_ON = k3_piecewise_graph.enabled()
    if not _PIECEWISE_ON:
        return None
    return _PIECEWISE_MODULE if _PIECEWISE_MODULE.active_session() else None


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""

    def _run() -> torch.Tensor:
        if _ubatch_active() and not _ubatch_no_yield():
            return _ubatch_all_reduce(input_)
        return get_tp_group().all_reduce(input_)

    driver = _piecewise()
    if driver is None:
        return _run()
    return driver.run_collective("ar", input_, _run)


def tensor_model_parallel_all_reduce_in_place(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce a dead input tensor without allocating an output tensor."""

    def _run() -> torch.Tensor:
        if _ubatch_active() and not _ubatch_no_yield():
            # The caller's buffer is its own projection output; deliver the
            # result there on the compute stream once the collective is done.
            input_.copy_(_ubatch_all_reduce(input_))
            return input_
        return get_tp_group().all_reduce_in_place(input_)

    driver = _piecewise()
    if driver is None:
        return _run()
    return driver.run_collective("ar_in_place", input_, _run)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""

    def _run() -> torch.Tensor:
        return get_tp_group().all_gather(input_, dim)

    driver = _piecewise()
    if driver is None:
        return _run()
    return driver.run_collective("ag", input_, _run)


def tensor_model_parallel_all_gatherv(
    input_: torch.Tensor, sizes: list[int], dim: int = 0
) -> torch.Tensor:
    """All-gather variable-length tensor slices across the model-parallel group."""
    tp_group = get_tp_group()
    if tp_group.world_size == 1:
        return input_
    return tp_group.all_gatherv(input_, dim=dim, sizes=sizes)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
