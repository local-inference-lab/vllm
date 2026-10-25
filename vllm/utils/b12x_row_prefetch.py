# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Advisory host-side prefetch of disk-backed PLE/Engram rows.

A disk-backed row table reads its rows through a host row cache. Waiting for
the device hash to name the rows costs a full disk round trip inside the
forward pass, so the runner predicts the tokens the next step will hash and
hands their row ids to the table's ``prefetch`` from a worker thread. The
engine thread only records non-blocking device-to-host copies and a CUDA event;
all hashing, de-duplication and queueing happens off it.

Prediction is advisory. The synchronous read path stays the correctness path,
so a wrong guess costs a cache miss and nothing else.
"""

from __future__ import annotations

import queue
import threading
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.metrics.b12x_row_cache import B12xRowCacheStats, B12xTableStats

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState

logger = init_logger(__name__)

# Row-cache counters are refreshed on their own thread at this cadence. A
# Prometheus scrape is far coarser, and the read queues behind the cache lock,
# so it must never happen on a step path.
ROW_CACHE_SAMPLE_SECONDS = 1.0

# b12x ``stats()`` key -> ``B12xTableStats`` field, summed across tables of one
# kind. Prefetch errors collapse the two b12x error counters: an operator
# distinguishes them in the b12x log, not on a dashboard.
_TABLE_FIELDS = (
    ("row_cache_hits", "hits"),
    ("row_cache_misses", "misses"),
    ("row_cache_resident", "resident_rows"),
    ("row_cache_bytes", "pool_bytes"),
    ("prefetch_rows", "prefetch_rows"),
    ("prefetch_rows_dropped", "prefetch_rows_dropped"),
    ("prefetch_rows_loaded", "prefetch_rows_loaded"),
    ("prefetch_rows_cancelled", "prefetch_rows_cancelled"),
    ("prefetch_queue_depth", "prefetch_queue_depth"),
)
_ERROR_FIELDS = ("prefetch_source_errors", "prefetch_job_errors")


def _accumulate(table: B12xTableStats, snapshot: dict[str, Any]) -> None:
    """Fold one b12x ``stats()`` snapshot into the per-kind totals."""
    for source, target in _TABLE_FIELDS:
        setattr(table, target, getattr(table, target) + int(snapshot.get(source, 0)))
    for source in _ERROR_FIELDS:
        table.prefetch_errors += int(snapshot.get(source, 0))


@dataclass
class _PrefetchCounters:
    """Advisory prefetch work counters; one writer each, so none is locked.

    The engine thread owns ``submissions`` and ``dropped_submissions``, the
    prefetch worker owns ``staged_rows``.
    """

    submissions: int = 0
    dropped_submissions: int = 0
    staged_rows: int = 0


def _refresh_sampler(
    sampler_ref: weakref.ReferenceType[RowCacheSampler],
    stop: threading.Event,
    interval: float,
) -> None:
    """Refresh the referenced sampler on its cadence, then exit.

    Module level with a weak reference on purpose: the thread must not be what
    keeps the sampler alive, and it must not outlive it either. The local is
    released before the wait, or holding it across the interval would pin the
    sampler for the whole wait and defeat the weak reference entirely.
    """
    while True:
        sampler = sampler_ref()
        if sampler is None:
            return
        sampler.refresh()
        del sampler
        if stop.wait(interval):
            return


class RowCacheSampler:
    """Row-cache counters for metrics, refreshed off the engine thread.

    ``tables`` holds one ``(kind, resolve_table)`` pair per disk table, where
    the resolver returns ``None`` while the table is still built lazily.

    The refresh runs on its own daemon thread and :meth:`stats` only returns
    the last published list; that separation is the whole design. Measured on
    an SM120 device, ``DiskRowCache.stats()`` takes a 7 us median when the
    cache is quiet but a 3.6 ms worst case while the prefetch worker holds the
    cache lock to reserve and confirm rows, because it queues behind that
    lock. Sampling it per step would therefore put a multi-millisecond outlier
    into decode latency to serve a scrape that arrives seconds later.
    """

    def __init__(
        self,
        tables: Sequence[tuple[str, Callable[[], Any | None]]],
        *,
        interval: float = ROW_CACHE_SAMPLE_SECONDS,
        background: bool = True,
    ) -> None:
        """Bind the table resolvers and, unless asked not to, start the thread.

        ``background=False`` leaves the sampler passive for a caller that owns
        the refresh cadence, which is how the tests drive it deterministically.
        """
        self._tables = tuple(tables)
        # A zero or negative interval would spin the daemon thread against the
        # GIL and the cache lock, so the cadence has a floor.
        self._interval = max(float(interval), 0.001)
        self._published: list[B12xTableStats] = []
        self._stop = threading.Event()
        # One last-good snapshot per resolver: a transient read failure repeats
        # the previous numbers rather than publishing lower totals, which the
        # frontend would read as a worker restart and drop the accumulated
        # count for.
        self._last: list[dict[str, Any] | None] = [None] * len(self._tables)
        self._thread: threading.Thread | None = None
        if background and self._tables:
            # The thread reaches the sampler through a weak reference: a bound
            # method would keep the sampler, and through it every disk cache it
            # resolves, alive for the whole process even after the worker drops
            # it. Finalizing stops the loop so a collected sampler leaves no
            # thread behind.
            weakref.finalize(self, self._stop.set)
            self._thread = threading.Thread(
                target=_refresh_sampler,
                args=(weakref.ref(self), self._stop, self._interval),
                name="b12x-row-cache-sampler",
                daemon=True,
            )
            self._thread.start()

    def refresh(self) -> list[B12xTableStats]:
        """Read every cache once and publish the totals. Never on a step path.

        A closed cache or a raising reader is logged and leaves that table on
        its last good snapshot: these counters are cumulative, and nothing
        about metrics may break a step, an engine shutdown, or a total.
        """
        for position, (_kind, resolve_table) in enumerate(self._tables):
            try:
                disk = resolve_table()
                if disk is not None:
                    self._last[position] = dict(disk.stats())
            except Exception as exc:
                logger.debug_once("B12X row-cache sample failed: %r", exc)
        totals: dict[str, B12xTableStats] = {}
        for (kind, _resolve_table), snapshot in zip(self._tables, self._last):
            table = totals.setdefault(kind, B12xTableStats(kind=kind))
            if snapshot is not None:
                _accumulate(table, snapshot)
        self._published = list(totals.values())
        return self._published

    def stats(self) -> list[B12xTableStats]:
        """The last published snapshot: an attribute read, no lock, no cache."""
        return self._published


def row_cache_stats(
    prefetcher: RowPrefetcher | None,
    sampler: RowCacheSampler | None,
) -> B12xRowCacheStats | None:
    """This worker's metrics payload, or None when it holds no disk tables."""
    if prefetcher is None and sampler is None:
        return None
    tables = sampler.stats() if sampler is not None else []
    if prefetcher is None:
        return B12xRowCacheStats(tables=tables)
    submissions, dropped_submissions, staged_rows = prefetcher.counters()
    return B12xRowCacheStats(
        submissions=submissions,
        dropped_submissions=dropped_submissions,
        staged_rows=staged_rows,
        disabled=int(prefetcher.disabled),
        tables=tables,
    )


# TODO: fixed per-step bound on prefill lookahead tokens. b12x prefetch()
# normalizes ids with a per-id Python loop; this caps GIL time taken from the
# engine thread.
PREFILL_PREFETCH_MAX_TOKENS = 2048

_STOP = object()


@dataclass(frozen=True)
class PrefetchTarget:
    """One disk table and the row-id function that predicts its next-step ids.

    ``row_ids`` maps an ``(R, H + Q)`` int64 CPU tensor of raw token ids, the
    ``H`` committed history columns followed by the ``Q`` query columns, to the
    flat global row ids those tokens hash to.
    """

    resolve_table: Callable[[], Any | None]
    row_ids: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class NextStepSpans:
    """The token windows the next step is expected to hash.

    ``decode_positions`` are the batch positions that decode next step; the
    others are still prefilling, described by request-state row, first token
    position and length of their next chunk.
    """

    decode_positions: np.ndarray
    prefill_rows: np.ndarray
    prefill_starts: np.ndarray
    prefill_lens: np.ndarray


def plan_next_step_spans(
    *,
    idx_mapping_np: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    num_computed_prefill_tokens: np.ndarray,
    prefill_len: np.ndarray,
    is_prefilling: np.ndarray,
    max_prefill_tokens: int,
) -> NextStepSpans:
    """Split this step's batch into next-step decodes and prefill lookahead.

    Every input is a length-``num_reqs`` array in batch order at the start of
    this step. A request still prefilling keeps taking tokens next step, so its
    next chunk is predicted as this step's scheduled length, capped by the
    tokens left in the prompt and by ``max_prefill_tokens`` across the batch.
    """
    num_reqs = int(idx_mapping_np.shape[0])
    scheduled = np.asarray(num_scheduled_tokens[:num_reqs], dtype=np.int64)
    computed = np.asarray(num_computed_prefill_tokens[:num_reqs], dtype=np.int64)
    total = np.asarray(prefill_len[:num_reqs], dtype=np.int64)
    prefilling = np.asarray(is_prefilling[:num_reqs], dtype=bool)

    next_start = computed + scheduled
    still_prefill = prefilling & (next_start < total)
    positions = np.flatnonzero(still_prefill)

    rows: list[int] = []
    starts: list[int] = []
    lens: list[int] = []
    budget = int(max_prefill_tokens)
    for position in positions.tolist():
        length = min(
            int(total[position] - next_start[position]), int(scheduled[position])
        )
        if length <= 0:
            continue
        length = min(length, budget)
        rows.append(position)
        starts.append(int(next_start[position]))
        lens.append(length)
        budget -= length
        if budget <= 0:
            break
    return NextStepSpans(
        decode_positions=np.flatnonzero(~still_prefill).astype(np.int64),
        prefill_rows=idx_mapping_np[rows].astype(np.int64),
        prefill_starts=np.asarray(starts, dtype=np.int64),
        prefill_lens=np.asarray(lens, dtype=np.int64),
    )


@dataclass(frozen=True)
class _Job:
    """One step's staged tokens, ready for the worker to hash and prefetch.

    Frozen so the engine thread can hand it over and forget it; the slot,
    event and counters are deliberately shared with the prefetcher.
    """

    slot_index: int
    slot: torch.Tensor
    event: torch.cuda.Event
    num_decode_rows: int
    query_width: int
    prefill_lens: np.ndarray
    history_len: int
    targets: tuple[PrefetchTarget, ...]
    free_slots: queue.Queue[int]
    disabled: threading.Event
    counters: _PrefetchCounters = field(default_factory=_PrefetchCounters)


def _prefetch_worker(jobs: queue.Queue[Any]) -> None:
    """Hash staged tokens and hand the rows to each disk table, off the engine."""
    while True:
        job = jobs.get()
        if job is _STOP:
            return
        try:
            _run_job(job)
        except Exception as exc:
            logger.warning_once(
                "Disabling PLE/Engram row prefetch after error: %r", exc
            )
            job.disabled.set()
        finally:
            job.free_slots.put(job.slot_index)


def _run_job(job: _Job) -> None:
    """Turn a staged slot into row ids and ask each disk table to warm them.

    Runs on the prefetch worker: it waits for the asynchronous staging copy,
    then slices the slot back into the decode and prefill blocks the token
    hashes were taken over.
    """
    job.event.synchronize()
    history_len = job.history_len
    blocks = []
    if job.num_decode_rows:
        count = job.num_decode_rows * (history_len + job.query_width)
        blocks.append(job.slot[:count].view(job.num_decode_rows, -1))
    offset = sum(block.numel() for block in blocks)
    for length in job.prefill_lens.tolist():
        count = history_len + int(length)
        blocks.append(job.slot[offset : offset + count].view(1, count))
        offset += count
    if not blocks:
        return
    for target in job.targets:
        table = target.resolve_table()
        if table is None:
            continue
        if not callable(getattr(table, "prefetch", None)):
            logger.warning_once(
                "VLLM_PLE_PREFETCH=1 requires a b12x build whose DiskTable "
                "provides prefetch(ids) (local-inference-lab/flashinfer#44). "
                "Disabling VLLM_PLE_PREFETCH."
            )
            job.disabled.set()
            return
        ids = torch.unique(torch.cat([target.row_ids(block) for block in blocks]))
        job.counters.staged_rows += int(ids.numel())
        if table.prefetch(ids) is False:
            logger.warning_once(
                "DiskRowCache.prefetch is inactive; set a "
                "B12X_ROW_CACHE_ROWS/B12X_ROW_CACHE_BYTES size (GDS tables "
                "cannot prefetch). Disabling VLLM_PLE_PREFETCH."
            )
            job.disabled.set()
            return


class RowPrefetcher:
    """Stage the next step's tokens and prefetch their rows for every target."""

    def __init__(
        self,
        targets: tuple[PrefetchTarget, ...],
        *,
        history_len: int,
        history_pad: int,
        max_num_reqs: int,
        num_speculative_tokens: int,
        device: torch.device,
    ) -> None:
        """Allocate the pinned slot ring and the thread that consumes its jobs.

        The ring is deliberately shallow: a step whose slots are all still in
        flight is dropped rather than making the engine thread wait on disk.
        """
        self._targets = tuple(targets)
        self._history_len = int(history_len)
        self._history_pad = int(history_pad)
        self._device = device
        self._max_query_width = 1 + int(num_speculative_tokens)
        capacity = (
            max_num_reqs * (self._history_len + self._max_query_width)
            + PREFILL_PREFETCH_MAX_TOKENS
            + max_num_reqs * self._history_len
        )
        self._slots = [
            torch.empty(capacity, dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(2)
        ]
        self._events = [torch.cuda.Event() for _ in self._slots]
        self._free_slots: queue.Queue[int] = queue.Queue()
        for index in range(len(self._slots)):
            self._free_slots.put(index)
        self._jobs: queue.Queue[Any] = queue.Queue()
        self._disabled = threading.Event()
        # Work counters, read by the engine thread when it builds the step's
        # output; see ``_PrefetchCounters`` for who writes each one.
        self._counters = _PrefetchCounters()
        threading.Thread(
            target=_prefetch_worker,
            args=(self._jobs,),
            name="vllm-row-prefetch",
            daemon=True,
        ).start()
        weakref.finalize(self, self._jobs.put, _STOP)

    @property
    def disabled(self) -> bool:
        """True once a failure has switched prefetching off for this process."""
        return self._disabled.is_set()

    def counters(self) -> tuple[int, int, int]:
        """Cumulative submissions, dropped submissions and staged rows."""
        counters = self._counters
        return (
            counters.submissions,
            counters.dropped_submissions,
            counters.staged_rows,
        )

    def submit(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        draft_tokens: torch.Tensor | None,
    ) -> None:
        """Record the next step's tokens and queue the prefetch work.

        Never blocks on the GPU: the only device interaction is asynchronous
        copies into a free pinned slot plus one event record. A saturated slot
        ring drops the request rather than waiting for a slot, and any failure
        disables the prefetcher so the engine thread pays no further cost.
        """
        if self._disabled.is_set() or input_batch.num_reqs == 0:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        try:
            slot_index = self._free_slots.get_nowait()
        except queue.Empty:
            self._counters.dropped_submissions += 1
            return
        try:
            job = self._stage(slot_index, input_batch, req_states, draft_tokens)
        except Exception as exc:
            self._free_slots.put(slot_index)
            self._disabled.set()
            logger.warning_once(
                "Disabling PLE/Engram row prefetch after error: %r", exc
            )
            return
        if job is not None:
            self._counters.submissions += 1
            self._jobs.put_nowait(job)

    def _stage(
        self,
        slot_index: int,
        input_batch: InputBatch,
        req_states: RequestState,
        draft_tokens: torch.Tensor | None,
    ) -> _Job | None:
        """Copy this step's predicted tokens into a pinned slot, or return None.

        None means the step needed no prefetch, so the slot goes straight back
        to the ring.
        """
        num_reqs = input_batch.num_reqs
        spans = plan_next_step_spans(
            idx_mapping_np=input_batch.idx_mapping_np[:num_reqs],
            num_scheduled_tokens=input_batch.num_scheduled_tokens[:num_reqs],
            num_computed_prefill_tokens=input_batch.num_computed_prefill_tokens_np[
                :num_reqs
            ],
            prefill_len=input_batch.prefill_len_np[:num_reqs],
            is_prefilling=input_batch.is_prefilling_np[:num_reqs],
            max_prefill_tokens=PREFILL_PREFETCH_MAX_TOKENS,
        )
        decode = self._decode_tokens(input_batch, req_states, spans, draft_tokens)
        prefill = self._prefill_tokens(req_states, spans)
        parts = [block.reshape(-1) for block in (decode, prefill) if block is not None]
        if not parts:
            self._free_slots.put(slot_index)
            return None
        query_width = self._max_query_width
        if decode is not None:
            query_width = int(decode.shape[1]) - self._history_len
        slot = self._slots[slot_index]
        flat = torch.cat(parts)
        slot[: flat.numel()].copy_(flat, non_blocking=True)
        event = self._events[slot_index]
        event.record(torch.cuda.current_stream())
        return _Job(
            slot_index=slot_index,
            slot=slot,
            event=event,
            num_decode_rows=int(spans.decode_positions.shape[0]),
            query_width=query_width,
            prefill_lens=spans.prefill_lens.copy(),
            history_len=self._history_len,
            targets=self._targets,
            free_slots=self._free_slots,
            disabled=self._disabled,
            counters=self._counters,
        )

    def _decode_tokens(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
        spans: NextStepSpans,
        draft_tokens: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Committed history, last sampled token and staged drafts per decode row."""
        positions = spans.decode_positions
        if positions.shape[0] == 0:
            return None
        history_len = self._history_len
        rows = torch.as_tensor(
            input_batch.idx_mapping_np[positions],
            dtype=torch.int64,
            device=self._device,
        )
        base = req_states.num_computed_tokens.gpu[rows].long()[:, None]
        cols = base + torch.arange(
            -history_len, 0, dtype=torch.int64, device=self._device
        )
        history = torch.where(
            cols >= 0,
            req_states.all_token_ids.gpu[rows[:, None], cols.clamp_min(0)].long(),
            self._history_pad,
        )
        query = req_states.last_sampled_tokens[rows]
        if (
            draft_tokens is not None
            and draft_tokens.dim() == 2
            and draft_tokens.shape[1] > 0
        ):
            batch_pos = torch.as_tensor(
                positions, dtype=torch.int64, device=self._device
            )
            query = torch.cat((query, draft_tokens[batch_pos].long()), dim=1)
        return torch.cat((history, query), dim=1)

    def _prefill_tokens(
        self, req_states: RequestState, spans: NextStepSpans
    ) -> torch.Tensor | None:
        """History plus next-chunk tokens of every still-prefilling request, flat."""
        if spans.prefill_rows.shape[0] == 0:
            return None
        history_len = self._history_len
        rows, cols = [], []
        for row, start, length in zip(
            spans.prefill_rows.tolist(),
            spans.prefill_starts.tolist(),
            spans.prefill_lens.tolist(),
        ):
            rows.append(np.full(history_len + length, row, dtype=np.int64))
            cols.append(np.arange(start - history_len, start + length, dtype=np.int64))
        row_ids = torch.as_tensor(
            np.concatenate(rows), dtype=torch.int64, device=self._device
        )
        token_pos = torch.as_tensor(
            np.concatenate(cols), dtype=torch.int64, device=self._device
        )
        gathered = req_states.all_token_ids.gpu[row_ids, token_pos.clamp_min(0)].long()
        return torch.where(token_pos >= 0, gathered, self._history_pad)
