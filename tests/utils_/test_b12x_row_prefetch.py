# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Next-step span prediction and staging for the disk row prefetcher."""

import gc
import queue
import threading
import time
import weakref

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v4_1.common.engram_prefetch import compress_engram_tokens
from vllm.utils.b12x_row_prefetch import (
    PrefetchTarget,
    RowCacheSampler,
    RowPrefetcher,
    _Job,
    _PrefetchCounters,
    _run_job,
    plan_next_step_spans,
    row_cache_stats,
)

# (a) decoding, (b) final prefill chunk, (c)/(d) still prefilling.
IDX_MAPPING = np.array([7, 3, 11, 5], dtype=np.intp)
SCHEDULED = np.array([4, 4, 6, 8], dtype=np.int32)
COMPUTED_PREFILL = np.array([10, 8, 0, 16], dtype=np.int32)
PREFILL_LEN = np.array([10, 12, 20, 40], dtype=np.int32)
IS_PREFILLING = np.array([False, True, True, True], dtype=np.bool_)


def _plan(max_prefill_tokens):
    return plan_next_step_spans(
        idx_mapping_np=IDX_MAPPING,
        num_scheduled_tokens=SCHEDULED,
        num_computed_prefill_tokens=COMPUTED_PREFILL,
        prefill_len=PREFILL_LEN,
        is_prefilling=IS_PREFILLING,
        max_prefill_tokens=max_prefill_tokens,
    )


def test_finished_prefill_decodes_and_next_chunk_is_scheduled_length():
    """A chunk that ends the prompt is not lookahead; a mid-prompt chunk is."""
    spans = _plan(2048)
    assert spans.decode_positions.tolist() == [0, 1]
    assert spans.prefill_starts.tolist() == [6, 24]
    assert spans.prefill_lens.tolist() == [6, 8]
    assert spans.prefill_rows.tolist() == IDX_MAPPING[[2, 3]].tolist()


def test_prefill_lookahead_is_capped_by_the_budget():
    """The crossing span is truncated and later spans are dropped."""
    assert _plan(10).prefill_lens.tolist() == [6, 4]
    assert _plan(5).prefill_lens.tolist() == [5]
    assert _plan(5).prefill_rows.tolist() == [IDX_MAPPING[2]]


HISTORY_LEN = 3
HISTORY_PAD = 7
# Request rows 5 and 9 decode; row 11 keeps prefilling from position 6.
BATCH_ROWS = np.array([5, 9, 11], dtype=np.intp)
TOKENS = torch.zeros(16, 64, dtype=torch.int32)
for _row in (5, 9, 11):
    TOKENS[_row] = torch.arange(64, dtype=torch.int32) % 900 + 1
NUM_COMPUTED = torch.zeros(16, dtype=torch.int32)
NUM_COMPUTED[5] = 10
NUM_COMPUTED[9] = 2
LAST_SAMPLED = torch.zeros(16, 1, dtype=torch.int64)
LAST_SAMPLED[5] = 21
LAST_SAMPLED[9] = 3
DRAFTS = torch.tensor([[30, 31], [32, 33], [0, 0]], dtype=torch.int64)


class _Buffer:
    def __init__(self, values):
        self.gpu = values


def _batch(num_reqs=3):
    return type(
        "InputBatch",
        (),
        {
            "num_reqs": num_reqs,
            "idx_mapping_np": BATCH_ROWS,
            "num_scheduled_tokens": np.ones(num_reqs, dtype=np.int32),
            "num_computed_prefill_tokens_np": np.array([10, 10, 2], dtype=np.int32),
            "prefill_len_np": np.array([10, 10, 20], dtype=np.int32),
            "is_prefilling_np": np.array([False, False, True], dtype=np.bool_),
        },
    )()


def _req_states():
    return type(
        "RequestState",
        (),
        {
            "all_token_ids": _Buffer(TOKENS),
            "num_computed_tokens": _Buffer(NUM_COMPUTED),
            "last_sampled_tokens": LAST_SAMPLED,
        },
    )()


def _prefetcher(targets=()):
    """A prefetcher without device allocations: the gathers are device-agnostic."""
    prefetcher = object.__new__(RowPrefetcher)
    prefetcher._targets = tuple(targets)
    prefetcher._history_len = HISTORY_LEN
    prefetcher._history_pad = HISTORY_PAD
    prefetcher._device = torch.device("cpu")
    prefetcher._max_query_width = 3
    prefetcher._slots = [torch.zeros(4096, dtype=torch.int64) for _ in range(2)]
    prefetcher._free_slots = queue.Queue()
    prefetcher._jobs = queue.Queue()
    prefetcher._disabled = threading.Event()
    prefetcher._counters = _PrefetchCounters()
    return prefetcher


class _Event:
    """Stand-in for the ``torch.cuda.Event`` the staging path records on."""

    def synchronize(self):
        """Report the record as complete, so reuse is never blocked."""
        return None

    def record(self, stream=None):
        """Accept the staging path's CUDA event record without CUDA."""
        return None


class _Table:
    """Stand-in for a b12x ``DiskTable``."""

    def __init__(self, result=True):
        self.result = result
        self.seen = []

    def prefetch(self, ids):
        self.seen.append(ids)
        return self.result


class _NoPrefetch:
    """A DiskTable from a b12x build that has no ``prefetch``."""


def _job(prefetcher, slot, num_decode_rows, query_width, prefill_lens, targets):
    return _Job(
        slot_index=0,
        slot=slot,
        event=_Event(),
        num_decode_rows=num_decode_rows,
        query_width=query_width,
        prefill_lens=np.asarray(prefill_lens, dtype=np.int64),
        history_len=HISTORY_LEN,
        targets=tuple(targets),
        free_slots=prefetcher._free_slots,
        disabled=prefetcher._disabled,
        counters=prefetcher._counters,
    )


def _staged_spans():
    """Two decode rows (5, 9) and one prefill span on row 11 from position 6."""
    return plan_next_step_spans(
        idx_mapping_np=BATCH_ROWS,
        num_scheduled_tokens=np.array([1, 1, 4], dtype=np.int32),
        num_computed_prefill_tokens=np.array([10, 10, 2], dtype=np.int32),
        prefill_len=np.array([10, 10, 20], dtype=np.int32),
        is_prefilling=np.array([False, False, True], dtype=np.bool_),
        max_prefill_tokens=2048,
    )


def test_decode_window_is_history_then_sampled_then_drafts():
    """Negative history positions pad, and the staged draft chain extends the row."""
    decode = _prefetcher()._decode_tokens(
        _batch(), _req_states(), _staged_spans(), DRAFTS
    )
    assert decode.shape == (2, HISTORY_LEN + 3)
    # Row 5 has positions 7,8,9 committed; row 9 has only two, so the oldest pads.
    assert decode[0].tolist() == [8, 9, 10, 21, 30, 31]
    assert decode[1].tolist() == [HISTORY_PAD, 1, 2, 3, 32, 33]


def test_prefill_window_carries_history_before_the_next_chunk():
    """The hashed window needs the preceding context, not just the new chunk."""
    spans = _staged_spans()
    assert spans.prefill_starts.tolist() == [6]
    assert spans.prefill_lens.tolist() == [4]
    prefill = _prefetcher()._prefill_tokens(_req_states(), spans)
    # History positions 3,4,5 followed by the chunk at 6..9 of request row 11.
    assert prefill.tolist() == [4, 5, 6, 7, 8, 9, 10]


def test_worker_prefetches_the_rows_of_both_staged_blocks():
    """Slot bytes round-trip into exactly the per-block rows the target predicts."""
    spans = _staged_spans()
    prefetcher = _prefetcher()
    decode = prefetcher._decode_tokens(_batch(), _req_states(), spans, DRAFTS)
    prefill = prefetcher._prefill_tokens(_req_states(), spans)
    slot = prefetcher._slots[0]
    slot[: decode.numel()].copy_(decode.reshape(-1))
    slot[decode.numel() : decode.numel() + prefill.numel()].copy_(prefill)
    table = _Table()
    # One id per block row, taken from its newest token: the last draft of each
    # decode row and the last token of the chunk. A wrongly sliced block moves
    # these.
    target = PrefetchTarget(
        resolve_table=lambda: table, row_ids=lambda block: block[:, -1]
    )
    _run_job(_job(prefetcher, slot, 2, 3, spans.prefill_lens, [target]))
    assert [ids.tolist() for ids in table.seen] == [[10, 31, 33]]


@pytest.mark.parametrize(
    "table", [_NoPrefetch(), _Table(result=False)], ids=["no-prefetch", "cache-off"]
)
def test_worker_disables_after_an_unusable_prefetch(table):
    """A table that cannot prefetch must stop the prefetcher, not raise."""
    prefetcher = _prefetcher()
    target = PrefetchTarget(
        resolve_table=lambda: table, row_ids=lambda block: block.reshape(-1)
    )
    _run_job(
        _job(
            prefetcher,
            torch.zeros(4096, dtype=torch.int64),
            num_decode_rows=1,
            query_width=3,
            prefill_lens=[],
            targets=[target],
        )
    )
    assert prefetcher.disabled


def test_compress_engram_tokens_marks_dead_tokens():
    """Out-of-vocabulary, negative and image-sentinel ids hash as the pad row."""
    from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

    token_map = torch.arange(0, 130_000 * 7, 7, dtype=torch.int64)
    raw = torch.tensor(
        [0, 4, 129_263, IMAGE_SENTINEL_BASE_ID, 130_000, -1], dtype=torch.int64
    )
    assert compress_engram_tokens(raw, token_map).tolist() == [
        0,
        28,
        129_263 * 7,
        -1,
        -1,
        -1,
    ]


def _counting_prefetcher(targets):
    """A prefetcher that stages tokens without a CUDA device present."""
    prefetcher = _prefetcher(targets)
    prefetcher._events = [_Event(), _Event()]
    return prefetcher


def test_submit_books_a_queued_step_but_not_a_saturated_ring(monkeypatch):
    """A step that finds no free slot is lost work; a queued one is a submission."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    table = _Table()
    prefetcher = _counting_prefetcher(
        [
            PrefetchTarget(
                resolve_table=lambda: table, row_ids=lambda block: block[:, -1]
            )
        ]
    )

    prefetcher.submit(_batch(), _req_states(), DRAFTS)
    assert prefetcher.counters() == (0, 1, 0)
    assert prefetcher._jobs.qsize() == 0

    prefetcher._free_slots.put(0)
    prefetcher.submit(_batch(), _req_states(), DRAFTS)
    assert prefetcher.counters() == (1, 1, 0)

    # The two decode rows contribute their newest tokens (31, 33) and the
    # prefilling row its single scheduled next token (4); unique() collapses
    # them, so the row counter has a reason to be three.
    _run_job(prefetcher._jobs.get_nowait())
    assert [ids.tolist() for ids in table.seen] == [[4, 31, 33]]
    assert prefetcher.counters() == (1, 1, 3)


def test_a_disabled_prefetcher_books_no_dropped_submission(monkeypatch):
    """Drops mean lost work, so a prefetcher that stopped itself books nothing."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: None)
    prefetcher = _counting_prefetcher(())
    prefetcher._disabled.set()
    prefetcher._free_slots.put(0)

    prefetcher.submit(_batch(), _req_states(), DRAFTS)
    assert prefetcher.counters() == (0, 0, 0)


class _Cache:
    """Stand-in for a ``DiskRowCache``, counting how often it is read."""

    def __init__(self, snapshot, raise_on_read=False):
        """Serve ``snapshot`` to every read, or raise if asked to fail."""
        self.snapshot = snapshot
        self.raise_on_read = raise_on_read
        self.reads = 0

    def stats(self):
        """Count the read, then hand back the canned b12x ``stats()`` dict."""
        self.reads += 1
        if self.raise_on_read:
            raise RuntimeError("shard still registering")
        return self.snapshot


PLE_SNAPSHOT = {
    "row_cache_hits": 70,
    "row_cache_misses": 30,
    "row_cache_resident": 25,
    "row_cache_bytes": 2250,
    "prefetch_rows": 12,
    "prefetch_rows_dropped": 3,
    "prefetch_rows_loaded": 9,
    "prefetch_rows_cancelled": 1,
    "prefetch_source_errors": 1,
    "prefetch_job_errors": 0,
    "prefetch_queue_depth": 2,
}


def test_sampler_totals_each_kind_and_survives_a_broken_cache():
    """Two PLE caches fold into one kind; a raising cache must not fail it."""
    first, second = _Cache(dict(PLE_SNAPSHOT)), _Cache(dict(PLE_SNAPSHOT))
    broken = _Cache({}, raise_on_read=True)
    sampler = RowCacheSampler(
        [("ple", lambda: first), ("ple", lambda: second), ("engram", lambda: broken)],
        background=False,
    )

    tables = {table.kind: table for table in sampler.refresh()}

    assert set(tables) == {"ple", "engram"}
    ple = tables["ple"]
    assert (ple.hits, ple.misses) == (140, 60)
    assert (ple.resident_rows, ple.pool_bytes) == (50, 4500)
    assert (ple.prefetch_rows, ple.prefetch_rows_dropped) == (24, 6)
    assert (ple.prefetch_rows_loaded, ple.prefetch_rows_cancelled) == (18, 2)
    # The two b12x error counters fold into one: which one failed belongs in
    # the b12x log, not on a dashboard.
    assert ple.prefetch_errors == 2
    assert ple.prefetch_queue_depth == 4
    assert (first.reads, second.reads, broken.reads) == (1, 1, 1)
    assert tables["engram"].hits == 0


def test_the_step_path_never_touches_the_caches():
    """``stats()`` is what the worker calls per step, so it must read nothing.

    The alternative was measured on SM120: a cache read costs a 7 us median
    but a 3.6 ms worst case while the prefetch worker holds the cache lock, so
    sampling on the step path would trade decode latency for scrape freshness.
    """
    cache = _Cache(dict(PLE_SNAPSHOT))
    sampler = RowCacheSampler([("ple", lambda: cache)], background=False)

    for _ in range(1000):
        assert sampler.stats() == []
    assert cache.reads == 0

    sampler.refresh()
    assert cache.reads == 1
    for _ in range(1000):
        assert len(sampler.stats()) == 1
    assert cache.reads == 1


def test_the_sampler_thread_publishes_without_being_asked():
    """The refresh cadence is the sampler's own thread's business."""
    cache = _Cache(dict(PLE_SNAPSHOT))
    sampler = RowCacheSampler([("ple", lambda: cache)], interval=0.01)
    try:
        deadline = time.monotonic() + 2.0
        while not sampler.stats() and time.monotonic() < deadline:
            time.sleep(0.002)
        published = sampler.stats()
        assert [table.kind for table in published] == ["ple"]
        assert published[0].hits == 70
        assert sampler._thread is not None and sampler._thread.is_alive()
    finally:
        sampler._stop.set()
        if sampler._thread is not None:
            sampler._thread.join(2)
            assert not sampler._thread.is_alive()


def test_row_cache_stats_composes_worker_work_with_cache_totals():
    """The payload carries what the prefetcher did and what the caches saw."""
    prefetcher = _prefetcher()
    prefetcher._counters.submissions = 5
    prefetcher._counters.dropped_submissions = 2
    prefetcher._counters.staged_rows = 11
    cache = _Cache(dict(PLE_SNAPSHOT))
    sampler = RowCacheSampler([("ple", lambda: cache)], background=False)
    sampler.refresh()
    payload = row_cache_stats(prefetcher, sampler)

    assert (payload.submissions, payload.dropped_submissions) == (5, 2)
    assert (payload.staged_rows, payload.disabled) == (11, 0)
    assert [table.kind for table in payload.tables] == ["ple"]
    assert payload.tables[0].hits == 70

    # With prefetch never enabled the worker still reports its cache.
    engram_sampler = RowCacheSampler([("engram", lambda: cache)], background=False)
    engram_sampler.refresh()
    payload = row_cache_stats(None, engram_sampler)
    assert (payload.submissions, payload.staged_rows, payload.disabled) == (0, 0, 0)
    assert [table.kind for table in payload.tables] == ["engram"]
    assert cache.reads == 2

    assert row_cache_stats(None, None) is None


def test_a_failing_cache_keeps_its_last_good_totals():
    """A transient read failure may not publish lower cumulative counters.

    The frontend turns cumulative counters into deltas and re-baselines when a
    value falls, so a partial sample would silently discard real work.
    """
    cache = _Cache(dict(PLE_SNAPSHOT))
    sampler = RowCacheSampler([("ple", lambda: cache)], background=False)

    assert sampler.refresh()[0].hits == 70

    cache.raise_on_read = True
    assert sampler.refresh()[0].hits == 70
    assert cache.reads == 2

    # The held value is a snapshot, not a live view of the cache's own dict.
    cache.raise_on_read = False
    cache.snapshot["row_cache_hits"] = 90
    assert sampler.refresh()[0].hits == 90


def test_an_unbuilt_cache_still_publishes_its_kind_and_keeps_its_totals():
    """A configured kind is never dropped, which would freeze a gauge.

    The disk table is built lazily, so the resolver answers None before the
    first real step and could answer None again after teardown. The collector
    depends on every configured kind appearing in every snapshot: a kind that
    vanished would leave its gauges showing the last value forever.
    """
    built: list[_Cache] = []
    sampler = RowCacheSampler(
        [("ple", lambda: built[0] if built else None)], background=False
    )

    assert [(table.kind, table.hits) for table in sampler.refresh()] == [("ple", 0)]

    built.append(_Cache(dict(PLE_SNAPSHOT)))
    assert sampler.refresh()[0].hits == 70

    built.clear()
    assert [(table.kind, table.hits) for table in sampler.refresh()] == [("ple", 70)]


def test_the_refresh_thread_does_not_keep_the_sampler_alive():
    """The daemon thread must not be what retains the sampler and its caches."""
    cache = _Cache(dict(PLE_SNAPSHOT))
    sampler = RowCacheSampler([("ple", lambda: cache)], interval=0.001)
    thread = sampler._thread
    assert thread is not None and thread.is_alive()
    # Park the thread inside its wait, the window in which it must not be
    # holding the sampler: a target that retained it would survive here.
    deadline = time.monotonic() + 2.0
    while cache.reads == 0 and time.monotonic() < deadline:
        time.sleep(0.002)
    assert cache.reads > 0, "the sampler thread never refreshed"
    alive = weakref.ref(sampler)

    del sampler
    gc.collect()

    assert alive() is None
    deadline = time.monotonic() + 2.0
    while thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert not thread.is_alive()
