# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FsGCManager, the fs secondary tier's capacity bound.

These use real files on disk with explicitly set mtimes, since mtime is the
GC's single source of truth for recency. The background thread is neutralized
by giving every manager a very long `interval_s` and driving `sweep()` directly,
so nothing here depends on timing except the two stamping tests, which poll.
"""

import hashlib
import os
import time

import pytest

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.fs.gc_manager import FsGCManager

_BLOCK_BYTES = 4096


def _make_file_mapper(root_dir: str) -> FileMapper:
    return FileMapper(
        root_dir=root_dir,
        model_name="test/model",
        tokens_per_hash=16,
        blocks_per_file=1,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        rank=0,
        dtype="float32",
    )


def _key(index: int):
    return make_offload_key(hashlib.sha256(str(index).encode()).digest()[:16], 0)


def _write_block(mapper: FileMapper, index: int, age_s: float) -> str:
    """Create a block file for `index` whose mtime is `age_s` in the past."""
    path = mapper.get_file_name(_key(index))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * _BLOCK_BYTES)
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


@pytest.fixture
def mapper(tmp_path):
    return _make_file_mapper(str(tmp_path))


def _make_gc(mapper: FileMapper, **kwargs) -> FsGCManager:
    params = dict(
        max_bytes=10 * _BLOCK_BYTES,
        low_watermark=0.8,
        # Long enough that the background thread never sweeps on its own; each
        # test calls sweep() explicitly so assertions are deterministic.
        interval_s=3600.0,
        stamp_interval_s=1.0,
        grace_s=60.0,
    )
    params.update(kwargs)
    return FsGCManager(mapper, **params)  # type: ignore[arg-type]


def test_rejects_grace_not_exceeding_stamp_interval(mapper):
    # Otherwise a block in active use can be stamped, then age past the grace
    # window before its next stamp is due, and be swept while still hot.
    with pytest.raises(ValueError, match="must exceed stamp_interval_s"):
        _make_gc(mapper, stamp_interval_s=60.0, grace_s=60.0)


def test_rejects_out_of_range_low_watermark(mapper):
    with pytest.raises(ValueError, match="low_watermark"):
        _make_gc(mapper, low_watermark=1.5)


def test_under_cap_is_a_noop(mapper):
    gc = _make_gc(mapper)
    try:
        paths = [_write_block(mapper, i, age_s=1000) for i in range(5)]
        assert gc.sweep() == 0
        assert all(os.path.exists(p) for p in paths)
    finally:
        gc.shutdown()


def test_evicts_in_mtime_order_down_to_the_low_watermark(mapper):
    gc = _make_gc(mapper)
    try:
        # 15 blocks over a 10-block cap, oldest first. The sweep must free down
        # to the 8-block watermark, i.e. remove exactly the 7 oldest.
        paths = [_write_block(mapper, i, age_s=1000 - i) for i in range(15)]

        freed = gc.sweep()

        assert freed == 7 * _BLOCK_BYTES
        assert not any(os.path.exists(p) for p in paths[:7])
        assert all(os.path.exists(p) for p in paths[7:])
    finally:
        gc.shutdown()


def test_grace_window_keeps_recently_used_blocks_even_over_cap(mapper):
    gc = _make_gc(mapper, grace_s=60.0)
    try:
        # Everything is inside the grace window, so the tier is knowingly left
        # over its cap rather than evicting blocks that are still in use.
        paths = [_write_block(mapper, i, age_s=1) for i in range(15)]

        assert gc.sweep() == 0
        assert all(os.path.exists(p) for p in paths)
    finally:
        gc.shutdown()


def test_protect_keeps_an_lru_block_a_sweep_would_otherwise_evict(mapper):
    """The one case the grace window cannot cover.

    A promotion stamps its keys when the scheduler matches them, but the read
    itself can execute much later -- queued behind other reads on a busy tier.
    Once the mtime ages past grace_s the sweep would happily unlink the file out
    from under the in-flight read, so submit_load/submit_store pin the keys.
    """
    gc = _make_gc(mapper)
    try:
        # All 15 are far outside the grace window, so recency protects nothing.
        paths = [_write_block(mapper, i, age_s=1000 - i) for i in range(15)]

        # The 3 oldest have reads in flight; they are exactly the blocks the
        # sweep wants to remove first.
        gc.protect([_key(0), _key(1), _key(2)])

        freed = gc.sweep()

        # Still frees the full 7 blocks needed to reach the watermark, just
        # taking the next-oldest unprotected ones instead.
        assert freed == 7 * _BLOCK_BYTES
        assert all(os.path.exists(p) for p in paths[:3]), "in-flight blocks unlinked"
        assert not any(os.path.exists(p) for p in paths[3:10])
        assert all(os.path.exists(p) for p in paths[10:])

        # Once the jobs finish the blocks become evictable again. Refill past
        # the cap first: the previous sweep left the tier at its watermark, so
        # another sweep would legitimately have nothing to do.
        gc.release([_key(0), _key(1), _key(2)])
        for i in range(15, 20):
            _write_block(mapper, i, age_s=1)

        # 13 blocks over a 10-block cap needs 5 freed, and the released blocks
        # are still the oldest on disk, so they are now the first to go.
        assert gc.sweep() == 5 * _BLOCK_BYTES
        assert not any(os.path.exists(p) for p in paths[:3]), (
            "still pinned after release"
        )
    finally:
        gc.shutdown()


def test_release_is_refcounted(mapper):
    """Two tiers' jobs can protect the same key; one finishing must not unpin it."""
    gc = _make_gc(mapper)
    try:
        paths = [_write_block(mapper, i, age_s=1000 - i) for i in range(15)]

        gc.protect([_key(0)])
        gc.protect([_key(0)])
        gc.release([_key(0)])

        gc.sweep()
        assert os.path.exists(paths[0]), "unpinned while a job was still in flight"
    finally:
        gc.shutdown()


def test_touch_stamps_mtime_so_eviction_order_follows_use(mapper):
    gc = _make_gc(mapper)
    try:
        path = _write_block(mapper, 0, age_s=1000)
        before = os.stat(path).st_mtime

        gc.touch([_key(0)])

        deadline = time.monotonic() + 5.0
        while os.stat(path).st_mtime == before and time.monotonic() < deadline:
            time.sleep(0.01)
        assert os.stat(path).st_mtime > before
    finally:
        gc.shutdown()


def test_touch_rate_limits_repeat_stamps_of_the_same_key(mapper):
    """touch() arrives once per request per scheduling attempt with every key of
    the request, so an unthrottled utime per call would be thousands of syscalls
    per step."""
    gc = _make_gc(mapper, stamp_interval_s=30.0, grace_s=60.0)
    try:
        path = _write_block(mapper, 0, age_s=1000)
        gc.touch([_key(0)])

        deadline = time.monotonic() + 5.0
        while os.stat(path).st_mtime < time.time() - 900 and (
            time.monotonic() < deadline
        ):
            time.sleep(0.01)
        stamped = os.stat(path).st_mtime

        # Backdate the file, then touch again well inside stamp_interval_s: the
        # second touch must be dropped, leaving the backdated mtime in place.
        old = time.time() - 1000
        os.utime(path, (old, old))
        gc.touch([_key(0)])
        time.sleep(0.5)

        assert os.stat(path).st_mtime == pytest.approx(old)
        assert stamped > old
    finally:
        gc.shutdown()


def test_touch_tolerates_keys_with_no_file(mapper):
    """A block can live only in the DRAM primary tier, or its cascade to disk
    can have failed; neither is an error."""
    gc = _make_gc(mapper)
    try:
        gc.touch([_key(0), _key(1)])
        time.sleep(0.2)
        assert gc.sweep() == 0
    finally:
        gc.shutdown()


def test_sweep_ignores_non_block_files(mapper):
    """store_block writes <dest>.tmp before os.replace, and config.json must
    survive; only .bin files are eviction candidates."""
    gc = _make_gc(mapper, max_bytes=_BLOCK_BYTES)
    try:
        paths = [_write_block(mapper, i, age_s=1000 - i) for i in range(4)]
        tmp_path = paths[0] + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(b"\0" * _BLOCK_BYTES)
        stamp = time.time() - 2000
        os.utime(tmp_path, (stamp, stamp))

        gc.sweep()

        assert os.path.exists(tmp_path), "in-progress store was unlinked"
    finally:
        gc.shutdown()
