# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for FileSystemTierManager.

These tests use real disk I/O to verify the filesystem tier implementation.
The tier manager writes KV cache blocks to disk and reads them back, verifying
data integrity throughout the process.
"""

import mmap
import os
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.distributed.kv_events import MEDIUM_FS
from vllm.v1.kv_offload.base import (
    Locality,
    LookupResult,
    OffloadingEvent,
    OffloadingKVEventsConfig,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.tiering.base import JobMetadata
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs.manager import (
    FileSystemTierManager,
)
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLOCK_ELEMENTS = 128 * mmap.PAGESIZE  # 2MB per block for pagesize 4096.
_DTYPE: torch.dtype = torch.float32
_CTX = ReqContext(req_id="test")


def _make_offloading_spec(enable_kv_cache_events: bool) -> MagicMock:
    """Mock spec with an explicit global KV events flag."""
    spec = MagicMock()
    spec.config = OffloadingConfig(
        groups=(),
        worker_kv_bytes_per_block=0,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config={},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float32"),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )
    spec.blocks_per_chunk = 1
    spec.kv_events_config = OffloadingKVEventsConfig(
        enable_kv_cache_events=enable_kv_cache_events,
        self_describing_kv_events=False,
    )
    return spec


_MOCK_OFFLOADING_SPEC = _make_offloading_spec(enable_kv_cache_events=False)


def key(n: int) -> OffloadKey:
    return make_offload_key(n.to_bytes(8, "big"), 0)


def make_job(
    job_id: int,
    keys: list[OffloadKey],
    block_ids: list[int] | None = None,
    is_promotion: bool = False,
    req_context: ReqContext = _CTX,
) -> JobMetadata:
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return JobMetadata(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids, dtype=np.int64),
        is_promotion=is_promotion,
        req_context=req_context,
    )


def drain(tier: FileSystemTierManager) -> list:
    """Block until all in-flight jobs finish, then collect results."""
    tier.drain_jobs()
    return list(tier.get_finished_jobs())


def lookup_and_wait(
    tier: FileSystemTierManager,
    keys: list[OffloadKey],
    ctx: ReqContext = _CTX,
    timeout: float = 1.0,
) -> list[LookupResult]:
    """Perform a full async lookup cycle and return resolved results."""
    for k in keys:
        tier.lookup(k, ctx)
    tier.on_schedule_end(ScheduleEndContext(new_req_ids=[], preempted_req_ids=()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not tier._lookup_manager._pending_results.empty():
            break
        time.sleep(0.01)
    return [tier.lookup(k, ctx) for k in keys]


def _page_aligned_zero_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    page_size = mmap.PAGESIZE
    dtype_num_bytes = torch.tensor([], dtype=dtype).element_size()

    num_bytes = num_blocks * block_elements * dtype_num_bytes
    num_bytes_aligned = num_bytes + page_size
    t = torch.zeros(num_bytes_aligned, dtype=torch.uint8)

    ptr = t.data_ptr()
    alignment_offset = ptr % page_size
    # Move tensor to next page regardless.
    shift = page_size - alignment_offset
    t = t[shift : shift + num_bytes]
    return t.view(dtype).view(num_blocks, block_elements)


def _page_aligned_rand_tensor(
    num_blocks: int, block_elements: int, dtype: torch.dtype = _DTYPE
) -> torch.Tensor:
    rand_tensor = _page_aligned_zero_tensor(num_blocks, block_elements)
    rand_tensor[:] = torch.rand(num_blocks, block_elements, dtype=dtype)
    return rand_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
    )
    yield tier, tensor
    tier.shutdown()


@pytest.fixture
def fs_tier_with_events(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=mock_view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=4,
        n_write_threads=4,
        enable_kv_events=True,
        locality="LOCAL",
    )
    yield tier
    tier.shutdown()


def _make_bounded_fs_tier(
    tmp_path,
    max_blocks: int,
    *,
    enable_events: bool = False,
    n_read_threads: int = 4,
    n_write_threads: int = 4,
):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    assert view.strides is not None
    block_size = view.strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=enable_events),
        primary_kv_view=view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=n_read_threads,
        n_write_threads=n_write_threads,
        enable_kv_events=enable_events,
        locality="LOCAL" if enable_events else None,
        max_cache_size_bytes=max_blocks * block_size,
    )
    return tier, tensor, block_size


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lookup_empty_tier(fs_tier):
    tier, _ = fs_tier
    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.MISS, LookupResult.MISS]


def test_store_creates_file_and_lookup_succeeds(fs_tier):
    tier, _ = fs_tier
    job = make_job(1, [key(1)], [0])
    tier.submit_store(job)
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert lookup_and_wait(tier, [key(1)]) == [LookupResult.HIT]
    dest = tier.file_mapper.get_file_name(key(1))
    assert os.path.exists(dest), f"Expected file at {dest}"


def test_store_then_load_roundtrip(fs_tier):
    tier, _ = fs_tier
    job_s = make_job(1, [key(1), key(2)], [0, 1])
    tier.submit_store(job_s)
    store_results = drain(tier)
    assert all(r.success for r in store_results)

    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]

    job_l = make_job(2, [key(1), key(2)], [2, 3], is_promotion=True)
    tier.submit_load(job_l)
    load_results = drain(tier)
    assert all(r.success for r in load_results)
    # Blocks stay on disk after load
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_invalid_path_raises_at_construction():
    """Construction must fail immediately when the config file cannot be written."""
    tensor = _page_aligned_zero_tensor(32, _BLOCK_ELEMENTS)
    mock_view = memoryview(tensor.numpy())

    with pytest.raises(OSError):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=mock_view,
            tier_type="fs",
            root_dir="/dev/null/invalid_path",
        )


@pytest.mark.parametrize("locality", ["local", ""])
def test_invalid_locality_raises_at_construction(tmp_path, locality):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)

    with pytest.raises(ValueError, match="Locality"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=memoryview(tensor.numpy()),
            tier_type="fs",
            root_dir=str(tmp_path),
            locality=locality,
        )


def test_factory_forwards_locality_to_fs_tier(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "locality": "LOCAL",
        },
        memoryview(tensor.numpy()),
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier.locality is Locality.LOCAL
    finally:
        tier.shutdown()


def test_factory_forwards_fs_capacity(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    assert view.strides is not None
    max_cache_size_bytes = 2 * view.strides[0]
    tier = SecondaryTierFactory.create_secondary_tier(
        {
            "type": "fs",
            "root_dir": str(tmp_path),
            "n_read_threads": 1,
            "n_write_threads": 1,
            "max_cache_size_bytes": max_cache_size_bytes,
        },
        view,
        _MOCK_OFFLOADING_SPEC,
    )
    try:
        assert isinstance(tier, FileSystemTierManager)
        assert tier._max_cache_size_bytes == max_cache_size_bytes
    finally:
        tier.shutdown()


def test_fs_capacity_validation(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    assert view.strides is not None

    for invalid in (True, 1.5, "1024"):
        with pytest.raises(TypeError, match="max_cache_size_bytes"):
            FileSystemTierManager(
                offloading_spec=_MOCK_OFFLOADING_SPEC,
                primary_kv_view=view,
                tier_type="fs",
                root_dir=str(tmp_path),
                max_cache_size_bytes=invalid,
            )

    with pytest.raises(ValueError, match="at least one"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=view,
            tier_type="fs",
            root_dir=str(tmp_path),
            max_cache_size_bytes=view.strides[0] - 1,
        )


def test_bounded_fs_tier_evicts_lru_block(tmp_path):
    tier, _, block_size = _make_bounded_fs_tier(tmp_path, max_blocks=2)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(result.success for result in drain(tier))
        tier.submit_store(make_job(2, [key(2)], [1]))
        assert all(result.success for result in drain(tier))

        tier.touch([key(1)], _CTX)
        tier.submit_store(make_job(3, [key(3)], [2]))
        assert all(result.success for result in drain(tier))

        assert os.path.exists(tier.file_mapper.get_file_name(key(1)))
        assert not os.path.exists(tier.file_mapper.get_file_name(key(2)))
        assert os.path.exists(tier.file_mapper.get_file_name(key(3)))
        assert tier._cache_size_bytes == 2 * block_size
    finally:
        tier.shutdown()


def test_lookup_hit_is_pinned_until_request_finishes(tmp_path):
    tier, _, _ = _make_bounded_fs_tier(tmp_path, max_blocks=1)
    ctx = ReqContext(req_id="pinned-request")
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(result.success for result in drain(tier))
        assert lookup_and_wait(tier, [key(1)], ctx) == [LookupResult.HIT]

        tier.submit_store(make_job(2, [key(2)], [1]))
        results = drain(tier)
        assert len(results) == 1
        assert not results[0].success
        assert os.path.exists(tier.file_mapper.get_file_name(key(1)))

        tier.on_request_finished(ctx)
        tier.submit_store(make_job(3, [key(2)], [1]))
        assert all(result.success for result in drain(tier))
        assert not os.path.exists(tier.file_mapper.get_file_name(key(1)))
        assert os.path.exists(tier.file_mapper.get_file_name(key(2)))
    finally:
        tier.shutdown()


def test_lookup_pin_transfers_to_load_job(tmp_path):
    tier, _, _ = _make_bounded_fs_tier(tmp_path, max_blocks=1)
    ctx = ReqContext(req_id="promoted-request")
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(result.success for result in drain(tier))
        assert lookup_and_wait(tier, [key(1)], ctx) == [LookupResult.HIT]
        assert tier._request_pins[ctx.req_id] == {key(1)}

        tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True, req_context=ctx))
        assert ctx.req_id not in tier._request_pins
        assert tier._load_job_keys[2] == [key(1)]
        assert tier._pin_counts[key(1)] == 1

        assert all(result.success for result in drain(tier))
        assert tier._load_job_keys == {}
        assert tier._pin_counts == {}
        tier.on_request_finished(ctx)
    finally:
        tier.shutdown()


def test_queued_load_source_is_pinned(tmp_path, monkeypatch):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier, _, _ = _make_bounded_fs_tier(tmp_path, max_blocks=1)
    load_started = threading.Event()
    release_load = threading.Event()
    original_load_block = mgr_mod.load_block

    def blocked_load(*args, **kwargs):
        load_started.set()
        assert release_load.wait(timeout=5.0)
        return original_load_block(*args, **kwargs)

    monkeypatch.setattr(mgr_mod, "load_block", blocked_load)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(result.success for result in drain(tier))

        tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
        assert load_started.wait(timeout=5.0)
        tier.submit_store(make_job(3, [key(2)], [0]))
        release_load.set()

        results = {result.job_id: result for result in drain(tier)}
        assert results[2].success
        assert not results[3].success
        assert os.path.exists(tier.file_mapper.get_file_name(key(1)))
    finally:
        release_load.set()
        tier.shutdown()


def test_concurrent_stores_reserve_capacity(tmp_path, monkeypatch):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier, _, block_size = _make_bounded_fs_tier(
        tmp_path, max_blocks=2, n_write_threads=4
    )
    release_stores = threading.Event()
    entered = 0
    entered_cv = threading.Condition()
    original_store_block = mgr_mod.store_block

    def blocked_store(*args, **kwargs):
        nonlocal entered
        with entered_cv:
            entered += 1
            entered_cv.notify_all()
        assert release_stores.wait(timeout=5.0)
        return original_store_block(*args, **kwargs)

    monkeypatch.setattr(mgr_mod, "store_block", blocked_store)
    try:
        tier.submit_store(make_job(1, [key(1), key(2), key(3)], [0, 1, 2]))
        with entered_cv:
            assert entered_cv.wait_for(lambda: entered == 2, timeout=5.0)
        time.sleep(0.1)
        assert entered == 2
        assert tier._pending_store_bytes == 2 * block_size

        release_stores.set()
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert tier._cache_size_bytes == 2 * block_size
    finally:
        release_stores.set()
        tier.shutdown()


def test_capacity_index_trims_existing_namespace_on_startup(tmp_path):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    assert view.strides is not None
    block_size = view.strides[0]
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
    )
    paths: list[str] = []
    try:
        for job_id in (1, 2, 3):
            tier.submit_store(make_job(job_id, [key(job_id)], [job_id - 1]))
            assert all(result.success for result in drain(tier))
            path = tier.file_mapper.get_file_name(key(job_id))
            paths.append(path)
            os.utime(path, ns=(job_id, job_id))
    finally:
        tier.shutdown()

    bounded = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
        max_cache_size_bytes=2 * block_size,
    )
    try:
        assert not os.path.exists(paths[0])
        assert os.path.exists(paths[1])
        assert os.path.exists(paths[2])
        assert bounded._cache_size_bytes == 2 * block_size
    finally:
        bounded.shutdown()


def test_capacity_eviction_scans_each_candidate_once(tmp_path, monkeypatch):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier, _, block_size = _make_bounded_fs_tier(tmp_path, max_blocks=2)
    paths = [tier.file_mapper.get_file_name(key(i)) for i in range(1, 5)]
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb"):
            pass

    with tier._capacity_cv:
        for i in range(1, 5):
            tier._cache_lru[key(i)] = block_size
        tier._cache_size_bytes = 4 * block_size
        tier._pin_counts[key(1)] = 1

    remove_attempts: list[str] = []
    original_remove = mgr_mod.os.remove

    def tracked_remove(path):
        remove_attempts.append(path)
        if path == paths[1]:
            raise PermissionError("injected eviction failure")
        original_remove(path)

    monkeypatch.setattr(mgr_mod.os, "remove", tracked_remove)
    try:
        with tier._capacity_cv:
            assert tier._evict_until_fits_locked(0)
        assert list(tier._cache_lru) == [key(1), key(2)]
        assert tier._cache_size_bytes == 2 * block_size
        assert paths[0] not in remove_attempts
        assert remove_attempts.count(paths[1]) == 1
        assert remove_attempts == paths[1:]
    finally:
        tier.shutdown()


def test_capacity_index_removes_only_owned_stale_temp_files(tmp_path):
    tensor = _page_aligned_zero_tensor(1, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    assert view.strides is not None
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
    )
    block_path = tier.file_mapper.get_file_name(key(1))
    tier.shutdown()

    os.makedirs(os.path.dirname(block_path), exist_ok=True)
    stale_temp_path = f"{block_path}_123.tmp"
    unrelated_temp_path = f"{block_path}_writer.tmp"
    for path in (stale_temp_path, unrelated_temp_path):
        with open(path, "wb") as file:
            file.write(b"stale")

    bounded = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
        max_cache_size_bytes=view.strides[0],
    )
    try:
        assert not os.path.exists(stale_temp_path)
        assert os.path.exists(unrelated_temp_path)
    finally:
        bounded.shutdown()


def test_capacity_index_fails_if_stale_temp_cannot_be_removed(tmp_path, monkeypatch):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tensor = _page_aligned_zero_tensor(1, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    assert view.strides is not None
    tier = FileSystemTierManager(
        offloading_spec=_MOCK_OFFLOADING_SPEC,
        primary_kv_view=view,
        tier_type="fs",
        root_dir=str(tmp_path),
        n_read_threads=1,
        n_write_threads=1,
    )
    block_path = tier.file_mapper.get_file_name(key(1))
    tier.shutdown()

    os.makedirs(os.path.dirname(block_path), exist_ok=True)
    stale_temp_path = f"{block_path}_123.tmp"
    with open(stale_temp_path, "wb") as file:
        file.write(b"stale")

    original_remove = mgr_mod.os.remove

    def fail_stale_temp_remove(path):
        if path == stale_temp_path:
            raise PermissionError("injected cleanup failure")
        original_remove(path)

    monkeypatch.setattr(mgr_mod.os, "remove", fail_stale_temp_remove)
    with pytest.raises(OSError, match="Unable to remove stale filesystem KV temp"):
        FileSystemTierManager(
            offloading_spec=_MOCK_OFFLOADING_SPEC,
            primary_kv_view=view,
            tier_type="fs",
            root_dir=str(tmp_path),
            n_read_threads=1,
            n_write_threads=1,
            max_cache_size_bytes=view.strides[0],
        )


def test_failed_load_missing_file(fs_tier):
    """Test that loading a block whose file does not exist results in a failed job."""
    tier, _ = fs_tier
    job = make_job(1, [key(99)], [0], is_promotion=True)
    tier.submit_load(job)
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success


def test_multiple_jobs_tracked_independently(fs_tier):
    tier, _ = fs_tier
    job1 = make_job(1, [key(1)], [0])
    job2 = make_job(2, [key(2)], [1])
    tier.submit_store(job1)
    tier.submit_store(job2)
    results = drain(tier)
    job_ids = {r.job_id for r in results}
    assert job_ids == {1, 2}
    assert lookup_and_wait(tier, [key(1), key(2)]) == [
        LookupResult.HIT,
        LookupResult.HIT,
    ]


def test_multi_block_job_partial_failure(fs_tier):
    """A load job where one block file is missing yields a single failed JobResult."""
    tier, _ = fs_tier
    # Store two of three keys
    tier.submit_store(make_job(1, [key(10), key(11)], [0, 1]))
    assert all(r.success for r in drain(tier))

    # Load all three — key(99) was never stored
    tier.submit_load(
        make_job(2, [key(10), key(11), key(99)], [0, 1, 2], is_promotion=True)
    )
    results = drain(tier)

    assert len(results) == 1
    assert results[0].job_id == 2
    assert not results[0].success


def test_shutdown_discards_pending_tasks(fs_tier):
    """Shutdown clears both queues and stops all worker threads without draining."""
    tier, _ = fs_tier
    # Submit many tasks to ensure some remain pending
    for i in range(10):
        tier.submit_store(make_job(i, [key(i)], [i % 4]))

    # Shutdown immediately without draining
    tier.shutdown()

    # Verify queues are cleared and threads stopped
    assert len(tier._pool._load_q) == 0
    assert len(tier._pool._store_q) == 0
    assert all(not t.is_alive() for t in tier._pool._threads)


def test_store_load_data_integrity(fs_tier):
    """Data written by store must be exactly recovered by load."""
    tier, tensor = fs_tier
    # Populate tensor with random data
    tensor[:] = _page_aligned_rand_tensor(4, _BLOCK_ELEMENTS)

    # Store first 2 blocks
    num_store = 2
    expected = tensor[:num_store].clone()

    store_ids = list(range(num_store))
    keys = [key(i) for i in range(num_store)]

    tier.submit_store(make_job(1, keys, store_ids))
    results = drain(tier)
    assert all(r.success for r in results)

    # Overwrite source blocks to prove data is read from disk
    tensor[:num_store] = 0.0

    # Load into last 2 blocks
    load_ids = [2, 3]
    tier.submit_load(make_job(2, keys, load_ids, is_promotion=True))
    results = drain(tier)
    assert all(r.success for r in results)

    for i, bid in enumerate(load_ids):
        assert torch.allclose(tensor[bid], expected[i]), (
            f"Block {bid} data mismatch after store+load"
        )


def test_wait_idle_blocks_until_tasks_complete():
    """wait_idle must not return while a task is still in flight."""
    pool = DualQueueThreadPool(n_read_threads=1, n_write_threads=1)
    gate = threading.Event()
    pool.enqueue_store(job_id=1, n_tasks=1, tasks=[lambda: gate.wait(timeout=5.0)])

    waiter = threading.Thread(target=pool.wait_idle)
    waiter.start()
    try:
        waiter.join(timeout=0.2)
        assert waiter.is_alive(), "wait_idle returned before task completed"
        gate.set()
        waiter.join(timeout=5.0)
        assert not waiter.is_alive(), "wait_idle did not unblock"
    finally:
        gate.set()
        pool.shutdown(wait=True)
        waiter.join(timeout=5.0)


def test_batch_lookup_c_extension(tmp_path):
    """Validates batch_lookup_C: empty, single, all-existing, all-missing,
    mixed ordering, and input type validation."""
    try:
        from vllm.fs_io_C import batch_lookup as batch_lookup_C
    except ImportError:
        pytest.skip("fs_io_C extension not built")

    # Setup
    all_exist = [str(tmp_path / f"e{i}.bin") for i in range(3)]
    for p in all_exist:
        open(p, "w").close()
    all_missing = [str(tmp_path / f"m{i}.bin") for i in range(3)]

    # Empty list
    assert batch_lookup_C([]) == []

    # Single existing / missing
    assert batch_lookup_C([all_exist[0]]) == [True]
    assert batch_lookup_C([all_missing[0]]) == [False]

    # All existing / all missing
    assert batch_lookup_C(all_exist) == [True, True, True]
    assert batch_lookup_C(all_missing) == [False, False, False]

    # Mixed — verifies index ordering is preserved
    paths = [val for pair in zip(all_exist, all_missing) for val in pair]
    assert batch_lookup_C(paths) == [True, False, True, False, True, False]

    # Input validation: non-list argument
    with pytest.raises(TypeError):
        batch_lookup_C(("/tmp/foo",))
    with pytest.raises(TypeError):
        batch_lookup_C(None)

    # Input validation: non-str elements in list
    with pytest.raises(TypeError):
        batch_lookup_C([None])
    with pytest.raises(TypeError):
        batch_lookup_C([b"/tmp/foo"])
    with pytest.raises(TypeError):
        batch_lookup_C([42])
    with pytest.raises(TypeError):
        batch_lookup_C([all_exist[0], None])  # valid first, invalid mid-list


@pytest.mark.parametrize("use_c_ext", [True, False])
def test_batch_lookup_dispatch(fs_tier, monkeypatch, use_c_ext):
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    if use_c_ext and not mgr_mod._HAS_BATCH_LOOKUP_C:
        pytest.skip("fs_io_C extension not built")

    monkeypatch.setattr(mgr_mod, "_HAS_BATCH_LOOKUP_C", use_c_ext)

    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    assert all(r.success for r in drain(tier))

    results = lookup_and_wait(tier, [key(1), key(2)])
    assert results == [LookupResult.HIT, LookupResult.MISS]


# ---------------------------------------------------------------------------
# KV events
# ---------------------------------------------------------------------------


def test_capacity_eviction_event_precedes_replacement_store(tmp_path):
    tier, _, _ = _make_bounded_fs_tier(tmp_path, max_blocks=2, enable_events=True)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(result.success for result in drain(tier))
        tier.submit_store(make_job(2, [key(2)], [1]))
        assert all(result.success for result in drain(tier))
        list(tier.take_events())

        tier.touch([key(1)], _CTX)
        tier.submit_store(make_job(3, [key(3)], [2]))
        assert all(result.success for result in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 2
        assert events[0].removed
        assert events[0].keys == [key(2)]
        assert not events[1].removed
        assert events[1].keys == [key(3)]
        assert all(event.medium == "FS" for event in events)
        assert all(event.locality is Locality.LOCAL for event in events)
    finally:
        tier.shutdown()


def test_eviction_before_store_poll_emits_invalidation_only(tmp_path):
    tier, _, _ = _make_bounded_fs_tier(tmp_path, max_blocks=1, enable_events=True)
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        tier.drain_jobs()

        tier.submit_store(make_job(2, [key(2)], [1]))
        results = drain(tier)
        assert all(result.success for result in results)

        events = list(tier.take_events())
        assert [(event.removed, event.keys) for event in events] == [
            (True, [key(1)]),
            (False, [key(2)]),
        ]
    finally:
        tier.shutdown()


def test_successful_store_emits_stored_event(fs_tier_with_events):
    """A completed store job emits one stored event with the job's keys."""
    tier = fs_tier_with_events
    keys = [key(1), key(2)]
    tier.submit_store(make_job(1, keys, [0, 1]))
    assert all(r.success for r in drain(tier))

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == keys
    # Literal medium pins the wire contract, not just the constant choice.
    assert events[0].medium == "FS"
    assert events[0].locality is Locality.LOCAL
    assert not events[0].removed
    # take_events drains the buffer.
    assert list(tier.take_events()) == []


@pytest.mark.parametrize(
    ("locality", "expected"),
    [(None, None), ("REMOTE", Locality.REMOTE)],
)
def test_store_event_uses_configured_locality(tmp_path, locality, expected):
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    locality_config = {} if locality is None else {"locality": locality}
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
        **locality_config,
    )
    try:
        tier.submit_store(make_job(1, [key(1)], [0]))
        assert all(r.success for r in drain(tier))

        events = list(tier.take_events())
        assert len(events) == 1
        assert events[0].locality is expected
    finally:
        tier.shutdown()


def test_load_job_emits_no_event(fs_tier_with_events):
    tier = fs_tier_with_events
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    list(tier.take_events())

    tier.submit_load(make_job(2, [key(1)], [1], is_promotion=True))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert list(tier.take_events()) == []


def test_mixed_job_results_emit_event_only_for_successful_job(
    fs_tier_with_events, monkeypatch
):
    """With a failed and a successful store job in flight, exactly one event
    is emitted and its keys belong to the successful job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(1))
    original_store_block = mgr_mod.store_block

    def flaky_store_block(dest_path, *args, **kwargs):
        if dest_path == failing_path:
            raise OSError("injected store failure")
        return original_store_block(dest_path, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "store_block", flaky_store_block)

    tier.submit_store(make_job(1, [key(1)], [0]))
    tier.submit_store(make_job(2, [key(2)], [1]))
    results = drain(tier)
    assert len(results) == 2
    by_id = {r.job_id: r for r in results}
    assert not by_id[1].success
    assert by_id[2].success

    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key(2)]


def test_partially_failed_store_emits_no_event(fs_tier_with_events, monkeypatch):
    """A store job with any failed block emits no event for the whole job."""
    import vllm.v1.kv_offload.tiering.fs.manager as mgr_mod

    tier = fs_tier_with_events
    failing_path = tier.file_mapper.get_file_name(key(2))
    original_store_block = mgr_mod.store_block

    def flaky_store_block(dest_path, *args, **kwargs):
        if dest_path == failing_path:
            raise OSError("injected store failure")
        return original_store_block(dest_path, *args, **kwargs)

    monkeypatch.setattr(mgr_mod, "store_block", flaky_store_block)

    tier.submit_store(make_job(1, [key(1), key(2)], [0, 1]))
    results = drain(tier)
    assert len(results) == 1
    assert not results[0].success
    assert list(tier.take_events()) == []
    assert tier._store_job_keys == {}


def test_events_disabled_by_default(fs_tier):
    tier, _ = fs_tier
    tier.submit_store(make_job(1, [key(1)], [0]))
    results = drain(tier)
    assert len(results) == 1
    assert results[0].success
    assert tier.events is None
    assert tier._store_job_keys == {}
    assert list(tier.take_events()) == []


def test_events_require_global_kv_events_flag(tmp_path):
    """Tier-level opt-in alone is not enough; the global flag gates events."""
    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=False),
        primary_kv_view=memoryview(tensor.numpy()),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    try:
        assert tier.events is None
        tier.submit_store(make_job(1, [key(1)], [0]))
        results = drain(tier)
        assert len(results) == 1
        assert results[0].success
        assert list(tier.take_events()) == []
        assert tier._store_job_keys == {}
    finally:
        tier.shutdown()


def test_cascade_store_emits_fs_event_through_tiering_manager(tmp_path):
    """A GPU->CPU->fs cascade surfaces the tier-owned FS stored event via the
    TieringOffloadingManager's aggregated take_events()."""
    from vllm.v1.kv_offload.tiering.manager import (
        CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager,
    )

    tensor = _page_aligned_zero_tensor(4, _BLOCK_ELEMENTS)
    view = memoryview(tensor.numpy())
    mock_region = MagicMock()
    mock_region.create_kv_memoryview.return_value = view
    primary = CPUPrimaryTierOffloadingManager(num_blocks=4, mmap_region=mock_region)
    tier = FileSystemTierManager(
        offloading_spec=_make_offloading_spec(enable_kv_cache_events=True),
        primary_kv_view=primary.get_kv_memoryview(),
        tier_type="fs",
        root_dir=str(tmp_path),
        enable_kv_events=True,
    )
    manager = TieringOffloadingManager(primary_tier=primary, secondary_tiers=[tier])
    try:
        keys = [key(1), key(2)]
        manager.on_new_request(_CTX)
        assert manager.prepare_store(keys, _CTX) is not None
        manager.complete_store(keys, _CTX)  # cascades to the fs tier

        events: list[OffloadingEvent] = []
        ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not events:
            manager.on_schedule_end(ctx)
            events.extend(manager.take_events())
            time.sleep(0.01)

        fs_events = [e for e in events if e.medium == MEDIUM_FS]
        assert len(fs_events) == 1
        assert set(fs_events[0].keys) == set(keys)
        assert not fs_events[0].removed
    finally:
        tier.shutdown()
