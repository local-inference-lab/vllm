# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cached-reader capacity, output-page growth, and release fences."""

from types import SimpleNamespace

import pytest
import torch

from tests.v1.core import test_boundary_admission as base
from tests.v1.core.test_boundary_admission import initialize_hash as initialize_hash
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    _estimate_max_model_len_from_groups,
    _pool_bytes_per_block,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test


def manager(spec=3, **kwargs):
    attention = MLAAttentionSpec(
        block_size=2048,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        model_version="glm5_next",
    )
    recurrent = MambaSpec(
        block_size=256,
        shapes=((1,),),
        dtypes=(torch.uint8,),
        mamba_cache_mode="align",
        num_speculative_blocks=spec,
    )
    groups = [KVCacheGroupSpec(["attention"], attention)]
    groups += [KVCacheGroupSpec([str(i)], recurrent) for i in range(3)]
    return base.make_kv_cache_manager(
        KVCacheConfig(num_blocks=1934, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=524288,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
        dcp_world_size=1,
        scheduler_block_size=2048,
        hash_block_size=256,
        enable_boundary_checkpoints=True,
        **kwargs,
    )


def request(name, salt="0", length=8192, cap=4096):
    req = base.make_request(
        name, list(range(length)), 256, base.sha256, cache_salt=salt
    )
    req.max_tokens = cap
    req.sampling_params.max_tokens = cap
    return req


def seed(cache, salt="0", length=8192, lookahead=4):
    req = request("producer-" + salt, salt, length)
    cache.get_computed_blocks(req)
    assert cache.allocate_slots(req, length, num_lookahead_tokens=lookahead) is not None
    req.num_computed_tokens = length
    base.drain(cache)
    checkpoint = cache.publish_boundary_checkpoint(req, length, kind="prompt")
    cache.free(req)
    return checkpoint


def restore(cache, req, lookahead=4):
    cache.new_step_starts()
    blocks, hit, _ = cache.get_computed_blocks(req)
    assert hit
    allocated = cache.allocate_slots(
        req,
        max(1, req.num_prompt_tokens - hit),
        hit,
        blocks,
        num_lookahead_tokens=lookahead,
    )
    if allocated is not None:
        req.num_computed_tokens = req.num_prompt_tokens
        req.status = RequestStatus.RUNNING
    return allocated


@pytest.mark.parametrize("num_blocks", [62, 65, 80])
def test_small_pool_capacity_covers_instruction_checkpoint_restore(num_blocks):
    """Accepted pool geometry must admit both cold prompts and their restores."""
    attention = MLAAttentionSpec(
        block_size=2048,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        model_version="glm5_next",
    )
    recurrent = MambaSpec(
        block_size=256,
        shapes=((1,),),
        dtypes=(torch.uint8,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    groups = [KVCacheGroupSpec([str(i)], recurrent) for i in range(6)]
    groups.append(KVCacheGroupSpec(["attention"], attention))
    config = SimpleNamespace(
        use_request_boundary_checkpoints=True,
        scheduler_config=SimpleNamespace(max_num_scheduled_tokens=4096),
        attention_config=SimpleNamespace(hisparse_config=None),
        model_config=SimpleNamespace(max_model_len=1048576),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
    )
    max_model_len = _estimate_max_model_len_from_groups(
        config, groups, (num_blocks - 1) * _pool_bytes_per_block(groups)
    )
    if num_blocks == 62:
        # Cold admission fits, but restore needs 64 non-null blocks. Startup
        # must reject this pool instead of leaving an idle scheduler waiting.
        assert max_model_len == 0
        return
    assert max_model_len >= 6357
    cache = base.make_kv_cache_manager(
        KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        max_model_len=max_model_len,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
        dcp_world_size=2,
        scheduler_block_size=4096,
        hash_block_size=256,
        enable_boundary_checkpoints=True,
    )
    producer = request("producer", length=6357, cap=1024)
    producer.recurrent_instruction_boundary = 6345
    cache.get_computed_blocks(producer)
    for count in (4096, 2249):
        cache.new_step_starts()
        assert cache.allocate_slots(producer, count, num_lookahead_tokens=3)
        producer.num_computed_tokens += count
        base.drain(cache)
    checkpoint = cache.publish_boundary_checkpoint(producer, 6345, kind="instruction")
    assert checkpoint is not None
    cache.free(producer)
    assert cache.block_pool.get_num_free_blocks() == num_blocks - 1

    reader = request("reader", length=6357, cap=1024)
    reader.recurrent_instruction_boundary = 6345
    assert restore(cache, reader, lookahead=3) is not None
    base.drain(cache)
    reader.append_output_token_ids([10000])
    pending = False
    held: list[KVCacheBlock] = []
    for step in range(128):
        cache.new_step_starts()
        assert cache.allocate_slots(reader, 4, num_lookahead_tokens=3) is not None
        reader.num_computed_tokens += 4
        reader.num_in_flight_tokens += 4
        copies = cache.take_kv_cache_block_copies()[1]
        if pending:
            reader.num_in_flight_tokens -= 4
            reader.append_output_token_ids([10001 + step] * 4)
        cache.block_pool.free_blocks(held)
        held = copies
        pending = True
        cache.cache_blocks(reader, min(reader.num_computed_tokens, reader.num_tokens))
    cache.free(reader)
    cache.block_pool.free_blocks(held)
    assert cache.block_pool.get_num_free_blocks() == num_blocks - 1


def test_all_16_short_warm_readers_admit_without_checkpoint_loss():
    cache = manager()
    for index in range(16):
        seed(cache, str(index))
    published = set(cache.boundary_checkpoints._entries)
    active: list[Request] = []
    for index in range(16):
        req = request("warm-" + str(index), str(index))
        allocated = restore(cache, req)
        assert allocated is not None, (
            len(active),
            cache.block_pool.get_num_free_blocks(),
        )
        base.drain(cache)
        active.append(req)
    assert len(active) == 16
    assert published == set(cache.boundary_checkpoints._entries)
    assert cache.block_pool.get_num_free_blocks() >= 1400
    for req in active:
        cache.free(req)
    assert not cache._boundary_readers
    assert cache.block_pool.get_num_free_blocks() == 1933


@pytest.mark.parametrize(
    "status",
    [
        RequestStatus.FINISHED_STOPPED,
        RequestStatus.FINISHED_LENGTH_CAPPED,
        RequestStatus.FINISHED_ABORTED,
        RequestStatus.PREEMPTED,
    ],
)
@pytest.mark.parametrize("deferred", [False, True])
def test_reader_cleanup_and_id_reuse(status, deferred):
    cache = manager()
    seed(cache)
    req = request("reused")
    assert restore(cache, req) is not None
    assert req.request_id in cache._boundary_readers
    held = cache.take_kv_cache_block_copies()[1]
    req.status = status
    if deferred:
        blocks = cache.pop_blocks_for_free(req)
        assert blocks and all(b.ref_cnt > 0 for b in blocks if not b.is_null)
        assert req.request_id not in cache._boundary_readers
        cache.block_pool.free_blocks(reversed(blocks))
    else:
        cache.free(req)
    cache.block_pool.free_blocks(held)
    assert not cache._boundary_readers
    reused = request("reused", cap=1)
    assert restore(cache, reused) is not None
    assert "reused" in cache._boundary_readers
    base.drain(cache)
    cache.free(reused)
    assert not cache._boundary_readers
    assert cache.block_pool.get_num_free_blocks() == 1933


def test_failed_checkpoint_acquire_never_pins_reader(monkeypatch):
    cache = manager()
    seed(cache)
    waiting = request("waiting")
    before = [b.ref_cnt for b in cache.block_pool.blocks]
    monkeypatch.setattr(
        cache.boundary_checkpoints, "acquire", lambda checkpoint_id: None
    )
    assert restore(cache, waiting) is None
    assert not cache._boundary_readers
    assert before == [b.ref_cnt for b in cache.block_pool.blocks]


@pytest.mark.parametrize("cap", [1, 4096])
@pytest.mark.parametrize("batches", [1, 2])
@pytest.mark.parametrize("spec,lookahead", [(3, 3), (3, 4), (7, 8)])
def test_output_page_growth_preserves_inflight_blocks(cap, batches, spec, lookahead):
    cache = manager(spec)
    prompt = 8192 if cap == 4096 else 8191
    seed(cache, length=prompt, lookahead=lookahead)
    req = request("reader", length=prompt, cap=cap)
    assert restore(cache, req, lookahead) is not None
    base.drain(cache)
    if cap > 1:
        req.append_output_token_ids([9000] * (cap - 1))
        assert (
            cache.allocate_slots(req, cap - 1, num_lookahead_tokens=lookahead)
            is not None
        )
        req.num_computed_tokens += cap - 1
        base.drain(cache)
    pending = []
    for _ in range(batches):
        assert (
            cache.allocate_slots(req, spec + 1, num_lookahead_tokens=lookahead)
            is not None
        )
        req.num_computed_tokens += spec + 1
        req.num_in_flight_tokens += spec + 1
        pending.extend(cache.take_kv_cache_block_copies()[1])
        attention = cache.coordinator.single_type_managers[0]
        table = attention.req_to_blocks[req.request_id]
        assert (
            len(table)
            <= (prompt + cap + batches * (spec + 1) + lookahead + 2047) // 2048
        )
        assert all(b.ref_cnt > 0 for b in table if not b.is_null)
    # A plain prompt-plus-cap bound would miss this extra attention page.
    assert len(table) > (prompt + cap) // 2048
    blocks = cache.pop_blocks_for_free(req)
    assert not cache._boundary_readers
    assert all(b.ref_cnt > 0 for b in blocks if not b.is_null)
    cache.block_pool.free_blocks(pending)
    cache.block_pool.free_blocks(reversed(blocks))
    assert cache.block_pool.get_num_free_blocks() == 1933


@pytest.mark.parametrize("action", ["abort", "preempt"])
def test_scheduler_abort_and_preemption_release_reader(action):
    import time

    from tests.v1.core import test_boundary_admission_review as review

    scheduler, first, _ = review.scheduler_state()
    cache = scheduler.kv_cache_manager
    assert first.request_id in cache._boundary_readers
    if action == "preempt":
        scheduler.running.remove(first)
        scheduler._preempt_request(first, time.monotonic())
    else:
        scheduler.finish_requests([first.request_id], RequestStatus.FINISHED_ABORTED)
    assert first.request_id not in cache._boundary_readers
    for _, blocks in scheduler.deferred_frees:
        assert all(b.ref_cnt > 0 for b in blocks if not b.is_null)


@pytest.mark.parametrize("concurrency", [2, 4, 16])
@pytest.mark.parametrize("cap", [4096, 65536, 250000])
def test_aged_cache_keeps_explicit_warm_queue_concurrent(concurrency, cap):
    cache = manager()
    for index in range(300):
        seed(cache, str(index), lookahead=3)
    queued = [
        request(f"warm-{index}", str(index), cap=cap)
        for index in range(300 - concurrency, 300)
    ]
    active = []
    try:
        for req in queued:
            assert restore(cache, req, lookahead=3) is not None
            base.drain(cache)
            active.append(req)
        assert len(active) == concurrency
        assert len(cache._boundary_readers) == concurrency
    finally:
        for req in active:
            cache.free(req)
    assert cache.block_pool.get_num_free_blocks() == 1933
    assert not cache._boundary_readers


@pytest.mark.parametrize("pressure", [80, 95, 99])
def test_checkpoint_reuse_does_not_override_actual_capacity(pressure):
    cache = manager()
    seed(cache)
    checkpoint = seed(cache, "future")
    active = request("active")
    assert restore(cache, active) is not None
    base.drain(cache)
    queue = cache.block_pool.free_block_queue
    expendable = [
        b
        for b in queue.get_all_free_blocks()
        if not cache.boundary_checkpoints.contains_block(b.block_id)
    ]
    held = expendable[: 1933 * pressure // 100]
    cache.block_pool.touch(held)
    base.victim_at_head(cache, checkpoint)
    waiting = request("waiting")
    before = [(b.block_id, b.ref_cnt) for b in queue.get_all_free_blocks()]
    result = restore(cache, waiting)
    if pressure == 99:
        assert result is None
        assert before == [(b.block_id, b.ref_cnt) for b in queue.get_all_free_blocks()]
        assert waiting.request_id not in cache._boundary_readers
    else:
        assert result is not None
        assert checkpoint.checkpoint_id not in cache.boundary_checkpoints._entries
        base.drain(cache)
        cache.free(waiting)
    cache.block_pool.free_blocks(held)
    cache.free(active)
    assert cache.block_pool.get_num_free_blocks() == 1933
