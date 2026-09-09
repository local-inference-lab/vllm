# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Continuation admission, scheduler fallback, and recurrent lifetime checks."""

import pytest
import torch

from tests.v1.core import test_boundary_admission as base
from tests.v1.core.test_boundary_admission import initialize_hash as initialize_hash
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("case", ["defer", "idle", "no_runnable", "cold", "external"])
@pytest.mark.parametrize("length", [166, 272, 480])
def test_continuation_guard_bypasses_only_without_local_reader_progress(case, length):
    cache = base.manager()
    base.seed(cache, "a")
    other = base.seed(cache, "b")
    active = base.request("active")
    if case != "idle":
        assert base.restore(cache, active) is not None
        base.drain(cache)
    base.victim_at_head(cache, other)
    req = base.request("continuation", "c" if case == "cold" else "a", length)
    cached, hit, _ = cache.get_computed_blocks(req)
    before = [
        (b.block_id, b.ref_cnt)
        for b in cache.block_pool.free_block_queue.get_all_free_blocks()
    ]
    blocks = cache.allocate_slots(
        req,
        min(32, length - hit),
        hit,
        cached,
        num_external_computed_tokens=1 if case == "external" else 0,
        num_lookahead_tokens=3,
        can_defer_boundary_restore=case != "no_runnable",
    )
    if case == "defer":
        assert hit == 140
        assert blocks is None
        assert before == [
            (b.block_id, b.ref_cnt)
            for b in cache.block_pool.free_block_queue.get_all_free_blocks()
        ]
        assert req.request_id not in cache._boundary_readers
        assert req.request_id not in cache._boundary_allocations
    else:
        assert blocks is not None
        base.drain(cache)
        cache.free(req)
    if case != "idle":
        cache.free(active)
    assert cache.block_pool.get_num_free_blocks() == 127


@pytest.mark.parametrize("fairness", [None, 0.4])
@pytest.mark.parametrize("length", [58, 256])
def test_nonisolated_continuation_deferral_serves_existing_decoder(
    fairness, length, monkeypatch
):
    from tests.v1.core.utils import create_requests

    scheduler = base.make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
        fairness_engine="compute_share" if fairness is not None else None,
        prefill_compute_share=fairness,
    )
    cache = scheduler.kv_cache_manager
    cache.boundary_checkpoints = base.BoundaryCheckpointCache(cache.block_pool)
    producer, first = create_requests(
        num_requests=2, num_tokens=32, same_prompt=True, req_ids=["producer", "first"]
    )
    (unrelated,) = create_requests(num_requests=1, num_tokens=48, req_ids=["unrelated"])
    for req in (producer, unrelated):
        cache.get_computed_blocks(req)
        assert cache.allocate_slots(req, req.num_prompt_tokens) is not None
        checkpoint = cache.publish_boundary_checkpoint(
            req, req.num_prompt_tokens, kind="prompt"
        )
        cache.free(req)
    scheduler.add_request(first)
    assert scheduler.schedule().boundary_logits_only
    assert scheduler.schedule().num_scheduled_tokens == {"first": 4}
    continuation = base.make_request(
        "continuation",
        list(producer.prompt_token_ids) + [1234] * (length - 32),
        scheduler.block_size,
        base.sha256,
    )
    scheduler.add_request(continuation)
    first.is_prefill_chunk = False
    if scheduler.compute_share_controller is not None:
        # Force the prefill turn whose empty admission must fall back to decode.
        monkeypatch.setattr(
            scheduler.compute_share_controller, "select", lambda **kwargs: "prefill"
        )
    base.victim_at_head(cache, checkpoint)
    assert not scheduler._has_waiting_boundary_logits()
    output = scheduler.schedule()
    assert not output.boundary_logits_only
    assert output.num_scheduled_tokens == {"first": 4}
    assert continuation.status == RequestStatus.WAITING
    assert continuation.request_id not in cache._boundary_readers
    assert checkpoint.checkpoint_id in cache.boundary_checkpoints._entries
    assert scheduler.max_num_running_reqs == 16


@pytest.mark.parametrize("tail", [0, 1, 252, 255])
@pytest.mark.parametrize("append", [26, 257, 8193, 16385])
@pytest.mark.parametrize("inflight", [1, 2])
@pytest.mark.parametrize("checkpoints", [0, 1])
def test_continuation_prefill_source_and_two_batch_state_bound(
    tail, append, inflight, checkpoints
):
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
        num_prefill_checkpoint_blocks=checkpoints,
    )
    groups = [KVCacheGroupSpec(["attention"], attention)]
    groups += [KVCacheGroupSpec([f"recurrent-{i}"], recurrent) for i in range(3)]
    cache = base.make_kv_cache_manager(
        KVCacheConfig(num_blocks=256, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=32768,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
        dcp_world_size=1,
        scheduler_block_size=2048,
        hash_block_size=256,
        enable_boundary_checkpoints=True,
    )
    prefix = 2048 + tail
    producer = base.make_request("producer", list(range(prefix)), 256, base.sha256)
    cache.get_computed_blocks(producer)
    assert cache.allocate_slots(producer, prefix, num_lookahead_tokens=3) is not None
    producer.num_computed_tokens = prefix
    base.drain(cache)
    checkpoint = cache.publish_boundary_checkpoint(producer, prefix, kind="prompt")
    cache.free(producer)
    req = base.make_request(
        "continuation", list(range(prefix + append)), 256, base.sha256
    )
    cached, hit, _ = cache.get_computed_blocks(req)
    assert hit == prefix
    pending = []
    while req.num_computed_tokens < req.num_prompt_tokens:
        cache.new_step_starts()
        start = req.num_computed_tokens or hit
        count = min(4096, req.num_prompt_tokens - start)
        protected = []
        if req.num_computed_tokens:
            processed = req.num_computed_tokens - req.num_in_flight_tokens
            for manager in cache.coordinator.single_type_managers[1:]:
                protected.extend(
                    b
                    for b in manager.req_to_blocks[req.request_id][
                        max(0, (processed - 1) // 256) :
                    ]
                    if not b.is_null
                )
        blocks = cache.allocate_slots(
            req,
            count,
            hit if not req.num_computed_tokens else 0,
            cached if not req.num_computed_tokens else None,
            num_lookahead_tokens=3,
            can_defer_boundary_restore=True,
        )
        assert blocks is not None
        assert all(b.ref_cnt > 0 for b in protected)
        req.num_computed_tokens = start + count
        req.num_in_flight_tokens += count
        req.status = RequestStatus.RUNNING
        held = cache.take_kv_cache_block_copies()[1]
        pending.append((count, held))
        for manager in cache.coordinator.single_type_managers[1:]:
            table = manager.req_to_blocks[req.request_id]
            assert (
                sum(not b.is_null for b in table)
                <= recurrent.num_speculative_blocks + 3
            )
            assert all(b.is_null or b.ref_cnt > 0 for b in table)
        assert all(
            cache.block_pool.blocks[i].ref_cnt > 0 for i in checkpoint.dependencies
        )
        if len(pending) >= inflight:
            count, held = pending.pop(0)
            req.num_in_flight_tokens -= count
            cache.block_pool.free_blocks(held)
    for count, held in pending:
        req.num_in_flight_tokens -= count
        cache.block_pool.free_blocks(held)
    assert req.num_in_flight_tokens == 0
    assert cache.publish_boundary_checkpoint(req, req.num_prompt_tokens, kind="prompt")
    cache.free(req)
    assert cache.block_pool.get_num_free_blocks() == 255
    assert not cache._boundary_readers and not cache._boundary_allocations


@pytest.mark.parametrize("length", [166, 480])
@pytest.mark.parametrize("can_defer", [False, True])
def test_continuation_reserve_includes_cached_source_and_future_growth(
    length, can_defer
):
    cache = base.manager()
    base.seed(cache, "a")
    checkpoint = base.seed(cache, "b")
    active = base.request("active")
    assert base.restore(cache, active) is not None
    base.drain(cache)
    req = base.request("continuation", length=length)
    cached, hit, _ = cache.get_computed_blocks(req)
    dependencies = req.boundary_checkpoint.dependencies
    queue = cache.block_pool.free_block_queue
    ordered = queue.get_all_free_blocks()
    expendable = [
        b
        for b in ordered
        if b.block_id not in dependencies
        and not cache.boundary_checkpoints.contains_block(b.block_id)
    ]
    victim = next(b for b in ordered if b.block_id in checkpoint.dependencies)
    assert len(expendable) >= 40
    prefix = expendable[:40] + [victim]
    for block in prefix:
        queue.remove(block)
    queue.prepend_n(prefix)
    before = [b.block_id for b in queue.get_all_free_blocks()]
    result = cache.allocate_slots(
        req,
        length - hit,
        hit,
        cached,
        num_lookahead_tokens=3,
        can_defer_boundary_restore=can_defer,
    )
    if can_defer:
        assert result is None
        assert before == [b.block_id for b in queue.get_all_free_blocks()]
        assert req.request_id not in cache._boundary_readers
    else:
        assert result is not None
        assert checkpoint.checkpoint_id in cache.boundary_checkpoints._entries
        base.drain(cache)
        cache.free(req)
    cache.free(active)
    assert cache.block_pool.get_num_free_blocks() == 127


@pytest.mark.parametrize("length", [58, 512])
def test_same_pass_continuations_use_already_scheduled_prefill_progress(length):
    from tests.v1.core.utils import create_requests

    scheduler = base.make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
    )
    cache = scheduler.kv_cache_manager
    cache.boundary_checkpoints = base.BoundaryCheckpointCache(cache.block_pool)
    (producer,) = create_requests(num_requests=1, num_tokens=32, req_ids=["producer"])
    (unrelated,) = create_requests(num_requests=1, num_tokens=48, req_ids=["unrelated"])
    for req in (producer, unrelated):
        cache.get_computed_blocks(req)
        assert cache.allocate_slots(req, req.num_prompt_tokens) is not None
        checkpoint = cache.publish_boundary_checkpoint(
            req, req.num_prompt_tokens, kind="prompt"
        )
        cache.free(req)
    queue = cache.block_pool.free_block_queue
    ordered = queue.get_all_free_blocks()
    expendable = [
        b for b in ordered if not cache.boundary_checkpoints.contains_block(b.block_id)
    ]
    victim = next(b for b in ordered if b.block_id in checkpoint.dependencies)
    prefix = expendable[:64] + [victim]
    assert len(prefix) == 65
    for block in prefix:
        queue.remove(block)
    queue.prepend_n(prefix)
    first, second = [
        base.make_request(
            name,
            list(producer.prompt_token_ids) + [1234] * (length - 32),
            scheduler.block_size,
            base.sha256,
        )
        for name in ["first", "second"]
    ]
    first.max_tokens = second.max_tokens = scheduler.max_model_len
    scheduler.add_request(first)
    scheduler.add_request(second)
    assert not scheduler.running
    output = scheduler.schedule()
    assert not output.boundary_logits_only
    assert output.num_scheduled_tokens == {"first": length - 32}
    assert second.status == RequestStatus.WAITING
    assert first.request_id in cache._boundary_readers
    assert second.request_id not in cache._boundary_readers
    assert checkpoint.checkpoint_id in cache.boundary_checkpoints._entries
    assert scheduler.max_num_running_reqs == 16
