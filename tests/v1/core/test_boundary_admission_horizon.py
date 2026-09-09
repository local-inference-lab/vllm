# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-reader attention horizons, low-pressure concurrency, and release fences."""

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


def restore(cache, req, lookahead=4, defer=True):
    cache.new_step_starts()
    blocks, hit, _ = cache.get_computed_blocks(req)
    assert hit
    allocated = cache.allocate_slots(
        req,
        max(1, req.num_prompt_tokens - hit),
        hit,
        blocks,
        num_lookahead_tokens=lookahead,
        can_defer_boundary_restore=defer,
    )
    if allocated is not None:
        req.num_computed_tokens = req.num_prompt_tokens
        req.status = RequestStatus.RUNNING
    return allocated


def test_all_16_short_warm_readers_admit_without_checkpoint_loss():
    cache = manager()
    for index in range(16):
        seed(cache, str(index))
    published = set(cache.boundary_checkpoints._entries)
    active = []
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
    assert not cache._boundary_reader_horizons
    assert cache.block_pool.get_num_free_blocks() == 1933


@pytest.mark.parametrize("cap", [0, 1, 4096, 524288])
@pytest.mark.parametrize("batches", [1, 2])
@pytest.mark.parametrize("spec,lookahead", [(0, 0), (3, 3), (3, 4), (7, 8)])
def test_horizon_uses_full_prompt_output_cap_and_checked_spec_margin(
    cap, batches, spec, lookahead
):
    cache = manager(
        spec, max_concurrent_batches=batches, num_lookahead_tokens=lookahead
    )
    seed(cache, lookahead=lookahead)
    req = request("reader", length=8192 + 257, cap=cap)
    assert restore(cache, req, lookahead) is not None
    margin = max(lookahead, spec + 1)
    horizon = min(
        cache.max_model_len,
        req.num_prompt_tokens + cap + batches * (margin + 1) + margin,
    )
    assert cache._boundary_reader_horizons[req.request_id] == horizon
    # At the output limit, every optimistic batch and drafter slot must fit.
    assert horizon >= min(
        cache.max_model_len,
        req.num_prompt_tokens + cap + batches * (spec + 1) + lookahead,
    )
    base.drain(cache)
    cache.free(req)
    assert not cache._boundary_reader_horizons


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
def test_horizon_cleanup_matches_reader_release_and_id_reuse(status, deferred):
    cache = manager()
    seed(cache)
    req = request("reused")
    assert restore(cache, req) is not None
    assert req.request_id in cache._boundary_reader_horizons
    held = cache.take_kv_cache_block_copies()[1]
    req.status = status
    if deferred:
        blocks = cache.pop_blocks_for_free(req)
        assert blocks and all(b.ref_cnt > 0 for b in blocks if not b.is_null)
        assert req.request_id not in cache._boundary_reader_horizons
        assert req.request_id not in cache._boundary_readers
        cache.block_pool.free_blocks(reversed(blocks))
    else:
        cache.free(req)
    cache.block_pool.free_blocks(held)
    assert not cache._boundary_reader_horizons and not cache._boundary_readers
    reused = request("reused", cap=1)
    assert restore(cache, reused) is not None
    assert cache._boundary_reader_horizons["reused"] < 8300
    base.drain(cache)
    cache.free(reused)
    assert not cache._boundary_reader_horizons
    assert cache.block_pool.get_num_free_blocks() == 1933


def test_deferred_admission_and_failed_acquire_never_leave_a_horizon(monkeypatch):
    cache = manager()
    seed(cache)
    victim = seed(cache, "other")
    first = request("first")
    assert restore(cache, first) is not None
    base.drain(cache)
    base.victim_at_head(cache, victim)
    waiting = request("waiting")
    assert restore(cache, waiting) is None
    assert set(cache._boundary_reader_horizons) == {first.request_id}
    cache.free(first)
    monkeypatch.setattr(
        cache.boundary_checkpoints, "acquire", lambda checkpoint_id: None
    )
    assert restore(cache, waiting, defer=False) is None
    assert not cache._boundary_reader_horizons and not cache._boundary_readers


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_scheduler_binds_actual_lookahead_and_batch_count(async_scheduling):
    scheduler = base.make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        async_scheduling=async_scheduling,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
    )
    cache = scheduler.kv_cache_manager
    assert (
        cache._boundary_restore_max_concurrent_batches
        == scheduler.vllm_config.max_concurrent_batches
    )
    assert cache._boundary_restore_lookahead >= scheduler.num_lookahead_tokens
    assert scheduler.max_num_running_reqs == 16


@pytest.mark.parametrize("cap", [1, 4096])
@pytest.mark.parametrize("batches", [1, 2])
@pytest.mark.parametrize("spec,lookahead", [(3, 3), (3, 4), (7, 8)])
def test_real_allocations_fit_horizon_at_output_page_boundary(
    cap, batches, spec, lookahead
):
    cache = manager(
        spec, max_concurrent_batches=batches, num_lookahead_tokens=lookahead
    )
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
            <= (cache._boundary_reader_horizons[req.request_id] + 2047) // 2048
        )
        assert all(b.ref_cnt > 0 for b in table if not b.is_null)
    # A plain prompt-plus-cap bound would miss this extra attention page.
    assert len(table) > (prompt + cap) // 2048
    blocks = cache.pop_blocks_for_free(req)
    assert not cache._boundary_reader_horizons
    assert all(b.ref_cnt > 0 for b in blocks if not b.is_null)
    cache.block_pool.free_blocks(pending)
    cache.block_pool.free_blocks(reversed(blocks))
    assert cache.block_pool.get_num_free_blocks() == 1933


@pytest.mark.parametrize("action", ["abort", "preempt"])
def test_scheduler_abort_and_preemption_clear_reader_horizon(action):
    import time

    from tests.v1.core import test_boundary_admission_review as review

    scheduler, first, _ = review.scheduler_state()
    cache = scheduler.kv_cache_manager
    assert first.request_id in cache._boundary_reader_horizons
    if action == "preempt":
        scheduler.running.remove(first)
        scheduler._preempt_request(first, time.monotonic())
    else:
        scheduler.finish_requests([first.request_id], RequestStatus.FINISHED_ABORTED)
    assert first.request_id not in cache._boundary_readers
    assert first.request_id not in cache._boundary_reader_horizons
    for _, blocks in scheduler.deferred_frees:
        assert all(b.ref_cnt > 0 for b in blocks if not b.is_null)
