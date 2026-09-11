# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent CPU probes of reserve geometry and scheduler behavior."""

import time

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


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
@pytest.mark.parametrize("tail", [0, 1, 252, 253, 254, 255])
@pytest.mark.parametrize("checkpoints", [0, 1])
def test_recurrent_reserve_across_acceptance_and_block_edges(
    accepted, tail, checkpoints
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
        KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=4096,
        max_in_flight_tokens=8192,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
        dcp_world_size=1,
        scheduler_block_size=2048,
        hash_block_size=256,
        enable_boundary_checkpoints=True,
    )
    producer = base.request("producer", length=512 + tail)
    cache.get_computed_blocks(producer)
    assert (
        cache.allocate_slots(producer, producer.num_tokens, num_lookahead_tokens=3)
        is not None
    )
    producer.num_computed_tokens = producer.num_tokens
    base.drain(cache)
    assert cache.publish_boundary_checkpoint(
        producer, producer.num_tokens, kind="prompt"
    )
    cache.free(producer)
    req = base.request("reader", length=512 + tail)
    assert base.restore(cache, req) is not None
    req.append_output_token_ids([10000])
    pending = 0
    held = cache.take_kv_cache_block_copies()[1]
    for step in range(600):
        cache.new_step_starts()
        assert cache.allocate_slots(req, 4, num_lookahead_tokens=3) is not None
        req.num_computed_tokens += 4
        req.num_in_flight_tokens += 4
        new_held = cache.take_kv_cache_block_copies()[1]
        for manager in cache.coordinator.single_type_managers[1:]:
            blocks = manager.req_to_blocks[req.request_id]
            assert sum(not b.is_null for b in blocks) <= 6
            assert all(b.is_null or b.ref_cnt > 0 for b in blocks)
        if pending:
            req.num_in_flight_tokens -= 4
            req.num_computed_tokens -= 4 - accepted
            req.append_output_token_ids([10000 + step] * accepted)
        cache.block_pool.free_blocks(held)
        held = new_held
        pending = 4
        cache.cache_blocks(req, min(req.num_computed_tokens, req.num_tokens))
    cache.free(req)
    cache.block_pool.free_blocks(held)
    assert cache.block_pool.get_num_free_blocks() == 127
    assert not cache._boundary_readers


def scheduler_state():
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
    producer, first, second = create_requests(
        num_requests=3,
        num_tokens=32,
        same_prompt=True,
        req_ids=["producer", "first", "second"],
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
    scheduler.add_request(second)
    assert scheduler.schedule().num_scheduled_tokens == {"first": 4}
    base.queue_matching(scheduler, unrelated)
    base.victim_at_head(cache, checkpoint)
    return scheduler, first, second


@pytest.mark.parametrize(
    "scope",
    [
        "supported",
        "dcp2",
        "pp2",
        "three_batches",
        "all_mode",
        "two_checkpoints",
        "wide_spec",
    ],
)
def test_actual_caller_scope(scope, monkeypatch):
    scheduler, first, second = scheduler_state()
    cache = scheduler.kv_cache_manager
    if scope == "dcp2":
        monkeypatch.setattr(
            scheduler.parallel_config, "decode_context_parallel_size", 2
        )
    if scope == "pp2":
        monkeypatch.setattr(scheduler, "use_pp", True)
    if scope == "three_batches":
        monkeypatch.setattr(
            type(scheduler.vllm_config), "max_concurrent_batches", property(lambda _: 3)
        )
    if scope in ("all_mode", "two_checkpoints", "wide_spec"):
        manager = cache.coordinator.single_type_managers[0]
        monkeypatch.setattr(
            manager,
            "kv_cache_spec",
            MambaSpec(
                block_size=256,
                shapes=((1,),),
                dtypes=(torch.uint8,),
                mamba_cache_mode="all" if scope == "all_mode" else "align",
                num_speculative_blocks=3,
                num_prefill_checkpoint_blocks=2 if scope == "two_checkpoints" else 0,
            ),
        )
        monkeypatch.setattr(manager, "block_size", 4 if scope == "wide_spec" else 256)
    calls = []

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        calls.append(kwargs["can_defer_boundary_restore"])
        raise Captured

    monkeypatch.setattr(cache, "allocate_slots", capture)
    with pytest.raises(Captured):
        scheduler.schedule()
    assert calls == [scope == "supported"]


def test_deferred_head_blocks_later_cold_request():
    from tests.v1.core.utils import create_requests

    scheduler, first, second = scheduler_state()
    (cold,) = create_requests(num_requests=1, num_tokens=24, req_ids=["later-cold"])
    scheduler.add_request(cold)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"first": 4}
    assert second.status == cold.status == RequestStatus.WAITING
    assert scheduler.max_num_running_reqs == 16


def test_coarse_runnable_can_return_empty_while_output_is_pending():
    scheduler, first, second = scheduler_state()
    first.num_output_placeholders = 4
    first.num_computed_tokens = first.num_prompt_tokens + first.max_tokens + 2
    assert scheduler._request_is_runnable_decode(
        first, scheduling_step=scheduler.current_step + 1
    )
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert second.status == RequestStatus.WAITING


def test_preemption_removes_reader_bookkeeping():
    scheduler, first, second = scheduler_state()
    cache = scheduler.kv_cache_manager
    scheduler.running.remove(first)
    scheduler._preempt_request(first, time.monotonic())
    assert first.request_id not in cache._boundary_readers
    assert first.request_id not in cache._boundary_allocations
    assert not cache._boundary_readers


@pytest.mark.parametrize("case", ["supported", "connector", "idle", "no_progress"])
def test_scheduler_deferral_requires_local_progress(case, monkeypatch):
    from unittest.mock import Mock

    scheduler, first, second = scheduler_state()
    cache = scheduler.kv_cache_manager
    if case == "connector":
        connector = Mock()
        connector.get_num_new_matched_tokens.return_value = (0, False)
        connector.build_connector_meta.return_value = None
        connector.boundary_checkpoint_external_tokens.return_value = 0
        monkeypatch.setattr(scheduler, "connector", connector)
    elif case == "idle":
        scheduler.finish_requests([first.request_id], RequestStatus.FINISHED_ABORTED)
    elif case == "no_progress":
        first.next_decode_eligible_step = scheduler.current_step + 100

    output = scheduler.schedule()
    if case == "supported":
        assert output.num_scheduled_tokens == {"first": 4}
        assert second.status == RequestStatus.WAITING
        assert second.request_id not in cache._boundary_readers
    else:
        assert output.boundary_logits_only
        assert output.num_scheduled_tokens == {"second": 1}
        assert second.request_id in cache._boundary_readers
    assert scheduler.max_num_running_reqs == 16
