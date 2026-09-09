# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership and scheduler-progress checks for the installed admission guard."""

from dataclasses import replace

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.boundary_checkpoint import BoundaryCheckpointCache
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def manager():
    attention = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.uint8,
        model_version="glm5_next",
    )
    recurrent = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.uint8,),
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    groups = [KVCacheGroupSpec(["attention"], attention)]
    groups += [KVCacheGroupSpec([f"recurrent-{i}"], recurrent) for i in range(3)]
    return make_kv_cache_manager(
        KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=groups),
        max_model_len=512,
        max_in_flight_tokens=32,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
        dcp_world_size=1,
        scheduler_block_size=64,
        hash_block_size=16,
        enable_boundary_checkpoints=True,
    )


def request(name, salt="a", length=140):
    result = make_request(name, list(range(length)), 16, sha256, cache_salt=salt)
    result.recurrent_instruction_boundary = 10
    result.max_tokens = result.sampling_params.max_tokens = 256
    return result


def drain(cache):
    _, copies = cache.take_kv_cache_block_copies()
    cache.block_pool.free_blocks(copies)


def seed(cache, salt):
    producer = request("producer-" + salt, salt)
    cache.get_computed_blocks(producer)
    assert cache.allocate_slots(producer, 10, num_lookahead_tokens=3) is not None
    producer.num_computed_tokens = 10
    drain(cache)
    assert cache.publish_boundary_checkpoint(producer, 10, kind="instruction")
    assert cache.allocate_slots(producer, 130, num_lookahead_tokens=3) is not None
    producer.num_computed_tokens = 140
    drain(cache)
    checkpoint = cache.publish_boundary_checkpoint(producer, 140, kind="prompt")
    producer.append_output_token_ids([900, 901, 902, 903, 904])
    assert cache.allocate_slots(producer, 4, num_lookahead_tokens=3) is not None
    producer.num_computed_tokens = 144
    producer.status = RequestStatus.FINISHED_STOPPED
    assert cache.publish_boundary_checkpoint(producer, 144, kind="response")
    cache.free(producer)
    return checkpoint


def restore(cache, req, can_defer=False):
    cache.new_step_starts()
    blocks, hit, _ = cache.get_computed_blocks(req)
    result = cache.allocate_slots(
        req,
        1,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        num_lookahead_tokens=3,
        can_defer_boundary_restore=can_defer,
    )
    if result is not None:
        req.num_computed_tokens = hit
        req.status = RequestStatus.RUNNING
    return result


def victim_at_head(cache, checkpoint):
    block = next(
        cache.block_pool.blocks[i]
        for i in checkpoint.dependencies
        if cache.block_pool.blocks[i].ref_cnt == 0
    )
    queue = cache.block_pool.free_block_queue
    queue.remove(block)
    queue.prepend_n([block])
    return block


@pytest.mark.parametrize("case", ["defer", "no_runnable", "idle", "cold"])
def test_guard_only_defers_exact_restore_with_runnable_reader(case):
    cache = manager()
    seed(cache, "a")
    other = seed(cache, "b")
    active = request("active")
    if case != "idle":
        assert restore(cache, active) is not None
        drain(cache)
    victim_at_head(cache, other)
    waiting = request("waiting", "c" if case == "cold" else "a")
    before = [b.ref_cnt for b in cache.block_pool.blocks]
    free_before = [
        b.block_id for b in cache.block_pool.free_block_queue.get_all_free_blocks()
    ]
    result = restore(cache, waiting, can_defer=case != "no_runnable")
    if case == "defer":
        assert result is None
        assert before == [b.ref_cnt for b in cache.block_pool.blocks]
        assert free_before == [
            b.block_id for b in cache.block_pool.free_block_queue.get_all_free_blocks()
        ]
        assert waiting.request_id not in cache._boundary_readers
        assert waiting.request_id not in cache._boundary_allocations
        cache.free(active)
        assert restore(cache, waiting, can_defer=True) is not None
    else:
        assert result is not None
        assert other.checkpoint_id not in cache.boundary_checkpoints._entries
    drain(cache)
    cache.free(waiting)
    if active.request_id in cache._boundary_allocations:
        cache.free(active)
    assert cache.block_pool.get_num_free_blocks() == 127


@pytest.mark.parametrize("action", ["replace", "discard"])
def test_pending_copy_pins_survive_reader_release_and_id_reuse(action):
    cache = manager()
    seed(cache, "a")
    req = request("reused-id")
    assert restore(cache, req) is not None
    drain(cache)
    assert len(req.boundary_checkpoint_blocks) == 3
    assert all(i > 0 for slot in req.boundary_checkpoint_blocks for i in slot)
    old = cache._boundary_readers[req.request_id]
    new = replace(old, checkpoint_id=cache.boundary_checkpoints.next_id())
    cache.boundary_checkpoints.stage(req, new, num_ranks=2)
    assert not cache.boundary_checkpoints.acknowledge(new.checkpoint_id, 0)
    assert new.checkpoint_id in cache.boundary_checkpoints._pending
    if action == "replace":
        assert cache.boundary_checkpoints.acknowledge(new.checkpoint_id, 1)
        assert old.checkpoint_id not in cache.boundary_checkpoints._entries
        assert cache._boundary_readers[req.request_id] is old
        assert all(cache.block_pool.blocks[i].ref_cnt > 0 for i in old.dependencies)
    cache.free(req)
    if action == "discard":
        assert all(cache.block_pool.blocks[i].ref_cnt > 0 for i in new.dependencies)
        cache.boundary_checkpoints.discard(new.checkpoint_id)
    assert cache.block_pool.get_num_free_blocks() == 127
    reused = request("reused-id")
    assert restore(cache, reused, can_defer=True) is not None
    assert len(cache._boundary_allocations[reused.request_id]) == 15
    drain(cache)
    cache.free(reused)
    assert cache.block_pool.get_num_free_blocks() == 127
    assert not cache._boundary_allocations and not cache._boundary_readers


@pytest.fixture(autouse=True)
def initialize_hash(tmp_path, monkeypatch):
    import json
    import sys
    from functools import partial

    from tests.v1.core.utils import create_scheduler

    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                architectures=["OPTForCausalLM"],
                model_type="opt",
                hidden_size=64,
                ffn_dim=256,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=2048,
                vocab_size=50272,
                word_embed_proj_dim=64,
                torch_dtype="float16",
            )
        )
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "make_scheduler",
        partial(
            create_scheduler,
            model=str(tmp_path),
            skip_tokenizer_init=True,
            device="cpu",
        ),
    )
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)


def make_scheduler(**kwargs):
    from tests.v1.core.utils import create_scheduler

    return create_scheduler(**kwargs)


@pytest.mark.parametrize("fairness", [None, 0.4])
def test_actual_scheduler_runs_decode_after_guard_defers_restore(fairness):
    from tests.v1.core.utils import create_requests

    scheduler = make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
        fairness_engine="compute_share" if fairness is not None else None,
        prefill_compute_share=fairness,
    )
    cache = scheduler.kv_cache_manager
    cache.boundary_checkpoints = BoundaryCheckpointCache(cache.block_pool)
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
    victim = victim_at_head(cache, checkpoint)
    blocked = scheduler.schedule()
    assert not blocked.boundary_logits_only
    assert blocked.num_scheduled_tokens == {"first": 4}
    assert second.status == RequestStatus.WAITING
    assert checkpoint.checkpoint_id in cache.boundary_checkpoints._entries
    queue = cache.block_pool.free_block_queue
    queue.remove(victim)
    queue.append(victim)
    admitted = scheduler.schedule()
    assert admitted.boundary_logits_only
    assert admitted.num_scheduled_tokens == {"second": 1}
    assert scheduler.max_num_running_reqs == 16


@pytest.mark.parametrize(
    "status", [RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED]
)
def test_response_checkpoint_remains_reusable_after_stop_or_length_cap(status):
    cache = manager()
    seed(cache, "a")
    req = request("response-producer")
    req.sampling_params.max_tokens = 3
    assert restore(cache, req) is not None
    drain(cache)
    req.append_output_token_ids([9010, 9011, 9012])
    assert cache.allocate_slots(req, 2, num_lookahead_tokens=3) is not None
    req.num_computed_tokens += 2
    req.status = status
    checkpoint = cache.publish_boundary_checkpoint(
        req, req.num_tokens - 1, kind="response"
    )
    assert checkpoint is not None and checkpoint.kind == "response"
    cache.free(req)
    appended = make_request(
        "appended", list(req.all_token_ids), 16, sha256, cache_salt="a"
    )
    assert cache.get_computed_blocks(appended)[1] == 142
    assert cache.block_pool.get_num_free_blocks() == 127


@pytest.mark.parametrize("can_defer", [False, True])
def test_future_working_reserve_blocks_beyond_immediate_allocation(can_defer):
    cache = manager()
    seed(cache, "a")
    checkpoint = seed(cache, "b")
    active = request("active")
    assert restore(cache, active) is not None
    drain(cache)
    waiting = request("waiting")
    cache.get_computed_blocks(waiting)
    dependencies = waiting.boundary_checkpoint.dependencies
    queue = cache.block_pool.free_block_queue
    ordered = queue.get_all_free_blocks()
    # Keep 35 expendable IDs before the first published dependency.
    expendable = [
        b
        for b in ordered
        if b.block_id not in dependencies
        and not cache.boundary_checkpoints.contains_block(b.block_id)
    ]
    victim = next(b for b in ordered if b.block_id in checkpoint.dependencies)
    assert len(expendable) >= 35
    prefix = expendable[:35] + [victim]
    for block in prefix:
        queue.remove(block)
    queue.prepend_n(prefix)
    before = [b.block_id for b in queue.get_all_free_blocks()]
    result = restore(cache, waiting, can_defer=can_defer)
    if can_defer:
        assert result is None
        assert before == [b.block_id for b in queue.get_all_free_blocks()]
        assert waiting.request_id not in cache._boundary_readers
    else:
        assert result is not None
        assert checkpoint.checkpoint_id in cache.boundary_checkpoints._entries
        cache.free(waiting)
    drain(cache)
    cache.free(active)
    assert cache.block_pool.get_num_free_blocks() == 127
