# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU ownership and scheduler-progress checks for the installed admission guard."""

import inspect
from dataclasses import replace
from typing import Any

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.boundary_checkpoint import (
    BoundaryCheckpointCache,
    boundary_checkpoint_slots,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def compute_share_fixture_options(share: float | None) -> dict[str, Any]:
    """Select the fixture's supported fairness API and exercise its controller.

    JJ exposes an explicit engine selector; the compute-sharing PR makes a
    configured share sufficient. Both interfaces must test enabled fairness,
    rather than silently dropping the coverage in the composed tree.
    """
    from tests.v1.core.utils import create_scheduler

    options: dict[str, Any] = {"prefill_compute_share": share}
    if "fairness_engine" in inspect.signature(create_scheduler).parameters:
        options["fairness_engine"] = "compute_share" if share is not None else None
    return options


def manager(dcp=1, num_blocks=128):
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
        KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        max_model_len=512,
        max_in_flight_tokens=32,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=1,
        dcp_world_size=dcp,
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


def reserve_import(cache, consumer, prefix=100):
    return cache.reserve_external_boundary_checkpoint(
        consumer,
        prefix,
        cache.boundary_checkpoint_page_positions(prefix),
        draft_prefix_len=prefix,
        kind="prompt",
        num_ranks=2,
        reserve_admission=True,
    )


@pytest.mark.parametrize("dcp", [1, 2])
@pytest.mark.parametrize("prefix", [16, 100, 128])
@pytest.mark.parametrize("chunk", [1, 16, 33, 127])
def test_external_import_owns_execution_capacity_through_chunked_prefill(
    dcp, prefix, chunk
):
    """Other admissions cannot consume a restored request's remaining suffix."""
    cache = manager(dcp)
    consumer = request("import", length=400)
    free = cache.block_pool.get_num_free_blocks()
    checkpoint = reserve_import(cache, consumer, prefix=prefix)
    assert checkpoint is not None
    credits = cache.external_boundary_reserved_blocks()
    assert credits > 0
    assert not cache.acknowledge_external_boundary_checkpoint(
        checkpoint.checkpoint_id, 0
    )
    assert cache.acknowledge_external_boundary_checkpoint(checkpoint.checkpoint_id, 1)
    assert not cache.reset_prefix_cache()
    pressure = cache.block_pool.get_new_blocks(
        cache.block_pool.get_num_free_blocks() - credits
    )
    assert cache.allocate_slots(request("competitor", "other"), 1) is None
    blocks, hits, _ = cache.get_computed_blocks(consumer)
    assert hits == prefix
    assert (
        cache.allocate_slots(
            consumer,
            chunk,
            num_new_computed_tokens=prefix,
            new_computed_blocks=blocks,
            num_lookahead_tokens=3,
            full_sequence_must_fit=True,
        )
        is not None
    )
    consumer.num_computed_tokens = prefix + chunk
    drain(cache)
    assert cache.has_external_boundary_admission(consumer.request_id)
    assert cache.external_boundary_reserved_blocks() > 0
    while consumer.num_computed_tokens < consumer.num_tokens:
        count = min(chunk, consumer.num_tokens - consumer.num_computed_tokens)
        assert cache.allocate_slots(consumer, count, num_lookahead_tokens=3) is not None
        consumer.num_computed_tokens += count
        drain(cache)
    assert not cache.has_external_boundary_admission(consumer.request_id)
    cache.free(consumer)
    cache.block_pool.free_blocks(pressure)
    assert cache.block_pool.get_num_free_blocks() == free


@pytest.mark.parametrize(
    "attention_kind,retained", [("sliding", 0), ("sliding", 48), ("chunked", 0)]
)
@pytest.mark.parametrize(
    "prefix,length", [(128, 400), (128, 128), (128, 129), (100, 100), (100, 101)]
)
def test_scheduler_preserves_recycling_import_capacity(
    attention_kind, retained, prefix, length
):
    """Null positions must not release credits while the live window moves."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import ModelRunnerOutput

    template = make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        max_num_batched_tokens=32,
        long_prefill_token_threshold=16,
        block_size=64,
        max_model_len=512,
    )
    config = template.vllm_config
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.num_gpu_blocks = 64
    attention_args = dict(block_size=64, num_kv_heads=1, head_size=1, dtype=torch.uint8)
    attention = (
        SlidingWindowSpec(
            **attention_args, sliding_window=128, extra_retained_tokens=retained
        )
        if attention_kind == "sliding"
        else ChunkedLocalAttentionSpec(**attention_args, attention_chunk_size=128)
    )
    recurrent = MambaSpec(
        block_size=16, shapes=((1,),), dtypes=(torch.uint8,), mamba_cache_mode="align"
    )
    scheduler = Scheduler(
        vllm_config=config,
        kv_cache_config=KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["attention"], attention),
                KVCacheGroupSpec(["recurrent"], recurrent),
            ],
        ),
        structured_output_manager=template.structured_output_manager,
        block_size=64,
        hash_block_size=16,
    )
    scheduler.use_v2_model_runner = True
    cache = scheduler.kv_cache_manager
    cache.boundary_checkpoints = BoundaryCheckpointCache(cache.block_pool)
    consumer = request("recycling-import", length=length)
    scheduler.add_request(consumer)
    checkpoint = reserve_import(cache, consumer, prefix=prefix)
    assert checkpoint is not None
    if length <= prefix + 1:
        # One attention page and two private recurrent states suffice for
        # this suffix; imported read-only history need not be replaced.
        assert cache.external_boundary_reserved_blocks() <= (
            3 + 3 * len(boundary_checkpoint_slots(consumer))
        )
    for rank in range(2):
        cache.acknowledge_external_boundary_checkpoint(checkpoint.checkpoint_id, rank)
    pressure = []
    while consumer.num_computed_tokens < consumer.num_tokens:
        spare = (
            cache.block_pool.get_num_free_blocks()
            - cache.external_boundary_reserved_blocks()
        )
        pressure.extend(cache.block_pool.get_new_blocks(max(0, spare)))
        expected = min(16, max(1, length - max(prefix, consumer.num_computed_tokens)))
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {consumer.request_id: expected}
        assert consumer.num_preemptions == 0
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[consumer.request_id],
                req_id_to_index={consumer.request_id: 0},
                sampled_token_ids=[[]],
            ),
        )
    cache.free(consumer)
    cache.block_pool.free_blocks(pressure)
    assert cache.external_boundary_reserved_blocks() == 0
    assert cache.block_pool.get_num_free_blocks() == 63


def publish(cache, checkpoint):
    for rank in range(2):
        cache.acknowledge_external_boundary_checkpoint(checkpoint.checkpoint_id, rank)


@pytest.mark.parametrize("cancel_at", ["copying", "ready"])
def test_external_import_cancellation_releases_only_pages_after_copy_drain(cancel_at):
    """Cancellation frees the slot and credits at once; pages wait for copies."""
    cache = manager()
    cache.set_external_boundary_admission_context(1, 0)
    consumer = request("cancel")
    free = cache.block_pool.get_num_free_blocks()
    checkpoint = reserve_import(cache, consumer)
    assert checkpoint is not None
    assert not cache.can_admit_external_boundary_request("other")
    if cancel_at == "ready":
        publish(cache, checkpoint)
    cache.free(consumer)
    assert cache.external_boundary_reserved_blocks() == 0
    assert cache.can_admit_external_boundary_request("other")
    if cancel_at == "copying":
        assert cache.block_pool.get_num_free_blocks() < free
        cache.discard_external_boundary_checkpoint(checkpoint.checkpoint_id)
    # The connector releases again once the cancelled copy has drained.
    cache.release_external_boundary_admission(consumer.request_id)
    assert cache.external_boundary_reserved_blocks() == 0
    assert cache.block_pool.get_num_free_blocks() == free


def test_external_imports_reserve_slots_without_serializing_available_imports():
    cache = manager()
    cache.set_external_boundary_admission_context(2, 0)
    first, second, third = [request(name, name) for name in ("one", "two", "three")]
    assert reserve_import(cache, first) is not None
    assert reserve_import(cache, second) is not None
    free = cache.block_pool.get_num_free_blocks()
    assert reserve_import(cache, third) is None
    assert not cache.can_admit_external_boundary_request(third.request_id)
    assert cache.block_pool.get_num_free_blocks() == free
    assert cache.can_admit_external_boundary_request(first.request_id)


def test_external_import_preflight_does_not_evict_cache_on_execution_pressure():
    cache = manager()
    consumer = request("consumer")
    cache.set_external_boundary_admission_context(
        1, cache.block_pool.get_num_free_blocks()
    )
    free = cache.block_pool.get_num_free_blocks()
    assert reserve_import(cache, consumer) is None
    assert cache.block_pool.get_num_free_blocks() == free
    assert cache.external_boundary_reserved_blocks() == 0


def refused_waiter_need(cache, waiter):
    """Refuse a restore on a full pool and return the space it waits for."""
    probe = manager()
    free = probe.block_pool.get_num_free_blocks()
    assert reserve_import(probe, waiter) is not None
    need = free - probe.block_pool.get_num_free_blocks()
    need += probe.external_boundary_reserved_blocks()
    pressure = cache.block_pool.get_new_blocks(cache.block_pool.get_num_free_blocks())
    assert reserve_import(cache, waiter) is None
    return pressure, need


@pytest.mark.parametrize("released", ["waiter_only", "both"])
def test_capacity_waiter_holds_back_only_admissions_that_would_take_its_space(
    released,
):
    cache = manager()
    older, newer = request("older"), request("newer", "newer")
    pressure, need = refused_waiter_need(cache, older)
    cache.block_pool.free_blocks(pressure[: need * (2 if released == "both" else 1)])
    assert cache.can_admit_external_boundary_request(newer.request_id)
    # A later restore may use only the space beyond the waiter's need.
    assert (reserve_import(cache, newer) is not None) == (released == "both")
    assert reserve_import(cache, older) is not None


def test_capacity_waiter_does_not_hold_back_requests_considered_before_it():
    cache = manager()
    older, newer = request("older"), request("newer", "newer")
    pressure, need = refused_waiter_need(cache, older)
    cache.new_step_starts()
    cache.block_pool.free_blocks(pressure[:need])
    assert reserve_import(cache, newer) is not None
    assert reserve_import(cache, older) is None


def test_cancelled_capacity_waiter_releases_admission_barrier():
    cache = manager()
    consumer, later = request("cancel-waiter"), request("later", "later")
    pressure, need = refused_waiter_need(cache, consumer)
    cache.block_pool.free_blocks(pressure[:need])
    cache.release_external_boundary_admission(consumer.request_id)
    assert reserve_import(cache, later) is not None


def bounded_wait_clock(monkeypatch, max_wait):
    """Drive restore waits from a fake clock and capture the manager's warnings."""
    import time

    from vllm.v1.core import kv_cache_manager

    clock = [100.0]
    warnings = []
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        kv_cache_manager.logger, "warning", lambda *args: warnings.append(args)
    )
    monkeypatch.setenv("VLLM_CHECKPOINT_RESTORE_MAX_WAIT_S", str(max_wait))
    return clock, warnings


def test_capacity_waiter_stops_waiting_after_bounded_wait(monkeypatch):
    clock, warnings = bounded_wait_clock(monkeypatch, 30)
    cache = manager()
    waiter, later = request("waiter"), request("later", "later")
    pressure, need = refused_waiter_need(cache, waiter)
    clock[0] += 29.0
    assert reserve_import(cache, waiter) is None
    assert not warnings and not cache.external_boundary_wait_expired("waiter")
    clock[0] += 1.5
    assert reserve_import(cache, waiter) is None
    assert cache.external_boundary_wait_expired(waiter.request_id)
    assert len(warnings) == 1 and warnings[0][2] == pytest.approx(30.5)
    # The request recomputes: it neither restores nor holds later requests back.
    cache.block_pool.free_blocks(pressure[:need])
    assert reserve_import(cache, waiter) is None
    assert reserve_import(cache, later) is not None
    assert len(warnings) == 1
    cache.release_external_boundary_admission(waiter.request_id)
    assert not cache.external_boundary_wait_expired(waiter.request_id)


def test_import_preflight_preserves_watermark_and_classifies_impossible_restore():
    cache = manager()
    consumer = request("watermark")
    free = cache.block_pool.get_num_free_blocks()
    cache.watermark_blocks = free
    cache.set_external_boundary_admission_context(1, 0, has_scheduled_reqs=True)
    assert reserve_import(cache, consumer) is None
    assert cache.block_pool.get_num_free_blocks() == free
    # The last runnable request finishing removes the normal admission watermark.
    cache.set_external_boundary_admission_context(1, 0, has_scheduled_reqs=False)
    assert reserve_import(cache, consumer) is not None
    cache = manager(num_blocks=16)
    free = cache.block_pool.get_num_free_blocks()
    with pytest.raises(ValueError, match="exceed the GPU pool"):
        reserve_import(cache, consumer)
    assert cache.block_pool.get_num_free_blocks() == free
    assert not cache.has_pending_external_boundary_admissions()


def test_eight_external_sessions_rotate_without_cold_prefill_or_leaked_blocks():
    """Aggregate checkpoints exceed the pool; each admitted turn still restores."""
    cache = manager(dcp=2, num_blocks=64)
    free = cache.block_pool.get_num_free_blocks()
    consumers = [request(f"session-{i}", str(i), length=400) for i in range(8)]
    pages = sum(map(len, cache.boundary_checkpoint_page_positions(400))) + 1
    assert pages * len(consumers) > free
    remaining = list(consumers)
    completed = []
    while remaining:
        admitted = []
        cache.set_external_boundary_admission_context(8, 0)
        for consumer in remaining:
            checkpoint = reserve_import(cache, consumer, prefix=400)
            if checkpoint is not None:
                admitted.append((consumer, checkpoint))
        assert admitted, "imports must make progress when previous owners drain"
        for consumer, checkpoint in admitted:
            for rank in range(2):
                cache.acknowledge_external_boundary_checkpoint(
                    checkpoint.checkpoint_id, rank
                )
            blocks, hits, _ = cache.get_computed_blocks(consumer)
            assert hits == consumer.num_tokens
            assert (
                cache.allocate_slots(
                    consumer,
                    1,
                    num_new_computed_tokens=hits,
                    new_computed_blocks=blocks,
                    num_lookahead_tokens=3,
                    full_sequence_must_fit=True,
                )
                is not None
            )
            drain(cache)
            cache.free(consumer)
            remaining.remove(consumer)
            completed.append(consumer.request_id)
    assert completed == [consumer.request_id for consumer in consumers]
    assert cache.block_pool.get_num_free_blocks() == free
    assert cache.external_boundary_reserved_blocks() == 0


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


def pending_kwargs(cache, requests=()):
    """Keep baseline execution compatible with the queued-request API."""
    parameters = inspect.signature(cache.allocate_slots).parameters
    if "pending_boundary_requests" in parameters:
        return {"pending_boundary_requests": tuple(requests)}
    return {}


def queue_matching(scheduler, producer):
    """Queue a future consumer of the checkpoint used by pressure fixtures."""
    queued = make_request(
        "queued-" + producer.request_id,
        list(producer.prompt_token_ids),
        scheduler.block_size,
        sha256,
    )
    scheduler.add_request(queued)
    return queued


def restore(cache, req, can_defer=False, pending=()):
    cache.new_step_starts()
    blocks, hit, _ = cache.get_computed_blocks(req)
    result = cache.allocate_slots(
        req,
        1,
        num_new_computed_tokens=hit,
        new_computed_blocks=blocks,
        num_lookahead_tokens=3,
        can_defer_boundary_restore=can_defer,
        **pending_kwargs(cache, pending),
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
    result = restore(
        cache,
        waiting,
        can_defer=case != "no_runnable",
        pending=[request("queued-b", "b")],
    )
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


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("lanes", [1, 2])
@pytest.mark.parametrize("limit", ["slots", "blocks"])
def test_ready_import_behind_blocked_head_gets_saved_logits_step(policy, lanes, limit):
    """Reserved imports remain reachable while unrelated requests wait."""
    from unittest.mock import Mock

    from tests.v1.core.utils import create_requests

    scheduler = make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        async_scheduling=True,
        max_num_seqs=2 if limit == "slots" else 3,
        max_parallel_prefills=lanes,
        scheduling_policy=policy,
    )
    cache = scheduler.kv_cache_manager
    cache.boundary_checkpoints = BoundaryCheckpointCache(cache.block_pool)
    running, consumer = create_requests(
        num_requests=2,
        num_tokens=32,
        req_ids=["running", "consumer"],
    )
    (cold,) = create_requests(num_requests=1, num_tokens=160, req_ids=["cold"])
    scheduler.add_request(running)
    assert scheduler.schedule().num_scheduled_tokens == {"running": 32}
    checkpoint = cache.reserve_external_boundary_checkpoint(
        consumer,
        32,
        cache.boundary_checkpoint_page_positions(32),
        draft_prefix_len=32,
        kind="prompt",
        num_ranks=2,
        reserve_admission=True,
    )
    assert checkpoint is not None
    for rank in range(2):
        cache.acknowledge_external_boundary_checkpoint(checkpoint.checkpoint_id, rank)
    pressure = []
    if limit == "blocks":
        pressure = cache.block_pool.get_new_blocks(
            cache.block_pool.get_num_free_blocks()
            - cache.external_boundary_reserved_blocks()
        )
    scheduler.connector = Mock()
    scheduler.connector.poll_boundary_checkpoint.return_value = True
    scheduler.connector.boundary_checkpoint_external_tokens.side_effect = lambda req: (
        32 if req is consumer else 0
    )
    scheduler.connector.get_num_new_matched_tokens.return_value = (0, False)
    scheduler.add_request(cold)
    scheduler.add_request(consumer)
    output = scheduler.schedule()
    assert output.boundary_logits_only
    assert output.num_scheduled_tokens == {"consumer": 1}
    assert consumer.prefill_stats.num_computed_tokens == 0
    assert consumer.prefill_stats.num_external_cached_tokens == 32
    assert cold.status == RequestStatus.WAITING
    cache.block_pool.free_blocks(pressure)


def test_streaming_owner_can_resume_ahead_of_capacity_waiter():
    from unittest.mock import Mock

    from tests.v1.core.utils import create_requests

    scheduler = make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        max_num_seqs=8,
        num_blocks=12,
    )
    cache = scheduler.kv_cache_manager
    cache.boundary_checkpoints = BoundaryCheckpointCache(cache.block_pool)
    (session,) = create_requests(num_requests=1, num_tokens=100, req_ids=["stream"])
    session.resumable = True
    scheduler.add_request(session)
    assert scheduler.schedule().num_scheduled_tokens == {"stream": 100}
    session.num_in_flight_tokens = 0
    scheduler.running.remove(session)
    assert not scheduler._handle_stopped_request(session)
    assert session.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    (consumer,) = create_requests(num_requests=1, num_tokens=64, req_ids=["restore"])
    connector = Mock()

    def poll(req):
        if req is not consumer:
            return True
        restored = cache.reserve_external_boundary_checkpoint(
            req,
            64,
            cache.boundary_checkpoint_page_positions(64),
            draft_prefix_len=64,
            kind="prompt",
            num_ranks=2,
            reserve_admission=True,
        )
        assert restored is None
        return False

    connector.poll_boundary_checkpoint.side_effect = poll
    connector.boundary_checkpoint_external_tokens.return_value = 0
    connector.get_num_new_matched_tokens.return_value = (0, False)
    scheduler.connector = connector
    scheduler.add_request(consumer)
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert consumer.request_id in cache._boundary_import_waiters
    (update,) = create_requests(num_requests=1, num_tokens=1, req_ids=["stream"])
    update.resumable = True
    scheduler.add_request(update)
    assert session.status == RequestStatus.WAITING
    assert scheduler.schedule().num_scheduled_tokens == {"stream": 1}


def import_scheduler(**options):
    """A real scheduler with checkpoint caching and a restore connector mock.

    Tests install the connector after any setup step that processes output.
    """
    from unittest.mock import Mock

    scheduler = make_scheduler(
        **{
            "enable_prefix_caching": True,
            "use_v2_model_runner": True,
            "async_scheduling": True,
            **options,
        }
    )
    cache = scheduler.kv_cache_manager
    cache.boundary_checkpoints = BoundaryCheckpointCache(cache.block_pool)
    connector = Mock()
    connector.poll_boundary_checkpoint.return_value = True
    connector.boundary_checkpoint_external_tokens.return_value = 0
    connector.get_num_new_matched_tokens.return_value = (0, False)
    connector.has_pending_block_frees.return_value = False
    return scheduler, cache, connector


def start_decode(scheduler, name="running"):
    from tests.v1.core.utils import create_requests

    (running,) = create_requests(num_requests=1, num_tokens=16, req_ids=[name])
    scheduler.add_request(running)
    assert scheduler.schedule().num_scheduled_tokens == {name: 16}
    return running


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
@pytest.mark.parametrize("state", ["copying", "ready"])
def test_blocked_queue_head_admits_only_restored_imports_past_it(policy, state):
    """Ordinary requests keep their queue order while imports are pending."""
    from tests.v1.core.utils import create_requests

    scheduler, cache, connector = import_scheduler(
        num_blocks=64, scheduling_policy=policy
    )
    importer = request("import", "import", length=48)
    checkpoint = reserve_import(cache, importer, prefix=32)
    assert checkpoint is not None
    if state == "ready":
        publish(cache, checkpoint)
    (big,) = create_requests(num_requests=1, num_tokens=2000, req_ids=["big"])
    small = request("small", "small", length=16)
    connector.poll_boundary_checkpoint.side_effect = lambda req: (
        req is not importer or state == "ready"
    )
    scheduler.connector = connector
    for req in (big, small, importer):
        scheduler.add_request(req)
    expected = {"import": 16} if state == "ready" else {}
    assert scheduler.schedule().num_scheduled_tokens == expected
    assert small.status == big.status == RequestStatus.WAITING


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_pending_import_step_cost_does_not_scale_with_blocked_queue(policy):
    """A copying import leaves a blocked 500-request queue at one lookup a step."""
    import time

    from tests.v1.core.utils import create_requests

    def blocked(with_import):
        scheduler, cache, connector = import_scheduler(
            num_blocks=256, scheduling_policy=policy
        )
        queued = create_requests(num_requests=500, num_tokens=2048)
        importer = request("import", "import", length=64)
        if with_import:
            assert reserve_import(cache, importer, prefix=32) is not None
        cache.block_pool.get_new_blocks(cache.block_pool.get_num_free_blocks())
        connector.poll_boundary_checkpoint.side_effect = lambda req: (
            req is not importer
        )
        scheduler.connector = connector
        for req in (*queued, importer):
            scheduler.add_request(req)
        assert scheduler.schedule().num_scheduled_tokens == {}
        connector.poll_boundary_checkpoint.reset_mock()
        steps = 20
        start = time.perf_counter()
        for _ in range(steps):
            assert scheduler.schedule().num_scheduled_tokens == {}
        elapsed = (time.perf_counter() - start) / steps
        return elapsed, connector.poll_boundary_checkpoint.call_count / steps

    baseline, baseline_polls = blocked(with_import=False)
    elapsed, polls = blocked(with_import=True)
    assert polls == baseline_polls == 1
    assert elapsed < 10 * baseline + 1e-3


@pytest.mark.parametrize("state", ["copying", "ready", "ready_but_skipped"])
def test_ready_import_behind_older_request_keeps_running_decode(state):
    """Running decodes skip a step only for a saved-logits step actually taken."""
    scheduler, cache, connector = import_scheduler()
    running = start_decode(scheduler)
    importer = request("import", "import", length=32)
    checkpoint = reserve_import(cache, importer, prefix=32)
    assert checkpoint is not None
    if state != "copying":
        publish(cache, checkpoint)
    older = request("older", "older", length=16)
    connector.poll_boundary_checkpoint.side_effect = lambda req: (
        req is not importer or state == "ready"
    )
    connector.boundary_checkpoint_external_tokens.side_effect = lambda req: (
        32 if req is importer else 0
    )
    scheduler.connector = connector
    scheduler.add_request(older)
    scheduler.add_request(importer)
    output = scheduler.schedule()
    if state == "ready":
        assert output.boundary_logits_only
        assert output.num_scheduled_tokens == {"import": 1}
        output = scheduler.schedule()
        assert output.num_scheduled_tokens["running"] == 1
        assert output.num_scheduled_tokens["older"] == 16
    else:
        assert not output.boundary_logits_only
        assert output.num_scheduled_tokens == {"running": 1, "older": 16}
    assert running.num_preemptions == 0


def test_unadmitted_import_credits_do_not_preempt_running_decode():
    """Reservations gate new admissions, never the growth of running requests."""
    scheduler, cache, connector = import_scheduler()
    running = start_decode(scheduler)
    importer = request("import", "import", length=64)
    checkpoint = reserve_import(cache, importer, prefix=16)
    assert checkpoint is not None
    publish(cache, checkpoint)
    credits = cache.external_boundary_reserved_blocks()
    assert credits > 0
    # The decode crosses a block boundary with only the import's credits free.
    cache.block_pool.get_new_blocks(cache.block_pool.get_num_free_blocks() - credits)
    scheduler.connector = connector
    scheduler.add_request(importer)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["running"] == 1
    assert running.status == RequestStatus.RUNNING
    assert running.num_preemptions == 0


@pytest.mark.parametrize("policy", ["fcfs", "priority"])
def test_capacity_waiter_does_not_block_requests_ahead_of_it(policy):
    """A restore waiting for capacity cannot overtake requests ordered before it."""
    scheduler, cache, connector = import_scheduler(
        num_blocks=64, scheduling_policy=policy
    )
    earlier = request("earlier", "earlier", length=16)
    waiter = request("waiter", "waiter", length=400)
    lookup_done = [False]

    def poll(req):
        if req is earlier:
            return lookup_done[0]
        assert reserve_import(cache, req, prefix=384) is None
        return False

    connector.poll_boundary_checkpoint.side_effect = poll
    scheduler.connector = connector
    scheduler.add_request(earlier)
    scheduler.add_request(waiter)
    pressure = cache.block_pool.get_new_blocks(cache.block_pool.get_num_free_blocks())
    assert scheduler.schedule().num_scheduled_tokens == {}
    assert waiter.request_id in cache._boundary_import_waiters
    # Space returns for the earlier request, but not for the restore.
    cache.block_pool.free_blocks(pressure[:16])
    lookup_done[0] = True
    assert scheduler.schedule().num_scheduled_tokens == {"earlier": 16}
    assert waiter.status == RequestStatus.WAITING


def test_restore_past_its_bounded_wait_is_admitted_to_recompute(monkeypatch):
    """The connector keeps deferring the refused restore; the scheduler admits it."""
    clock, warnings = bounded_wait_clock(monkeypatch, 30)
    scheduler, cache, connector = import_scheduler(
        num_blocks=64, long_prefill_token_threshold=64
    )
    # A first chunk fits where the whole restore does not.
    scheduler.scheduler_reserve_full_isl = False
    waiter = request("waiter", "waiter", length=400)

    def poll(req):
        assert reserve_import(cache, req, prefix=384) is None
        return False

    connector.poll_boundary_checkpoint.side_effect = poll
    scheduler.connector = connector
    scheduler.add_request(waiter)
    pressure = cache.block_pool.get_new_blocks(cache.block_pool.get_num_free_blocks())
    assert scheduler.schedule().num_scheduled_tokens == {}
    cache.block_pool.free_blocks(pressure[:20])
    assert scheduler.schedule().num_scheduled_tokens == {}
    clock[0] += 31.0
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"waiter": 64}
    assert len(warnings) == 1 and warnings[0][2] == pytest.approx(31.0)
    assert not cache.external_boundary_wait_expired(waiter.request_id)


def run_contended_steps(scheduler, steps):
    """Schedule steps, charging each contended one to its compute class."""
    outputs = []
    for _ in range(steps):
        output = scheduler.schedule()
        if output.compute_service_class is not None:
            scheduler.record_compute_time(
                output.compute_service_class, 0.01, contended=True
            )
        outputs.append(output)
    return outputs


@pytest.mark.parametrize("slots", ["full", "free"])
def test_decodes_on_a_full_pool_preempt_while_restores_wait(slots):
    """A prefill turn that schedules nothing still lets decodes preempt.

    Waiting restores keep compute sharing contended, so every step after the
    first decode quantum is a prefill turn, whether the restores wait behind
    full running slots or for GPU capacity. When a decode then needs a block
    from a full pool, that turn must preempt like a decode turn; otherwise
    every later step is empty and nothing changes.
    """
    from tests.v1.core.utils import create_requests

    scheduler, cache, connector = import_scheduler(
        max_num_seqs=2 if slots == "full" else 3,
        num_blocks=17,
        **compute_share_fixture_options(0.4),
    )
    first, second = create_requests(
        num_requests=2, num_tokens=60, max_tokens=64, req_ids=["first", "second"]
    )
    for req in (first, second):
        scheduler.add_request(req)
    assert scheduler.schedule().num_scheduled_tokens == {"first": 60, "second": 60}
    assert cache.block_pool.get_num_free_blocks() == 0
    restores = [request(name, name) for name in ("restore-a", "restore-b")]
    polled = []

    def poll(req):
        polled.append(req.request_id)
        if req not in restores:
            return True
        assert reserve_import(cache, req, prefix=128) is None
        return False

    connector.poll_boundary_checkpoint.side_effect = poll
    scheduler.connector = connector
    for req in restores:
        scheduler.add_request(req)
    # Four decode steps fill the last block; the fifth needs a new one.
    outputs = run_contended_steps(scheduler, 8)
    assert all(output.num_scheduled_tokens for output in outputs)
    assert second.num_preemptions == 1
    assert first.status == RequestStatus.RUNNING
    assert first.num_computed_tokens == 68
    # With a slot free the restores compete, and their wait is bounded, again.
    assert {"restore-a", "restore-b"} <= set(polled)
    assert cache._boundary_import_waiters.keys() == {"restore-a", "restore-b"}


def test_reset_with_running_requests_releases_ready_unadmitted_import():
    """A cache reset drops a finished restore; its request then recomputes."""
    scheduler, cache, connector = import_scheduler()
    running = start_decode(scheduler)
    importer = request("import", "import", length=32)
    checkpoint = reserve_import(cache, importer, prefix=16)
    assert checkpoint is not None
    publish(cache, checkpoint)
    scheduler.connector = connector
    scheduler.add_request(importer)
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert running.status == RequestStatus.PREEMPTED
    assert importer.boundary_checkpoint is None
    assert not cache.has_external_boundary_admission(importer.request_id)
    assert cache.block_pool.get_num_free_blocks() == cache.block_pool.num_gpu_blocks - 1
    output = scheduler.schedule()
    assert output.num_scheduled_tokens["import"] == 32


@pytest.mark.parametrize("release", ["move_victim", "finish_reader"])
@pytest.mark.parametrize("fairness", [None, 0.4])
def test_actual_scheduler_runs_decode_after_guard_defers_restore(fairness, release):
    from tests.v1.core.utils import create_requests

    scheduler = make_scheduler(
        enable_prefix_caching=True,
        use_v2_model_runner=True,
        async_scheduling=True,
        num_speculative_tokens=3,
        speculative_method="ngram_gpu",
        **compute_share_fixture_options(fairness),
    )
    assert (scheduler.compute_share_controller is not None) == (fairness is not None)
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
    future = queue_matching(scheduler, unrelated)
    victim = victim_at_head(cache, checkpoint)
    blocked = scheduler.schedule()
    assert not blocked.boundary_logits_only
    assert blocked.num_scheduled_tokens == {"first": 4}
    assert second.status == RequestStatus.WAITING
    assert checkpoint.checkpoint_id in cache.boundary_checkpoints._entries
    if release == "move_victim":
        queue = cache.block_pool.free_block_queue
        queue.remove(victim)
        queue.append(victim)
    else:
        # Reader completion must unblock admission without rearranging the victim.
        scheduler.finish_requests([first.request_id], RequestStatus.FINISHED_STOPPED)
    admitted = scheduler.schedule()
    assert admitted.boundary_logits_only
    assert admitted.num_scheduled_tokens == {"second": 1}
    assert future.status == RequestStatus.WAITING
    scheduler.finish_requests(
        [second.request_id] + ([first.request_id] if release == "move_victim" else []),
        RequestStatus.FINISHED_ABORTED,
    )
    final = scheduler.schedule()
    assert final.boundary_logits_only
    assert final.num_scheduled_tokens == {future.request_id: 1}
    assert future.status == RequestStatus.RUNNING
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
    result = restore(
        cache, waiting, can_defer=can_defer, pending=[request("queued-b", "b")]
    )
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
