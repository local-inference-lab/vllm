# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill-lane livelock regressions (max_parallel_prefills > 1).

A lane-selected async-load candidate whose allocation fails must not
strand the waiting loop with the remaining lane-holders still holding
lanes (intra-step leak).

A never-admissible rank-(-1) waiting cohort must not monopolize the
lanes across steps: lane aging has to hand a running chunk a lane in a
preempt-capable pass so preemption eventually frees blocks.

The other cross-step monopoly cohort - requests scheduled once and
then preempted keep their last_scheduled_step lane rank. That rank must
age out, and the post-waiting interleave pass must be able to preempt on
a step that scheduled nothing (only while the connector is quiescent),
or the frozen-rank pair starves the running chunks forever at free == 0.

With mamba "align" cache mode, a lane share below the block align
quantum zeroes the chunk in the aligned split before the allocator is
reached: doomed lane candidates must not fan the token budget out below
the quantum, or no chunk can ever advance.
"""

import pytest

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16


def _partial_prefill_output(req_ids: list[str]) -> ModelRunnerOutput:
    """Model-runner output for a step that only advanced prefills."""
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _empty_output() -> ModelRunnerOutput:
    return _partial_prefill_output([])


def test_async_admission_failure_releases_lane_for_other_candidates():
    """Failed async lane-holders must not block other lane candidates.

    Three async-load candidates (connector CPU-tier hits) can never pass
    the async admission gate while the running chunk reserves its
    remaining prompt. A failed lane-holder parks and releases its lane,
    lanes rotate to the chunk, every lane-holding candidate is processed
    within the step, and the chunk keeps advancing every step.
    """
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=1024,
        max_parallel_prefills=2,
        num_blocks=48,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=True),
    )
    looked_up: list[str] = []

    def fake_lookup(request, num_computed_tokens):
        looked_up.append(request.request_id)
        if request.request_id == "chunk":
            return 0, False
        # Whole-prompt CPU-tier hit: admission pays the full-sequence
        # requirement and must fail against the chunk's reservation.
        return 240, True

    assert scheduler.connector is not None
    scheduler.connector.get_num_new_matched_tokens = fake_lookup

    (chunk,) = create_requests(
        num_requests=1,
        num_tokens=640,
        req_ids=["chunk"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(chunk)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens.get("chunk", 0) == 64
    scheduler.update_from_output(output, _partial_prefill_output(["chunk"]))
    assert chunk.is_prefill_chunk

    for req_id in ("r1", "r2", "r3"):
        (req,) = create_requests(
            num_requests=1,
            num_tokens=640,
            req_ids=[req_id],
            block_size=BLOCK_SIZE,
        )
        scheduler.add_request(req)

    for _ in range(3):
        looked_up.clear()
        output = scheduler.schedule()
        assert output.num_scheduled_tokens.get("chunk", 0) == 64, (
            "running chunk starved: failed async lane-holders kept the lanes"
        )
        # Every lane-holding candidate was processed this step: the waiting
        # loop did not break with unprocessed lane-holders in place.
        assert set(looked_up) == {"r1", "r2", "r3"}
        scheduler.update_from_output(output, _partial_prefill_output(["chunk"]))

    for req_id in ("r1", "r2", "r3"):
        assert scheduler.requests[req_id].status == RequestStatus.WAITING
    assert chunk.num_computed_tokens == 64 + 3 * 64


def test_never_admissible_cohort_cannot_starve_runnable_chunk_forever():
    """A rank-(-1) cohort must age out so preemption can fire.

    A filler plus two running prefill chunks own the whole pool
    (free == 0, their blocks non-evictable) and four promoted-waiting
    requests can never be admitted. The connector is held
    non-quiescent so the guarded post-waiting interleave pass cannot
    preempt: cross-step lane aging is the isolated recovery mechanism,
    rotating the cohort behind the chunks within LANE_AGING_STEPS
    steps until a chunk reaches the preempt-capable first pass and
    preemption of the last running request frees blocks.
    """

    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=32,
        max_model_len=1024,
        max_parallel_prefills=4,
        num_blocks=33,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    # Keep the connector non-quiescent: the post-waiting interleave pass
    # may preempt on a no-progress step only while it holds no in-flight
    # jobs, so this test exercises aging as the sole recovery path.
    assert scheduler.connector is not None
    scheduler.connector.has_pending_push_work = lambda: True

    # Filler: prefills to completion, then holds its blocks pinned
    # (non-evictable) without contributing tokens (no sampled output).
    (filler,) = create_requests(
        num_requests=1,
        num_tokens=128,
        req_ids=["filler"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(filler)
    while filler.num_computed_tokens < 128:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        scheduler.update_from_output(
            output, _partial_prefill_output(list(output.num_scheduled_tokens))
        )
    assert filler.num_computed_tokens == 128

    c1, c2 = create_requests(
        num_requests=2,
        num_tokens=320,
        req_ids=["c1", "c2"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(c1)
    scheduler.add_request(c2)

    # Fill the pool with the two chunks (lane token-split keeps them even;
    # 24 blocks remain, both need 20 -> neither completes).
    while scheduler.kv_cache_manager.block_pool.get_num_free_blocks() > 0:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        scheduler.update_from_output(
            output, _partial_prefill_output(list(output.num_scheduled_tokens))
        )
    assert c1.is_prefill_chunk and c2.is_prefill_chunk

    # Promoted-waiting cohort: async loads completed (num_computed preset),
    # their blocks were never allocated, and free == 0 makes them
    # permanently inadmissible.
    for i in range(4):
        (promoted,) = create_requests(
            num_requests=1,
            num_tokens=256,
            req_ids=[f"p{i}"],
            block_size=BLOCK_SIZE,
        )
        promoted.num_computed_tokens = 240
        scheduler.add_request(promoted)

    # The intra-step rotation alone cannot recover: while the cohort holds
    # rank -1 at step start, the chunks only reach the post-waiting pass,
    # which cannot preempt while the connector reports in-flight jobs, so
    # steps stay empty.
    for _ in range(10):
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens == 0
        scheduler.update_from_output(output, _empty_output())

    # Aging must hand the chunks a lane in the preempt-capable first pass
    # within a bounded number of steps.
    progressed = False
    for _ in range(64):
        output = scheduler.schedule()
        if output.total_num_scheduled_tokens > 0:
            progressed = True
            break
        scheduler.update_from_output(output, _empty_output())
    assert progressed, (
        "never-scheduled rank-(-1) cohort starved the running chunks "
        "forever: lane aging did not rotate lanes to a preempt-capable pass"
    )
    assert c1.num_preemptions + c2.num_preemptions >= 1, (
        "expected preemption of a running chunk to free blocks"
    )


def _drive_to_preempted_after_scheduled_cohort(scheduler):
    """Build the preempted-after-scheduled roster and return (c1, c2, s1, s2).

    Two chunks fill the pool (45 blocks; both stop below their 24-block
    completion), a never-scheduled pair is scheduled once for one
    32-token chunk each (last_scheduled_step set), the scheduler's own
    preemption path (pop the running tail + _preempt_request, as the
    preempt-capable first pass does when the pool goes dry) preempts
    the pair, and the chunks consume the freed blocks until free == 0.
    The pair then waits, preempted, with its lane rank set and ahead of
    the chunks'.
    """
    free_blocks = scheduler.kv_cache_manager.block_pool.get_num_free_blocks
    c1, c2 = create_requests(
        num_requests=2,
        num_tokens=384,
        req_ids=["c1", "c2"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(c1)
    scheduler.add_request(c2)
    # Advance both chunks to 14 blocks each, leaving 16 free blocks:
    # enough for the pair's full-sequence admission check (224 tokens
    # = 14 blocks each) and an even drain (four blocks per step) back
    # to free == 0, while the first chunk advance step drops free below
    # the pair's full-sequence requirement so it can never re-qualify.
    while free_blocks() > 16:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        scheduler.update_from_output(
            output, _partial_prefill_output(list(output.num_scheduled_tokens))
        )
    assert c1.is_prefill_chunk and c2.is_prefill_chunk

    s1, s2 = create_requests(
        num_requests=2,
        num_tokens=224,
        req_ids=["s1", "s2"],
        block_size=BLOCK_SIZE,
    )
    scheduler.add_request(s1)
    scheduler.add_request(s2)
    # The never-scheduled pair outranks the chunks and takes both lanes
    # for exactly one step.
    output = scheduler.schedule()
    assert output.num_scheduled_tokens.get("s1") == 32
    assert output.num_scheduled_tokens.get("s2") == 32
    scheduler.update_from_output(output, _partial_prefill_output(["s1", "s2"]))

    # Preempt the pair the way the preempt-capable first pass does when
    # the pool goes dry: pop the running tail and preempt. Preemption
    # does not clear last_scheduled_step.
    for victim in (s2, s1):
        assert scheduler.running.pop() is victim
        scheduler._preempt_request(victim, timestamp=0.0)

    # The chunks consume the freed blocks down to free == 0. The pair's
    # frozen rank keeps it at the lane front, where its admission fails
    # the full-sequence check every step and it parks, so the chunks
    # advance through the post-waiting pass.
    while free_blocks() > 0:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        scheduler.update_from_output(
            output, _partial_prefill_output(list(output.num_scheduled_tokens))
        )
    for req in (s1, s2):
        assert req.status == RequestStatus.PREEMPTED
        assert req.num_preemptions == 1
    assert c1.is_prefill_chunk and c2.is_prefill_chunk
    controller = scheduler.prefill_interleave_controller
    # The pair's rank is set and ahead of the chunks' (scheduled once,
    # preempted without its last_scheduled_step being cleared), so it
    # wins both lanes at every step start.
    assert "s1" in controller.last_scheduled_step
    assert "s2" in controller.last_scheduled_step
    assert controller.last_scheduled_step["s1"] < controller.last_scheduled_step["c1"]
    assert controller.last_scheduled_step["s2"] < controller.last_scheduled_step["c2"]
    return c1, c2, s1, s2


def test_preempted_after_scheduled_cohort_cannot_starve_runnable_chunks():
    """The preempted-after-scheduled cohort must not starve the chunks.

    The pair keeps its last_scheduled_step rank and wins both lanes at
    step start, so the chunks only reach the post-waiting interleave
    pass: with the connector quiescent it must preempt on a step that
    scheduled nothing, and preemption of the last running chunk frees
    blocks for a forward.
    """
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=1024,
        max_parallel_prefills=2,
        num_blocks=45,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
    )
    c1, c2, s1, s2 = _drive_to_preempted_after_scheduled_cohort(scheduler)

    progressed = False
    for _ in range(64):
        output = scheduler.schedule()
        if output.total_num_scheduled_tokens > 0:
            progressed = True
            break
        scheduler.update_from_output(output, _empty_output())
    assert progressed, (
        "preempted-after-scheduled cohort starved the running chunks: "
        "the frozen-rank pair held both lanes at step start and the "
        "post-waiting interleave pass scheduled nothing"
    )
    assert c1.num_preemptions + c2.num_preemptions >= 1, (
        "expected recovery through preemption of a running request"
    )
    for req in (s1, s2):
        assert req.status == RequestStatus.PREEMPTED


def test_busy_connector_guards_preemption_and_aging_recovers():
    """With connector jobs in flight the post-waiting interleave pass
    must not preempt (the preemption's flush would reach an unbounded
    worker-side wait()); recovery then comes from aging the stale
    ranks out of the lane front so a chunk reaches the preempt-capable
    first pass.
    """
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=64,
        max_model_len=1024,
        max_parallel_prefills=2,
        num_blocks=45,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=False),
    )
    assert scheduler.connector is not None
    scheduler.connector.has_pending_push_work = lambda: True
    c1, c2, s1, s2 = _drive_to_preempted_after_scheduled_cohort(scheduler)

    # The guard holds while the pair's rank is still fresh: no-progress
    # steps stay empty instead of preempting.
    for _ in range(10):
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens == 0, (
            "the post-waiting interleave pass preempted while the "
            "connector reported in-flight jobs"
        )
        scheduler.update_from_output(output, _empty_output())

    progressed = False
    for _ in range(64):
        output = scheduler.schedule()
        if output.total_num_scheduled_tokens > 0:
            progressed = True
            break
        scheduler.update_from_output(output, _empty_output())
    assert progressed, (
        "stale frozen ranks never aged out of the lane front: the "
        "preempted-after-scheduled cohort starved the chunks with the "
        "interleave pass guarded"
    )
    assert c1.num_preemptions + c2.num_preemptions >= 1, (
        "expected recovery through preemption in the preempt-capable "
        "first running pass"
    )


def test_lane_share_below_align_quantum_cannot_starve_chunk():
    """A fanout share below the align quantum must not zero the chunk.

    In a mamba "align" deployment the align quantum is the block size
    (64 here); a chunk whose share ends before the next block boundary is
    floored to zero tokens by the aligned split, consumes its lane, and
    never reaches the allocator. With more lane candidates than lanes,
    every chunk's fanout share falls below the quantum and no prefill
    can advance while doomed async candidates hold lanes and fail the
    full-sequence admission gate: empty steps forever. The lane share
    must fall back to the global step budget whenever that budget can
    cover a full quantum.
    """
    scheduler = create_scheduler(
        max_num_seqs=16,
        max_num_batched_tokens=131,  # 2 * 64 + 3: two full quanta per step
        max_model_len=1024,
        max_parallel_prefills=4,
        num_blocks=11,
        block_size=64,
        enable_prefix_caching=False,
        use_kv_connector=mock_kv(matched_tokens=0, is_async=True),
    )
    # The scheduler-side flag of a mamba "align" deployment (derived at
    # construction from has_mamba_layers and mamba_cache_mode == "align").
    scheduler.need_mamba_block_aligned_split = True

    def fake_lookup(request, num_computed_tokens):
        if request.request_id == "chunk":
            return 0, False
        # Whole-prompt CPU-tier hit: the async admission gate pays the
        # full-sequence requirement (8 blocks) and must fail against
        # free (6) minus the chunk's remaining-block reservation (4).
        return 240, True

    assert scheduler.connector is not None
    scheduler.connector.get_num_new_matched_tokens = fake_lookup

    (chunk,) = create_requests(
        num_requests=1,
        num_tokens=512,
        req_ids=["chunk"],
        block_size=64,
    )
    scheduler.add_request(chunk)
    # Advance to a block-aligned mid-prefill position with two full
    # quanta pending (the aligned split floors every step to 128 here).
    while chunk.num_computed_tokens < 256:
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        scheduler.update_from_output(
            output, _partial_prefill_output(list(output.num_scheduled_tokens))
        )
    assert chunk.is_prefill_chunk

    for req_id in ("d1", "d2"):
        (doomed,) = create_requests(
            num_requests=1,
            num_tokens=512,
            req_ids=[req_id],
            block_size=64,
        )
        scheduler.add_request(doomed)

    # All three candidates hold lanes (4 lanes, 3 candidates), so the
    # chunk's fanout share is ceil(131/3) = 44 < 64: below the
    # quantum, the aligned split would zero the chunk and consume the
    # lane at the zero-token branch. With the share cap dropped, the
    # chunk takes two full quanta from the step budget.
    progressed = False
    for _ in range(64):
        output = scheduler.schedule()
        if output.total_num_scheduled_tokens > 0:
            progressed = True
            break
        scheduler.update_from_output(output, _empty_output())
    assert progressed, (
        "lane fanout shares below the align quantum zeroed the chunk: "
        "the lane-selected chunk never reached the allocator and starved "
        "behind the doomed async candidates"
    )
    assert chunk.num_computed_tokens == 384
    for req_id in ("d1", "d2"):
        assert scheduler.requests[req_id].status == RequestStatus.WAITING
