# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for checkpoint-restore scheduler admission guards.

Four mechanisms, all bounded-failure hardening for deployments where KV
transfers can wedge:

- The FCFS preemption victim must not have been scheduled earlier in the
  same model step: unlike the PRIORITY branch, the FCFS arm performs no
  same-step rollback, so a same-step-scheduled victim would silently free
  blocks the step output still claims. The guard raises instead.
- Admission-starved requests (waiting-pass allocation failures) are stamped
  on first failure and rate-log the starvation, so a wedge is explainable
  from stock logs.
- A starved admission is relieved by preempting the largest freeable
  running peer — age-gated (only where natural resolution demonstrably
  failed), cooldown-gated (>= the flush bound so preemptions never stack
  flushes), and capped per starvation episode.
- Finished requests whose block free was deferred for a KV-transfer
  completion that never arrives are reaped after a deadline, so a dropped
  completion cannot leak the request and its blocks forever.
"""

import time
from types import SimpleNamespace

import pytest

from vllm.v1.core.sched import scheduler as sched_module
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


STARVE_LOG_INTERVAL = 30.0


def _model_runner_output(req_ids: list[str], sampled: list[list[int]]):
    from vllm.v1.outputs import ModelRunnerOutput

    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_fcfs_preemption_guard_rejects_scheduled_victim():
    """The FCFS preempt arm must refuse a victim scheduled this step.

    The PRIORITY branch rolls a same-step-scheduled victim back out of the
    step output; the FCFS arm has no rollback, so the only safe action is
    to fail loudly.
    """
    scheduled = {"victim": 4}
    victim = SimpleNamespace(request_id="victim")
    with pytest.raises(AssertionError, match="FCFS preemption victim"):
        sched_module._ensure_fcfs_victim_not_scheduled(victim, scheduled)

    # An unscheduled victim is the normal, safe case.
    other = SimpleNamespace(request_id="other")
    sched_module._ensure_fcfs_victim_not_scheduled(other, scheduled)


def test_admission_starve_loggable_rate_limits():
    """First failure logs immediately and stamps first-seen; afterwards at
    most one line per interval, with the interval state riding the request.
    """
    request = SimpleNamespace()
    now = 100.0
    assert sched_module._admission_starve_loggable(request, now, 30.0)
    assert request._starve_first == 100.0
    assert request._starve_last == 100.0

    # Inside the interval: silent, stamps unchanged.
    assert not sched_module._admission_starve_loggable(request, 110.0, 30.0)
    assert request._starve_last == 100.0

    # Past the interval: one more line, first-seen preserved.
    assert sched_module._admission_starve_loggable(request, 140.0, 30.0)
    assert request._starve_first == 100.0
    assert request._starve_last == 140.0


def test_starved_admission_break_stamps_and_logs():
    """A waiting request that cannot allocate stamps its first failure and
    logs the starvation; the age gate keeps relief off on first sight.
    """
    scheduler = create_scheduler(
        device="cpu",
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=12,
        enable_prefix_caching=False,
    )
    (victim,) = create_requests(
        num_requests=1, num_tokens=80, block_size=16, req_ids=["victim"]
    )
    (starved,) = create_requests(
        num_requests=1, num_tokens=1000, block_size=16, req_ids=["starved"]
    )
    scheduler.add_request(victim)
    scheduler.schedule()

    scheduler.add_request(starved)
    scheduler.schedule()

    # The starved request broke at the admission arm: stamped and aged.
    assert hasattr(starved, "_starve_first")
    assert hasattr(starved, "_starve_last")
    assert starved._starve_last - starved._starve_first >= 0.0

    # Fresh stamp means the age gate held: no relief preemption yet.
    assert victim.request_id in {r.request_id for r in scheduler.running}
    assert victim.status == RequestStatus.RUNNING


def test_starved_admission_preempts_largest_freeable_peer(monkeypatch):
    """Past the age gate, the starved admission preempts the largest
    freeable running peer (not scheduled this step), and the starved
    request is requeued ahead of the preempted victim.
    """
    monkeypatch.setenv("LMCACHE_ADMISSION_PREEMPT_AGE", "0")
    scheduler = create_scheduler(
        device="cpu",
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=12,
        enable_prefix_caching=False,
    )
    (big,) = create_requests(
        num_requests=1, num_tokens=96, block_size=16, req_ids=["big"]
    )
    (small,) = create_requests(
        num_requests=1, num_tokens=80, block_size=16, req_ids=["small"]
    )
    (starved,) = create_requests(
        num_requests=1, num_tokens=1000, block_size=16, req_ids=["starved"]
    )
    scheduler.add_request(big)
    scheduler.schedule()
    scheduler.add_request(small)
    scheduler.schedule()
    assert big.status == RequestStatus.RUNNING
    assert small.status == RequestStatus.RUNNING

    scheduler.add_request(starved)
    output = scheduler.schedule()

    # The victim is the peer holding the most blocks ("big": 6 vs 5), and
    # the smaller peer ("small") is untouched.
    preempted = output.preempted_req_ids
    assert "big" in preempted
    assert big.status == RequestStatus.PREEMPTED
    assert small.status == RequestStatus.RUNNING
    assert "small" in {r.request_id for r in scheduler.running}

    # The starved request was not admitted (it still cannot fit) and the
    # episode counter recorded the single relief preemption.
    assert starved.status == RequestStatus.WAITING
    assert starved._admission_preempts == 1

    # Requeued at the head of the waiting queue, ahead of the re-queued
    # victim, so the freed blocks cannot be re-claimed by the victim.
    assert next(iter(scheduler.waiting)) is starved


def test_starved_admission_gates_age_cooldown_and_cap(monkeypatch):
    """The relief is age-gated, cooldown-gated, and episode-capped."""
    monkeypatch.setenv("LMCACHE_ADMISSION_PREEMPT_MAX_PER_EPISODE", "2")
    scheduler = create_scheduler(
        device="cpu",
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=12,
        enable_prefix_caching=False,
    )
    (big,) = create_requests(
        num_requests=1, num_tokens=96, block_size=16, req_ids=["big"]
    )
    (small,) = create_requests(
        num_requests=1, num_tokens=80, block_size=16, req_ids=["small"]
    )
    (starved,) = create_requests(
        num_requests=1, num_tokens=1000, block_size=16, req_ids=["starved"]
    )
    scheduler.add_request(big)
    scheduler.schedule()
    scheduler.add_request(small)
    scheduler.schedule()
    scheduler.add_request(starved)

    def fire() -> None:
        scheduler._maybe_preempt_for_starved_admission(
            starved,
            num_scheduled_tokens={},
            scheduled_timestamp=time.monotonic(),
            prefill_interleave_step=None,
            request_queue=scheduler.waiting,
        )

    # Age gate: a fresh stamp grants nothing.
    starved._starve_first = time.monotonic()
    fire()
    assert big.status == RequestStatus.RUNNING
    assert small.status == RequestStatus.RUNNING
    assert "_admission_preempts" not in vars(starved)

    # Aged stamp: the largest freeable peer ("big") is preempted.
    starved._starve_first = time.monotonic() - 400.0
    fire()
    assert big.status == RequestStatus.PREEMPTED
    assert starved._admission_preempts == 1

    # Cooldown: the second fire is suppressed even though "small" is a
    # valid victim (default cooldown 300 s dwarfs the test runtime).
    starved._starve_first = time.monotonic() - 400.0
    fire()
    assert small.status == RequestStatus.RUNNING
    assert starved._admission_preempts == 1

    # Cooldown elapsed: the next peer ("small") is preempted.
    scheduler._last_admission_preempt = float("-inf")
    fire()
    assert small.status == RequestStatus.PREEMPTED
    assert starved._admission_preempts == 2


def test_starved_admission_respects_episode_cap(monkeypatch):
    """Within an episode the cap bounds relief: a starved request that
    already consumed its preemption grants no further ones.
    """
    monkeypatch.setenv("LMCACHE_ADMISSION_PREEMPT_AGE", "0")
    scheduler = create_scheduler(
        device="cpu",
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=12,
        enable_prefix_caching=False,
    )
    (victim,) = create_requests(
        num_requests=1, num_tokens=80, block_size=16, req_ids=["victim"]
    )
    (starved,) = create_requests(
        num_requests=1, num_tokens=1000, block_size=16, req_ids=["starved"]
    )
    scheduler.add_request(victim)
    scheduler.schedule()
    scheduler.add_request(starved)

    starved._starve_first = time.monotonic() - 400.0
    starved._admission_preempts = 1
    scheduler._maybe_preempt_for_starved_admission(
        starved,
        num_scheduled_tokens={},
        scheduled_timestamp=time.monotonic(),
        prefill_interleave_step=None,
        request_queue=scheduler.waiting,
    )
    assert victim.status == RequestStatus.RUNNING
    assert starved._admission_preempts == 1


def test_delayed_free_requests_reaped_after_deadline(monkeypatch):
    """A finished request whose block free was deferred for a KV-transfer
    completion is force-freed after the deadline; a request inside the
    deadline is left for the connector.
    """
    monkeypatch.setenv("LMCACHE_DELAYED_FREE_TIMEOUT", "1.0")
    scheduler = create_scheduler(
        device="cpu",
        max_num_batched_tokens=100,
        block_size=16,
        num_blocks=12,
        enable_prefix_caching=False,
    )
    (stale,) = create_requests(
        num_requests=1, num_tokens=80, block_size=16, req_ids=["stale"]
    )
    (fresh,) = create_requests(
        num_requests=1, num_tokens=80, block_size=16, req_ids=["fresh"]
    )
    scheduler.add_request(stale)
    scheduler.schedule()
    scheduler.add_request(fresh)
    scheduler.schedule()
    pool = scheduler.kv_cache_manager.block_pool
    free_before = pool.get_num_free_blocks()

    # Simulate the deferred-free state: the request finished but its block
    # free is waiting on a KV-transfer completion that never arrived.
    stale.status = RequestStatus.FINISHED_STOPPED
    fresh.status = RequestStatus.FINISHED_STOPPED
    now = time.monotonic()
    scheduler._free_request(stale, delay_free_blocks=True)
    scheduler._free_request(fresh, delay_free_blocks=True)
    scheduler._delayed_free_reqs[stale.request_id] = now - 10.0
    scheduler._delayed_free_reqs[fresh.request_id] = now

    scheduler.schedule()

    # The stale one is reaped: gone from the scheduler, blocks returned.
    assert stale.request_id not in scheduler.requests
    assert fresh.request_id in scheduler.requests
    assert pool.get_num_free_blocks() == free_before + 5

    # A normal completion still cleans the stamp (fresh is freed normally
    # once the connector reports completion).
    scheduler._free_request(fresh, delay_free_blocks=False)
    assert fresh.request_id not in scheduler._delayed_free_reqs
