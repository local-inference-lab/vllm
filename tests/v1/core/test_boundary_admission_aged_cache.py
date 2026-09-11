# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Old checkpoints must not serialize otherwise admissible cached readers."""

import json
import inspect
import pytest
from tests.v1.core import test_boundary_admission as base
from tests.v1.core import test_boundary_admission_horizon as horizon
from tests.v1.core.test_boundary_admission import initialize_hash as initialize_hash
from vllm.v1.request import RequestStatus

pytestmark = pytest.mark.cpu_test


def aged_probe(defer):
    cache = horizon.manager()
    for i in range(300):
        horizon.seed(cache, str(i), lookahead=3)
    admitted = []
    waiting = [horizon.request("warm-" + str(i), str(i)) for i in (298, 299)]
    for req in waiting:
        cache.new_step_starts()
        blocks, hit, _ = cache.get_computed_blocks(req)
        assert hit == 8192
        options = {}
        if (
            "pending_boundary_requests"
            in inspect.signature(cache.allocate_slots).parameters
        ):
            options["pending_boundary_requests"] = waiting
        allocation = cache.allocate_slots(
            req,
            1,
            hit,
            blocks,
            num_lookahead_tokens=3,
            can_defer_boundary_restore=defer,
            **options,
        )
        if allocation is None:
            break
        req.num_computed_tokens = hit
        req.status = RequestStatus.RUNNING
        admitted.append(req)
        base.drain(cache)
    count = len(admitted)
    for req in admitted:
        cache.free(req)
    assert cache.block_pool.get_num_free_blocks() == 1933
    return count


def test_aged_cache_admits_two_warm_readers():
    guarded, unguarded = aged_probe(True), aged_probe(False)
    print(
        "PR721_PROBE "
        + json.dumps(
            dict(cold_requests=300, admitted=guarded, unguarded_admitted=unguarded)
        )
    )
    assert unguarded == 2
    assert guarded == 2, "Aged-cache guard serialized two warm readers"


def test_real_scheduler_admits_past_unrelated_old_checkpoint():
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
    (old,) = create_requests(num_requests=1, num_tokens=48, req_ids=["old"])
    for req in (producer, old):
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
    base.victim_at_head(cache, checkpoint)
    step = scheduler.schedule()
    assert step.boundary_logits_only, (
        "An unrelated old checkpoint blocked the queue head"
    )
    assert step.num_scheduled_tokens == {"second": 1}
    assert first.status == second.status == RequestStatus.RUNNING
    assert scheduler.max_num_running_reqs == 16
