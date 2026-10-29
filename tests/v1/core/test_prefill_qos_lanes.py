# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Size-classed prefill lanes: short work keeps flowing beside long prefills.

Geometry for every test: 4 lanes (auto), 16-token steps, block size 4, and a
long threshold of 32 uncached tokens with one long lane.
"""

import pytest

from vllm.config import SchedulerConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.sched.prefill_interleave import PrefillInterleaveController
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

THRESHOLD = 32


@pytest.fixture
def opt_model_path(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"architectures":["OPTForCausalLM"],"model_type":"opt",'
        '"max_position_embeddings":32768}'
    )
    return str(tmp_path)


def _scheduler(opt_model_path, **kwargs):
    kwargs.setdefault("max_parallel_prefills", "auto")
    kwargs.setdefault("prefill_long_threshold", THRESHOLD)
    kwargs.setdefault("max_long_prefills", 1)
    return create_scheduler(
        model=opt_model_path,
        skip_tokenizer_init=True,
        device="cpu",
        max_num_seqs=kwargs.pop("max_num_seqs", 8),
        max_num_batched_tokens=kwargs.pop("max_num_batched_tokens", 16),
        max_model_len=kwargs.pop("max_model_len", 512),
        block_size=kwargs.pop("block_size", 4),
        **kwargs,
    )


def _update(scheduler, output) -> None:
    req_ids = list(output.num_scheduled_tokens)
    sampled_token_ids = [
        [] if scheduler.requests[req_id].is_prefill_chunk else [0] for req_id in req_ids
    ]
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: index for index, req_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _start(scheduler, req_id: str, num_tokens: int):
    (request,) = create_requests(
        num_requests=1, num_tokens=num_tokens, req_ids=[req_id]
    )
    scheduler.add_request(request)
    return request


def test_size_classes_require_parallel_lanes():
    with pytest.raises(ValueError, match="prefill_long_threshold"):
        SchedulerConfig(
            max_model_len=128,
            is_encoder_decoder=False,
            max_parallel_prefills=1,
            prefill_long_threshold=THRESHOLD,
        )


def test_size_classes_cli_contract():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    namespace = parser.parse_args(
        [
            "--max-parallel-prefills",
            "auto",
            "--prefill-long-threshold",
            "32768",
            "--max-long-prefills",
            "2",
        ]
    )
    engine_args = EngineArgs.from_cli_args(namespace)
    assert engine_args.prefill_long_threshold == 32768
    assert engine_args.max_long_prefills == 2


def test_long_prefill_alone_keeps_the_full_budget(opt_model_path):
    scheduler = _scheduler(opt_model_path)
    _start(scheduler, "long", 200)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"long": 16}


def test_long_prefill_keeps_a_lane_when_short_requests_arrive(opt_model_path):
    """Plain round-robin gives every lane to never-scheduled newcomers; size
    classes keep one lane for the running long prefill (its progress floor)."""
    scheduler = _scheduler(opt_model_path)
    _start(scheduler, "long", 200)
    _update(scheduler, scheduler.schedule())
    for index in range(4):
        _start(scheduler, f"s{index}", 8)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens["long"] >= 4
    shorts = [f"s{index}" for index in range(4)]
    assert sum(req_id in output.num_scheduled_tokens for req_id in shorts) == 3
    assert sum(output.num_scheduled_tokens.values()) == 16


def test_without_size_classes_newcomers_take_every_lane(opt_model_path):
    """The legacy round-robin behaviour is unchanged when size classes are off."""
    scheduler = _scheduler(opt_model_path, prefill_long_threshold=0)
    _start(scheduler, "long", 200)
    _update(scheduler, scheduler.schedule())
    for index in range(4):
        _start(scheduler, f"s{index}", 8)

    output = scheduler.schedule()

    assert "long" not in output.num_scheduled_tokens
    assert sorted(output.num_scheduled_tokens) == ["s0", "s1", "s2", "s3"]


def test_second_long_admission_skips_over_without_blocking_short_work(
    opt_model_path,
):
    scheduler = _scheduler(opt_model_path)
    long1 = _start(scheduler, "long1", 200)
    _update(scheduler, scheduler.schedule())
    long2 = _start(scheduler, "long2", 200)
    _start(scheduler, "s0", 8)
    _start(scheduler, "s1", 8)

    output = scheduler.schedule()

    assert "long1" in output.num_scheduled_tokens
    assert "long2" not in output.num_scheduled_tokens
    assert "s0" in output.num_scheduled_tokens
    assert "s1" in output.num_scheduled_tokens
    # long2 was looked up once and parked, not admitted.
    assert long2.qos_cached_tokens_hint == 0
    assert long2.status == RequestStatus.WAITING
    assert long1.status == RequestStatus.RUNNING

    # While long1 is in flight, the parked long2 gets no lane at all.
    _update(scheduler, output)
    output = scheduler.schedule()
    assert "long2" not in output.num_scheduled_tokens
    assert "long1" in output.num_scheduled_tokens


def test_parked_long_is_admitted_once_long_work_drops_below_threshold(
    opt_model_path,
):
    scheduler = _scheduler(opt_model_path)
    long1 = _start(scheduler, "long1", 64)
    _update(scheduler, scheduler.schedule())
    long2 = _start(scheduler, "long2", 200)

    admitted_at = None
    for step in range(12):
        output = scheduler.schedule()
        if "long2" in output.num_scheduled_tokens:
            admitted_at = step
            break
        _update(scheduler, output)

    assert admitted_at is not None
    # long2 waited only until long1's remaining work fell below the threshold.
    assert long1.num_tokens - 1 - long1.num_computed_tokens < THRESHOLD
    assert long2.status == RequestStatus.RUNNING


def test_short_requests_finish_while_long_prefill_progresses(opt_model_path):
    scheduler = _scheduler(opt_model_path)
    long_request = _start(scheduler, "long", 400)
    _update(scheduler, scheduler.schedule())
    shorts = [_start(scheduler, f"s{index}", 12) for index in range(6)]

    for _ in range(10):
        before = long_request.num_computed_tokens
        output = scheduler.schedule()
        _update(scheduler, output)
        # The long prefill advances on every step: no starvation.
        assert long_request.num_computed_tokens > before

    # Every short request finished its prefill and is decoding.
    for request in shorts:
        assert request.num_computed_tokens >= request.num_prompt_tokens
    assert long_request.num_computed_tokens < long_request.num_prompt_tokens


def test_blocked_request_does_not_take_a_lane():
    """A request preempted mid-prefill keeps is_prefill_chunk; while it waits
    for a remote KV load it must not hold a prefill lane."""
    blocked, ready = create_requests(
        num_requests=2, num_tokens=64, req_ids=["blocked", "ready"]
    )
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    blocked.is_prefill_chunk = True

    step = PrefillInterleaveController().begin_step(
        running=[],
        waiting=[ready],
        skipped_waiting=[blocked],
        request_lookup={"blocked": blocked, "ready": ready},
        is_local_prefill=Scheduler._request_has_local_prefill,
        max_parallel_prefills=2,
        policy="round-robin",
        num_runnable_decodes=0,
        decode_refill_target=2,
        current_step=1,
        respect_priority=False,
    )

    assert step is not None
    assert step.is_selected("ready")
    assert not step.is_selected("blocked")


@pytest.mark.parametrize("threshold, admitted", [(THRESHOLD, False), (0, True)])
def test_long_admission_reserves_in_flight_prefill_blocks(
    opt_model_path, threshold, admitted
):
    """With size classes on, a long admission must fit beside the blocks an
    in-flight long prefill still needs (full-ISL check with reservation)."""
    scheduler = _scheduler(
        opt_model_path,
        prefill_long_threshold=threshold,
        max_long_prefills=2,
        num_blocks=80,
    )
    _start(scheduler, "long1", 200)
    _update(scheduler, scheduler.schedule())
    _start(scheduler, "long2", 140)

    output = scheduler.schedule()

    assert ("long2" in output.num_scheduled_tokens) is admitted
