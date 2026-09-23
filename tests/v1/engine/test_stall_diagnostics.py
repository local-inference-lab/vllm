# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request timelines and engine-loop stall reports."""

import asyncio
import time
from types import SimpleNamespace

from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.engine.output_processor import RequestOutputCollector
from vllm.v1.engine.stall_diagnostics import EngineLoopWatchdog, RequestTimeline
from vllm.v1.metrics.stats import RequestStateStats


def _output(request_id: str, finished: bool) -> RequestOutput:
    stats = RequestStateStats(num_generation_tokens=7, queued_ts=10.0)
    stats.scheduled_ts = 10.25
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=list(range(100)),
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="",
                token_ids=[1],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop" if finished else None,
            )
        ],
        finished=finished,
        metrics=stats,
        num_cached_tokens=96,
    )


def _timeline(now: float, **stages: float) -> RequestTimeline:
    timeline = RequestTimeline("req-1", now - stages.pop("arrival_age"), now - 5.0)
    for stage, age in stages.items():
        setattr(timeline, stage, now - age)
    return timeline


def test_hold_before_submission_is_reported(caplog_vllm):
    now = time.time()
    # Rendered 125 s ago, but the generator only started 5 s ago.
    timeline = _timeline(
        now,
        arrival_age=125.0,
        processed=4.9,
        submitted=4.8,
        first_output=1.0,
        final_output=0.1,
    )
    timeline.output(_output("req-1", finished=True))
    timeline.final_output = now - 0.1
    timeline.close("finished", threshold=60.0, always=False)

    record = next(r for r in caplog_vllm.records if "req-1" in r.getMessage())
    message = record.getMessage()
    assert record.levelname == "WARNING"
    assert "Request req-1 finished after 125." in message
    assert "arrival to generator start 120.00 s" in message
    assert "engine queue 0.25 s" in message
    assert "prompt 100 tokens (96 cached)" in message
    assert "7 generated" in message
    assert "finish reason stop" in message


def test_fast_request_is_not_reported(caplog_vllm):
    now = time.time()
    timeline = _timeline(
        now,
        arrival_age=5.5,
        processed=4.9,
        submitted=4.8,
        first_output=1.0,
        final_output=0.1,
    )
    timeline.close("finished", threshold=60.0, always=False)
    assert "req-1" not in caplog_vllm.text


def test_request_aborted_before_any_output_is_reported(caplog_vllm):
    now = time.time()
    timeline = _timeline(now, arrival_age=90.0, processed=4.9, submitted=4.8)
    timeline.started = now - 89.9
    timeline.close("aborted", threshold=60.0, always=False)
    assert "Request req-1 aborted after 90." in caplog_vllm.text
    assert "submission to first output -" in caplog_vllm.text


def test_handler_stopping_after_final_output_counts_as_finished(caplog_vllm):
    now = time.time()
    timeline = _timeline(now, arrival_age=5.5, processed=4.9, submitted=4.8)
    timeline.output(_output("req-1", finished=True))
    timeline.close("aborted", threshold=60.0, always=True)
    assert "Request req-1 finished after" in caplog_vllm.text


def test_generate_records_each_stage(caplog_vllm):
    async def add_request(request_id, prompt, params, **kwargs):
        collector = RequestOutputCollector(RequestOutputKind.DELTA, request_id)
        collector.put(_output(request_id, finished=True))
        return collector

    engine = SimpleNamespace(
        add_request=add_request,
        log_requests=False,
        _request_stall_warning_s=60.0,
        _log_request_timeline=True,
    )
    prompt = {"type": "token", "prompt_token_ids": [1, 2], "arrival_time": 0.0}
    prompt["arrival_time"] = time.time() - 2.0

    async def consume():
        return [
            out
            async for out in AsyncLLM.generate(
                engine, prompt, SamplingParams(), "req-1"
            )
        ]

    outputs = asyncio.run(consume())
    assert len(outputs) == 1 and outputs[0].finished
    message = next(
        r.getMessage() for r in caplog_vllm.records if "req-1" in r.getMessage()
    )
    assert message.startswith("Request req-1 finished after 2.")
    assert "arrival to generator start 2." in message
    for stage in (
        "input processing",
        "engine submission",
        "submission to first output",
        "first to final output",
        "final output to stream end",
    ):
        assert f"{stage} -" not in message


def test_engine_loop_watchdog_reports_a_blocked_step(caplog_vllm):
    watchdog = EngineLoopWatchdog(threshold=0.2, poll_interval=0.02)

    def step_waiting_on_a_transfer():
        time.sleep(0.6)

    try:
        watchdog.busy("engine step")
        step_waiting_on_a_transfer()
        watchdog.idle()
    finally:
        watchdog.stop()
    text = caplog_vllm.text
    assert "Engine core has spent 0 s in one engine step" in text
    assert "step_waiting_on_a_transfer" in text
    assert "Engine core engine step finished after 0.6" in text


def test_engine_loop_watchdog_ignores_an_idle_loop(caplog_vllm):
    watchdog = EngineLoopWatchdog(threshold=0.1, poll_interval=0.02)
    try:
        watchdog.busy("engine step")
        watchdog.idle()
        time.sleep(0.3)
    finally:
        watchdog.stop()
    assert "Engine core" not in caplog_vllm.text


def test_add_request_waiting_in_the_input_queue_is_reported(caplog_vllm):
    added = []
    engine = SimpleNamespace(
        _loop_watchdog=None,
        _stall_warning_s=1.0,
        _add_received={"req-1": time.monotonic() - 5.0},
        _reject_add_in_shutdown=lambda request: False,
        add_request=lambda request, wave: added.append((request.request_id, wave)),
    )
    request = SimpleNamespace(request_id="req-1")
    EngineCoreProc._handle_client_request(
        engine, EngineCoreRequestType.ADD, (request, 0)
    )
    assert added == [("req-1", 0)]
    assert engine._add_received == {}
    assert "Request req-1 waited 5." in caplog_vllm.text
