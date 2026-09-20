# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler drain ownership; no model or checkpoint is needed."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.v1.core.sched.interface import PauseState
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState


@pytest.mark.parametrize("deferred", [False, True])
def test_shutdown_acknowledges_release_only_after_completed_drain(deferred):
    value = core()
    pause: Future[None] | None = Future() if deferred else None
    value.pause_scheduler = MagicMock(return_value=pause)
    executor = value.model_executor
    result = value.prepare_shutdown()
    assert value.prepare_shutdown() is result
    if deferred:
        assert pause is not None
        executor.shutdown.assert_not_called()
        pause.set_result(None)
    assert result.result() is None
    executor.shutdown.assert_called_once()
    assert value.model_executor is None
    with pytest.raises(RuntimeError, match="closing"):
        value.resume_scheduler()
    with pytest.raises(RuntimeError, match="closing"):
        value.residency_maintenance({})


def test_failed_worker_close_stays_paused_and_reports_failure():
    value = core()
    value.pause_scheduler = MagicMock(return_value=None)
    value.model_executor.shutdown.side_effect = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        value.prepare_shutdown().result()
    assert value.scheduler.pause_state == PauseState.PAUSED_ALL
    with pytest.raises(RuntimeError, match="closing"):
        value.resume_scheduler()


def test_async_close_waits_through_cancellation_and_closes_once(monkeypatch):
    async def scenario():
        value = object.__new__(AsyncLLM)
        released = asyncio.Event()
        value.engine_core = MagicMock()
        value.engine_core.prepare_shutdown_async = AsyncMock(side_effect=released.wait)
        value.renderer = MagicMock()
        prometheus = MagicMock()
        monkeypatch.setattr("vllm.v1.engine.async_llm.shutdown_prometheus", prometheus)
        task = asyncio.create_task(value.shutdown_async())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        value.engine_core.shutdown.assert_not_called()
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        value.engine_core.shutdown.assert_called_once()
        value.renderer.shutdown.assert_called_once()
        # Explicit completion must not call module globals during finalization.
        monkeypatch.setattr("vllm.v1.engine.async_llm.shutdown_prometheus", None)
        value.shutdown()
        value.__del__()
        prometheus.assert_called_once()

    asyncio.run(scenario())


def core(deferred=False):
    value = object.__new__(EngineCoreProc)
    value.vllm_config = SimpleNamespace(
        additional_config={"b12x_expert_cache": {"mode": "adaptive"}},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            enable_expert_parallel=False,
        ),
    )
    value.model_executor = MagicMock()
    value.scheduler = MagicMock(pause_state=PauseState.UNPAUSED)
    value.scheduler.set_pause_state.side_effect = lambda state: setattr(
        value.scheduler, "pause_state", state
    )
    value.scheduler.has_requests.return_value = False
    value.batch_queue = None
    value.engines_running = deferred
    value._idle_state_callbacks = []
    return value


@pytest.mark.parametrize("deferred", [False, True])
def test_maintenance_waits_for_scheduler_and_device_without_clearing_kv(deferred):
    value = core(deferred)
    order = []

    def rpc(method, **kwargs):
        order.append(method)
        assert value.scheduler.pause_state == PauseState.PAUSED_ALL
        return [{"status": "complete"}]

    value.model_executor.collective_rpc.side_effect = rpc
    value._reset_caches = MagicMock()
    result = value.residency_maintenance({"session": "one"})
    assert isinstance(result, Future)
    if deferred:
        assert not result.done() and order == []
        with pytest.raises(RuntimeError, match="pending"):
            value.resume_scheduler()
        with pytest.raises(RuntimeError, match="owns"):
            value.pause_scheduler(mode="abort")
        value.engines_running = False
        value._notify_idle_state_callbacks()
    assert result.result()["worker"]["status"] == "complete"
    assert order == ["synchronize_device", "b12x_residency_maintenance"]
    assert value.scheduler.pause_state == PauseState.UNPAUSED
    value._reset_caches.assert_not_called()


@pytest.mark.parametrize("stage", ["synchronize_device", "b12x_residency_maintenance"])
def test_failed_maintenance_cannot_be_resumed(stage):
    value = core()

    def rpc(method, **kwargs):
        if method == stage:
            raise RuntimeError("injected failure")
        return [{"status": "complete"}]

    value.model_executor.collective_rpc.side_effect = rpc
    with pytest.raises(RuntimeError, match="injected"):
        value.residency_maintenance({}).result()
    assert value.scheduler.pause_state == PauseState.PAUSED_ALL
    with pytest.raises(RuntimeError, match="reload"):
        value.resume_scheduler()


def test_ordinary_engine_has_no_maintenance_behavior():
    value = core()
    value.vllm_config.additional_config = {}
    with pytest.raises(ValueError, match="opted-in"):
        value.residency_maintenance({})
    value.model_executor.collective_rpc.assert_not_called()
    value.scheduler.set_pause_state.assert_not_called()


def test_idle_callback_resume_does_not_wait_for_another_client_message():
    value = core(deferred=True)
    value.scheduler.has_requests.side_effect = lambda: (
        value.scheduler.pause_state == PauseState.UNPAUSED
    )
    value.model_executor.collective_rpc.return_value = [{"status": "complete"}]
    result = value.residency_maintenance({})
    assert not result.done()
    value.engines_running = False
    value.shutdown_state = EngineShutdownState.RUNNING
    value.input_queue = MagicMock()
    value.input_queue.empty.return_value = True
    value._process_input_queue()
    assert result.result()["worker"]["status"] == "complete"
    value.input_queue.get.assert_not_called()


def test_client_cancellation_waits_for_engine_owned_transaction():
    async def run():
        started, release = asyncio.Event(), asyncio.Event()
        finished = []

        async def operation(config):
            started.set()
            await release.wait()
            finished.append(True)
            return {}

        client = SimpleNamespace(
            engine_core=SimpleNamespace(residency_maintenance_async=operation)
        )
        task = asyncio.create_task(AsyncLLM.residency_maintenance(client, {}))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished == [True]

    asyncio.run(run())


def test_shutdown_drain_failure_never_releases_live_worker():
    value = core()
    value.pause_scheduler = MagicMock(side_effect=RuntimeError("drain failed"))
    with pytest.raises(RuntimeError, match="drain failed"):
        value.prepare_shutdown().result()
    value.model_executor.shutdown.assert_not_called()
    assert value.scheduler.pause_state == PauseState.PAUSED_ALL
