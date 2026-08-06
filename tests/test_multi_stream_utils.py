# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager

import pytest
import torch

from vllm.utils.multi_stream_utils import (
    CUDAGraphCaptureEventPool,
    execute_in_parallel,
    is_vllm_cudagraph_capture_active,
    vllm_cudagraph_capture_scope,
)


class _FakeEvent:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def record(self) -> None:
        self.calls.append(f"{self.name}.record")

    def wait(self) -> None:
        self.calls.append(f"{self.name}.wait")


def test_vllm_cudagraph_capture_scope_is_nested_and_exception_safe() -> None:
    assert not is_vllm_cudagraph_capture_active()
    with pytest.raises(RuntimeError), vllm_cudagraph_capture_scope():
        assert is_vllm_cudagraph_capture_active()
        with vllm_cudagraph_capture_scope():
            assert is_vllm_cudagraph_capture_active()
        raise RuntimeError("capture failed")
    assert not is_vllm_cudagraph_capture_active()


def test_cudagraph_capture_event_pool_isolates_capture_generations(monkeypatch):
    created = []

    def fake_event():
        event = object()
        created.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", fake_event)
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    pool = CUDAGraphCaptureEventPool(2)
    with pool.lease() as default_events:
        assert default_events is pool.default_events

    with pool.lease(private_eager=True) as eager_a:
        pass
    with pool.lease(private_eager=True) as eager_b:
        assert eager_a is not eager_b
        assert set(map(id, eager_a)).isdisjoint(map(id, eager_b))
    assert not pool._captured_event_sets

    with (
        vllm_cudagraph_capture_scope(),
        pool.lease(private_eager=True) as scoped_capture,
    ):
        assert set(map(id, eager_a)).isdisjoint(map(id, scoped_capture))
    assert pool._captured_event_sets == [scoped_capture]

    capturing = True
    with pool.lease() as capture_a:
        pass
    with pool.lease() as capture_b:
        assert capture_a is not capture_b
        assert set(map(id, capture_a)).isdisjoint(map(id, capture_b))
        assert set(map(id, pool.default_events)).isdisjoint(map(id, capture_a))
    assert len(pool._captured_event_sets) == 3
    assert len(created) == 12


@pytest.mark.parametrize("enqueue_default_first", [False, True])
def test_execute_in_parallel_enqueue_order(monkeypatch, enqueue_default_first):
    calls: list[str] = []

    @contextmanager
    def fake_stream(_stream):
        calls.append("aux.enter")
        try:
            yield
        finally:
            calls.append("aux.exit")

    monkeypatch.setattr(torch.cuda, "stream", fake_stream)
    start = _FakeEvent("start", calls)
    done = _FakeEvent("done", calls)

    default_result, aux_results = execute_in_parallel(
        lambda: calls.append("default") or "default-result",
        [lambda: calls.append("aux") or "aux-result"],
        start,
        [done],
        [object()],
        enable=True,
        enqueue_default_first=enqueue_default_first,
    )

    assert default_result == "default-result"
    assert aux_results == ["aux-result"]
    assert calls == (
        [
            "start.record",
            "default",
            "aux.enter",
            "start.wait",
            "aux",
            "done.record",
            "aux.exit",
            "done.wait",
        ]
        if enqueue_default_first
        else [
            "start.record",
            "aux.enter",
            "start.wait",
            "aux",
            "done.record",
            "aux.exit",
            "default",
            "done.wait",
        ]
    )
