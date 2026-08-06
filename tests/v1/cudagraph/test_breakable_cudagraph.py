# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the breakable cudagraph primitives.
"""

from __future__ import annotations

import os
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"


def test_memory_profile_cleanup_resets_backend_cache_bindings(monkeypatch):
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    events: list[str] = []
    old_cache = torch.ones((1,), dtype=torch.uint8)

    class Impl:
        def reset_kv_cache_binding_state(self):
            events.append("reset")
            assert layer.kv_cache is old_cache

    layer = SimpleNamespace(impl=Impl(), kv_cache=old_cache)
    runner = model_runner_module.GPUModelRunner.__new__(
        model_runner_module.GPUModelRunner
    )
    runner.cudagraph_manager = None
    runner.speculator = None
    runner.kv_caches = [old_cache]
    runner.attn_groups = []
    runner.kv_cache_config = object()
    runner.block_tables = object()
    runner.kernel_block_sizes = object()
    runner.kv_connector = object()
    runner.kv_block_zeroer = object()
    runner.verification_capacity_manager = object()
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.compilation_config = SimpleNamespace(
        static_forward_context={"layer": layer}
    )

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(model_runner_module.gc, "collect", lambda: None)

    runner._cleanup_cudagraph_memory_profile()

    assert events == ["reset"]
    assert runner.kv_caches == []
    assert layer.kv_cache.numel() == 0
    assert runner.cache_config.num_gpu_blocks is None


def test_cudagraph_manager_clear_releases_capture_state():
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    manager = ModelCudaGraphManager.__new__(ModelCudaGraphManager)
    manager.graphs = {object(): object()}
    manager._graphs_captured = True
    manager.breakable_cg_runner = object()
    manager.hidden_states = object()
    manager.aux_hidden_states = [object()]
    manager.intermediate_tensors = object()

    manager.clear()

    assert manager.graphs == {}
    assert not manager._graphs_captured
    assert manager.breakable_cg_runner is None
    assert manager.hidden_states is None
    assert manager.aux_hidden_states == []
    assert manager.intermediate_tensors is None


def test_memory_profile_destroys_graphs_before_restoring_pools(monkeypatch):
    from vllm.v1.worker import workspace as workspace_module
    from vllm.v1.worker.gpu import model_runner as model_runner_module

    profile_pool = object()
    production_pool = object()
    events: list[str] = []
    lazy_wrappers: list[FakeWrapper] = []

    class FakeManager:
        def __init__(self):
            self.pool = production_pool

        def needs_capture(self):
            return True

        def capture(self, *args, **kwargs):
            assert self.pool is profile_pool
            assert wrapper.graph_pool is profile_pool
            lazy_wrapper = FakeWrapper()
            lazy_wrapper.graph_pool = self.pool
            lazy_wrappers.append(lazy_wrapper)
            events.append("capture")

    class FakeWrapper:
        def __init__(self):
            self.graph_pool = production_pool

    class FakeSpeculator:
        def get_cudagraph_managers(self):
            return []

        def capture(self):
            assert workspace_module._workspace_lane.get() == 1
            events.append("spec_capture")

    manager = FakeManager()
    wrapper = FakeWrapper()
    runner = model_runner_module.GPUModelRunner.__new__(
        model_runner_module.GPUModelRunner
    )
    runner.vllm_config = object()
    runner.cudagraph_manager = manager
    runner.speculator = FakeSpeculator()
    runner.lora_config = None
    runner.model = object()
    runner.model_state = object()
    runner.input_buffers = object()
    runner.intermediate_tensors = object()
    runner.block_tables = object()
    runner.attn_groups = object()
    runner.kv_cache_config = object()
    runner.use_aux_hidden_state_outputs = False
    runner._init_minimal_kv_cache_for_profiling = lambda: None
    runner.maybe_setup_dummy_loras = lambda _: nullcontext()
    runner._zero_cudagraph_capture_kv_blocks = lambda: None

    def cleanup():
        events.append("cleanup")
        assert manager.pool is profile_pool
        assert wrapper.graph_pool is profile_pool
        assert lazy_wrappers[0].graph_pool is profile_pool

    runner._cleanup_cudagraph_memory_profile = cleanup

    memory_info = iter(((1000, 0), (900, 0), (950, 0)))
    monkeypatch.setattr(
        model_runner_module, "set_current_vllm_config", lambda _: nullcontext()
    )
    monkeypatch.setattr(
        model_runner_module,
        "current_platform",
        SimpleNamespace(
            graph_pool_handle=lambda: profile_pool,
            get_global_graph_pool=lambda: production_pool,
        ),
    )
    monkeypatch.setattr(
        model_runner_module.CUDAGraphWrapper, "_all_instances", [wrapper]
    )
    monkeypatch.setattr(
        model_runner_module.BreakableCUDAGraphWrapper,
        "_all_instances",
        lazy_wrappers,
    )
    monkeypatch.setattr(model_runner_module.gc, "collect", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "get_memory_info", lambda: next(memory_info))
    monkeypatch.setattr(
        model_runner_module,
        "checkpoint_b12x_graph_channels",
        lambda: events.append("checkpoint") or ("channel-checkpoint",),
    )
    monkeypatch.setattr(
        model_runner_module,
        "rollback_b12x_graph_channels",
        lambda checkpoint: (
            events.append("rollback")
            if checkpoint == ("channel-checkpoint",)
            else pytest.fail("rollback received the wrong channel checkpoint")
        ),
    )

    assert runner.profile_cudagraph_memory() == 50
    assert events == [
        "checkpoint",
        "capture",
        "spec_capture",
        "cleanup",
        "rollback",
    ]
    assert workspace_module._workspace_lane.get() == 0
    assert manager.pool is production_pool
    assert wrapper.graph_pool is production_pool
    assert lazy_wrappers[0].graph_pool is production_pool


def test_breakable_runner_inherits_active_manager_pool(monkeypatch):
    from vllm.v1.worker.gpu import cudagraph_utils

    active_pool = object()

    class FakeBreakableWrapper:
        def __init__(self, model, vllm_config):
            self.model = model
            self.vllm_config = vllm_config
            self.graph_pool = object()

    monkeypatch.setattr(
        cudagraph_utils, "BreakableCUDAGraphWrapper", FakeBreakableWrapper
    )
    manager = cudagraph_utils.CudaGraphManager.__new__(cudagraph_utils.CudaGraphManager)
    manager.breakable_cg_runner = None
    manager.vllm_config = object()
    manager.pool = active_pool
    model = object()

    manager.init_breakable_cg_runner(model)

    assert manager.breakable_cg_runner is not None
    assert manager.breakable_cg_runner.model is model
    assert manager.breakable_cg_runner.graph_pool is active_pool


def test_piecewise_capture_builds_fresh_metadata_for_both_passes():
    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import (
        BatchExecutionDescriptor,
        CudaGraphManager,
    )

    manager = CudaGraphManager.__new__(CudaGraphManager)
    desc = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 8, None)
    manager.device = torch.device("cpu")
    manager._capture_descs = {CUDAGraphMode.PIECEWISE: [desc]}
    manager._graphs_captured = False
    manager.use_breakable_cg = True

    create_calls = []
    forward_calls = []

    def create_forward_fn(desc_arg, warmup):
        assert desc_arg == desc
        metadata = {"layer": object()}
        create_calls.append((warmup, metadata))

        def forward_fn(cg_mode):
            assert metadata
            forward_calls.append((warmup, cg_mode, metadata))

        return forward_fn

    with (
        patch(
            "vllm.v1.worker.gpu.cudagraph_utils.graph_capture",
            return_value=nullcontext(),
        ),
        patch(
            "vllm.v1.worker.gpu.cudagraph_utils.is_global_first_rank",
            return_value=False,
        ),
    ):
        manager.capture(create_forward_fn)

    assert [warmup for warmup, _ in create_calls] == [True, False]
    assert [mode for _, mode, _ in forward_calls] == [
        CUDAGraphMode.NONE,
        CUDAGraphMode.PIECEWISE,
    ]
    assert create_calls[0][1] is not create_calls[1][1]


@pytest.fixture(autouse=True)
def _reset_breakable_tls():
    """Defensively clear thread-local capture state between tests so a
    failure in one test can't leak "nested capture" errors into the next."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    BreakableCUDAGraphCapture._tls.active = None
    yield
    BreakableCUDAGraphCapture._tls.active = None


@pytest.fixture
def cuda_capture_stream():
    """A non-default CUDA stream suitable for cudagraph capture.

    ``CUDAGraph.capture_begin`` refuses to capture from the default
    stream, so all capture-using tests need to run under
    ``torch.cuda.stream(...)`` for a separate stream.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        yield stream
    torch.cuda.current_stream().wait_stream(stream)


# ---------------------------------------------------------------------------
# eager_break_during_capture: outside capture
# ---------------------------------------------------------------------------


def test_decorator_passthrough_outside_capture():
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    calls = []

    @eager_break_during_capture
    def f(x):
        calls.append(x)
        return x * 2

    assert f(3) == 6
    assert calls == [3]


# ---------------------------------------------------------------------------
# BreakableCUDAGraphCapture: thread-local + nested rejection
# ---------------------------------------------------------------------------


def test_current_is_none_when_inactive():
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    assert BreakableCUDAGraphCapture.current() is None
    assert BreakableCUDAGraphCapture.is_active() is False


def test_thread_local_active_during_context(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    cap = BreakableCUDAGraphCapture()
    with cap:
        assert BreakableCUDAGraphCapture.current() is cap
        assert BreakableCUDAGraphCapture.is_active() is True
    assert BreakableCUDAGraphCapture.current() is None


def test_nested_capture_raises(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    outer = BreakableCUDAGraphCapture()
    inner = BreakableCUDAGraphCapture()
    with outer, pytest.raises(RuntimeError, match="Nested.*not supported"), inner:
        pass


def test_active_state_isolated_across_threads(cuda_capture_stream):
    """Verify the thread-local 'active capture' slot is per-thread.

    We don't run concurrent captures here -- CUDA only supports one
    in-flight capture per stream and we keep tests cheap. We just check
    that the worker thread sees its own slot as None while the main
    thread has a capture active.
    """
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    worker_view: dict[str, BreakableCUDAGraphCapture | None] = {}

    def worker():
        worker_view["state"] = BreakableCUDAGraphCapture.current()

    main_cap = BreakableCUDAGraphCapture()
    with main_cap:
        # Main thread has a live capture.
        assert BreakableCUDAGraphCapture.current() is main_cap
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    # Worker thread saw None -- thread-local separation works.
    assert worker_view["state"] is None
    # Main thread's slot is cleared on exit.
    assert BreakableCUDAGraphCapture.current() is None


# ---------------------------------------------------------------------------
# Segment list construction
# ---------------------------------------------------------------------------


def test_capture_with_no_eager_break_records_one_graph(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    x = torch.zeros(4, device="cuda")
    cap = BreakableCUDAGraphCapture()
    with cap:
        x.add_(1.0)
    assert len(cap.segments) == 1
    assert cap.num_graphs == 1
    assert cap.num_eager_breaks == 0


def test_add_eager_creates_alternating_graph_eager_graph(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    x = torch.zeros(4, device="cuda")
    counter = {"eager_calls": 0}

    def eager_step():
        counter["eager_calls"] += 1
        x.add_(10.0)

    cap = BreakableCUDAGraphCapture()
    with cap:
        x.add_(1.0)
        cap.add_eager(eager_step)
        x.add_(1.0)
        cap.add_eager(eager_step)
        x.add_(1.0)
    # 3 graph segments + 2 eager segments, interleaved as G E G E G.
    assert len(cap.segments) == 5
    assert cap.num_graphs == 3
    assert cap.num_eager_breaks == 2
    # Eager fn is stored as-is in the segment list, so we can confirm
    # the alternation pattern by identity check.
    assert cap.segments[1] is eager_step
    assert cap.segments[3] is eager_step
    assert counter["eager_calls"] == 2  # only the in-capture invocation


# ---------------------------------------------------------------------------
# Capture vs eager numerical equivalence
# ---------------------------------------------------------------------------


def test_capture_replay_matches_eager_simple(cuda_capture_stream):
    """Verify that replay reproduces the same end-state as a single eager
    forward, with an eager break in the middle.

    Note: during capture, the *captured* kernels are recorded but NOT
    executed (that's CUDA-graph semantics). Only the eager segments
    actually mutate state at capture time. So we check correctness after
    ``replay()``, not after ``with cap:`` exits.
    """
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    x = torch.zeros(8, device="cuda")
    log: list[str] = []

    def eager_break_op():
        x.mul_(2.0)
        log.append("eager")

    cap = BreakableCUDAGraphCapture()
    with cap:
        x.add_(1.0)  # recorded into graph[0]
        cap.add_eager(eager_break_op)  # runs eagerly: x *= 2
        x.add_(5.0)  # recorded into graph[1]

    # Capture-time: graph kernels were recorded only; eager segment ran
    # once on x == 0, leaving x == 0.
    torch.accelerator.synchronize()
    assert torch.equal(x, torch.zeros(8, device="cuda"))
    assert log == ["eager"]

    # Replay with a fresh input: 10 -> 11 -> 22 -> 27.
    x.fill_(10.0)
    cap.replay()
    torch.accelerator.synchronize()
    assert torch.equal(x, torch.full((8,), 27.0, device="cuda"))
    assert log == ["eager", "eager"]

    # Replay again with another input: 100 -> 101 -> 202 -> 207.
    x.fill_(100.0)
    cap.replay()
    torch.accelerator.synchronize()
    assert torch.equal(x, torch.full((8,), 207.0, device="cuda"))
    assert log == ["eager", "eager", "eager"]


def test_decorator_breaks_when_invoked_inside_capture(cuda_capture_stream):
    """Verify @eager_break_during_capture correctly routes through
    add_eager when inside a capture context, and runs straight through
    when there's no active capture."""
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    @eager_break_during_capture
    def attention_like(t: torch.Tensor) -> None:
        # In-place double; stands in for "real" attention work.
        t.mul_(2.0)

    x = torch.zeros(4, device="cuda")

    # Outside capture: decorator should just call through.
    x.fill_(3.0)
    attention_like(x)
    torch.accelerator.synchronize()
    assert torch.equal(x, torch.full((4,), 6.0, device="cuda"))

    # Inside capture: decorator should split the graph. Only the eager
    # segment actually mutates state during capture.
    x.fill_(0.0)
    cap = BreakableCUDAGraphCapture()
    with cap:
        x.add_(5.0)  # recorded
        attention_like(x)  # eager: x *= 2 (on x == 0, no-op)
        x.add_(1.0)  # recorded
    torch.accelerator.synchronize()
    assert torch.equal(x, torch.zeros(4, device="cuda"))
    # 2 graph segments + 1 eager segment, ordered G E G; the arithmetic
    # equivalence check below verifies the ordering.
    assert len(cap.segments) == 3
    assert cap.num_graphs == 2
    assert cap.num_eager_breaks == 1

    # Replay: 2 -> 7 -> 14 -> 15.
    x.fill_(2.0)
    cap.replay()
    torch.accelerator.synchronize()
    assert torch.equal(x, torch.full((4,), 15.0, device="cuda"))


# ---------------------------------------------------------------------------
# Replay ordering
# ---------------------------------------------------------------------------


def test_replay_invokes_eager_segments_in_order(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    log: list[str] = []
    x = torch.zeros(1, device="cuda")

    def make_eager(name):
        def step():
            log.append(name)
            x.add_(1.0)

        return step

    cap = BreakableCUDAGraphCapture()
    with cap:
        x.add_(1.0)
        cap.add_eager(make_eager("A"))
        x.add_(1.0)
        cap.add_eager(make_eager("B"))
        x.add_(1.0)
        cap.add_eager(make_eager("C"))
        x.add_(1.0)

    # Capture-time invocation order
    assert log == ["A", "B", "C"]

    log.clear()
    cap.replay()
    torch.accelerator.synchronize()
    assert log == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Capture cleanup releases thread-local even if body raises
# ---------------------------------------------------------------------------


def test_exception_in_body_clears_active(cuda_capture_stream):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

    cap = BreakableCUDAGraphCapture()
    with pytest.raises(RuntimeError, match="boom"), cap:
        raise RuntimeError("boom")

    # active must be reset even after an exception inside the body
    assert BreakableCUDAGraphCapture.current() is None


# ---------------------------------------------------------------------------
# Nested decorated ops: inner op must not trigger a recursive eager break
# ---------------------------------------------------------------------------


def test_nested_decorated_op_runs_inline(cuda_capture_stream):
    """A decorated op invoked from inside another decorated op's eager
    body must execute inline -- starting a second eager break mid-flight
    corrupts the segment state and explodes ``_begin_segment``'s assert.

    This mirrors the deepseek_v4_attention case where the outer attention
    op's impl internally dispatches sparse_attn_indexer (also decorated).
    """
    from vllm.compilation.breakable_cudagraph import (
        BreakableCUDAGraphCapture,
        eager_break_during_capture,
    )

    x = torch.zeros(4, device="cuda")
    inner_calls = 0

    @eager_break_during_capture
    def inner_op(t: torch.Tensor) -> None:
        nonlocal inner_calls
        inner_calls += 1
        t.add_(1.0)

    @eager_break_during_capture
    def outer_op(t: torch.Tensor) -> None:
        # outer body calls another decorated op -- this is the case that
        # used to assert in _begin_segment.
        inner_op(t)
        t.add_(10.0)

    cap = BreakableCUDAGraphCapture()
    with cap:
        x.add_(2.0)  # recorded in graph[0]
        outer_op(x)  # one eager break, inner runs inline
        x.add_(100.0)  # recorded in graph[1]

    # Exactly one eager break (the outer); inner must NOT add a second.
    assert cap.num_graphs == 2
    assert cap.num_eager_breaks == 1
    assert inner_calls == 1  # only the capture-time invocation

    x.fill_(0.0)
    cap.replay()
    torch.accelerator.synchronize()
    # 0 -> +2 -> +1 (inner) -> +10 (outer) -> +100 = 113
    assert torch.equal(x, torch.full((4,), 113.0, device="cuda"))
    assert inner_calls == 2  # replay invokes the outer's lambda again
