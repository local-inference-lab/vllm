#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GPUModelRunner (V2) CUDA graph memory profiling.

These exercise the orchestration of ``profile_cudagraph_memory`` on CPU by
building a runner via ``__new__`` and faking the GPU-only helpers, so the
control flow (bootstrap -> sample FULL graphs into a throwaway pool ->
extrapolate -> teardown) is covered without a GPU.
See https://github.com/vllm-project/vllm/issues/49224.
"""

import contextlib
import gc
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from weakref import ref

import pytest
import torch

from tests.utils import create_new_process_for_each_test
from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu import model_runner as mrv2

GLOBAL_POOL = "global-pool"
THROWAWAY_POOL = "throwaway-pool"


class _FakeCudaGraphManager(cgu.CudaGraphManager):
    def __init__(
        self, needs_capture: bool, num_full_descs: int, piecewise_only: bool = False
    ) -> None:
        self._needs_capture = needs_capture
        self.pool: Any = GLOBAL_POOL
        descs = [
            SimpleNamespace(num_tokens=num_tokens)
            for num_tokens in range(num_full_descs, 0, -1)
        ]
        if piecewise_only:
            self._capture_descs = {CUDAGraphMode.PIECEWISE: descs}
        else:
            self._capture_descs = {CUDAGraphMode.FULL: descs} if needs_capture else {}
        # Profiling hooks set by profile_cudagraph_memory.
        self._sample_full_descs = False
        self._capture_mem_samples: list[Any] | None = None
        self.use_breakable_cg = False
        self.graphs: dict[Any, Any] = {}
        self.graph_capture_resources: dict[Any, list[Any]] = {}
        self._graphs_captured = False

    def needs_capture(self) -> bool:
        return self._needs_capture


class _RecordingGraph:
    def __init__(self, lifecycle: list[str], name: str) -> None:
        self.lifecycle = lifecycle
        self.name = name

    def reset(self) -> None:
        self.lifecycle.append(f"reset-{self.name}")


class _RecordingDict(dict[Any, Any]):
    def __init__(self, lifecycle: list[str], name: str) -> None:
        super().__init__()
        self.lifecycle = lifecycle
        self.name = name

    def clear(self) -> None:
        self.lifecycle.append(f"clear-{self.name}")
        super().clear()


def _make_profiling_runner(
    cudagraph_mode: CUDAGraphMode,
    *,
    needs_capture: bool = True,
    num_full_descs: int = 3,
    piecewise_only: bool = False,
    captured_bytes: int = 7 << 30,
    mem_samples: list[tuple[int, int, int]] | None = None,
) -> Any:
    """Build a runner whose capture records ``(desc index, growth, cost)``."""
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
    runner.cudagraph_manager = _FakeCudaGraphManager(
        needs_capture, num_full_descs, piecewise_only
    )
    runner.vllm_config = SimpleNamespace()
    runner.speculator = None

    events: list[str] = []
    runner.events = events
    runner.pool_during_capture = None

    def _capture_model(*, profile_only: bool = False) -> int:
        assert profile_only
        events.append("capture")
        runner.pool_during_capture = runner.cudagraph_manager.pool
        # Simulate the manager's per-FULL-graph memory sampling.
        manager = runner.cudagraph_manager
        samples = manager._capture_mem_samples
        if samples is not None and mem_samples:
            descs = manager._capture_descs[CUDAGraphMode.FULL]
            samples.extend(
                cgu._FullGraphMemorySample(descs[index], growth, cost)
                for index, growth, cost in mem_samples
            )
        return captured_bytes

    runner.capture_model = _capture_model
    return runner


class _FakePlatform:
    """Stands in for current_platform; the global graph pool is a class attr,
    matching vllm.platforms.Platform's lazy singleton."""

    _global_graph_pool: Any = GLOBAL_POOL

    @staticmethod
    def graph_pool_handle() -> Any:
        return THROWAWAY_POOL

    def get_global_graph_pool(self) -> Any:
        return type(self)._global_graph_pool


def _patch_module(monkeypatch) -> None:
    @contextlib.contextmanager
    def _fake_set_current_vllm_config(_cfg):
        yield

    monkeypatch.setattr(cgu, "set_current_vllm_config", _fake_set_current_vllm_config)
    _FakePlatform._global_graph_pool = GLOBAL_POOL
    monkeypatch.setattr(cgu, "current_platform", _FakePlatform())
    monkeypatch.setattr(
        cgu, "_init_minimal_kv_cache_for_profiling", lambda r: r.events.append("init")
    )
    monkeypatch.setattr(
        cgu, "_teardown_profiling_state", lambda r: r.events.append("teardown")
    )
    # The profiler reads free GPU memory before/after to compute what it
    # retained; default to a constant (nothing retained).
    monkeypatch.setattr(cgu.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(cgu.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(cgu.torch.accelerator, "memory_reserved", lambda: 0)
    monkeypatch.setattr(
        cgu.torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )


def test_profile_cudagraph_memory_disabled_returns_zero(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.NONE)

    result = cgu.profile_cudagraph_memory(runner)

    assert result == 0
    # No KV-cache bootstrap or teardown when cudagraphs are disabled.
    assert runner.events == []


def test_profile_cudagraph_memory_no_graphs_tears_down(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL, needs_capture=False)

    result = cgu.profile_cudagraph_memory(runner)

    assert result == 0
    # Bootstrapped then cleaned up, without capturing or touching the pool.
    assert runner.events == ["init", "teardown"]
    assert runner.cudagraph_manager.pool == GLOBAL_POOL


def test_profile_cudagraph_memory_samples_and_extrapolates(monkeypatch):
    _patch_module(monkeypatch)
    mib = 1 << 20
    # The measured delta includes the sampled FULL graphs of 4, 3 and 1
    # tokens. The unsampled 2-token graph costs the per-graph samples
    # interpolated between 1 token (10 MiB) and 3 tokens (30 MiB).
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL,
        num_full_descs=4,
        captured_bytes=1000 * mib,
        mem_samples=[
            (0, 100 * mib, 100 * mib),
            (1, 30 * mib, 30 * mib),
            (3, 0, 10 * mib),
        ],
    )

    result = cgu.profile_cudagraph_memory(
        runner, lambda: runner.events.append("prepare")
    )

    assert result == (1000 + 20) * mib
    # Bootstrap, capture, and teardown run in order.
    assert runner.events == ["init", "prepare", "capture", "teardown"]
    # Capture must use a throwaway pool, not the persistent global pool.
    assert runner.pool_during_capture == THROWAWAY_POOL
    assert runner.cudagraph_manager.pool == GLOBAL_POOL
    # FULL capture must be limited to a sample of the graphs.
    assert runner.cudagraph_manager._sample_full_descs


def test_profile_cudagraph_memory_debug_accounting_captures_every_full_graph(
    monkeypatch,
):
    _patch_module(monkeypatch)
    monkeypatch.setattr(cgu, "_DEBUG_GRAPH_MEMORY_ACCOUNTING", True)
    mib = 1 << 20
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL,
        num_full_descs=2,
        captured_bytes=1000 * mib,
        mem_samples=[(0, 100 * mib, 100 * mib), (1, 0, 50 * mib)],
    )

    # Every graph was captured, so the measured delta is the estimate.
    assert cgu.profile_cudagraph_memory(runner) == 1000 * mib
    assert not runner.cudagraph_manager._sample_full_descs


def test_profile_cudagraph_memory_piecewise_only_returns_measured(monkeypatch):
    _patch_module(monkeypatch)
    captured_bytes = 5 << 30
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        piecewise_only=True,
        captured_bytes=captured_bytes,
    )

    result = cgu.profile_cudagraph_memory(runner)

    # No FULL graphs to sample or extrapolate: the measured delta is exact.
    assert result == captured_bytes


@pytest.mark.parametrize(
    "init_native,capture_native,retained_native",
    [
        (0, 256, 256),
        (512, 256, 64),
        (0, 256, 0),
        (512, 0, 0),
        (0, 0, 256),
        (0, 256, 512),
    ],
)
def test_profile_cudagraph_memory_records_native_growth_inside_capture_only(
    monkeypatch, init_native, capture_native, retained_native
):
    """Worker reconciles capture growth after all prepared plans are released."""
    _patch_module(monkeypatch)
    mib = 1 << 20
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL_AND_PIECEWISE,
        piecewise_only=True,
        captured_bytes=1000 * mib,
    )

    def memory_info():
        native = init_native
        if "teardown" in runner.events:
            native += retained_native
        elif "capture" in runner.events:
            native += capture_native
        # Torch-owned memory is not native initialization and is fully reserved.
        return ((4096 - native - 512) * mib, 4096 * mib)

    monkeypatch.setattr(cgu.torch.accelerator, "get_memory_info", memory_info)
    monkeypatch.setattr(cgu.torch.accelerator, "memory_reserved", lambda: 512 * mib)

    assert cgu.profile_cudagraph_memory(runner) == 1000 * mib
    assert runner.cudagraph_native_memory_profile == (
        init_native * mib,
        (init_native + capture_native) * mib,
        1000 * mib,
    )


def test_profile_cudagraph_memory_tears_down_on_capture_error(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)

    def _boom(*, profile_only: bool = False) -> int:
        runner.events.append("capture")
        raise RuntimeError("capture failed")

    runner.capture_model = _boom

    try:
        cgu.profile_cudagraph_memory(runner)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected capture error to propagate")

    # Teardown still runs even if capture raises.
    assert runner.events == ["init", "capture", "teardown"]


def test_profile_cudagraph_memory_tears_down_on_partial_init_error(monkeypatch):
    real_teardown = cgu._teardown_profiling_state
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)
    runner.compilation_config.static_forward_context = {}
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: runner.events.append("teardown")

    def _partial_init(runner) -> None:
        runner.events.append("init")
        runner.kv_caches = [object()]
        runner.attn_groups = [[object()]]
        runner.kv_cache_config = object()
        raise RuntimeError("minimal cache initialization failed")

    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", _partial_init)
    monkeypatch.setattr(cgu, "_teardown_profiling_state", real_teardown)
    monkeypatch.setattr(cgu.torch.accelerator, "synchronize", lambda: None)

    try:
        cgu.profile_cudagraph_memory(runner)
    except RuntimeError as error:
        assert str(error) == "minimal cache initialization failed"
    else:
        raise AssertionError("expected initialization error to propagate")

    assert runner.events == ["init", "teardown"]
    assert runner.kv_caches == []
    assert runner.attn_groups == []
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cudagraph_manager is None
    assert runner.cache_config.num_gpu_blocks is None
    assert cgu.current_platform.get_global_graph_pool() == GLOBAL_POOL


def test_profile_cudagraph_memory_restores_compilation_counters(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)

    def _capture_model(*, profile_only: bool = False) -> int:
        compilation_counter.num_cudagraph_captured += 5
        compilation_counter.num_gpu_runner_capture_triggers += 1
        return 1 << 30

    runner.capture_model = _capture_model
    captured_before = compilation_counter.num_cudagraph_captured
    triggers_before = compilation_counter.num_gpu_runner_capture_triggers

    cgu.profile_cudagraph_memory(runner)

    # Profiling captures are discarded, so they must not inflate the
    # compilation counters; the real capture_model() runs later.
    assert compilation_counter.num_cudagraph_captured == captured_before
    assert compilation_counter.num_gpu_runner_capture_triggers == triggers_before


def test_model_runner_delegates_to_cudagraph_utils(monkeypatch):
    runner = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    prepare = lambda: None
    monkeypatch.setattr(
        mrv2,
        "_profile_cudagraph_memory",
        lambda r, callback: (r, callback),
    )
    assert runner.profile_cudagraph_memory(prepare) == (runner, prepare)


def _full_descs(*token_counts: int) -> list[cgu.BatchExecutionDescriptor]:
    return [
        cgu.BatchExecutionDescriptor(CUDAGraphMode.FULL, num_tokens, num_tokens)
        for num_tokens in token_counts
    ]


def test_profiling_samples_two_largest_and_smallest_full_graphs():
    descs = _full_descs(40, 32, 24, 16, 8)

    assert cgu._profiling_full_descs(descs) == [descs[0], descs[1], descs[-1]]
    assert cgu._profiling_full_descs(descs[:3]) == descs[:3]


def test_extrapolate_full_graph_memory():
    mib = 1 << 20
    descs = _full_descs(40, 32, 24, 16, 8)

    def sample(index: int, growth: int, cost: int | None = None) -> Any:
        return cgu._FullGraphMemorySample(
            descs[index], growth * mib, (growth if cost is None else cost) * mib
        )

    # No samples (e.g. no FULL graphs): nothing to add.
    assert cgu._extrapolate_full_graph_memory([], []) == 0
    # Sampled graphs cost exactly their measured growth.
    assert cgu._extrapolate_full_graph_memory([sample(0, 100)], descs[:1]) == (
        100 * mib
    )
    # Unsampled graphs interpolate the per-graph samples by token count.
    assert (
        cgu._extrapolate_full_graph_memory(
            [sample(0, 100), sample(1, 32), sample(4, 8)], descs
        )
        == (100 + 32 + 8 + 24 + 16) * mib
    )
    # A fixed retained cost is charged to every unsampled graph, even when
    # the sampled graphs found room for it in the pool, and one-time pool
    # growth of a sampled graph is not.
    assert (
        cgu._extrapolate_full_graph_memory(
            [sample(0, 300), sample(1, 80, 122), sample(4, 6, 122)], descs
        )
        == (300 + 80 + 6 + 2 * 122) * mib
    )
    # The largest graph's sample is not a per-graph reference, and the
    # per-graph cost is floored for driver overhead.
    assert (
        cgu._extrapolate_full_graph_memory(
            [sample(0, 300), sample(1, 0), sample(4, 0)], descs
        )
        == (300 + 2 * 1) * mib
    )


_MIB = 1 << 20
# Decode page metadata that a buggy attention backend retains in every FULL
# graph (the DeepSeek-V4.1 per-graph copy fixed in vLLM #918).
_METADATA_BYTES = 116 * _MIB
_GRAPH_EXEC_BYTES = 6 * _MIB


class _FakeDevice:
    """One device as the graph memory sampler reads it, in bytes.

    Eager blocks stay cached after they are freed until ``empty_cache()``.
    The private graph pool reuses its free blocks but never returns memory,
    and every captured graph adds ``_GRAPH_EXEC_BYTES`` outside the Torch
    allocator, like an instantiated CUDA graph.
    """

    def __init__(self) -> None:
        self.eager_allocated = 0
        self.eager_reserved = 0
        self.pool_allocated = 0
        self.pool_reserved = 0
        self.native = 0
        self.capturing = False
        # Cached bytes that torch.cuda.graph's own flush found on entry.
        self.flushed_by_capture: list[int] = []

    def get_memory_info(self) -> tuple[int, int]:
        total = 96 << 30
        used = self.eager_reserved + self.pool_reserved + self.native
        return total - used, total

    def memory_allocated(self) -> int:
        return self.eager_allocated + self.pool_allocated

    def memory_reserved(self) -> int:
        return self.eager_reserved + self.pool_reserved

    def empty_cache(self) -> None:
        self.eager_reserved = self.eager_allocated

    def allocate(self, nbytes: int) -> None:
        if self.capturing:
            self.pool_allocated += nbytes
            self.pool_reserved = max(self.pool_reserved, self.pool_allocated)
        else:
            self.eager_allocated += nbytes
            self.eager_reserved = max(self.eager_reserved, self.eager_allocated)

    def free(self, nbytes: int) -> None:
        if self.capturing:
            self.pool_allocated -= nbytes
        else:
            self.eager_allocated -= nbytes

    @contextlib.contextmanager
    def cuda_graph(self, *_args, **_kwargs):
        # Like torch.cuda.graph, flush the allocator cache before capturing.
        self.flushed_by_capture.append(self.eager_reserved - self.eager_allocated)
        self.empty_cache()
        self.capturing = True
        try:
            yield
        finally:
            self.capturing = False
            self.native += _GRAPH_EXEC_BYTES

    def create_forward_fn(self, retain_metadata: bool):
        """Decode forward: activations scale with tokens, metadata is fixed."""

        def create_forward_fn(desc, warmup):
            def forward_fn(_mode):
                activations = 2 * _MIB * desc.num_tokens
                self.allocate(activations + _METADATA_BYTES)
                self.free(activations)
                if not (self.capturing and retain_metadata):
                    self.free(_METADATA_BYTES)

            return forward_fn

        return create_forward_fn

    def capture_model(self, manager: cgu.CudaGraphManager, create_forward_fn):
        """Mirror GPUModelRunner.capture_model's measured delta."""
        self.empty_cache()
        free_before = self.get_memory_info()[0]
        manager.capture(create_forward_fn)
        return free_before - self.get_memory_info()[0]


class _CapturingGraphManager(cgu.CudaGraphManager):
    """Runs the real capture loop over FULL decode graphs of the given sizes."""

    def __init__(self, token_counts: list[int]) -> None:
        self.device = torch.device("cpu")
        self.pool: Any = GLOBAL_POOL
        self.ubatch_runner = None
        self.use_breakable_cg = False
        self._capture_descs = {
            CUDAGraphMode.FULL: _full_descs(*sorted(token_counts, reverse=True))
        }
        self._sample_full_descs = False
        self._capture_mem_samples: list[Any] | None = None
        self.graphs: dict[Any, Any] = {}
        self.graph_capture_resources: dict[Any, list[Any]] = {}
        self._graphs_captured = False

    def needs_capture(self) -> bool:
        return True

    def _capture_stream(self, desc):
        return None


def _patch_capture(monkeypatch, device: _FakeDevice) -> None:
    accelerator = cgu.torch.accelerator
    monkeypatch.setattr(accelerator, "synchronize", lambda: None)
    for name in ("get_memory_info", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(accelerator, name, getattr(device, name))
    monkeypatch.setattr(accelerator, "empty_cache", device.empty_cache)
    monkeypatch.setattr(cgu.torch.cuda, "graph", device.cuda_graph)
    monkeypatch.setattr(cgu.torch.cuda, "CUDAGraph", MagicMock)
    monkeypatch.setattr(cgu, "graph_capture", lambda **_: contextlib.nullcontext())
    monkeypatch.setattr(cgu, "get_offloader", MagicMock)
    monkeypatch.setattr(cgu, "set_graph_pool_id", lambda _pool: None)
    monkeypatch.setattr(cgu, "is_global_first_rank", lambda: False)


# Decode graph sizes shaped like DeepSeek-V4.1's DSpark verification graphs.
_DECODE_TOKENS = [*range(128, 0, -8), *range(7, 0, -1)]


@pytest.mark.parametrize("retain_metadata", [False, True])
def test_full_graph_sample_excludes_released_warmup_cache(monkeypatch, retain_metadata):
    """The baseline is read after the warmup's cached blocks are flushed.

    ``torch.cuda.graph`` empties the allocator cache when it is entered. A
    baseline read before that flush subtracts the eager warmup's cached
    activations and metadata from the graph's own memory.
    """
    device = _FakeDevice()
    _patch_capture(monkeypatch, device)
    manager = _CapturingGraphManager([128, 120, 1])
    samples: list[Any] = []
    manager._capture_mem_samples = samples

    manager.capture(device.create_forward_fn(retain_metadata))

    # Nothing is left for the capture's own flush to release in the sample.
    assert device.flushed_by_capture == [0, 0, 0]
    # With retained metadata, the 120-token graph needs 100 MiB more pool and
    # the 1-token graph fits in blocks freed by earlier graphs; each graph
    # still keeps its own metadata.
    retained = _METADATA_BYTES if retain_metadata else 0
    pool_growth = [256 * _MIB + _METADATA_BYTES, 100 * _MIB if retained else 0, 0]
    assert [sample.growth for sample in samples] == [
        pool + _GRAPH_EXEC_BYTES for pool in pool_growth
    ]
    assert [sample.cost for sample in samples] == [retained + _GRAPH_EXEC_BYTES] * 3


@pytest.mark.parametrize("retain_metadata", [False, True])
def test_profile_estimate_covers_every_full_graph(monkeypatch, retain_metadata):
    """Three sampled graphs must account for memory each graph retains.

    Compare the profiled estimate with a capture of every graph. A per-graph
    retained buffer must shrink the KV cache instead of running out of memory
    after the real capture.
    """
    _patch_module(monkeypatch)
    full_device = _FakeDevice()
    _patch_capture(monkeypatch, full_device)
    all_graphs_bytes = full_device.capture_model(
        _CapturingGraphManager(_DECODE_TOKENS),
        full_device.create_forward_fn(retain_metadata),
    )

    device = _FakeDevice()
    _patch_capture(monkeypatch, device)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)
    runner.cudagraph_manager = _CapturingGraphManager(_DECODE_TOKENS)
    runner.capture_model = lambda *, profile_only: device.capture_model(
        runner.cudagraph_manager, device.create_forward_fn(retain_metadata)
    )

    estimate = cgu.profile_cudagraph_memory(runner)

    assert all_graphs_bytes <= estimate <= all_graphs_bytes * 21 // 20
    assert device.flushed_by_capture == [0, 0, 0]
    if not retain_metadata:
        assert estimate == all_graphs_bytes


def test_profile_cudagraph_memory_clears_captured_graphs(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    lifecycle: list[str] = []
    monkeypatch.setattr(
        cgu.torch.accelerator,
        "synchronize",
        lambda: lifecycle.append("synchronize"),
    )
    monkeypatch.setattr(
        cgu.CUDAGraphWrapper,
        "reset_all_graphs",
        classmethod(lambda cls: lifecycle.append("reset-piecewise")),
    )
    monkeypatch.setattr(
        cgu.BreakableCUDAGraphWrapper,
        "reset_all_graphs",
        classmethod(lambda cls: lifecycle.append("reset-breakable")),
    )
    monkeypatch.setattr(
        cgu.CUDAGraphWrapper,
        "clear_all_graphs",
        classmethod(lambda cls: lifecycle.append("clear-piecewise")),
    )
    monkeypatch.setattr(
        cgu.BreakableCUDAGraphWrapper,
        "clear_all_graphs",
        classmethod(lambda cls: lifecycle.append("clear-breakable")),
    )
    runner.cudagraph_manager.graphs = _RecordingDict(lifecycle, "full")
    runner.cudagraph_manager.graphs["profile"] = _RecordingGraph(lifecycle, "full")
    runner.cudagraph_manager.graph_capture_resources = _RecordingDict(
        lifecycle, "resources"
    )
    runner.cudagraph_manager.graph_capture_resources["profile"] = [object()]

    cgu.profile_cudagraph_memory(runner)

    # CUDA graph executables must be destroyed and synchronized before their
    # B12X channel checkpoints and tensor workspaces are released.
    assert lifecycle == [
        "synchronize",  # Native-memory measurement before capture.
        "synchronize",  # Native-memory measurement after capture.
        "synchronize",
        "reset-piecewise",
        "reset-breakable",
        "reset-full",
        "synchronize",
        "clear-piecewise",
        "clear-breakable",
        "clear-full",
        "clear-resources",
    ]


def test_cuda_graph_wrappers_reset_executables_without_releasing_resources():
    lifecycle: list[str] = []
    piecewise = object.__new__(cgu.CUDAGraphWrapper)
    piecewise_entry = SimpleNamespace(
        cudagraph=_RecordingGraph(lifecycle, "piecewise"),
        output=object(),
    )
    piecewise.concrete_cudagraph_entries = {"profile": piecewise_entry}

    class _RecordingCapture:
        def reset(self) -> None:
            lifecycle.append("reset-breakable")

    breakable = object.__new__(cgu.BreakableCUDAGraphWrapper)
    breakable_entry = SimpleNamespace(
        capture=_RecordingCapture(),
        resources=[object()],
    )
    breakable.entries = {"profile": breakable_entry}

    piecewise.reset_graphs()
    breakable.reset_graphs()

    assert lifecycle == ["reset-piecewise", "reset-breakable"]
    assert piecewise_entry.cudagraph is None
    assert piecewise_entry.output is not None
    assert piecewise.concrete_cudagraph_entries == {"profile": piecewise_entry}
    assert breakable_entry.capture is None
    assert breakable_entry.resources
    assert breakable.entries == {"profile": breakable_entry}


def test_profile_cudagraph_memory_redirects_wrapper_pools(monkeypatch):
    """Piecewise wrappers must capture into the throwaway pool too.

    Profiling graphs captured into the persistent global pool and then
    discarded drop the pool's use_count to 0, tripping the c10 allocator's
    create_or_incref_pool assert when the real capture reuses that pool.
    """
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    class _FakeWrapper:
        def __init__(self) -> None:
            self.graph_pool: Any = GLOBAL_POOL
            self.pool_during_capture: Any = None

        def clear_graphs(self) -> None:
            pass

        def reset_graphs(self) -> None:
            pass

    wrapper = _FakeWrapper()
    cgu.CUDAGraphWrapper._all_instances.add(wrapper)
    try:
        capture_model = runner.capture_model

        def _capture_model(*, profile_only: bool = False) -> int:
            wrapper.pool_during_capture = wrapper.graph_pool
            return capture_model(profile_only=profile_only)

        runner.capture_model = _capture_model

        cgu.profile_cudagraph_memory(runner)

        assert wrapper.pool_during_capture == THROWAWAY_POOL
        assert wrapper.graph_pool == GLOBAL_POOL
    finally:
        cgu.CUDAGraphWrapper._all_instances.discard(wrapper)


def test_profile_cudagraph_memory_swaps_and_drops_speculator_managers(monkeypatch):
    """Speculator cudagraph managers must also capture into the throwaway pool.

    They are created during the profiling KV-cache bootstrap, binding the
    (swapped) global graph pool, and are re-created by the real
    initialize_kv_cache; profiling-captured graphs must be dropped at
    teardown rather than released against the persistent global pool, which
    would trip the c10 create_or_incref_pool assert at the real capture.
    """
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    def _init(r):
        r.events.append("init")
        # Mirror production: the speculator's cudagraph managers are created
        # during the profiling KV-cache bootstrap and bind the global pool
        # (which profiling has already pointed at the throwaway pool).
        manager = _FakeCudaGraphManager(True, 0)
        manager.pool = cgu.current_platform.get_global_graph_pool()
        r.speculator = SimpleNamespace(
            cudagraph_manager=manager, reset_attn=lambda: None
        )

    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", _init)

    pools_seen: list[Any] = []
    capture_model = runner.capture_model

    def _capture_model(*, profile_only: bool = False) -> int:
        pools_seen.append(runner.speculator.cudagraph_manager.pool)
        return capture_model(profile_only=profile_only)

    runner.capture_model = _capture_model

    cgu.profile_cudagraph_memory(runner)

    assert pools_seen == [THROWAWAY_POOL]
    assert runner.speculator.cudagraph_manager is None
    # The real global pool is restored afterwards.
    assert _FakePlatform._global_graph_pool == GLOBAL_POOL


@create_new_process_for_each_test("spawn")
@pytest.mark.skipif(not cgu.current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("native_bytes", [0, 256 << 20])
def test_profile_cudagraph_memory_frees_throwaway_pool(monkeypatch, native_bytes):
    """Release profiling graphs and avoid reserving persistent cudaMalloc twice."""
    from cuda.bindings import runtime

    @contextlib.contextmanager
    def _fake_set_current_vllm_config(_cfg):
        yield

    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    runner.compilation_config.static_forward_context = {}
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.kv_caches = []
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace()
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: None
    allocation_bytes = 64 << 20
    memory: dict[str, int] = {}
    native_pointer = None

    torch.accelerator.synchronize()
    gc.collect()
    torch.accelerator.empty_cache()
    memory["before"] = torch.accelerator.memory_reserved()

    def _init(r):
        r.events.append("init")
        kv_cache = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda")
        r.kv_caches = [kv_cache]
        r.compilation_config.static_forward_context = {
            "layer": SimpleNamespace(kv_cache=kv_cache)
        }
        manager = _FakeCudaGraphManager(True, 0)
        manager.pool = cgu.current_platform.get_global_graph_pool()
        r.speculator = SimpleNamespace(
            cudagraph_manager=manager, reset_attn=lambda: None
        )

    def _capture_model(*, profile_only: bool = False) -> int:
        nonlocal native_pointer
        before_free = torch.accelerator.get_memory_info()[0]
        if native_bytes:
            error, native_pointer = runtime.cudaMalloc(native_bytes)
            assert error == runtime.cudaError_t.cudaSuccess
        for owner in (
            runner.cudagraph_manager,
            runner.speculator.cudagraph_manager,
        ):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=owner.pool):
                output = torch.empty(allocation_bytes, dtype=torch.uint8, device="cuda")
                output.fill_(1)
            owner.graphs["profile"] = graph
            owner.graph_capture_resources["profile"] = [output]
        torch.accelerator.synchronize()
        memory["captured"] = torch.accelerator.memory_reserved()
        memory["measured"] = before_free - torch.accelerator.get_memory_info()[0]
        return memory["measured"]

    monkeypatch.setattr(cgu, "set_current_vllm_config", _fake_set_current_vllm_config)
    monkeypatch.setattr(cgu, "_init_minimal_kv_cache_for_profiling", _init)
    runner.capture_model = _capture_model

    try:
        graph_estimate = cgu.profile_cudagraph_memory(runner)
        memory["after"] = torch.accelerator.memory_reserved()

        assert memory["captured"] - memory["before"] >= 3 * allocation_bytes
        assert memory["after"] == memory["before"]
        native_before, native_after, measured = runner.cudagraph_native_memory_profile
        assert measured == graph_estimate == memory["measured"]
        assert native_after - native_before >= native_bytes
    finally:
        if native_pointer is not None:
            assert (
                runtime.cudaFree(native_pointer)[0] == runtime.cudaError_t.cudaSuccess
            )


def test_teardown_profiling_state_clears_mamba_align_metadata(monkeypatch):
    """Profiling-cached Mamba align metadata must be invalidated at teardown.

    ``MambaHybridModelState`` lazily caches ``_mamba_group_ids`` and
    ``_mamba_spec`` from whichever KVCacheConfig it first sees. When the
    profiling config's group layout differs from the real (e.g. PP-projected)
    config, reusing the stale metadata mismatches the real block tables
    ("expected 3 block tables, got 4" at
    ``MambaSpecDecodeGPUContext.initialize_from_forward_context``).
    """
    runner: Any = mrv2.GPUModelRunner.__new__(mrv2.GPUModelRunner)
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.model_state = SimpleNamespace(
        supports_mm_inputs=False,
        _mamba_ctx=object(),
        _mamba_group_ids=[0, 1],
        _mamba_spec=object(),
    )
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.kv_caches = []
    runner.attn_groups = []
    runner.kv_cache_config = SimpleNamespace()
    runner.cudagraph_manager = object()
    runner.lora_config = None
    runner.maybe_remove_all_loras = lambda _: None

    monkeypatch.setattr(cgu.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(cgu.torch.accelerator, "empty_cache", lambda: None)

    cgu._teardown_profiling_state(runner)

    assert runner.model_state._mamba_ctx is None
    assert runner.model_state._mamba_group_ids == []
    assert runner.model_state._mamba_spec is None


def test_profile_cudagraph_memory_redirects_late_created_wrappers(monkeypatch):
    """Wrappers created by AOT warmup must not touch the persistent pool."""
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    class _FakeWrapper:
        def __init__(self) -> None:
            self.graph_pool = cgu.current_platform.get_global_graph_pool()
            self.pool_during_capture: Any = None

        def clear_graphs(self) -> None:
            pass

        def reset_graphs(self) -> None:
            pass

    wrapper: _FakeWrapper | None = None
    capture_model = runner.capture_model

    def _capture_model(*, profile_only: bool = False) -> int:
        nonlocal wrapper
        wrapper = _FakeWrapper()
        cgu.CUDAGraphWrapper._all_instances.add(wrapper)
        wrapper.pool_during_capture = wrapper.graph_pool
        return capture_model(profile_only=profile_only)

    runner.capture_model = _capture_model
    try:
        cgu.profile_cudagraph_memory(runner)

        assert wrapper is not None
        assert wrapper.pool_during_capture == THROWAWAY_POOL
        assert wrapper.graph_pool == GLOBAL_POOL
        assert cgu.current_platform.get_global_graph_pool() == GLOBAL_POOL
    finally:
        if wrapper is not None:
            cgu.CUDAGraphWrapper._all_instances.discard(wrapper)


def test_profile_cudagraph_memory_redirects_speculator_managers(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)
    prefill_manager = _FakeCudaGraphManager(True, 2)
    decode_manager = _FakeCudaGraphManager(True, 2)
    runner.speculator = SimpleNamespace(
        prefill_cudagraph_manager=prefill_manager,
        decode_cudagraph_manager=decode_manager,
    )

    capture_model = runner.capture_model
    pools_during_capture: tuple[Any, Any] | None = None

    def _capture_model(*, profile_only: bool = False) -> int:
        nonlocal pools_during_capture
        pools_during_capture = (prefill_manager.pool, decode_manager.pool)
        lifecycle: list[str] = []
        prefill_manager.graphs["profile"] = _RecordingGraph(lifecycle, "prefill")
        decode_manager.graphs["profile"] = _RecordingGraph(lifecycle, "decode")
        prefill_manager._graphs_captured = True
        decode_manager._graphs_captured = True
        return capture_model(profile_only=profile_only)

    runner.capture_model = _capture_model
    cgu.profile_cudagraph_memory(runner)

    assert pools_during_capture == (THROWAWAY_POOL, THROWAWAY_POOL)
    assert prefill_manager.pool == GLOBAL_POOL
    assert decode_manager.pool == GLOBAL_POOL
    assert not prefill_manager.graphs
    assert not decode_manager.graphs
    assert not prefill_manager._graphs_captured
    assert not decode_manager._graphs_captured


def test_v2_profiling_teardown_runs_cache_lifecycle_hooks(monkeypatch):
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    events: list[str] = []

    class _Layer:
        def __init__(self) -> None:
            self.kv_cache = object()

        def unbind_kv_cache(self) -> None:
            events.append("layer")
            self.kv_cache = None

    class _ModelState:
        supports_mm_inputs = False

        def reset_kv_cache_state(self) -> None:
            events.append("model-state")

    class _Speculator:
        def reset_attn(self) -> None:
            events.append("speculator")
            DraftModelSpeculator.reset_attn(self)

    class _UBatchRunner:
        def abort_pending_run(self):
            events.append("ubatch")

    layer = _Layer()
    runner = SimpleNamespace(
        kv_caches=[object()],
        attn_groups=[[object()]],
        kv_cache_config=object(),
        cudagraph_manager=object(),
        block_tables=object(),
        pcp_manager=object(),
        adaptive_verification=object(),
        ubatch_runner=_UBatchRunner(),
        model_state=_ModelState(),
        speculator=_Speculator(),
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        lora_config=None,
        maybe_remove_all_loras=lambda _config: events.append("loras"),
    )
    runner.speculator.pcp_manager = torch.empty(1)
    pcp_ref = ref(runner.speculator.pcp_manager)
    ubatch_ref = ref(runner.ubatch_runner)

    def empty_cache():
        assert pcp_ref() is None
        assert ubatch_ref() is None

    monkeypatch.setattr(cgu.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(cgu.torch.accelerator, "empty_cache", empty_cache)

    cgu._teardown_profiling_state(runner)

    assert events == ["ubatch", "layer", "model-state", "speculator", "loras"]
    assert layer.kv_cache is None
    assert runner.kv_caches == []
    assert runner.attn_groups == []
    assert runner.cudagraph_manager is None
    assert not hasattr(runner, "block_tables")
    assert runner.pcp_manager is None
    assert runner.adaptive_verification is None
    assert runner.ubatch_runner is None
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks is None


def test_legacy_profiling_teardown_unbinds_layers_and_mamba_buffers(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as legacy_runner_module

    events: list[str] = []

    class _Layer:
        kv_cache = object()

        def unbind_kv_cache(self) -> None:
            events.append("layer")
            self.kv_cache = None

    layer = _Layer()
    runner = legacy_runner_module.GPUModelRunner.__new__(
        legacy_runner_module.GPUModelRunner
    )
    runner.kv_caches = [object()]
    runner.attn_groups = [[object()]]
    runner.kv_cache_config = object()
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.drafter = SimpleNamespace(draft_attn_groups=[object()])
    runner.compilation_config = SimpleNamespace(static_forward_context={"layer": layer})
    runner._mamba_bufs = object()
    monkeypatch.setattr(
        legacy_runner_module.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        legacy_runner_module.torch.accelerator, "empty_cache", lambda: None
    )

    runner._cleanup_profiling_kv_cache()

    assert events == ["layer"]
    assert layer.kv_cache is None
    assert runner.kv_caches == []
    assert runner.attn_groups == []
    assert runner.drafter.draft_attn_groups == []
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks is None
    assert runner._mamba_bufs is None


def test_legacy_minimal_init_restores_block_override_on_error(monkeypatch):
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.worker import gpu_model_runner as legacy_runner_module

    runner = legacy_runner_module.GPUModelRunner.__new__(
        legacy_runner_module.GPUModelRunner
    )
    runner.get_kv_cache_spec = lambda: {}
    runner.vllm_config = object()
    runner.max_num_reqs = 4
    runner.compilation_config = SimpleNamespace(max_cudagraph_capture_size=2)
    runner.cache_config = SimpleNamespace(num_gpu_blocks_override=17)
    monkeypatch.setattr(
        legacy_runner_module.KVCacheSpecRegistry,
        "check_kv_cache_spec_registry",
        lambda _spec: None,
    )
    monkeypatch.setattr(kv_cache_utils, "get_kv_cache_groups", lambda *_args: [])

    def _raise_config_error(*_args, **_kwargs):
        raise RuntimeError("config failed")

    monkeypatch.setattr(
        kv_cache_utils,
        "get_kv_cache_config_from_groups",
        _raise_config_error,
    )

    with pytest.raises(RuntimeError, match="config failed"):
        runner._init_minimal_kv_cache_for_profiling()

    assert runner.cache_config.num_gpu_blocks_override == 17


def test_legacy_profile_tears_down_after_partial_init_error(monkeypatch):
    from vllm.v1.worker import gpu_model_runner as legacy_runner_module

    runner = legacy_runner_module.GPUModelRunner.__new__(
        legacy_runner_module.GPUModelRunner
    )
    runner.vllm_config = object()
    runner.cache_config = SimpleNamespace(num_gpu_blocks=1)
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_caches = [object()]
    runner.attn_groups = [[object()]]
    runner.kv_cache_config = object()
    runner._mamba_bufs = object()
    runner.drafter = SimpleNamespace(draft_attn_groups=[object()])

    def _partial_init() -> None:
        raise RuntimeError("init failed")

    runner._init_minimal_kv_cache_for_profiling = _partial_init
    monkeypatch.setattr(
        legacy_runner_module,
        "set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        legacy_runner_module.torch.accelerator, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        legacy_runner_module.torch.accelerator, "empty_cache", lambda: None
    )

    with pytest.raises(RuntimeError, match="init failed"):
        runner.profile_cudagraph_memory()

    assert runner.kv_caches == []
    assert runner.attn_groups == []
    assert runner.drafter.draft_attn_groups == []
    assert not hasattr(runner, "kv_cache_config")
    assert runner.cache_config.num_gpu_blocks is None
    assert runner._mamba_bufs is None


def test_legacy_profile_samples_exclude_cached_blocks(monkeypatch):
    """Model runner V1 samples each graph with the allocator cache empty."""
    from vllm.v1.worker import gpu_model_runner as legacy_runner_module

    device = _FakeDevice()
    _patch_capture(monkeypatch, device)
    runner = legacy_runner_module.GPUModelRunner.__new__(
        legacy_runner_module.GPUModelRunner
    )
    runner.vllm_config = object()
    runner.lora_config = None
    runner.device = torch.device("cpu")
    runner.max_model_len = 1024
    runner.max_num_tokens = 128
    runner.cudagraph_dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [(CUDAGraphMode.FULL, _full_descs(128, 120, 1))],
        cudagraph_keys={},
        keys_initialized=True,
    )
    runner._init_minimal_kv_cache_for_profiling = lambda: None
    runner._cleanup_profiling_kv_cache = lambda: None
    runner._create_encoder_cudagraph_manager = lambda: None
    runner._freeze_gc = contextlib.nullcontext
    runner.maybe_remove_all_loras = lambda _config: None
    create_forward_fn = device.create_forward_fn(retain_metadata=True)

    def warmup_and_capture(desc, cudagraph_runtime_mode, profile_seq_lens=None):
        forward_fn = create_forward_fn(desc, warmup=True)
        forward_fn(CUDAGraphMode.NONE)
        with device.cuda_graph():
            forward_fn(cudagraph_runtime_mode)
        # The dummy run's eager outputs stay cached after the capture.
        device.allocate(64 * _MIB)
        device.free(64 * _MIB)

    runner._warmup_and_capture = warmup_and_capture
    monkeypatch.setattr(
        legacy_runner_module,
        "set_current_vllm_config",
        lambda _config: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        legacy_runner_module, "graph_capture", lambda **_: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        legacy_runner_module, "set_cudagraph_capturing_enabled", lambda _: None
    )
    monkeypatch.setattr(
        legacy_runner_module,
        "current_platform",
        SimpleNamespace(graph_pool_handle=object, is_rocm=lambda: False),
    )

    estimate = runner.profile_cudagraph_memory()

    # The largest graph costs its pool and instantiated graph; the next one
    # the pool growth for its retained metadata, charged to both others.
    first_graph = 256 * _MIB + _METADATA_BYTES + _GRAPH_EXEC_BYTES
    per_graph = 100 * _MIB + _GRAPH_EXEC_BYTES
    assert estimate == first_graph + 2 * per_graph
