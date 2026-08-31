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
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.compilation.counter import compilation_counter
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import cudagraph_utils as cgu
from vllm.v1.worker.gpu import model_runner as mrv2

GLOBAL_POOL = "global-pool"
THROWAWAY_POOL = "throwaway-pool"


class _FakePlatform:
    _global_graph_pool: Any = GLOBAL_POOL

    def graph_pool_handle(self) -> Any:
        return THROWAWAY_POOL

    def get_global_graph_pool(self) -> Any:
        return self.__class__._global_graph_pool


class _FakeCudaGraphManager(cgu.CudaGraphManager):
    def __init__(
        self, needs_capture: bool, num_full_descs: int, piecewise_only: bool = False
    ) -> None:
        self._needs_capture = needs_capture
        self.pool: Any = GLOBAL_POOL
        descs = [object() for _ in range(num_full_descs)]
        if piecewise_only:
            self._capture_descs = {CUDAGraphMode.PIECEWISE: descs}
        else:
            self._capture_descs = {CUDAGraphMode.FULL: descs} if needs_capture else {}
        # Profiling hooks set by profile_cudagraph_memory.
        self._max_full_descs_to_capture: int | None = None
        self._capture_mem_samples: list[int] | None = None
        self.use_breakable_cg = False
        self.graphs: dict[Any, Any] = {}
        self.graph_capture_resources: dict[Any, Any] = {}
        self._graphs_captured = False

    def needs_capture(self) -> bool:
        return self._needs_capture


def _make_profiling_runner(
    cudagraph_mode: CUDAGraphMode,
    *,
    needs_capture: bool = True,
    num_full_descs: int = 3,
    piecewise_only: bool = False,
    captured_bytes: int = 7 << 30,
    mem_samples: list[int] | None = None,
) -> Any:
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

    def _capture_model() -> int:
        events.append("capture")
        runner.pool_during_capture = runner.cudagraph_manager.pool
        # Simulate the manager's per-FULL-graph memory sampling.
        samples = runner.cudagraph_manager._capture_mem_samples
        if samples is not None:
            samples.extend(mem_samples or [])
        return captured_bytes

    runner.capture_model = _capture_model
    return runner


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


def test_profile_cudagraph_memory_small_full_set_returns_exact_measurement(monkeypatch):
    _patch_module(monkeypatch)
    captured_bytes = 7 << 30
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL,
        num_full_descs=cgu._MAX_EXACT_FULL_GRAPH_PROFILING_GRAPHS,
        captured_bytes=captured_bytes,
    )

    result = cgu.profile_cudagraph_memory(runner)

    assert result == captured_bytes
    assert runner.events == ["init", "capture", "teardown"]
    assert runner.pool_during_capture == THROWAWAY_POOL
    assert runner.cudagraph_manager.pool == GLOBAL_POOL
    assert runner.cudagraph_manager._max_full_descs_to_capture is None


def test_profile_cudagraph_memory_large_full_set_extrapolates(monkeypatch):
    _patch_module(monkeypatch)
    gib = 1 << 30
    # Measured delta 1000 MiB includes the sampled FULL graphs (100 + 20 MiB).
    # Extrapolated FULL cost for 9 graphs: 100 + 8 * 20 = 260 MiB.
    runner = _make_profiling_runner(
        CUDAGraphMode.FULL,
        num_full_descs=cgu._MAX_EXACT_FULL_GRAPH_PROFILING_GRAPHS + 1,
        captured_bytes=1000 * gib,
        mem_samples=[100 * gib, 20 * gib],
    )

    result = cgu.profile_cudagraph_memory(runner)

    assert result == (1000 - (100 + 20) + (100 + 8 * 20)) * gib
    # Bootstrap, capture, and teardown run in order.
    assert runner.events == ["init", "capture", "teardown"]
    # Capture must use a throwaway pool, not the persistent global pool.
    assert runner.pool_during_capture == THROWAWAY_POOL
    assert runner.cudagraph_manager.pool == GLOBAL_POOL
    # FULL capture must be limited to the largest few graphs.
    assert (
        runner.cudagraph_manager._max_full_descs_to_capture
        == cgu._FULL_GRAPH_PROFILING_SAMPLES
    )


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


def test_profile_cudagraph_memory_tears_down_on_capture_error(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL)

    def _boom() -> int:
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

    def _capture_model() -> int:
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
    monkeypatch.setattr(mrv2, "_profile_cudagraph_memory", lambda r: 42)
    assert runner.profile_cudagraph_memory() == 42


def test_extrapolate_full_graph_memory():
    mib = 1 << 20
    # No samples (e.g. no FULL graphs): nothing to add.
    assert cgu._extrapolate_full_graph_memory([], 0) == 0
    # A single graph costs exactly its sample.
    assert cgu._extrapolate_full_graph_memory([100 * mib], 1) == 100 * mib
    # First capture + per-graph cost for the rest.
    assert (
        cgu._extrapolate_full_graph_memory([100 * mib, 20 * mib], 5)
        == (100 + 4 * 20) * mib
    )
    # Per-graph cost is floored to account for driver overhead.
    assert cgu._extrapolate_full_graph_memory([100 * mib, 0], 3) == (100 + 2 * 1) * mib


def test_profile_cudagraph_memory_clears_captured_graphs(monkeypatch):
    _patch_module(monkeypatch)
    runner = _make_profiling_runner(CUDAGraphMode.FULL_AND_PIECEWISE)

    cleared: list[str] = []
    monkeypatch.setattr(
        cgu.CUDAGraphWrapper,
        "clear_all_graphs",
        classmethod(lambda cls: cleared.append("piecewise")),
    )
    monkeypatch.setattr(
        cgu.BreakableCUDAGraphWrapper,
        "clear_all_graphs",
        classmethod(lambda cls: cleared.append("breakable")),
    )

    cgu.profile_cudagraph_memory(runner)

    # Profiling captures are discarded so the real capture re-captures them
    # against the KV cache.
    assert cleared == ["piecewise", "breakable"]


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

    wrapper = _FakeWrapper()
    cgu.CUDAGraphWrapper._all_instances.add(wrapper)
    try:
        capture_model = runner.capture_model

        def _capture_model() -> int:
            wrapper.pool_during_capture = wrapper.graph_pool
            return capture_model()

        runner.capture_model = _capture_model

        cgu.profile_cudagraph_memory(runner)

        assert wrapper.pool_during_capture == THROWAWAY_POOL
        assert wrapper.graph_pool == GLOBAL_POOL
    finally:
        cgu.CUDAGraphWrapper._all_instances.discard(wrapper)


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

    wrapper: _FakeWrapper | None = None
    capture_model = runner.capture_model

    def _capture_model() -> int:
        nonlocal wrapper
        wrapper = _FakeWrapper()
        cgu.CUDAGraphWrapper._all_instances.add(wrapper)
        wrapper.pool_during_capture = wrapper.graph_pool
        return capture_model()

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

    def _capture_model() -> int:
        nonlocal pools_during_capture
        pools_during_capture = (prefill_manager.pool, decode_manager.pool)
        prefill_manager.graphs["profile"] = object()
        decode_manager.graphs["profile"] = object()
        prefill_manager._graphs_captured = True
        decode_manager._graphs_captured = True
        return capture_model()

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

    layer = _Layer()
    runner = SimpleNamespace(
        kv_caches=[object()],
        attn_groups=[[object()]],
        kv_cache_config=object(),
        cudagraph_manager=object(),
        block_tables=object(),
        pcp_manager=object(),
        adaptive_verification=object(),
        model_state=_ModelState(),
        speculator=_Speculator(),
        compilation_config=SimpleNamespace(static_forward_context={"layer": layer}),
        cache_config=SimpleNamespace(num_gpu_blocks=1),
        lora_config=None,
        maybe_remove_all_loras=lambda _config: events.append("loras"),
    )
    monkeypatch.setattr(cgu.torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(cgu.torch.accelerator, "empty_cache", lambda: None)

    cgu._teardown_profiling_state(runner)

    assert events == ["layer", "model-state", "speculator", "loras"]
    assert layer.kv_cache is None
    assert runner.kv_caches == []
    assert runner.attn_groups == []
    assert runner.cudagraph_manager is None
    assert runner.block_tables is None
    assert runner.pcp_manager is None
    assert runner.adaptive_verification is None
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
