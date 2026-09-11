# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import gpu_worker, startup_plan
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


@pytest.mark.parametrize(
    "final_free_memory,cudagraph_estimate,expected_available_memory",
    [
        # No late persistent memory, no graph estimate.
        (80, 0, 78),
        # Five bytes retained after the graph profile, charged in full.
        (75, 0, 73),
        # The graph estimate already covers three of the five retained bytes.
        (75, 3, 73),
        # The graph estimate covers every retained byte; only it is charged.
        (75, 6, 72),
    ],
)
@pytest.mark.parametrize("estimate_graphs", [False, True])
@pytest.mark.parametrize("resolves_kernels", [True, False])
def test_kv_memory_profile_uses_repeatable_peak_before_cudagraphs(
    monkeypatch,
    final_free_memory,
    cudagraph_estimate,
    expected_available_memory,
    estimate_graphs,
    resolves_kernels,
):
    """KV sizing must retain the warmed allocator and graph high-waters and
    the persistent allocations made after the activation profile, charging
    the part of them inside the CUDA-graph estimate once. The expected
    budgets of the table apply when the estimate is applied; without it the
    retained memory is charged in full and nothing is subtracted for graphs.
    The repeated profile runs only when the B12X warm-up resolved kernels;
    otherwise the single profile's headroom stands and no correction is
    subtracted."""
    events: list[object] = []
    first_profile = SimpleNamespace(
        free_memory=84,
        torch_allocated=7,
        torch_memory=8,
        non_torch_memory=2,
    )
    after_warmup = SimpleNamespace(
        free_memory=83,
        torch_allocated=8,
        torch_memory=9,
        non_torch_memory=2,
    )
    repeatable_profile = SimpleNamespace(
        free_memory=82,
        torch_allocated=9,
        torch_memory=16,
        non_torch_memory=2,
    )
    final = SimpleNamespace(
        free_memory=final_free_memory,
        torch_allocated=8,
        torch_memory=9,
        non_torch_memory=3,
    )
    snapshots = iter(
        [first_profile, after_warmup, repeatable_profile, final]
        if resolves_kernels
        else [first_profile, final]
    )

    def profile_cudagraph_memory():
        events.append("profile_cudagraph_memory")
        return cudagraph_estimate

    def reserve_sampler_workspace():
        events.append("reserve_sampler_workspace")
        return 0

    model_runner = SimpleNamespace(
        model_memory_usage=0,
        reserve_sampler_workspace=reserve_sampler_workspace,
        profile_run=lambda: events.append("profile_run"),
        profile_glm_dcp_attention=lambda: events.append("profile_glm_dcp_attention"),
        profile_cudagraph_memory=profile_cudagraph_memory,
    )
    profile_result = SimpleNamespace(
        weights_memory=0,
        total_consumed=10,
        transient_peak_headroom=5,
        before_profile=SimpleNamespace(free_memory=85),
        after_profile=SimpleNamespace(
            free_memory=80,
            torch_allocated=8,
            torch_memory=9,
            non_torch_memory=3,
        ),
        non_kv_cache_memory=10,
    )

    @contextmanager
    def fake_memory_profiling(*args, **kwargs):
        yield profile_result

    worker = SimpleNamespace(
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None,
            gpu_memory_utilization=0.9,
        ),
        model_runner=model_runner,
        init_snapshot=SimpleNamespace(
            free_memory=100,
            total_memory=100,
            torch_allocated=1,
            torch_memory=1,
            non_torch_memory=1,
        ),
        requested_memory=90,
        device="cuda:0",
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                cudagraph_mode=gpu_worker.CUDAGraphMode.PIECEWISE,
                cudagraph_capture_sizes=[8, 4],
            )
        ),
    )

    monkeypatch.setattr(gpu_worker, "maybe_apply_startup_plan", lambda worker: None)
    monkeypatch.setenv(
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", str(int(estimate_graphs))
    )
    monkeypatch.setattr(gpu_worker, "memory_profiling", fake_memory_profiling)
    monkeypatch.setattr(
        gpu_worker,
        "MemorySnapshot",
        lambda *, device: next(snapshots),
    )
    monkeypatch.setattr(
        gpu_worker,
        "current_platform",
        SimpleNamespace(is_cuda_alike=lambda: True),
    )

    def fake_b12x_warmup(worker, sizes):
        events.append(("b12x_warmup", tuple(sizes)))
        return resolves_kernels

    monkeypatch.setattr(gpu_worker, "b12x_warmup", fake_b12x_warmup)
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "reset_peak_memory_stats",
        lambda device: events.append(("reset_peak", device)),
    )
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        gpu_worker,
        "reserve_mm_ipc_gpu_memory",
        lambda requested, *args: requested,
    )

    available = gpu_worker.Worker.determine_available_memory(worker)

    # The cached allocator blocks are released before the final snapshot, as
    # the activation profile released them before its own after-profile
    # snapshot, so only allocations retained after the CUDA-graph profile
    # count as late persistent memory.
    repeated_profile: list[object] = (
        [("reset_peak", "cuda:0"), "profile_run", "profile_glm_dcp_attention"]
        if resolves_kernels
        else []
    )
    assert events == [
        "reserve_sampler_workspace",
        "profile_run",
        "profile_glm_dcp_attention",
        ("b12x_warmup", (8, 4)),
        *repeated_profile,
        "profile_cudagraph_memory",
        "empty_cache",
    ]
    # The repeatable profile retained seven bytes above its cleanup state.
    # Five are already covered by the live-allocation peak, leaving two bytes
    # of allocator-reservation headroom to deduct from KV capacity, plus the
    # free memory retained after the activation profile. Without the repeat
    # those two bytes are not deducted.
    headroom_correction = 2 if resolves_kernels else 0
    late_persistent_memory = 80 - final_free_memory
    if estimate_graphs:
        assert available == expected_available_memory + 2 - headroom_correction
    else:
        assert available == 80 - headroom_correction - late_persistent_memory
    # The activation peak stays activation-only (the repeatable allocator
    # headroom, seven bytes here, or the single profile's five); post-capture
    # recommendations add measured graph memory to it, and the admission
    # budget subtracts the estimate separately.
    assert worker.peak_activation_memory == 5 + headroom_correction
    assert worker.total_consumed == 10 + late_persistent_memory
    assert worker.cudagraph_memory_estimate == cudagraph_estimate


@pytest.mark.parametrize("estimated_gib", [0, 4])
@pytest.mark.parametrize("measured_gib", [3, 7])
def test_post_capture_recommendation_counts_measured_graph_memory_once(
    monkeypatch, estimated_gib, measured_gib
):
    """The saved KV budget uses measured graph storage, not its estimate."""
    compilation = SimpleNamespace(
        mode=gpu_worker.CompilationMode.NONE,
        compilation_time=0.0,
        encoder_compilation_time=0.0,
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(compilation_config=compilation),
        compilation_config=compilation,
        model_runner=SimpleNamespace(
            lora_config=None,
            maybe_remove_all_loras=lambda config: None,
            capture_model=lambda: measured_gib * GiB_bytes,
        ),
        model_config=SimpleNamespace(enforce_eager=False, seed=0),
        cache_config=SimpleNamespace(
            kv_cache_memory_bytes=None, gpu_memory_utilization=0.9
        ),
        init_snapshot=SimpleNamespace(
            free_memory=100 * GiB_bytes, total_memory=100 * GiB_bytes
        ),
        requested_memory=90 * GiB_bytes,
        total_consumed=10 * GiB_bytes,
        peak_activation_memory=5 * GiB_bytes,
        cudagraph_memory_estimate=estimated_gib * GiB_bytes,
        available_kv_cache_memory_bytes=(75 - estimated_gib) * GiB_bytes,
        use_v2_model_runner=False,
        observability_config=SimpleNamespace(
            jit_monitor_mode="off", jit_monitor_verbose=False
        ),
    )
    saved = []
    monkeypatch.setattr(
        gpu_worker, "maybe_save_startup_plan", lambda w, budget: saved.append(budget)
    )
    monkeypatch.setattr(
        gpu_worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False)
    )
    for name in (
        "kernel_warmup",
        "set_random_seed",
        "freeze_gc_heap",
        "maybe_attach_gc_debug_callback",
        "enable_gpu_sync_check",
        "set_torch_threads_for_runtime",
    ):
        monkeypatch.setattr(gpu_worker, name, lambda *args: None)
    monkeypatch.setattr("vllm.utils.jit_monitor.activate", lambda **kwargs: None)

    gpu_worker.Worker.compile_or_warm_up_model(worker)

    assert saved == [(90 - 10 - 5 - measured_gib) * GiB_bytes - 150 * (1 << 20)]
