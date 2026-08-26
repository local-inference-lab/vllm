# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as gpu_model_runner_v2_module
import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)


def test_gpu_worker_keeps_cuda_warmup_import_lazy(monkeypatch: pytest.MonkeyPatch):
    # Forkserver preloads this module. Binding kernel_warmup at module scope
    # imports CUDA-initializing warmup dependencies into the server snapshot.
    assert gpu_worker_module.kernel_warmup.__module__ == gpu_worker_module.__name__

    run_kernel_warmup = Mock()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.warmup.kernel_warmup",
        SimpleNamespace(kernel_warmup=run_kernel_warmup),
    )
    worker = object()
    gpu_worker_module.kernel_warmup(worker)

    run_kernel_warmup.assert_called_once_with(worker)


@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 0, 1024])
def test_release_allocator_slack_before_manual_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_memory_bytes: int | None,
):
    runner = object.__new__(GPUModelRunnerV2)
    runner.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=kv_cache_memory_bytes,
    )
    runner.device = torch.device("cuda:3")
    calls: list[object] = []
    monkeypatch.setattr(
        gpu_model_runner_v2_module.torch.accelerator,
        "synchronize",
        lambda device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2_module.gc,
        "collect",
        lambda: calls.append("collect"),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2_module.torch.accelerator,
        "empty_cache",
        lambda: calls.append("empty_cache"),
    )

    runner._release_allocator_slack_before_manual_kv_cache()

    if kv_cache_memory_bytes is None:
        assert calls == []
    else:
        assert calls == [
            ("synchronize", torch.device("cuda:3")),
            "collect",
            "empty_cache",
        ]


@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 0, 1024])
def test_manual_kv_storage_precedes_attention_metadata(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_memory_bytes: int | None,
):
    runner = object.__new__(GPUModelRunnerV2)
    runner.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=kv_cache_memory_bytes,
    )
    runner.device = torch.device("cuda:3")
    runner.vllm_config = object()
    kv_cache_config = object()
    storage = object()
    calls: list[object] = []
    monkeypatch.setattr(
        runner,
        "_release_allocator_slack_before_manual_kv_cache",
        lambda: calls.append("release"),
    )
    monkeypatch.setattr(
        gpu_model_runner_v2_module,
        "allocate_kv_cache",
        lambda config, vllm_config, device: (
            calls.append(("allocate", config, vllm_config, device)) or storage
        ),
    )

    result = runner._allocate_manual_kv_cache_before_metadata(kv_cache_config)

    if kv_cache_memory_bytes is None:
        assert result is None
        assert calls == []
    else:
        assert result is storage
        assert calls == [
            "release",
            (
                "allocate",
                kv_cache_config,
                runner.vllm_config,
                torch.device("cuda:3"),
            ),
        ]


def test_kernel_warmup_runs_once(monkeypatch: pytest.MonkeyPatch):
    worker = object.__new__(Worker)
    calls = []
    monkeypatch.setattr(
        gpu_worker_module,
        "kernel_warmup",
        lambda warmed_worker: calls.append(warmed_worker),
    )

    worker._warmup_kernels_once()
    worker._warmup_kernels_once()

    assert calls == [worker]
    assert worker._kernel_warmup_complete is True


def test_runtime_kernel_warmup_waits_for_kv_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace()
    calls: list[tuple[str, object]] = []

    def persistent_warmup(warmed_worker: object) -> bool:
        calls.append(("persistent", warmed_worker))
        return False

    monkeypatch.setattr(
        gpu_worker_module,
        "kernel_warmup",
        persistent_warmup,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "runtime_kernel_warmup",
        lambda warmed_worker: calls.append(("runtime", warmed_worker)),
    )

    worker._warmup_kernels_once()
    worker.model_runner.block_tables = object()
    worker._warmup_kernels_once()
    worker._warmup_kernels_once()

    assert calls == [("persistent", worker), ("runtime", worker)]
    assert worker._kernel_warmup_complete is True
    assert worker._runtime_kernel_warmup_complete is True


def test_failed_kernel_warmup_is_retryable(monkeypatch: pytest.MonkeyPatch):
    worker = object.__new__(Worker)
    calls = 0

    def fail_once(_worker):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(gpu_worker_module, "kernel_warmup", fail_once)

    with pytest.raises(RuntimeError, match="warmup failed"):
        worker._warmup_kernels_once()
    worker._warmup_kernels_once()

    assert calls == 2
    assert worker._kernel_warmup_complete is True


def test_memory_profile_replays_model_after_kernel_warmup(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = object.__new__(Worker)
    calls: list[object] = []
    worker.device = "cuda:0"
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: calls.append("profile"),
        profile_single_request_prefill=lambda: calls.append("single-prefill"),
    )
    worker.vllm_config = SimpleNamespace()
    worker.get_model = lambda: object()
    worker.get_kv_cache_spec = lambda: object()
    monkeypatch.setattr(
        gpu_worker_module,
        "deepseek_v4_compressor_triton_warmup",
        lambda *args: calls.append("compressor"),
    )
    monkeypatch.setattr(
        worker,
        "_warmup_kernels_once",
        lambda: calls.append("kernel_warmup"),
    )
    monkeypatch.setattr(
        gpu_worker_module.torch.accelerator,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset_peak", device)),
    )

    worker._profile_model_with_kernel_warmup()

    assert calls == [
        "profile",
        "compressor",
        "kernel_warmup",
        ("reset_peak", "cuda:0"),
        "profile",
        "single-prefill",
    ]


@pytest.mark.parametrize("kv_cache_memory_bytes", [0, 1024])
def test_manual_kv_profile_releases_unoccupied_memory(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_memory_bytes: int,
):
    worker = object.__new__(Worker)
    calls: list[object] = []
    worker.device = "cuda:0"
    worker.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=kv_cache_memory_bytes,
    )
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: calls.append("profile"),
    )
    worker.vllm_config = SimpleNamespace()
    worker.init_snapshot = SimpleNamespace(free_memory=2048)
    worker.model_config = SimpleNamespace(multimodal_config=None)
    worker.parallel_config = SimpleNamespace(_api_process_count=1)
    worker.get_model = lambda: object()
    worker.get_kv_cache_spec = lambda: object()
    monkeypatch.setattr(gpu_worker_module, "maybe_apply_startup_plan", lambda _: None)
    monkeypatch.setattr(
        gpu_worker_module,
        "deepseek_v4_compressor_triton_warmup",
        lambda *args: calls.append("compressor"),
    )
    monkeypatch.setattr(
        worker,
        "_warmup_kernels_once",
        lambda: calls.append("kernel_warmup"),
    )
    monkeypatch.setattr(
        gpu_worker_module.torch.accelerator,
        "synchronize",
        lambda device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        gpu_worker_module.gc,
        "collect",
        lambda: calls.append("collect"),
    )
    monkeypatch.setattr(
        gpu_worker_module.torch.accelerator,
        "empty_cache",
        lambda: calls.append("empty_cache"),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "reserve_mm_ipc_gpu_memory",
        lambda size, *args: size,
    )

    assert worker.determine_available_memory() == kv_cache_memory_bytes
    assert calls == [
        ("synchronize", "cuda:0"),
        "collect",
        "empty_cache",
        "profile",
        ("synchronize", "cuda:0"),
        "collect",
        "empty_cache",
        "compressor",
        "kernel_warmup",
        ("synchronize", "cuda:0"),
        "collect",
        "empty_cache",
    ]


@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 0, 1024])
def test_manual_kv_graph_capture_releases_unoccupied_memory(
    kv_cache_memory_bytes: int | None,
):
    worker = object.__new__(Worker)
    calls: list[str] = []
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes)

    def capture_model() -> int:
        calls.append("capture")
        return 17

    worker.model_runner = SimpleNamespace(capture_model=capture_model)
    worker._release_unoccupied_accelerator_memory = lambda: calls.append("release")

    assert worker._capture_model_with_reclaimed_manual_kv_cache() == 17
    expected = ["capture"] if kv_cache_memory_bytes is None else ["release", "capture"]
    assert calls == expected


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
