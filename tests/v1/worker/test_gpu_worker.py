# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as gpu_model_runner_module
import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)


class _AllocationScope:
    def __init__(self) -> None:
        self.active = False

    def __enter__(self):
        assert not self.active
        self.active = True
        return self

    def __exit__(self, *_args) -> None:
        assert self.active
        self.active = False


@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 1024])
def test_manual_kv_storage_is_allocated_after_reclaim(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_memory_bytes: int | None,
):
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(
        kv_cache_memory_bytes=kv_cache_memory_bytes,
    )
    runner.device = torch.device("cuda:3")
    config = object()
    storage = object()
    scope = _AllocationScope()
    calls: list[object] = []
    monkeypatch.setattr(
        gpu_model_runner_module.torch.accelerator,
        "synchronize",
        lambda device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.gc,
        "collect",
        lambda: calls.append("collect"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.accelerator,
        "empty_cache",
        lambda: calls.append("empty_cache"),
    )

    def allocate(actual_config, device):
        assert scope.active
        calls.append(("allocate", actual_config, device))
        return storage

    monkeypatch.setattr(
        gpu_model_runner_module,
        "allocate_kv_cache_storage",
        allocate,
    )

    result = runner._allocate_manual_kv_cache_storage(config, scope)

    if kv_cache_memory_bytes is None:
        assert result is None
        assert calls == []
    else:
        assert result is storage
        assert calls == [
            ("synchronize", torch.device("cuda:3")),
            "collect",
            "empty_cache",
            ("allocate", config, torch.device("cuda:3")),
        ]
    assert not scope.active


def test_persistent_and_runtime_kernel_warmups_run_once(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace()
    calls: list[tuple[str, object]] = []

    def persistent_warmup(warmed_worker: object) -> bool:
        calls.append(("persistent", warmed_worker))
        return False

    monkeypatch.setattr(gpu_worker_module, "kernel_warmup", persistent_warmup)
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


def test_failed_persistent_kernel_warmup_is_retryable(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace()
    calls = 0

    def fail_once(_worker: object) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("warmup failed")
        return False

    monkeypatch.setattr(gpu_worker_module, "kernel_warmup", fail_once)

    with pytest.raises(RuntimeError, match="warmup failed"):
        worker._warmup_kernels_once()
    worker._warmup_kernels_once()

    assert calls == 2
    assert worker._kernel_warmup_complete is True


def test_memory_profile_replays_model_after_persistent_warmup(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = object.__new__(Worker)
    worker.device = torch.device("cuda:3")
    calls: list[object] = []
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: calls.append("profile"),
    )
    monkeypatch.setattr(
        worker,
        "_warmup_kernels_once",
        lambda: calls.append("kernel_warmup"),
    )
    monkeypatch.setattr(
        torch.accelerator,
        "reset_peak_memory_stats",
        lambda device: calls.append(("reset_peak", device)),
    )

    worker._profile_model_with_kernel_warmup()

    assert calls == [
        "profile",
        "kernel_warmup",
        ("reset_peak", torch.device("cuda:3")),
        "profile",
    ]


def test_release_unoccupied_accelerator_memory(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = object.__new__(Worker)
    worker.device = torch.device("cuda:3")
    calls: list[object] = []
    monkeypatch.setattr(
        torch.accelerator,
        "synchronize",
        lambda device: calls.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_worker.gc.collect",
        lambda: calls.append("collect"),
    )
    monkeypatch.setattr(
        torch.accelerator,
        "empty_cache",
        lambda: calls.append("empty_cache"),
    )

    worker._release_unoccupied_accelerator_memory()

    assert calls == [
        ("synchronize", torch.device("cuda:3")),
        "collect",
        "empty_cache",
    ]


def test_manual_kv_profile_warms_kernels_before_reservation(
    monkeypatch: pytest.MonkeyPatch,
):
    worker = object.__new__(Worker)
    worker.device = torch.device("cuda:3")
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=1024)
    worker.model_runner = SimpleNamespace()
    worker.model_config = SimpleNamespace(multimodal_config=None)
    worker.parallel_config = SimpleNamespace(_api_process_count=1)
    worker.init_snapshot = SimpleNamespace(free_memory=2048)
    calls: list[object] = []
    worker.model_runner.profile_run = lambda: calls.append("profile")
    monkeypatch.setattr(gpu_worker_module, "maybe_apply_startup_plan", lambda _: None)
    monkeypatch.setattr(
        worker,
        "_release_unoccupied_accelerator_memory",
        lambda: calls.append("release"),
    )
    monkeypatch.setattr(
        worker,
        "_warmup_kernels_once",
        lambda: calls.append("kernel_warmup"),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "reserve_mm_ipc_gpu_memory",
        lambda size, *_args: size,
    )

    assert worker.determine_available_memory() == 1024
    assert calls == [
        "release",
        "profile",
        "release",
        "kernel_warmup",
        "release",
    ]


@pytest.mark.parametrize("kv_cache_memory_bytes", [None, 0, 1024])
def test_manual_kv_graph_capture_reclaims_allocator_slack(
    kv_cache_memory_bytes: int | None,
):
    worker = object.__new__(Worker)
    calls: list[str] = []
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=kv_cache_memory_bytes)
    worker.model_runner = SimpleNamespace(
        capture_model=lambda: calls.append("capture") or 17,
    )
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
