# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A worker owns prepared resources before EngineCore initialization completes."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.worker import gpu_worker


@pytest.mark.parametrize("error", [RuntimeError("warmup"), KeyboardInterrupt()])
def test_warmup_failure_releases_worker_and_preserves_error(monkeypatch, error):
    monkeypatch.setattr(gpu_worker, "set_current_vllm_config", lambda _: nullcontext())
    worker = SimpleNamespace(
        vllm_config=None,
        _compile_or_warm_up_model_after_preparation=Mock(side_effect=error),
        shutdown=Mock(),
    )
    with pytest.raises(type(error)) as caught:
        gpu_worker.Worker.compile_or_warm_up_model(worker)
    assert caught.value is error
    worker.shutdown.assert_called_once()


def test_successful_warmup_keeps_worker_alive(monkeypatch):
    monkeypatch.setattr(gpu_worker, "set_current_vllm_config", lambda _: nullcontext())
    worker = SimpleNamespace(
        vllm_config=None,
        _compile_or_warm_up_model_after_preparation=Mock(return_value="ready"),
        shutdown=Mock(),
    )
    assert gpu_worker.Worker.compile_or_warm_up_model(worker) == "ready"
    worker.shutdown.assert_not_called()


@pytest.mark.parametrize("offload_gb", [0, 27])
def test_offload_pool_is_released_before_shutdown_acknowledgement(
    monkeypatch, offload_gb
):
    events = []
    monkeypatch.setattr(gpu_worker, "ensure_kv_transfer_shutdown", None)
    monkeypatch.setattr(gpu_worker, "ensure_ec_transfer_shutdown", None)
    monkeypatch.setattr(gpu_worker.current_platform, "is_cuda_alike", lambda: False)
    monkeypatch.setattr(gpu_worker.gc, "unfreeze", lambda: None)
    monkeypatch.setattr(gpu_worker.gc, "collect", lambda: events.append("collect"))
    monkeypatch.setattr(gpu_worker.torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        gpu_worker.torch.accelerator,
        "empty_host_cache",
        lambda: events.append("host_pool"),
    )
    worker = SimpleNamespace(
        vllm_config=SimpleNamespace(
            offload_config=SimpleNamespace(
                uva=SimpleNamespace(cpu_offload_gb=offload_gb)
            )
        ),
        profiler=None,
        elastic_ep_executor=SimpleNamespace(shutdown=lambda: None),
        model_runner=SimpleNamespace(shutdown=lambda: events.append("readers_owners")),
        _record_b12x_lifecycle=lambda stage: events.append(stage),
    )
    gpu_worker.Worker.shutdown(worker)
    expected = ["before_worker_shutdown", "readers_owners", "collect"]
    if offload_gb:
        expected.append("host_pool")
    assert events == expected + ["after_worker_shutdown"]
