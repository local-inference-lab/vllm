# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm.v1.kv_offload.cpu import host_mem_ops


def _driver_result(code: int) -> tuple[SimpleNamespace]:
    return (SimpleNamespace(value=code),)


def test_cuda_registration_uses_driver_api(monkeypatch) -> None:
    driver = MagicMock()
    driver.cuMemHostRegister.return_value = _driver_result(2)
    driver.cuMemHostUnregister.return_value = _driver_result(0)
    runtime_factory = MagicMock()
    monkeypatch.setattr(host_mem_ops.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(host_mem_ops, "_get_cuda_driver", lambda: driver)
    monkeypatch.setattr(host_mem_ops, "CudaRTLibrary", runtime_factory)

    assert host_mem_ops.register_host_memory(0x1000, 4096) == 2
    assert host_mem_ops.unregister_host_memory(0x1000) == 0

    driver.cuMemHostRegister.assert_called_once_with(0x1000, 4096, 0)
    driver.cuMemHostUnregister.assert_called_once_with(0x1000)
    runtime_factory.assert_not_called()


def test_rocm_registration_clears_runtime_failures(monkeypatch) -> None:
    cudart = MagicMock()
    cudart.cudaHostRegister.return_value = SimpleNamespace(value=1)
    cudart.cudaHostUnregister.return_value = SimpleNamespace(value=2)
    runtime = MagicMock()
    monkeypatch.setattr(host_mem_ops.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(host_mem_ops, "CudaRTLibrary", lambda: runtime)

    assert host_mem_ops.register_host_memory(0x2000, 8192) == 1
    assert host_mem_ops.unregister_host_memory(0x2000) == 2

    assert runtime.cudaGetLastError.call_count == 2
