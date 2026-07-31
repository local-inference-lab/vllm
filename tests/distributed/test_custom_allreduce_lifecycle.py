# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.parallel_state import GroupCoordinator


def make_b12x_custom_allreduce() -> tuple[CustomAllreduce, MagicMock]:
    runtime = MagicMock()
    custom_allreduce = object.__new__(CustomAllreduce)
    custom_allreduce.disabled = False
    custom_allreduce._pcie_runtime = runtime
    custom_allreduce._pcie_dma = None
    custom_allreduce._ptr = 0
    return custom_allreduce, runtime


def test_b12x_destructor_skips_collective_close_without_retention() -> None:
    custom_allreduce, runtime = make_b12x_custom_allreduce()
    dma = MagicMock()
    custom_allreduce._pcie_dma = dma
    custom_allreduce_ref = weakref.ref(custom_allreduce)

    del custom_allreduce
    gc.collect()

    dma.close.assert_not_called()
    runtime.close.assert_not_called()
    assert custom_allreduce_ref() is None


def test_b12x_explicit_close_remains_coordinated() -> None:
    custom_allreduce, runtime = make_b12x_custom_allreduce()
    dma = MagicMock()
    custom_allreduce._pcie_dma = dma
    close_order = []
    runtime.close.side_effect = lambda: close_order.append("runtime")
    dma.close.side_effect = lambda: close_order.append("dma")

    custom_allreduce.close()

    assert close_order == ["runtime", "dma"]
    dma.close.assert_called_once_with()
    runtime.close.assert_called_once_with()
    assert custom_allreduce._pcie_dma is None
    assert custom_allreduce._pcie_runtime is None


def test_b12x_close_retains_owners_when_runtime_close_fails() -> None:
    custom_allreduce, runtime = make_b12x_custom_allreduce()
    dma = MagicMock()
    runtime.close.side_effect = RuntimeError("close failed")
    custom_allreduce._pcie_dma = dma

    with pytest.raises(RuntimeError, match="close failed"):
        custom_allreduce.close()

    assert custom_allreduce._pcie_runtime is runtime
    assert custom_allreduce._pcie_dma is dma
    dma.close.assert_not_called()


def test_cuda_communicator_closes_custom_allreduce_before_nccl() -> None:
    communicator = object.__new__(CudaCommunicator)
    communicator.ca_comm = MagicMock()
    communicator.pynccl_comm = MagicMock()
    communicator.aiter_ar_comm = None
    communicator.fi_ar_comm = None
    communicator.all2all_manager = None
    close_order = []
    communicator.ca_comm.close.side_effect = lambda: close_order.append("custom")
    communicator.pynccl_comm.destroy.side_effect = lambda: close_order.append("nccl")

    communicator.destroy()

    assert close_order == ["custom", "nccl"]
    assert communicator.ca_comm is None
    assert communicator.pynccl_comm is None


def test_cuda_communicator_retains_owners_when_custom_close_fails() -> None:
    communicator = object.__new__(CudaCommunicator)
    custom_allreduce = MagicMock()
    custom_allreduce.close.side_effect = RuntimeError("close failed")
    communicator.ca_comm = custom_allreduce
    communicator.pynccl_comm = MagicMock()
    communicator.aiter_ar_comm = None
    communicator.fi_ar_comm = None
    communicator.all2all_manager = None

    with pytest.raises(RuntimeError, match="close failed"):
        communicator.destroy()

    assert communicator.ca_comm is custom_allreduce
    communicator.pynccl_comm.destroy.assert_not_called()


def test_group_destroy_keeps_process_groups_live_for_communicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = object.__new__(GroupCoordinator)
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    coordinator.device_communicator = MagicMock()
    coordinator.mq_broadcaster = None
    device_group = coordinator.device_group
    destroy_order = []
    coordinator.device_communicator.destroy.side_effect = lambda: destroy_order.append(
        "communicator"
    )
    monkeypatch.setattr(
        torch.distributed,
        "destroy_process_group",
        lambda group: destroy_order.append(
            "device" if group is device_group else "cpu"
        ),
    )

    coordinator.destroy()

    assert destroy_order == ["communicator", "device", "cpu"]


def test_group_destroy_retains_process_groups_when_communicator_close_fails() -> None:
    coordinator = object.__new__(GroupCoordinator)
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    coordinator.device_communicator = MagicMock()
    coordinator.device_communicator.destroy.side_effect = RuntimeError("close failed")
    coordinator.mq_broadcaster = None

    with pytest.raises(RuntimeError, match="close failed"):
        coordinator.destroy()

    assert hasattr(coordinator, "device_group")
    assert hasattr(coordinator, "cpu_group")
