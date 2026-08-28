# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

import vllm.distributed.communication_op as communication_op
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.parallel_state import GroupCoordinator


def _make_group(world_size: int) -> GroupCoordinator:
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = world_size
    group.device_communicator = None
    return group


def test_tensor_model_parallel_all_reduce_in_place_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.arange(4, dtype=torch.float32)
    group = Mock()

    def reduce_in_place(value: torch.Tensor) -> torch.Tensor:
        value.add_(1)
        return value

    group.all_reduce_in_place.side_effect = reduce_in_place
    monkeypatch.setattr(communication_op, "get_tp_group", lambda: group)

    result = communication_op.tensor_model_parallel_all_reduce_in_place(tensor)

    assert result is tensor
    torch.testing.assert_close(tensor, torch.arange(4, dtype=torch.float32) + 1)
    group.all_reduce_in_place.assert_called_once_with(tensor)


def test_group_all_reduce_in_place_preserves_single_rank_storage() -> None:
    tensor = torch.arange(4)
    group = _make_group(world_size=1)

    assert group.all_reduce_in_place(tensor) is tensor


def test_group_all_reduce_in_place_delegates_to_device_communicator() -> None:
    tensor = torch.arange(4)
    group = _make_group(world_size=2)
    communicator = Mock()
    communicator.all_reduce_in_place.return_value = tensor
    group.device_communicator = communicator

    result = group.all_reduce_in_place(tensor)

    assert result is tensor
    communicator.all_reduce_in_place.assert_called_once_with(tensor)


def test_cuda_all_reduce_in_place_uses_pynccl_alias() -> None:
    tensor = torch.arange(4, dtype=torch.float32)
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    pynccl = Mock()
    pynccl.disabled = False

    def reduce(value: torch.Tensor, *, out_tensor: torch.Tensor) -> torch.Tensor:
        assert value is tensor
        assert out_tensor is tensor
        out_tensor.add_(2)
        return out_tensor

    pynccl.all_reduce.side_effect = reduce
    communicator.pynccl_comm = pynccl

    result = communicator.all_reduce_in_place(tensor)

    assert result is tensor
    torch.testing.assert_close(tensor, torch.arange(4, dtype=torch.float32) + 2)


def test_cuda_all_reduce_in_place_uses_torch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.arange(4, dtype=torch.float32)
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.pynccl_comm = None
    communicator.device_group = object()

    def reduce(value: torch.Tensor, *, group: object) -> None:
        assert value is tensor
        assert group is communicator.device_group
        value.add_(3)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    result = communicator.all_reduce_in_place(tensor)

    assert result is tensor
    torch.testing.assert_close(tensor, torch.arange(4, dtype=torch.float32) + 3)
