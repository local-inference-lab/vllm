# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_reduce_in_place(
    input_: torch.Tensor, *, borrow_output: bool = False
) -> torch.Tensor:
    """All-reduce a dead input tensor without allocating an output tensor.

    With ``borrow_output`` the result may be communicator-owned storage that
    the next same-shape reduction overwrites (see
    ``GroupCoordinator.all_reduce_in_place``).
    """
    if borrow_output:
        return get_tp_group().all_reduce_in_place(input_, borrow_output=True)
    return get_tp_group().all_reduce_in_place(input_)


def tensor_model_parallel_is_borrowed_storage(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` aliases storage a borrowed reduction returned."""
    return get_tp_group().is_borrowed_reduction_storage(tensor)


def tensor_model_parallel_pcie_all_gather_pair(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, Any] | None:
    """Gather two rank-local ``[rows, c]`` blocks on the TP group's
    copy-engine ring side stream; ``None`` when unavailable."""
    return get_tp_group().pcie_all_gather_pair(first, second)


def tensor_model_parallel_prepare_pcie_reduce_scatter(wire: str) -> bool:
    """Compile the TP ring's reduce-scatter kernels for ``wire`` ahead of
    any kernel freeze or graph capture; ``False`` without a ring."""
    return get_tp_group().pcie_prepare_reduce_scatter(wire)


def tensor_model_parallel_pcie_reduce_scatter_columns(
    input_: torch.Tensor, *, wire: str, cols: int
) -> torch.Tensor | None:
    """Reduce ``input_`` across the TP group on the copy-engine ring and
    return this rank's ``[rows, cols]`` column block; ``None`` when
    unavailable."""
    return get_tp_group().pcie_reduce_scatter_columns(input_, wire=wire, cols=cols)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_all_gatherv(
    input_: torch.Tensor, sizes: list[int], dim: int = 0
) -> torch.Tensor:
    """All-gather variable-length tensor slices across the model-parallel group."""
    tp_group = get_tp_group()
    if tp_group.world_size == 1:
        return input_
    return tp_group.all_gatherv(input_, dim=dim, sizes=sizes)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
