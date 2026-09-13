# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    load_tensor_parallel_shard,
)


def test_tensor_parallel_shard_padding_is_explicit() -> None:
    weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    with pytest.raises(RuntimeError):
        load_tensor_parallel_shard(weight, 0, 8, 4)

    shard = load_tensor_parallel_shard(weight, 0, 8, 4, allow_padding=True)

    torch.testing.assert_close(shard[:2], weight[8:])
    assert torch.count_nonzero(shard[2:]) == 0


def test_column_parameter_zero_fills_padded_checkpoint_tail(monkeypatch) -> None:
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 3
    )
    param = ModelWeightParameter(
        data=torch.empty(4, 2),
        input_dim=1,
        output_dim=0,
        weight_loader=lambda *_: None,
    )
    param.tp_rank = 2
    param.tp_size = 3
    param.allow_tp_padding = True
    weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    param.load_column_parallel_weight(weight)

    torch.testing.assert_close(param[:2], weight[8:])
    assert torch.count_nonzero(param[2:]) == 0


def test_row_parameter_zero_fills_padded_checkpoint_tail(monkeypatch) -> None:
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 3
    )
    param = ModelWeightParameter(
        data=torch.empty(2, 4),
        input_dim=1,
        output_dim=0,
        weight_loader=lambda *_: None,
    )
    param.tp_rank = 2
    param.tp_size = 3
    param.allow_tp_padding = True
    weight = torch.arange(20, dtype=torch.float32).reshape(2, 10)

    param.load_row_parallel_weight(weight)

    torch.testing.assert_close(param[:, :2], weight[:, 8:])
    assert torch.count_nonzero(param[:, 2:]) == 0
