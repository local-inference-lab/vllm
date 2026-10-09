# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    copy_tensor_parallel_shard,
)
from vllm.model_executor.weight_transfer import weight_transfer


def test_tensor_parallel_shard_padding_is_explicit() -> None:
    weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    shard = torch.empty(4, 2)

    with pytest.raises(RuntimeError):
        copy_tensor_parallel_shard(shard, weight, 0, 8, 4)

    copy_tensor_parallel_shard(shard, weight, 0, 8, 4, allow_padding=True)

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


@pytest.mark.parametrize("rank", [0, 2, 3])
@pytest.mark.parametrize(
    "loader", ["column", "row", "merged", "qkv", "legacy_column", "legacy_row"]
)
def test_padded_shards_preserve_lazy_checkpoint_source(monkeypatch, rank, loader):
    """Padding must retain file provenance and write into existing weight storage."""
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: rank
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 4
    )
    dim = 1 if "row" in loader else 0
    fused = loader in ("merged", "qkv")
    shape = [2, 2]
    shape[dim] = 6 if fused else 4
    param = ModelWeightParameter(
        data=torch.full(shape, -1.0),
        input_dim=1,
        output_dim=0,
        weight_loader=lambda *_: None,
    )
    param.allow_tp_padding = True
    shape[dim] = 10
    values = torch.arange(20, dtype=torch.float32).reshape(shape)
    source = torch.empty_like(values, device="meta")
    destination_storage = param.untyped_storage()._cdata

    def direct_read(destination, view):
        assert view.untyped_storage()._cdata == source.untyped_storage()._cdata
        assert view._version == 0
        assert destination.untyped_storage()._cdata == destination_storage
        destination.copy_(
            values.as_strided(view.shape, view.stride(), view.storage_offset())
        )
        return True

    with weight_transfer(direct_read):
        if loader.startswith("legacy"):
            layer = SimpleNamespace(tp_rank=rank)
            cls = RowParallelLinear if dim else ColumnParallelLinear
            cls.weight_loader(layer, param, source)
        elif fused:
            kwargs = {"shard_offset": 1, "shard_size": 4}
            if loader == "qkv":
                param.load_qkv_weight(source, **kwargs, shard_id="q", num_heads=1)
            else:
                param.load_merged_column_weight(source, **kwargs)
        else:
            getattr(param, f"load_{loader}_parallel_weight")(source)

    expected = torch.full_like(param, -1.0)
    shard = expected.narrow(dim, 1 if fused else 0, 4)
    shard.zero_()
    available = max(0, min(4, 10 - rank * 4))
    if available:
        shard.narrow(dim, 0, available).copy_(values.narrow(dim, rank * 4, available))
    torch.testing.assert_close(param, expected)
