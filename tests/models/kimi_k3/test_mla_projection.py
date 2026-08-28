# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia import mla as mla_module
from vllm.models.kimi_k3.nvidia.mla import (
    KimiShardedMergedQKVGateLinear,
    _restore_merged_output_order,
    _shard_qkv_a_projection,
)


class _UnquantizedOutputLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int, tp_size: int = 2):
        super().__init__()
        self.quant_method = UnquantizedLinearMethod()
        self.input_is_parallel = True
        self.input_size_per_partition = input_size
        self.output_size = output_size
        self.reduce_results = True
        self.tp_size = tp_size
        self.register_parameter("bias", None)
        self.weight = nn.Parameter(
            torch.randn(output_size, input_size),
            requires_grad=False,
        )


def test_output_gate_uses_bounded_compile_config() -> None:
    compiler_config = mla_module._gate_sigmoid_mul.get_compiler_config()

    assert compiler_config["triton.autotune_pointwise"] is False


def test_mla_projects_prefill_into_caller_storage(monkeypatch) -> None:
    layer = object.__new__(mla_module.MultiHeadLatentAttention)
    nn.Module.__init__(layer)
    layer.o_proj = _UnquantizedOutputLinear(input_size=3, output_size=4)
    attn_out = torch.randn(1024, 3)
    output = torch.empty(1024, 4)
    expected = torch.mm(attn_out, layer.o_proj.weight.t())
    reduce_in_place = Mock(side_effect=lambda value, tp_size: value)
    monkeypatch.setattr(
        mla_module,
        "reduce_kimi_full_width_projection",
        reduce_in_place,
    )

    with torch.inference_mode():
        actual = layer._project_output_into(attn_out, output)

    assert actual is output
    torch.testing.assert_close(actual, expected)
    reduce_in_place.assert_called_once_with(output, 2)


def test_mla_caller_output_selection_preserves_decode_path() -> None:
    layer = object.__new__(mla_module.MultiHeadLatentAttention)
    nn.Module.__init__(layer)
    layer.o_proj = _UnquantizedOutputLinear(input_size=3, output_size=4)

    assert not layer.should_use_caller_output(torch.empty(8, 4))
    assert layer.should_use_caller_output(torch.empty(1024, 4))
    assert not layer.should_use_caller_output(torch.empty(1024, 3))


def test_mla_caller_output_rejects_projection_input_alias() -> None:
    layer = object.__new__(mla_module.MultiHeadLatentAttention)
    nn.Module.__init__(layer)
    layer.o_proj = _UnquantizedOutputLinear(input_size=4, output_size=4)
    aliased = torch.empty(1024, 4)

    with pytest.raises(ValueError, match="must not alias"):
        layer._project_output_into(aliased, aliased)


def test_restore_merged_output_order() -> None:
    rank_major = torch.tensor(
        [
            [0, 1, 100, 101, 102],
            [2, 3, 103, 104, 105],
            [4, 5, 106, 107, 108],
            [6, 7, 109, 110, 111],
        ]
    ).flatten()

    output = _restore_merged_output_order(rank_major, [8, 12], tp_size=4)

    torch.testing.assert_close(
        output,
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
            ]
        ),
    )


def test_restore_merged_output_order_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="Unexpected gathered merged projection"):
        _restore_merged_output_order(torch.zeros(19), [8, 12], tp_size=4)


@pytest.mark.parametrize(
    ("additional_config", "tp_size", "expected"),
    [
        ({}, 16, False),
        ({"kimi_shard_qkv_a": True}, 1, False),
        ({"kimi_shard_qkv_a": True}, 16, True),
    ],
)
def test_shard_qkv_a_projection_selection(
    additional_config: dict[str, bool],
    tp_size: int,
    expected: bool,
) -> None:
    config = SimpleNamespace(additional_config=additional_config)

    assert _shard_qkv_a_projection(config, [1536, 576], tp_size) is expected


def test_shard_qkv_a_projection_rejects_fractional_shards() -> None:
    config = SimpleNamespace(additional_config={"kimi_shard_qkv_a": True})

    with pytest.raises(ValueError, match="divisible by TP=10"):
        _shard_qkv_a_projection(config, [1536, 576], tp_size=10)


def test_sharded_qkv_gate_gathers_only_latent_projection(monkeypatch) -> None:
    layer = object.__new__(KimiShardedMergedQKVGateLinear)
    nn.Module.__init__(layer)
    layer.tp_size = 2
    layer.qkv_output_sizes = [4, 4]
    layer.qkv_local_width = 4
    layer.gate_local_width = 2
    layer.use_b12x = True
    local_output = torch.tensor([[0, 1, 100, 101, 900, 901]])
    rank_major_qkv = torch.tensor([[0, 1, 100, 101, 2, 3, 102, 103]])
    monkeypatch.setattr(
        mla_module.MergedColumnParallelLinear,
        "forward",
        lambda self, x: (local_output, None),
    )
    gather = Mock(return_value=rank_major_qkv)
    monkeypatch.setattr(mla_module, "gather_kimi_sharded_projection", gather)

    output, bias = layer(torch.empty(1, 1))

    assert bias is None
    torch.testing.assert_close(
        output,
        torch.tensor([[0, 1, 2, 3, 100, 101, 102, 103, 900, 901]]),
    )
    gather.assert_called_once()
    torch.testing.assert_close(gather.call_args.args[0], local_output[:, :4])
    assert gather.call_args.kwargs == {"use_b12x": True}
