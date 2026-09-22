# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Projection loading and GQA invariants for padded DSpark tensor parallelism."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.qwen3_dflash import (
    _dflash_padded_width,
    _enable_dflash_tp_padding,
)


def _weight(shape):
    parameter = nn.Parameter(
        torch.full(shape, float("nan"), dtype=torch.float64), requires_grad=False
    )
    parameter.input_dim = 1
    parameter.output_dim = 0
    layer = nn.Module()
    layer.register_parameter("weight", parameter)
    layer.quant_method = UnquantizedLinearMethod()
    _enable_dflash_tp_padding(layer)
    return parameter


@pytest.mark.parametrize("tp,padded_kv", [(8, 16), (10, 20), (12, 24)])
def test_padded_gqa_projection_matches_unsharded_attention(tp, padded_kv):
    """Real checkpoint heads contribute once; absent heads contribute zero."""
    generator = torch.Generator().manual_seed(123)
    hidden, heads, kv_heads, dim, rows = 4, 96, 16, 2, 3
    x = torch.randn(rows, hidden, generator=generator, dtype=torch.float64)
    weights = [
        torch.randn(h * dim, hidden, generator=generator, dtype=x.dtype)
        for h in (heads, kv_heads, kv_heads)
    ]
    output_weight = torch.randn(hidden, heads * dim, generator=generator, dtype=x.dtype)

    def attention(q, k, v, num_heads, num_kv):
        q = q.view(rows, num_heads, dim).transpose(0, 1)
        k = k.view(rows, num_kv, dim).transpose(0, 1)
        v = v.view(rows, num_kv, dim).transpose(0, 1)
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return out.transpose(0, 1).reshape(rows, num_heads * dim)

    q, k, v = (F.linear(x, w) for w in weights)
    reference = F.linear(attention(q, k, v, heads, kv_heads), output_weight)
    local_kv = padded_kv // tp
    local_q = local_kv * (heads // kv_heads)
    widths = [local_q * dim, local_kv * dim, local_kv * dim]
    partials = []
    for rank in range(tp):
        qkv = _weight((sum(widths), hidden))
        projection = SimpleNamespace(
            tp_rank=rank,
            num_heads=local_q,
            num_kv_heads=local_kv,
            num_kv_head_replicas=1,
            head_size=dim,
            v_head_size=dim,
        )
        projection.validate_shard_id = MethodType(
            QKVParallelLinear.validate_shard_id, projection
        )
        for key, weight in zip(("q", "k", "v"), weights):
            QKVParallelLinear.weight_loader(projection, qkv, weight, key)
        q, k, v = F.linear(x, qkv).split(widths, dim=-1)
        out_weight = _weight((hidden, local_q * dim))
        RowParallelLinear.weight_loader(
            SimpleNamespace(tp_rank=rank), out_weight, output_weight
        )
        result = F.linear(attention(q, k, v, local_q, local_kv), out_weight)
        assert torch.isfinite(result).all()
        if rank * local_kv >= kv_heads:
            assert torch.count_nonzero(qkv) == 0
            assert torch.count_nonzero(out_weight) == 0
            assert torch.count_nonzero(result) == 0
        partials.append(result)
    torch.testing.assert_close(sum(partials), reference, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("width", [7168, 14336])
def test_draft_projection_padding_retains_supported_shapes(width):
    assert _dflash_padded_width(width, 16) == width
    padded = _dflash_padded_width(width, 10)
    assert padded >= width and padded % (10 * 128) == 0


def test_draft_padding_rejects_serialized_quantization():
    layer = nn.Module()
    layer.quant_method = object()
    with pytest.raises(ValueError, match="BF16/FP16 checkpoints"):
        _enable_dflash_tp_padding(layer)
