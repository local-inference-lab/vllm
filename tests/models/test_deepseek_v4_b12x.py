# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12x DeepSeek V4 preparation declarations."""

from types import SimpleNamespace

import pytest
import torch


def test_wo_preparation_declares_fixed_and_dynamic_capacity_plans(monkeypatch) -> None:
    from vllm.models.deepseek_v4.nvidia import b12x
    from vllm.utils.b12x import B12xWorkload, PreparationResourceUnavailableError

    class Plan:
        def __init__(self, rows, dynamic_tokens):
            self.rows = rows
            self.dynamic_tokens = dynamic_tokens

        def request(self, name, **kwargs):
            return SimpleNamespace(name=name, plan=self, **kwargs)

    layer = b12x.DeepseekV4B12xAttention.__new__(b12x.DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "test.v4"
    layer._b12x_wo_resolver = None
    layer._b12x_wo_projection_weights = object()
    layer.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty(1))
    layer.n_local_heads, layer.n_local_groups = 8, 2
    layer.head_dim, layer.nope_head_dim, layer.rope_head_dim = 128, 96, 32
    declarations = []
    def declare(rows, dynamic_tokens):
        declarations.append((rows, dynamic_tokens))
        return Plan(rows, dynamic_tokens)

    monkeypatch.setattr(layer, "_declare_b12x_wo_plan", declare)
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 4, 32), fixed_token_counts=(1, 4),
        output_dtype=torch.bfloat16, max_tokens=32, max_seqs=2, max_model_len=32,
    )
    with pytest.raises(PreparationResourceUnavailableError):
        layer._b12x_wo_plan(1)
    (unit,) = layer.get_b12x_preparation_units(layer, workload)
    assert declarations == [(1, False), (4, False), (32, True)]
    assert [request.plan.rows for request in unit.requests] == [1, 4, 32]
    assert layer._b12x_wo_plan(4).dynamic_tokens is False
    assert layer._b12x_wo_plan(17).rows == 32
    assert declarations == [(1, False), (4, False), (32, True)]
    with pytest.raises(PreparationResourceUnavailableError):
        layer._b12x_wo_plan(33)
