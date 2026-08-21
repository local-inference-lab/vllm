# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.linear as linear
import vllm.model_executor.parameter as parameter
from vllm.config.virtual_tp import VIRTUAL_TP_PLAN_ATTR
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    apply_rope,
    mm_projector_forward,
)
from vllm.transformers_utils.configs.kimi_k3 import KimiK3VisionConfig


@pytest.mark.parametrize(
    ("use_data_parallel", "parallel_hidden_size", "linear_tp_size"),
    [(False, 24, 3), (True, 16, 1)],
)
def test_kimi_projector_shards_only_in_weight_tp_mode(
    monkeypatch: pytest.MonkeyPatch,
    use_data_parallel: bool,
    parallel_hidden_size: int,
    linear_tp_size: int,
):
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)

    config = KimiK3VisionConfig(
        vt_hidden_size=4,
        vt_intermediate_size=16,
        merge_kernel_size=(2, 2),
        text_hidden_size=8,
    )
    setattr(
        config,
        VIRTUAL_TP_PLAN_ATTR,
        {
            "vision_projector_hidden_size": {
                "original_size": 16,
                "padded_size": parallel_hidden_size,
                "tp_size": linear_tp_size,
                "local_size": parallel_hidden_size // linear_tp_size,
            }
        },
    )

    projector = KimiK25MultiModalProjector(
        config,
        use_data_parallel=use_data_parallel,
    )

    assert isinstance(projector.linear_1, ColumnParallelLinear)
    assert isinstance(projector.linear_2, RowParallelLinear)
    assert projector.parallel_hidden_size == parallel_hidden_size
    assert projector.linear_1.tp_size == linear_tp_size
    assert projector.linear_1.output_size_per_partition == (
        parallel_hidden_size // linear_tp_size
    )
    assert projector.linear_2.tp_size == linear_tp_size
    assert projector.linear_2.input_size_per_partition == (
        parallel_hidden_size // linear_tp_size
    )


class _SerializedFp8Projector(nn.Module):
    def __init__(self):
        super().__init__()
        self.serialized_weight = nn.Parameter(
            torch.empty(1, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.pre_norm = nn.LayerNorm(4, dtype=torch.bfloat16)
        self.input_dtype: torch.dtype | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.input_dtype = inputs.dtype
        return inputs


def test_kimi_projector_uses_norm_activation_dtype_for_fp8_weights():
    projector = _SerializedFp8Projector()

    outputs = mm_projector_forward(
        projector,
        [torch.randn(2, 4), torch.randn(1, 4)],
    )

    assert projector.input_dtype == torch.bfloat16
    assert [output.shape for output in outputs] == [(2, 4), (1, 4)]
    assert all(output.dtype == torch.bfloat16 for output in outputs)


def test_kimi_vision_rope_reuses_packed_qk_buffers():
    packed_qkv = torch.randn(17, 3, 4, 8, dtype=torch.bfloat16)
    query, key, _ = torch.unbind(packed_qkv, dim=1)
    phases = torch.randn(17, 4)
    frequencies = torch.polar(torch.ones_like(phases), phases)

    def reference(inputs: torch.Tensor) -> torch.Tensor:
        complex_inputs = torch.view_as_complex(
            inputs.float().view(*inputs.shape[:-1], -1, 2)
        )
        rotated = complex_inputs * frequencies.unsqueeze(-2)
        return torch.view_as_real(rotated).flatten(-2).to(inputs.dtype)

    expected_query = reference(query)
    expected_key = reference(key)
    query_pointer = query.data_ptr()
    key_pointer = key.data_ptr()

    output_query, output_key = apply_rope(query, key, frequencies)

    assert output_query.data_ptr() == query_pointer
    assert output_key.data_ptr() == key_pointer
    torch.testing.assert_close(output_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(output_key, expected_key, rtol=0, atol=0)
