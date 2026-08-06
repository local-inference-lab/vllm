# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.linear as linear_mod
import vllm.model_executor.layers.mamba.gdn.base as gdn_base
import vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn as kimi_gdn
import vllm.model_executor.parameter as parameter_mod
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_mxfp8 as mxfp8_scheme,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Mxfp8SerializedLinearMethod,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


class _FakeLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | list[int],
        *args,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        del args, kwargs
        self.input_size = input_size
        self.output_sizes = (
            list(output_size) if isinstance(output_size, list) else [output_size]
        )
        self.output_size = sum(self.output_sizes)
        self.prefix = prefix
        self.bias = None
        self.weight = nn.Parameter(
            torch.empty(self.output_size, input_size),
            requires_grad=False,
        )
        self.fixed_output: torch.Tensor | None = None

    def forward(self, x: torch.Tensor):
        output = self.fixed_output
        if output is None:
            output = x.new_zeros((x.shape[0], self.output_size))
        return output, None


class _FakeRMSNormGated(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        del gate
        return x


def _make_kda(
    monkeypatch: pytest.MonkeyPatch,
) -> kimi_gdn.KimiGatedDeltaNetAttention:
    merged_calls: list[tuple[int, list[int], str]] = []

    def merged_linear(input_size, output_sizes, *args, prefix="", **kwargs):
        merged_calls.append((input_size, list(output_sizes), prefix))
        return _FakeLinear(
            input_size,
            output_sizes,
            *args,
            prefix=prefix,
            **kwargs,
        )

    monkeypatch.setattr(gdn_base, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(gdn_base, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(kimi_gdn, "MergedColumnParallelLinear", merged_linear)
    monkeypatch.setattr(kimi_gdn, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(kimi_gdn, "ReplicatedLinear", _FakeLinear)
    monkeypatch.setattr(kimi_gdn, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(kimi_gdn, "FusedRMSNormGated", _FakeRMSNormGated)

    config = SimpleNamespace(
        hidden_size=32,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(),
        quant_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(
        kimi_gdn,
        "get_current_vllm_config",
        lambda: vllm_config,
    )

    layer = kimi_gdn.KimiGatedDeltaNetAttention(
        config,
        vllm_config,
        prefix="model.layers.0.self_attn",
    )
    assert merged_calls == [
        (
            32,
            [32, 32, 32],
            "model.layers.0.self_attn.qkv_proj",
        )
    ]
    return layer


def test_kda_constructs_one_merged_qkv_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_kda(monkeypatch)

    assert layer.qkv_proj.output_sizes == [32, 32, 32]
    assert not hasattr(layer, "q_proj")
    assert not hasattr(layer, "k_proj")
    assert not hasattr(layer, "v_proj")


def test_kda_conv_state_reserves_speculative_rollback_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_kda(monkeypatch)
    layer.num_spec = 5

    conv_shape, recurrent_shape = layer.get_state_shape()

    assert conv_shape == (8, 96)
    assert recurrent_shape == (4, 8, 8)


def test_kda_forward_splits_merged_qkv_in_checkpoint_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_kda(monkeypatch)
    hidden_states = torch.zeros(2, 32)
    layer.qkv_proj.fixed_output = torch.cat(
        [
            torch.full((2, 32), 1.0),
            torch.full((2, 32), 2.0),
            torch.full((2, 32), 3.0),
        ],
        dim=-1,
    )
    captured: dict[str, torch.Tensor] = {}

    def kda_attention(q, k, v, g1, beta, core_attn_out, layer_name):
        del g1, beta, layer_name
        captured.update(q=q.clone(), k=k.clone(), v=v.clone())
        core_attn_out.zero_()

    monkeypatch.setattr(torch.ops.vllm, "kda_attention", kda_attention)
    output = torch.empty_like(hidden_states)

    layer(hidden_states, torch.arange(2), output)

    torch.testing.assert_close(captured["q"], torch.full((2, 32), 1.0))
    torch.testing.assert_close(captured["k"], torch.full((2, 32), 2.0))
    torch.testing.assert_close(captured["v"], torch.full((2, 32), 3.0))


def test_kda_spec_decode_uses_speculative_state_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _make_kda(monkeypatch)
    num_tokens = 3
    state_indices = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    accepted_tokens = torch.tensor([2], dtype=torch.int32)
    query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=num_tokens,
        num_actual_tokens=num_tokens,
        spec_query_start_loc=query_start_loc,
        spec_state_indices_tensor=state_indices,
        spec_sequence_masks=torch.tensor([True]),
        spec_token_indx=torch.arange(num_tokens, dtype=torch.int32),
        non_spec_token_indx=torch.empty(0, dtype=torch.int32),
        num_accepted_tokens=accepted_tokens,
    )
    monkeypatch.setattr(
        kimi_gdn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
    )

    conv_calls: list[dict] = []

    def causal_conv1d_update(x, *args, **kwargs):
        del args
        conv_calls.append(kwargs)
        return x

    recurrent_call: dict = {}

    def fused_recurrent_kda(**kwargs):
        recurrent_call.update(kwargs)
        return kwargs["q"].clone(), kwargs["initial_state"]

    monkeypatch.setattr(kimi_gdn, "causal_conv1d_update", causal_conv1d_update)
    monkeypatch.setattr(kimi_gdn, "fused_kda_gate", lambda x, *args, **kwargs: x)
    monkeypatch.setattr(kimi_gdn, "fused_recurrent_kda", fused_recurrent_kda)

    layer.kv_cache = (
        torch.zeros(4, 96, 3),
        torch.zeros(4, 4, 8, 8),
    )
    q = torch.arange(num_tokens * 32, dtype=torch.float32).reshape(num_tokens, 32)
    core_attn_out = torch.zeros(1, num_tokens, 4, 8)

    layer._forward(
        q_proj_states=q,
        k_proj_states=q + 1,
        v_proj_states=q + 2,
        g1=torch.zeros(1, num_tokens, 4, 8),
        beta=torch.zeros(1, num_tokens, 4),
        core_attn_out=core_attn_out,
    )

    assert len(conv_calls) == 3
    for call in conv_calls:
        torch.testing.assert_close(call["conv_state_indices"], state_indices[:, 0])
        assert call["num_accepted_tokens"] is accepted_tokens
        assert call["query_start_loc"] is query_start_loc
    assert recurrent_call["ssm_state_indices"] is state_indices
    assert recurrent_call["num_accepted_tokens"] is accepted_tokens
    torch.testing.assert_close(
        core_attn_out,
        q.reshape(1, num_tokens, 4, 8),
    )


def test_serialized_mxfp8_qkv_loader_preserves_tp_shard_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeKernel:
        def process_weights_after_loading(self, layer) -> None:
            del layer

        def apply_weights(self, layer, x, bias=None):
            raise NotImplementedError

    class QuantConfig:
        def get_quant_method(self, layer, prefix):
            del layer, prefix
            return Mxfp8SerializedLinearMethod()

    monkeypatch.setattr(
        mxfp8_scheme,
        "init_mxfp8_linear_kernel",
        lambda: FakeKernel(),
    )
    for module in (linear_mod, parameter_mod):
        monkeypatch.setattr(
            module,
            "get_tensor_model_parallel_world_size",
            lambda: 12,
        )
        monkeypatch.setattr(
            module,
            "get_tensor_model_parallel_rank",
            lambda: 3,
        )

    layer = MergedColumnParallelLinear(
        32,
        [24, 24, 24],
        bias=False,
        quant_config=QuantConfig(),
        prefix="model.layers.0.self_attn.qkv_proj",
    )
    for shard_id in range(3):
        weight = torch.full(
            (24, 32),
            shard_id + 1,
            dtype=torch.float8_e4m3fn,
        )
        scale = torch.full(
            (24, 1),
            127 + shard_id,
            dtype=torch.uint8,
        )
        layer.weight_loader(layer.weight, weight, shard_id)
        layer.weight_loader(layer.weight_scale, scale, shard_id)

    expected_weight = torch.cat(
        [
            torch.full((2, 32), shard_id + 1, dtype=torch.float8_e4m3fn)
            for shard_id in range(3)
        ]
    )
    expected_scale = torch.cat(
        [torch.full((2, 1), 127 + shard_id, dtype=torch.uint8) for shard_id in range(3)]
    )
    assert torch.equal(layer.weight, expected_weight)
    assert torch.equal(layer.weight_scale, expected_scale)
