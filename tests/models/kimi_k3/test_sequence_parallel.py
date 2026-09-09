# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.config import ParallelConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.models.common.ops import sequence_parallel as sp_ops
from vllm.models.kimi_k3.nvidia import mla as kimi_mla
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import mtp as kimi_mtp
from vllm.platforms import current_platform


class _IdentityNorm(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size), requires_grad=False)
        self.variance_epsilon = 1e-5

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _RecordingMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_tokens = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.num_tokens = hidden_states.shape[0]
        return hidden_states


class _Projection(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(1, hidden_size),
            requires_grad=False,
        )


class _PartialRowProjection(RowParallelLinear):
    def __init__(self, weight: torch.Tensor) -> None:
        nn.Module.__init__(self)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.reduce_results = False

    def forward(self, hidden_states: torch.Tensor):
        return torch.nn.functional.linear(hidden_states, self.weight), None


class _SequenceParallelMTPBlock:
    use_sequence_parallel = True

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        assert residual is None
        return hidden_states * 2, None, hidden_states * 3


@pytest.mark.parametrize(
    ("dcp_world_size", "backend_owns", "expected"),
    ((1, True, False), (8, False, False), (8, True, True)),
)
def test_kimi_detects_backend_owned_decode_dcp(
    dcp_world_size: int, backend_owns: bool, expected: bool
) -> None:
    impl = SimpleNamespace(owns_decode_dcp_collectives=backend_owns)
    assert kimi_mla._backend_owns_decode_dcp(impl, dcp_world_size) is expected


def _mock_sequence_parallel_collectives(monkeypatch):
    monkeypatch.setattr(
        kimi_model,
        "sp_reduce_scatter",
        lambda tensor: tensor.chunk(2, dim=0)[0],
    )
    monkeypatch.setattr(
        kimi_model,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        kimi_model,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )


@pytest.mark.parametrize(
    ("num_tokens", "is_padding", "tp_rank", "expected"),
    [
        (1, None, 0, [False]),
        (1, None, 1, [True]),
        (5, None, 2, [False, True]),
        (5, None, 3, [True, True]),
        (5, [False, True, False, False, False], 0, [False, True]),
    ],
)
def test_sp_padding_mask_marks_added_rows(
    monkeypatch,
    num_tokens: int,
    is_padding: list[bool] | None,
    tp_rank: int,
    expected: list[bool],
):
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: tp_rank)

    hidden_states = torch.empty(num_tokens, 2)
    padding = torch.tensor(is_padding) if is_padding is not None else None
    actual = sp_ops.sp_padding_mask(padding, hidden_states)

    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize(
    ("data_parallel_size", "expected"),
    [
        (1, False),
        (2, True),
    ],
)
def test_moe_sequence_parallel_requires_data_parallel(
    monkeypatch,
    data_parallel_size: int,
    expected: bool,
):
    monkeypatch.setattr(current_platform, "device_count", lambda: 2)
    parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )

    assert parallel_config.use_sequence_parallel_moe is expected


def test_kimi_decoder_layer_keeps_moe_states_sequence_sharded(monkeypatch):
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = False
    layer.use_sequence_parallel = True
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.mlp = _RecordingMoE()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states,
        layer,
    )

    _mock_sequence_parallel_collectives(monkeypatch)

    positions = torch.arange(3)
    full_hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    hidden_states = kimi_model.sp_shard(full_hidden_states)
    hidden_states, prefix_sum, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=None,
    )

    assert prefix_sum is None
    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.mlp.num_tokens == 2

    hidden_states, prefix_sum, residual = layer(
        positions=positions,
        hidden_states=hidden_states,
        residual=residual,
    )

    assert prefix_sum is None
    assert hidden_states.shape == residual.shape == (2, 2)
    assert layer.mlp.num_tokens == 2


def test_kimi_attn_residual_states_stay_sequence_sharded(monkeypatch):
    layer = object.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.use_sequence_parallel = True
    layer.prev_valid_blocks = 0
    layer.block_write_idx = 0
    layer.is_block_write_layer = False
    layer.input_layernorm = _IdentityNorm()
    layer.post_attention_layernorm = _IdentityNorm()
    layer.self_attention_res_norm = _IdentityNorm()
    layer.mlp_res_norm = _IdentityNorm()
    layer.self_attention_res_proj = _Projection()
    layer.mlp_res_proj = _Projection()
    layer.mlp = _RecordingMoE()
    layer._run_self_attn = MethodType(
        lambda self, positions, hidden_states: hidden_states,
        layer,
    )

    _mock_sequence_parallel_collectives(monkeypatch)
    monkeypatch.setattr(
        kimi_model,
        "attn_res",
        lambda prefix_sum, hidden_states, *args, **kwargs: (
            prefix_sum if hidden_states is None else prefix_sum + hidden_states
        ),
    )

    prefix_sum = kimi_model.sp_shard(torch.arange(6, dtype=torch.float32).view(3, 2))
    block_residual = torch.zeros(2, 1, 2)
    hidden_states, prefix_sum, block_residual = layer(
        positions=torch.arange(3),
        hidden_states=None,
        prefix_sum=prefix_sum,
        residual=block_residual,
    )

    assert hidden_states.shape == prefix_sum.shape == (2, 2)
    assert block_residual.shape == (2, 1, 2)
    assert layer.mlp.num_tokens == 2


def test_kimi_mtp_restores_sequence_parallel_output(monkeypatch):
    layer = object.__new__(kimi_mtp.KimiK3MultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = _IdentityNorm()
    layer.hnorm = _IdentityNorm()
    layer.eh_proj = nn.Identity()
    object.__setattr__(layer, "mtp_block", _SequenceParallelMTPBlock())

    final_norm = Mock(side_effect=lambda hidden_states: hidden_states + 1)
    object.__setattr__(
        layer,
        "shared_head",
        SimpleNamespace(norm=final_norm),
    )

    monkeypatch.setattr(
        kimi_mtp,
        "fused_mtp_input",
        lambda positions, inputs_embeds, *args: inputs_embeds,
    )
    monkeypatch.setattr(
        kimi_mtp,
        "sp_shard",
        lambda tensor: torch.nn.functional.pad(tensor, (0, 0, 0, 1))[:2],
    )
    monkeypatch.setattr(
        kimi_mtp,
        "sp_all_gather",
        lambda tensor: torch.cat([tensor, tensor], dim=0),
    )

    inputs_embeds = torch.arange(6, dtype=torch.float32).view(3, 2)
    logits_hidden_states, hidden_states = layer(
        input_ids=torch.zeros(3, dtype=torch.long),
        positions=torch.arange(3),
        previous_hidden_states=torch.zeros_like(inputs_embeds),
        inputs_embeds=inputs_embeds,
    )

    sharded_states = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, 1))[:2]
    expected_hidden_states = torch.cat(
        [sharded_states * 5, sharded_states * 5],
        dim=0,
    )[:3]
    torch.testing.assert_close(hidden_states, expected_hidden_states)
    torch.testing.assert_close(logits_hidden_states, expected_hidden_states + 1)
    final_norm.assert_called_once()
    torch.testing.assert_close(final_norm.call_args.args[0], expected_hidden_states)


@pytest.mark.parametrize(
    ("enabled", "use_sequence_parallel", "tp_size", "expected"),
    [
        (True, True, 8, True),
        (False, True, 8, False),  # opt-in only
        (True, False, 8, False),  # replication only exists under SP
        (True, True, 1, False),  # nothing to shard
        (True, True, 5, False),  # 6144 % 5 -- would fail divide()
    ],
)
def test_shard_sequence_parallel_mlp_gating(
    monkeypatch,
    enabled: bool,
    use_sequence_parallel: bool,
    tp_size: int,
    expected: bool,
):
    monkeypatch.setattr(kimi_model.envs, "VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT", enabled)
    monkeypatch.setattr(
        kimi_model, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    assert (
        kimi_model.shard_sequence_parallel_mlp(
            hidden_size=7168,
            intermediate_size=6144,
            use_sequence_parallel=use_sequence_parallel,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("use_sequence_parallel", "tp_size", "expected"),
    [
        (False, 6, True),
        (True, 6, False),
        (False, 1, False),
    ],
)
def test_auxiliary_projection_sharding_requires_shared_token_rows(
    monkeypatch,
    use_sequence_parallel: bool,
    tp_size: int,
    expected: bool,
):
    monkeypatch.setattr(
        kimi_model, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    assert kimi_model.shard_auxiliary_projections(use_sequence_parallel) is expected


@pytest.mark.parametrize(
    ("quantization", "moe_backend", "b12x_env", "expected"),
    [
        ("mxfp4", "b12x", False, True),
        ("mxfp4", "auto", True, True),
        ("mxfp4", "auto", False, False),
        ("mxfp4", "flashinfer_cutlass", True, False),
        (None, "b12x", True, False),
    ],
)
def test_native_mxfp4_moe_shard_requires_b12x_backend(
    monkeypatch: pytest.MonkeyPatch,
    quantization: str | None,
    moe_backend: str,
    b12x_env: bool,
    expected: bool,
):
    monkeypatch.setattr(kimi_model.envs, "VLLM_USE_B12X_MOE", b12x_env)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(quantization=quantization),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )

    assert kimi_model._uses_native_b12x_mxfp4_intermediate_size(vllm_config) is expected


def test_partial_routed_output_transform_adds_partial_residual(monkeypatch):
    monkeypatch.delenv("VLLM_KQUANT_CAPTURE_DIR", raising=False)
    weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    projection = _PartialRowProjection(weight)
    transform = kimi_model.KimiRoutedOutputTransform(None, projection, layer_idx=0)
    hidden_states = torch.arange(6, dtype=torch.float32).view(2, 3)
    residual = torch.full((2, 4), 2.0)

    actual = transform(hidden_states, residual=residual)
    expected = torch.nn.functional.linear(hidden_states, weight) + residual

    assert transform.output_is_tp_partial
    torch.testing.assert_close(actual, expected)


def test_kimi_column_parallel_loader_zero_fills_tail():
    layer = object.__new__(kimi_model.KimiPaddedColumnParallelLinear)
    layer.tp_rank = 1
    param = nn.Parameter(torch.empty(3, 4), requires_grad=False)
    param.output_dim = 0
    loaded_weight = torch.arange(20, dtype=torch.float32).view(5, 4)

    layer.weight_loader(param, loaded_weight)

    expected = torch.cat([loaded_weight[3:], torch.zeros(1, 4)])
    torch.testing.assert_close(param, expected)


def test_kimi_padded_column_gathers_and_removes_logical_tail(monkeypatch):
    layer = object.__new__(kimi_model.KimiPaddedColumnParallelLinear)
    nn.Module.__init__(layer)
    layer.kimi_gather_output = True
    layer.logical_output_size = 5
    local_output = torch.arange(3, dtype=torch.float32).view(1, 3)
    gathered_output = torch.arange(6, dtype=torch.float32).view(1, 6)
    monkeypatch.setattr(
        kimi_model.ColumnParallelLinear,
        "forward",
        lambda _self, _x: (local_output, None),
    )
    monkeypatch.setattr(
        kimi_model,
        "gather_kimi_sharded_projection",
        lambda output: gathered_output,
    )

    output, bias = layer(torch.empty(1, 1))

    torch.testing.assert_close(output, gathered_output[:, :5])
    assert output.is_contiguous()
    assert bias is None


def test_kimi_gate_local_projection_preserves_linear_return_contract():
    layer = object.__new__(kimi_model.KimiColumnParallelGate)
    nn.Module.__init__(layer)
    layer.weight = nn.Parameter(torch.arange(12).view(3, 4).float())
    hidden_states = torch.arange(8).view(2, 4).float()

    output, bias = layer.forward_local(hidden_states)

    torch.testing.assert_close(
        output,
        torch.nn.functional.linear(hidden_states, layer.weight),
    )
    assert output.dtype == torch.float32
    assert bias is None


def test_kimi_gate_gathers_fp32_local_projection(monkeypatch):
    layer = object.__new__(kimi_model.KimiColumnParallelGate)
    nn.Module.__init__(layer)
    layer.logical_output_size = 5
    layer.weight = nn.Parameter(torch.arange(12).view(3, 4).float())
    hidden_states = torch.arange(8).view(2, 4).float()
    gathered = torch.arange(12).view(2, 6).float()
    received: dict[str, torch.Tensor] = {}

    def gather(output: torch.Tensor) -> torch.Tensor:
        received["local"] = output
        return gathered

    monkeypatch.setattr(kimi_model, "gather_kimi_sharded_projection", gather)

    output, bias = layer(hidden_states)

    torch.testing.assert_close(
        received["local"],
        torch.nn.functional.linear(hidden_states, layer.weight),
    )
    torch.testing.assert_close(output, gathered[:, :5])
    assert bias is None


def test_kimi_router_decodes_precomputed_topk_payload_without_copy():
    router = kimi_model.KimiK3PrecomputedTopKRouter(
        top_k=16,
        global_num_experts=896,
        e_score_correction_bias=nn.Parameter(torch.zeros(896)),
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    num_tokens = 3
    hidden_states = torch.empty(num_tokens, 4)
    payload = torch.empty(num_tokens * 2, 16, dtype=torch.float32)
    expected_weights = torch.arange(num_tokens * 16, dtype=torch.float32).view(
        num_tokens, 16
    )
    expected_ids = torch.arange(num_tokens * 16, dtype=torch.int32).view(num_tokens, 16)
    payload[:num_tokens].copy_(expected_weights)
    payload[num_tokens:].view(torch.int32).copy_(expected_ids)

    weights, ids = router._compute_routing(hidden_states, payload, torch.int32)

    assert weights.data_ptr() == payload.data_ptr()
    assert ids.data_ptr() == payload[num_tokens:].data_ptr()
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids, expected_ids)


def test_kimi_router_marks_precomputed_padding_routes_inactive(monkeypatch):
    router = kimi_model.KimiK3PrecomputedTopKRouter(
        top_k=16,
        global_num_experts=896,
        e_score_correction_bias=nn.Parameter(torch.zeros(896)),
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    num_tokens = 3
    hidden_states = torch.empty(num_tokens, 4)
    payload = torch.empty(num_tokens * 2, 16, dtype=torch.float32)
    expected_weights = torch.arange(num_tokens * 16, dtype=torch.float32).view(
        num_tokens, 16
    )
    expected_ids = torch.arange(num_tokens * 16, dtype=torch.int32).view(num_tokens, 16)
    payload[:num_tokens].copy_(expected_weights)
    payload[num_tokens:].view(torch.int32).copy_(expected_ids)
    monkeypatch.setattr(kimi_model.envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(kimi_model, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        kimi_model,
        "get_forward_context",
        lambda: SimpleNamespace(is_padding=torch.tensor([False, True, False])),
    )

    weights, ids = router._compute_routing(hidden_states, payload, torch.int32)

    assert weights.data_ptr() == payload.data_ptr()
    assert ids.data_ptr() == payload[num_tokens:].data_ptr()
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(ids[0], expected_ids[0])
    torch.testing.assert_close(ids[1], torch.full((16,), -1, dtype=torch.int32))
    torch.testing.assert_close(ids[2], expected_ids[2])


def test_kimi_router_selects_experts_with_b12x_for_full_logits(monkeypatch):
    correction_bias = nn.Parameter(torch.zeros(896))
    router = kimi_model.KimiK3PrecomputedTopKRouter(
        top_k=16,
        global_num_experts=896,
        e_score_correction_bias=correction_bias,
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    hidden_states = torch.empty(8, 4)
    router_logits = torch.empty(8, 896, dtype=torch.float32)
    expected = (torch.empty(8, 16), torch.empty(8, 16, dtype=torch.int32))
    received: dict[str, torch.Tensor] = {}

    def select_experts(logits: torch.Tensor, bias: torch.Tensor):
        received.update(logits=logits, bias=bias)
        return expected

    monkeypatch.setattr(
        kimi_model,
        "try_select_kimi_routed_experts",
        select_experts,
    )

    actual = router._compute_routing(
        hidden_states,
        router_logits,
        torch.int32,
    )

    assert actual is expected
    assert received.pop("logits") is router_logits
    assert received.pop("bias").data_ptr() == correction_bias.data.data_ptr()
    assert received == {}


def test_kimi_router_uses_standard_routing_for_non_payload(monkeypatch):
    router = kimi_model.KimiK3PrecomputedTopKRouter(
        top_k=16,
        global_num_experts=896,
        e_score_correction_bias=nn.Parameter(torch.zeros(896)),
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    hidden_states = torch.empty(2, 4)
    router_logits = torch.empty(2, 896)
    input_ids = torch.tensor([1, 2])
    expected = (torch.empty(2, 16), torch.empty(2, 16, dtype=torch.int32))
    received: dict[str, object] = {}

    def standard_routing(
        _self,
        hidden_states_arg,
        router_logits_arg,
        indices_type_arg,
        *,
        input_ids=None,
    ):
        received.update(
            hidden_states=hidden_states_arg,
            router_logits=router_logits_arg,
            indices_type=indices_type_arg,
            input_ids=input_ids,
        )
        return expected

    monkeypatch.setattr(
        kimi_model.FusedTopKBiasRouter,
        "_compute_routing",
        standard_routing,
    )

    actual = router._compute_routing(
        hidden_states,
        router_logits,
        torch.int32,
        input_ids=input_ids,
    )

    assert actual is expected
    assert received["hidden_states"] is hidden_states
    assert received["router_logits"] is router_logits
    assert received["indices_type"] is torch.int32
    assert received["input_ids"] is input_ids


def _make_paired_projection_moe():
    moe = object.__new__(kimi_model.KimiMoE)
    nn.Module.__init__(moe)
    moe.use_mega_moe = False
    moe.gate = object.__new__(kimi_model.KimiColumnParallelGate)
    nn.Module.__init__(moe.gate)
    moe.gate.logical_output_size = 5
    moe.gate.e_score_correction_bias = nn.Parameter(torch.zeros(5))
    moe.routed_expert_down_proj = object.__new__(
        kimi_model.KimiPaddedColumnParallelLinear
    )
    nn.Module.__init__(moe.routed_expert_down_proj)
    moe.routed_expert_down_proj.logical_output_size = 3
    moe._down_proj_events = (None, None)
    moe._down_proj_stream = None
    return moe


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_kimi_moe_uses_precomputed_projection_routing_payload(
    monkeypatch, num_tokens: int
):
    moe = _make_paired_projection_moe()
    hidden_states = torch.empty(num_tokens, 4)
    local_router = torch.empty(num_tokens, 3)
    local_down = torch.empty(num_tokens, 2)
    gathered_down = torch.empty(num_tokens, 3)
    routing_payload = torch.empty(num_tokens * 2, 16)
    monkeypatch.setattr(
        kimi_model.KimiColumnParallelGate,
        "forward_local",
        lambda _self, _hidden: (local_router, None),
    )
    monkeypatch.setattr(
        kimi_model.KimiPaddedColumnParallelLinear,
        "forward_local",
        lambda _self, _hidden: (local_down, None),
    )
    monkeypatch.setattr(
        kimi_model,
        "maybe_execute_in_parallel",
        lambda first, second, *_args: (first(), second()),
    )
    monkeypatch.setattr(
        kimi_model,
        "try_gather_kimi_sharded_projection_pair_topk",
        lambda down, router, bias: (gathered_down, routing_payload),
    )
    monkeypatch.setattr(
        kimi_model,
        "gather_kimi_sharded_projection_pair",
        lambda *_args: pytest.fail(
            "precomputed routing must skip the model-level paired gather"
        ),
    )

    routed_hidden, router_output, topk_ids = moe._maybe_overlap_router_and_down_proj(
        hidden_states
    )

    assert routed_hidden is gathered_down
    assert router_output is routing_payload
    assert topk_ids is None


def test_kimi_moe_paired_projection_uses_exact_router_fallback(monkeypatch):
    moe = _make_paired_projection_moe()
    hidden_states = torch.empty(2, 4)
    local_router = torch.empty(2, 3)
    local_down = torch.empty(2, 2)
    gathered_router = torch.arange(12, dtype=torch.float32).view(2, 6)
    gathered_down = torch.arange(8, dtype=torch.float32).view(2, 4)
    monkeypatch.setattr(
        kimi_model.KimiColumnParallelGate,
        "forward_local",
        lambda _self, _hidden: (local_router, None),
    )
    monkeypatch.setattr(
        kimi_model.KimiPaddedColumnParallelLinear,
        "forward_local",
        lambda _self, _hidden: (local_down, None),
    )
    monkeypatch.setattr(
        kimi_model,
        "maybe_execute_in_parallel",
        lambda first, second, *_args: (first(), second()),
    )
    monkeypatch.setattr(
        kimi_model,
        "try_gather_kimi_sharded_projection_pair_topk",
        lambda *_args: None,
    )
    received: dict[str, int | None] = {}

    def gather_pair(down, router, first_columns=None, second_columns=None):
        # The paired gather returns the logical widths the caller asks for.
        received.update(first_columns=first_columns, second_columns=second_columns)
        return (
            gathered_down[:, :first_columns].contiguous(),
            gathered_router[:, :second_columns].contiguous(),
        )

    monkeypatch.setattr(kimi_model, "gather_kimi_sharded_projection_pair", gather_pair)

    routed_hidden, router_output, topk_ids = moe._maybe_overlap_router_and_down_proj(
        hidden_states
    )

    assert received == {"first_columns": 3, "second_columns": 5}

    torch.testing.assert_close(routed_hidden, gathered_down[:, :3])
    torch.testing.assert_close(router_output, gathered_router[:, :5])
    assert routed_hidden.is_contiguous()
    assert router_output.is_contiguous()
    assert topk_ids is None


def test_kimi_merged_projection_restores_logical_output_order(monkeypatch):
    layer = object.__new__(kimi_mla.KimiShardedMergedColumnParallelLinear)
    nn.Module.__init__(layer)
    layer.tp_size = 4
    layer.output_sizes = [8, 4]
    local_output = torch.empty(1, 3)
    rank_major_output = torch.tensor(
        [[0, 1, 8, 2, 3, 9, 4, 5, 10, 6, 7, 11]],
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        kimi_mla.MergedColumnParallelLinear,
        "forward",
        lambda _self, _x: (local_output, None),
    )
    monkeypatch.setattr(
        kimi_mla,
        "gather_kimi_sharded_projection",
        lambda output: rank_major_output,
    )

    output, bias = layer(torch.empty(1, 1))

    torch.testing.assert_close(output, torch.arange(12).view(1, 12).float())
    assert bias is None


@pytest.mark.parametrize(
    ("output_sizes", "tp_size", "width", "message"),
    [
        ([7, 4], 4, 11, "must be divisible"),
        ([8, 4], 4, 11, "Unexpected gathered"),
    ],
)
def test_kimi_merged_projection_rejects_invalid_shard_geometry(
    output_sizes: list[int],
    tp_size: int,
    width: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        kimi_mla._restore_merged_output_order(
            torch.empty(1, width),
            output_sizes,
            tp_size,
        )


def test_kimi_row_parallel_loader_zero_fills_tail():
    layer = object.__new__(kimi_model.KimiPaddedRowParallelLinear)
    layer.tp_rank = 1
    param = nn.Parameter(torch.empty(4, 3), requires_grad=False)
    param.input_dim = 1
    loaded_weight = torch.arange(20, dtype=torch.float32).view(4, 5)

    layer.weight_loader(param, loaded_weight)

    expected = torch.cat([loaded_weight[:, 3:], torch.zeros(4, 1)], dim=1)
    torch.testing.assert_close(param, expected)


def test_sharded_sequence_parallel_mlp_matches_replicated(default_vllm_config):
    """Sharded SP MLP must reproduce the replicated result for every token.

    Each rank owns a *disjoint* token shard, so a weight shard alone cannot
    finish a rank's own tokens: the ranks must gather the full token set,
    compute partial sums over their intermediate shard, and reduce-scatter.
    Splicing per-rank feature slices together instead silently mixes different
    tokens and produces plausible-looking garbage.
    """
    tp_size, hidden, intermediate, tokens_per_rank = 4, 16, 12, 3
    torch.manual_seed(0)
    num_tokens = tp_size * tokens_per_rank
    x = torch.randn(num_tokens, hidden)
    gate_weight = torch.randn(intermediate, hidden)
    up_weight = torch.randn(intermediate, hidden)
    down_weight = torch.randn(hidden, intermediate)
    act_fn = SiluAndMul()

    replicated = act_fn(x @ torch.cat([gate_weight, up_weight]).T) @ down_weight.T

    shard = intermediate // tp_size
    # Every rank all-gathers the full token set, then computes its partial.
    partials = [
        act_fn(
            x
            @ torch.cat(
                [
                    gate_weight[r * shard : (r + 1) * shard],
                    up_weight[r * shard : (r + 1) * shard],
                ]
            ).T
        )
        @ down_weight[:, r * shard : (r + 1) * shard].T
        for r in range(tp_size)
    ]
    reduced = torch.stack(partials).sum(0)
    # Reduce-scatter: rank r keeps only its own token shard.
    for r in range(tp_size):
        mine = reduced[r * tokens_per_rank : (r + 1) * tokens_per_rank]
        expected = replicated[r * tokens_per_rank : (r + 1) * tokens_per_rank]
        torch.testing.assert_close(mine, expected, atol=1e-5, rtol=1e-5)


def test_sp_all_gather_uses_custom_kernel(monkeypatch):
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    expected = torch.cat([hidden_states, hidden_states])
    custom_all_gather = Mock(return_value=expected)
    device_communicator = SimpleNamespace(
        custom_all_gather=custom_all_gather,
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_communicator),
    )
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", fallback)

    output = sp_ops.sp_all_gather(hidden_states)

    torch.testing.assert_close(output, expected)
    custom_all_gather.assert_called_once_with(hidden_states)
    fallback.assert_not_called()


def test_sp_reduce_scatter_uses_custom_kernel_after_padding(monkeypatch):
    hidden_states = torch.arange(6, dtype=torch.float32).view(3, 2)
    expected = torch.arange(4, dtype=torch.float32).view(2, 2)
    custom_reduce_scatter = Mock(return_value=expected)
    device_communicator = SimpleNamespace(
        custom_reduce_scatter=custom_reduce_scatter,
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_communicator),
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_reduce_scatter", fallback)

    output = sp_ops.sp_reduce_scatter(hidden_states)

    torch.testing.assert_close(output, expected)
    padded = custom_reduce_scatter.call_args.args[0]
    assert padded.shape == (4, 2)
    torch.testing.assert_close(padded[:3], hidden_states)
    torch.testing.assert_close(padded[3], torch.zeros(2))
    fallback.assert_not_called()


@pytest.mark.parametrize("shape", [(3,), (3, 2, 2)])
def test_sp_shard_pads_only_the_token_axis(monkeypatch, shape):
    hidden_states = torch.arange(math.prod(shape), dtype=torch.float32).view(shape)
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_rank", lambda: 1)

    output = sp_ops.sp_shard(hidden_states)

    padding = hidden_states.new_zeros((1, *shape[1:]))
    expected = torch.cat([hidden_states, padding])[2:]
    torch.testing.assert_close(output, expected)


def test_sp_collectives_fall_back_without_custom_kernel(monkeypatch):
    hidden_states = torch.arange(4, dtype=torch.float32).view(2, 2)
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=None),
    )
    monkeypatch.setattr(
        sp_ops,
        "get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    all_gather = Mock(return_value=hidden_states)
    reduce_scatter = Mock(return_value=hidden_states)
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", all_gather)
    monkeypatch.setattr(
        sp_ops,
        "tensor_model_parallel_reduce_scatter",
        reduce_scatter,
    )

    torch.testing.assert_close(sp_ops.sp_all_gather(hidden_states), hidden_states)
    torch.testing.assert_close(sp_ops.sp_reduce_scatter(hidden_states), hidden_states)
    all_gather.assert_called_once_with(hidden_states, 0)
    reduce_scatter.assert_called_once_with(hidden_states, 0)
