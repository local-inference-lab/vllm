# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner_module
from vllm.config import ParallelConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.models.common.ops import sequence_parallel as sp_ops
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


class _PartialOutputTransform(nn.Module):
    output_is_tp_partial = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


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
    ("enabled", "use_sequence_parallel", "eligible", "tp_size", "expected"),
    [
        (True, True, True, 8, True),
        (False, True, True, 8, False),  # opt-in only
        (True, False, True, 8, False),  # replication only exists under SP
        (True, True, False, 8, False),  # FusedMoE path owns the reduction
        (True, True, True, 1, False),  # nothing to shard
        (True, True, True, 5, False),  # 6144 % 5 -- would fail divide()
    ],
)
def test_shard_sequence_parallel_mlp_gating(
    monkeypatch,
    enabled: bool,
    use_sequence_parallel: bool,
    eligible: bool,
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
            eligible=eligible,
        )
        is expected
    )


@pytest.mark.parametrize(
    (
        "use_sequence_parallel",
        "tp_size",
        "num_experts",
        "routed_hidden_size",
        "expected",
    ),
    [
        (False, 16, 896, 3584, True),
        (False, 8, 896, 3584, True),
        (False, 12, 896, 3584, False),
        (True, 16, 896, 3584, False),
        (False, 1, 896, 3584, False),
    ],
)
def test_auxiliary_projection_sharding_requires_shared_rows_and_even_widths(
    monkeypatch,
    use_sequence_parallel: bool,
    tp_size: int,
    num_experts: int,
    routed_hidden_size: int,
    expected: bool,
):
    monkeypatch.setattr(
        kimi_model, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    assert (
        kimi_model.shard_auxiliary_projections(
            use_sequence_parallel=use_sequence_parallel,
            num_experts=num_experts,
            routed_hidden_size=routed_hidden_size,
        )
        is expected
    )


def test_column_parallel_gate_preserves_rank_major_fp32_logits(monkeypatch):
    gate = object.__new__(kimi_model.KimiColumnParallelGate)
    nn.Module.__init__(gate)
    gate.weight = nn.Parameter(
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    gate.tp_size = 2
    hidden_states = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32)
    local_logits = torch.nn.functional.linear(hidden_states, gate.weight)
    gathered_logits = torch.cat((local_logits, local_logits + 100.0), dim=-1)
    gather = Mock(return_value=gathered_logits)
    monkeypatch.setattr(kimi_model, "tensor_model_parallel_all_gather", gather)

    actual, bias = gate(hidden_states)

    assert bias is None
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, gathered_logits)
    gather.assert_called_once()
    torch.testing.assert_close(gather.call_args.args[0], local_logits)


def test_partial_routed_output_transform_adds_partial_residual():
    weight = torch.arange(12, dtype=torch.float32).view(4, 3)
    projection = _PartialRowProjection(weight)
    transform = kimi_model.KimiRoutedOutputTransform(None, projection)
    hidden_states = torch.arange(6, dtype=torch.float32).view(2, 3)
    residual = torch.full((2, 4), 2.0)

    actual = transform(hidden_states, residual=residual)
    expected = torch.nn.functional.linear(hidden_states, weight) + residual

    assert transform.output_is_tp_partial
    torch.testing.assert_close(actual, expected)


def test_tp_partial_output_transform_defers_shared_reduce(monkeypatch):
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_output_transform = _PartialOutputTransform()
    runner.routed_input_transform = None
    runner.routed_scaling_factor = 1.0
    runner.router = None
    runner.layer_name = "test"
    runner.moe_config = SimpleNamespace(
        hidden_dim_unpadded=4,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        tp_size=2,
        ep_size=1,
    )
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            has_unpadded_output=False,
            moe_kernel=SimpleNamespace(output_is_reduced=lambda: True),
        )
    )
    runner._maybe_pad_hidden_states = MethodType(
        lambda self, shared, routed: (routed, None, None),
        runner,
    )

    shared_output = torch.full((2, 4), 2.0)
    fused_output = torch.full((2, 4), 3.0)
    runner._forward_entry = Mock(return_value=(shared_output, fused_output))
    reduced_inputs = []

    def all_reduce(hidden_states: torch.Tensor) -> torch.Tensor:
        reduced_inputs.append(hidden_states.clone())
        return hidden_states * 2

    monkeypatch.setattr(
        moe_runner_module,
        "tensor_model_parallel_all_reduce",
        all_reduce,
    )

    hidden_states = torch.zeros_like(shared_output)
    actual = runner.forward(
        hidden_states,
        router_logits=torch.empty(2, 1),
        shared_experts_input=hidden_states,
    )

    assert len(reduced_inputs) == 1
    torch.testing.assert_close(reduced_inputs[0], shared_output + fused_output)
    torch.testing.assert_close(actual, (shared_output + fused_output) * 2)


@pytest.mark.parametrize(
    ("quantization", "moe_backend", "expected"),
    [
        ("mxfp4", "b12x", True),
        ("mxfp4", "auto", False),
        ("mxfp4", "flashinfer_cutlass", False),
        (None, "b12x", False),
    ],
)
def test_native_mxfp4_moe_shard_requires_b12x_backend(
    quantization: str | None,
    moe_backend: str,
    expected: bool,
):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(quantization=quantization),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
    )

    assert kimi_model._uses_native_b12x_mxfp4_intermediate_size(vllm_config) is expected


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
