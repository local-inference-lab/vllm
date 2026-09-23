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
from vllm.models.common.ops import sequence_parallel as sp_ops
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import mtp as kimi_mtp
from vllm.platforms import current_platform


@torch.inference_mode()
@pytest.mark.parametrize("rows", [1, 1024, 4096])
def test_prefill_projection_donation_preserves_storage_and_decode(rows, monkeypatch):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.models.kimi_k3.nvidia import tp_projection

    layer = SimpleNamespace(
        quant_method=UnquantizedLinearMethod(),
        weight=torch.randn(16, 8),
        input_is_parallel=True,
        bias=None,
        output_size=16,
        input_size_per_partition=8,
        reduce_results=True,
        tp_size=2,
    )
    activation = torch.randn(rows, 8)
    output = torch.empty(rows, 16)
    eligible = tp_projection.can_reuse_projection_output(layer, output)
    assert eligible == (rows >= 1024)
    if not eligible:
        return
    expected = torch.mm(activation, layer.weight.T) * 2
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_all_reduce_in_place",
        lambda value: value.mul_(2),
    )
    actual = tp_projection.project_into_consumed_output(layer, activation, output)
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="disjoint"):
        tp_projection.project_into_consumed_output(layer, output[:, :8], output)


def test_in_place_allreduce_donates_pynccl_output():
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from vllm.distributed.parallel_state import GroupCoordinator

    group = GroupCoordinator.__new__(GroupCoordinator)
    group.world_size = 2
    comm = CudaCommunicator.__new__(CudaCommunicator)
    comm.pynccl_comm = Mock(disabled=False)
    group.device_communicator = comm
    values = torch.arange(16, dtype=torch.float32)

    def reduce(inp, *, out_tensor):
        assert inp is values and out_tensor is inp
        return out_tensor.mul_(2)

    comm.pynccl_comm.all_reduce.side_effect = reduce
    assert group.all_reduce_in_place(values) is values
    torch.testing.assert_close(values, torch.arange(16, dtype=torch.float32) * 2)


@pytest.mark.parametrize("padding", [False, True])
def test_precomputed_routing_preserves_ids_weights_capture_and_padding(
    monkeypatch, padding
):
    router = kimi_model.KimiPrecomputedTopKRouter(
        top_k=16,
        global_num_experts=896,
        e_score_correction_bias=torch.zeros(896),
    )
    payload = torch.empty(6, 16)
    weights = torch.rand(3, 16)
    weights /= weights.sum(-1, keepdim=True)
    ids = torch.arange(48, dtype=torch.int32).view(3, 16)
    payload[:3].copy_(weights)
    payload[3:].view(torch.int32).copy_(ids)
    mask = torch.tensor([False, padding, False])
    monkeypatch.setattr(kimi_model.envs, "VLLM_MOE_SKIP_PADDING", True)
    monkeypatch.setattr(kimi_model, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        kimi_model, "get_forward_context", lambda: SimpleNamespace(is_padding=mask)
    )
    capture = Mock()
    router.capture_fn = capture
    actual_weights, actual_ids = router._select_experts(
        torch.empty(3, 4), payload, torch.int32
    )
    expected_ids = ids.masked_fill(mask[:, None], -1)
    torch.testing.assert_close(actual_weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    assert actual_weights.data_ptr() == payload.data_ptr()
    assert actual_ids.data_ptr() == payload[3:].data_ptr()
    capture.assert_called_once_with(actual_ids)


def test_precomputed_router_preserves_full_logit_prefill(monkeypatch):
    router = kimi_model.KimiPrecomputedTopKRouter(
        top_k=16,
        global_num_experts=896,
        e_score_correction_bias=torch.zeros(896),
    )
    expected = (torch.ones(9, 16), torch.zeros(9, 16, dtype=torch.int32))
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(kimi_model.FusedTopKBiasRouter, "_compute_routing", fallback)
    hidden, logits = torch.empty(9, 4), torch.zeros(9, 896)
    actual = router._compute_routing(hidden, logits, torch.int32)
    assert actual is expected
    fallback.assert_called_once_with(hidden, logits, torch.int32, input_ids=None)


@torch.inference_mode()
@pytest.mark.parametrize("rows", [1024, 4096])
def test_routed_prefill_donation_matches_allocating_bf16_path(
    rows,
    monkeypatch,
    default_vllm_config,
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.kimi_k3.nvidia import tp_projection

    for module in (linear, parameter, tp_projection):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 16)
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 15)
    torch.manual_seed(141)
    projection = tp_projection.KimiPaddedRowParallelLinear(3584, 7168, "test.up")
    projection = projection.to(device="cuda", dtype=torch.bfloat16)
    projection.weight.copy_(torch.randn_like(projection.weight) * 0.02)
    projection.quant_method.process_weights_after_loading(projection)
    transform = kimi_model.KimiRoutedOutputTransform(
        RMSNorm(3584).to(device="cuda", dtype=torch.bfloat16),
        projection,
    )
    latent = torch.randn(rows, 3584, device="cuda", dtype=torch.bfloat16)
    expected = transform(latent.clone())
    output = torch.empty(rows, 7168, device="cuda", dtype=torch.bfloat16)
    actual = transform(latent.clone(), output=output)
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [2, 8, 12, 16])
@pytest.mark.parametrize("shard_latents", [False, True])
@pytest.mark.parametrize("prepared_transport", [False, True])
def test_merged_mla_projection_preserves_latent_and_local_gate_order(
    tp_size, shard_latents, prepared_transport, monkeypatch, default_vllm_config
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp_size)
    rank = tp_size - 1
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(
        parameter, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    torch.manual_seed(811)
    widths = [1536, 576, 96 * 128]
    weights = [torch.randn(width, 32) for width in widths]
    x = torch.randn(3, 32)
    expected = [torch.nn.functional.linear(x, weight) for weight in weights]
    layer = linear.KimiK3MergedQKVGateLinear(
        32, 1536, 512, 64, 96, 128, shard_latents=shard_latents
    )
    for shard, weight in enumerate(weights):
        layer.weight.weight_loader(layer.weight, weight, shard)
    layer.quant_method.process_weights_after_loading(layer)

    def gather(local, dim):
        assert dim == -1
        pieces = [value.chunk(tp_size, dim=-1) for value in expected[:2]]
        torch.testing.assert_close(local, torch.cat([p[rank] for p in pieces], -1))
        return torch.cat(
            [torch.cat([p[r] for p in pieces], -1) for r in range(tp_size)], -1
        )

    monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", gather)
    if prepared_transport and shard_latents:
        local_width = sum(widths[:2]) // tp_size
        padded_width = (local_width + 7) // 8 * 8

        def gather_heads(query):
            assert query.shape == (x.shape[0], 1, padded_width)
            assert not query[..., local_width:].count_nonzero()
            result = gather(query[:, 0, :local_width], -1).unflatten(
                -1, (tp_size, local_width)
            )
            return torch.nn.functional.pad(result, (0, padded_width - local_width))

        layer._latent_transport = SimpleNamespace(
            query_dim=padded_width, gather=gather_heads
        )
    actual, bias = layer(x)
    assert bias is None
    torch.testing.assert_close(
        actual,
        torch.cat([*expected[:2], expected[2].chunk(tp_size, dim=-1)[rank]], -1),
    )
    expected_rows = (
        sum(widths) // tp_size
        if shard_latents
        else sum(widths[:2]) + widths[2] // tp_size
    )
    assert layer.weight.shape == (expected_rows, 32)


@pytest.mark.parametrize("tp_size", [8, 9, 10, 12, 16])
@pytest.mark.parametrize("projection", ["gate", "down", "up"])
def test_padded_auxiliary_projections_reconstruct_checkpoint(
    tp_size, projection, monkeypatch, default_vllm_config
):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.kimi_k3.nvidia import tp_projection

    for module in (linear, parameter, tp_projection):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: tp_size
        )
    torch.manual_seed(243)
    input_size, output_size = (
        (3584, 32)
        if projection == "up"
        else (32, 896 if projection == "gate" else 3584)
    )
    weight = torch.randn(output_size, input_size)
    x = torch.randn(3, input_size)
    constructor = {
        "gate": tp_projection.KimiColumnParallelGate,
        "down": tp_projection.KimiPaddedColumnParallelLinear,
        "up": tp_projection.KimiPaddedRowParallelLinear,
    }[projection]
    parts = []
    layers = []
    for rank in range(tp_size):
        for module in (linear, parameter):
            monkeypatch.setattr(
                module, "get_tensor_model_parallel_rank", lambda rank=rank: rank
            )
        layer = constructor(input_size, output_size, prefix="test.projection")
        layer.weight.weight_loader(layer.weight, weight)
        axis = 1 if projection == "up" else 0
        width = layer.weight.shape[axis]
        available = max(0, min(weight.shape[axis] - rank * width, width))
        if available < width:
            assert (
                torch.count_nonzero(
                    layer.weight.narrow(axis, available, width - available)
                )
                == 0
            )
        layer.quant_method.process_weights_after_loading(layer)
        result, bias = layer(x) if projection == "up" else layer.forward_local(x)
        assert bias is None and result.dtype == torch.float32
        parts.append(result)
        if projection == "up":
            with torch.inference_mode():
                donated = torch.empty_like(result)
                actual, _ = layer.forward_into(x, donated)
                assert actual is donated
                torch.testing.assert_close(actual, result, rtol=0, atol=0)
        layers.append(layer)
    expected = torch.nn.functional.linear(x, weight)
    if projection == "up":
        actual = torch.stack(parts).sum(0)
        assert all(layer.reduce_results is False for layer in layers)
    else:
        gathered = torch.cat(parts, -1)
        monkeypatch.setattr(
            tp_projection,
            "tensor_model_parallel_all_gather",
            lambda tensor, dim: gathered,
        )
        actual, _ = layers[-1](x)
        assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@torch.inference_mode()
@pytest.mark.parametrize("tp_size", [9, 10, 12])
@pytest.mark.parametrize("last_rank", [False, True])
def test_aligned_auxiliary_decode_preserves_weights_and_fp64_accuracy(
    tp_size, last_rank, monkeypatch, default_vllm_config
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.models.kimi_k3.nvidia import tp_projection as tp

    rank = tp_size - 1 if last_rank else 0
    for module in (linear, parameter, tp):
        monkeypatch.setattr(
            module, "get_tensor_model_parallel_world_size", lambda: tp_size
        )
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    torch.manual_seed(189 + rank)
    down = tp.KimiPaddedColumnParallelLinear(7168, 3584, "test.down")
    gate = tp.KimiColumnParallelGate(7168, 896, "test.gate")
    up = tp.KimiPaddedRowParallelLinear(3584, 7168, "test.up")
    for layer in (down, gate, up):
        layer.to(device="cuda", dtype=torch.bfloat16)
        layer.weight.normal_(std=0.02)
    saved = [layer.weight.clone() for layer in (down, gate, up)]
    control_x = torch.randn(9, 7168, device="cuda", dtype=torch.bfloat16)
    control_up = torch.randn(9, 3584, device="cuda", dtype=torch.bfloat16)
    control_d, _ = down.forward_local(control_x)
    control_g, _ = gate.forward_local(control_x)
    control_u, _ = up(control_up)
    paired = tp.prepare_paired_decode_projection(down, gate)
    tp.prepare_aligned_decode_projection(up)
    for layer, original in zip((down, gate, up), saved):
        assert torch.equal(layer.weight, original)
        assert layer.weight.is_contiguous()
    assert (
        down.weight.untyped_storage().data_ptr() == paired.untyped_storage().data_ptr()
    )
    assert (
        gate.weight.untyped_storage().data_ptr() == paired.untyped_storage().data_ptr()
    )
    assert (
        torch.count_nonzero(paired[down.weight.shape[0] + gate.weight.shape[0] :]) == 0
    )
    assert torch.equal(down.forward_local(control_x)[0], control_d)
    assert torch.equal(gate.forward_local(control_x)[0], control_g)
    assert torch.equal(up(control_up)[0], control_u)

    for rows in (3, 4, 6, 8):
        x = control_x[:rows].clone()
        latent = control_up[:rows].clone()

        def call(x=x, latent=latent):
            d, g = tp.paired_decode_projection(
                x, paired, saved[0].shape[0], saved[1].shape[0]
            )
            u, _ = up(latent)
            return d, g, u

        call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = call()
        for mutation in range(3):
            x.normal_()
            latent.normal_()
            local = torch.nn.functional.pad(latent, (0, up.input_pad))[
                :,
                rank * up.input_size_per_partition : (rank + 1)
                * up.input_size_per_partition,
            ]
            oracles = [
                x.double() @ saved[0].double().T,
                x.double() @ saved[1].double().T,
                local.double() @ saved[2].double().T,
            ]
            for output in outputs:
                output.fill_(float("nan"))
            allocated = torch.accelerator.memory_allocated()
            graph.replay()
            torch.accelerator.synchronize()
            assert torch.accelerator.memory_allocated() == allocated
            repeats = [output.clone() for output in outputs]
            for output, oracle in zip(outputs, oracles):
                assert torch.isfinite(output).all() and torch.count_nonzero(output)
                relative = (output.double() - oracle).norm() / oracle.norm()
                assert relative < (0.0018 if output.dtype == torch.bfloat16 else 2e-6)
            for _ in range(5):
                graph.replay()
            torch.accelerator.synchronize()
            assert all(torch.equal(a, b) for a, b in zip(outputs, repeats))
        graph.reset()


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
    layer.reuse_attn_res_output = False
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
