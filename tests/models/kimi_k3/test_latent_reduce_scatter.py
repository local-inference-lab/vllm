# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Column reduce-scatter of the Kimi-K3 routed latent
(``VLLM_K3_LATENT_REDUCE=rs_fp32|rs_bf16``): the gate, the runner hook, the
column-block RMSNorm against an fp64 reference, and the up-projection's
consumption of its own input shard."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import tp_projection
from vllm.models.kimi_k3.nvidia.model import (
    KimiPaddedRowParallelLinear,
    KimiRoutedOutputTransform,
)

WORLD = 9
LATENT = 3584
SHARD = 400  # kimi_projection_shard_width(3584, 9)


@pytest.fixture
def latent_reduce(monkeypatch: pytest.MonkeyPatch):
    def _set(mode: str) -> None:
        monkeypatch.setenv("VLLM_K3_LATENT_REDUCE", mode)
        tp_projection.kimi_latent_reduce_mode.cache_clear()

    yield _set
    tp_projection.kimi_latent_reduce_mode.cache_clear()


def _rms_norm(width: int, weight: torch.Tensor) -> RMSNorm:
    norm = object.__new__(RMSNorm)
    nn.Module.__init__(norm)
    norm.hidden_size = width
    norm.variance_epsilon = 1e-5
    norm.variance_size_override = None
    norm.has_weight = True
    norm.pass_weight = True
    norm.pass_weight_add = True
    norm.weight = nn.Parameter(weight, requires_grad=False)
    return norm


def _projection(
    rank: int, tp_size: int, shard: int, logical: int, output_size: int
) -> KimiPaddedRowParallelLinear:
    projection = object.__new__(KimiPaddedRowParallelLinear)
    nn.Module.__init__(projection)
    projection.input_pad = shard * tp_size - logical
    projection.logical_input_size = logical
    projection.input_is_parallel = False
    projection.tp_size = tp_size
    projection.tp_rank = rank
    projection.output_size = output_size
    projection.reduce_results = False
    projection.quant_method = UnquantizedLinearMethod()
    projection.register_parameter("bias", None)
    weight = torch.randn(output_size, shard)
    start = rank * shard
    valid = max(0, min(shard, logical - start))
    weight[:, valid:] = 0  # the loader zero-fills the tail past the logical input
    projection.weight = nn.Parameter(weight, requires_grad=False)
    return projection


def test_latent_reduce_mode_gate(latent_reduce):
    latent_reduce("allreduce")
    assert tp_projection.kimi_latent_reduce_scatter_wire() is None
    latent_reduce("rs_fp32")
    assert tp_projection.kimi_latent_reduce_scatter_wire() == "fp32"
    latent_reduce("rs_bf16")
    assert tp_projection.kimi_latent_reduce_scatter_wire() == "bf16"
    latent_reduce("ring")
    with pytest.raises(ValueError, match="VLLM_K3_LATENT_REDUCE"):
        tp_projection.kimi_latent_reduce_mode()


def test_try_reduce_scatter_reaches_the_ring_only_when_enabled(
    monkeypatch, latent_reduce
):
    calls: list[dict] = []

    def fake_rs(partial, *, wire, cols):
        calls.append({"ptr": partial.data_ptr(), "wire": wire, "cols": cols})
        return partial[:, :cols]

    monkeypatch.setattr(
        tp_projection, "tensor_model_parallel_pcie_reduce_scatter_columns", fake_rs
    )
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: WORLD
    )
    partial = torch.zeros(1024, LATENT, dtype=torch.bfloat16)
    latent_reduce("allreduce")
    assert tp_projection.try_reduce_scatter_kimi_latent(partial, cols=SHARD) is None
    latent_reduce("rs_fp32")
    assert tp_projection.try_reduce_scatter_kimi_latent(partial[:8], cols=SHARD) is None
    block = tp_projection.try_reduce_scatter_kimi_latent(partial, cols=SHARD)
    assert block is not None and block.shape == (1024, SHARD)
    assert calls == [{"ptr": partial.data_ptr(), "wire": "fp32", "cols": SHARD}]
    latent_reduce("rs_bf16")
    tp_projection.try_reduce_scatter_kimi_latent(partial, cols=SHARD)
    assert calls[-1]["wire"] == "bf16"


def test_transform_reduce_scatter_hook_respects_gate_and_capture(
    monkeypatch, latent_reduce
):
    projection = _projection(0, WORLD, SHARD, LATENT, output_size=16)
    transform = KimiRoutedOutputTransform(None, projection, layer_idx=0)
    partial = torch.zeros(1024, LATENT, dtype=torch.bfloat16)
    seen: list[int] = []
    monkeypatch.setattr(
        kimi_model,
        "try_reduce_scatter_kimi_latent",
        lambda p, *, cols: seen.append(cols) or p[:, :cols],
    )
    latent_reduce("rs_fp32")
    block = transform.reduce_scatter_tp_partial(partial)
    assert block is not None and block.shape == (1024, SHARD)
    assert seen == [SHARD]
    monkeypatch.setenv("VLLM_KQUANT_CAPTURE_DIR", "/tmp/capture")
    assert transform.reduce_scatter_tp_partial(partial) is None
    monkeypatch.delenv("VLLM_KQUANT_CAPTURE_DIR")
    projection.reduce_results = True
    assert transform.reduce_scatter_tp_partial(partial) is None


def _ulps(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    exponent = torch.floor(torch.log2(reference.abs().float().clamp_min(2.0**-126)))
    return (actual.float() - reference.float()).abs() / torch.pow(2.0, exponent - 7)


def test_column_block_rms_norm_is_at_least_as_precise_as_the_full_width_kernel(
    monkeypatch,
):
    """Nine ranks normalize their 400-column blocks with fp64 partial sums of
    squares combined by an (emulated) fp64 all-reduce; the assembled result
    is compared with the fp64 RMSNorm of the same bf16 latent, against the
    served full-width arithmetic (fp32 variance, bf16(bf16(x*s)*w))."""
    torch.manual_seed(0)
    rows = 64
    latent = (torch.randn(rows, LATENT) * torch.logspace(-2, 2, LATENT)).to(
        torch.bfloat16
    )
    weight = (1.0 + 0.1 * torch.randn(LATENT)).to(torch.bfloat16)
    eps = 1e-5
    blocks = []
    for rank in range(WORLD):
        block = torch.zeros(rows, SHARD, dtype=torch.bfloat16)
        start = rank * SHARD
        valid = max(0, min(SHARD, LATENT - start))
        block[:, :valid] = latent[:, start : start + valid]
        blocks.append(block)
    total_sumsq = sum(b.double().square().sum(dim=-1, keepdim=True) for b in blocks)

    def fake_all_reduce(t: torch.Tensor) -> torch.Tensor:
        assert t.dtype == torch.float64 and t.shape == (rows, 1)
        return total_sumsq.clone()

    monkeypatch.setattr(kimi_model, "tensor_model_parallel_all_reduce", fake_all_reduce)
    normed = torch.empty(rows, LATENT, dtype=torch.bfloat16)
    for rank in range(WORLD):
        projection = _projection(rank, WORLD, SHARD, LATENT, output_size=8)
        transform = KimiRoutedOutputTransform(_rms_norm(LATENT, weight), projection, 0)
        out = transform.normalize_column_block(blocks[rank])
        assert out.shape == (rows, SHARD) and out.dtype == torch.bfloat16
        start = rank * SHARD
        valid = max(0, min(SHARD, LATENT - start))
        assert torch.equal(out[:, valid:], torch.zeros_like(out[:, valid:]))
        normed[:, start : start + valid] = out[:, :valid]

    x64 = latent.double()
    ref64 = (
        x64
        * torch.rsqrt(x64.square().mean(dim=-1, keepdim=True) + eps)
        * weight.double()
    )
    reference = ref64.to(torch.bfloat16)
    x32 = latent.float()
    served = (
        (x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps))
        .to(torch.bfloat16)
        .float()
        * weight.float()
    ).to(torch.bfloat16)

    block_mismatch = int((normed != reference).sum())
    served_mismatch = int((served != reference).sum())
    assert block_mismatch <= served_mismatch
    assert float(_ulps(normed, reference).max()) <= 1.0
    block_err = (normed.double() - ref64).norm() / ref64.norm()
    served_err = (served.double() - ref64).norm() / ref64.norm()
    assert block_err <= served_err


@pytest.mark.parametrize("rank", [0, 1])
def test_up_projection_consumes_its_column_block_like_the_full_width_shard(rank):
    """tp_size 2, 8-column shards over a 13-wide logical input: rank 1's block
    carries 5 real columns and 3 zeros, and every entry point produces the
    projection the full-width strided window produces."""
    torch.manual_seed(rank)
    projection = _projection(rank, 2, 8, 13, output_size=6)
    full = torch.randn(1024, 13)
    start = rank * 8
    valid = min(8, 13 - start)
    block = torch.zeros(1024, 8)
    block[:, :valid] = full[:, start : start + valid]
    expected = torch.mm(
        full[:, start : start + valid], projection.weight[:, :valid].t()
    )

    out_block, _ = projection.forward_local_block(block)
    torch.testing.assert_close(out_block, expected)
    into = torch.empty(1024, 6)
    projection.forward_into(block, into, x_is_local_block=True)
    torch.testing.assert_close(into, expected)
    full_into = torch.empty(1024, 6)
    projection.forward_into(full, full_into)
    torch.testing.assert_close(full_into, expected)
    with pytest.raises(ValueError, match="Column block has"):
        projection.forward_local_block(block[:, :4])

    transform = KimiRoutedOutputTransform(None, projection, layer_idx=0)
    residual = torch.zeros(1024, 6)
    projection._ACCUMULATION_TILE_ROWS = 6
    assert transform.can_accumulate_residual(block, residual)
    result = transform(block, residual=residual, column_block=True)
    torch.testing.assert_close(result, expected)
    output = torch.empty(1024, 6)
    assert transform.can_write_output(block, output)
    assert (
        transform(block, output=output, column_block=True).data_ptr()
        == output.data_ptr()
    )
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(transform(block, column_block=True), expected)


def test_runner_reduce_hook_prefers_the_transform_reduce_scatter(monkeypatch):
    runner = MoERunner.__new__(MoERunner)
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=False, tp_size=WORLD, ep_size=1
    )
    runner.reduction_borrow_output = False
    partial = torch.zeros(1024, LATENT, dtype=torch.bfloat16)
    block = partial[:, :SHARD]
    transform = Mock()
    transform.reduce_scatter_tp_partial = Mock(return_value=block)
    runner.routed_output_transform = transform
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner."
        "tensor_model_parallel_all_reduce",
        lambda t: pytest.fail("all-reduce must not run when the block is available"),
    )
    out, reduced, column_block = runner._maybe_reduce_routed_output_before_transform(
        partial, False
    )
    assert out is block and reduced and column_block

    transform.reduce_scatter_tp_partial = Mock(return_value=None)
    reduced_calls: list[int] = []
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner."
        "tensor_model_parallel_all_reduce",
        lambda t: reduced_calls.append(t.data_ptr()) or t,
    )
    out, reduced, column_block = runner._maybe_reduce_routed_output_before_transform(
        partial, False
    )
    assert out is partial and reduced and not column_block
    assert reduced_calls == [partial.data_ptr()]

    # The flag reaches the transform only when set.
    transform.reset_mock()
    transform.return_value = block
    runner.apply_routed_output_transform(block, column_block=True)
    assert transform.call_args.kwargs == {"column_block": True}
    runner.apply_routed_output_transform(partial)
    assert transform.call_args.kwargs == {}
