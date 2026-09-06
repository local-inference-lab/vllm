# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paired DMA-ring gather of the Kimi-K3 router logits and routed latent
(``VLLM_K3_PROJECTION_GATHER=dma_pair``): block assembly equals the NCCL
layout, the gate and fallbacks hold, the shared experts run before the
gather is consumed, and the runner adopts their result."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import tp_projection

WORLD = 9


@pytest.fixture
def gather_mode(monkeypatch: pytest.MonkeyPatch):
    def _set(mode: str) -> None:
        monkeypatch.setenv("VLLM_K3_PROJECTION_GATHER", mode)
        tp_projection.kimi_projection_gather_mode.cache_clear()

    yield _set
    tp_projection.kimi_projection_gather_mode.cache_clear()


@pytest.mark.parametrize(
    ("cols", "width", "dtype"),
    [(104, 896, torch.float32), (400, 3584, torch.bfloat16), (8, 72, torch.bfloat16)],
)
def test_assemble_rank_major_blocks_matches_concatenation(cols, width, dtype):
    blocks = torch.randn(WORLD, 6, cols).to(dtype)
    expected = torch.cat(list(blocks), dim=-1)[:, :width].contiguous()
    actual = tp_projection.assemble_rank_major_blocks(blocks, width)
    assert actual.shape == (6, width) and actual.is_contiguous()
    assert torch.equal(
        actual.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
        expected.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
    )


def test_assemble_rank_major_blocks_rejects_bad_width():
    with pytest.raises(ValueError, match="does not fit"):
        tp_projection.assemble_rank_major_blocks(torch.zeros(WORLD, 2, 4), 40)


def test_gather_mode_rejects_unknown_values(gather_mode):
    gather_mode("ring")
    with pytest.raises(ValueError, match="VLLM_K3_PROJECTION_GATHER"):
        tp_projection.kimi_projection_gather_mode()


def test_async_gather_keeps_nccl_unless_enabled(monkeypatch, gather_mode):
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_pcie_all_gather_pair",
        lambda *a: pytest.fail("ring must not be used"),
    )
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: WORLD
    )
    first = torch.zeros(1024, 104)
    second = torch.zeros(1024, 400, dtype=torch.bfloat16)
    gather_mode("nccl")
    assert (
        tp_projection.try_gather_kimi_projection_pair_async(first, 896, second, 3584)
        is None
    )
    gather_mode("dma_pair")
    # Decode-sized rows keep the NCCL path.
    assert (
        tp_projection.try_gather_kimi_projection_pair_async(
            first[:8], 896, second[:8], 3584
        )
        is None
    )


def test_async_gather_issues_on_the_ring_and_assembles_on_wait(
    monkeypatch, gather_mode
):
    gather_mode("dma_pair")
    first = torch.randn(1024, 104)
    second = torch.randn(1024, 400).to(torch.bfloat16)
    stacked = (
        torch.stack([first * (r + 1) for r in range(WORLD)]),
        torch.stack([second * (r + 1) for r in range(WORLD)]),
    )
    calls = []

    def fake_pair(a, b):
        calls.append((a.data_ptr(), b.data_ptr()))
        return stacked[0], stacked[1], None

    monkeypatch.setattr(
        tp_projection, "tensor_model_parallel_pcie_all_gather_pair", fake_pair
    )
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: WORLD
    )
    monkeypatch.setattr(
        tp_projection,
        "_get_kimi_projection_group",
        lambda: SimpleNamespace(world_size=WORLD),
    )
    pending = tp_projection.gather_kimi_projection_pair_prefill(
        first, 896, second, 3584
    )
    assert isinstance(pending, tp_projection.PendingProjectionGather)
    assert calls == [(first.data_ptr(), second.data_ptr())]
    router, latent = pending.wait()
    assert torch.equal(
        router, tp_projection.assemble_rank_major_blocks(stacked[0], 896)
    )
    assert torch.equal(
        latent, tp_projection.assemble_rank_major_blocks(stacked[1], 3584)
    )


def test_prefill_gather_falls_back_to_nccl_when_the_ring_declines(
    monkeypatch, gather_mode
):
    gather_mode("dma_pair")
    first = torch.randn(1024, 104)
    second = torch.randn(1024, 400).to(torch.bfloat16)
    monkeypatch.setattr(
        tp_projection, "tensor_model_parallel_pcie_all_gather_pair", lambda a, b: None
    )
    monkeypatch.setattr(
        tp_projection, "get_tensor_model_parallel_world_size", lambda: WORLD
    )
    monkeypatch.setattr(
        tp_projection,
        "_get_kimi_projection_group",
        lambda: SimpleNamespace(world_size=WORLD),
    )
    monkeypatch.setattr(
        tp_projection,
        "gather_kimi_sharded_projection",
        lambda t: torch.cat([t] * WORLD, dim=-1),
    )
    gathered = tp_projection.gather_kimi_projection_pair_prefill(
        first, 896, second, 3584
    )
    assert isinstance(gathered, tp_projection.CompletedProjectionGather)
    router, latent = gathered.wait()
    assert router.shape == (1024, 896) and latent.shape == (1024, 3584)
    assert torch.equal(router, torch.cat([first] * WORLD, dim=-1)[:, :896])


def test_custom_all_reduce_pair_declines_without_ring_or_during_capture():
    ca = CustomAllreduce.__new__(CustomAllreduce)
    ca.disabled = False
    ca._IS_CAPTURING = False
    ca._pcie_dma = None
    first = torch.zeros(4, 8)
    assert ca.pcie_dma_all_gather_pair(first, first) is None
    ring = Mock()
    ring.should_all_gather_pair.return_value = True
    ca._pcie_dma = ring
    ca._IS_CAPTURING = True
    assert ca.pcie_dma_all_gather_pair(first, first) is None
    ring.all_gather_pair.assert_not_called()


def _shared_experts(order: SharedExpertsOrder) -> tuple[SharedExperts, Mock]:
    shared = SharedExperts.__new__(SharedExperts)
    nn.Module.__init__(shared)
    layer = Mock(side_effect=lambda *a, **k: pytest.fail("layer must not run"))
    shared._layer = layer
    shared._output = [None, None]
    shared._precomputed = [None, None]
    shared.enable_dbo = False
    shared._determine_shared_experts_order = lambda x: order
    return shared, layer


def test_shared_experts_adopt_a_precomputed_output():
    shared, layer = _shared_experts(SharedExpertsOrder.NO_OVERLAP)
    x = torch.randn(4, 8)
    result = torch.randn(4, 8)
    shared.install_precomputed_output(result)
    # The non-matching order call is a no-op, as for computed outputs.
    assert shared.forward(x, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED) is None
    assert shared._precomputed[0] is result
    shared.forward(x, SharedExpertsOrder.NO_OVERLAP)
    assert shared.output is result
    layer.assert_not_called()


def test_shared_experts_precomputed_output_must_alias_for_input_reuse():
    shared, _ = _shared_experts(SharedExpertsOrder.NO_OVERLAP)
    x = torch.randn(4, 8)
    shared.install_precomputed_output(torch.randn(4, 8))
    with pytest.raises(ValueError, match="occupy the shared input storage"):
        shared.forward(x, SharedExpertsOrder.NO_OVERLAP, reuse_input=True)
    shared._precomputed = [None, None]
    shared.install_precomputed_output(x)
    shared.forward(x, SharedExpertsOrder.NO_OVERLAP, reuse_input=True)
    assert shared.output is x


class _LocalProjection(nn.Module):
    def __init__(self, cols: int, logical: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.logical_output_size = logical
        self.cols = cols
        self.dtype = dtype

    def forward_local(self, x: torch.Tensor):
        return torch.zeros(x.shape[0], self.cols, dtype=self.dtype), None


class _Runner(nn.Module):
    def __init__(self, reuse: bool) -> None:
        super().__init__()
        self.shared_experts = SimpleNamespace(can_reuse_input=lambda x: reuse)
        self.calls: list[dict] = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["shared_experts_input"]


@pytest.mark.parametrize("reuse", [True, False])
def test_moe_forward_runs_shared_experts_under_the_gather(
    monkeypatch, gather_mode, reuse
):
    gather_mode("dma_pair")
    moe = kimi_model.KimiMoE.__new__(kimi_model.KimiMoE)
    nn.Module.__init__(moe)
    moe.use_mega_moe = False
    moe.use_latent_moe = True
    gate = object.__new__(kimi_model.KimiColumnParallelGate)
    nn.Module.__init__(gate)
    gate.logical_output_size = 896
    gate.forward_local = lambda x: (torch.zeros(x.shape[0], 104), None)
    down = object.__new__(kimi_model.KimiPaddedColumnParallelLinear)
    nn.Module.__init__(down)
    down.logical_output_size = 3584
    down.forward_local = lambda x: (
        torch.zeros(x.shape[0], 400, dtype=torch.bfloat16),
        None,
    )
    moe.gate = gate
    moe.routed_expert_down_proj = down
    shared_calls: list[dict] = []
    order: list[str] = []

    def shared_experts(x, **kwargs):
        order.append("shared")
        shared_calls.append(kwargs)
        return x if "output" in kwargs else x * 2

    moe.shared_experts = shared_experts
    moe.experts = _Runner(reuse)
    router = torch.ones(1024, 896)
    latent = torch.ones(1024, 3584, dtype=torch.bfloat16)

    class _Gather:
        def wait(self):
            order.append("wait")
            return router, latent

    def fake_gather(*args):
        order.append("issue")
        return _Gather()

    monkeypatch.setattr(kimi_model, "gather_kimi_projection_pair_prefill", fake_gather)
    hidden = torch.randn(1024, 16)

    assert moe._prefill_projection_gather_eligible(hidden)
    assert not moe._prefill_projection_gather_eligible(hidden[:8])
    out = moe.forward(hidden)

    assert order == ["issue", "shared", "wait"]
    assert out.data_ptr() == hidden.data_ptr()
    (call,) = moe.experts.calls
    assert call["router_logits"] is router
    assert call["hidden_states"] is latent
    # forward() re-views the input; storage identity is what matters.
    assert call["shared_experts_input"].data_ptr() == hidden.data_ptr()
    if reuse:
        assert [list(c) for c in shared_calls] == [["output"]]
        assert shared_calls[0]["output"].data_ptr() == hidden.data_ptr()
        assert call["shared_output"].data_ptr() == hidden.data_ptr()
    else:
        assert shared_calls == [{}]
        assert call["shared_output"].data_ptr() != hidden.data_ptr()
