# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Borrowed DMA-ring outputs for Kimi-K3 prefill reductions
(``VLLM_K3_RING_STATIC_IO``): the keyword reaches the ring only when the
gate is on, and the two retention sites copy a borrowed tensor out."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.kimi_k3.nvidia import model as kimi_model
from vllm.models.kimi_k3.nvidia import tp_projection


@pytest.fixture
def ring_static_io(monkeypatch: pytest.MonkeyPatch):
    def _set(enabled: bool) -> None:
        monkeypatch.setenv("VLLM_K3_RING_STATIC_IO", "1" if enabled else "0")
        tp_projection.kimi_ring_static_io_enabled.cache_clear()

    yield _set
    tp_projection.kimi_ring_static_io_enabled.cache_clear()


class _Reducer:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict]] = []

    def __call__(self, value: torch.Tensor, **kwargs) -> torch.Tensor:
        self.calls.append((value.data_ptr(), kwargs))
        return value


def test_full_width_projection_borrows_only_when_enabled(monkeypatch, ring_static_io):
    reducer = _Reducer()
    monkeypatch.setattr(
        tp_projection, "tensor_model_parallel_all_reduce_in_place", reducer
    )
    output = torch.empty(1024, 4)

    ring_static_io(False)
    tp_projection.reduce_kimi_full_width_projection(output, 9)
    ring_static_io(True)
    tp_projection.reduce_kimi_full_width_projection(output, 9)

    assert reducer.calls == [
        (output.data_ptr(), {}),
        (output.data_ptr(), {"borrow_output": True}),
    ]


def test_materialize_copies_only_borrowed_reductions(monkeypatch, ring_static_io):
    borrowed = torch.zeros(4)
    owned = torch.zeros(4)
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_is_borrowed_storage",
        lambda t: t.data_ptr() == borrowed.data_ptr(),
    )
    ring_static_io(True)
    assert tp_projection.kimi_reduction_is_borrowed(borrowed)
    assert not tp_projection.kimi_reduction_is_borrowed(owned)
    copy = tp_projection.materialize_kimi_reduction(borrowed)
    assert copy.data_ptr() != borrowed.data_ptr()
    assert tp_projection.materialize_kimi_reduction(owned) is owned

    ring_static_io(False)
    assert tp_projection.materialize_kimi_reduction(borrowed) is borrowed
    assert not tp_projection.kimi_reduction_is_borrowed(borrowed)


def test_runner_in_place_reduction_forwards_the_borrow_flag(monkeypatch):
    runner = MoERunner.__new__(MoERunner)
    reducer = _Reducer()
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.runner.moe_runner."
        "tensor_model_parallel_all_reduce_in_place",
        reducer,
    )
    states = torch.empty(2048, 8)
    runner.reduction_borrow_output = False
    runner._all_reduce_in_place(states)
    runner.reduction_borrow_output = True
    runner._all_reduce_in_place(states)
    assert reducer.calls == [
        (states.data_ptr(), {}),
        (states.data_ptr(), {"borrow_output": True}),
    ]


def _custom_all_reduce_with_ring() -> tuple[CustomAllreduce, Mock]:
    ring = Mock()
    ring.should_allreduce.return_value = True
    ring.all_reduce.side_effect = lambda inp, out=None, **kw: inp
    ring.is_ring_storage.side_effect = lambda t: t.shape[0] == 3
    ca = CustomAllreduce.__new__(CustomAllreduce)
    ca._pcie_runtime = SimpleNamespace()
    ca._pcie_dma = ring
    ca._pcie_twoshot = None
    ca._pcie_twoshot_max_bytes = 0
    ca._pcie_allreduce_max_size = 16
    ca._pcie_capture_stream = None
    ca._IS_CAPTURING = False
    ca.disabled = False
    return ca, ring


def test_custom_all_reduce_forwards_borrow_to_the_dma_ring():
    ca, ring = _custom_all_reduce_with_ring()
    inp = torch.empty(1024, 8, dtype=torch.bfloat16)

    ca.all_reduce(inp)
    ca.all_reduce(inp, borrow_output=True)

    assert ring.all_reduce.call_args_list[0].kwargs == {"out": None}
    assert ring.all_reduce.call_args_list[1].kwargs == {
        "out": None,
        "borrow_output": True,
    }
    assert ca.is_pcie_ring_storage(torch.empty(3, 1))
    assert not ca.is_pcie_ring_storage(torch.empty(4, 1))


def test_cuda_communicator_in_place_reduction_forwards_the_borrow_flag():
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    ca_comm = Mock()
    ca_comm.disabled = False
    ca_comm.should_custom_ar.return_value = True
    ca_comm.custom_all_reduce.side_effect = lambda inp, **kw: inp
    ca_comm.is_pcie_ring_storage.side_effect = lambda t: t.shape[0] == 3
    communicator.ca_comm = ca_comm
    tensor = torch.empty(1024, 4)

    communicator.all_reduce_in_place(tensor)
    communicator.all_reduce_in_place(tensor, borrow_output=True)

    assert ca_comm.custom_all_reduce.call_args_list[0].kwargs == {}
    assert ca_comm.custom_all_reduce.call_args_list[1].kwargs == {"borrow_output": True}
    assert communicator.is_borrowed_reduction_storage(torch.empty(3, 1))
    assert not communicator.is_borrowed_reduction_storage(torch.empty(2, 1))


def _decoder_layer(block_write: bool) -> kimi_model.KimiDecoderLayer:
    layer = kimi_model.KimiDecoderLayer.__new__(kimi_model.KimiDecoderLayer)
    nn.Module.__init__(layer)
    layer.use_attn_res = True
    layer.is_block_write_layer = block_write
    layer.is_final_block_write_layer = False
    layer.reuse_attn_res_output = True
    layer.prev_valid_blocks = 1
    layer.mlp_res_norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    layer.mlp_res_proj = SimpleNamespace(weight=torch.ones(1, 4))
    layer.post_attention_layernorm = SimpleNamespace(
        weight=torch.ones(4), variance_epsilon=1e-5
    )
    return layer


@pytest.mark.parametrize("block_write", [True, False])
def test_post_attention_norm_materializes_a_borrowed_prefix(
    monkeypatch, ring_static_io, block_write
):
    """At a block-write layer the attention-output reduction becomes the
    retained AttnRes prefix, so a borrowed one is copied out; elsewhere it
    is consumed in place."""
    ring_static_io(True)
    hidden = torch.arange(8.0).view(2, 4)
    prefix_old = torch.zeros(2, 4)
    residual = torch.zeros(2, 1, 4)
    monkeypatch.setattr(
        tp_projection,
        "tensor_model_parallel_is_borrowed_storage",
        lambda t: t.data_ptr() == hidden.data_ptr(),
    )
    seen: dict[str, int] = {}

    def fake_attn_res(prefix, delta, blocks, *args, output=None, **kwargs):
        seen["prefix_ptr"] = prefix.data_ptr()
        seen["delta_ptr"] = delta.data_ptr() if delta is not None else -1
        seen["output_ptr"] = output.data_ptr() if output is not None else -1
        return output if output is not None else prefix.clone()

    monkeypatch.setattr(kimi_model, "attn_res", fake_attn_res)
    layer = _decoder_layer(block_write)

    _, prefix_new, _ = layer._post_attn_norm(hidden, residual, prefix_old)

    if block_write:
        assert prefix_new.data_ptr() != hidden.data_ptr()
        torch.testing.assert_close(prefix_new, hidden)
        assert seen["prefix_ptr"] == prefix_new.data_ptr()
        assert seen["output_ptr"] == prefix_old.data_ptr()
    else:
        assert prefix_new is prefix_old
        assert seen["delta_ptr"] == hidden.data_ptr()
        assert seen["output_ptr"] == hidden.data_ptr()
