# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convolution outputs occupy workspace disjoint from the KDA consumer."""

import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as kda
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
from vllm.v1.worker.workspace import WorkspaceManager


def _layer():
    layer: Any = object.__new__(kda.KimiGatedDeltaNetAttention)
    torch.nn.Module.__init__(layer)
    layer.kda_prefill_backend = "b12x"
    layer.get_state_dtype = lambda: (torch.bfloat16, torch.float32)
    layer._b12x_prefill_plan = SimpleNamespace(
        scratch_specs=lambda: [SimpleNamespace(shape=(1024,), dtype=torch.uint8)]
    )
    layer._b12x_prefill_max_tokens = 32
    layer.local_num_heads = 2
    layer.head_dim = 128
    layer.local_projection_size = 256
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    return layer


@pytest.mark.parametrize("capacity", [1024, 100000])
def test_convolution_suffix_preserves_kda_workspace(monkeypatch, capacity):
    manager = WorkspaceManager(torch.device("cpu"))
    manager.reserve_all(((capacity,), torch.uint8))
    manager.lock()
    monkeypatch.setattr(kda, "current_workspace_manager", lambda: manager)
    layer = _layer()
    outputs = layer._borrow_b12x_prefill_transients(32, torch.bfloat16)
    if capacity == 1024:
        assert outputs is None
        return
    for index, output in enumerate(outputs):
        output.fill_(index + 1)
    scratch, target = layer._get_b12x_prefill_workspace()
    scratch.fill_(255)
    target.fill_(17)
    for index, output in enumerate(outputs):
        assert output.shape == (32, 256) and output.is_contiguous()
        assert torch.all(output == index + 1)


@pytest.mark.parametrize(
    "backend,dtype", [("flashkda", torch.bfloat16), ("b12x", torch.float32)]
)
def test_convolution_suffix_keeps_unsupported_paths(backend, dtype):
    layer = _layer()
    layer.kda_prefill_backend = backend
    assert layer._borrow_b12x_prefill_transients(32, dtype) is None


def test_mixed_selection_survives_convolution_and_kda_writes(monkeypatch):
    manager = WorkspaceManager(torch.device("cpu"))
    manager.reserve_all(((200000,), torch.uint8))
    manager.lock()
    monkeypatch.setattr(kda, "current_workspace_manager", lambda: manager)
    layer = _layer()
    source = torch.randn(32, 768, dtype=torch.bfloat16)
    indices = torch.tensor([21, 2, 4, 0, 31, 17, 8, 7], dtype=torch.int64)
    specs = (((8, 768), source.dtype),)
    outputs = layer._borrow_b12x_prefill_transients(8, source.dtype, specs)
    selected = outputs[3]
    torch.index_select(source, 0, indices, out=selected)
    conv = layer._borrow_b12x_prefill_transients(8, source.dtype)
    for buffer in conv:
        buffer.fill_(5)
    scratch, target = layer._get_b12x_prefill_workspace()
    scratch.fill_(255)
    target.fill_(9)
    torch.testing.assert_close(selected, source[indices], rtol=0, atol=0)


@pytest.mark.parametrize("full_rank_gate", [False, True])
def test_consumed_kda_projections_release_before_output_gemm(full_rank_gate):
    layer = _layer()
    layer.local_num_heads = 2
    layer.head_dim = 4
    layer.local_projection_size = 8
    layer.use_full_rank_gate = full_rank_gate
    layer.in_proj_padding = 2 if full_rank_gate else 0
    owners = []

    def project(x, columns):
        result = torch.ones(x.shape[0], columns)
        owners.append(weakref.ref(result))
        return result, None

    layer.in_proj_qkvgfab = lambda x: project(x, 40 if full_rank_gate else 30)
    layer.g_a_proj = lambda x: project(x, 4)
    layer.g_b_proj = lambda x: project(x, 8)
    layer.f_b_proj = lambda x: project(x, 8)

    def recurrence(*, mixed_qkv, g1, g2, beta, core_attn_out):
        core_attn_out.copy_(
            mixed_qkv[:, :8].view(1, -1, 2, 4)
            + g1
            + g2.unsqueeze(0)
            + beta.unsqueeze(-1)
        )

    layer._forward = recurrence

    def output_projection(x):
        assert all(owner() is None for owner in owners)
        return x * 2, None

    layer.o_proj = output_projection
    output = torch.empty(17, 8)
    layer.forward(torch.ones(17, 8), torch.arange(17), output)
    torch.testing.assert_close(output, torch.full_like(output, 8), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [33, 2048, 3072])
@pytest.mark.parametrize("speculative_slots", [0, 3])
def test_caller_owned_convolution_preserves_output_and_history(rows, speculative_slots):
    channels, width = 4096, 4
    storage = torch.empty(rows * 4 * channels, device="cuda", dtype=torch.bfloat16)
    source = storage[rows * channels :].view(rows, 3 * channels)
    source.normal_()
    original = source.clone()
    weight = torch.randn(channels, width, device="cuda", dtype=source.dtype)
    starts = torch.tensor([0, rows // 2, rows], dtype=torch.int32, device="cuda")
    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    initial = torch.tensor([False, True], device="cuda")
    history = torch.randn(
        4, channels, width - 1 + speculative_slots, device="cuda", dtype=source.dtype
    )
    expected_history = history.clone()
    supplied = storage[: rows * channels].view(rows, channels)
    value = source[:, channels : 2 * channels].transpose(0, 1)
    kwargs = dict(
        query_start_loc=starts,
        cache_indices=slots,
        has_initial_state=initial,
        activation="silu",
    )
    expected = causal_conv1d_fn(value, weight, None, expected_history, **kwargs)
    actual = causal_conv1d_fn(
        value, weight, None, history, out=supplied.transpose(0, 1), **kwargs
    )
    assert actual.data_ptr() == supplied.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(history, expected_history, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    with pytest.raises(ValueError, match="disjoint"):
        causal_conv1d_fn(value, weight, None, history, out=value, **kwargs)
