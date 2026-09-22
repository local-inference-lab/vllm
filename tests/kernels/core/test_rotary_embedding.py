# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for miscellaneous utilities
"""

import pytest
import torch

from tests.kernels.utils import opcheck
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


def rotary_embedding_opcheck(
    rot,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None = None,
):
    cos_sin_cache = rot.cos_sin_cache.to(query.device, dtype=query.dtype)

    # ops.rotary_embedding() is a in-place operation
    # that updates the query and key tensors.
    opcheck(
        torch.ops._C.rotary_embedding,
        (positions, query, key, rot.head_size, cos_sin_cache, rot.is_neox_style),
    )


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("max_position", [11, 4096, 32768])
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("rotary_dim", [32])
@pytest.mark.parametrize("head_size", [32, 108])
@pytest.mark.parametrize("seq_len", [11, 1024])
@pytest.mark.parametrize("use_key", [True, False])
@pytest.mark.parametrize("head_stride_is_contiguous", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotary_embedding_opcheck(
    default_vllm_config,
    dist_init,
    device,
    max_position,
    is_neox_style,
    rotary_dim,
    head_size,
    seq_len,
    use_key,
    head_stride_is_contiguous,
    dtype,
):
    batch_size = 1
    base = 10000
    num_heads = 7
    rot = RotaryEmbedding(
        head_size, rotary_dim, max_position, base, is_neox_style, dtype
    )

    positions = torch.randint(0, max_position, (batch_size, seq_len), device=device)
    head_stride = head_size + (64 if head_stride_is_contiguous else 0)

    query = torch.randn(
        batch_size, seq_len, num_heads, head_stride, dtype=dtype, device=device
    )
    key = torch.randn_like(query) if use_key else None
    query = query[..., :head_size]
    key = key[..., :head_size] if key is not None else None

    rotary_embedding_opcheck(rot, positions, query, key)

    # if we have a contiguous head stride, test the alternate
    # [..., num_heads * head_dim] shape/layout
    if head_stride_is_contiguous:
        rotary_embedding_opcheck(
            rot,
            positions,
            query.flatten(start_dim=-2),
            key.flatten(start_dim=-2) if key is not None else None,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("is_neox_style", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_compact_rotary_preserves_full_table_and_changed_input_graphs(
    default_vllm_config, is_neox_style, dtype
):
    """Global positions above 500k retain exact table values and rotated Q/K."""
    from vllm.model_executor.layers.rotary_embedding.compact import (
        CompactRotaryEmbedding,
    )

    capacity, head_dim, max_position = 4096, 128, 1048576
    with torch.device("cuda:0"):
        reference = RotaryEmbedding(
            head_dim, head_dim, max_position, 2000000, is_neox_style, dtype
        )
        compact = CompactRotaryEmbedding(
            head_dim,
            head_dim,
            max_position,
            2000000,
            is_neox_style,
            dtype,
            capacity=capacity,
        )
    positions = torch.randint(max_position, (capacity,), device="cuda:0")
    positions[:8] = torch.tensor(
        [0, 1, 4095, 500000, 524287, 524288, 950000, 1048575], device="cuda:0"
    )
    query = torch.randn(capacity, head_dim * 2, dtype=dtype, device="cuda:0")
    key = torch.randn(capacity, head_dim, dtype=dtype, device="cuda:0")
    pointers = [buffer.data_ptr() for buffer in compact.buffers()]
    assert (
        sum(t.nbytes for t in compact.buffers()) < reference.cos_sin_cache.nbytes / 100
    )
    for rows in (1, 8, 129, capacity):
        q, k = query[:rows].clone(), key[:rows].clone()
        expected_q, expected_k = reference.forward_cuda(positions[:rows], q, k)
        q, k = query[:rows].clone(), key[:rows].clone()
        actual_q, actual_k = compact.forward_cuda(positions[:rows], q, k)
        torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(
            compact.cos_sin_cache[:rows],
            reference.cos_sin_cache[positions[:rows]],
            rtol=0,
            atol=0,
        )

    rows = 8
    q, k = query[:rows].clone(), key[:rows].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        compact.forward_cuda(positions[:rows], q, k)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        compact.forward_cuda(positions[:rows], q, k)
    positions[:rows].add_(123).remainder_(max_position)
    expected_q, expected_k = reference.forward_cuda(
        positions[:rows], query[:rows].clone(), key[:rows].clone()
    )
    q.copy_(query[:rows])
    k.copy_(key[:rows])
    compact.cos_sin_cache.fill_(float("nan"))
    compact.freqs_workspace.fill_(float("nan"))
    allocated = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
    assert pointers == [buffer.data_ptr() for buffer in compact.buffers()]

    with pytest.raises(ValueError, match="exceed prepared capacity"):
        compact.materialize(
            torch.zeros(capacity + 1, device="cuda:0", dtype=torch.int64)
        )
