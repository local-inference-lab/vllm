# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA GEMMs must not read outside small, interleaved input allocations."""

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _bmm_with_disjoint_batches,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 32])
@pytest.mark.parametrize("projection", ["query", "value"])
@torch.inference_mode()
def test_mla_bmm_at_mapped_allocation_end(rows, projection):
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability_family(120)
    ):
        pytest.skip("Regression requires the SM120/121 cuBLAS path")
    if not hasattr(torch.cuda.MemPool, "snapshot"):
        pytest.skip("Allocator boundary inspection requires MemPool.snapshot")

    k, n = (256, 512) if projection == "query" else (512, 256)
    pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(pool):
        owner = torch.ones(10 * 1024 * 1024, device="cuda", dtype=torch.bfloat16)
    end = owner.data_ptr() + owner.numel() * owner.element_size()
    assert any(s["address"] + s["total_size"] == end for s in pool.snapshot())
    source = owner[-rows * 16 * k :].view(rows, 16, k)
    lhs = source.transpose(0, 1)
    rhs = torch.ones((16, k, n), device="cuda", dtype=torch.bfloat16)
    result = torch.empty((rows, 16, n), device="cuda", dtype=torch.bfloat16)
    out = result.transpose(0, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _bmm_with_disjoint_batches(lhs, rhs, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _bmm_with_disjoint_batches(lhs, rhs, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    for value in (1, 3, -5):
        source.fill_(value)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(
            result, torch.full_like(result, value * k), rtol=0, atol=0
        )
    graph.reset()


@pytest.mark.parametrize("rows", [1, 4, 8, 32, 128, 4096])
@pytest.mark.parametrize("projection", ["query", "value"])
@torch.inference_mode()
def test_mla_bmm_random_values_and_graph_replay(rows, projection):
    if not current_platform.is_cuda():
        pytest.skip("CUDA GEMM qualification")
    torch.manual_seed(42)
    k, n = (256, 512) if projection == "query" else (512, 256)
    source = torch.randn((rows, 16, k), device="cuda", dtype=torch.bfloat16)
    lhs = source.transpose(0, 1)
    rhs = torch.randn((16, k, n), device="cuda", dtype=torch.bfloat16)
    result = torch.empty((rows, 16, n), device="cuda", dtype=torch.bfloat16)
    out = result.transpose(0, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _bmm_with_disjoint_batches(lhs, rhs, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _bmm_with_disjoint_batches(lhs, rhs, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    for sign in (1, -1):
        source.mul_(sign)
        graph.replay()
        reference = torch.bmm(lhs.float().contiguous(), rhs.float()).bfloat16()
        torch.testing.assert_close(out, reference, rtol=0.02, atol=0.25)
    graph.reset()
