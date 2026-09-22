# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordering and storage lifetime for the bounded MLA context prefetch ring."""

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.ops.dcp_prefetch import DCPContextPrefetch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream ordering test")
@pytest.mark.parametrize("world_size", [2, 8, 10, 16])
@pytest.mark.parametrize("fp8_transport", [False, True])
def test_context_prefetch_preserves_chunks_and_graph_replay(world_size, fp8_transport):
    class CopyCommunicator:
        disabled = False

        def all_gather(self, output, source, stream):
            assert torch.cuda.current_stream() == stream
            for rank in range(world_size):
                output[rank * source.shape[0] : (rank + 1) * source.shape[0]].copy_(
                    source
                )
                if fp8_transport:
                    header = output[rank * source.shape[0] : rank * source.shape[0] + 4]
                    header.view(torch.float32).mul_(rank + 1)

    manager = SimpleNamespace(
        _kv_gather=functools.partial(torch.distributed.all_gather_into_tensor),
        group=SimpleNamespace(
            world_size=world_size,
            device_communicator=SimpleNamespace(pynccl_comm=CopyCommunicator()),
        ),
    )
    workspace = torch.empty(
        (8 * (world_size + 1), 64), device="cuda", dtype=torch.bfloat16
    )
    pipeline = DCPContextPrefetch(manager, workspace, fp8_transport=fp8_transport)
    scale = torch.tensor([0.1], device="cuda", dtype=torch.float32)
    chunks = [
        SimpleNamespace(num_local_context_tokens=n, index=i)
        for i, n in enumerate((8, 3, 7, 1, 8, 4, 2))
    ]
    sources = [
        torch.randn(
            (c.num_local_context_tokens, 64), device="cuda", dtype=workspace.dtype
        )
        for c in chunks
    ]
    outputs = [
        torch.empty(
            (c.num_local_context_tokens * world_size, 64),
            device="cuda",
            dtype=workspace.dtype,
        )
        for c in chunks
    ]

    def run():
        def extract(chunk, local):
            source = sources[chunk.index]
            if fp8_transport:
                source = source.to(torch.float8_e4m3fn).view(torch.uint8)
            local.copy_(source)

        for chunk, gathered in pipeline.gather_chunks(
            chunks, extract, scale if fp8_transport else None
        ):
            outputs[chunk.index].copy_(gathered)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    addresses = [buffer.data_ptr() for buffer in pipeline.buffers]
    for _ in range(5):
        for source in sources:
            source.normal_()
        for buffer in pipeline.buffers:
            buffer.fill_(255 if fp8_transport else float("nan"))
        for output in outputs:
            output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for source, output in zip(sources, outputs):
            expected = (
                torch.cat(
                    [
                        (
                            source.to(torch.float8_e4m3fn).float()
                            * (scale * (rank + 1))
                        ).bfloat16()
                        for rank in range(world_size)
                    ]
                )
                if fp8_transport
                else source.repeat(world_size, 1)
            )
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
        assert [buffer.data_ptr() for buffer in pipeline.buffers] == addresses
    assert list(pipeline.gather_chunks([], lambda *_: None, scale)) == []
