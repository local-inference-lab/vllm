# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 router compilation and graph replay preserve FP32 logits."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability((12, 0)), reason="requires SM120"
)


def _gate(monkeypatch):
    for name in (
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(f"{name}.get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(f"{name}.get_tensor_model_parallel_world_size", lambda: 1)
    gate = GateLinear(
        4096, 256, params_dtype=torch.bfloat16, out_dtype=torch.float32
    ).cuda()
    with torch.no_grad():
        gate.weight.normal_(std=0.01)
    assert gate._sm120_graph_pool_lifetime_guard
    return gate


@pytest.mark.parametrize("compiled", [False, True])
@torch.inference_mode()
def test_sm120_router_fullgraph_and_shared_pool_replay(monkeypatch, compiled):
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.manual_seed(41)
    gate = _gate(monkeypatch)
    forward = (
        torch.compile(gate, fullgraph=True, backend="eager", dynamic=True)
        if compiled
        else gate
    )
    pool = torch.cuda.graph_pool_handle()
    captures = []
    for rows in (32, 64):
        x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                result, bias = forward(x)
        torch.cuda.current_stream().wait_stream(stream)
        assert result.dtype == torch.float32 and bias is None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            output, _ = forward(x)
        captures.append((graph, x, output))
    # Replay in capture order while both differently sized graphs remain alive.
    for _ in range(3):
        for graph, x, output in captures:
            x.normal_()
            graph.replay()
            reference = torch.nn.functional.linear(x.float(), gate.weight.float())
            torch.testing.assert_close(output, reference, atol=0.002, rtol=0.0001)
