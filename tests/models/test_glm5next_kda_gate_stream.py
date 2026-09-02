# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3 KDA gate side stream: same result as the sequential forward,
gate projections off the main stream, graph-capturable."""

import pytest
import torch

from vllm.models.glm5next.nvidia import kda as kda_module
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HIDDEN, HEADS, HEAD_DIM, GATE_RANK = 64, 2, 16, 8
PROJ = HEADS * HEAD_DIM


class _Linear:
    """Stands in for a vLLM linear layer: returns (y, None) and records the
    CUDA stream it ran on."""

    def __init__(self, name, weight, log):
        self.name, self.weight, self.log = name, weight, log

    def __call__(self, x):
        self.log.append((self.name, torch.cuda.current_stream(x.device).cuda_stream))
        return x @ self.weight, None


def _make_layer(device, log, generator):
    def w(rows, cols):
        return (torch.randn(rows, cols, generator=generator) / rows**0.5).to(
            device, torch.bfloat16
        )

    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.use_full_rank_gate = False
    layer.local_projection_size = PROJ
    layer.local_num_heads = HEADS
    layer.head_dim = HEAD_DIM
    layer.in_proj_qkvgfab = _Linear(
        "in_proj", w(HIDDEN, 3 * PROJ + HEADS + HEAD_DIM), log
    )
    layer.g_a_proj = _Linear("g_a", w(HIDDEN, GATE_RANK), log)
    layer.g_b_proj = _Linear("g_b", w(GATE_RANK, PROJ), log)
    layer.f_b_proj = _Linear("f_b", w(HEAD_DIM, PROJ), log)
    layer.o_proj = _Linear("o_proj", w(PROJ, HIDDEN), log)

    def core(*, mixed_qkv, g1, g2, beta, core_attn_out):
        n = mixed_qkv.shape[0]
        q = mixed_qkv[:, :PROJ].reshape(1, n, HEADS, HEAD_DIM)
        core_attn_out.copy_(q * g2 + g1 * beta.reshape(1, n, HEADS, 1))

    layer._forward = core
    return layer


def _run(layer, hidden_states):
    out = layer(hidden_states, positions=None)
    torch.cuda.synchronize(hidden_states.device)
    return out


def test_gate_side_stream_matches_sequential_forward(monkeypatch):
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(11)
    log: list[tuple[str, int]] = []
    layer = _make_layer(device, log, generator)
    x = (torch.randn(8, HIDDEN, generator=generator)).to(device, torch.bfloat16)

    monkeypatch.setattr(kda_module, "_GATE_SIDE_STREAM", True)
    overlapped = _run(layer, x)
    main = torch.cuda.current_stream(device).cuda_stream
    streams = dict(log)
    assert streams["g_a"] != main and streams["g_b"] == streams["g_a"]
    assert streams["in_proj"] == main and streams["o_proj"] == main
    log.clear()

    monkeypatch.setattr(kda_module, "_GATE_SIDE_STREAM", False)
    sequential = _run(layer, x)
    assert all(stream == main for _, stream in log)
    assert torch.equal(overlapped, sequential)


def test_gate_side_stream_is_graph_capturable(monkeypatch):
    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(5)
    log: list[tuple[str, int]] = []
    layer = _make_layer(device, log, generator)
    static = (torch.randn(4, HIDDEN, generator=generator)).to(device, torch.bfloat16)
    monkeypatch.setattr(kda_module, "_GATE_SIDE_STREAM", True)
    eager = _run(layer, static)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(2):
            layer(static, positions=None)
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = layer(static, positions=None)
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(captured, eager)
