# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM-5.3 KDA gate side stream: same result as the sequential forward,
gate projections off the main stream, graph-capturable."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.forward_context import override_forward_context
from vllm.models.glm5next.nvidia import kda as kda_module
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
from vllm.v1.worker import workspace

HIDDEN, HEADS, HEAD_DIM, GATE_RANK = 64, 2, 16, 8
PROJ = HEADS * HEAD_DIM


@pytest.fixture(autouse=True)
def forward_context():
    with override_forward_context(SimpleNamespace(attn_metadata=None)):
        yield


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
    torch.accelerator.synchronize(hidden_states.device)
    return out


def test_gate_overlap_is_disabled_for_the_entire_breakable_capture(monkeypatch):
    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: object()),
    )
    assert not kda_module._gate_overlap_allowed()


def test_gate_overlap_is_disabled_for_uncaptured_graph_warmup(monkeypatch):
    from vllm.compilation import monitor

    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not kda_module._gate_overlap_allowed()


def test_gate_overlap_is_enabled_for_regular_full_capture(monkeypatch):
    from vllm.compilation import monitor

    monkeypatch.setattr(
        BreakableCUDAGraphCapture,
        "current",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert kda_module._gate_overlap_allowed()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gate_side_stream_matches_sequential_forward(monkeypatch):
    from vllm.compilation import monitor

    device = torch.device("cuda", 0)
    generator = torch.Generator().manual_seed(11)
    log: list[tuple[str, int]] = []
    layer = _make_layer(device, log, generator)
    x = (torch.randn(8, HIDDEN, generator=generator)).to(device, torch.bfloat16)

    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: False)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
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
    torch.accelerator.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = layer(static, positions=None)
    graph.replay()
    torch.accelerator.synchronize(device)
    assert torch.equal(captured, eager)
    for _ in range(3):
        static.copy_(torch.randn_like(static))
        expected = _run(layer, static)
        graph.replay()
        torch.accelerator.synchronize(device)
        assert torch.equal(captured, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gate_overlap_keeps_projection_scratch_disjoint(monkeypatch):
    """Concurrent gate/main linears must not overwrite each other's scratch."""
    from vllm.compilation import monitor

    device = torch.device("cuda", 0)
    manager = workspace.WorkspaceManager(device)
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: False)
    generator = torch.Generator().manual_seed(19)
    layer = _make_layer(device, [], generator)
    allocations = {}

    class ScratchLinear(_Linear):
        @property
        def quant_method(self):
            return self

        def get_workspace_size(self, layer, rows):
            return rows * self.weight.shape[1] * self.weight.element_size()

        def __call__(self, x):
            size = self.get_workspace_size(self, x.shape[0])
            scratch = workspace.current_preallocated_workspace()
            if scratch is None:
                (scratch,) = manager.get_simultaneous(((size,), torch.uint8))
            allocations[self.name] = (scratch.data_ptr(), size)
            result, _ = super().__call__(x)
            temporary = scratch[:size].view(result.dtype).view_as(result)
            temporary.copy_(result)
            return temporary.clone(), None

    for name in ("g_a_proj", "g_b_proj", "in_proj_qkvgfab", "f_b_proj"):
        linear = getattr(layer, name)
        setattr(layer, name, ScratchLinear(linear.name, linear.weight, linear.log))
    x = torch.randn(4, HIDDEN, device=device, dtype=torch.bfloat16)
    monkeypatch.setattr(kda_module, "_GATE_SIDE_STREAM", False)
    expected = _run(layer, x)
    monkeypatch.setattr(kda_module, "_GATE_SIDE_STREAM", True)
    # Warmup is serialized but must reserve enough for the captured fork.
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: True)
    _run(layer, x)
    manager.lock()
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: False)
    actual = _run(layer, x)
    for gate in ("g_a", "g_b"):
        gate_ptr, gate_bytes = allocations[gate]
        for main in ("in_proj", "f_b"):
            main_ptr, main_bytes = allocations[main]
            assert (
                gate_ptr + gate_bytes <= main_ptr or main_ptr + main_bytes <= gate_ptr
            ), f"{gate} scratch overlaps {main} scratch"
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(x, positions=None)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        monkeypatch.setattr(kda_module, "_GATE_SIDE_STREAM", False)
        expected = _run(layer, x)
        graph.replay()
        torch.accelerator.synchronize(device)
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("overlap", [False, True])
def test_projection_owners_release_before_glm_output_gemm(monkeypatch, overlap):
    """GLM returns the output projection directly after releasing inputs."""
    from vllm.compilation import monitor

    device = torch.device("cuda", 0)
    layer = _make_layer(device, [], torch.Generator().manual_seed(27))
    owners = []

    class TrackedLinear(_Linear):
        def __call__(self, x):
            result, bias = super().__call__(x)
            owners.append(weakref.ref(result))
            return result, bias

    for name in ("g_a_proj", "g_b_proj", "in_proj_qkvgfab", "f_b_proj"):
        linear = getattr(layer, name)
        setattr(layer, name, TrackedLinear(linear.name, linear.weight, linear.log))
    original = layer.o_proj
    outputs = []

    def output_projection(x):
        assert all(owner() is None for owner in owners)
        result = original(x)
        outputs.append(result[0])
        return result

    layer.o_proj = output_projection
    monkeypatch.setattr(kda_module, "_GATE_SIDE_STREAM", overlap)
    monkeypatch.setattr(monitor, "is_cudagraph_capturing_enabled", lambda: False)
    x = torch.randn(64, HIDDEN, device=device, dtype=torch.bfloat16)
    actual = _run(layer, x)
    assert actual is outputs[0]
