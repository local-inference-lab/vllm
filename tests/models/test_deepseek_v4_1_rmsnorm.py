# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RMSNorm preparation, numerical recipe, and dynamic graph inputs."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("hidden,weight_dtype", [
    (128, torch.bfloat16), (512, torch.bfloat16),
    (512, torch.float32), (1280, torch.bfloat16), (5120, torch.bfloat16),
])
@pytest.mark.parametrize("eps", [1e-20, 1e-5])
@torch.no_grad()
def test_rmsnorm_prepared_forward_and_replay(monkeypatch, hidden, weight_dtype, eps):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x RMSNorm requires SM12x")
    from b12x.preparation import PreparationSession
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.models.deepseek_v4_1 import b12x_layers
    from vllm.utils.b12x import B12xWorkload, PreparationResourceUnavailableError

    device = torch.device("cuda", torch.cuda.current_device())
    capacity = 64
    torch.manual_seed(41)
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=capacity))
    monkeypatch.setattr(b12x_layers, "get_current_vllm_config", lambda: config)
    layer = b12x_layers.B12xRMSNorm(hidden, eps)
    layer.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, hidden, device=device,
                                                   dtype=weight_dtype), requires_grad=False)
    source = torch.randn(capacity, hidden, dtype=torch.bfloat16, device=device)
    source[0].mul_(1e-11)
    source[1].zero_()
    source[2].mul_(100)
    with pytest.raises(PreparationResourceUnavailableError, match="no declared plan"):
        layer(source)
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 8, capacity), fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16, max_tokens=capacity, max_seqs=8, max_model_len=capacity,
    )
    allocated = torch.cuda.memory_allocated(device)
    units = tuple(_units_from_modules(layer, workload))
    assert len(units) == 1 and units[0].stage == "weights"
    assert torch.cuda.memory_allocated(device) == allocated

    def reference(x):
        value = x.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
                * layer.weight.float()).bfloat16()

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(units[0].requests)
        assert torch.cuda.memory_allocated(device) == allocated
        assert layer._plan.prepared is not None
        session.freeze()
        for rows in (1, 8, capacity):
            actual = layer(source[:rows])
            torch.testing.assert_close(actual, reference(source[:rows]), rtol=0.008, atol=1e-5)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        shaped = source.view(8, 8, hidden).transpose(0, 1)
        torch.testing.assert_close(layer(shaped), reference(shaped), rtol=0.008, atol=1e-5)
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                replayed = layer(source)
            pointer = replayed.data_ptr()
            source.neg_()
            layer.weight.mul_(0.5)
            replayed.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            assert replayed.data_ptr() == pointer
            torch.testing.assert_close(replayed, reference(source), rtol=0.008, atol=1e-5)
            assert torch.isfinite(replayed).all() and torch.count_nonzero(replayed) > 0
        finally:
            graph.reset()
        with pytest.raises(ValueError, match="exceeds capacity"):
            layer(torch.empty(capacity + 1, hidden, dtype=torch.bfloat16, device=device))
