# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch of the weight-first fused router + shared gate_up projection in
the MoE runner (``MoERunner._weight_first_projection``) and the shared-expert
wrapper that consumes its gate_up activation. The b12x kernel itself is
tested in the b12x repository; here the projection is a stand-in recorded
through ``monkeypatch``.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsFromGateUp,
)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(8, 4, bias=False, dtype=torch.bfloat16)
        self.act_fn = lambda t: t * 2
        self.down_proj = _Down()
        self.shard_sequence_parallel = "marker"
        self.calls: list[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append("mlp")
        return x + 1


class _Down(nn.Module):
    def forward(self, x: torch.Tensor):
        return x + 100, None


class _FakeProjection:
    def __init__(self, weights, *, depth, pdl):
        self.weights = weights
        self.depth = depth
        self.pdl = pdl
        self.n = sum(int(w.shape[0]) for w in weights)
        self.k = int(weights[0].shape[1])
        self.nt, self.kt = 48, 768
        self.calls = 0

    def supports(self, x: torch.Tensor) -> bool:
        return 1 <= x.shape[0] <= 16

    def __call__(self, x: torch.Tensor):
        self.calls += 1
        return x[:, :2].clone(), x[:, 2:6].clone()


def _fake_b12x(monkeypatch: pytest.MonkeyPatch, *, supported: bool = True):
    import sys
    import types
    from typing import Any

    module: Any = types.ModuleType("b12x.gemm.weight_first_gemv")
    module.WeightFirstProjection = _FakeProjection
    module.is_supported = lambda device=None: supported
    module.supports = lambda weights, x=None: all(
        w.dtype == torch.bfloat16 for w in weights
    )
    gemm: Any = types.ModuleType("b12x.gemm")
    gemm.weight_first_gemv = module
    package: Any = types.ModuleType("b12x")
    package.gemm = gemm
    monkeypatch.setitem(sys.modules, "b12x", package)
    monkeypatch.setitem(sys.modules, "b12x.gemm", gemm)
    monkeypatch.setitem(sys.modules, "b12x.gemm.weight_first_gemv", module)
    return module


def _runner(*, gate=True, shared=True, fse_fuse_gate=False):
    runner = moe_runner.MoERunner.__new__(moe_runner.MoERunner)
    runner.gate = (
        SimpleNamespace(weight=torch.zeros(2, 8, dtype=torch.bfloat16), bias=None)
        if gate
        else None
    )
    runner._shared_experts = (
        SimpleNamespace(_layer=_Mlp(), _stream=None) if shared else None
    )
    runner._fse_fuse_gate = fse_fuse_gate
    return runner


def test_wrapper_consumes_gate_up_once_and_forwards_attributes() -> None:
    mlp = _Mlp()
    wrapper = SharedExpertsFromGateUp(mlp)
    assert wrapper.shard_sequence_parallel == "marker"
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    gate_up = torch.ones(2, 4, dtype=torch.bfloat16)
    wrapper.gate_up = gate_up
    out = wrapper(x)
    # act_fn doubles, down_proj adds 100: the wrapped MLP forward did not run.
    assert torch.equal(out, torch.full((2, 4), 102, dtype=torch.bfloat16))
    assert wrapper.gate_up is None
    assert mlp.calls == []
    assert torch.equal(wrapper(x), x + 1)
    assert mlp.calls == ["mlp"]


def test_projection_built_once_and_used_at_decode_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_b12x(monkeypatch)
    monkeypatch.setattr(moe_runner, "_WF_GATE_ENABLED", True)
    monkeypatch.setattr(moe_runner, "_WF_STAGE_DEPTH", 3)
    runner = _runner()
    mlp = runner._shared_experts._layer
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    projection = runner._weight_first_projection(x, x)
    assert isinstance(projection, _FakeProjection)
    assert projection.depth == 3 and projection.pdl is True
    assert projection.weights[0] is runner.gate.weight
    assert projection.weights[1] is mlp.gate_up_proj.weight
    assert isinstance(runner._shared_experts._layer, SharedExpertsFromGateUp)
    assert runner._shared_experts._layer.mlp is mlp
    # The build happens once; later calls reuse the instance.
    assert runner._weight_first_projection(x, x) is projection
    # Rows beyond the kernel contract (the same tensor as the shared-expert
    # input, so the row limit itself decides) and a different shared-expert
    # input return None without rebuilding.
    x_17 = torch.zeros(17, 8, dtype=torch.bfloat16)
    assert runner._weight_first_projection(x_17, x_17) is None
    other = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert runner._weight_first_projection(x, other) is None
    assert runner.__dict__["_wf_projection"] is projection


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate": False},
        {"shared": False},
        {"fse_fuse_gate": True},
    ],
)
def test_projection_absent_without_gate_or_shared_experts(
    monkeypatch: pytest.MonkeyPatch, kwargs
) -> None:
    _fake_b12x(monkeypatch)
    monkeypatch.setattr(moe_runner, "_WF_GATE_ENABLED", True)
    runner = _runner(**kwargs)
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert runner._weight_first_projection(x, x) is None
    assert "_wf_projection" not in runner.__dict__


def test_projection_disabled_by_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_b12x(monkeypatch)
    monkeypatch.setattr(moe_runner, "_WF_GATE_ENABLED", False)
    runner = _runner()
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert runner._weight_first_projection(x, x) is None
    assert "_wf_projection" not in runner.__dict__


def test_projection_disabled_when_weights_outside_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_b12x(monkeypatch)
    monkeypatch.setattr(moe_runner, "_WF_GATE_ENABLED", True)
    runner = _runner()
    runner.gate.weight = torch.zeros(2, 8, dtype=torch.float16)
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert runner._weight_first_projection(x, x) is None
    assert runner.__dict__["_wf_projection"] is False
    assert isinstance(runner._shared_experts._layer, _Mlp)


def test_projection_disabled_when_device_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_b12x(monkeypatch, supported=False)
    monkeypatch.setattr(moe_runner, "_WF_GATE_ENABLED", True)
    runner = _runner()
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert runner._weight_first_projection(x, x) is None
    assert runner.__dict__["_wf_projection"] is False
