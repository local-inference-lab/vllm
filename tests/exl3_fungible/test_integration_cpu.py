# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for the fungible-quant integration hook (M1).

No GPU, no built vllm required: integration.py resolves its vllm
symbols through the ``_moe_module_types`` / ``_collector_cls`` seam
functions, which these tests monkeypatch with fakes. The fakes
duck-type exactly what the hook touches: ``runner.model.modules()``,
``runner.device``, ``module.layer_id``, ``module.router``,
``router.global_num_experts``, ``router.set_capture_fn``.
"""
import importlib.util
from pathlib import Path

import pytest
import torch

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        integration as fq_integration,
    )
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )
except ImportError:  # standalone run against an env without built vllm._C
    _pkg = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")

    def _load(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, _pkg / filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    FqStatsCollector = _load("fq_stats_standalone", "stats.py").FqStatsCollector
    fq_integration = _load("fq_integration_standalone", "integration.py")


class FakeRouter:
    def __init__(self, global_num_experts: int = 8):
        self.global_num_experts = global_num_experts
        self.capture_fn = None

    def set_capture_fn(self, fn):
        self.capture_fn = fn


class FakeMoERunner:
    def __init__(self, layer_id: int, router):
        self.layer_id = layer_id
        self.router = router


class FakeModel:
    def __init__(self, modules):
        self._modules = list(modules)

    def modules(self):
        return iter(self._modules)


class FakeRunner:
    def __init__(self, modules):
        self.model = FakeModel(modules)
        self.device = "cpu"


def make_runner(num_layers: int = 2, num_experts: int = 8) -> FakeRunner:
    mods: list = [object()]  # non-MoE module: must be skipped
    for lid in range(num_layers):
        mods.append(FakeMoERunner(lid, FakeRouter(num_experts)))
    # MoERunner whose router is NOT a BaseRouter: must be skipped too
    mods.append(FakeMoERunner(99, object()))
    return FakeRunner(mods)


def moe_routers(runner: FakeRunner) -> list[FakeRouter]:
    return [m.router for m in runner.model._modules
            if isinstance(m, FakeMoERunner) and isinstance(m.router, FakeRouter)]


@pytest.fixture
def fakes(monkeypatch):
    """Inject the fake isinstance targets + the (real) collector class."""
    monkeypatch.setattr(fq_integration, "_moe_module_types",
                        lambda: (FakeMoERunner, FakeRouter))
    monkeypatch.setattr(fq_integration, "_collector_cls",
                        lambda: FqStatsCollector)
    for env in (fq_integration.FQ_ENABLE_ENV,
                fq_integration.FQ_WINDOW_LEN_ENV,
                fq_integration.FQ_WINDOW_STRIDE_ENV,
                fq_integration.FQ_DECAY_ENV):
        monkeypatch.delenv(env, raising=False)
    return monkeypatch


def test_env_off_returns_none_and_binds_nothing(fakes):
    runner = make_runner()
    assert fq_integration.maybe_init_fq_collector(runner) is None
    assert all(r.capture_fn is None for r in moe_routers(runner))
    fakes.setenv(fq_integration.FQ_ENABLE_ENV, "0")  # explicit off
    assert fq_integration.maybe_init_fq_collector(runner) is None
    assert all(r.capture_fn is None for r in moe_routers(runner))


def test_env_off_imports_nothing(fakes):
    """The env-off path must return before touching the lazy seams —
    that is the zero-import-cost guarantee for the default path."""
    def _boom():
        raise AssertionError("lazy import touched with env off")

    fakes.setattr(fq_integration, "_moe_module_types", _boom)
    fakes.setattr(fq_integration, "_collector_cls", _boom)
    assert fq_integration.maybe_init_fq_collector(make_runner()) is None


def test_env_on_binds_all_routers_and_chains(fakes):
    fakes.setenv(fq_integration.FQ_ENABLE_ENV, "1")
    runner = make_runner(num_layers=3)
    routers = moe_routers(runner)

    # Pre-bind a capture fn on layer 1's router, as the routed-experts
    # capturer would have (enable_return_routed_experts path).
    seen: list = []
    routers[1].capture_fn = lambda ids: seen.append(ids.shape)

    collector = fq_integration.maybe_init_fq_collector(runner)
    assert collector is not None
    assert collector.num_experts == 8
    assert collector.device == torch.device("cpu")
    assert all(r.capture_fn is not None for r in routers)
    assert set(collector.count_buf) == {0, 1, 2}

    # Fire the bound fns: stats recorded AND the previous fn still fires.
    ids = torch.tensor([[0, 1], [1, 7]], dtype=torch.int32)
    routers[1].capture_fn(ids)
    assert collector.count_buf[1][:8].tolist() == [1, 2, 0, 0, 0, 0, 0, 1]
    assert seen == [ids.shape], "previously bound capture fn must chain"
    routers[0].capture_fn(torch.tensor([[4]]))
    assert collector.count_buf[0][4] == 1


def test_no_moe_model_returns_none(fakes):
    fakes.setenv(fq_integration.FQ_ENABLE_ENV, "1")
    runner = FakeRunner([object(), object()])
    assert fq_integration.maybe_init_fq_collector(runner) is None
    # MoERunner present but with a non-BaseRouter router: still no-MoE.
    runner = FakeRunner([FakeMoERunner(0, object())])
    assert fq_integration.maybe_init_fq_collector(runner) is None


def test_window_knobs_read_from_env(fakes):
    fakes.setenv(fq_integration.FQ_ENABLE_ENV, "1")
    fakes.setenv(fq_integration.FQ_WINDOW_LEN_ENV, "5")
    fakes.setenv(fq_integration.FQ_WINDOW_STRIDE_ENV, "3")
    fakes.setenv(fq_integration.FQ_DECAY_ENV, "0.5")
    collector = fq_integration.maybe_init_fq_collector(make_runner())
    assert collector.window_len == 5
    assert collector.window_stride == 3
    assert collector.decay == 0.5


def test_window_knobs_default_to_stats_signature(fakes):
    fakes.setenv(fq_integration.FQ_ENABLE_ENV, "1")
    collector = fq_integration.maybe_init_fq_collector(make_runner())
    assert collector.window_len == 64
    assert collector.window_stride == 32
    assert collector.decay == 0.95
