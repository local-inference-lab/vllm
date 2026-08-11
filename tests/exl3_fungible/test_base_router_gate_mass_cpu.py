# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The router half of the gate-mass contract, exercised on CPU.

Gate weights are a LOCAL in ``BaseRouter._select_experts``: nothing on
the router object holds them, and the capture hook historically passed
``topk_ids`` only. That is why the fungible-quant collector's
``topk_weights_getter`` was never bound in the live path and ``mass``
came back byte-identical to ``count``. The fix is at that call site: a
capture fn tagged ``wants_topk_weights`` is handed both.

``base_router.py`` cannot simply be imported here — the source tree's
``vllm`` is not built (no ``vllm._C``), which is why every test in this
directory runs ``--noconftest`` against file-loaded modules. So we load
the REAL file by path with its five vllm imports stubbed out. That
still executes the actual ``_select_experts`` we ship, rather than a
paraphrase of it.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_SRC = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
        / "layers" / "fused_moe" / "router" / "base_router.py")


class _StubFusedMoERouter:
    def __init__(self, eplb_state=None):
        self._routing_replay_out = None
        self.eplb_state = eplb_state


def _load_base_router():
    """Load base_router.py with its vllm dependencies stubbed.

    ``current_platform.is_cuda_alike() -> False`` selects the non-triton
    branch, so no GPU toolchain is touched.
    """
    stubs: dict[str, types.ModuleType] = {}

    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        stubs[name] = m
        return m

    mod("vllm.distributed.eplb.eplb_state", EplbLayerState=object)
    mod("vllm.model_executor.layers.fused_moe.router.fused_moe_router",
        FusedMoERouter=_StubFusedMoERouter)
    mod("vllm.platforms",
        current_platform=types.SimpleNamespace(is_cuda_alike=lambda: False))
    mod("vllm.triton_utils", tl=None, triton=None)
    mod("vllm.v1.worker.ubatching", dbo_current_ubatch_id=lambda: 0)

    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "fq_base_router_under_test", _SRC)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.BaseRouter
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


BaseRouter = _load_base_router()


class ToyRouter(BaseRouter):
    """Concrete router returning fixed routing, like a real subclass."""

    def __init__(self, weights, ids, num_experts=8):
        super().__init__(top_k=ids.shape[-1], global_num_experts=num_experts)
        self._weights = weights
        self._ids = ids

    @property
    def routing_method_type(self):
        return None

    def _compute_routing(self, hidden_states, router_logits, indices_type,
                         *, input_ids=None):
        return self._weights, self._ids


def make_router():
    return ToyRouter(torch.tensor([[0.1, 0.9]], dtype=torch.float32),
                     torch.tensor([[2, 5]], dtype=torch.int64))


def drive(router):
    return router._select_experts(torch.zeros(1, 4), torch.zeros(1, 8))


def test_default_capture_fn_still_gets_ids_only():
    """Backward compatibility: the routed-experts capturer binds a
    one-argument callback and must keep working untouched."""
    r = make_router()
    seen = []
    r.set_capture_fn(lambda ids: seen.append(ids))
    assert r.capture_fn_wants_weights is False
    drive(r)
    assert len(seen) == 1 and torch.equal(seen[0], r._ids)


def test_tagged_capture_fn_receives_the_gate_weights():
    """The fix: a capture fn tagged ``wants_topk_weights`` is called with
    (topk_ids, topk_weights). Without it there is no reachable source of
    gate mass anywhere on the router."""
    r = make_router()
    seen = []

    def cap(ids, weights):
        seen.append((ids, weights))

    cap.wants_topk_weights = True
    r.set_capture_fn(cap)
    assert r.capture_fn_wants_weights is True
    drive(r)
    assert len(seen) == 1, "tagged capture fn was called with ids only"
    ids, weights = seen[0]
    assert torch.equal(ids, r._ids)
    assert torch.equal(weights, r._weights)


def test_weights_are_the_pre_eplb_gate_weights_of_this_call():
    """The weights handed over must be this call's routing output, not a
    stale snapshot: a stashed-on-the-router scheme would read last
    step's tensor under graph replay."""
    r = make_router()
    got = []
    cap = lambda ids, w: got.append(w.clone())  # noqa: E731
    cap.wants_topk_weights = True
    r.set_capture_fn(cap)

    drive(r)
    r._weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
    drive(r)

    assert got[0].flatten().tolist() == pytest.approx([0.1, 0.9], abs=1e-6)
    assert got[1].flatten().tolist() == pytest.approx([0.7, 0.3], abs=1e-6), \
        "capture saw a stale weights tensor"


def test_clearing_the_capture_fn_clears_the_opt_in():
    r = make_router()
    cap = lambda ids, w: None  # noqa: E731
    cap.wants_topk_weights = True
    r.set_capture_fn(cap)
    assert r.capture_fn_wants_weights is True
    r.set_capture_fn(None)
    assert r.capture_fn_wants_weights is False
    drive(r)  # must not blow up on a None callback


def test_collector_bound_to_a_real_base_router_records_real_mass():
    """End-to-end on the real router class: bind the collector, route,
    and check that mass is the gate-weight sum and not the hit count."""
    stats_path = (Path(__file__).resolve().parents[2] / "vllm"
                  / "model_executor" / "layers" / "quantization"
                  / "exl3_fungible" / "stats.py")
    spec = importlib.util.spec_from_file_location("fq_stats_br", stats_path)
    stats_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stats_mod)

    r = ToyRouter(torch.tensor([[0.05, 0.95]], dtype=torch.float32),
                  torch.tensor([[3, 6]], dtype=torch.int64))
    c = stats_mod.FqStatsCollector(
        8, window_len=2, window_stride=1, decay=1.0, device="cpu",
        record_mass=True)
    c.bind_router(0, r)
    assert c.mass_is_real(0) is True

    drive(r)
    drive(r)
    assert c.count_buf[0][3].item() == pytest.approx(2.0)
    assert c.count_buf[0][6].item() == pytest.approx(2.0)
    assert c.mass_buf[0][3].item() == pytest.approx(0.10, abs=1e-6)
    assert c.mass_buf[0][6].item() == pytest.approx(1.90, abs=1e-6)
