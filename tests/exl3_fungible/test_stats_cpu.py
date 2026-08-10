# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests for FqStatsCollector (M1) — no GPU required.

Graph-safety of the capture path is validated on-GPU by T1; these tests
pin the host-side contracts: chaining, window roll/decay math, dummy-step
semantics, and the no-host-ops discipline of the capture fn source.
"""
import inspect

import pytest
import torch

try:
    from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
        FqStatsCollector,
    )
except ImportError:  # standalone run against an env without built vllm._C
    import importlib.util
    from pathlib import Path

    _p = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible" / "stats.py")
    _spec = importlib.util.spec_from_file_location("fq_stats_standalone", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    FqStatsCollector = _mod.FqStatsCollector


class FakeRouter:
    def __init__(self):
        self.capture_fn = None

    def set_capture_fn(self, fn):
        self.capture_fn = fn


def make_collector(**kw) -> FqStatsCollector:
    kw.setdefault("device", "cpu")
    kw.setdefault("window_len", 4)
    kw.setdefault("window_stride", 2)
    kw.setdefault("decay", 0.5)
    return FqStatsCollector(8, **kw)


def test_capture_counts_and_chains_previous_fn():
    seen = []
    router = FakeRouter()
    router.capture_fn = lambda ids: seen.append(ids.shape)

    c = make_collector()
    c.bind_router(0, router)
    ids = torch.tensor([[0, 1], [1, 7]], dtype=torch.int32)
    router.capture_fn(ids)

    assert c.count_buf[0].tolist() == [1, 2, 0, 0, 0, 0, 0, 1]
    assert seen == [ids.shape], "previous capture fn must still fire"


def test_window_roll_and_zeroing():
    router = FakeRouter()
    c = make_collector()
    c.bind_router(0, router)
    router.capture_fn(torch.tensor([[3]]))
    c.step()               # step 1: no roll (stride 2)
    assert c.count_buf[0][3] == 1
    c.step()               # step 2: roll
    assert c.count_buf[0][3] == 0, "accumulator zeroed on roll"
    assert c._count_win[0][0][3] == 1


def test_dummy_step_discards_without_recording():
    router = FakeRouter()
    c = make_collector()
    c.bind_router(0, router)
    router.capture_fn(torch.tensor([[5]]))
    c.step(is_dummy=True)
    assert c.count_buf[0].sum() == 0
    c.step()
    assert c._windows_rolled == 0 or c._count_win[0].sum() == 0


def test_decayed_weighting_newest_first():
    router = FakeRouter()
    c = make_collector(window_stride=1)
    c.bind_router(0, router)
    router.capture_fn(torch.tensor([[0]]))   # window 0: expert 0
    c.step()
    router.capture_fn(torch.tensor([[1]]))   # window 1: expert 1
    c.step()
    count, _ = c.decayed(0)
    # newest (expert 1) weight 1.0; older (expert 0) weight λ=0.5
    assert count[1] == pytest.approx(1.0)
    assert count[0] == pytest.approx(0.5)


def test_decay_ring_wraps():
    router = FakeRouter()
    c = make_collector(window_stride=1, window_len=2, decay=0.5)
    c.bind_router(0, router)
    for expert in (0, 1, 2):                 # 3 rolls into a 2-slot ring
        router.capture_fn(torch.tensor([[expert]]))
        c.step()
    count, _ = c.decayed(0)
    assert count[0] == 0, "oldest window evicted"
    assert count[2] == pytest.approx(1.0)
    assert count[1] == pytest.approx(0.5)


def test_summary_shape():
    router = FakeRouter()
    c = make_collector(window_stride=1)
    c.bind_router(0, router)
    c.bind_router(1, FakeRouter())
    router.capture_fn(torch.tensor([[2, 2]]))
    c.step()
    s = c.summary()
    assert set(s["layers"]) == {0, 1}
    assert s["layers"][0]["count"][2] == pytest.approx(2.0)


def test_capture_fn_source_has_no_host_ops():
    """Static discipline check: the capture path must never call host-sync
    ops (graph-safety contract, 01 §2)."""
    c = make_collector()
    c.bind_router(0, FakeRouter())
    src = inspect.getsource(c.make_capture_fn.__code__.co_consts[
        [i for i, k in enumerate(c.make_capture_fn.__code__.co_consts)
         if getattr(k, "co_name", "") == "_capture"][0]])
    for banned in (".item(", ".cpu(", ".tolist(", ".numpy(", "print("):
        assert banned not in src, f"host op {banned} in capture path"
