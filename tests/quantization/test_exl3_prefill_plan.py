# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dispatch-contract tests for the dual-plan rank-sliced EXL3 MoE runtime.

The planned Trellis API and the ExLlamaV3 extension are mocked; these tests
prove backend policy only:

* decode window m in [min, max] binds the decode plan;
* max < m <= capacity binds the prefill plan (block_size_m=64 by default);
* m < min stays on the parity path with chunk-capped staging buffers;
* VLLM_EXL3_PREFILL_TRELLIS=0 restores the single-plan parity behavior
  with full-capacity staging;
* m above planned capacity raises.

CPU-only; no CUDA, sparkinfer, or exllamav3_ext required.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module
from vllm.model_executor.layers.quantization.exl3 import Exl3MoEMethod

HIDDEN = 128
INTERMEDIATE = 128
EXPERTS = 8
TOPK = 4
MAX_BATCHED = 256


class _FakePlan:
    def __init__(self, caps):
        self.caps = caps

    def scratch_specs(self):
        return (
            SimpleNamespace(
                shape=(64,),
                dtype=torch.uint8,
                device=self.caps["device"],
            ),
        )


class _FakeTrellisApi:
    def __init__(self):
        self.planned = []
        self.bound = []

    def Caps(self, **kwargs):
        return kwargs

    def plan(self, caps):
        plan = _FakePlan(caps)
        self.planned.append(caps)
        return plan

    def bind(self, plan, *, scratch, a, weights, topk_weights, topk_ids):
        del scratch, weights, topk_weights, topk_ids
        self.bound.append((plan, int(a.shape[0])))
        return SimpleNamespace(plan=plan, m=int(a.shape[0]))

    def run(self, *, binding):
        return torch.zeros((binding.m, HIDDEN), dtype=torch.float32)


class _FakeExt:
    """Parity extension without exl3_moe_fused: chunk loop only."""

    def __init__(self):
        self.moe_calls = []

    def exl3_moe_max_concurrency(self, device):
        del device
        return 2

    def exl3_moe(self, xh, out32, *args):
        del args
        self.moe_calls.append((int(xh.shape[0]), int(out32.shape[0])))


def _make_layer():
    return SimpleNamespace(
        exl3_max_num_batched_tokens=MAX_BATCHED,
        exl3_hidden_size=HIDDEN,
        exl3_intermediate_size_per_partition=INTERMEDIATE,
        local_num_experts=EXPERTS,
        exl3_trellis_tile_config=(64, 128, 64, 128),
        exl3_trellis_weights=object(),
        exl3_pointer_tables=(),
        exl3_expert_map=torch.arange(EXPERTS, dtype=torch.int64),
    )


def _make_method():
    method = object.__new__(Exl3MoEMethod)
    method.quant_config = SimpleNamespace(
        bits=3.0,
        rank_sliced_metadata={"tp": 4},
    )
    return method


class _Harness:
    def __init__(self, env=None):
        self._env = dict(env or {})
        self._saved_env = {}
        self._saved_capturing = None
        self.api = _FakeTrellisApi()
        self.ext = _FakeExt()

    def __enter__(self):
        for name, value in self._env.items():
            self._saved_env[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved_loaders = (
            exl3_module._load_sparkinfer_trellis,
            exl3_module._load_exl3_ext,
        )
        exl3_module._load_sparkinfer_trellis = lambda: self.api
        exl3_module._load_exl3_ext = lambda: self.ext
        self._saved_capturing = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: False
        self._saved_current_device = torch.cuda.current_device
        torch.cuda.current_device = lambda: 0
        exl3_module._RANK_SLICED_RUNTIMES.clear()
        return self

    def __exit__(self, *exc):
        (
            exl3_module._load_sparkinfer_trellis,
            exl3_module._load_exl3_ext,
        ) = self._saved_loaders
        torch.cuda.is_current_stream_capturing = self._saved_capturing
        torch.cuda.current_device = self._saved_current_device
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        exl3_module._RANK_SLICED_RUNTIMES.clear()
        return False


def _apply(method, layer, m):
    x = torch.zeros((m, HIDDEN), dtype=torch.bfloat16)
    weights = torch.zeros((m, TOPK), dtype=torch.float32)
    ids = torch.zeros((m, TOPK), dtype=torch.int64)
    return method._apply_rank_sliced(layer, x, weights, ids)


def test_dual_plan_construction_and_dispatch():
    with _Harness() as h:
        method = _make_method()
        layer = _make_layer()

        out = _apply(method, layer, 16)
        assert out.dtype == torch.bfloat16 and out.shape == (16, HIDDEN)
        # Two plans: decode (32, block 8) then prefill (capacity, block 64).
        assert [
            (caps["max_tokens"], caps["block_size_m"]) for caps in h.planned_caps()
        ] == [(32, 8), (MAX_BATCHED, 64)]
        assert h.api.bound[-1][0].caps["max_tokens"] == 32

        _apply(method, layer, 200)
        assert h.api.bound[-1][0].caps["max_tokens"] == MAX_BATCHED
        assert h.api.bound[-1][1] == 200
        assert not h.ext.moe_calls

        _apply(method, layer, 2)
        assert h.ext.moe_calls == [(2, 2)]

        runtime = next(iter(exl3_module._RANK_SLICED_RUNTIMES.values()))
        assert runtime["parity_rows"] == 128
        assert runtime["xh"].shape[0] == 128
        assert runtime["token_sorted"].numel() == 128 * TOPK

        # Batches above the scheduler contract must fail before allocating a
        # replacement runtime or a larger Trellis arena during serving.
        runtime_keys = tuple(exl3_module._RANK_SLICED_RUNTIMES)
        runtime_count = len(exl3_module._RANK_SLICED_RUNTIMES)
        plan_count = len(h.planned_caps())
        with pytest.raises(
            ValueError,
            match=rf"m={MAX_BATCHED + 1}, capacity={MAX_BATCHED}",
        ):
            _apply(method, layer, MAX_BATCHED + 1)
        assert tuple(exl3_module._RANK_SLICED_RUNTIMES) == runtime_keys
        assert len(exl3_module._RANK_SLICED_RUNTIMES) == runtime_count
        assert len(h.planned_caps()) == plan_count


def test_prefill_trellis_disabled_restores_parity():
    with _Harness(env={"VLLM_EXL3_PREFILL_TRELLIS": "0"}) as h:
        method = _make_method()
        layer = _make_layer()

        _apply(method, layer, 200)
        # Single decode plan only; large m runs the parity chunk loop.
        assert [
            (caps["max_tokens"], caps["block_size_m"]) for caps in h.planned_caps()
        ] == [(32, 8)]
        assert not h.api.bound
        assert h.ext.moe_calls == [(128, 128), (72, 72)]

        runtime = next(iter(exl3_module._RANK_SLICED_RUNTIMES.values()))
        assert runtime["prefill_plan"] is None
        assert runtime["parity_rows"] == MAX_BATCHED
        assert runtime["xh"].shape[0] == MAX_BATCHED


def test_prefill_block_m_env_override():
    with _Harness(env={"VLLM_EXL3_PREFILL_BLOCK_M": "48"}) as h:
        method = _make_method()
        layer = _make_layer()
        _apply(method, layer, 40)
        assert h.planned_caps()[-1]["block_size_m"] == 48
        assert h.api.bound[-1][0].caps["max_tokens"] == MAX_BATCHED


def test_parity_window_capacity_is_validated_before_planning():
    env = {
        "VLLM_EXL3_TRELLIS_MIN_M": "160",
        "VLLM_EXL3_TRELLIS_MAX_M": "192",
        "VLLM_EXL3_PREFILL_CHUNK": "128",
    }
    with _Harness(env=env) as h:
        with pytest.raises(ValueError, match="cannot cover the EXL3 parity window"):
            _apply(_make_method(), _make_layer(), 16)
        assert not h.planned_caps()


def test_disabled_prefill_plan_keeps_full_parity_capacity():
    env = {
        "VLLM_EXL3_TRELLIS_MIN_M": "160",
        "VLLM_EXL3_TRELLIS_MAX_M": "192",
        "VLLM_EXL3_PREFILL_CHUNK": "128",
        "VLLM_EXL3_PREFILL_TRELLIS": "0",
    }
    with _Harness(env=env):
        _apply(_make_method(), _make_layer(), 159)
        runtime = next(iter(exl3_module._RANK_SLICED_RUNTIMES.values()))
        assert runtime["parity_rows"] == MAX_BATCHED


def test_parity_path_guarded_against_capture():
    with _Harness() as h:
        method = _make_method()
        layer = _make_layer()
        # Plan eagerly, then flip into "capturing" state.
        _apply(method, layer, 16)
        torch.cuda.is_current_stream_capturing = lambda: True
        # Both trellis plans stay capture-safe.
        _apply(method, layer, 16)
        _apply(method, layer, 200)
        assert h.api.bound[-1][1] == 200
        # The eager parity path must refuse to be recorded.
        try:
            _apply(method, layer, 2)
        except RuntimeError as err:
            assert "capture" in str(err)
        else:
            raise AssertionError("parity path must raise during capture")


def _install_planned_caps_helper():
    def planned_caps(self):
        return self.api.planned

    _Harness.planned_caps = planned_caps


_install_planned_caps_helper()


if __name__ == "__main__":
    test_dual_plan_construction_and_dispatch()
    test_prefill_trellis_disabled_restores_parity()
    test_prefill_block_m_env_override()
    test_parity_path_guarded_against_capture()
    print("EXL3_PREFILL_PLAN_TESTS_OK")
