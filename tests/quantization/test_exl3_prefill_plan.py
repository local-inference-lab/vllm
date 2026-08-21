# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dispatch-contract tests for the dual-plan rank-sliced EXL3 MoE runtime.

The planned Trellis API and the ExLlamaV3 extension are mocked; these tests
prove backend policy only:

* decode window m in [min, max] binds the decode plan;
* max < m <= scheduler capacity binds capacity-bounded prefill slices;
* the opt-in prefill capacity defaults to the scheduler capacity;
* m < min stays on the parity path with chunk-capped staging buffers;
* VLLM_EXL3_PREFILL_TRELLIS=0 restores the single-plan parity behavior
  with full-capacity staging;
* m above planned capacity raises.

CPU-only; no CUDA, b12x, or exllamav3_ext required.
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


class _FakeFusedMoeApi:
    def __init__(self):
        self.planned = []
        self.bound = []
        self.routed = []

    def Caps(self, **kwargs):
        return kwargs

    def plan(self, caps):
        plan = _FakePlan(caps)
        self.planned.append(caps)
        return plan

    def bind(
        self,
        plan,
        *,
        scratch,
        a,
        experts,
        topk_weights,
        topk_ids,
        route_expert_map=None,
        output_expert_map=None,
        output=None,
    ):
        del scratch, experts
        self.bound.append((plan, int(a.shape[0])))
        self.routed.append((topk_weights.clone(), topk_ids.clone()))
        return SimpleNamespace(
            plan=plan,
            a=a,
            route_expert_map=route_expert_map,
            output_expert_map=output_expert_map,
            output=output,
        )

    def run(self, *, binding):
        if binding.route_expert_map is not None:
            tier_output = binding.a.to(torch.float32).mul(0.5)
            if binding.output is not None:
                binding.output.copy_(tier_output)
                return binding.output
            return tier_output
        return binding.a.to(torch.float32)


class _FakeMixedTrellisApi:
    def __init__(self):
        self.compiled = []
        self.compiled3 = []
        self.routed = []
        self.calls = []
        self.calls3 = []
        self.bindings = []
        self.bindings3 = []

    def max_packed_route_slots(self, routed_rows, block_size, experts):
        del block_size, experts
        return routed_rows

    def compile_mixed_trellis(self, **kwargs):
        launch = SimpleNamespace(**kwargs)
        self.compiled.append(launch)
        return launch

    def compile_mixed_trellis3(self, **kwargs):
        launch = SimpleNamespace(**kwargs)
        self.compiled3.append(launch)
        return launch

    def make_mixed_trellis_buffers(self, launch, **kwargs):
        del kwargs
        return SimpleNamespace(scratch=torch.empty(launch.size_m, dtype=torch.uint8))

    def make_mixed_trellis3_buffers(self, launch, **kwargs):
        del kwargs
        return SimpleNamespace(scratch=torch.empty(launch.size_m, dtype=torch.uint8))

    def bind_mixed_trellis(self, *args, **kwargs):
        tier0, tier1 = args[:2]
        launch = args[5]
        binding = SimpleNamespace(
            tiers=(tier0, tier1),
            launch=launch,
            kwargs=kwargs,
        )
        self.bindings.append(binding)
        return binding

    def run_bound_mixed_trellis(self, x, topk_weights, topk_ids, binding, buffers):
        del buffers
        self.calls.append((int(x.shape[0]), *binding.tiers))
        self.routed.append((topk_weights.clone(), topk_ids.clone()))
        return x.to(torch.float32)

    def bind_mixed_trellis3(self, *args, **kwargs):
        tier0, tier1, tier2 = args[:3]
        launch = args[6]
        binding = SimpleNamespace(
            tiers=(tier0, tier1, tier2),
            launch=launch,
            kwargs=kwargs,
        )
        self.bindings3.append(binding)
        return binding

    def run_bound_mixed_trellis3(self, x, topk_weights, topk_ids, binding, buffers):
        del buffers
        self.calls3.append((int(x.shape[0]), *binding.tiers))
        self.routed.append((topk_weights.clone(), topk_ids.clone()))
        return x.to(torch.float32)


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
        exl3_trellis_weights=SimpleNamespace(plan=object()),
        exl3_pointer_tables=(),
        exl3_expert_map=torch.arange(EXPERTS, dtype=torch.int64),
    )


def _make_mixed_layer():
    layer = _make_layer()
    layer.exl3_mixed_bitrate = True
    layer.exl3_mixed_trellis = {
        "tiers": (object(), object()),
        "prefill_tiers": (object(), object()),
        "tier_ids": ((0, 1, 2, 3), (4, 5, 6, 7)),
        "tier_bits": (3, 4),
        "trellis_codebook": "mcg",
        "global_to_combined": object(),
        "descriptor_map": object(),
        "rotations": object(),
        "broadcast_suh": False,
        "broadcast_svh": False,
        "tile_config": (64, 128, 64, 128),
        "prefill_tile_config": (128, 128, 128, 128),
    }
    return layer


def _make_mixed3_layer():
    layer = _make_layer()
    layer.exl3_mixed_bitrate = True
    layer.exl3_mixed_trellis = {
        "tiers": (object(), object(), object()),
        "prefill_tiers": (object(), object(), object()),
        "tier_ids": ((0, 1, 2), (3, 4, 5), (6, 7)),
        "tier_bits": (3, 4, 5),
        "trellis_codebook": "mcg",
        "global_to_combined": object(),
        "descriptor_map": object(),
        "rotations": object(),
        "broadcast_suh": False,
        "broadcast_svh": False,
        "tile_config": (64, 128, 64, 128),
        "prefill_tile_config": (128, 128, 128, 128),
    }
    return layer


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
        self.api = _FakeFusedMoeApi()
        self.ext = _FakeExt()
        self.mixed_api = _FakeMixedTrellisApi()

    def __enter__(self):
        for name, value in self._env.items():
            self._saved_env[name] = os.environ.get(name)
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved_loaders = (
            exl3_module._load_b12x_fused_moe,
            exl3_module._load_b12x_mixed_trellis,
            exl3_module._load_exl3_ext,
        )
        exl3_module._load_b12x_fused_moe = lambda: self.api
        exl3_module._load_b12x_mixed_trellis = lambda: self.mixed_api
        exl3_module._load_exl3_ext = lambda: self.ext
        self._saved_capturing = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: False
        self._saved_current_device_index = torch.accelerator.current_device_index
        torch.accelerator.current_device_index = lambda: 0
        self._saved_device_properties = torch.cuda.get_device_properties
        torch.cuda.get_device_properties = lambda device: SimpleNamespace(
            multi_processor_count=1,
            shared_memory_per_block_optin=1,
        )
        exl3_module._RANK_SLICED_RUNTIMES.clear()
        exl3_module._MIXED_TRELLIS_RUNTIMES.clear()
        exl3_module._MIXED_TRELLIS_BUFFERS.clear()
        return self

    def __exit__(self, *exc):
        (
            exl3_module._load_b12x_fused_moe,
            exl3_module._load_b12x_mixed_trellis,
            exl3_module._load_exl3_ext,
        ) = self._saved_loaders
        torch.cuda.is_current_stream_capturing = self._saved_capturing
        torch.accelerator.current_device_index = self._saved_current_device_index
        torch.cuda.get_device_properties = self._saved_device_properties
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        exl3_module._RANK_SLICED_RUNTIMES.clear()
        exl3_module._MIXED_TRELLIS_RUNTIMES.clear()
        exl3_module._MIXED_TRELLIS_BUFFERS.clear()
        return False

    def planned_caps(self):
        return self.api.planned


def _apply(method, layer, m):
    x = torch.zeros((m, HIDDEN), dtype=torch.bfloat16)
    weights = torch.zeros((m, TOPK), dtype=torch.float32)
    ids = torch.zeros((m, TOPK), dtype=torch.int64)
    return method._apply_rank_sliced(layer, x, weights, ids)


def test_opt_in_prefill_capacity_slices_dispatch():
    with _Harness(env={"VLLM_EXL3_PREFILL_CAPACITY": "128"}) as h:
        method = _make_method()
        layer = _make_layer()

        out = _apply(method, layer, 16)
        assert out.dtype == torch.bfloat16 and out.shape == (16, HIDDEN)
        # Two plans: decode (32, block 8), then bounded prefill (block 64).
        assert [
            (caps["max_tokens"], caps["w4a16_block_size_m"])
            for caps in h.planned_caps()
        ] == [(32, 8), (128, 64)]
        assert all(caps["route_num_experts"] == 0 for caps in h.planned_caps())
        assert h.api.bound[-1][0].caps["max_tokens"] == 32

        _apply(method, layer, 200)
        assert all(bound[0].caps["max_tokens"] == 128 for bound in h.api.bound[-2:])
        assert [bound[1] for bound in h.api.bound[-2:]] == [128, 72]
        assert not h.ext.moe_calls

        _apply(method, layer, 2)
        assert h.api.bound[-1][0].caps["max_tokens"] == 32
        assert h.api.bound[-1][1] == 2
        assert not h.ext.moe_calls

        runtime = next(iter(exl3_module._RANK_SLICED_RUNTIMES.values()))
        assert runtime["parity_rows"] == 128
        assert runtime["xh"].shape[0] == 128
        assert runtime["token_sorted"].numel() == 128 * TOPK
        assert runtime["prefill_capacity"] == 128

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


@pytest.mark.parametrize(
    ("rows", "expected_slice_rows"),
    ((127, [127]), (128, [128]), (129, [128, 1])),
)
def test_prefill_capacity_boundaries_preserve_rows_and_routing(
    rows, expected_slice_rows
):
    with _Harness(env={"VLLM_EXL3_PREFILL_CAPACITY": "128"}) as h:
        method = _make_method()
        layer = _make_layer()
        x = torch.arange(rows, dtype=torch.bfloat16).unsqueeze(1).expand(-1, HIDDEN)
        weights = torch.arange(rows * TOPK, dtype=torch.float32).reshape(rows, TOPK)
        ids = torch.arange(rows * TOPK, dtype=torch.int64).reshape(rows, TOPK)
        ids = ids.remainder(EXPERTS)

        out = method._apply_rank_sliced(layer, x, weights, ids)

        assert [bound[1] for bound in h.api.bound] == expected_slice_rows
        assert torch.equal(out, x)
        assert torch.equal(torch.cat([route[0] for route in h.api.routed]), weights)
        assert torch.equal(torch.cat([route[1] for route in h.api.routed]), ids)
        assert all(bound[0] is h.api.bound[0][0] for bound in h.api.bound)


def test_default_prefill_capacity_keeps_single_dispatch():
    with _Harness() as h:
        out = _apply(_make_method(), _make_layer(), 200)

        assert h.planned_caps()[-1]["max_tokens"] == MAX_BATCHED
        assert [bound[1] for bound in h.api.bound] == [200]
        assert out.shape == (200, HIDDEN)


def test_mixed_prefill_capacity_slices_rows_and_routing():
    with _Harness(env={"VLLM_EXL3_PREFILL_CAPACITY": "128"}) as h:
        method = _make_method()
        layer = _make_mixed_layer()
        rows = 200
        x = torch.arange(rows, dtype=torch.bfloat16).unsqueeze(1).expand(-1, HIDDEN)
        weights = torch.arange(rows * TOPK, dtype=torch.float32).reshape(rows, TOPK)
        ids = torch.arange(rows * TOPK, dtype=torch.int64).reshape(rows, TOPK)
        ids = ids.remainder(EXPERTS)

        out = method._apply_rank_sliced(layer, x, weights, ids)

        assert [launch.size_m for launch in h.mixed_api.compiled] == [32, 128]
        assert [launch.moe_block_size for launch in h.mixed_api.compiled] == [8, 64]
        assert all(
            launch.route_num_experts == EXPERTS for launch in h.mixed_api.compiled
        )
        assert h.mixed_api.compiled[1].force_tile_config == (128, 128, 128, 128)
        assert not h.planned_caps()
        assert not h.api.bound
        assert [call[0] for call in h.mixed_api.calls] == [128, 72]
        assert all(
            call[1:] == layer.exl3_mixed_trellis["prefill_tiers"]
            for call in h.mixed_api.calls
        )
        assert torch.equal(out, x)
        assert torch.equal(
            torch.cat([route[0] for route in h.mixed_api.routed]), weights
        )
        assert torch.equal(torch.cat([route[1] for route in h.mixed_api.routed]), ids)


def test_three_tier_prefill_uses_native_b12x_dispatch():
    with _Harness(env={"VLLM_EXL3_PREFILL_CAPACITY": "128"}) as h:
        method = _make_method()
        layer = _make_mixed3_layer()
        rows = 200
        x = torch.arange(rows, dtype=torch.bfloat16).unsqueeze(1).expand(-1, HIDDEN)
        weights = torch.arange(rows * TOPK, dtype=torch.float32).reshape(rows, TOPK)
        ids = torch.arange(rows * TOPK, dtype=torch.int64).reshape(rows, TOPK)
        ids = ids.remainder(EXPERTS)

        out = method._apply_rank_sliced(layer, x, weights, ids)

        assert not h.mixed_api.compiled
        assert [launch.size_m for launch in h.mixed_api.compiled3] == [32, 128]
        assert all(launch.trellis_codebook == "mcg" for launch in h.mixed_api.compiled3)
        assert all(
            launch.route_num_experts == EXPERTS for launch in h.mixed_api.compiled3
        )
        assert [call[0] for call in h.mixed_api.calls3] == [128, 72]
        assert all(
            call[1:] == layer.exl3_mixed_trellis["prefill_tiers"]
            for call in h.mixed_api.calls3
        )
        assert torch.equal(out, x)
        assert torch.equal(
            torch.cat([route[0] for route in h.mixed_api.routed]), weights
        )
        assert torch.equal(torch.cat([route[1] for route in h.mixed_api.routed]), ids)


def test_two_tier_layers_share_launches_but_bind_fixed_artifacts_separately():
    with _Harness(env={"VLLM_EXL3_PREFILL_CAPACITY": "128"}) as h:
        method = _make_method()
        first_layer = _make_mixed_layer()
        second_layer = _make_mixed_layer()
        x = torch.zeros((16, HIDDEN), dtype=torch.bfloat16)
        weights = torch.ones((16, TOPK), dtype=torch.float32)
        ids = torch.zeros((16, TOPK), dtype=torch.int64)

        first_runtime = method._mixed_rank_sliced_runtime(first_layer, x, ids)
        first_bindings = dict(first_layer.exl3_mixed_trellis["runtime_bindings"])
        second_runtime = method._mixed_rank_sliced_runtime(second_layer, x, ids)
        second_bindings = dict(second_layer.exl3_mixed_trellis["runtime_bindings"])

        assert first_runtime is second_runtime
        assert len(h.mixed_api.compiled) == 2
        assert len(h.mixed_api.bindings) == 4
        assert first_bindings.keys() == second_bindings.keys()
        for launch_id in first_bindings:
            first_binding = first_bindings[launch_id]
            second_binding = second_bindings[launch_id]
            assert first_binding is not second_binding
            assert first_binding.tiers != second_binding.tiers

        method._apply_rank_sliced(first_layer, x, weights, ids)
        assert len(h.mixed_api.bindings) == 4
        assert h.mixed_api.calls[-1][1:] == first_layer.exl3_mixed_trellis["tiers"]


def test_three_tier_layers_share_launches_but_bind_fixed_artifacts_separately():
    with _Harness(env={"VLLM_EXL3_PREFILL_CAPACITY": "128"}) as h:
        method = _make_method()
        first_layer = _make_mixed3_layer()
        second_layer = _make_mixed3_layer()
        x = torch.zeros((16, HIDDEN), dtype=torch.bfloat16)
        weights = torch.ones((16, TOPK), dtype=torch.float32)
        ids = torch.zeros((16, TOPK), dtype=torch.int64)

        first_runtime = method._mixed_rank_sliced_runtime(first_layer, x, ids)
        first_bindings = dict(first_layer.exl3_mixed_trellis["runtime_bindings"])
        second_runtime = method._mixed_rank_sliced_runtime(second_layer, x, ids)
        second_bindings = dict(second_layer.exl3_mixed_trellis["runtime_bindings"])

        assert first_runtime is second_runtime
        assert len(h.mixed_api.compiled3) == 2
        assert len(h.mixed_api.bindings3) == 4
        assert first_bindings.keys() == second_bindings.keys()
        for launch_id in first_bindings:
            first_binding = first_bindings[launch_id]
            second_binding = second_bindings[launch_id]
            assert first_binding is not second_binding
            assert first_binding.tiers != second_binding.tiers

        method._apply_rank_sliced(first_layer, x, weights, ids)
        assert len(h.mixed_api.bindings3) == 4
        assert h.mixed_api.calls3[-1][1:] == first_layer.exl3_mixed_trellis["tiers"]


def test_mixed_runtime_policy_is_resolved_once():
    calls = 0

    def properties(device):
        nonlocal calls
        del device
        calls += 1
        return SimpleNamespace(
            major=12,
            multi_processor_count=1,
            shared_memory_per_block_optin=1,
        )

    with _Harness() as h:
        torch.cuda.get_device_properties = properties
        method = _make_method()
        layer = _make_mixed_layer()
        x = torch.zeros((16, HIDDEN), dtype=torch.bfloat16)
        ids = torch.zeros((16, TOPK), dtype=torch.int64)

        first = method._mixed_rank_sliced_runtime(layer, x, ids)
        second = method._mixed_rank_sliced_runtime(layer, x, ids)

        assert first is second
        assert calls == 1
        assert len(h.mixed_api.compiled) == 2


def test_mixed_buffer_pool_is_scoped_to_target_or_draft_owner():
    with _Harness() as h:
        launch = SimpleNamespace(
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            size_m=32,
            max_m_blocks=4,
            tier0_num_experts=4,
            tier1_num_experts=4,
            rotation_input_dtype="bf16",
            route_ids_dtype=torch.int64,
        )
        args = (
            h.mixed_api,
            launch,
            torch.device("cpu"),
            1,
            32,
            8,
            (64, 128, 64, 128),
            TOPK,
        )
        target = exl3_module._shared_mixed_buffers((1, False), *args)
        target_again = exl3_module._shared_mixed_buffers((1, False), *args)
        draft = exl3_module._shared_mixed_buffers((1, True), *args)

        assert target_again is target
        assert draft is not target

        launch3 = SimpleNamespace(
            **{
                **vars(launch),
                "tier2_num_experts": 1,
            }
        )
        three_tier = exl3_module._shared_mixed_buffers(
            (1, False), h.mixed_api, launch3, *args[2:]
        )
        assert three_tier is not target


def test_mixed_runtime_forwards_shared_h_broadcast_contract():
    with _Harness() as h:
        method = _make_method()
        layer = _make_mixed_layer()
        layer.exl3_mixed_trellis["broadcast_suh"] = True
        layer.exl3_mixed_trellis["broadcast_svh"] = True
        x = torch.zeros((16, HIDDEN), dtype=torch.bfloat16)
        ids = torch.zeros((16, TOPK), dtype=torch.int64)

        method._mixed_rank_sliced_runtime(layer, x, ids)

        assert len(h.mixed_api.compiled) == 2
        assert all(launch.broadcast_suh is True for launch in h.mixed_api.compiled)
        assert all(launch.broadcast_svh is True for launch in h.mixed_api.compiled)


def test_prefill_capacity_cannot_exceed_scheduler_bound():
    env = {"VLLM_EXL3_PREFILL_CAPACITY": str(MAX_BATCHED + 1)}
    with _Harness(env=env) as h:
        with pytest.raises(ValueError, match="cannot exceed max_num_batched_tokens"):
            _apply(_make_method(), _make_layer(), 16)
        assert not h.planned_caps()


def test_prefill_trellis_disabled_restores_parity():
    with _Harness(env={"VLLM_EXL3_PREFILL_TRELLIS": "0"}) as h:
        method = _make_method()
        layer = _make_layer()

        _apply(method, layer, 200)
        # Single decode plan only; large m runs the parity chunk loop.
        assert [
            (caps["max_tokens"], caps["w4a16_block_size_m"])
            for caps in h.planned_caps()
        ] == [(32, 8)]
        assert not h.api.bound
        assert h.ext.moe_calls == [(128, 128), (72, 72)]

        runtime = next(iter(exl3_module._RANK_SLICED_RUNTIMES.values()))
        assert runtime["prefill_plan"] is None
        assert runtime["parity_rows"] == MAX_BATCHED


def test_prefill_block_m_env_override():
    env = {
        "VLLM_EXL3_PREFILL_BLOCK_M": "48",
        "VLLM_EXL3_PREFILL_CAPACITY": "128",
    }
    with _Harness(env=env) as h:
        method = _make_method()
        layer = _make_layer()
        _apply(method, layer, 40)
        assert h.planned_caps()[-1]["w4a16_block_size_m"] == 48
        assert h.api.bound[-1][0].caps["max_tokens"] == 128


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


def test_explicit_parity_path_guarded_against_capture():
    env = {
        "VLLM_EXL3_TRELLIS_MIN_M": "4",
        "VLLM_EXL3_PREFILL_CAPACITY": "128",
    }
    with _Harness(env=env) as h:
        method = _make_method()
        layer = _make_layer()
        # Plan eagerly, then flip into "capturing" state.
        _apply(method, layer, 16)
        torch.cuda.is_current_stream_capturing = lambda: True
        # Both trellis plans stay capture-safe.
        _apply(method, layer, 16)
        _apply(method, layer, 200)
        assert [bound[1] for bound in h.api.bound[-2:]] == [128, 72]
        # The eager parity path must refuse to be recorded.
        with pytest.raises(RuntimeError, match="capture"):
            _apply(method, layer, 2)


if __name__ == "__main__":
    test_opt_in_prefill_capacity_slices_dispatch()
    test_prefill_trellis_disabled_restores_parity()
    test_prefill_block_m_env_override()
    test_explicit_parity_path_guarded_against_capture()
    print("EXL3_PREFILL_PLAN_TESTS_OK")
