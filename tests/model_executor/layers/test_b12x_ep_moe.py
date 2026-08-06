# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.b12x_ep_moe as b12x_ep_moe
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


def _parallel_config(**overrides) -> FusedMoEParallelConfig:
    values = dict(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=2,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    values.update(overrides)
    return FusedMoEParallelConfig(**values)


def _fake_ep_experts() -> b12x_ep_moe.B12xEPExperts:
    experts = object.__new__(b12x_ep_moe.B12xEPExperts)
    experts._prepared_ep_expert_map = None
    experts._prepared_experts = None
    experts._source_parameters_released = False
    experts._unit_scale_by_device = {}
    experts.moe_config = SimpleNamespace(in_dtype=torch.bfloat16)
    experts.quant_config = SimpleNamespace(
        quant_dtype="nvfp4",
        weight_quant_dtype="nvfp4",
    )
    return experts


def test_b12x_tp_and_ep_parallel_contracts_are_exclusive() -> None:
    ep_config = _parallel_config()
    tp_config = _parallel_config(use_ep=False, ep_size=1, tp_size=2)

    assert b12x_ep_moe.B12xEPExperts._supports_parallel_config(ep_config)
    assert not B12xExperts._supports_parallel_config(ep_config)
    assert B12xExperts._supports_parallel_config(tp_config)
    assert not b12x_ep_moe.B12xEPExperts._supports_parallel_config(tp_config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dp_size": 2},
        {"pcp_size": 2},
        {"sp_size": 2},
        {"enable_eplb": True},
    ],
)
def test_b12x_ep_rejects_disruptive_parallel_variants(overrides: dict) -> None:
    assert not b12x_ep_moe.B12xEPExperts._supports_parallel_config(
        _parallel_config(**overrides)
    )


def test_b12x_ep_forces_w4a16_and_accepts_expert_map() -> None:
    experts = _fake_ep_experts()

    assert experts._quant_mode() == "w4a16"
    assert experts.supports_expert_map()
    assert not experts._activation_amax_enabled_for_layer()


def test_b12x_ep_rejects_topology_with_empty_expert_ranks() -> None:
    supported, reason = b12x_ep_moe.B12xEPExperts.is_supported_config(
        b12x_ep_moe.B12xEPExperts,
        SimpleNamespace(
            in_dtype=torch.bfloat16,
            num_experts=1,
            moe_parallel_config=_parallel_config(ep_size=2),
        ),
        None,
        None,
        b12x_ep_moe.B12xEPExperts.activation_format(),
    )

    assert not supported
    assert reason == "kernel requires at least one local expert on every EP rank"


def test_b12x_ep_oracles_prefer_ep_specialization_before_tp() -> None:
    import vllm.model_executor.layers.fused_moe.oracle.mxfp4 as mxfp4_oracle
    import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle

    assert nvfp4_oracle.backend_to_kernel_cls(nvfp4_oracle.NvFp4MoeBackend.B12X) == [
        b12x_ep_moe.B12xEPExperts,
        B12xExperts,
    ]
    assert mxfp4_oracle.backend_to_kernel_cls(mxfp4_oracle.Mxfp4MoeBackend.B12X) == [
        b12x_ep_moe.B12xEPExperts,
        B12xExperts,
    ]


def test_b12x_ep_runner_uses_b12x_op_and_late_allreduce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        moe_runner,
        "current_platform",
        SimpleNamespace(is_tpu=lambda: False, is_cpu=lambda: False),
    )
    experts = object.__new__(b12x_ep_moe.B12xEPExperts)
    kernel = SimpleNamespace(
        fused_experts=experts,
        output_is_reduced=lambda: False,
    )
    runner = object.__new__(MoERunner)
    runner._shared_experts = None
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(moe_kernel=kernel)
    )
    runner.moe_config = SimpleNamespace(
        is_sequence_parallel=False,
        tp_size=1,
        ep_size=2,
    )
    reductions = []
    monkeypatch.setattr(
        moe_runner,
        "tensor_model_parallel_all_reduce",
        lambda states: reductions.append(states) or states.add(1),
    )

    forward_entry = runner._select_forward()
    states = torch.zeros(2, 4)
    reduced = runner._maybe_reduce_final_output(states, trunc_size=None)

    assert forward_entry._qualified_op_name == "vllm::b12x_moe_forward"
    assert len(reductions) == 1
    assert reductions[0] is states
    torch.testing.assert_close(reduced, torch.ones_like(states))


def test_b12x_ep_apply_passes_validated_map_to_separate_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _fake_ep_experts()
    prepared = SimpleNamespace(num_experts=2)
    experts._get_or_prepare_experts = MethodType(
        lambda self, **_kwargs: prepared,
        experts,
    )
    experts._b12x_swiglu_params = MethodType(
        lambda self, _activation: (None, None, None),
        experts,
    )
    fake_plan = object()
    scratch = torch.empty(64, dtype=torch.uint8)
    calls = []
    monkeypatch.setattr(
        b12x_ep_moe,
        "_plan_b12x_ep_moe_fp4_scratch",
        lambda **kwargs: calls.append(("plan", kwargs)) or fake_plan,
    )
    monkeypatch.setattr(
        b12x_ep_moe,
        "_workspace2_as_b12x_scratch",
        lambda workspace2, plan: scratch,
    )
    monkeypatch.setattr(
        b12x_ep_moe,
        "_run_b12x_ep_moe_fp4",
        lambda **kwargs: calls.append(("run", kwargs)),
    )

    hidden_states = torch.randn(3, 8, dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    topk_ids = torch.tensor([[0, 1], [2, 3], [1, 2]], dtype=torch.int64)
    topk_weights = torch.full((3, 2), 0.5, dtype=torch.float32)
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(0),
        w2=torch.empty(0),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=4,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=torch.empty(1),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    plan_call = next(kwargs for kind, kwargs in calls if kind == "plan")
    run_call = next(kwargs for kind, kwargs in calls if kind == "run")
    assert plan_call["global_num_experts"] == 4
    assert plan_call["experts"] is prepared
    assert run_call["plan"] is fake_plan
    assert run_call["scratch"] is scratch
    assert run_call["expert_map"].tensor is expert_map
    assert run_call["topk_ids"].dtype == torch.int32


def test_b12x_ep_workspace_plan_uses_global_and_local_expert_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _fake_ep_experts()
    prepared = SimpleNamespace(
        num_experts=3,
        w1_fp4=torch.empty(1),
        plan=SimpleNamespace(quant_modes=frozenset({"w4a16"})),
    )
    experts._prepared_experts = prepared
    experts._b12x_swiglu_params = MethodType(
        lambda self, _activation: (None, None, None),
        experts,
    )
    calls = []
    fake_plan = SimpleNamespace(
        scratch_specs=lambda: [
            SimpleNamespace(dtype=torch.uint8, shape=(101,)),
        ]
    )
    monkeypatch.setattr(
        b12x_ep_moe,
        "_plan_b12x_ep_moe_fp4_scratch",
        lambda **kwargs: calls.append(kwargs) or fake_plan,
    )

    shapes = experts.workspace_shapes(
        M=7,
        N=32,
        K=16,
        topk=2,
        global_num_experts=10,
        local_num_experts=3,
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )

    assert shapes == ((0,), (51,), (7, 16))
    assert calls[0]["global_num_experts"] == 10
    assert calls[0]["experts"] is prepared


def test_b12x_ep_reuses_static_map_and_rejects_mutation() -> None:
    experts = _fake_ep_experts()
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32)

    first = experts._prepare_ep_expert_map(
        expert_map,
        global_num_experts=4,
        local_num_experts=2,
        device=expert_map.device,
    )
    second = experts._prepare_ep_expert_map(
        expert_map,
        global_num_experts=4,
        local_num_experts=2,
        device=expert_map.device,
    )
    assert second is first

    expert_map.copy_(torch.tensor([-1, 0, -1, 1], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="mutated"):
        experts._prepare_ep_expert_map(
            expert_map,
            global_num_experts=4,
            local_num_experts=2,
            device=expert_map.device,
        )


def test_b12x_ep_prepares_map_during_weight_postprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _fake_ep_experts()
    experts._prepared_experts = SimpleNamespace(
        num_experts=2,
        w1_fp4=torch.empty(0),
        plan=SimpleNamespace(quant_modes=frozenset({"w4a16"})),
    )
    parent_calls = []
    monkeypatch.setattr(
        B12xExperts,
        "process_weights_after_loading",
        lambda self, layer: parent_calls.append((self, layer)),
    )
    layer = SimpleNamespace(
        expert_map=torch.tensor([0, -1, 1, -1], dtype=torch.int32),
        global_num_experts=4,
    )

    experts.process_weights_after_loading(layer)

    assert parent_calls == [(experts, layer)]
    assert experts._prepared_ep_expert_map.tensor is layer.expert_map


def test_b12x_ep_warmup_uses_mapped_path_for_each_serving_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experts = _fake_ep_experts()
    prepared = SimpleNamespace(num_experts=2)
    meta = SimpleNamespace(
        w1=torch.empty(0),
        w2=torch.empty(0),
        activation=MoEActivation.SILU,
        activation_name="silu",
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_experts=2,
        k=8,
        n=16,
        topk=2,
        apply_router_weight_on_input=False,
        swiglu_limit=None,
        swiglu_alpha=None,
        swiglu_beta=None,
    )
    experts._warmup_metadata = MethodType(lambda self, _layer: meta, experts)
    experts._get_or_prepare_experts = MethodType(
        lambda self, **_kwargs: prepared,
        experts,
    )
    layer = SimpleNamespace(
        expert_map=torch.tensor([0, -1, 1, -1], dtype=torch.int32),
        global_num_experts=4,
    )
    fake_plan = SimpleNamespace(
        scratch_specs=lambda: [
            SimpleNamespace(dtype=torch.uint8, shape=(64,)),
        ]
    )
    plan_calls = []
    run_calls = []
    monkeypatch.setattr(
        b12x_ep_moe,
        "_plan_b12x_ep_moe_fp4_scratch",
        lambda **kwargs: plan_calls.append(kwargs) or fake_plan,
    )
    monkeypatch.setattr(
        b12x_ep_moe,
        "_run_b12x_ep_moe_fp4",
        lambda **kwargs: run_calls.append(kwargs),
    )

    signature = experts.warmup_dynamic_signature(layer)
    warmed = experts.warmup_dynamic_launches(layer, token_counts=(3, 1, 3))

    assert signature is not None
    assert signature[0] == "replicated-input-ep-w4a16"
    assert warmed == 2
    assert [call["tokens"] for call in plan_calls] == [1, 3]
    assert len(run_calls) == 2
    assert all(call["expert_map"].tensor is layer.expert_map for call in run_calls)
