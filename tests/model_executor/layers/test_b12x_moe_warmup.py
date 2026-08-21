# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.b12x_moe as b12x_moe
import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 as ct_mxfp4  # noqa: E501
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


class _FakePlan:
    def __init__(self, execution: object = "dynamic-m16") -> None:
        self.launch_plan = SimpleNamespace(
            implementation="dynamic",
            execution=execution,
        )

    @staticmethod
    def scratch_specs():
        return [SimpleNamespace(dtype=torch.uint8, shape=(64,))]


def _make_fake_prepared_experts(
    *,
    quant_mode: str = "nvfp4",
    activation: str = "swigluoai_uninterleave",
    num_experts: int = 8,
    hidden_size: int = 64,
    intermediate_size: int = 16,
    plan=None,
):
    if plan is None:
        plan = SimpleNamespace(
            quant_modes=frozenset({quant_mode}),
            io_dtype="bfloat16",
            activation=activation,
            discards_source_parameters=False,
        )
    return SimpleNamespace(
        plan=plan,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation=activation,
        w1_fp4=torch.empty(1, dtype=torch.uint8),
    )


def _make_fake_b12x_experts() -> b12x_moe.B12xExperts:
    num_experts = 8
    experts = object.__new__(b12x_moe.B12xExperts)
    experts.moe_config = SimpleNamespace(
        in_dtype=torch.bfloat16,
        experts_per_token=4,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        swiglu_limit=7.0,
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
    )
    experts.quant_config = SimpleNamespace(
        quant_dtype="nvfp4",
        weight_quant_dtype="nvfp4",
        gemm1_clamp_limit=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        a1_gscale=torch.ones(num_experts, dtype=torch.float32),
        a2_gscale=torch.ones(num_experts, dtype=torch.float32),
        w1_scale=torch.ones(num_experts, 16, 16, dtype=torch.float32),
        w2_scale=torch.ones(num_experts, 16, 16, dtype=torch.float32),
        g1_alphas=torch.ones(num_experts, dtype=torch.float32),
        g2_alphas=torch.ones(num_experts, dtype=torch.float32),
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        per_act_token_quant=False,
        per_out_ch_quant=False,
        w1_zp=None,
        w2_zp=None,
        w1_bias=None,
        w2_bias=None,
    )
    experts._prepared_experts = _make_fake_prepared_experts()
    experts._source_parameters_released = False
    experts._unit_scale_by_device = {}
    experts._activation_amax_base_num_layers = None
    experts._activation_amax_state_key = None
    experts._activation_amax_layer_idx = None
    return experts


def _make_fake_moe_runner(fused_experts: object) -> MoERunner:
    runner = object.__new__(MoERunner)
    runner._shared_experts = None
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(fused_experts=fused_experts)
        )
    )
    return runner


def test_b12x_moe_runner_uses_functional_custom_op(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_runner,
        "current_platform",
        SimpleNamespace(is_tpu=lambda: False, is_cpu=lambda: False),
    )

    runner = _make_fake_moe_runner(object.__new__(b12x_moe.B12xExperts))

    forward_entry = runner._select_forward()

    assert forward_entry._qualified_op_name == "vllm::b12x_moe_forward"


def test_b12x_moe_runner_uses_shared_input_reuse_custom_op(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_runner,
        "current_platform",
        SimpleNamespace(is_tpu=lambda: False, is_cpu=lambda: False),
    )

    runner = _make_fake_moe_runner(object.__new__(b12x_moe.B12xExperts))

    forward_entry = runner._select_shared_input_reuse_forward()

    assert forward_entry._qualified_op_name == (
        "vllm::b12x_moe_forward_shared_input_reuse"
    )


def test_non_b12x_moe_runner_keeps_generic_custom_op(monkeypatch) -> None:
    monkeypatch.setattr(
        moe_runner,
        "current_platform",
        SimpleNamespace(is_tpu=lambda: False, is_cpu=lambda: False),
    )

    runner = _make_fake_moe_runner(object())

    forward_entry = runner._select_forward()

    assert forward_entry._qualified_op_name == "vllm::moe_forward"


def test_non_b12x_moe_runner_keeps_generic_shared_input_reuse_op(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        moe_runner,
        "current_platform",
        SimpleNamespace(is_tpu=lambda: False, is_cpu=lambda: False),
    )

    runner = _make_fake_moe_runner(object())

    forward_entry = runner._select_shared_input_reuse_forward()

    assert forward_entry._qualified_op_name == "vllm::moe_forward_shared_input_reuse"


def test_b12x_moe_custom_op_matches_generic_mutation_contract() -> None:
    b12x_schema = str(torch.ops.vllm.b12x_moe_forward.default._schema)
    b12x_shared_schema = str(torch.ops.vllm.b12x_moe_forward_shared.default._schema)
    generic_schema = str(torch.ops.vllm.moe_forward.default._schema)
    generic_shared_schema = str(torch.ops.vllm.moe_forward_shared.default._schema)
    b12x_reuse_schema = str(
        torch.ops.vllm.b12x_moe_forward_shared_input_reuse.default._schema
    )
    generic_reuse_schema = str(
        torch.ops.vllm.moe_forward_shared_input_reuse.default._schema
    )

    assert "Tensor(a0!) hidden_states" in generic_schema
    assert "Tensor(a0!) hidden_states" in generic_shared_schema
    assert "Tensor(a0!) hidden_states" in b12x_schema
    assert "Tensor(a0!) hidden_states" in b12x_shared_schema
    assert "Tensor(a0!) hidden_states" in generic_reuse_schema
    assert "!)? shared_experts_input" in generic_reuse_schema
    assert "Tensor(a0!) hidden_states" in b12x_reuse_schema
    assert "!)? shared_experts_input" in b12x_reuse_schema


def test_b12x_moe_run_binds_only_the_prepared_expert_owner(monkeypatch) -> None:
    b12x_fused_moe = pytest.importorskip("b12x.moe.fused_moe")

    execute_calls = []

    class Plan:
        bind_kwargs = None

        def bind(self, **kwargs):
            self.bind_kwargs = kwargs
            return "binding"

    monkeypatch.setattr(
        b12x_fused_moe,
        "run",
        lambda *, binding: execute_calls.append(binding),
    )
    plan = Plan()
    experts = object()
    a = torch.empty(2, 16, dtype=torch.bfloat16)
    output = torch.empty_like(a)
    topk_ids = torch.zeros(2, 4, dtype=torch.int32)
    topk_weights = torch.full((2, 4), 0.25, dtype=torch.float32)
    scratch = torch.empty(64, dtype=torch.uint8)

    binding = b12x_moe._run_b12x_moe_fp4(
        a=a,
        experts=experts,
        output=output,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        input_scales_static=True,
        unit_scale_contract=False,
        plan=plan,
        scratch=scratch,
    )

    assert binding == "binding"
    assert execute_calls == ["binding"]
    assert plan.bind_kwargs == {
        "scratch": scratch,
        "a": a,
        "experts": experts,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "output": output,
        "input_scales_static": True,
        "unit_scale_contract": False,
        "activation_amax": None,
        "layer_idx": None,
        "route_expert_map": None,
    }


def test_b12x_source_release_leaves_prepared_owner_as_storage_owner() -> None:
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.empty(8, 32, 16, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.empty(8, 64, 8, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.empty(8, 32, 2, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.empty(8, 64, 1, dtype=torch.uint8),
        requires_grad=False,
    )
    experts = _make_fake_b12x_experts()
    experts.quant_config._w1 = SimpleNamespace(scale=layer.w13_weight_scale)
    experts.quant_config._w2 = SimpleNamespace(scale=layer.w2_weight_scale)
    owner = SimpleNamespace(
        w1_fp4=layer.w13_weight,
        w2_fp4=layer.w2_weight,
        w1_blockscale=layer.w13_weight_scale,
        w2_blockscale=layer.w2_weight_scale,
    )
    experts._prepared_experts = owner
    owner_ptrs = tuple(
        tensor.untyped_storage().data_ptr()
        for tensor in (
            owner.w1_fp4,
            owner.w2_fp4,
            owner.w1_blockscale,
            owner.w2_blockscale,
        )
    )

    experts._release_source_parameters(layer)
    experts._release_source_parameters(layer)

    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight.numel() == 0
    assert layer.w13_weight_scale.numel() == 0
    assert layer.w2_weight_scale.numel() == 0
    assert experts.quant_config._w1.scale.numel() == 0
    assert experts.quant_config._w2.scale.numel() == 0
    assert (
        tuple(
            tensor.untyped_storage().data_ptr()
            for tensor in (
                owner.w1_fp4,
                owner.w2_fp4,
                owner.w1_blockscale,
                owner.w2_blockscale,
            )
        )
        == owner_ptrs
    )


def test_compressed_tensors_mxfp4_releases_only_superseded_scales() -> None:
    layer = torch.nn.Module()
    aliased_scale = torch.nn.Parameter(
        torch.empty(8, 32, 2, dtype=torch.uint8),
        requires_grad=False,
    )
    released_scale = torch.nn.Parameter(
        torch.empty(8, 64, 1, dtype=torch.uint8),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", aliased_scale)
    layer.register_parameter("w2_weight_scale", released_scale)
    representation = SimpleNamespace(w13_scale=aliased_scale)
    prepared = SimpleNamespace(
        representation_for=lambda quant_mode: representation,
    )
    fused_experts = SimpleNamespace(
        _lookup_prepared_experts=lambda: prepared,
    )
    method = object.__new__(ct_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod)
    method.moe_kernel = SimpleNamespace(fused_experts=fused_experts)

    method._release_superseded_scale_storage(layer)

    assert layer.w13_weight_scale is aliased_scale
    assert layer.w13_weight_scale.numel() > 0
    assert layer.w2_weight_scale is released_scale
    assert layer.w2_weight_scale.numel() == 0


def test_compressed_tensors_situ_mxfp4_selects_b12x(monkeypatch) -> None:
    class FakeB12xExperts:
        pass

    monkeypatch.setattr(
        ct_mxfp4.CutlassExpertsMxfp4,
        "_supports_current_device",
        lambda: False,
    )
    monkeypatch.setattr(
        ct_mxfp4,
        "select_mxfp4_moe_backend",
        lambda moe: (ct_mxfp4.Mxfp4MoeBackend.B12X, FakeB12xExperts),
    )

    method = ct_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod(
        SimpleNamespace(activation=MoEActivation.SITU)
    )

    assert method.mxfp4_backend == ct_mxfp4.Mxfp4MoeBackend.B12X
    assert method.experts_cls is FakeB12xExperts
    assert not method.use_cutlass_mxfp4


def test_b12x_moe_warmup_uses_minimax_swiglu_params(monkeypatch) -> None:
    plan_calls = []
    run_calls = []

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return _FakePlan()

    def fake_run(**kwargs):
        run_calls.append(kwargs)

    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)
    monkeypatch.setattr(
        b12x_moe,
        "_plan_b12x_moe_execution",
        lambda **kwargs: _FakePlan().launch_plan,
    )
    monkeypatch.setattr(b12x_moe, "_run_b12x_moe_fp4", fake_run)
    monkeypatch.setattr(
        b12x_moe,
        "_dynamic_moe_warmup_tokens",
        lambda *, topk, quant_mode, requested_tokens: 7,
    )

    experts = _make_fake_b12x_experts()
    layer = SimpleNamespace(
        w13_weight=torch.empty(8, 32, 32, dtype=torch.uint8),
        w2_weight=torch.empty(8, 64, 8, dtype=torch.uint8),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        apply_router_weight_on_input=False,
    )

    warmed = experts.warmup_dynamic_launches(layer, token_counts=(3,))

    assert warmed == 1
    assert len(plan_calls) == 1
    assert len(run_calls) == 1
    assert plan_calls[0]["tokens"] == 7
    assert plan_calls[0]["quant_mode"] == "nvfp4"
    assert plan_calls[0]["swiglu_limit"] == 7.0
    assert plan_calls[0]["swiglu_alpha"] == 1.702
    assert plan_calls[0]["swiglu_beta"] == 1.0
    assert plan_calls[0]["experts"] is experts._prepared_experts

    assert run_calls[0]["a"].shape == (7, 64)
    assert run_calls[0]["output"].shape == (7, 64)
    assert run_calls[0]["topk_ids"].dtype == torch.int32
    assert run_calls[0]["topk_weights"].dtype == torch.float32
    assert run_calls[0]["scratch"].dtype == torch.uint8
    assert run_calls[0]["scratch"].numel() == 64
    assert run_calls[0]["experts"] is experts._prepared_experts
    assert run_calls[0]["input_scales_static"] is True


def test_b12x_moe_warmup_runs_one_launch_per_planner_regime(monkeypatch) -> None:
    planned_tokens = []
    scratch_tokens = []
    run_tokens = []

    def fake_execution_plan(**kwargs):
        tokens = kwargs["tokens"]
        planned_tokens.append(tokens)
        execution = "dynamic-m16" if tokens < 8 else "dynamic-m32"
        return _FakePlan(execution).launch_plan

    def fake_scratch_plan(**kwargs):
        scratch_tokens.append(kwargs["tokens"])
        return _FakePlan()

    monkeypatch.setattr(
        b12x_moe,
        "_plan_b12x_moe_execution",
        fake_execution_plan,
    )
    monkeypatch.setattr(
        b12x_moe,
        "_plan_b12x_moe_fp4_scratch",
        fake_scratch_plan,
    )
    monkeypatch.setattr(
        b12x_moe,
        "_run_b12x_moe_fp4",
        lambda **kwargs: run_tokens.append(kwargs["a"].shape[0]),
    )
    monkeypatch.setattr(
        b12x_moe,
        "_dynamic_moe_warmup_tokens",
        lambda *, topk, quant_mode, requested_tokens: requested_tokens,
    )

    experts = _make_fake_b12x_experts()
    layer = SimpleNamespace(
        w13_weight=torch.empty(8, 32, 32, dtype=torch.uint8),
        w2_weight=torch.empty(8, 64, 8, dtype=torch.uint8),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        apply_router_weight_on_input=False,
    )

    warmed = experts.warmup_dynamic_launches(
        layer,
        token_counts=(3, 4, 8, 16),
    )

    assert warmed == 2
    assert planned_tokens == [3, 4, 8, 16]
    assert scratch_tokens == [3, 8]
    assert run_tokens == [3, 8]


def test_b12x_moe_warmup_uses_workspace_token_limit(monkeypatch) -> None:
    planned_tokens = []
    run_tokens = []

    monkeypatch.setenv("B12X_MOE_WORKSPACE_TOKEN_LIMIT", "4")
    monkeypatch.setattr(
        b12x_moe,
        "_dynamic_moe_warmup_tokens",
        lambda *, topk, quant_mode, requested_tokens: requested_tokens,
    )

    def record_plan(**kwargs):
        planned_tokens.append(kwargs["tokens"])
        return _FakePlan().launch_plan

    monkeypatch.setattr(
        b12x_moe,
        "_plan_b12x_moe_execution",
        record_plan,
    )
    monkeypatch.setattr(
        b12x_moe,
        "_plan_b12x_moe_fp4_scratch",
        lambda **kwargs: _FakePlan(),
    )
    monkeypatch.setattr(
        b12x_moe,
        "_run_b12x_moe_fp4",
        lambda **kwargs: run_tokens.append(kwargs["a"].shape[0]),
    )

    experts = _make_fake_b12x_experts()
    layer = SimpleNamespace(
        w13_weight=torch.empty(8, 32, 32, dtype=torch.uint8),
        w2_weight=torch.empty(8, 64, 8, dtype=torch.uint8),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        apply_router_weight_on_input=False,
    )

    warmed = experts.warmup_dynamic_launches(layer, token_counts=(8,))

    assert warmed == 1
    assert planned_tokens == [4]
    assert run_tokens == [4]


def test_b12x_moe_workspace_limit_is_disabled_for_kquant_capture(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_MOE_WORKSPACE_TOKEN_LIMIT", "2")
    monkeypatch.setenv("VLLM_KQUANT_CAPTURE_DIR", "/tmp/kquant-capture")

    assert b12x_moe._moe_workspace_tokens(5) == 5


def test_b12x_force_a16_nvfp4_selects_w4a16(monkeypatch) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")

    experts = _make_fake_b12x_experts()

    assert experts._quant_mode() == "w4a16"


def test_b12x_moe_workspace_limit_bounds_scratch_not_output(monkeypatch) -> None:
    planned_tokens = []

    def fake_plan(**kwargs):
        planned_tokens.append(kwargs["tokens"])
        return _FakePlan()

    monkeypatch.setenv("B12X_MOE_WORKSPACE_TOKEN_LIMIT", "2")
    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)

    experts = _make_fake_b12x_experts()
    workspace13, workspace2, output = experts.workspace_shapes(
        M=5,
        N=32,
        K=64,
        topk=4,
        global_num_experts=8,
        local_num_experts=8,
        expert_tokens_meta=None,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )

    assert planned_tokens == [2]
    assert workspace13 == (0,)
    assert workspace2 == (32,)
    assert output == (5, 64)


def test_b12x_moe_workspace_limit_chunks_token_independent_launches(
    monkeypatch,
) -> None:
    planned_tokens = []
    run_calls = []

    def fake_plan(**kwargs):
        planned_tokens.append(kwargs["tokens"])
        return _FakePlan()

    def fake_run(**kwargs):
        run_calls.append(kwargs)
        kwargs["output"].copy_(kwargs["a"])

    monkeypatch.setenv("B12X_MOE_WORKSPACE_TOKEN_LIMIT", "2")
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)
    monkeypatch.setattr(b12x_moe, "_run_b12x_moe_fp4", fake_run)

    local_num_experts = 8
    global_num_experts = 12
    experts = _make_fake_b12x_experts()
    experts._prepared_experts = _make_fake_prepared_experts(
        quant_mode="w4a16",
        num_experts=local_num_experts,
    )
    expert_map = torch.tensor(
        [0, -1, 1, -1, 2, -1, 3, 4, -1, 5, 6, 7],
        dtype=torch.int32,
    )
    hidden_states = torch.arange(5 * 64, dtype=torch.bfloat16).view(5, 64)
    output = torch.empty_like(hidden_states)
    topk_ids = torch.arange(5 * 4, dtype=torch.int32).view(5, 4) % 12
    topk_weights = torch.full((5, 4), 0.25, dtype=torch.float32)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(0, dtype=torch.uint8),
        w2=torch.empty(0, dtype=torch.uint8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=torch.empty(64, dtype=torch.uint8),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert planned_tokens == [2, 2, 1]
    assert [call["a"].shape[0] for call in run_calls] == [2, 2, 1]
    assert all(call["route_expert_map"] is expert_map for call in run_calls)
    assert all(
        call["plan"] is not None and call["output"].shape == call["a"].shape
        for call in run_calls
    )
    assert torch.equal(torch.cat([call["topk_ids"] for call in run_calls]), topk_ids)
    assert torch.equal(
        torch.cat([call["topk_weights"] for call in run_calls]), topk_weights
    )
    assert torch.equal(output, hidden_states)


def test_b12x_activation_amax_registers_stable_vllm_owned_tensor(
    monkeypatch,
) -> None:
    b12x_moe._reset_b12x_moe_activation_amax_for_tests()
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX", "1")

    experts = _make_fake_b12x_experts()
    experts._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.3.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=8,
    )

    activation_amax, layer_idx = experts._activation_amax_args(
        device=torch.device("cpu"),
        num_experts=8,
    )

    assert activation_amax is not None
    assert activation_amax.shape == (4, 8, 2)
    assert activation_amax.dtype == torch.float32
    assert layer_idx == 3
    data_ptr = activation_amax.data_ptr()

    late_experts = _make_fake_b12x_experts()
    with pytest.raises(RuntimeError, match="would reallocate after use"):
        late_experts._register_activation_amax(
            layer=SimpleNamespace(layer_name="model.layers.4.mlp.experts"),
            device=torch.device("cpu"),
            num_experts=8,
        )
    assert activation_amax.data_ptr() == data_ptr


def test_b12x_activation_amax_is_passed_as_separate_binding_arg(
    monkeypatch,
) -> None:
    b12x_moe._reset_b12x_moe_activation_amax_for_tests()
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX", "1")

    plan_calls = []
    run_calls = []

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return _FakePlan()

    def fake_run(**kwargs):
        run_calls.append(kwargs)

    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)
    monkeypatch.setattr(b12x_moe, "_run_b12x_moe_fp4", fake_run)

    num_experts = 8
    hidden_size = 16
    experts = _make_fake_b12x_experts()
    experts._prepared_experts = _make_fake_prepared_experts(
        quant_mode="w4a16",
        activation="swigluoai_uninterleave",
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=32,
    )
    experts._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.2.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=num_experts,
    )

    hidden_states = torch.zeros(3, hidden_size, dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    topk_ids = torch.zeros(3, 4, dtype=torch.int32)
    topk_weights = torch.full((3, 4), 0.25, dtype=torch.float32)
    workspace2 = torch.empty(64, dtype=torch.uint8)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(0, dtype=torch.uint8),
        w2=torch.empty(0, dtype=torch.uint8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=num_experts,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert len(plan_calls) == 1
    assert len(run_calls) == 1
    assert plan_calls[0]["collect_activation_amax"] is True
    assert run_calls[0]["activation_amax"] is not workspace2
    assert run_calls[0]["activation_amax"].shape == (3, num_experts, 2)
    assert run_calls[0]["layer_idx"] == 2


def test_b12x_w4a16_expert_map_sizes_workspace_for_global_routes(
    monkeypatch,
) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    plan_calls = []

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return _FakePlan()

    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)
    experts = _make_fake_b12x_experts()
    experts._prepared_experts = _make_fake_prepared_experts(quant_mode="w4a16")

    experts.workspace_shapes(
        M=1,
        N=32,
        K=64,
        topk=4,
        global_num_experts=12,
        local_num_experts=8,
        expert_tokens_meta=None,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )

    assert plan_calls[0]["route_num_experts"] == 12


def test_b12x_w4a16_expert_map_plans_and_binds_global_routes(monkeypatch) -> None:
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")

    plan_calls = []
    run_calls = []

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return _FakePlan()

    def fake_run(**kwargs):
        run_calls.append(kwargs)

    monkeypatch.setattr(b12x_moe, "_plan_b12x_moe_fp4_scratch", fake_plan)
    monkeypatch.setattr(b12x_moe, "_run_b12x_moe_fp4", fake_run)

    local_num_experts = 8
    global_num_experts = 12
    hidden_size = 16
    experts = _make_fake_b12x_experts()
    experts._prepared_experts = _make_fake_prepared_experts(
        quant_mode="w4a16",
        num_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=32,
    )
    expert_map = torch.tensor(
        [0, -1, 1, -1, 2, -1, 3, 4, -1, 5, 6, 7],
        dtype=torch.int32,
    )
    hidden_states = torch.zeros(1, hidden_size, dtype=torch.bfloat16)
    output = torch.empty_like(hidden_states)
    topk_ids = torch.tensor([[0, 1, 10, 11]], dtype=torch.int32)
    topk_weights = torch.full((1, 4), 0.25, dtype=torch.float32)

    assert experts.supports_expert_map()
    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=torch.empty(0, dtype=torch.uint8),
        w2=torch.empty(0, dtype=torch.uint8),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=torch.empty(64, dtype=torch.uint8),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert len(plan_calls) == 1
    assert plan_calls[0]["route_num_experts"] == global_num_experts
    assert len(run_calls) == 1
    assert run_calls[0]["route_expert_map"] is expert_map
    assert torch.equal(run_calls[0]["topk_ids"], topk_ids)
    assert torch.equal(run_calls[0]["topk_weights"], topk_weights)


def test_b12x_activation_amax_save_every_writes_main_and_mtp_files(
    monkeypatch,
    tmp_path,
) -> None:
    b12x_moe._reset_b12x_moe_activation_amax_for_tests()
    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX", "1")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX_SAVE_EVERY", "2")
    monkeypatch.setenv("VLLM_B12X_MOE_ACTIVATION_AMAX_FILE", str(tmp_path / "amax.pt"))

    main = _make_fake_b12x_experts()
    main._activation_amax_base_num_layers = 60
    main._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.3.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=8,
    )
    mtp = _make_fake_b12x_experts()
    mtp._activation_amax_base_num_layers = 60
    mtp._register_activation_amax(
        layer=SimpleNamespace(layer_name="model.layers.60.mlp.experts"),
        device=torch.device("cpu"),
        num_experts=8,
    )

    main_amax, main_layer = main._activation_amax_args(
        device=torch.device("cpu"),
        num_experts=8,
    )
    mtp_amax, mtp_layer = mtp._activation_amax_args(
        device=torch.device("cpu"),
        num_experts=8,
    )
    assert main_amax is not None and mtp_amax is not None
    assert main_layer == 3
    assert mtp_layer == 0
    main_amax[main_layer, 1, 0] = 11.0
    mtp_amax[mtp_layer, 2, 1] = 17.0

    b12x_moe.maybe_save_b12x_moe_activation_amax()
    assert not list(tmp_path.glob("*.pt"))

    b12x_moe.maybe_save_b12x_moe_activation_amax()
    files = sorted(tmp_path.glob("*.pt"))
    assert len(files) == 2
    loaded = [torch.load(path, weights_only=False) for path in files]
    payloads = {payload["model"]: payload for payload in loaded}
    assert set(payloads) == {"main", "mtp"}
    assert payloads["main"]["activation_amax"][3, 1, 0] == 11.0
    assert payloads["mtp"]["activation_amax"][0, 2, 1] == 17.0
    assert payloads["main"]["layers"][3]["prefix"] == "model.layers.3.mlp.experts"
    assert payloads["mtp"]["layers"][0]["external_layer_idx"] == 60


def test_b12x_force_a8_mxfp4_prepares_one_expert_owner(monkeypatch) -> None:
    b12x_fused_moe = pytest.importorskip("b12x.moe.fused_moe")

    monkeypatch.setenv("B12X_MOE_FORCE_A16", "1")
    monkeypatch.setenv("B12X_FORCE_MOE_A8", "1")

    plan_calls = []
    prepare_calls = []
    weight_plan = SimpleNamespace(
        quant_modes=frozenset({"w4a8_mx"}),
        io_dtype="bfloat16",
        activation="silu",
        discards_source_parameters=True,
    )

    def fake_plan(**kwargs):
        plan_calls.append(kwargs)
        return weight_plan

    def fake_prepare(**kwargs):
        prepare_calls.append(kwargs)
        return _make_fake_prepared_experts(
            quant_mode="w4a8_mx",
            activation="silu",
            num_experts=8,
            hidden_size=256,
            intermediate_size=128,
            plan=weight_plan,
        )

    monkeypatch.setattr(
        b12x_fused_moe,
        "plan_weights",
        fake_plan,
    )
    monkeypatch.setattr(
        b12x_fused_moe,
        "prepare_weights",
        fake_prepare,
    )

    experts = _make_fake_b12x_experts()
    experts._prepared_experts = None
    experts.quant_config.quant_dtype = "mxfp4"
    experts.quant_config.weight_quant_dtype = "mxfp4"
    experts.quant_config.g1_alphas = None
    experts.quant_config.g2_alphas = None
    experts.quant_config.a1_gscale = None
    experts.quant_config.a2_gscale = None
    experts.quant_config.w1_scale = torch.empty((8, 256, 8), dtype=torch.uint8)
    experts.quant_config.w2_scale = torch.empty((8, 256, 4), dtype=torch.uint8)
    w1 = torch.empty((8, 256, 128), dtype=torch.uint8)
    w2 = torch.empty((8, 256, 64), dtype=torch.uint8)

    prepared = experts._get_or_prepare_experts(
        w1=w1,
        w2=w2,
        activation=MoEActivation.SILU,
        params_dtype=torch.bfloat16,
    )

    assert experts._quant_mode() == "w4a8_mx"
    assert prepared is experts._prepared_experts
    assert len(plan_calls) == 1
    assert plan_calls[0]["quant_modes"] == "w4a8_mx"
    assert plan_calls[0]["source_format"] == "fp4_e8m0_k32"
    assert plan_calls[0]["w13_layout"] == "w31"
    assert len(prepare_calls) == 1
    assert prepare_calls[0]["plan"] is weight_plan
    assert prepare_calls[0]["w1_fp4"] is w1
    assert prepare_calls[0]["w2_fp4"] is w2
    assert torch.equal(prepare_calls[0]["w1_global_scale"], torch.ones(8))
    assert torch.equal(prepare_calls[0]["w2_global_scale"], torch.ones(8))


def test_b12x_supports_situ_activation() -> None:
    assert MoEActivation.from_str("situ") is MoEActivation.SITU
    assert b12x_moe.B12xExperts._supports_activation(MoEActivation.SITU)


def test_warmup_b12x_moe_dynamic_dedupes_signatures(monkeypatch) -> None:
    calls = []

    def fake_signature(self, layer):
        return ("same-signature",)

    def fake_warmup(self, layer, *, token_counts):
        calls.append((self, layer, token_counts))
        return 3

    monkeypatch.setattr(
        b12x_moe.B12xExperts,
        "warmup_dynamic_signature",
        fake_signature,
    )
    monkeypatch.setattr(
        b12x_moe.B12xExperts,
        "warmup_dynamic_launches",
        fake_warmup,
    )

    experts_0 = object.__new__(b12x_moe.B12xExperts)
    experts_1 = object.__new__(b12x_moe.B12xExperts)
    modules = [
        SimpleNamespace(
            routed_experts=SimpleNamespace(
                quant_method=SimpleNamespace(
                    moe_kernel=SimpleNamespace(fused_experts=experts_0)
                )
            )
        ),
        SimpleNamespace(
            routed_experts=SimpleNamespace(
                quant_method=SimpleNamespace(
                    moe_kernel=SimpleNamespace(fused_experts=experts_1)
                )
            )
        ),
    ]
    model = SimpleNamespace(modules=lambda: iter(modules))

    warmed = b12x_moe.warmup_b12x_moe_dynamic(
        model,
        max_tokens=5,
        token_counts=(3,),
    )

    assert warmed == 3
    assert len(calls) == 1
    assert calls[0][2] == (1, 2, 3, 4, 5)


def test_b12x_moe_warmup_token_counts_cover_serving_range() -> None:
    assert b12x_moe._b12x_moe_warmup_token_counts(
        max_tokens=10,
        token_counts=(3, 8, 12, 0),
    ) == (1, 2, 3, 4, 8, 10)
