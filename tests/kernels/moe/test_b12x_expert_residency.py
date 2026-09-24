# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts for b12x expert residency (HBM and Grace MoE experts, SM103)."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("b12x.moe.fused_moe.automatic")

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.b12x as b12x
import vllm.model_executor.layers.fused_moe.b12x_residency as residency
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.config.expert_residency import ExpertResidencyConfig
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.utils.b12x import B12xWorkload

EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 4, 256, 256, 2


def _quant_config() -> FusedMoEQuantConfig:
    scale = torch.ones(1, dtype=torch.float32)
    return FusedMoEQuantConfig.make(
        quant_dtype="mxfp8",
        weight_dtype="mxfp4",
        w1_scale=scale,
        w2_scale=scale,
        g1_alphas=scale,
        g2_alphas=scale,
        a1_gscale=scale,
        a2_gscale=scale,
    )


def _layer(name: str, experts: int = EXPERTS) -> SimpleNamespace:
    return SimpleNamespace(
        layer_name=name,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=False,
        w13_weight=torch.zeros(
            experts, 2 * INTERMEDIATE, HIDDEN // 2, dtype=torch.uint8
        ),
        w2_weight=torch.zeros(experts, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8),
        w13_weight_scale=torch.full(
            (experts, 2 * INTERMEDIATE, HIDDEN // 32), 127, dtype=torch.uint8
        ),
        w2_weight_scale=torch.full(
            (experts, HIDDEN, INTERMEDIATE // 32), 127, dtype=torch.uint8
        ),
    )


def _declared(monkeypatch, name: str, experts: int = EXPERTS):
    monkeypatch.setattr(b12x, "expert_residency_enabled", lambda: True)
    monkeypatch.setattr(
        b12x, "_register_b12x_moe_output_collective", lambda *a, **k: None
    )
    config = make_dummy_moe_config(
        num_experts=experts,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN,
        intermediate_size=INTERMEDIATE,
        in_dtype=torch.bfloat16,
    )
    provider = B12xExperts(config, _quant_config())
    layer = _layer(name, experts)
    provider.process_weights_after_loading(layer)
    return provider, layer


def _workload(max_tokens: int = 8) -> B12xWorkload:
    return B12xWorkload(
        stage="weights",
        token_counts=(1, max_tokens),
        fixed_token_counts=(1,),
        output_dtype=torch.bfloat16,
        max_tokens=max_tokens,
        max_seqs=4,
        max_model_len=128,
    )


def _platform(monkeypatch, capability=(10, 3)):
    import vllm.platforms as platforms

    monkeypatch.setattr(
        platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability=lambda value, device_id=0: tuple(value) == capability,
            is_device_capability_family=lambda family, device_id=0: (
                capability[0] * 10 == family
            ),
        ),
    )


def _deployment(**changes):
    values = dict(
        model_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
        ),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1 << 30),
        load_config=SimpleNamespace(load_format="auto"),
        moe_backend="b12x",
    )
    values.update(changes)
    return values


def test_config_admits_one_sm103_gpu_with_reserved_kv(monkeypatch):
    _platform(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_B12X_SM103", True)
    ExpertResidencyConfig().verify(**_deployment())


@pytest.mark.parametrize(
    "change,message",
    [
        ({"moe_backend": "flashinfer_cutlass"}, "moe-backend b12x"),
        ({"cache_config": SimpleNamespace(kv_cache_memory_bytes=None)}, "kv-cache"),
        ({"load_config": SimpleNamespace(load_format="instanttensor")}, "load_format"),
        (
            {
                "parallel_config": SimpleNamespace(
                    tensor_parallel_size=2,
                    pipeline_parallel_size=1,
                    data_parallel_size=1,
                    enable_expert_parallel=False,
                )
            },
            "one GPU",
        ),
    ],
)
def test_config_rejects_unsupported_deployments(monkeypatch, change, message):
    _platform(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_B12X_SM103", True)
    with pytest.raises(ValueError, match=message):
        ExpertResidencyConfig().verify(**_deployment(**change))


def test_config_requires_sm103_and_the_opt_in(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_B12X_SM103", True)
    _platform(monkeypatch, capability=(12, 0))
    with pytest.raises(ValueError, match="SM103 GPU"):
        ExpertResidencyConfig().verify(**_deployment())
    _platform(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_B12X_SM103", False)
    with pytest.raises(ValueError, match="VLLM_B12X_SM103"):
        ExpertResidencyConfig().verify(**_deployment())
    with pytest.raises(ValueError, match="min_hot_experts"):
        ExpertResidencyConfig(min_hot_experts=3, max_hot_experts=2)


@pytest.mark.parametrize(
    "capability,opt_in,expected",
    [
        ((12, 0), False, True),
        ((10, 3), False, False),
        ((10, 3), True, True),
        ((10, 0), True, False),
    ],
)
def test_b12x_native_device_needs_the_sm103_opt_in(
    monkeypatch, capability, opt_in, expected
):
    from vllm.utils.b12x import b12x_native_device

    _platform(monkeypatch, capability=capability)
    monkeypatch.setattr(envs, "VLLM_B12X_SM103", opt_in)
    assert b12x_native_device() is expected


def test_mxfp4_experts_declare_residency_instead_of_preparing(monkeypatch):
    from b12x.moe.fused_moe import ExecutionCapacity, ResidencyLayerSpec

    def forbidden(*args, **kwargs):
        pytest.fail("residency must not prepare ordinary b12x MoE weights")

    monkeypatch.setattr(B12xExperts, "_prepare_experts", forbidden)
    provider, layer = _declared(monkeypatch, "layers.0.mlp.experts")
    declaration = provider.residency_declaration
    assert declaration is not None and declaration.plan is None
    assert declaration.w13 is layer.w13_weight and declaration.top_k == TOP_K
    assert layer.b12x_preparation_provider is provider
    spec = ResidencyLayerSpec.from_weight_plan(
        layer="layers.0.mlp.experts",
        weight_plan=declaration.weight_plan,
        capacity=ExecutionCapacity(max_tokens=8, top_k=TOP_K),
    )
    assert (spec.experts, spec.hidden, spec.intermediate) == (
        EXPERTS,
        HIDDEN,
        INTERMEDIATE,
    )
    assert spec.gate_first


def test_host_staging_is_required(monkeypatch):
    monkeypatch.setattr(b12x, "expert_residency_enabled", lambda: True)
    provider = B12xExperts(
        make_dummy_moe_config(
            num_experts=EXPERTS,
            experts_per_token=TOP_K,
            hidden_dim=HIDDEN,
            intermediate_size=INTERMEDIATE,
            in_dtype=torch.bfloat16,
        ),
        _quant_config(),
    )
    layer = _layer("layers.0.mlp.experts")
    layer.w13_weight = torch.empty(0, device="meta")
    with pytest.raises(ValueError, match="host memory"):
        provider.process_weights_after_loading(layer)


def _model(*providers):
    modules = [
        SimpleNamespace(b12x_preparation_provider=provider) for provider in providers
    ]
    return SimpleNamespace(modules=lambda: iter(modules))


def test_declaration_places_target_layers_and_keeps_draft_in_hbm(monkeypatch):
    from b12x.moe import fused_moe

    target = [
        _declared(monkeypatch, f"model.layers.{index}.mlp.experts")[0]
        for index in range(2)
    ]
    draft, _ = _declared(monkeypatch, "model.layers.2.mlp.experts")
    capacity = fused_moe.ExecutionCapacity(max_tokens=8, top_k=TOP_K)
    specs = [
        fused_moe.ResidencyLayerSpec.from_weight_plan(
            layer=p.residency_declaration.layer_name,
            weight_plan=p.residency_declaration.weight_plan,
            capacity=capacity,
        )
        for p in (*target, draft)
    ]
    overhead = sum(
        s.memory(0).scratch_bytes + s.memory(0).route_map_bytes for s in specs
    )
    kv = 1 << 20
    # Room for every draft expert and one expert per target layer.
    hbm = overhead + (EXPERTS + 2) * specs[0].expert_bytes + kv

    def available(cls, *, device, grace_bytes=None, **reservations):
        return cls(hbm_bytes=hbm, grace_bytes=1 << 30, **reservations)

    monkeypatch.setattr(
        fused_moe.ModelExpertMemoryBudget, "from_available", classmethod(available)
    )
    monkeypatch.setattr(
        fused_moe.ResidencyHardware,
        "detect",
        classmethod(
            lambda cls, device: cls(compute_capability=(10, 3), grace_coherent=True)
        ),
    )
    monkeypatch.setattr(residency, "checkpoint_fingerprint", lambda _: "sha256:test")
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda: pytest.fail("placement must not initialize CUDA"),
    )
    vllm_config = SimpleNamespace(
        expert_residency_config=ExpertResidencyConfig(
            hbm_reserve_gb=0, grace_reserve_gb=0
        ),
        model_config=SimpleNamespace(revision=None, tokenizer_revision=None),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv),
    )
    residency.declare_expert_residency(
        vllm_config,
        model=_model(*target),
        draft=_model(draft),
        workload=_workload(),
        draft_workload=replace(_workload(), lane=1),
        device=torch.device("cuda", 0),
    )
    plans = [p.residency_declaration.plan for p in (*target, draft)]
    assert all(plan is not None and plan.prepared is None for plan in plans)
    assert all(p.residency_declaration.max_tokens == 8 for p in (*target, draft))
    # Draft experts all fit in HBM; each target layer keeps one hot expert.
    hot = [plan.query.hot_experts for plan in plans]
    assert hot == [1, 1, EXPERTS]


def test_checkpoint_fingerprint_tracks_safetensors_headers(tmp_path):
    def write(name, header):
        payload = json.dumps(header).encode()
        (tmp_path / name).write_bytes(len(payload).to_bytes(8, "little") + payload)

    model_config = SimpleNamespace(model=str(tmp_path), revision=None)
    with pytest.raises(ValueError, match="safetensors"):
        residency.checkpoint_fingerprint(model_config)
    header = {"w": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    write("model-00001.safetensors", header)
    first = residency.checkpoint_fingerprint(model_config)
    assert first == residency.checkpoint_fingerprint(model_config)
    header["w"]["shape"] = [2, 2]
    write("model-00001.safetensors", header)
    assert residency.checkpoint_fingerprint(model_config) != first


def test_apply_binds_the_residency_plan_without_caller_workspace(monkeypatch):
    provider, _ = _declared(monkeypatch, "layers.0.mlp.experts")
    declaration = provider.residency_declaration
    declaration.plan, declaration.max_tokens = object(), 8
    calls = []
    fake = SimpleNamespace(
        bind=lambda plan, **kwargs: calls.append((plan, kwargs)) or "binding",
        run=lambda *, binding: calls.append(binding),
    )
    monkeypatch.setattr(b12x, "_require_b12x_fused_moe", lambda: fake)
    monkeypatch.setattr(b12x, "_is_current_stream_capturing", lambda: False)
    assert provider.workspace_shapes(
        5, 2 * INTERMEDIATE, HIDDEN, TOP_K, EXPERTS, EXPERTS, None, MoEActivation.SILU
    ) == ((0,), (1,), (5, HIDDEN))

    def apply(tokens):
        provider.apply(
            output=torch.empty(tokens, HIDDEN, dtype=torch.bfloat16),
            hidden_states=torch.empty(tokens, HIDDEN, dtype=torch.bfloat16),
            w1=torch.empty(0),
            w2=torch.empty(0),
            topk_weights=torch.ones(tokens, TOP_K, dtype=torch.bfloat16),
            topk_ids=torch.zeros(tokens, TOP_K, dtype=torch.int32),
            activation=MoEActivation.SILU,
            global_num_experts=EXPERTS,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    apply(0)
    assert not calls
    apply(3)
    plan, kwargs = calls[0]
    assert plan is declaration.plan and calls[1] == "binding"
    assert kwargs["topk_weights"].dtype == torch.float32
    assert set(kwargs) == {"a", "topk_ids", "topk_weights", "output"}
    with pytest.raises(ValueError, match="residency capacity"):
        apply(9)
