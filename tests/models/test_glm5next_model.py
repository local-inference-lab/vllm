# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from vllm.model_executor.layers import mla as mla_layer
from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.models.glm4_1v import Glm4vForConditionalGeneration
from vllm.model_executor.models.interfaces import supports_eagle3, supports_pp
from vllm.models.glm5next.nvidia import attention as glm5next_attention
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
from vllm.models.glm5next.nvidia.model import (
    Glm5NextDecoderLayer,
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    Glm5NextMoE,
    _load_glm5next_fused_conv1d,
    _remap_glm5next_weight_name,
)
from vllm.models.glm5next.nvidia.mtp import (
    Glm5NextMTP,
    Glm5NextMultiTokenPredictor,
)
from vllm.transformers_utils.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from vllm.transformers_utils.processors import glm5next as glm5next_processor
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xGLM5NextMLASparseBackend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _make_eagle_draft_vllm_config,
)


def test_glm5next_config_preserves_official_sparse_moe_fields() -> None:
    text_config = Glm5NextTextConfig(
        topk_method="noaux_tc",
        norm_topk_prob=False,
        indexer_rope_interleave=True,
        logit_scale=0.5,
        swiglu_limit=10.0,
    )
    vision_config = Glm5NextVisionConfig(swiglu_limit=10.0)

    assert text_config.topk_method == "noaux_tc"
    assert not text_config.moe_renormalize
    assert text_config.indexer_rope_interleave
    assert text_config.logit_scale == 0.5
    assert text_config.swiglu_limit == 10.0
    assert vision_config.swiglu_limit == 10.0


def test_glm5next_config_accepts_prebuilt_subconfigs() -> None:
    text_config = Glm5NextTextConfig(hidden_size=1024)
    vision_config = Glm5NextVisionConfig(hidden_size=768)

    config = Glm5NextConfig(
        text_config=text_config,
        vision_config=vision_config,
    )

    assert config.text_config is text_config
    assert config.vision_config is vision_config


@pytest.mark.parametrize(
    ("checkpoint_name", "parameter_name"),
    [
        (
            "model.layers.0.self_attn.forget_gate.f_b_proj.weight",
            "model.layers.0.self_attn.f_b_proj.weight",
        ),
        (
            "model.layers.0.self_attn.forget_gate.A_log",
            "model.layers.0.self_attn.A_log",
        ),
        (
            "model.layers.3.attn_hc.fn",
            "model.layers.3.hc_attn_fn",
        ),
        (
            "model.layers.3.ffn_hc.scale",
            "model.layers.3.hc_ffn_scale",
        ),
    ],
)
def test_glm5next_checkpoint_weight_name_remapping(
    checkpoint_name: str,
    parameter_name: str,
) -> None:
    assert _remap_glm5next_weight_name(checkpoint_name) == parameter_name


def test_glm5next_kda_adapts_shared_out_buffer_forward(monkeypatch) -> None:
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    hidden_states = torch.randn(2, 4)
    positions = torch.arange(2)

    def fake_forward(self, hidden_states, positions, output) -> None:
        output.copy_(hidden_states + positions[:, None])

    monkeypatch.setattr(KimiGatedDeltaNetAttention, "forward", fake_forward)

    actual = layer(hidden_states, positions)

    torch.testing.assert_close(actual, hidden_states + positions[:, None])


def test_glm5next_moe_applies_external_gate_once() -> None:
    layer = Glm5NextMoE.__new__(Glm5NextMoE)
    torch.nn.Module.__init__(layer)
    layer.is_sequence_parallel = False

    class CountingGate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states):
            self.calls += 1
            return hidden_states + 1, None

    class RecordingExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router_input = None

        def forward(self, *, hidden_states, router_logits):
            self.router_input = router_logits
            return hidden_states

    layer.gate = CountingGate()
    layer.experts = RecordingExperts()
    hidden_states = torch.randn(2, 4)

    actual = layer(hidden_states)

    assert layer.gate.calls == 1
    torch.testing.assert_close(layer.experts.router_input, hidden_states + 1)
    torch.testing.assert_close(actual, hidden_states)


def test_glm5next_moe_does_not_give_gate_to_runner(monkeypatch) -> None:
    from vllm.models.glm5next.nvidia import model as glm5next_model

    class FakeGate(torch.nn.Module):
        out_dtype = torch.float32
        e_score_correction_bias = None

        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeExperts(torch.nn.Module):
        pass

    factory_kwargs = {}

    def fake_factory(**kwargs):
        factory_kwargs.update(kwargs)
        return FakeExperts()

    ep_group = SimpleNamespace(
        device_group=SimpleNamespace(size=lambda: 1),
        rank_in_group=0,
    )
    monkeypatch.setattr(
        glm5next_model, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(glm5next_model, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(glm5next_model, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(
        glm5next_model, "_get_moe_router_dtype", lambda config: torch.float32
    )
    monkeypatch.setattr(glm5next_model, "GateLinear", FakeGate)
    monkeypatch.setattr(glm5next_model, "FusedMoEFactory", fake_factory)

    config = SimpleNamespace(
        routed_scaling_factor=1.0,
        n_routed_experts=8,
        n_shared_experts=None,
        hidden_act="silu",
        hidden_size=16,
        topk_method=None,
        moe_intermediate_size=8,
        num_experts_per_token=2,
        moe_renormalize=False,
    )
    parallel_config = SimpleNamespace(
        use_sequence_parallel_moe=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        enable_eplb=False,
    )

    layer = Glm5NextMoE(config, parallel_config)

    assert isinstance(layer.gate, FakeGate)
    assert "gate" not in factory_kwargs


def test_glm5next_kda_splits_mixed_decode_prefill_batch(monkeypatch) -> None:
    from vllm.models.kimi_k3.nvidia.ops.third_party import kda as kda_ops
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    prefix = "model.layers.0.self_attn"
    chunk_indices = torch.tensor([[0, 0]], dtype=torch.int32)
    chunk_offsets = torch.tensor([0], dtype=torch.int32)
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=4,
        has_initial_state=torch.tensor([True, True, False]),
        non_spec_query_start_loc=torch.tensor([0, 1, 2, 4], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([1, 2, 3], dtype=torch.int32),
        prefill_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        prefill_state_indices=torch.tensor([3], dtype=torch.int32),
        prefill_has_initial_state=torch.tensor([False]),
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )

    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = prefix
    layer.head_dim = 1
    layer.local_projection_size = 1
    layer.local_num_heads = 1
    layer.gate_lower_bound = -5.0
    layer.A_log = torch.ones(1)
    layer.dt_bias = torch.ones(1)
    layer._b12x_kda_plan = None
    layer.conv1d = SimpleNamespace(
        weight=torch.ones(3, 1, 3),
        bias=torch.zeros(3),
    )
    layer.kv_cache = (torch.zeros(4, 3, 3), torch.zeros(4, 1, 1, 1))

    class IdentityGate(torch.nn.Module):
        def forward(self, value, gate):
            return value

    layer.o_norm = IdentityGate()

    calls: dict[str, object] = {}

    def fake_conv(x, *args, **kwargs):
        return x

    def fake_recurrent(*, q, cu_seqlens, ssm_state_indices, **kwargs):
        calls["decode_q_len"] = q.shape[1]
        calls["decode_query_start_loc"] = cu_seqlens.clone()
        calls["decode_state_indices"] = ssm_state_indices.clone()
        return torch.full_like(q, 11), None

    def fake_chunk(
        *, q, initial_state, cu_seqlens, chunk_indices, chunk_offsets, **kwargs
    ):
        calls["prefill_q_len"] = q.shape[1]
        calls["prefill_query_start_loc"] = cu_seqlens.clone()
        calls["chunk_indices"] = chunk_indices
        calls["chunk_offsets"] = chunk_offsets
        return torch.full_like(q, 22), torch.full_like(initial_state, 33)

    def fake_gather(state, indices, has_initial_state):
        calls["prefill_state_indices"] = indices.clone()
        calls["prefill_has_initial_state"] = has_initial_state.clone()
        return state.index_select(0, indices.long())

    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={prefix: metadata}),
    )
    monkeypatch.setattr(kimi_gdn_linear_attn, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(kimi_gdn_linear_attn, "causal_conv1d_fn", fake_conv)
    monkeypatch.setattr(kimi_gdn_linear_attn, "gather_initial_states", fake_gather)
    monkeypatch.setattr(kda_ops, "fused_recurrent_kda", fake_recurrent)
    monkeypatch.setattr(kda_ops, "chunk_kda_with_fused_gate", fake_chunk)

    core_attn_out = torch.empty(1, 4, 1, 1)
    layer._forward(
        mixed_qkv=torch.arange(12, dtype=torch.float32).view(4, 3),
        g1=torch.ones(1, 4, 1, 1),
        g2=torch.ones(4, 1, 1),
        beta=torch.ones(1, 4, 1),
        core_attn_out=core_attn_out,
    )

    assert calls["decode_q_len"] == 2
    assert torch.equal(
        calls["decode_query_start_loc"], torch.tensor([0, 1, 2], dtype=torch.int32)
    )
    assert torch.equal(
        calls["decode_state_indices"], torch.tensor([1, 2], dtype=torch.int32)
    )
    assert calls["prefill_q_len"] == 2
    assert torch.equal(
        calls["prefill_query_start_loc"], torch.tensor([0, 2], dtype=torch.int32)
    )
    assert calls["chunk_indices"] is chunk_indices
    assert calls["chunk_offsets"] is chunk_offsets
    assert torch.equal(
        calls["prefill_state_indices"], torch.tensor([3], dtype=torch.int32)
    )
    assert torch.equal(calls["prefill_has_initial_state"], torch.tensor([False]))
    assert torch.equal(core_attn_out[:, :2], torch.full((1, 2, 1, 1), 11.0))
    assert torch.equal(core_attn_out[:, 2:], torch.full((1, 2, 1, 1), 22.0))
    assert torch.equal(layer.kv_cache[1][3], torch.full((1, 1, 1), 33.0))


def test_glm5next_alone_opts_into_b12x_kda_decode() -> None:
    assert not KimiGatedDeltaNetAttention.enable_b12x_kda_decode
    assert Glm5NextLinearAttention.enable_b12x_kda_decode
    assert KimiGatedDeltaNetAttention.b12x_kda_null_state_index is None
    assert Glm5NextLinearAttention.b12x_kda_null_state_index == 0


@pytest.mark.parametrize(
    ("backend", "speculative", "uses_b12x"),
    [
        ("auto", False, False),
        ("auto", True, True),
        ("b12x", False, True),
        ("b12x", True, True),
        ("triton", False, False),
        ("triton", True, False),
    ],
)
def test_glm5next_selects_configured_kda_decode_backend(
    monkeypatch,
    backend: str,
    speculative: bool,
    uses_b12x: bool,
) -> None:
    monkeypatch.setattr(
        KimiGatedDeltaNetAttention,
        "__init__",
        lambda self, config, vllm_config, prefix: None,
    )
    vllm_config = SimpleNamespace(
        additional_config={"glm53_kda_decode_backend": backend},
        speculative_config=object() if speculative else None,
    )

    layer = Glm5NextLinearAttention(object(), vllm_config)

    assert layer.enable_b12x_kda_decode is uses_b12x


def test_glm5next_defaults_to_hybrid_kda_decode(monkeypatch) -> None:
    monkeypatch.setattr(
        KimiGatedDeltaNetAttention,
        "__init__",
        lambda self, config, vllm_config, prefix: None,
    )
    vllm_config = SimpleNamespace(additional_config={}, speculative_config=None)

    layer = Glm5NextLinearAttention(object(), vllm_config)

    assert layer._glm53_kda_decode_backend == "auto"
    assert not layer.enable_b12x_kda_decode


def test_glm5next_rejects_unknown_kda_decode_backend() -> None:
    vllm_config = SimpleNamespace(
        additional_config={"glm53_kda_decode_backend": "unknown"}
    )

    with pytest.raises(ValueError, match="KDA decode backend"):
        Glm5NextLinearAttention(object(), vllm_config)


def test_glm5next_b12x_mhc_builds_first_layer_broadcast_fn() -> None:
    hidden_size = 4
    hc_mult = 4
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.hidden_size = hidden_size
    layer.n = hc_mult
    layer.hc_attn_fn = torch.nn.Parameter(
        torch.arange(24 * hc_mult * hidden_size, dtype=torch.float32).view(
            24, hc_mult * hidden_size
        )
    )
    layer.hc_attn_fn_broadcast = None
    layer._b12x_mhc = object()

    model = Glm5NextModel.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 1
    model.layers = torch.nn.ModuleList([layer])

    model.finalize_mhc_broadcast_weights()

    expected = layer.hc_attn_fn.detach().view(24, hc_mult, hidden_size).sum(dim=1)
    torch.testing.assert_close(layer.hc_attn_fn_broadcast, expected)
    assert layer.hc_attn_fn_broadcast.shape == (24, hidden_size)
    assert layer.hc_attn_fn_broadcast.is_contiguous()

    broadcast_data_ptr = layer.hc_attn_fn_broadcast.data_ptr()
    with torch.no_grad():
        layer.hc_attn_fn.add_(1)
    model.finalize_mhc_broadcast_weights()

    expected = layer.hc_attn_fn.detach().view(24, hc_mult, hidden_size).sum(dim=1)
    torch.testing.assert_close(layer.hc_attn_fn_broadcast, expected)
    assert layer.hc_attn_fn_broadcast.data_ptr() == broadcast_data_ptr


def test_glm5next_dflash_contracts_completed_mhc_hidden_state() -> None:
    completed = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    class FakeMhcLayer:
        mhc = True
        n = 4

        def hc_post(self, hidden_states, residual, post, comb):
            return completed

    model = Glm5NextModel.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.dflash_capture = True

    actual = model._prepare_aux_hidden_state(
        FakeMhcLayer(),
        torch.zeros(2, 3),
        torch.zeros(2, 4, 3),
        torch.zeros(2, 4),
        torch.zeros(2, 4, 4),
    )

    torch.testing.assert_close(actual, completed.mean(dim=1))
    assert actual.shape == (2, 3)


def test_glm5next_eagle_capture_preserves_completed_mhc_streams() -> None:
    completed = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    class FakeMhcLayer:
        mhc = True
        n = 4

        def hc_post(self, hidden_states, residual, post, comb):
            return completed

    model = Glm5NextModel.__new__(Glm5NextModel)
    torch.nn.Module.__init__(model)
    model.dflash_capture = False

    actual = model._prepare_aux_hidden_state(
        FakeMhcLayer(),
        torch.zeros(2, 3),
        torch.zeros(2, 4, 3),
        torch.zeros(2, 4),
        torch.zeros(2, 4, 4),
    )

    torch.testing.assert_close(actual, completed.flatten(1))
    assert actual.shape == (2, 12)


def test_glm5next_dflash_maps_target_layers_to_completed_outputs() -> None:
    draft_hf_config = SimpleNamespace(
        dflash_config={"target_layer_ids": [5, 14, 24, 33, 42]}
    )
    spec_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config)
    )

    assert get_eagle3_aux_layers_from_config(spec_config) == (6, 15, 25, 34, 43)
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (1, True),
        (8, True),
        (9, False),
    ],
)
def test_glm5next_b12x_mhc_dispatches_decode_sized_batches(
    num_tokens: int, expected: bool
) -> None:
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer._b12x_mhc = object()
    layer._b12x_mhc_max_tokens = 8

    assert layer._use_b12x_mhc(torch.empty(num_tokens, 4)) is expected


def test_glm5next_b12x_mhc_dispatch_requires_available_backend() -> None:
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer._b12x_mhc = None
    layer._b12x_mhc_max_tokens = 8

    assert not layer._use_b12x_mhc(torch.empty(1, 4))


def test_glm5next_conditional_post_load_finalizes_language_model() -> None:
    class FakeLanguageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.finalize_calls = 0

        def process_weights_after_loading(self) -> None:
            self.finalize_calls += 1

    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.language_model = FakeLanguageModel()

    model.process_weights_after_loading()

    assert model.language_model.finalize_calls == 1


def test_glm5next_b12x_kda_plan_reserves_null_state_zero(monkeypatch) -> None:
    captured_caps = {}

    class FakeApi:
        @staticmethod
        def Caps(**kwargs):
            captured_caps.update(kwargs)
            return kwargs

        @staticmethod
        def plan(caps):
            return caps

    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer._b12x_kda_api = FakeApi()
    layer._b12x_kda_max_tokens = 16
    layer._b12x_kda_max_seqs = 4
    layer._b12x_kda_state_index_columns = 4
    layer.local_num_heads = 8
    layer.head_dim = 128
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)

    monkeypatch.setattr(
        KimiGatedDeltaNetAttention,
        "get_state_dtype",
        lambda self: (torch.bfloat16, torch.float32),
    )
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "current_platform",
        SimpleNamespace(current_device=lambda: "cuda:0"),
    )

    plan = layer._make_b12x_kda_plan(max_state_slots=32)

    assert plan == captured_caps
    assert captured_caps["null_state_index"] == 0


def test_glm5next_b12x_kda_binds_caller_scratch_per_run(monkeypatch) -> None:
    scratch = torch.empty(32, dtype=torch.uint8)
    bindings: list[dict[str, object]] = []
    runs: list[object] = []

    class FakePlan:
        @staticmethod
        def shapes_and_dtypes():
            return (((32,), torch.uint8),)

    class FakeApi:
        @staticmethod
        def bind_kda(plan, **kwargs):
            binding = {"plan": plan, **kwargs}
            bindings.append(binding)
            return binding

        @staticmethod
        def run_kda(binding, **kwargs):
            runs.append(binding)

    workspace = SimpleNamespace(get_simultaneous=lambda *specs: (scratch,))
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "current_workspace_manager",
        lambda: workspace,
    )

    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer._b12x_kda_api = FakeApi()
    layer._b12x_kda_plan = FakePlan()
    layer._b12x_kda_max_tokens = 4
    layer._b12x_kda_max_seqs = 2
    layer._b12x_kda_state_index_columns = 2
    layer.local_num_heads = 1
    layer.head_dim = 2
    layer.gate_lower_bound = -5.0
    layer.A_log = torch.ones(1)
    layer.dt_bias = torch.ones(2)
    layer.o_norm = SimpleNamespace(weight=torch.ones(2), eps=1e-6)
    layer.kv_cache = (torch.empty(0), torch.zeros(3, 1, 2, 2))
    layer._b12x_kda_mixed_qkv = torch.empty(4, 6)
    layer._b12x_kda_raw_g = torch.empty(4, 1, 2)
    layer._b12x_kda_raw_beta = torch.empty(4, 1)
    layer._b12x_kda_z = torch.empty(4, 1, 2)
    layer._b12x_kda_output = torch.zeros(4, 1, 2)
    layer._b12x_kda_query_start_loc = torch.zeros(3, dtype=torch.int32)
    layer._b12x_kda_num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    layer._b12x_kda_state_indices = torch.zeros(2, 2, dtype=torch.int32)
    layer._b12x_kda_num_seqs = torch.zeros(1, dtype=torch.int32)
    layer._b12x_kda_num_tokens = torch.zeros(1, dtype=torch.int32)

    kwargs = dict(
        mixed_qkv=torch.ones(1, 6),
        raw_g=torch.ones(1, 1, 2),
        raw_beta=torch.ones(1, 1),
        z=torch.ones(1, 1, 2),
        output=torch.empty(1, 1, 2),
        state_indices=torch.tensor([[1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_accepted_tokens=None,
        num_requests=1,
    )
    layer._run_b12x_kda_decode_post_conv(**kwargs)
    layer._run_b12x_kda_decode_post_conv(**kwargs)

    assert len(bindings) == 2
    assert bindings[0] is not bindings[1]
    assert bindings[0]["scratch"] is scratch
    assert bindings[1]["scratch"] is scratch
    assert len(runs) == 2
    assert runs[0] is bindings[0]
    assert runs[1] is bindings[1]


def test_glm5next_sparse_mla_selects_b12x_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.qrep_active = False

    class FakeIndexer(torch.nn.Module):
        topk_tokens = 2048

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.indexer_op = None

    class FakeMLAAttention(torch.nn.Module):
        def __init__(self, **kwargs: object) -> None:
            super().__init__()
            captured.update(kwargs)
            self.layer_name = str(kwargs["prefix"])

    for name in (
        "DeepSeekV2FusedQkvAProjLinear",
        "ColumnParallelLinear",
        "ReplicatedLinear",
        "RowParallelLinear",
        "RMSNorm",
    ):
        monkeypatch.setattr(glm5next_attention, name, FakeLinear)
    monkeypatch.setattr(glm5next_attention, "Glm5NextPooledIndexer", FakeIndexer)
    monkeypatch.setattr(
        glm5next_attention, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(mla_layer, "MLAAttention", FakeMLAAttention)

    glm5next_attention.Glm5NextMLAAttention(
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(backend=AttentionBackendEnum.B12X)
        ),
        config=SimpleNamespace(rms_norm_eps=1e-5, index_topk=2048),
        hidden_size=8,
        num_heads=1,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        cache_config=SimpleNamespace(),
        topk_indices_buffer=torch.empty((2, 2051), dtype=torch.int32),
        skip_rope=True,
    )

    assert captured["attn_backend"] is B12xGLM5NextMLASparseBackend
    assert captured["use_sparse"] is True


def test_glm5next_rejects_pipeline_parallelism() -> None:
    assert not supports_pp(Glm5NextForCausalLM)
    assert not supports_pp(Glm5NextForConditionalGeneration)


def test_glm5next_processor_resolves_repository_video_config(
    monkeypatch, tmp_path
) -> None:
    processor_config = tmp_path / "processor_config.json"
    processor_config.write_text(
        json.dumps(
            {
                "video_processor": {
                    "video_processor_type": "Glm5NextVideoProcessor",
                    "max_image_tokens": 240_000,
                }
            }
        )
    )
    calls = {}
    tokenizer = object()

    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda model, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        glm5next_processor,
        "get_image_processor_config",
        lambda model, **kwargs: {"image_processor_type": "ignored"},
    )

    def fake_cached_file(model, filename, **kwargs):
        calls["cached_file"] = (model, filename, kwargs)
        return str(processor_config)

    monkeypatch.setattr(glm5next_processor, "cached_file", fake_cached_file)
    monkeypatch.setattr(
        glm5next_processor,
        "Glm5NextImageProcessor",
        lambda **kwargs: ("image", kwargs),
    )
    monkeypatch.setattr(
        glm5next_processor,
        "Glm5NextVideoProcessor",
        lambda **kwargs: ("video", kwargs),
    )

    def fake_init(self, **kwargs) -> None:
        self.loaded_components = kwargs

    monkeypatch.setattr(glm5next_processor.Glm5NextProcessor, "__init__", fake_init)

    processor = glm5next_processor.Glm5NextProcessor.from_pretrained(
        "zai-org/GLM-5.3-Flash",
        revision="test-revision",
        local_files_only=True,
    )

    assert calls["cached_file"] == (
        "zai-org/GLM-5.3-Flash",
        "processor_config.json",
        {"local_files_only": True, "revision": "test-revision"},
    )
    assert processor.loaded_components["tokenizer"] is tokenizer
    assert processor.loaded_components["video_processor"] == (
        "video",
        {"max_image_tokens": 30_000},
    )


def test_glm5next_processing_info_pins_processor_revision(monkeypatch) -> None:
    from vllm.models.glm5next.nvidia.multimodal import Glm5NextProcessingInfo

    calls = []
    processor = object()

    def fake_from_pretrained(model, **kwargs):
        calls.append((model, kwargs))
        return processor

    monkeypatch.setattr(
        glm5next_processor.Glm5NextProcessor,
        "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    info = SimpleNamespace(
        ctx=SimpleNamespace(
            model_config=SimpleNamespace(
                model="local-inference-lab/GLM-5.3-Flash-NVFP4",
                revision="checkpoint-commit",
            )
        )
    )

    assert Glm5NextProcessingInfo.get_hf_processor(info) is processor
    assert Glm5NextProcessingInfo.get_hf_processor(info) is processor
    assert calls == [
        (
            "local-inference-lab/GLM-5.3-Flash-NVFP4",
            {"revision": "checkpoint-commit"},
        )
    ]


def test_glm5next_processor_counts_video_only_tokens() -> None:
    class FakeVideoProcessor:
        merge_size = 2

        @staticmethod
        def get_number_of_video_patches(*args) -> int:
            return 20

    processor = glm5next_processor.Glm5NextProcessor.__new__(
        glm5next_processor.Glm5NextProcessor
    )
    processor.video_processor = FakeVideoProcessor()

    actual = processor._get_num_multimodal_tokens(video_sizes=[(1, 2, 3)])

    assert actual.num_video_tokens == [5]


def test_glm5next_fused_conv1d_loads_three_logical_shards() -> None:
    param = torch.nn.Parameter(torch.empty(12, 1, 4))
    loaded = torch.arange(12 * 4).reshape(12, 1, 4)
    calls = []

    def weight_loader(param, loaded_weight, shard_id) -> None:
        calls.append((param, loaded_weight.clone(), shard_id))

    param.weight_loader = weight_loader

    _load_glm5next_fused_conv1d(param, loaded)

    assert [shard_id for _, _, shard_id in calls] == [0, 1, 2]
    assert all(loaded_param is param for loaded_param, _, _ in calls)
    assert torch.equal(calls[0][1], loaded[:4])
    assert torch.equal(calls[1][1], loaded[4:8])
    assert torch.equal(calls[2][1], loaded[8:])


def test_glm5next_loads_separate_conv1d_shards() -> None:
    param = torch.nn.Parameter(torch.empty(12, 1, 4))
    calls = []

    def weight_loader(param, loaded_weight, shard_id) -> None:
        calls.append((param, loaded_weight.clone(), shard_id))

    param.weight_loader = weight_loader

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                is_moe=False,
                is_linear_attn=True,
                mla_nope=False,
                qk_rope_head_dim=0,
            )

        def named_parameters(self):
            return iter([("layers.0.self_attn.conv1d.weight", param)])

    weights = [
        (f"layers.0.self_attn.{name}_conv1d.weight", torch.full((4, 1, 4), i))
        for i, name in enumerate(("q", "k", "v"))
    ]

    loaded_params = Glm5NextModel.load_weights(FakeModel(), weights)

    assert loaded_params == {"layers.0.self_attn.conv1d.weight"}
    assert [shard_id for _, _, shard_id in calls] == [0, 1, 2]
    assert all(loaded_param is param for loaded_param, _, _ in calls)
    assert all(
        torch.equal(weight, weights[i][1]) for i, (_, weight, _) in enumerate(calls)
    )


def test_glm5next_mtp_uses_draft_kernel_overrides() -> None:
    @dataclass
    class KernelConfig:
        moe_backend: str

    @dataclass
    class AttentionConfig:
        backend: AttentionBackendEnum | None

    @dataclass
    class CacheConfig:
        cache_dtype: str

    @dataclass
    class VllmConfig:
        kernel_config: KernelConfig
        attention_config: AttentionConfig
        cache_config: CacheConfig
        speculative_config: SimpleNamespace

    target_config = VllmConfig(
        kernel_config=KernelConfig(moe_backend="b12x"),
        attention_config=AttentionConfig(backend=AttentionBackendEnum.FLASH_ATTN),
        cache_config=CacheConfig(cache_dtype="fp8"),
        speculative_config=SimpleNamespace(
            moe_backend="humming",
            attention_backend=AttentionBackendEnum.B12X,
            kv_cache_dtype=None,
        ),
    )

    draft_config = _make_eagle_draft_vllm_config(target_config)  # type: ignore[arg-type]

    assert draft_config.kernel_config.moe_backend == "humming"
    assert draft_config.attention_config.backend == AttentionBackendEnum.B12X
    assert target_config.kernel_config.moe_backend == "b12x"
    assert target_config.attention_config.backend == AttentionBackendEnum.FLASH_ATTN


def test_glm5next_mtp_maps_multimodal_quantization_prefix() -> None:
    quantized_layers = {
        "model.language_model.layers.45.mlp.experts": {"quant_algo": "MXFP8"}
    }

    mapped = Glm5NextMTP.hf_to_vllm_mapper.apply_dict(quantized_layers)

    assert mapped == {"model.layers.45.mlp.experts": {"quant_algo": "MXFP8"}}


def test_glm5next_mtp_reuses_wrapped_mla_topk_indices() -> None:
    topk_indices_buffer = torch.tensor(
        [[10, 11], [20, 21], [30, 31], [40, 41]], dtype=torch.int32
    )
    mla_attn = SimpleNamespace(
        skip_topk=False,
        topk_indices_buffer=topk_indices_buffer,
    )
    predictor = SimpleNamespace(
        layers={
            "45": SimpleNamespace(
                mtp_block=SimpleNamespace(self_attn=SimpleNamespace(mla_attn=mla_attn))
            )
        }
    )

    Glm5NextMultiTokenPredictor.set_skip_topk(predictor, True)
    Glm5NextMultiTokenPredictor.compact_topk_indices(
        predictor, torch.tensor([2, 0], dtype=torch.int64)
    )

    assert mla_attn.skip_topk
    assert torch.equal(
        mla_attn.topk_indices_buffer[:2],
        torch.tensor([[30, 31], [10, 11]], dtype=torch.int32),
    )


def test_glm5next_mtp_rolls_back_selector_interval_starts() -> None:
    calls: list[str] = []

    class FakeIndexer:
        def snapshot_speculative_interval_starts(self) -> None:
            calls.append("snapshot")

        def restore_speculative_interval_starts(self) -> None:
            calls.append("restore")

    predictor = SimpleNamespace(
        layers={
            "45": SimpleNamespace(
                mtp_block=SimpleNamespace(
                    self_attn=SimpleNamespace(indexer=FakeIndexer())
                )
            ),
            "46": SimpleNamespace(
                mtp_block=SimpleNamespace(self_attn=SimpleNamespace(indexer=None))
            ),
        }
    )

    Glm5NextMultiTokenPredictor.snapshot_qsa_interval_starts(predictor)
    Glm5NextMultiTokenPredictor.restore_qsa_interval_starts(predictor)

    assert calls == ["snapshot", "restore"]


def test_glm5next_mtp_maps_target_normalized_quantization_prefix() -> None:
    quantized_layers = {
        "model.language_model.layers.45.mlp.experts": {"quant_algo": "MXFP8"}
    }
    target_mapped = Glm4vForConditionalGeneration.hf_to_vllm_mapper.apply_dict(
        quantized_layers
    )

    mapped = Glm5NextMTP.hf_to_vllm_mapper.apply_dict(target_mapped)

    assert mapped == {"model.layers.45.mlp.experts": {"quant_algo": "MXFP8"}}


@pytest.mark.parametrize("map_through_target", [False, True])
def test_glm5next_mtp_resolves_mxfp8_quantization(
    map_through_target: bool,
) -> None:
    quantized_layers = {
        "model.language_model.layers.45.mlp.experts": {"quant_algo": "MXFP8"}
    }
    if map_through_target:
        quantized_layers = Glm4vForConditionalGeneration.hf_to_vllm_mapper.apply_dict(
            quantized_layers
        )
    quantized_layers = Glm5NextMTP.hf_to_vllm_mapper.apply_dict(quantized_layers)
    quant_config = ModelOptMixedPrecisionConfig.__new__(ModelOptMixedPrecisionConfig)
    quant_config.quantized_layers = quantized_layers
    quant_config.packed_modules_mapping = {}

    assert quant_config._resolve_quant_algo("model.layers.45.mlp.experts") == "MXFP8"
