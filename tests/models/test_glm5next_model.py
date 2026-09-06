# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

from vllm.model_executor.layers import mla as mla_layer
from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
    resolve_kda_prefill_backend,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
)
from vllm.model_executor.models.glm4_1v import Glm4vForConditionalGeneration
from vllm.model_executor.models.interfaces import supports_eagle3, supports_pp
from vllm.models.glm5next.nvidia import attention as glm5next_attention
from vllm.models.glm5next.nvidia import model as glm5next_model
from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention
from vllm.models.glm5next.nvidia.model import (
    GLM5NEXT_PACKED_MODULES_MAPPING,
    Glm5NextDecoderLayer,
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
    Glm5NextMoE,
    _load_glm5next_fused_conv1d,
    _remap_glm5next_weight_name,
    _try_load_fp8_attn_proj,
)
from vllm.models.glm5next.nvidia.mtp import (
    Glm5NextMTP,
    Glm5NextMultiTokenPredictor,
    Glm5NextMultiTokenPredictorLayer,
)
from vllm.platforms import current_platform
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
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
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


def test_glm5next_mixed_precision_resolves_fused_attention_projections() -> None:
    quant_config = ModelOptMixedPrecisionConfig.__new__(ModelOptMixedPrecisionConfig)
    quant_config.exclude_modules = []
    quant_config.quantized_layers = {
        f"model.language_model.layers.{layer}.self_attn.{projection}": {
            "quant_algo": "MXFP8"
        }
        for layer, projection in (
            (0, "q_proj"),
            (0, "k_proj"),
            (0, "v_proj"),
            (0, "b_proj"),
            (0, "f_a_proj"),
            (3, "q_a_proj"),
            (3, "kv_a_proj_with_mqa"),
        )
    }
    quant_config.packed_modules_mapping = GLM5NEXT_PACKED_MODULES_MAPPING
    quant_config.apply_vllm_mapper(
        Glm5NextForConditionalGeneration.hf_to_vllm_mapper.get_rename_mapper()
    )

    assert (
        quant_config._resolve_quant_algo(
            "language_model.model.layers.0.self_attn.in_proj_qkvgfab"
        )
        == "MXFP8"
    )
    assert (
        quant_config._resolve_quant_algo(
            "language_model.model.layers.3.self_attn.fused_qkv_a_proj"
        )
        == "MXFP8"
    )
    assert Glm5NextForCausalLM.packed_modules_mapping is GLM5NEXT_PACKED_MODULES_MAPPING
    assert (
        Glm5NextForConditionalGeneration.packed_modules_mapping
        is GLM5NEXT_PACKED_MODULES_MAPPING
    )
    assert Glm5NextMTP.packed_modules_mapping is GLM5NEXT_PACKED_MODULES_MAPPING


def test_glm5next_mtp_resolves_mapped_mxfp8_projection() -> None:
    quant_config = ModelOptMixedPrecisionConfig.__new__(ModelOptMixedPrecisionConfig)
    quant_config.exclude_modules = []
    quant_config.quantized_layers = {
        f"model.language_model.layers.45.self_attn.{projection}": {
            "quant_algo": "MXFP8"
        }
        for projection in ("q_a_proj", "kv_a_proj_with_mqa")
    }
    quant_config.packed_modules_mapping = GLM5NEXT_PACKED_MODULES_MAPPING
    quant_config.apply_vllm_mapper(Glm5NextMTP.hf_to_vllm_mapper.get_rename_mapper())

    assert (
        quant_config._resolve_quant_algo("model.layers.45.self_attn.fused_qkv_a_proj")
        == "MXFP8"
    )


def test_glm5next_mtp_selects_only_draft_checkpoint_weights() -> None:
    mtp = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1),
        has_own_lm_head=False,
    )

    assert Glm5NextMTP._checkpoint_weight_name_prefixes(mtp) == (
        "model.language_model.layers.45.",
        "language_model.model.layers.45.",
        "model.layers.45.",
        "layers.45.",
    )


@pytest.fixture
def glm_mtp_head_loader(monkeypatch):
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.models.glm5next.nvidia import mtp as mtp_module

    monkeypatch.setattr(
        mtp_module, "fused_moe_make_expert_params_mapping", lambda *a, **kw: []
    )
    model = Glm5NextMTP.__new__(Glm5NextMTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        n_routed_experts=0,
        mla_nope=False,
    )
    model.has_own_lm_head = True
    layer = torch.nn.Module()
    layer.shared_head = torch.nn.Module()
    layer.shared_head.head = torch.nn.Linear(4, 8, bias=False)
    layer.shared_head.head.weight.weight_loader = default_weight_loader
    layer.shared_head.norm = torch.nn.LayerNorm(4, bias=False)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleDict({"45": layer})
    model.model.mtp_start_layer_idx = 45
    model.model.num_mtp_layers = 1
    return model


@pytest.mark.parametrize("draft_order", [None, "before", "after"])
@pytest.mark.parametrize("target_prefix", ["", "model.language_model."])
def test_glm5next_mtp_loads_target_head_fallback_and_prefers_draft_weights(
    glm_mtp_head_loader, draft_order, target_prefix
) -> None:
    model = glm_mtp_head_loader
    target = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    draft = target + 100
    head_name = "model.layers.45.shared_head.head.weight"
    weights = [
        ("model.language_model.layers.45.shared_head.norm.weight", torch.ones(4)),
        (f"{target_prefix}lm_head.weight", target),
    ]
    if draft_order is not None:
        weights.insert(
            0 if draft_order == "before" else len(weights), (head_name, draft)
        )

    loaded = model.load_weights(weights)

    assert head_name in loaded
    torch.testing.assert_close(
        model.model.layers["45"].shared_head.head.weight,
        target if draft_order is None else draft,
    )
    assert f"{target_prefix}lm_head." in model._checkpoint_weight_name_prefixes()


@pytest.mark.parametrize("missing_head", [False, True])
def test_glm5next_mtp_requires_both_draft_layer_and_head_weights(
    glm_mtp_head_loader, missing_head
) -> None:
    weights = (
        [("model.layers.45.shared_head.norm.weight", torch.ones(4))]
        if missing_head
        else [("lm_head.weight", torch.ones(8, 4))]
    )
    match = "requires an unquantized" if missing_head else "layer 45 weights missing"
    with pytest.raises(ValueError, match=match):
        glm_mtp_head_loader.load_weights(weights)


def test_glm5next_mtp_preserves_position_zero_embedding() -> None:
    class CaptureProjection(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inputs: torch.Tensor | None = None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.inputs = inputs
            return inputs

    class IdentityBlock(torch.nn.Module):
        def forward(
            self,
            *,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
            output_indices: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
            return hidden_states, torch.zeros_like(hidden_states), None, None

    class IdentityHead(torch.nn.Module):
        def norm(
            self, hidden_states: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, None]:
            return hidden_states, None

    layer = Glm5NextMultiTokenPredictorLayer.__new__(Glm5NextMultiTokenPredictorLayer)
    torch.nn.Module.__init__(layer)
    projection = CaptureProjection()
    layer.enorm = torch.nn.Identity()
    layer.hnorm = torch.nn.Identity()
    layer.eh_proj = projection
    layer.mtp_block = IdentityBlock()
    layer.shared_head = IdentityHead()

    inputs_embeds = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    previous_hidden_states = torch.zeros_like(inputs_embeds)
    layer(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        previous_hidden_states=previous_hidden_states,
        inputs_embeds=inputs_embeds,
    )

    assert projection.inputs is not None
    torch.testing.assert_close(projection.inputs[:, :2], inputs_embeds)


def test_glm5next_mixed_precision_reaches_mla_projections(monkeypatch) -> None:
    captured = {}

    class FakeModule(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    class FakeMLAAttention(FakeModule):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(glm5next_model, "Glm5NextMLAAttention", FakeMLAAttention)
    monkeypatch.setattr(glm5next_model, "Glm5NextMLP", FakeModule)
    monkeypatch.setattr(glm5next_model, "RMSNorm", FakeModule)

    quant_config = ModelOptMixedPrecisionConfig.__new__(ModelOptMixedPrecisionConfig)
    config = SimpleNamespace(
        hidden_size=16,
        is_moe=False,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        n_routed_experts=None,
        mhc=False,
        is_kda_layer=lambda layer_idx: False,
        v_head_dim=4,
        kv_lora_rank=4,
        num_attention_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=0,
        q_lora_rank=4,
        max_position_embeddings=128,
        mla_nope=True,
        mlp_layer_types=["dense"],
        intermediate_size=32,
        hidden_act="silu",
        swiglu_limit=None,
    )
    vllm_config = SimpleNamespace(
        cache_config=None,
        quant_config=quant_config,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
    )

    Glm5NextDecoderLayer(vllm_config, config, 0, prefix="model.layers.0")

    assert captured["quant_config"] is quant_config


def test_glm5next_fp8_dequantizer_yields_to_mxfp8_projection() -> None:
    layer_prefix = "layers.3.self_attn"
    params_dict = {
        f"{layer_prefix}.fused_qkv_a_proj.weight": torch.nn.Parameter(torch.empty(1)),
        f"{layer_prefix}.fused_qkv_a_proj.weight_scale": torch.nn.Parameter(
            torch.empty(1)
        ),
    }
    pending: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

    handled = _try_load_fp8_attn_proj(
        f"{layer_prefix}.q_a_proj.weight",
        torch.empty((1, 1), dtype=torch.float8_e4m3fn),
        pending,
        params_dict,
        set(),
        0,
    )

    assert not handled
    assert not pending


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("reuse_buffer", [False, True])
@pytest.mark.parametrize("quantization", ["mxfp8", "block_fp8"])
def test_glm5next_dequantizes_buffered_attention_projections(
    scale_first: bool, reuse_buffer: bool, quantization: str
) -> None:
    """Paired tensors must survive a streaming loader reusing its buffers."""
    layer_prefix = "layers.3.self_attn"
    if quantization == "mxfp8":
        projection = "indexer.weights_proj"
        target = projection
        scale_name = "weight_scale"
        scale = torch.full((1, 1), 128, dtype=torch.uint8)
    else:
        projection = "q_a_proj"
        target = "fused_qkv_a_proj"
        scale_name = "weight_scale_inv"
        scale = torch.full((1, 1), 2.0, dtype=torch.float32)
    parameter_name = f"{layer_prefix}.{target}.weight"
    param = torch.nn.Parameter(torch.empty((1, 32), dtype=torch.bfloat16))

    def weight_loader(param, loaded_weight, shard_id=None) -> None:
        param.data.copy_(loaded_weight)

    param.weight_loader = weight_loader
    weight = (
        torch.arange(32, dtype=torch.float32).reshape(1, 32).to(torch.float8_e4m3fn)
    )
    expected = weight.to(torch.bfloat16) * 2

    class FakeModel(torch.nn.Module):
        config = SimpleNamespace(
            is_moe=False,
            is_linear_attn=True,
            mla_nope=False,
            qk_rope_head_dim=0,
        )

        def __init__(self) -> None:
            super().__init__()

        def named_parameters(self):
            return iter([(parameter_name, param)])

    pairs = [
        (f"{layer_prefix}.{projection}.weight", weight),
        (f"{layer_prefix}.{projection}.{scale_name}", scale),
    ]
    if scale_first:
        pairs.reverse()

    def weights():
        for name, tensor in pairs:
            yield name, tensor
            if reuse_buffer:
                tensor.zero_()

    loaded_params = Glm5NextModel.load_weights(FakeModel(), weights())

    assert loaded_params == {parameter_name}
    torch.testing.assert_close(param, expected)


@pytest.mark.parametrize(
    ("checkpoint_name", "parameter_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_proj.weight_scale",
            "layers.0.self_attn.in_proj_qkvgfab.weight_scale",
            0,
        ),
        (
            "layers.3.self_attn.q_a_proj.weight_scale",
            "layers.3.self_attn.fused_qkv_a_proj.weight_scale",
            0,
        ),
    ],
)
def test_glm5next_loads_mxfp8_fused_projection_scales(
    checkpoint_name: str,
    parameter_name: str,
    shard_id: int,
) -> None:
    calls = []
    param = torch.nn.Parameter(torch.empty(1))

    def weight_loader(param, loaded_weight, loaded_shard_id) -> None:
        calls.append((param, loaded_weight, loaded_shard_id))

    param.weight_loader = weight_loader

    class FakeModel(torch.nn.Module):
        config = SimpleNamespace(
            is_moe=False,
            is_linear_attn=True,
            mla_nope=False,
            qk_rope_head_dim=0,
        )

        def __init__(self) -> None:
            super().__init__()

        def named_parameters(self):
            return iter([(parameter_name, param)])

    loaded_weight = torch.ones(1, dtype=torch.uint8)
    loaded_params = Glm5NextModel.load_weights(
        FakeModel(), [(checkpoint_name, loaded_weight)]
    )

    assert loaded_params == {parameter_name}
    assert len(calls) == 1
    loaded_param, actual_weight, actual_shard_id = calls[0]
    assert loaded_param is param
    assert actual_weight is loaded_weight
    assert actual_shard_id == shard_id


def test_glm5next_kda_adapts_shared_out_buffer_forward(monkeypatch) -> None:
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.use_full_rank_gate = True
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


class _FakeKdaPrefillApi:
    """Record what the layer binds and runs through the b12x prefill op."""

    def __init__(self, calls: dict[str, object]) -> None:
        self._calls = calls

    def bind(self, plan, **kwargs):
        del plan
        self._calls["prefill_q_len"] = int(kwargs["q"].shape[0])
        self._calls["prefill_query_start_loc"] = kwargs["cu_seqlens"].clone()
        for name in (
            "initial_state_indices",
            "final_state_indices",
            "checkpoint_state_indices",
            "checkpoint_offsets",
        ):
            self._calls[name] = kwargs[name].clone()
        self._calls["num_seqs"] = int(kwargs["num_seqs"].item())
        self._calls["num_tokens"] = int(kwargs["num_tokens"].item())
        scratch_start = kwargs["scratch"].data_ptr()
        scratch_end = scratch_start + kwargs["scratch"].nbytes
        output_start = kwargs["output"].data_ptr()
        output_end = output_start + kwargs["output"].nbytes
        self._calls["scratch_output_overlap"] = (
            scratch_start < output_end and output_start < scratch_end
        )
        return SimpleNamespace(
            output=kwargs["output"], recurrent_state=kwargs["recurrent_state"]
        )

    def run(self, binding, *, lower_bound, max_live_tokens, max_live_seqs):
        self._calls["lower_bound"] = lower_bound
        self._calls["max_live_tokens"] = max_live_tokens
        self._calls["max_live_seqs"] = max_live_seqs
        binding.output.fill_(22)
        binding.recurrent_state[3] = 33
        return binding.output


@pytest.mark.parametrize("prefill_backend", ["triton", "flashkda", "b12x"])
def test_glm5next_kda_splits_mixed_decode_prefill_batch(
    monkeypatch, prefill_backend: str
) -> None:
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
    layer.kda_prefill_backend = prefill_backend
    layer._flashkda_buffer_specs = (
        ((1, 4, 1, 1), torch.float32),
        ((1, 1, 1, 1), torch.float32),
        ((1, 1, 1, 1), torch.float32),
        ((1,), torch.uint8),
    )
    layer.A_log = torch.ones(1)
    layer.dt_bias = torch.ones(1)
    layer._b12x_kda_plan = None
    layer._b12x_kda_scratch = None
    layer.model_config = SimpleNamespace(dtype=torch.float32)
    layer._b12x_prefill_api = None
    layer._b12x_prefill_plan = None
    layer._b12x_prefill_max_tokens = 4
    layer._b12x_prefill_max_seqs = 2
    if prefill_backend == "b12x":
        calls_sink: dict[str, object] = {}
        layer._b12x_prefill_api = _FakeKdaPrefillApi(calls_sink)
        layer._b12x_prefill_plan = SimpleNamespace(
            scratch_specs=lambda: (SimpleNamespace(shape=(16,), dtype=torch.uint8),)
        )
        layer._b12x_prefill_num_seqs = torch.zeros(1, dtype=torch.int32)
        layer._b12x_prefill_num_tokens = torch.zeros(1, dtype=torch.int32)
        layer._b12x_prefill_initial_indices = torch.zeros(2, dtype=torch.int32)
        layer._b12x_prefill_null_indices = torch.full(
            (2,), NULL_BLOCK_ID, dtype=torch.int32
        )
        layer._b12x_prefill_zero_offsets = torch.zeros(2, dtype=torch.int32)
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
    if prefill_backend == "b12x":
        calls = calls_sink  # noqa: F821 - bound above for this backend

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

    def fake_flashkda(
        *, q, beta, initial_state, cu_seqlens, out, final_state, **kwargs
    ):
        calls["prefill_q_len"] = q.shape[1]
        calls["prefill_query_start_loc"] = cu_seqlens.clone()
        calls["prefill_beta"] = beta.clone()
        out.fill_(22)
        final_state.fill_(33)
        return out, final_state

    class FakeWorkspaceManager:
        def __init__(self) -> None:
            self.storage = torch.empty(1024, dtype=torch.uint8)

        def get_simultaneous(self, *specs):
            outputs = []
            offset = 0
            for shape, dtype in specs:
                nbytes = math.prod(shape) * dtype.itemsize
                outputs.append(
                    self.storage[offset : offset + nbytes].view(dtype).view(shape)
                )
                offset += nbytes
            return outputs

    workspace_manager = FakeWorkspaceManager()

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
    monkeypatch.setattr(kimi_gdn_linear_attn, "_flashkda_prefill", fake_flashkda)
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "current_workspace_manager",
        lambda: workspace_manager,
    )
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
    if prefill_backend == "b12x":
        # The op addresses state slots directly: no dense gather, and the
        # pool keeps whatever the op wrote rather than a scattered result.
        assert "prefill_state_indices" not in calls
        assert torch.equal(
            calls["initial_state_indices"],
            torch.tensor([NULL_BLOCK_ID], dtype=torch.int32),
        )
        assert torch.equal(
            calls["final_state_indices"], torch.tensor([3], dtype=torch.int32)
        )
        assert calls["num_seqs"] == 1
        assert calls["num_tokens"] == 2
        assert calls["max_live_tokens"] == 2
        assert calls["max_live_seqs"] == 1
        assert calls["lower_bound"] == -5.0
        assert not calls["scratch_output_overlap"]
        assert torch.equal(core_attn_out[:, :2], torch.full((1, 2, 1, 1), 11.0))
        assert torch.equal(core_attn_out[:, 2:], torch.full((1, 2, 1, 1), 22.0))
        assert torch.equal(layer.kv_cache[1][3], torch.full((1, 1, 1), 33.0))
        return
    if prefill_backend == "triton":
        assert calls["chunk_indices"] is chunk_indices
        assert calls["chunk_offsets"] is chunk_offsets
    else:
        assert torch.equal(calls["prefill_beta"], torch.ones(1, 2, 1))
    assert torch.equal(
        calls["prefill_state_indices"], torch.tensor([3], dtype=torch.int32)
    )
    assert torch.equal(calls["prefill_has_initial_state"], torch.tensor([False]))
    assert torch.equal(core_attn_out[:, :2], torch.full((1, 2, 1, 1), 11.0))
    assert torch.equal(core_attn_out[:, 2:], torch.full((1, 2, 1, 1), 22.0))
    assert torch.equal(layer.kv_cache[1][3], torch.full((1, 1, 1), 33.0))


@pytest.mark.parametrize(
    ("configured", "supported", "expected"),
    [
        ("auto", True, "flashkda"),
        ("auto", False, "triton"),
        ("flashkda", True, "flashkda"),
        ("triton", True, "triton"),
    ],
)
def test_glm5next_kda_prefill_backend_resolution(
    monkeypatch, configured: str, supported: bool, expected: str
) -> None:
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "is_flashkda_supported",
        lambda *args: supported,
    )

    assert (
        resolve_kda_prefill_backend(configured, 128, torch.bfloat16, -5.0) == expected
    )


def test_glm5next_explicit_flashkda_rejects_unsupported_layer(monkeypatch) -> None:
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "is_flashkda_supported",
        lambda *args: False,
    )

    with pytest.raises(RuntimeError, match="FlashKDA requires"):
        resolve_kda_prefill_backend("flashkda", 64, torch.float16, None)


def test_glm5next_kda_prefill_backend_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unsupported KDA prefill backend"):
        resolve_kda_prefill_backend("unknown", 128, torch.bfloat16, -5.0)


def test_glm5next_kda_prefill_backend_selects_b12x_by_name(monkeypatch) -> None:
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "is_b12x_kda_prefill_supported",
        lambda *args: True,
    )
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "is_flashkda_supported",
        lambda *args: True,
    )

    assert (
        resolve_kda_prefill_backend("b12x", 128, torch.bfloat16, -5.0, torch.float32)
        == "b12x"
    )
    # auto keeps FlashKDA until the b12x serving qualification lands.
    assert (
        resolve_kda_prefill_backend("auto", 128, torch.bfloat16, -5.0, torch.float32)
        == "flashkda"
    )


def test_glm5next_explicit_b12x_rejects_unsupported_layer(monkeypatch) -> None:
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "is_b12x_kda_prefill_supported",
        lambda *args: False,
    )

    with pytest.raises(RuntimeError, match="b12x KDA prefill backend requires"):
        resolve_kda_prefill_backend("b12x", 128, torch.bfloat16, -5.0, torch.float32)


def test_kimi_k3_kda_prefill_rejects_b12x() -> None:
    """Only the GLM linear-attention layer offers the b12x prefill backend."""
    from vllm.models.kimi_k3.nvidia import kda as kimi_k3_kda

    with pytest.raises(ValueError, match="Unsupported KDA prefill backend"):
        kimi_k3_kda.resolve_kda_prefill_backend("b12x", 128, torch.bfloat16, -5.0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_glm5next_b12x_kda_prefill_matches_the_triton_chunk_path() -> None:
    """The b12x prefill path agrees with the Triton chunk path it replaces.

    Covers the contract the two share: the gate and beta activations, the
    recurrent-state orientation, and slot addressing in place of the dense
    gather and scatter.
    """
    b12x_api = pytest.importorskip("b12x.sequence.kda_prefill")
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )

    if not b12x_api.is_supported(torch.device("cuda", 0)):
        pytest.skip("b12x KDA prefill is unsupported on this device")

    torch.manual_seed(7)
    device = torch.device("cuda", 0)
    heads, head_dim, lower_bound = 4, 128, -5.0
    lengths = [37, 128, 1]
    slots = [3, 5, 7]
    has_initial = [True, False, True]
    tokens, requests = sum(lengths), len(lengths)

    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    q, k, v, raw_g = (
        torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device=device)
        for _ in range(4)
    )
    raw_beta = torch.randn(tokens, heads, dtype=torch.bfloat16, device=device)
    a_log = torch.randn(heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(heads * head_dim, dtype=torch.float32, device=device)
    pool = torch.randn(
        16, heads, head_dim, head_dim, dtype=torch.float32, device=device
    )
    pool[NULL_BLOCK_ID].zero_()
    state_indices = torch.tensor(slots, dtype=torch.int32, device=device)
    initial_mask = torch.tensor(has_initial, dtype=torch.bool, device=device)

    # The Triton path overwrites its value tensor, so it gets copies.
    reference_pool = pool.clone()
    initial_state = torch.where(
        initial_mask[:, None, None, None],
        reference_pool[state_indices.long()],
        torch.zeros_like(reference_pool[:requests]),
    )
    reference_out, reference_last = chunk_kda_with_fused_gate(
        q=q.clone().unsqueeze(0),
        k=k.clone().unsqueeze(0),
        v=v.clone().unsqueeze(0),
        raw_g=raw_g.clone().unsqueeze(0),
        raw_beta=raw_beta.clone().unsqueeze(0),
        A_log=a_log,
        g_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    reference_pool[state_indices.long()] = reference_last

    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.head_dim, layer.local_num_heads = head_dim, heads
    layer.gate_lower_bound = lower_bound
    layer.A_log, layer.dt_bias = a_log, dt_bias
    layer._b12x_prefill_api = b12x_api
    layer._b12x_prefill_max_tokens = 512
    layer._b12x_prefill_max_seqs = 8
    plan = b12x_api.plan(
        b12x_api.Caps(
            device=device,
            max_tokens=512,
            max_seqs=8,
            max_state_slots=int(pool.shape[0]),
            heads=heads,
            head_dim=head_dim,
            model_dtype=torch.bfloat16,
            state_dtype=torch.float32,
            qk_l2norm=True,
            checkpoint_export=True,
            null_state_index=NULL_BLOCK_ID,
            metadata_validation="trusted",
        )
    )
    layer._b12x_prefill_plan = plan
    scratch = torch.empty(
        plan.scratch_specs()[0].shape, dtype=torch.uint8, device=device
    )
    layer._b12x_prefill_num_seqs = torch.zeros(1, dtype=torch.int32, device=device)
    layer._b12x_prefill_num_tokens = torch.zeros(1, dtype=torch.int32, device=device)
    layer._b12x_prefill_initial_indices = torch.zeros(
        8, dtype=torch.int32, device=device
    )
    layer._b12x_prefill_null_indices = torch.full(
        (8,), NULL_BLOCK_ID, dtype=torch.int32, device=device
    )
    layer._b12x_prefill_zero_offsets = torch.zeros(8, dtype=torch.int32, device=device)

    got_pool = pool.clone()
    out = torch.zeros(tokens, heads, head_dim, dtype=torch.bfloat16, device=device)
    layer._run_b12x_kda_prefill(
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        raw_beta=raw_beta,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        has_initial_state=initial_mask,
        checkpoint=None,
        recurrent_state=got_pool,
        scratch=scratch,
        output=out,
    )
    torch.accelerator.synchronize(device)

    def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
        got, want = got.float(), want.float()
        scale = want.pow(2).mean().sqrt().clamp_min(1e-12)
        return ((got - want).pow(2).mean().sqrt() / scale).item()

    assert relative_error(out, reference_out[0]) < 1e-2
    for slot in slots:
        assert relative_error(got_pool[slot], reference_pool[slot]) < 1e-2
    untouched = [slot for slot in range(pool.shape[0]) if slot not in slots]
    assert torch.equal(got_pool[untouched], pool[untouched])


@pytest.mark.parametrize("request_boundaries", [False, True])
def test_glm5next_b12x_prefill_requests_a_checkpoint_block(
    request_boundaries: bool,
) -> None:
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    spec = SimpleNamespace(num_prefill_checkpoint_blocks=0)

    def fake_super_spec(self, vllm_config):
        del self, vllm_config
        return spec

    for backend, expected in (("b12x", 1), ("flashkda", 1), ("triton", 0)):
        layer.kda_prefill_backend = backend
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                kimi_gdn_linear_attn.GatedDeltaNetAttention,
                "get_kv_cache_spec",
                fake_super_spec,
            )
            patch.setattr(kimi_gdn_linear_attn, "MambaSpec", SimpleNamespace)
            patch.setattr(
                kimi_gdn_linear_attn,
                "replace",
                lambda base, **kwargs: SimpleNamespace(**kwargs),
            )
            resolved = layer.get_kv_cache_spec(
                SimpleNamespace(use_request_boundary_checkpoints=request_boundaries)
            )
        assert resolved.num_prefill_checkpoint_blocks == (
            0 if request_boundaries else expected
        ), backend


def test_glm5next_alone_opts_into_b12x_kda_decode() -> None:
    assert not KimiGatedDeltaNetAttention.enable_b12x_kda_decode
    assert Glm5NextLinearAttention.enable_b12x_kda_decode
    assert KimiGatedDeltaNetAttention.b12x_kda_null_state_index is None
    assert Glm5NextLinearAttention.b12x_kda_null_state_index == 0


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


def test_glm5next_mtp_compacts_outputs_after_attention_before_moe() -> None:
    class Attention(torch.nn.Module):
        def forward(self, hidden_states, positions):
            return hidden_states + positions.unsqueeze(1)

    class AddNorm(torch.nn.Module):
        def forward(self, hidden_states, residual):
            return hidden_states + residual, residual

    class RecordingMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_input_tokens = 0

        def forward(self, hidden_states):
            self.num_input_tokens = hidden_states.shape[0]
            return hidden_states * 2

    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.mhc = False
    layer.is_mtp_layer = True
    layer.input_layernorm = torch.nn.Identity()
    layer.self_attn = Attention()
    layer.post_attention_layernorm = AddNorm()
    layer.mlp = RecordingMLP()

    hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    positions = torch.arange(5, dtype=torch.float32)
    output_indices = torch.tensor([1, 4])

    full_output, full_residual, _, _ = layer(positions, hidden_states)
    compact_output, compact_residual, _, _ = layer(
        positions, hidden_states, output_indices=output_indices
    )

    torch.testing.assert_close(compact_output, full_output[output_indices])
    torch.testing.assert_close(compact_residual, full_residual[output_indices])
    assert layer.mlp.num_input_tokens == output_indices.numel()


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
    assert captured_caps["kda_metadata_validation"] == "trusted"


@pytest.mark.parametrize(
    "speculative,uniform", [(False, False), (True, False), (True, True)]
)
def test_b12x_kda_shares_counts_but_preserves_each_layers_state_indices(
    monkeypatch, speculative: bool, uniform: bool
) -> None:
    calls: dict[str, list] = {"bind": [], "run": []}

    class FakeApi:
        @staticmethod
        def bind_kda(plan, **kwargs):
            binding = SimpleNamespace(plan=plan, **kwargs)
            calls["bind"].append(binding)
            return binding

        @staticmethod
        def run_kda(binding, **kwargs):
            calls["run"].append((binding, kwargs))

    forward_context = SimpleNamespace(additional_kwargs={})
    monkeypatch.setattr(
        kimi_gdn_linear_attn,
        "get_forward_context",
        lambda: forward_context,
    )

    plan = SimpleNamespace(caps=SimpleNamespace(max_state_slots=32))
    api = FakeApi()

    def make_layer():
        layer = KimiGatedDeltaNetAttention.__new__(KimiGatedDeltaNetAttention)
        torch.nn.Module.__init__(layer)
        layer._b12x_kda_api = api
        layer._b12x_kda_plan = plan
        layer._b12x_kda_scratch = torch.empty(1)
        layer._b12x_kda_num_accepted_tokens = torch.zeros(2, dtype=torch.int32)
        layer._b12x_kda_num_seqs = torch.zeros(1, dtype=torch.int32)
        layer._b12x_kda_num_tokens = torch.zeros(1, dtype=torch.int32)
        layer._b12x_kda_max_tokens = 2
        layer._b12x_kda_max_seqs = 2
        layer._b12x_kda_state_index_columns = 1
        layer.gate_lower_bound = -5.0
        layer.local_num_heads = 1
        layer.head_dim = 1
        layer.A_log = torch.ones(1)
        layer.dt_bias = torch.ones(1)
        layer.o_norm = SimpleNamespace(eps=1e-6, weight=torch.ones(1))
        layer.kv_cache = [None, torch.empty(32, 1, 1, 1)]
        return layer

    layers = [make_layer(), make_layer()]
    mixed_qkv = torch.arange(6, dtype=torch.float32).view(2, 3)
    raw_g = torch.ones(2, 1, 1)
    raw_beta = torch.ones(2, 1)
    z = torch.ones(2, 1, 1)
    outputs = [torch.empty(2, 1, 1), torch.empty(2, 1, 1)]
    state_indices = [
        torch.tensor([[3], [4]], dtype=torch.int32),
        torch.tensor([[13], [14]], dtype=torch.int32),
    ]
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    accepted = torch.tensor([2, 1], dtype=torch.int32) if speculative else None

    for layer, output, indices in zip(layers, outputs, state_indices):
        layer._run_b12x_kda_decode_post_conv(
            metadata=SimpleNamespace(is_uniform_spec_decode=uniform),
            mixed_qkv=mixed_qkv,
            raw_g=raw_g,
            raw_beta=raw_beta,
            z=z,
            output=output,
            state_indices=indices,
            query_start_loc=query_start_loc.clone() if uniform else query_start_loc[:],
            num_accepted_tokens=accepted[:] if accepted is not None else None,
            num_requests=2,
        )

    assert len(calls["bind"]) == 2
    assert len(calls["run"]) == 2
    for binding, output, indices in zip(calls["bind"], outputs, state_indices):
        assert binding.mixed_qkv is mixed_qkv
        assert binding.raw_g is raw_g
        assert binding.raw_beta is raw_beta
        assert binding.z is z
        assert binding.output is output
        assert binding.state_indices.data_ptr() == indices.data_ptr()
    for (run_binding, _), bind_binding in zip(calls["run"], calls["bind"]):
        assert run_binding is bind_binding
    for name in (
        "query_start_loc",
        "num_accepted_tokens",
        "num_seqs",
        "num_tokens",
    ):
        assert getattr(calls["bind"][0], name) is getattr(calls["bind"][1], name)
    assert torch.equal(
        layers[0]._b12x_kda_num_accepted_tokens,
        torch.zeros(2, dtype=torch.int32)
        if speculative
        else torch.ones(2, dtype=torch.int32),
    )
    assert layers[0]._b12x_kda_num_seqs.item() == 2
    assert layers[0]._b12x_kda_num_tokens.item() == 2
    assert layers[1]._b12x_kda_num_seqs.item() == 0
    assert layers[1]._b12x_kda_num_tokens.item() == 0


@pytest.mark.parametrize("is_mtp_layer", [False, True])
def test_glm5next_sparse_mla_selects_b12x_backend(
    monkeypatch, is_mtp_layer: bool
) -> None:
    captured: dict[str, object] = {}

    class FakeLinear(torch.nn.Module):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            self.qrep_active = False

    indexer_kwargs: dict[str, object] = {}

    class FakeIndexer(torch.nn.Module):
        topk_tokens = 2048

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__()
            indexer_kwargs.update(kwargs)
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
        is_mtp_layer=is_mtp_layer,
    )

    assert captured["attn_backend"] is B12xGLM5NextMLASparseBackend
    assert captured["use_sparse"] is True
    assert indexer_kwargs["emit_physical_selection"] == (not is_mtp_layer)


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
