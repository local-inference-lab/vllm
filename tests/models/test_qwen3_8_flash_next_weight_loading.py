# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only package and weight-routing tests for Qwen3.8-Flash-Next."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.qwen3_8_flash_next.model import (
    _remap_qsa_cache_scale_name,
)
from vllm.models.qwen3_8_flash_next.mtp import (
    Qwen3_8FlashNextMTP,
    Qwen3_8FlashNextMultiTokenPredictor,
    _remap_mtp_quantized_layers,
    _remap_mtp_weight_name,
)
from vllm.models.qwen3_8_flash_next.ple_layer import (
    Qwen3_8FlashNextNGramEmbedding,
    Qwen3_8FlashNextPLELayer,
)


@pytest.mark.parametrize(
    ("checkpoint_name", "model_name"),
    [
        ("mtp.fc_embedding.weight", "model.fc_embedding.weight"),
        (
            "mtp.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "model.language_model.mtp.pre_fc_norm_hidden.weight",
            "model.pre_fc_norm_hidden.weight",
        ),
        (
            "model.language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ),
        ("lm_head.weight", "lm_head.weight"),
        ("model.language_model.layers.0.mlp.down_proj.weight", None),
    ],
)
def test_mtp_checkpoint_prefix_mapping(
    checkpoint_name: str,
    model_name: str | None,
) -> None:
    assert _remap_mtp_weight_name(checkpoint_name) == model_name
    assert checkpoint_name.startswith(
        Qwen3_8FlashNextMTP.checkpoint_weight_name_prefixes
    ) == (model_name is not None)


@pytest.mark.parametrize(
    "checkpoint_prefix", ["", "model.language_model.", "language_model."]
)
def test_mtp_loading_skips_target_shards_and_weights(
    tmp_path, monkeypatch, checkpoint_prefix
):
    """MTP opens only draft/shared shards, retaining every supported weight alias."""
    target_path = tmp_path / "target.safetensors"
    draft_path = tmp_path / "draft.safetensors"
    target_name = checkpoint_prefix + "layers.0.mlp.down_proj.weight"
    weights = {
        checkpoint_prefix + name: torch.tensor([float(index)])
        for index, name in enumerate(
            (
                "mtp.fc_embedding.weight",
                "model.mtp.fc_hidden.weight",
                "embed_tokens.weight",
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.lm_head.weight",
                "shared_head.head.weight",
                "model.shared_head.head.weight",
                "mtp.shared_head.head.weight",
            )
        )
    }
    save_file({target_name: torch.zeros(1)}, target_path)
    save_file(
        {**weights, checkpoint_prefix + "layers.1.weight": torch.zeros(1)}, draft_path
    )
    weight_map = {name: draft_path.name for name in weights}
    weight_map[target_name] = target_path.name
    weight_map[checkpoint_prefix + "layers.1.weight"] = draft_path.name
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    opened = []
    safe_open = weight_utils.safe_open

    def record_open(filename, **kwargs):
        opened.append(filename)
        return safe_open(filename, **kwargs)

    monkeypatch.setattr(weight_utils, "safe_open", record_open)
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    prefixes = Qwen3_8FlashNextMTP.checkpoint_weight_name_prefixes
    loaded = dict(
        loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None),
            SimpleNamespace(checkpoint_weight_name_prefixes=prefixes),
        )
    )

    assert opened == [str(draft_path)]
    assert loaded.keys() == weights.keys()
    for name, weight in loaded.items():
        assert _remap_mtp_weight_name(name) is not None
        torch.testing.assert_close(weight, weights[name])


def test_mtp_qsa_scale_uses_runtime_module_index() -> None:
    remapped = _remap_mtp_weight_name("mtp.layers.0.self_attn.k_proj.k_scale")
    assert remapped is not None
    assert _remap_qsa_cache_scale_name(remapped, frozenset({0})) == (
        "model.layers.0.self_attn._k_scale"
    )
    assert _remap_qsa_cache_scale_name(remapped, frozenset({48})) == remapped


def test_mtp_quantized_layers_use_runtime_module_index() -> None:
    quantized_layers = {
        "mtp.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.visual.blocks.0.mlp.linear_fc2": {"quant_algo": "NVFP4"},
    }

    assert _remap_mtp_quantized_layers(quantized_layers, 48) == {
        "mtp.layers.48.mlp.experts": {"quant_algo": "W4A16_NVFP4"},
        "model.visual.blocks.0.mlp.linear_fc2": {"quant_algo": "NVFP4"},
    }


class _PLELoaderAudit(nn.Module):
    load_weights = Qwen3_8FlashNextNGramEmbedding.load_weights
    _validate_embedding_loaded = (
        Qwen3_8FlashNextNGramEmbedding._validate_embedding_loaded
    )

    def __init__(self, quant_mode: str) -> None:
        super().__init__()
        self._quant_mode = quant_mode
        self._embedding_load_ranges: set[tuple[int, int]] = set()
        self._scale_load_ranges: set[tuple[int, int]] = set()
        self._weight_scale_loaded = False
        self._weight_scale_2_loaded = False
        self._embedding_validated = False
        self.split_ngram_parts = 4
        scale_shape: tuple[int, ...] | None
        scale_dtype: torch.dtype | None
        scale_2_shape: tuple[int, ...] | None
        scale_2_dtype: torch.dtype | None
        if quant_mode == "bf16":
            weight_dtype = torch.bfloat16
            weight_shape = (4, 2)
            scale_shape = scale_dtype = scale_2_shape = scale_2_dtype = None
        elif quant_mode == "fp8_e4m3_per_tensor":
            weight_dtype = torch.float8_e4m3fn
            weight_shape = (4, 2)
            scale_shape = (1,)
            scale_dtype = torch.bfloat16
            scale_2_shape = scale_2_dtype = None
        else:
            assert quant_mode == "nvfp4_group16"
            weight_dtype = torch.uint8
            weight_shape = (4, 8)
            scale_shape = (4, 1)
            scale_dtype = torch.float8_e4m3fn
            scale_2_shape = (1,)
            scale_2_dtype = torch.float32
        self._plan = SimpleNamespace(
            padded_vocab_size=8,
            shard_start=2,
            shard_end=6,
            weight_shape=weight_shape,
            weight_dtype=weight_dtype,
            weight_scale_shape=scale_shape,
            weight_scale_dtype=scale_dtype,
            weight_scale_2_shape=scale_2_shape,
            weight_scale_2_dtype=scale_2_dtype,
        )
        self.register_buffer("layer_multipliers", torch.tensor([11, 13]))
        self.register_buffer("ngram_heads_offsets", torch.tensor([0, 4]))
        self.register_buffer("ngram_heads_vocab_sizes", torch.tensor([4, 4]))
        embedding = nn.Module()
        embedding.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(self._plan.weight_shape, dtype=self._plan.weight_dtype),
                requires_grad=False,
            ),
        )
        if scale_shape is None:
            embedding.register_parameter("weight_scale", None)
        else:
            embedding.register_parameter(
                "weight_scale",
                nn.Parameter(
                    torch.empty(scale_shape, dtype=scale_dtype),
                    requires_grad=False,
                ),
            )
        if scale_2_shape is None:
            embedding.register_parameter("weight_scale_2", None)
        else:
            embedding.register_parameter(
                "weight_scale_2",
                nn.Parameter(
                    torch.empty(scale_2_shape, dtype=scale_2_dtype),
                    requires_grad=False,
                ),
            )
        self.add_module("ngram_embedding", embedding)


def _fp8_shard(start: int) -> torch.Tensor:
    return torch.tensor(
        [[start, start + 1], [start + 2, start + 3]],
        dtype=torch.float8_e4m3fn,
    )


def test_fp8_ple_loader_keeps_local_table_quantized() -> None:
    model = _PLELoaderAudit("fp8_e4m3_per_tensor")
    shard_1 = _fp8_shard(4)
    shard_2 = _fp8_shard(8)

    loaded = model.load_weights(
        [
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.shard_2.weight", shard_2),
            (
                "ngram_embedding.weight_scale",
                torch.tensor([0.25], dtype=torch.bfloat16),
            ),
        ]
    )
    model._validate_embedding_loaded()

    assert loaded == {"ngram_embedding.weight", "ngram_embedding.weight_scale"}
    assert model.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        model.ngram_embedding.weight.float(),
        torch.cat((shard_1, shard_2)).float(),
    )
    torch.testing.assert_close(
        model.ngram_embedding.weight_scale,
        torch.tensor([0.25], dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    ("weights", "error"),
    [
        (
            [
                ("ngram_embedding.shard_1.weight", _fp8_shard(4)),
                ("ngram_embedding.shard_2.weight", _fp8_shard(8)),
            ],
            "missing weight_scale",
        ),
        (
            [
                ("ngram_embedding.shard_1.weight", _fp8_shard(4)),
                (
                    "ngram_embedding.weight_scale",
                    torch.tensor([0.25], dtype=torch.bfloat16),
                ),
            ],
            "do not cover the local table",
        ),
    ],
)
def test_fp8_ple_loader_rejects_incomplete_checkpoint(weights, error: str) -> None:
    model = _PLELoaderAudit("fp8_e4m3_per_tensor")
    model.load_weights(weights)
    with pytest.raises(ValueError, match=error):
        model._validate_embedding_loaded()


def test_fp8_ple_loader_accepts_scale_in_later_streaming_callback() -> None:
    model = _PLELoaderAudit("fp8_e4m3_per_tensor")
    assert model.load_weights(
        [
            ("ngram_embedding.shard_1.weight", _fp8_shard(4)),
            ("ngram_embedding.shard_2.weight", _fp8_shard(8)),
        ]
    ) == {"ngram_embedding.weight"}

    assert model.load_weights(
        [
            (
                "ngram_embedding.weight_scale",
                torch.tensor([0.25], dtype=torch.bfloat16),
            )
        ]
    ) == {"ngram_embedding.weight_scale"}
    model._validate_embedding_loaded()


def test_fp8_ple_loader_rejects_unquantized_shard() -> None:
    with pytest.raises(TypeError, match="must have dtype"):
        _PLELoaderAudit("fp8_e4m3_per_tensor").load_weights(
            [
                ("ngram_embedding.shard_1.weight", _fp8_shard(4).bfloat16()),
                ("ngram_embedding.shard_2.weight", _fp8_shard(8)),
                (
                    "ngram_embedding.weight_scale",
                    torch.tensor([0.25], dtype=torch.bfloat16),
                ),
            ]
        )


def test_bf16_ple_loader_keeps_only_local_overlap() -> None:
    model = _PLELoaderAudit("bf16")
    model.ngram_embedding.weight.data.fill_(-1)
    shard_0 = torch.tensor([[0, 1], [2, 3]], dtype=torch.bfloat16)
    shard_1 = torch.tensor([[4, 5], [6, 7]], dtype=torch.bfloat16)
    shard_2 = torch.tensor([[8, 9], [10, 11]], dtype=torch.bfloat16)
    shard_3 = torch.tensor([[12, 13], [14, 15]], dtype=torch.bfloat16)

    loaded = model.load_weights(
        [
            ("ngram_embedding.shard_0.weight", shard_0),
            ("ngram_embedding.shard_1.weight", shard_1),
            ("ngram_embedding.shard_2.weight", shard_2),
            ("ngram_embedding.shard_3.weight", shard_3),
        ]
    )
    model._validate_embedding_loaded()

    assert loaded == {"ngram_embedding.weight"}
    assert model.ngram_embedding.weight.dtype == torch.bfloat16
    torch.testing.assert_close(
        model.ngram_embedding.weight,
        torch.cat((shard_1, shard_2)),
    )
    assert model.ngram_embedding.weight_scale is None
    assert model.ngram_embedding.weight_scale_2 is None


def test_nvfp4_ple_loader_keeps_local_table_packed() -> None:
    model = _PLELoaderAudit("nvfp4_group16")
    weight_1 = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    weight_2 = torch.arange(16, 32, dtype=torch.uint8).reshape(2, 8)
    scale_1 = torch.tensor([[1], [2]], dtype=torch.float8_e4m3fn)
    scale_2 = torch.tensor([[3], [4]], dtype=torch.float8_e4m3fn)

    loaded = model.load_weights(
        [
            ("ngram_embedding.shard_1.weight", weight_1),
            ("ngram_embedding.shard_1.weight_scale", scale_1),
            ("ngram_embedding.shard_2.weight", weight_2),
            ("ngram_embedding.shard_2.weight_scale", scale_2),
            ("ngram_embedding.weight_scale_2", torch.tensor(0.25)),
        ]
    )
    model._validate_embedding_loaded()

    assert loaded == {
        "ngram_embedding.weight",
        "ngram_embedding.weight_scale",
        "ngram_embedding.weight_scale_2",
    }
    assert model.ngram_embedding.weight.dtype == torch.uint8
    torch.testing.assert_close(
        model.ngram_embedding.weight, torch.cat((weight_1, weight_2))
    )
    torch.testing.assert_close(
        model.ngram_embedding.weight_scale.float(),
        torch.cat((scale_1, scale_2)).float(),
    )
    torch.testing.assert_close(
        model.ngram_embedding.weight_scale_2, torch.tensor([0.25])
    )


def test_ple_embedding_uses_query_offsets_for_live_token_count() -> None:
    embedding = Qwen3_8FlashNextNGramEmbedding.__new__(Qwen3_8FlashNextNGramEmbedding)
    nn.Module.__init__(embedding)
    embedding.max_total_tokens = 8
    embedding.max_num_reqs = 4
    embedding.ngram_size = 3
    embedding.eos_token_id = 0
    embedding._token_ids = torch.empty(8, dtype=torch.int64)
    embedding._query_start_loc = torch.empty(5, dtype=torch.int32)
    embedding._committed_history = torch.empty(4, 2, dtype=torch.int64)
    embedding._num_seqs = torch.zeros(1, dtype=torch.int32)
    embedding._num_tokens = torch.zeros(1, dtype=torch.int32)

    token_capacity = embedding._prepare_inputs(
        torch.tensor([41, 42, 0, 0], dtype=torch.int32),
        torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32),
        torch.zeros((4, 2), dtype=torch.int64),
    )

    assert token_capacity == 4
    assert embedding._num_seqs.item() == 4
    assert embedding._num_tokens.item() == 2


def test_ple_mixed_op_uses_query_offsets_for_live_token_count() -> None:
    layer = Qwen3_8FlashNextPLELayer.__new__(Qwen3_8FlashNextPLELayer)
    nn.Module.__init__(layer)
    layer.max_tokens = 8
    layer.max_seqs = 4
    layer._query_start_loc = torch.empty(5, dtype=torch.int32)
    layer._state_slot_ids = torch.empty(4, dtype=torch.int64)
    layer._state_is_fresh = torch.empty(4, dtype=torch.bool)
    layer._num_accepted_tokens = torch.empty(4, dtype=torch.int32)
    layer._request_is_prefill = torch.empty(4, dtype=torch.bool)
    layer._num_seqs = torch.zeros(1, dtype=torch.int32)
    layer._num_tokens = torch.zeros(1, dtype=torch.int32)
    metadata = SimpleNamespace(num_reqs=4, num_decodes=0, num_prefills=0)

    layer._prepare_metadata(
        metadata,
        torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32),
        token_count=4,
    )

    assert layer._num_seqs.item() == 4
    assert layer._num_tokens.item() == 2


_NATIVE_MTP_FIXED_KEYS = [
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.hyper_connection_mixer.hc_norm.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.mlp.gate.weight",
    "mtp.layers.0.mlp.shared_expert.down_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.mlp.shared_expert_gate.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]
_EXPERT_WEIGHT_SUFFIXES = [
    "down_proj.weight",
    "down_proj.weight_scale_inv",
    "gate_proj.weight",
    "gate_proj.weight_scale_inv",
    "up_proj.weight",
    "up_proj.weight_scale_inv",
]


def _native_mtp_checkpoint_keys() -> list[str]:
    expert_keys = [
        f"mtp.layers.0.mlp.experts.{expert_id}.{suffix}"
        for expert_id in range(512)
        for suffix in _EXPERT_WEIGHT_SUFFIXES
    ]
    keys = sorted([*_NATIVE_MTP_FIXED_KEYS, *expert_keys])
    assert len(keys) == 3101
    return keys


def _final_parameter_name(checkpoint_name: str) -> str:
    outer_name = _remap_mtp_weight_name(checkpoint_name)
    assert outer_name is not None
    outer_name = _remap_qsa_cache_scale_name(outer_name, frozenset({0}))
    if not outer_name.startswith("model."):
        return outer_name
    inner_name = outer_name.removeprefix("model.")
    mapped = Qwen3_8FlashNextMultiTokenPredictor.hf_to_vllm_mapper._map_name_with_shard(
        inner_name
    )
    assert mapped is not None
    return f"model.{mapped[0]}"


def _register_parameter(
    root: nn.Module,
    name: str,
    loaded_ids: list[int],
) -> None:
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        child = module._modules.get(part)
        if child is None:
            child = nn.Module()
            module.add_module(part, child)
        module = child
    parameter = nn.Parameter(torch.zeros(()), requires_grad=False)

    def record_load(param: nn.Parameter, loaded: torch.Tensor) -> None:
        del param
        loaded_ids.append(int(loaded.item()))

    parameter.weight_loader = record_load
    module.register_parameter(parts[-1], parameter)


class _LoaderAuditPredictor(nn.Module):
    hf_to_vllm_mapper = Qwen3_8FlashNextMultiTokenPredictor.hf_to_vllm_mapper
    load_weights = Qwen3_8FlashNextMultiTokenPredictor.load_weights

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_experts=512)
        self.num_mtp_layers = 1
        self.mtp_start_layer_idx = 48


class _LoaderAuditMTP(nn.Module):
    load_weights = Qwen3_8FlashNextMTP.load_weights

    def __init__(self, parameter_names: set[str], loaded_ids: list[int]) -> None:
        super().__init__()
        self.model = _LoaderAuditPredictor()
        for name in sorted(parameter_names):
            _register_parameter(self, name, loaded_ids)


def test_native_mtp_tensor_corpus_routes_through_auto_weights_loader() -> None:
    checkpoint_names = [
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        *_native_mtp_checkpoint_keys(),
    ]
    parameter_names = {_final_parameter_name(name) for name in checkpoint_names}
    assert len(checkpoint_names) == 3103
    # Four packed destinations consume nine constituent tensors: Q/K/V,
    # shared gate/up, and two HC down/injection pairs.
    assert len(parameter_names) == 3098

    loaded_ids: list[int] = []
    model = _LoaderAuditMTP(parameter_names, loaded_ids)
    loaded = model.load_weights(
        (name, torch.tensor(index)) for index, name in enumerate(checkpoint_names)
    )

    assert loaded == parameter_names
    assert sorted(loaded_ids) == list(range(3103))


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextForConditionalGeneration",
        "Qwen3_8FlashNextMTP",
    ],
)
def test_lazy_package_exports_match_registry(architecture: str) -> None:
    registered_cls = ModelRegistry.models[architecture].load_model_cls()

    assert registered_cls.__name__ == architecture
    assert registered_cls.__module__.startswith("vllm.models.qwen3_8_flash_next.")
    assert ModelRegistry.models[architecture].inspect_model_cls() is not None
