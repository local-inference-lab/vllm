# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    Qwen3ModelArchConfigConvertor,
)


def _write_draft(path, *, mla=True):
    path.mkdir()
    config = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "num_hidden_layers": 5,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 163840,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
        "is_causal": False,
        "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
        "sliding_window": 4096,
        "dflash_config": {
            "attention_mode": "mla" if mla else "gqa",
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [19, 37, 66, 78, 90],
            "mask_token_id": 163592,
        },
    }
    (path / "config.json").write_text(json.dumps(config))


@pytest.mark.parametrize("tp", [8, 9, 10, 12, 16])
def test_mla_dflash2_uses_padded_heads_and_latent_kv(tmp_path, tp):
    path = tmp_path / "draft"
    _write_draft(path)
    model = ModelConfig(model=str(path), tokenizer_mode="skip", runner="draft")
    assert model.architectures == ["DFlash2KimiK3Model"]
    assert model.is_deepseek_mla and model.use_mla
    parallel = ParallelConfig(
        tensor_parallel_size=tp, distributed_executor_backend="external_launcher"
    )
    model.verify_with_parallel_config(parallel)
    model.verify_with_parallel_config(parallel)
    assert model.get_head_size() == 576
    assert model.get_num_kv_heads(parallel) == 1
    assert model.get_num_attention_heads(parallel) == (64 + tp - 1) // tp
    assert model.hf_text_config.original_num_attention_heads == 64
    assert model.hf_text_config.intermediate_size == 14336
    model.verify_with_parallel_config(ParallelConfig())
    assert model.hf_text_config.num_attention_heads == 64


def test_mla_dflash2_eagle_wrapper_preserves_architecture(tmp_path):
    path = tmp_path / "draft"
    _write_draft(path)
    target = ModelConfig(
        model=str(path), tokenizer_mode="skip", runner="draft", max_model_len=32768
    )
    spec = SpeculativeConfig(
        model=str(path),
        method="dflash",
        num_speculative_tokens=7,
        target_model_config=target,
        target_parallel_config=ParallelConfig(),
    )
    draft = spec.draft_model_config
    assert draft.architectures == ["DFlash2KimiK3Model"]
    assert draft.hf_config.model_type == "eagle"
    assert draft.use_mla and draft.get_head_size() == 576


def test_gqa_dflash2_architecture_is_unchanged(tmp_path):
    path = tmp_path / "draft"
    _write_draft(path, mla=False)
    model = ModelConfig(model=str(path), tokenizer_mode="skip", runner="draft")
    assert model.architectures == ["DFlash2DraftModel"]
    assert not model.use_mla


def test_non_dflash_qwen_architecture_is_unchanged():
    from transformers import Qwen3Config

    config = Qwen3Config(architectures=["Qwen3ForCausalLM"])
    convertor = Qwen3ModelArchConfigConvertor(config, config)
    assert convertor.get_architectures() == ["Qwen3ForCausalLM"]
    assert not convertor.is_deepseek_mla()
