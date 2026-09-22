# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.config import ModelConfig, ParallelConfig
from vllm.transformers_utils.configs.speculators.base import SpeculatorsConfig

pytestmark = pytest.mark.skip_global_cleanup


def _make_dspark_config(architecture: str) -> dict:
    return {
        "speculators_model_type": "dspark",
        "architectures": [architecture],
        "sample_from_anchor": True,
        "use_aux_hidden_state": True,
        "draft_vocab_size": 32,
        "target_hidden_size": 16,
        "mask_token_id": 31,
        "markov_rank": 4,
        "markov_head_type": "vanilla",
        "block_size": 3,
        "aux_hidden_state_layer_ids": [2, 5],
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 16,
        },
    }


def test_dspark_updater_preserves_qwen3_omni_contract() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("Qwen3OmniDSparkModel")
    )

    assert config["architectures"] == ["Qwen3OmniDSparkModel"]
    assert config["sample_from_anchor"] is True
    assert config["dspark_bonus_anchor"] is False
    assert config["use_aux_hidden_state"] is True
    assert config["target_layer_ids"] == [1, 4]


def test_dspark_updater_keeps_legacy_qwen3_checkpoint_loadable() -> None:
    config = SpeculatorsConfig.extract_transformers_pre_trained_config(
        _make_dspark_config("DSparkSpeculator")
    )

    assert config["architectures"] == ["Qwen3DSparkModel"]


def test_dspark_updater_maps_bonus_anchor_semantics() -> None:
    outer_config = _make_dspark_config("Qwen3DSparkModel")
    outer_config["sample_from_anchor"] = False

    config = SpeculatorsConfig.extract_transformers_pre_trained_config(outer_config)

    assert config["sample_from_anchor"] is False
    assert config["dspark_bonus_anchor"] is True


@pytest.mark.parametrize(
    "tp,heads,kv_heads",
    [
        (1, 96, 16),
        (8, 96, 16),
        (10, 120, 20),
        (12, 144, 24),
        (16, 96, 16),
        (32, 96, 16),
    ],
)
def test_dspark_parallel_config_preserves_complete_gqa_groups(
    tmp_path, tp, heads, kv_heads
):
    """Padding must not change which learned KV head serves a query head."""
    config = {
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
        "hidden_size": 7168,
        "intermediate_size": 14336,
        "num_hidden_layers": 5,
        "num_attention_heads": 96,
        "num_key_value_heads": 16,
        "head_dim": 64,
        "vocab_size": 163840,
        "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    model = ModelConfig(model=str(tmp_path), tokenizer_mode="skip", runner="draft")
    parallel = ParallelConfig(
        tensor_parallel_size=tp, distributed_executor_backend="external_launcher"
    )
    for _ in range(2):
        model.verify_with_parallel_config(parallel)
        assert model.hf_text_config.num_attention_heads == heads
        assert model.hf_text_config.num_key_value_heads == kv_heads
        assert model.get_num_attention_heads(parallel) == heads // tp
        assert model.get_num_kv_heads(parallel) == max(1, kv_heads // tp)
        assert model.get_head_size() == 64
    model.verify_with_parallel_config(ParallelConfig())
    assert model.hf_text_config.num_attention_heads == 96
    assert model.hf_text_config.num_key_value_heads == 16
