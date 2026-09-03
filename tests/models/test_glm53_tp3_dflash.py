# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import qwen3_dflash
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Model,
    _get_dflash_draft_vocab_size,
    _get_glm53_tp3_head_geometry,
    _get_glm53_tp3_vocab_kwargs,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def _tp3_dflash_config(**overrides):
    values = {
        "glm53_tp3_padding": True,
        "glm53_tp3_vocab_padding_size": 192,
        "glm53_tp3_vocab_storage_size": 154944,
        "num_attention_heads": 36,
        "num_key_value_heads": 9,
        "original_num_attention_heads": 32,
        "original_num_key_value_heads": 8,
        "original_vocab_size": 154880,
        "draft_vocab_size": 154880,
        "vocab_size": 154880,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dflash_tp3_geometry_and_vocab_storage(monkeypatch) -> None:
    config = _tp3_dflash_config()

    assert _get_glm53_tp3_head_geometry(config) == (32, 8)
    assert _get_dflash_draft_vocab_size(config) == 154880
    vocab_kwargs = _get_glm53_tp3_vocab_kwargs(config)
    assert vocab_kwargs == {"padding_size": 192}

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 3,
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_rank",
        lambda: 2,
    )
    embedding = VocabParallelEmbedding(
        _get_dflash_draft_vocab_size(config),
        8,
        **vocab_kwargs,
    )
    assert embedding.num_embeddings == 154880
    assert embedding.num_embeddings_padded == 154944
    assert embedding.num_embeddings_per_partition == 51648
    assert embedding.weight.shape == (51648, 8)


def test_dflash_tp3_sink_bias_pads_only_rank_local_tail(monkeypatch) -> None:
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 2)
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model.config = _tp3_dflash_config()
    loaded = torch.arange(32, dtype=torch.float32)

    [(name, local)] = model._preprocess([("attention_sink_bias", loaded)])

    assert name == "attention_sink_bias"
    assert local.shape == (12,)
    torch.testing.assert_close(local[:8], loaded[24:32])
    torch.testing.assert_close(local[8:], torch.zeros(4))
    assert local.untyped_storage().data_ptr() != loaded.untyped_storage().data_ptr()


def test_dflash_tp4_paths_are_exact_noops(monkeypatch) -> None:
    config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=154880,
    )
    original_fields = vars(config).copy()
    assert _get_glm53_tp3_head_geometry(config) is None
    assert _get_glm53_tp3_vocab_kwargs(config) == {}
    assert _get_dflash_draft_vocab_size(config) == 154880
    assert vars(config) == original_fields

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_rank",
        lambda: 2,
    )
    embedding = VocabParallelEmbedding(config.vocab_size, 8)
    assert embedding.num_embeddings == 154880
    assert embedding.num_embeddings_padded == 154880
    assert embedding.num_embeddings_per_partition == 38720

    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 2)
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model.config = config
    loaded = torch.arange(32, dtype=torch.float32)
    [(_, local)] = model._preprocess([("attention_sink_bias", loaded)])
    torch.testing.assert_close(local, loaded[16:24])
    assert local.untyped_storage().data_ptr() == loaded.untyped_storage().data_ptr()


def test_dense_dflash_tp3_drops_target_expert_parallel(monkeypatch) -> None:
    from vllm.transformers_utils.configs import glm53_tp3

    monkeypatch.setattr(glm53_tp3, "is_glm53_config", lambda _: True)
    applied = []
    monkeypatch.setattr(
        glm53_tp3,
        "apply_glm53_tp3_draft_geometry",
        lambda *args: applied.append(args),
    )
    placement = {
        "prefill_context_parallel_size": 1,
        "data_parallel_size": 2,
        "data_parallel_size_local": 2,
        "data_parallel_rank": 1,
        "data_parallel_rank_local": 1,
        "data_parallel_master_ip": "127.0.0.1",
        "data_parallel_rpc_port": 1234,
        "data_parallel_master_port": 4321,
        "data_parallel_backend": "mp",
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
    }
    target_parallel = SimpleNamespace(
        tensor_parallel_size=3,
        enable_expert_parallel=True,
        **placement,
    )
    draft_parallel = SimpleNamespace(
        tensor_parallel_size=3,
        enable_expert_parallel=True,
    )
    spec = object.__new__(SpeculativeConfig)
    object.__setattr__(spec, "method", "dflash")
    object.__setattr__(spec, "target_model_config", object())
    object.__setattr__(spec, "target_parallel_config", target_parallel)
    object.__setattr__(spec, "draft_model_config", object())
    object.__setattr__(spec, "draft_parallel_config", draft_parallel)

    spec._apply_glm53_tp3_draft_geometry()

    assert draft_parallel.enable_expert_parallel is False
    for name, value in placement.items():
        assert getattr(draft_parallel, name) == value
    assert len(applied) == 1


def test_dflash7_uses_eight_target_kda_state_columns(monkeypatch) -> None:
    def fake_base_init(self, vllm_config, device) -> None:
        self.max_num_tokens = 16
        self.hidden_size = 4
        self.dtype = torch.float32
        self.num_speculative_steps = 7
        self.max_num_reqs = 2
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=0,
                dflash_config={"mask_token_id": 0},
            )
        )

    monkeypatch.setattr(DraftModelSpeculator, "__init__", fake_base_init)
    speculator = DFlashSpeculator(SimpleNamespace(), torch.device("cpu"))

    # The target KDA keeps one initial-state column plus one per draft token.
    assert speculator.num_speculative_steps == 7
    assert speculator.num_query_per_req == 8
    assert speculator.sample_col.shape == (14,)
