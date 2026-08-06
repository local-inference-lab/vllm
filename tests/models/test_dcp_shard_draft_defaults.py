# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache


def _vllm_config(num_hidden_layers: int = 78):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=num_hidden_layers)
        ),
    )


def _indexer_cache(layer_id: int = 78):
    cache = object.__new__(DeepseekV32IndexerCache)
    cache.head_dim = 512
    cache.dtype = torch.bfloat16
    cache.prefix = f"model.layers.{layer_id}.self_attn.indexer"
    cache.cache_config = SimpleNamespace(block_size=256)
    return cache


def test_dcp_shard_draft_defaults_to_sharded(monkeypatch):
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)

    spec = _indexer_cache().get_kv_cache_spec(_vllm_config())

    assert spec.dcp_replicated is False


def test_dcp_shard_draft_can_restore_replicated_legacy_mode(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_SHARD_DRAFT", "0")

    spec = _indexer_cache().get_kv_cache_spec(_vllm_config())

    assert spec.dcp_replicated is True
