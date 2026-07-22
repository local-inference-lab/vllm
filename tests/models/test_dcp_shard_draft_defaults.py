# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.models.deepseek_v2 as deepseek_v2
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache,
    _replicate_indexer_cache_under_dcp,
)


def _vllm_config(
    num_hidden_layers: int = 78,
    dcp_size: int = 4,
    pcp_size: int = 1,
):
    return SimpleNamespace(
        use_v2_model_runner=True,
        cache_config=SimpleNamespace(block_size=256),
        compilation_config=SimpleNamespace(static_forward_context={}),
        attention_config=SimpleNamespace(backend="B12X_MLA_SPARSE"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(num_hidden_layers=num_hidden_layers),
            max_model_len=4096,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            prefill_context_parallel_size=pcp_size,
        ),
    )


def _indexer_cache(layer_id: int = 78):
    cache = object.__new__(DeepseekV32IndexerCache)
    cache.head_dim = 512
    cache.dtype = torch.bfloat16
    cache.prefix = f"model.layers.{layer_id}.self_attn.indexer"
    cache.cache_config = SimpleNamespace(block_size=256)
    return cache


def _get_indexer_spec(layer_id: int, config):
    cache = _indexer_cache(layer_id)
    cache.dcp_replicated = _replicate_indexer_cache_under_dcp(cache.prefix, config)
    return cache, cache.get_kv_cache_spec(config)


def test_target_indexer_replication_defaults_off(monkeypatch):
    monkeypatch.delenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", raising=False)

    cache, spec = _get_indexer_spec(12, _vllm_config())

    assert spec.dcp_replicated is False
    assert spec.block_size == cache.cache_config.block_size


def test_dcp_shard_draft_defaults_to_sharded(monkeypatch):
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")

    cache, spec = _get_indexer_spec(78, _vllm_config())

    assert spec.dcp_replicated is False
    assert spec.block_size == cache.cache_config.block_size


def test_dcp_shard_draft_can_restore_replicated_legacy_mode(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_SHARD_DRAFT", "0")
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")

    cache, spec = _get_indexer_spec(78, _vllm_config())

    assert spec.dcp_replicated is True
    assert spec.block_size == cache.cache_config.block_size


@pytest.mark.parametrize("dcp_size", [2, 4, 6, 8])
def test_target_indexer_replication_equalizes_global_block_coverage(
    monkeypatch, dcp_size: int
):
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")
    monkeypatch.setattr(deepseek_v2, "use_b12x_sparse_indexer", lambda: True)

    config = _vllm_config(dcp_size=dcp_size)
    cache, spec = _get_indexer_spec(12, config)

    assert spec.dcp_replicated is True
    assert spec.block_size == dcp_size * cache.cache_config.block_size


def test_target_indexer_replication_rejects_pcp(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")
    monkeypatch.setattr(deepseek_v2, "use_b12x_sparse_indexer", lambda: True)

    config = _vllm_config(dcp_size=4, pcp_size=2)
    with pytest.raises(NotImplementedError, match="DCP2 through DCP8 with PCP1"):
        _replicate_indexer_cache_under_dcp("model.layers.12.indexer.k_cache", config)


def test_target_indexer_replication_requires_v2_model_runner(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")

    config = _vllm_config()
    config.use_v2_model_runner = False
    with pytest.raises(NotImplementedError, match="V2 model runner"):
        _replicate_indexer_cache_under_dcp("model.layers.12.indexer.k_cache", config)


def test_real_cache_construction_replicates_target_but_not_draft(monkeypatch):
    monkeypatch.setenv("VLLM_DCP_REPLICATE_INDEXER_CACHE", "1")
    monkeypatch.delenv("VLLM_DCP_SHARD_DRAFT", raising=False)
    monkeypatch.setattr(deepseek_v2, "use_b12x_sparse_indexer", lambda: True)

    config = _vllm_config()
    monkeypatch.setattr(deepseek_v2, "get_current_vllm_config", lambda: config)

    target = DeepseekV32IndexerCache(
        head_dim=132,
        dtype=torch.uint8,
        prefix="model.layers.12.self_attn.indexer.k_cache",
        cache_config=config.cache_config,
    )
    draft = DeepseekV32IndexerCache(
        head_dim=132,
        dtype=torch.uint8,
        prefix="model.layers.78.self_attn.indexer.k_cache",
        cache_config=config.cache_config,
    )

    assert target.dcp_replicated is True
    assert target.get_kv_cache_spec(config).dcp_replicated is True
    assert draft.dcp_replicated is False
    assert draft.get_kv_cache_spec(config).dcp_replicated is False
    assert config.compilation_config.static_forward_context == {
        target.prefix: target,
        draft.prefix: draft,
    }


def test_indexer_constructor_forwards_cache_replication(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCache(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dcp_replicated = True

    def fake_sparse_attn_indexer(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return torch.nn.Identity()

    monkeypatch.setattr(deepseek_v2, "DeepseekV32IndexerCache", FakeCache)
    monkeypatch.setattr(deepseek_v2, "SparseAttnIndexer", fake_sparse_attn_indexer)
    monkeypatch.setattr(
        deepseek_v2,
        "ReplicatedLinear",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        deepseek_v2,
        "MergedColumnParallelLinear",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(
        deepseek_v2,
        "LayerNorm",
        lambda *args, **kwargs: torch.nn.Identity(),
    )
    monkeypatch.setattr(deepseek_v2.current_platform, "is_cuda", lambda: False)

    vllm_config = _vllm_config()
    model_config = SimpleNamespace(
        index_topk=4,
        index_n_heads=1,
        index_head_dim=128,
        qk_rope_head_dim=64,
    )
    indexer = deepseek_v2.Indexer(
        vllm_config=vllm_config,
        config=model_config,
        hidden_size=256,
        q_lora_rank=128,
        quant_config=None,
        cache_config=vllm_config.cache_config,
        topk_indices_buffer=torch.empty((2, 4), dtype=torch.int32),
        prefix="model.layers.12.self_attn.indexer",
    )

    sparse_args = captured["args"]
    sparse_kwargs = captured["kwargs"]
    assert isinstance(sparse_args, tuple)
    assert isinstance(sparse_kwargs, dict)
    assert sparse_args[0] is indexer.k_cache
    assert sparse_kwargs["dcp_replicated"] is True
