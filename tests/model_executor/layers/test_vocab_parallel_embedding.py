# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.model_executor.layers.vocab_parallel_embedding as vocab_module


@pytest.mark.parametrize(
    "layer_cls",
    [vocab_module.VocabParallelEmbedding, vocab_module.ParallelLMHead],
)
def test_tp3_pads_global_vocab_to_partition_alignment(
    monkeypatch, default_vllm_config, layer_cls
):
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 2)

    layer = layer_cls(129280, 16)

    assert layer.padding_size == 192
    assert layer.org_vocab_size_padded == 129408
    assert layer.num_embeddings_padded == 129408
    assert layer.num_embeddings_per_partition == 43136
    assert layer.shard_indices.org_vocab_start_index == 86272
    assert layer.shard_indices.org_vocab_end_index == 129280
    assert layer.shard_indices.num_org_vocab_padding == 128
    mapping = layer.get_sharded_to_full_mapping()
    assert mapping == list(range(129408))


def test_power_of_two_tp_preserves_default_padding(monkeypatch):
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 3)

    layer = vocab_module.VocabParallelEmbedding(129280, 16)

    assert layer.padding_size == vocab_module.DEFAULT_VOCAB_PADDING_SIZE
    assert layer.num_embeddings_padded == 129280
    assert layer.num_embeddings_per_partition == 32320


def test_disable_tp_preserves_requested_padding(monkeypatch):
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(vocab_module, "get_tensor_model_parallel_rank", lambda: 2)

    layer = vocab_module.VocabParallelEmbedding(129280, 16, disable_tp=True)

    assert layer.tp_size == 1
    assert layer.padding_size == vocab_module.DEFAULT_VOCAB_PADDING_SIZE
    assert layer.num_embeddings_padded == 129280
