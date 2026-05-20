# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.layers.logits_processor as logits_processor_module
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)


class _StaticLogitsMethod:

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def apply(self, layer, hidden_states, bias=None):
        return self.logits.clone()


def test_get_top_tokens_masks_padding_and_maps_added_vocab(monkeypatch):
    monkeypatch.setattr(
        logits_processor_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    # Local layout:
    #   [0:10]  original vocab tokens 0..9
    #   [10:12] original vocab padding
    #   [12:14] added vocab tokens 10..11
    #   [14:20] added vocab padding
    shard_indices = VocabParallelEmbeddingShardIndices(
        padded_org_vocab_start_index=0,
        padded_org_vocab_end_index=12,
        padded_added_vocab_start_index=10,
        padded_added_vocab_end_index=18,
        org_vocab_start_index=0,
        org_vocab_end_index=10,
        added_vocab_start_index=10,
        added_vocab_end_index=12,
    )
    logits = torch.full((3, 20), -10.0)
    logits[0, 11] = 100.0  # original-vocab padding, must be ignored.
    logits[0, 12] = 1.0  # added token id 10.
    logits[1, 15] = 100.0  # added-vocab padding, must be ignored.
    logits[1, 3] = 1.0  # original token id 3.
    logits[2, 13] = 100.0  # added token id 11.

    lm_head = SimpleNamespace(
        quant_method=_StaticLogitsMethod(logits),
        shard_indices=shard_indices,
    )
    hidden_states = torch.empty((3, 1))

    top_tokens = LogitsProcessor(vocab_size=12, org_vocab_size=10).get_top_tokens(
        lm_head, hidden_states
    )

    assert top_tokens.tolist() == [10, 3, 11]
