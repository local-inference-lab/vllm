# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""max_token_id is the highest valid id, never the vocabulary size."""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm.tokenizers.hf import get_cached_tokenizer


def _tokenizer(cls=PreTrainedTokenizerFast) -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"a": 0, "b": 1, "c": 2}, unk_token="a"))
    tokenizer = cls(tokenizer_object=backend)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<x>"]})  # id 3
    return tokenizer


class _CountsAddedTokens(PreTrainedTokenizerFast):
    """A backend whose vocab_size includes added tokens, like fastokens."""

    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())


class _HidesSpecialTokens(PreTrainedTokenizerFast):
    """Special ids beyond get_vocab() that vocab_size counts (QwenTokenizer)."""

    @property
    def vocab_size(self) -> int:
        return 6

    def get_vocab(self) -> dict[str, int]:
        return {"a": 0, "b": 1, "c": 2}


def test_max_token_id_is_the_highest_id():
    assert get_cached_tokenizer(_tokenizer()).max_token_id == 3


def test_vocab_size_counting_added_tokens_admits_no_extra_id():
    tokenizer = _tokenizer(_CountsAddedTokens)
    assert tokenizer.vocab_size == 4
    assert get_cached_tokenizer(tokenizer).max_token_id == 3


def test_ids_only_vocab_size_knows_about_are_still_covered():
    tokenizer = _tokenizer(_HidesSpecialTokens)
    assert get_cached_tokenizer(tokenizer).max_token_id == 5
