# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the vocab-parallel top-k reduction of ``LogitsProcessor``."""

import types
from unittest import mock

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor

_ALL_GATHER = (
    "vllm.model_executor.layers.logits_processor.tensor_model_parallel_all_gather"
)


class _PlainLinear:
    """A platform-independent stand-in for the lm_head's quant method."""

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class _FakeShardHead:
    """One vocab shard of an lm_head, as ``get_top_k_tokens`` sees it."""

    def __init__(self, weight: torch.Tensor, vocab_start: int, tp_size: int):
        self.weight = weight
        self.quant_method = _PlainLinear()
        self.shard_indices = types.SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=vocab_start
        )
        self.tp_size = tp_size


def _shard_heads(weight: torch.Tensor, tp_size: int) -> list[_FakeShardHead]:
    shard = weight.shape[0] // tp_size
    return [
        _FakeShardHead(weight[r * shard : (r + 1) * shard], r * shard, tp_size)
        for r in range(tp_size)
    ]


def _assert_top_k(
    ids: torch.Tensor,
    values: torch.Tensor,
    full: torch.Tensor,
    k: int,
    transform=lambda v: v.float(),
) -> None:
    """``(ids, values)`` is a top-k of ``full``; ties may order differently."""
    expected_values, _ = torch.topk(full, k, dim=-1)
    assert ids.dtype == torch.int64
    assert ids.shape == values.shape == (full.shape[0], k)
    for row in ids:
        assert len(set(row.tolist())) == k, "duplicate ids in one row"
    torch.testing.assert_close(values, transform(expected_values))
    torch.testing.assert_close(values, transform(full.gather(-1, ids)))


def _simulate_tp(
    lp: LogitsProcessor,
    heads: list[_FakeShardHead],
    hidden_states: torch.Tensor,
    k: int,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
    """Run ``get_top_k_tokens`` on every rank with a simulated all-gather.

    The first pass records each rank's gather input; the second pass answers
    every rank with the concatenation of all recorded inputs, exactly as the
    collective would.  Returns the per-rank results and the recorded inputs.
    """

    inputs: list[torch.Tensor] = []

    def record(inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1
        inputs.append(inp.clone())
        return torch.cat([inp] * len(heads), dim=-1)

    with mock.patch(_ALL_GATHER, side_effect=record):
        for head in heads:
            lp.get_top_k_tokens(head, hidden_states, k)
    assert len(inputs) == len(heads), "one all-gather per rank"

    def answer(inp: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert dim == -1
        assert any(torch.equal(inp, seen) for seen in inputs)
        return torch.cat(inputs, dim=-1)

    results = []
    with mock.patch(_ALL_GATHER, side_effect=answer):
        for head in heads:
            results.append(lp.get_top_k_tokens(head, hidden_states, k))
    return results, inputs


@pytest.mark.parametrize("tp_size", [2, 4])
@pytest.mark.parametrize("k", [16, 7])
def test_top_k_tokens_single_packed_gather(default_vllm_config, tp_size, k):
    """Values and ids travel in one fp32 exchange and reduce to the global top-k."""
    torch.manual_seed(0)
    vocab_size, hidden_size, batch = 256, 32, 5
    hidden_states = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lp = LogitsProcessor(vocab_size)

    results, inputs = _simulate_tp(lp, _shard_heads(weight, tp_size), hidden_states, k)

    for inp in inputs:
        assert inp.dtype == torch.float32
        assert inp.shape == (batch, 2 * k)
        # ids ride along bit-exact as int32 in the second half of the row
        ids = inp[:, k:].contiguous().view(torch.int32)
        assert ids.min() >= 0 and ids.max() < vocab_size

    full = torch.nn.functional.linear(hidden_states, weight)
    for ids, values in results:
        _assert_top_k(ids, values, full, k)


def test_top_k_tokens_applies_scale_and_soft_cap_after_reduction(default_vllm_config):
    torch.manual_seed(1)
    vocab_size, hidden_size, batch, k, tp_size = 128, 16, 3, 4, 2
    hidden_states = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lp = LogitsProcessor(vocab_size, scale=0.5, soft_cap=2.0)

    results, _ = _simulate_tp(lp, _shard_heads(weight, tp_size), hidden_states, k)

    full = torch.nn.functional.linear(hidden_states, weight)
    for ids, values in results:
        _assert_top_k(
            ids, values, full, k, lambda v: torch.tanh(v.float() * 0.5 / 2.0) * 2.0
        )


def test_top_k_tokens_tp1_skips_the_gather(default_vllm_config):
    torch.manual_seed(2)
    vocab_size, hidden_size, batch, k = 64, 16, 2, 8
    hidden_states = torch.randn(batch, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lp = LogitsProcessor(vocab_size)
    head = _FakeShardHead(weight, 0, tp_size=1)

    with mock.patch(_ALL_GATHER) as all_gather:
        ids, values = lp.get_top_k_tokens(head, hidden_states, k)
    all_gather.assert_not_called()

    full = torch.nn.functional.linear(hidden_states, weight)
    _assert_top_k(ids, values, full, k)
