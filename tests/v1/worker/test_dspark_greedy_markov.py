# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact full-vocabulary DSpark greedy tokens, including graph replay."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.greedy import (
    sample_greedy_markov,
    scratch_shape,
)


@pytest.mark.parametrize("base_vocab,bias_vocab", [(2049, 2048), (2048, 2049)])
def test_greedy_markov_rejects_mismatched_vocabulary_before_dispatch(
    base_vocab, bias_vocab
):
    """An invalid model head must fail before any GPU buffer is accessed."""
    base = torch.empty((2, base_vocab))
    bias = torch.empty((2, bias_vocab))
    output = torch.empty(2, dtype=torch.int64)
    shape = scratch_shape(2, max(base_vocab, bias_vocab))
    with pytest.raises(ValueError, match="identical shapes"):
        sample_greedy_markov(
            base,
            bias,
            output,
            torch.empty(shape),
            torch.empty(shape, dtype=torch.int32),
        )


@pytest.mark.parametrize("short_values", [False, True])
def test_greedy_markov_rejects_insufficient_scratch_before_dispatch(short_values):
    base = torch.empty((2, 2049))
    bias = torch.empty_like(base)
    output = torch.empty(2, dtype=torch.int64)
    values = torch.empty(3 if short_values else 4)
    indices = torch.empty(4 if short_values else 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="scratch is smaller"):
        sample_greedy_markov(base, bias, output, values, indices)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("vocab", [1, 2047, 2048, 2049, 129280])
@pytest.mark.parametrize("rows", [1, 4])
def test_greedy_markov_exact_replay(dtype, vocab, rows):
    torch.manual_seed(41612)
    storage = torch.randn((rows, 7, vocab), device="cuda", dtype=dtype)
    base = storage[:, 3]
    bias = torch.randn_like(base)
    result = torch.full((rows, 7), -77, device="cuda", dtype=torch.int64)
    output = result[:, 3]
    shape = scratch_shape(rows, vocab)
    values = torch.full(shape, float("nan"), device="cuda")
    indices = torch.full(shape, -1, device="cuda", dtype=torch.int32)
    run = lambda: sample_greedy_markov(base, bias, output, values, indices)
    run()
    torch.testing.assert_close(output, (base + bias).argmax(-1), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for case in ("mutated", "ties", "negative-infinity", "nan"):
        if case == "mutated":
            base.normal_()
            bias.normal_()
        elif case == "ties":
            base.fill_(-1)
            bias.zero_()
            base[:, 0] = 2
            base[:, -1] = 2
        elif case == "negative-infinity":
            base.fill_(-float("inf"))
            bias.zero_()
        else:
            base.zero_()
            base[:, -1] = float("nan")
            base[:, vocab // 2] = float("nan")
        values.fill_(float("nan"))
        indices.fill_(-1)
        allocated = torch.accelerator.memory_allocated()
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.accelerator.memory_allocated() == allocated
        torch.testing.assert_close(output, (base + bias).argmax(-1), rtol=0, atol=0)
        assert torch.all(result[:, :3] == -77)
        assert torch.all(result[:, 4:] == -77)
