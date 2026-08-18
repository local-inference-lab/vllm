# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4 import attention as attention_mod


class _ScoringReached(Exception):
    pass


class _EventPool:
    def lease(self, *, private_eager: bool):
        assert private_eager
        return nullcontext((object(), object()))


class _ShortContextKernel:
    def __init__(self) -> None:
        self.launches = 0

    def __getitem__(self, grid):
        assert grid == (1,)

        def launch(*args, **kwargs) -> None:
            del args, kwargs
            self.launches += 1

        return launch


@pytest.mark.parametrize("capturing", [False, True])
def test_indexer_short_context_shortcut_is_eager_only(
    monkeypatch: pytest.MonkeyPatch,
    capturing: bool,
) -> None:
    """CUDA graph capture must record learned indexer scoring.

    Breakable graph capture supplies short dummy metadata, while replay can
    receive a long cached prefix. Recording the eager all-candidates shortcut
    would make replay select the first candidates without scoring them.
    """

    indexer = object.__new__(attention_mod.DeepseekV4Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.compress_ratio = 4
    indexer.topk_tokens = 512
    indexer.topk_indices_buffer = torch.empty((1, 512), dtype=torch.int32)
    indexer.k_cache = SimpleNamespace(prefix="indexer.k_cache")
    indexer.event_pool = _EventPool()
    indexer.aux_stream = None

    compressor_calls = 0

    def compressor(*args) -> None:
        nonlocal compressor_calls
        del args
        compressor_calls += 1

    indexer.compressor = compressor
    metadata = SimpleNamespace(
        max_seq_len=2048,
        num_decode_tokens=1,
        num_prefill_tokens=0,
    )
    monkeypatch.setattr(
        attention_mod,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={indexer.k_cache.prefix: metadata},
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing,
    )
    short_context_kernel = _ShortContextKernel()
    monkeypatch.setattr(
        attention_mod,
        "_fill_short_context_topk_indices",
        short_context_kernel,
    )
    monkeypatch.setattr(
        attention_mod.triton,
        "next_power_of_2",
        lambda value: value,
        raising=False,
    )

    def scoring_path(*args, **kwargs):
        del args, kwargs
        raise _ScoringReached

    monkeypatch.setattr(attention_mod, "maybe_execute_in_parallel", scoring_path)
    inputs = (
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        torch.empty((1, 1)),
        torch.tensor([2047]),
        SimpleNamespace(cos_sin_cache=torch.empty(0)),
    )

    if capturing:
        with pytest.raises(_ScoringReached):
            indexer(*inputs)
        assert compressor_calls == 0
        assert short_context_kernel.launches == 0
    else:
        assert indexer(*inputs) == (None, None, None)
        assert compressor_calls == 1
        assert short_context_kernel.launches == 1
