# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers import l2_prefetch


class _Linear(nn.Module):
    def __init__(self, rows: int = 4, columns: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(rows, columns, dtype=torch.bfloat16),
            requires_grad=False,
        )


class _AttentionWrapper(nn.Module):
    def __init__(self, *, skip_topk: bool) -> None:
        super().__init__()
        self.indexer = nn.Module()
        self.indexer.wq_b = _Linear()
        self.skip_topk = skip_topk


class _Attention(nn.Module):
    def __init__(self, *, skip_topk: bool, o_proj: bool = True) -> None:
        super().__init__()
        self.fused_qkv_a_proj = _Linear()
        self.q_b_proj = _Linear()
        if o_proj:
            self.o_proj = _Linear()
        wrapper = _AttentionWrapper(skip_topk=skip_topk)
        self.indexer = wrapper.indexer
        self.skip_topk = wrapper.skip_topk


class _Mlp(nn.Module):
    def __init__(self, *, moe: bool) -> None:
        super().__init__()
        if moe:
            self.experts = nn.Module()


class _DecoderLayer(nn.Module):
    def __init__(self, *, moe: bool, skip_topk: bool, o_proj: bool = True) -> None:
        super().__init__()
        self.self_attn = _Attention(skip_topk=skip_topk, o_proj=o_proj)
        self.mlp = _Mlp(moe=moe)


@pytest.mark.parametrize(
    ("mode", "capturing", "expected"),
    [
        (CUDAGraphMode.FULL, False, True),
        (CUDAGraphMode.PIECEWISE, True, False),
        (CUDAGraphMode.NONE, True, True),
        (CUDAGraphMode.NONE, False, False),
    ],
)
def test_active_requires_full_graph_or_capture(
    monkeypatch: pytest.MonkeyPatch,
    mode: CUDAGraphMode,
    capturing: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(l2_prefetch, "ENABLED", True)
    monkeypatch.setattr(l2_prefetch, "MAX_TOKENS", 16)
    monkeypatch.setattr(l2_prefetch, "WINDOWS", frozenset({"moe", "attn"}))
    monkeypatch.setattr(l2_prefetch, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        l2_prefetch,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=mode),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)

    assert l2_prefetch._active(16) is expected
    assert not l2_prefetch._active(17)


@pytest.mark.parametrize(
    ("windows", "num_tokens", "moe_expected", "attn_expected"),
    [
        (frozenset({"moe", "attn"}), 8, True, True),
        (frozenset({"moe", "attn"}), 9, True, False),
        (frozenset({"moe", "attn"}), 17, False, False),
        (frozenset({"moe"}), 4, True, False),
        (frozenset({"attn"}), 4, False, True),
        (frozenset(), 4, False, False),
    ],
)
def test_windows_gate_independently(
    monkeypatch: pytest.MonkeyPatch,
    windows: frozenset[str],
    num_tokens: int,
    moe_expected: bool,
    attn_expected: bool,
) -> None:
    monkeypatch.setattr(l2_prefetch, "ENABLED", True)
    monkeypatch.setattr(l2_prefetch, "MAX_TOKENS", 16)
    monkeypatch.setattr(l2_prefetch, "ATTN_MAX_TOKENS", 8)
    monkeypatch.setattr(l2_prefetch, "WINDOWS", windows)
    monkeypatch.setattr(l2_prefetch, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        l2_prefetch,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
    )

    assert l2_prefetch._active(num_tokens, l2_prefetch.MOE_WINDOW) is moe_expected
    assert l2_prefetch._active(num_tokens, l2_prefetch.ATTN_WINDOW) is attn_expected


def test_default_program_counts() -> None:
    """The input-projection window loads with 64 programs and the output
    projection window with 32 unless the environment overrides them."""
    assert (
        int(os.environ.get("VLLM_L2_PREFETCH_CTAS", "64")) == l2_prefetch.NUM_PROGRAMS
    )
    assert (
        int(os.environ.get("VLLM_L2_PREFETCH_ATTN_CTAS", "32"))
        == l2_prefetch.ATTN_NUM_PROGRAMS
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", 1), ("64", 64), (str(1 << 16), 1 << 16)],
)
def test_program_count_accepts_one_word_per_program(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    monkeypatch.setenv("VLLM_L2_PREFETCH_CTAS", value)
    assert l2_prefetch._program_count("VLLM_L2_PREFETCH_CTAS", "64") == expected


@pytest.mark.parametrize("value", ["0", "-1", str((1 << 16) + 1), "many"])
def test_program_count_rejects_counts_outside_the_sink(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Every program stores one word into the sink, so the count must stay
    within the sink's words; a non-integer value is rejected as well."""
    monkeypatch.setenv("VLLM_L2_PREFETCH_ATTN_CTAS", value)
    with pytest.raises(ValueError):
        l2_prefetch._program_count("VLLM_L2_PREFETCH_ATTN_CTAS", "32")


def test_program_count_uses_the_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_L2_PREFETCH_CTAS", raising=False)
    assert l2_prefetch._program_count("VLLM_L2_PREFETCH_CTAS", "64") == 64


def test_window_program_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(l2_prefetch, "NUM_PROGRAMS", 32)
    monkeypatch.setattr(l2_prefetch, "ATTN_NUM_PROGRAMS", 16)
    assert l2_prefetch._window_num_programs(l2_prefetch.MOE_WINDOW) == 32
    assert l2_prefetch._window_num_programs(l2_prefetch.ATTN_WINDOW) == 16


def test_register_decoder_layers_attaches_fixed_next_layer_weights() -> None:
    layers = nn.ModuleList(
        [
            _DecoderLayer(moe=False, skip_topk=True),
            _DecoderLayer(moe=True, skip_topk=False),
            _DecoderLayer(moe=True, skip_topk=True),
        ]
    )

    assert l2_prefetch.register_attention_layers(layers, 0, len(layers)) == 2

    assert layers[0]._l2_prefetch_weights is None

    dense_targets = layers[1]._l2_prefetch_weights
    assert dense_targets == [
        layers[1].self_attn.fused_qkv_a_proj.weight,
        layers[1].self_attn.q_b_proj.weight,
        layers[1].self_attn.indexer.wq_b.weight,
    ]

    moe_targets = layers[2]._l2_prefetch_weights
    assert moe_targets == [
        layers[2].self_attn.fused_qkv_a_proj.weight,
        layers[2].self_attn.q_b_proj.weight,
    ]


def test_register_decoder_layers_attaches_output_projection_per_layer() -> None:
    layers = nn.ModuleList(
        [
            _DecoderLayer(moe=False, skip_topk=True),
            _DecoderLayer(moe=True, skip_topk=False),
            _DecoderLayer(moe=True, skip_topk=True, o_proj=False),
        ]
    )

    l2_prefetch.register_attention_layers(layers, 0, len(layers))

    # Every layer, including the first local one, loads its own o_proj; the
    # router and shared-expert weights are not part of the list.
    assert layers[0]._l2_prefetch_attn_weights == [layers[0].self_attn.o_proj.weight]
    assert layers[1]._l2_prefetch_attn_weights == [layers[1].self_attn.o_proj.weight]
    assert layers[2]._l2_prefetch_attn_weights is None
    assert layers[0]._l2_prefetch_weights is None


def test_issue_passes_window_to_op(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, str]] = []

    def record(hidden_states, weights, sink, window):
        calls.append((len(weights), window))

    monkeypatch.setattr(torch.ops.vllm, "l2_weight_prefetch", record)
    monkeypatch.setattr(l2_prefetch, "_sink", lambda device: None)
    hidden_states = torch.empty(4, 8, dtype=torch.bfloat16)
    cuda_like = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    l2_prefetch.issue(cuda_like, [hidden_states], l2_prefetch.ATTN_WINDOW)
    l2_prefetch.issue(cuda_like, [hidden_states, hidden_states])
    l2_prefetch.issue(cuda_like, None, l2_prefetch.ATTN_WINDOW)
    assert calls == [(1, "attn"), (2, "moe")]


def test_issue_and_join_ignore_cpu_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("CPU tensors must not enter CUDA prefetch operations")

    monkeypatch.setattr(torch.ops.vllm, "l2_weight_prefetch", fail)
    monkeypatch.setattr(torch.ops.vllm, "l2_weight_prefetch_join", fail)
    hidden_states = torch.empty(4, 8, dtype=torch.bfloat16)

    l2_prefetch.issue(hidden_states, [hidden_states])
    l2_prefetch.join(hidden_states)
