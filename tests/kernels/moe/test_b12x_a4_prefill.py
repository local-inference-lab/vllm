# SPDX-License-Identifier: Apache-2.0
"""Host-side contract of the opt-in A4 prefill for b12x W4A16 MoE.

Covers the enabling variable, the leading decode-row count read from the
forward context, and the per-row-type choice: decode rows run W4A16, prefill
rows A4. The
b12x library itself is replaced by a recorder; no GPU is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm.forward_context as forward_context
import vllm.model_executor.layers.fused_moe.b12x as b12x
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x import B12xExperts


def test_min_tokens_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", raising=False)
    assert b12x._w4a16_a4_prefill_min_tokens() == 0
    assert not b12x._w4a16_a4_prefill_enabled()
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", "1536")
    assert b12x._w4a16_a4_prefill_min_tokens() == 1536
    assert b12x._w4a16_a4_prefill_enabled()
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", "-4")
    assert b12x._w4a16_a4_prefill_min_tokens() == 0


def _fake_context(monkeypatch, starts, gdn_prefills=None):
    metadata = {
        "layer.0": SimpleNamespace(
            query_start_loc=torch.tensor(starts, dtype=torch.int32),
            num_actual_tokens=starts[-1],
        )
    }
    if gdn_prefills is not None:
        # A GDN/KDA layer's host-side counts.
        metadata["layer.1"] = SimpleNamespace(
            num_prefills=gdn_prefills, num_spec_decode_tokens=starts[-1]
        )
    context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    return context


@pytest.mark.parametrize(
    "starts, rows",
    [
        ([0, 4, 8, 12, 2312], 12),  # three MTP3 decode requests, then a prefill chunk
        ([0, 2300], 0),  # prefill only
        ([0, 1, 2, 3], 3),  # decode only
        ([0, 4, 2304, 2308], 4),  # a decode request after the prefill is not leading
    ],
)
def test_leading_decode_rows(monkeypatch: pytest.MonkeyPatch, starts, rows) -> None:
    context = _fake_context(monkeypatch, starts)
    assert b12x._num_leading_decode_tokens(starts[-1]) == rows
    # Cached on the forward context: one host read per step.
    assert context._b12x_a4_decode_rows == rows
    context.attn_metadata = {}
    assert b12x._num_leading_decode_tokens(starts[-1]) == rows


def test_a_decode_only_step_needs_no_device_read(monkeypatch: pytest.MonkeyPatch):
    context = _fake_context(monkeypatch, [0, 4, 8], gdn_prefills=0)
    context.attn_metadata["layer.0"].query_start_loc = None  # would need a sync
    assert b12x._num_leading_decode_tokens(8) == 8


def test_leading_decode_rows_without_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: False)
    assert b12x._num_leading_decode_tokens(8) == 0


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bool | None]] = []

    def bind(
        self, plan, *, scratch, a, experts, topk_weights, topk_ids, output, **kwargs
    ):
        del plan, scratch, experts
        assert (
            a.shape[0] == topk_weights.shape[0] == topk_ids.shape[0] == output.shape[0]
        )
        return SimpleNamespace(
            rows=int(a.shape[0]),
            first=int(a.data_ptr()),
            a4=kwargs.get("a4_prefill"),
        )

    def run(self, *, binding) -> None:
        self.calls.append((binding.rows, binding.first, binding.a4))


def _experts(monkeypatch, quant_mode: str, a4_scales: bool = True):
    recorder = _Recorder()
    monkeypatch.setattr(b12x, "_require_b12x_fused_moe", lambda: recorder)
    monkeypatch.setattr(b12x, "_is_current_stream_capturing", lambda: False)
    experts = object.__new__(B12xExperts)
    experts._apply_router_weight_on_input = False
    experts._quant_mode = quant_mode
    prepared = SimpleNamespace(_impl=SimpleNamespace(a4_prefill_scales=a4_scales))
    experts._prepared = lambda: prepared
    experts._plan_for_tokens = lambda tokens, **kwargs: object()
    return experts, recorder


def _apply(experts, tokens):
    hidden = torch.zeros(tokens, 8, dtype=torch.bfloat16)
    experts.apply(
        output=torch.zeros_like(hidden),
        hidden_states=hidden,
        w1=None,
        w2=None,
        topk_weights=torch.ones(tokens, 2),
        topk_ids=torch.zeros(tokens, 2, dtype=torch.int32),
        activation=MoEActivation.SILU,
        global_num_experts=4,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=torch.zeros(64, dtype=torch.uint8),
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return hidden


@pytest.mark.parametrize("tokens", [2312, 64])
def test_decode_rows_stay_w4a16_and_prefill_rows_run_a4(
    monkeypatch: pytest.MonkeyPatch, tokens
) -> None:
    """Small and large prefills alike: the row type decides, not the size."""
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", "1536")
    _fake_context(monkeypatch, [0, 4, 8, tokens])
    experts, recorder = _experts(monkeypatch, "w4a16")
    hidden = _apply(experts, tokens)
    row = hidden.element_size() * hidden.shape[1]
    assert recorder.calls == [
        (8, hidden.data_ptr(), False),
        (tokens - 8, hidden.data_ptr() + 8 * row, True),
    ]


def test_a_prefill_only_step_runs_a4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", "1536")
    _fake_context(monkeypatch, [0, 300])
    experts, recorder = _experts(monkeypatch, "w4a16")
    _apply(experts, 300)
    assert [(rows, a4) for rows, _, a4 in recorder.calls] == [(300, True)]


def test_a_decode_only_step_stays_w4a16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", "1536")
    _fake_context(monkeypatch, [0, 4, 8, 12], gdn_prefills=0)
    experts, recorder = _experts(monkeypatch, "w4a16")
    _apply(experts, 12)
    assert [(rows, a4) for rows, _, a4 in recorder.calls] == [(12, False)]


def test_a_captured_graph_stays_w4a16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", "1536")
    experts, recorder = _experts(monkeypatch, "w4a16")
    monkeypatch.setattr(b12x, "_is_current_stream_capturing", lambda: True)
    _apply(experts, 32)
    assert [(rows, a4) for rows, _, a4 in recorder.calls] == [(32, False)]


@pytest.mark.parametrize(
    "quant_mode, threshold, a4_scales",
    [
        ("w4a16", "0", True),  # option off
        ("nvfp4", "1536", True),  # not a W4A16 plan
        ("w4a16", "1536", False),  # no activation scales (the MTP draft layer)
    ],
)
def test_no_choice_without_an_a4_path(
    monkeypatch: pytest.MonkeyPatch, quant_mode, threshold, a4_scales
) -> None:
    monkeypatch.setenv("B12X_W4A16_A4_PREFILL_MIN_TOKENS", threshold)
    _fake_context(monkeypatch, [0, 4, 2312])
    experts, recorder = _experts(monkeypatch, quant_mode, a4_scales)
    _apply(experts, 2312)
    assert [(rows, a4) for rows, _, a4 in recorder.calls] == [(2312, None)]
