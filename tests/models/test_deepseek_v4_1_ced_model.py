# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CED orchestration: full encoder context, bounded decoder, original output ABI.

The real model/layer forwards drive a small deterministic mathematical model;
attention kernels and mapped metadata are qualified by the GPU CED tests.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4_1.nvidia import model as native


def _gather(source, indices):
    result = source[indices.clamp_min(0)].clone()
    result[indices < 0] = 0
    return result


def _scatter(source, indices, full_rows):
    result = source.new_zeros((full_rows, *source.shape[1:]))
    valid = indices >= 0
    result[indices[valid]] = source[valid]
    return result


class _Mix:
    def post(self, x, residual, post, comb):
        return residual + x[:, None, :] * 0.25

    def pre(self, residual, fn, scale, base, norm, pre, **previous):
        if previous.get("previous_output") is not None:
            residual = self.post(previous["previous_output"], residual, None, None)
        if residual.ndim == 2:
            residual = residual[:, None, :].expand(-1, 4, -1).clone()
        rows = residual.shape[0]
        mix = residual.new_zeros((rows, 4))
        mix[:, 0] = 1
        return (
            residual,
            residual.new_full((rows, 4), 0.25),
            torch.eye(4).expand(rows, 4, 4),
            residual[:, 0, :],
            mix,
        )

    def post_pre(self, x, residual, post, comb, fn, scale, base, norm, pre):
        return self.pre(self.post(x, residual, post, comb), fn, scale, base, norm, pre)


class _Attention(nn.Module):
    def __init__(self, index, shared, rows):
        super().__init__()
        self.index, self.shared, self.rows = index, shared, rows

    def prepare_global_kv(self, positions, hidden):
        self.shared["global"] = hidden.mean(dim=0)
        self.shared["global_rows"] = hidden.shape[0]

    def forward(self, positions, hidden, scaling, *, global_kv_ready=None):
        if self.index == 2 and "global" not in self.shared:
            self.prepare_global_kv(positions, hidden)
        self.rows.append(("attention", self.index, hidden.shape[0]))
        result = hidden + self.index + 1 + positions[:, None].float() * 0.01
        if self.index >= 2:
            result = result + self.shared["global"]
        return result


class _Experts(nn.Module):
    def __init__(self, index, rows):
        super().__init__()
        self.index, self.rows = index, rows

    def forward(self, hidden, ids):
        self.rows.append(("experts", self.index, hidden.shape[0]))
        return hidden * 0.5 + ids[:, None].float() * 0.02


class _Layer(nn.Module):
    forward = native.DeepseekV4DecoderLayer.forward

    def __init__(self, index, shared, rows):
        super().__init__()
        self.engram = None
        self.hc_mult = 4
        self._b12x_mhc = _Mix()
        self.attn = _Attention(index, shared, rows)
        self.ffn = _Experts(index, rows)
        for name in (
            "hc_attn_fn_broadcast",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
        ):
            setattr(self, name, torch.ones(1))
        self.attn_norm = self.ffn_norm = SimpleNamespace(weight=torch.ones(2))


@pytest.mark.parametrize("compact", [False, True])
def test_ced_global_context_and_full_row_output_abi(monkeypatch, compact):
    monkeypatch.setattr(
        native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(native, "gather_rows", _gather)
    monkeypatch.setattr(native, "scatter_rows", _scatter)
    monkeypatch.setattr(native, "collapse", lambda hidden, mix: hidden[:, 0])
    monkeypatch.setattr(native, "stream_mean", lambda hidden: hidden.mean(dim=1))
    ids = torch.arange(8)
    positions = torch.arange(100, 108)
    inputs = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 8
    indices = torch.tensor([2, 3, 6, 7, -1, -1]) if compact else None
    shared, row_work = {}, []
    buffer = torch.empty(8, 8)
    target = SimpleNamespace(
        embed_input_ids=lambda _: inputs,
        use_mega_moe=False,
        use_sequence_parallel=False,
        disk_engram=False,
        engram_hash=None,
        start_layer=0,
        end_layer=4,
        ced_decoder_start=2,
        layers=nn.ModuleList(_Layer(i, shared, row_work) for i in range(4)),
        aux_hidden_state_layers={3, 4},
        _mtp_hidden_buffer=buffer,
        norm=lambda hidden: hidden,
    )
    output, aux = native.DeepseekV4Model.forward(
        target, ids, positions, None, ced_indices=indices
    )

    # Independent arithmetic oracle; decoder global context includes encoder
    # rows that will never enter decoder attention or experts.
    expected = inputs.clone()
    expected_aux = []
    global_context = None
    for index in range(4):
        if index == 2:
            global_context = expected.mean(dim=0)
        attention = expected + index + 1 + positions[:, None].float() * 0.01
        if index >= 2:
            attention = attention + global_context
        after_attention = expected + attention * 0.25
        expected = (
            after_attention
            + (after_attention * 0.5 + ids[:, None].float() * 0.02) * 0.25
        )
        if index >= 2:
            expected_aux.append(expected.clone())
    if compact:
        keep = torch.zeros(8, dtype=torch.bool)
        keep[indices[indices >= 0]] = True
        expected[~keep] = 0
        for value in expected_aux:
            value[~keep] = 0
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)
    for actual, value in zip(aux, expected_aux, strict=True):
        torch.testing.assert_close(actual, value, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        buffer.view(8, 4, 2),
        expected[:, None, :].expand(-1, 4, -1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert shared["global_rows"] == 8
    decoder_rows = len(indices) if compact else 8
    assert row_work == [
        (kind, i, 8 if i < 2 else decoder_rows)
        for i in range(4)
        for kind in ("attention", "experts")
    ]
