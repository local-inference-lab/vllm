# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A mixed DCP batch splits into exchanged decode rows and gathered prefill rows."""

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v32.attention import DeepseekV32Attention


class _RecordingImpl:
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, int, int, int]] = []

    def forward_mqa(
        self, q, kv_cache, attn_metadata, layer, *, row_start=0, route=None
    ):
        self.calls.append((route, row_start, q.shape[0], q.shape[1]))
        if route == "extend":
            return q[..., :512].clone(), torch.zeros(q.shape[:2])
        return q[..., :512] + 100.0, torch.zeros(q.shape[:2])


class _DCP2Manager:
    """Doubles the heads on gather and keeps the local half on combine."""

    def query_gather(self, q: torch.Tensor) -> torch.Tensor:
        return torch.cat([q, q], dim=1)

    def combine(self, out, lse, seq_lens=None, query_start_loc=None):
        assert seq_lens is None and query_start_loc is None
        return out[:, : out.shape[1] // 2]


def test_mixed_dcp_batch_exchanges_decode_rows_and_gathers_prefill_rows() -> None:
    layer = DeepseekV32Attention.__new__(DeepseekV32Attention)
    torch.nn.Module.__init__(layer)
    layer.impl = _RecordingImpl()
    layer.dcp_manager = _DCP2Manager()
    q_nope, q_pe = torch.randn(6, 4, 512), torch.randn(6, 4, 64)

    out = layer._split_ckv_dcp_mqa(
        (q_nope, q_pe),
        kv_cache=None,
        attn_metadata=SimpleNamespace(num_decode_tokens=2),
        num_actual=6,
    )

    # Decode rows: all DCP heads over the local shard; prefill rows: local
    # heads over the gathered cache, addressed from the first prefill row.
    assert layer.impl.calls == [("extend", 0, 2, 8), ("ckv_extend", 2, 4, 4)]
    torch.testing.assert_close(out[:2], q_nope[:2])
    torch.testing.assert_close(out[2:], q_nope[2:] + 100.0)
