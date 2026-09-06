# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer, per-row-block digests of the Kimi-K3 residual stream.

Diagnostic for the split-prefill exactness study
(research/prefill-campaign-20260906/r1-split-prefill.md): with
``VLLM_K3_RESIDUAL_DIGEST_DIR=<dir>`` every prefill forward writes one file
``digest-rank<r>-<seq>.pt`` holding an int64 tensor ``[layers, blocks]``
where entry ``(l, b)`` is an exact integer digest of the bf16 bits of rows
``[b * BLOCK_ROWS, (b + 1) * BLOCK_ROWS)`` of the layer output (the MLP
output after the layer's final all-reduce) of layer ``l``; the file also
records the absolute position of the forward's first row, so a half-chunk
forward is aligned to the unsplit chunk by position. Two forwards are
bit-identical on a row block exactly when their digests agree on that
block, so the first divergent layer and block can be located without
dumping activations (``evidence/r1/compare_digests.py``).

The digest is computed on the device and copied to the host once per
forward. It is inactive without the environment variable.
"""

from __future__ import annotations

import os

import torch

BLOCK_ROWS = 256
_state: dict = {"seq": 0, "weights": None}


def enabled() -> bool:
    return bool(os.getenv("VLLM_K3_RESIDUAL_DIGEST_DIR", ""))


def _weights(width: int, device: torch.device) -> torch.Tensor:
    w = _state["weights"]
    if w is None or w.shape[0] != width or w.device != device:
        # Odd multipliers spread the bit patterns; the sum of
        # BLOCK_ROWS * width terms of |v| < 2^15 * 2^20 stays within int64.
        w = torch.arange(width, device=device, dtype=torch.int64) * 2 + 1
        w = (w * 0x9E3779B1) % (1 << 20) | 1
        _state["weights"] = w
    return w


def block_digests(x: torch.Tensor, row_offset: int) -> torch.Tensor:
    """Digest ``x`` (``[rows, width]`` bf16) per BLOCK_ROWS rows.

    ``row_offset`` is the position of ``x``'s first row in the chunk, so a
    half-chunk forward labels its blocks like the unsplit chunk.
    """
    x = x.detach()
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    rows, width = x.shape
    bits = x.view(torch.int16).to(torch.int64)
    w = _weights(width, x.device)
    per_row = (bits * w).sum(dim=1)
    n_blocks = (row_offset + rows + BLOCK_ROWS - 1) // BLOCK_ROWS
    out = torch.zeros(n_blocks, dtype=torch.int64, device=x.device)
    block_idx = (torch.arange(rows, device=x.device) + row_offset) // BLOCK_ROWS
    out.index_add_(0, block_idx, per_row)
    return out


class ForwardDigest:
    """Collects one forward's per-layer digests and writes them at the end."""

    def __init__(self, num_layers: int, first_position: int, rank: int) -> None:
        self.rows: list[torch.Tensor] = []
        self.first_position = first_position
        self.rank = rank
        self.num_layers = num_layers

    def add(self, residual: torch.Tensor) -> None:
        self.rows.append(block_digests(residual, 0))

    def flush(self) -> None:
        if not self.rows:
            return
        width = max(r.shape[0] for r in self.rows)
        table = torch.zeros(len(self.rows), width, dtype=torch.int64)
        for i, r in enumerate(self.rows):
            table[i, : r.shape[0]] = r.cpu()
        directory = os.getenv("VLLM_K3_RESIDUAL_DIGEST_DIR", "")
        os.makedirs(directory, exist_ok=True)
        seq = _state["seq"]
        _state["seq"] = seq + 1
        torch.save(
            {
                "digests": table,
                "first_position": self.first_position,
                "block_rows": BLOCK_ROWS,
                "rank": self.rank,
            },
            os.path.join(directory, f"digest-rank{self.rank}-{seq:05d}.pt"),
        )
