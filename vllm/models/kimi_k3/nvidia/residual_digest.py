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

``VLLM_K3_RESIDUAL_DUMP_TAPS=<name>,<name>,...`` additionally stores the
first occurrence per forward of each named tap as a raw host tensor under
``raw`` in the same file (the first MLA layer for the ``mla.*`` taps), on the
ranks listed in ``VLLM_K3_RESIDUAL_DUMP_RANKS`` (default ``0``), so a
divergent digest can be quantified and located by row.
"""

from __future__ import annotations

import os
import threading
import time

import torch

BLOCK_ROWS = 256
# ``active`` maps a thread id to the forward digest running on that thread:
# the two-thread split prefill runs one forward per thread, and each tap must
# reach the digest of its own forward. ``meta`` is the runner's description of
# the scheduler step being executed; every digest created during the step
# records a copy of it.
_state: dict = {"seq": 0, "weights": None, "active": {}, "meta": {}}


def _current() -> "ForwardDigest | None":
    active = _state["active"]
    digest = active.get(threading.get_ident())
    if digest is None and len(active) == 1:
        digest = next(iter(active.values()))
    return digest


def tap(name: str, x: torch.Tensor) -> None:
    """Digest an intermediate of the active forward under ``name`` (for
    example ``attn`` for a layer's attention output or ``mlp`` for its MLP
    output). No-op without an active digest."""
    digest = _current()
    if digest is not None:
        digest.add_tap(name, x)


def set_step_meta(meta: dict) -> None:
    """Describe the scheduler step about to run (recorded by every forward
    digest of the step under ``meta``)."""
    _state["meta"] = dict(meta)


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


def _dump_taps(rank: int) -> frozenset[str]:
    """Tap names whose first occurrence per forward is stored raw on ``rank``."""
    names = os.getenv("VLLM_K3_RESIDUAL_DUMP_TAPS", "")
    if not names:
        return frozenset()
    ranks = os.getenv("VLLM_K3_RESIDUAL_DUMP_RANKS", "0")
    if str(rank) not in {r.strip() for r in ranks.split(",") if r.strip()}:
        return frozenset()
    return frozenset(n.strip() for n in names.split(",") if n.strip())


class ForwardDigest:
    """Collects one forward's per-layer digests and writes them at the end."""

    def __init__(self, num_layers: int, first_position: int, rank: int) -> None:
        self.rows: list[torch.Tensor] = []
        self.taps: dict[str, list[torch.Tensor]] = {}
        self.raw: dict[str, torch.Tensor] = {}
        self.first_position = first_position
        self.rank = rank
        self.num_layers = num_layers
        self.meta = dict(_state["meta"])
        self.meta["created"] = time.time()
        self.meta["thread"] = threading.get_ident()
        _state["active"][threading.get_ident()] = self

    def add(self, residual: torch.Tensor) -> None:
        self.rows.append(block_digests(residual, 0))

    def add_tap(self, name: str, x: torch.Tensor) -> None:
        x = x.detach()
        if x.ndim != 2:
            x = x.reshape(x.shape[0], -1)
        if not x.is_contiguous():
            x = x.contiguous()
        if name not in self.raw and name in _dump_taps(self.rank):
            self.raw[name] = x.cpu().clone()
        self.taps.setdefault(name, []).append(block_digests(x, 0))

    @staticmethod
    def _table(rows: list[torch.Tensor]) -> torch.Tensor:
        width = max(r.shape[0] for r in rows)
        table = torch.zeros(len(rows), width, dtype=torch.int64)
        for i, r in enumerate(rows):
            table[i, : r.shape[0]] = r.cpu()
        return table

    def flush(self) -> None:
        active = _state["active"]
        for tid, digest in list(active.items()):
            if digest is self:
                del active[tid]
        if not self.rows:
            return
        table = self._table(self.rows)
        directory = os.getenv("VLLM_K3_RESIDUAL_DIGEST_DIR", "")
        os.makedirs(directory, exist_ok=True)
        seq = _state["seq"]
        _state["seq"] = seq + 1
        torch.save(
            {
                "digests": table,
                "taps": {name: self._table(rows) for name, rows in self.taps.items()},
                "raw": self.raw,
                "meta": self.meta,
                "first_position": self.first_position,
                "block_rows": BLOCK_ROWS,
                "rank": self.rank,
            },
            os.path.join(directory, f"digest-rank{self.rank}-{seq:05d}.pt"),
        )
