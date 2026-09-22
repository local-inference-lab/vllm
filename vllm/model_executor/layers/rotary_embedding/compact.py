# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded position tables for unscaled rotary embeddings."""

import torch

from vllm import _custom_ops as ops

from .base import RotaryEmbedding


class CompactRotaryEmbedding(RotaryEmbedding):
    """Materialize only positions consumed by a forward in retained storage.

    Frequencies and trigonometric functions use the full-table implementation's
    FP32 operations, followed by the same cache-dtype conversion. Capacity is a
    maximum row count, not a maximum global position. Buffers are instance-owned
    and reused only by serial forwards, including CUDA Graph replay.
    """

    def __init__(self, *args, capacity: int, **kwargs):
        if capacity <= 0:
            raise ValueError("compact rotary capacity must be positive")
        super().__init__(*args, **kwargs, init_cache=False)
        self.capacity = capacity
        inv_freq = self._compute_inv_freq(self.base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "freqs_workspace",
            torch.empty(
                capacity,
                self.rotary_dim // 2,
                dtype=torch.float32,
                device=inv_freq.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "cos_sin_cache",
            torch.empty(
                capacity, self.rotary_dim, dtype=self.dtype, device=inv_freq.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "local_positions",
            torch.arange(capacity, dtype=torch.int64, device=inv_freq.device),
            persistent=False,
        )

    def materialize(self, positions):
        if positions.ndim != 1 or positions.numel() > self.capacity:
            raise ValueError("rotary position rows exceed prepared capacity")
        rows = positions.numel()
        freqs = self.freqs_workspace[:rows]
        cache = self.cos_sin_cache[:rows]
        half = self.rotary_dim // 2
        torch.mul(positions[:, None], self.inv_freq[None, :], out=freqs)
        torch.cos(freqs, out=cache[:, :half])
        torch.sin(freqs, out=cache[:, half:])
        return self.local_positions[:rows], cache

    def forward_cuda(self, positions, query, key=None):
        if query.dtype != self.dtype:
            raise ValueError("compact rotary cache and query dtypes must match")
        local_positions, cache = self.materialize(positions)
        ops.rotary_embedding(
            local_positions, query, key, self.head_size, cache, self.is_neox_style
        )
        return query, key

    def forward_native(self, positions, query, key=None):
        local_positions, cache = self.materialize(positions)
        return self.forward_static(
            local_positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cache,
            self.is_neox_style,
        )
