# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retained BF16 input storage for sequential auxiliary-state producers."""

from __future__ import annotations

import torch


class Bf16InputAccumulator:
    """Copy feature slices without quantization, then invoke one local linear.

    The caller supplies a BF16-input quantization method. Output gathering is
    owned by the caller: invoking ``quant_method.apply`` avoids running a
    column-parallel layer's collective twice. No source slice remains live
    after its copy completes on the caller's stream.
    """

    def __init__(self, layer, reference: torch.Tensor, slice_width: int):
        self.input_width = int(layer.input_size)
        if reference.ndim != 2 or reference.dtype != torch.bfloat16:
            raise ValueError("BF16 staging requires a BF16 [capacity,N] reference")
        if slice_width <= 0 or self.input_width % slice_width:
            raise ValueError("BF16 staging requires complete feature slices")
        self.max_tokens = int(reference.shape[0])
        if self.max_tokens <= 0:
            raise ValueError("BF16 staging capacity must be positive")
        self.slice_width = slice_width
        self.layer = layer
        self.values = torch.empty(
            (self.max_tokens, self.input_width),
            dtype=reference.dtype,
            device=reference.device,
        )
        self._tokens = self._column = 0

    def begin(self, tokens: int) -> None:
        if not 0 < tokens <= self.max_tokens:
            raise ValueError("BF16 staging row count is outside reserved capacity")
        self._tokens, self._column = tokens, 0

    def append(self, source: torch.Tensor) -> None:
        rows, column, width = self._tokens, self._column, self.slice_width
        if (
            rows == 0
            or source.shape != (rows, width)
            or source.dtype != self.values.dtype
            or source.device != self.values.device
            or column + width > self.input_width
        ):
            raise ValueError("BF16 staging slice has incompatible geometry or dtype")
        self.values[:rows, column : column + width].copy_(source)
        self._column += width

    def finish(self) -> torch.Tensor:
        if self._tokens == 0 or self._column != self.input_width:
            raise RuntimeError("BF16 staging input is incomplete")
        rows = self._tokens
        self._tokens = 0
        return self.layer.quant_method.apply(
            self.layer, self.values[:rows], getattr(self.layer, "bias", None)
        )
