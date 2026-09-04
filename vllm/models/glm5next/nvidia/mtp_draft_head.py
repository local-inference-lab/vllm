# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantized vocabulary-head copies for GLM-5.3 speculative proposals.

``VLLM_GLM53_MTP_DRAFT_HEAD`` selects the weight used only to produce MTP
draft logits:

* ``bf16`` (default) shares the target model's unquantized head;
* ``nvfp4`` creates an independent block-scaled NVFP4 copy and uses
  FlashInfer's W4A16 CuTe-DSL GEMM.

The target model retains its BF16 vocabulary head.  The verifier therefore
keeps the target distribution unchanged; quantization can only change proposal
acceptance and speculative-decoding efficiency.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

_NVFP4_GLOBAL_MAX = 448.0 * 6.0
_SUPPORTED_MODES = frozenset(("bf16", "nvfp4"))


def configured_draft_head_mode() -> str:
    """Return the validated GLM MTP draft-head storage mode."""
    mode = os.getenv("VLLM_GLM53_MTP_DRAFT_HEAD", "bf16").strip().lower()
    if mode not in _SUPPORTED_MODES:
        choices = ", ".join(sorted(_SUPPORTED_MODES))
        raise ValueError(
            f"VLLM_GLM53_MTP_DRAFT_HEAD must be one of {choices}; got {mode!r}"
        )
    return mode


class _DraftHeadQuantMethod:
    """Adapt a quantized draft head to ``LogitsProcessor``'s linear contract."""

    def apply(
        self,
        layer: QuantizedDraftHead,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("quantized GLM MTP draft heads do not support bias")
        return layer(hidden_states)


class QuantizedDraftHead(nn.Module):
    """Vocab-parallel NVFP4 copy of a BF16 vocabulary-head shard."""

    quant_method = _DraftHeadQuantMethod()

    def __init__(self, source_head: nn.Module, mode: str) -> None:
        super().__init__()
        if mode != "nvfp4":
            raise ValueError(f"quantized draft-head mode must be nvfp4; got {mode!r}")
        weight = getattr(source_head, "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise TypeError("the shared target vocabulary head has no tensor weight")
        if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_cuda:
            raise ValueError(
                "quantized GLM MTP draft heads require a two-dimensional CUDA "
                f"BF16 weight; got shape={tuple(weight.shape)}, dtype={weight.dtype}, "
                f"device={weight.device}"
            )
        if not weight.is_contiguous():
            raise ValueError(
                "the shared target vocabulary-head weight must be contiguous"
            )

        self.tp_size = int(getattr(source_head, "tp_size", 1))
        self.shard_indices: Any = getattr(source_head, "shard_indices", None)
        if self.shard_indices is None:
            raise TypeError("the shared target vocabulary head has no shard metadata")

        major, minor = torch.cuda.get_device_capability(weight.device)
        if (major, minor) != (12, 0):
            raise ValueError(
                "the NVFP4 W4A16 draft vocabulary head requires CUDA capability "
                f"12.0; got {major}.{minor}"
            )
        import flashinfer

        weight_global_scale = (
            _NVFP4_GLOBAL_MAX / weight.float().abs().nan_to_num().max()
        )
        weight_fp4, weight_sf = flashinfer.nvfp4_quantize(
            weight,
            weight_global_scale,
            sfLayout=flashinfer.SfLayout.layout_128x4,
            do_shuffle=False,
            backend="cuda",
        )
        weight_fp4, weight_sf, output_scale = flashinfer.prepare_bf16_fp4_weights(
            weight_fp4.view(torch.uint8),
            weight_sf,
            weight_global_scale.reciprocal().reshape(1),
            backend="cute-dsl",
        )
        self.register_buffer("weight", weight_fp4, persistent=False)
        self.register_buffer("weight_scale", weight_sf, persistent=False)
        self.register_buffer("weight_global_scale", output_scale, persistent=False)

    @property
    def storage_bytes(self) -> int:
        """Return bytes retained by the independent quantized weight copy."""
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.buffers()
            if tensor is not None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, shape[-1])
        import flashinfer

        logits = flashinfer.mm_bf16_fp4(
            hidden,
            self.weight,
            self.weight_scale,
            self.weight_global_scale,
            backend="cute-dsl",
            out_dtype=hidden.dtype,
        )
        return logits.reshape(*shape[:-1], -1)


def make_quantized_draft_head(source_head: nn.Module) -> QuantizedDraftHead | None:
    """Build the configured draft-only head copy after target weights load."""
    mode = configured_draft_head_mode()
    if mode == "bf16":
        return None
    result = QuantizedDraftHead(source_head, mode)
    logger.info(
        "Using a draft-only %s GLM MTP vocabulary head copy (%0.2f MiB per rank); "
        "the target verifier vocabulary head remains BF16.",
        mode.upper(),
        result.storage_bytes / (1024 * 1024),
    )
    return result
