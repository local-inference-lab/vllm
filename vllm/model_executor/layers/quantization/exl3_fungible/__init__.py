# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible quant: runtime per-expert bit-width reallocation for EXL3 MoE.

Components (research/fungible-quant/implementation in the research branch):
stats collector (M1), policy engine + store (M2), swap engine (M4).
"""

from vllm.model_executor.layers.quantization.exl3_fungible.stats import (
    FqStatsCollector,
)

__all__ = ["FqStatsCollector"]
