# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility names for checkpoints published before Qwen4Exp."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in {
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextForConditionalGeneration",
        "Qwen3_8FlashNextMTP",
    }:
        from vllm.models import qwen4_exp

        return getattr(qwen4_exp, name.replace("Qwen3_8FlashNext", "Qwen4Exp"))
    raise AttributeError(name)
