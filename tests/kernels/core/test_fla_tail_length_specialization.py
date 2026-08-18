# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops.kda import (
    layer_norm_gated_fwd_kernel,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import (
    l2norm_fwd_kernel2,
)


def _specialization(kernel, *args):
    _, _, _, _, binder = kernel.create_binder()
    return binder(*args)[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_l2norm_tail_length_reuses_aligned_specialization() -> None:
    """A non-aligned prefill tail must reuse the startup-warmed kernel."""
    x = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    y = torch.empty_like(x)

    aligned = _specialization(l2norm_fwd_kernel2, x, y, 1e-6, 1536, 128, 128, 32)
    tail = _specialization(l2norm_fwd_kernel2, x, y, 1e-6, 1177, 128, 128, 32)

    assert tail == aligned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kda_gated_norm_tail_length_reuses_aligned_specialization() -> None:
    """A non-aligned prefill tail must reuse the startup-warmed kernel."""
    x = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    rstd = torch.empty(1, dtype=torch.float32, device="cuda")
    kernel = cast(Any, layer_norm_gated_fwd_kernel).fn

    def bind(length: int):
        return _specialization(
            kernel,
            x,
            x,
            x,
            x,
            None,
            None,
            None,
            None,
            rstd,
            1e-5,
            length,
            1,
            128,
            128,
            16,
            128,
            "swish",
            True,
            False,
            False,
            True,
            False,
        )

    assert bind(1177) == bind(1536)
