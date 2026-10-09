# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="GDN Triton/FLA kernels require a CUDA-alike device.",
)


def _copy_with_poisoned_tail(value: torch.Tensor, tail_value: float) -> torch.Tensor:
    padded_seq_len = math.ceil(value.shape[1] / 64) * 64
    tail_elements = math.prod(value.shape[2:]) * (padded_seq_len - value.shape[1])
    storage = torch.empty(
        value.numel() + tail_elements,
        device=value.device,
        dtype=value.dtype,
    )
    result = storage[: value.numel()].view(value.shape)
    result.copy_(value)
    storage[value.numel() :].fill_(tail_value)
    assert result.is_contiguous()
    return result


@pytest.mark.parametrize("seq_len", [1, 2, 3, 63, 64, 65, 4096])
def test_prefill_ignores_chunk_tail(seq_len: int) -> None:
    device = torch.device(current_platform.device_type)
    generator = torch.Generator(device=device).manual_seed(1234 + seq_len)

    num_k_heads = 4
    num_v_heads = 12
    head_k_dim = 128
    head_v_dim = 128
    shape_qk = (1, seq_len, num_k_heads, head_k_dim)
    shape_v = (1, seq_len, num_v_heads, head_v_dim)

    q = F.normalize(
        torch.randn(shape_qk, device=device, dtype=torch.float32, generator=generator),
        dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn(shape_qk, device=device, dtype=torch.float32, generator=generator),
        dim=-1,
    ).to(torch.bfloat16)
    v = (
        torch.randn(shape_v, device=device, dtype=torch.float32, generator=generator)
        * 0.1
    ).to(torch.bfloat16)
    g = -torch.rand(
        (1, seq_len, num_v_heads),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    beta = torch.sigmoid(
        torch.randn(
            (1, seq_len, num_v_heads),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    )
    history = (
        torch.randn(
            (1, num_v_heads, head_v_dim, head_k_dim),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        * 0.05
    ).to(torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)

    for initial_state in (torch.zeros_like(history), history):
        first_chunk_output = None
        first_chunk_state = None
        for tail_value in (0.0, float("nan")):
            chunk_output, chunk_state = chunk_gated_delta_rule(
                q=_copy_with_poisoned_tail(q, tail_value),
                k=_copy_with_poisoned_tail(k, tail_value),
                v=_copy_with_poisoned_tail(v, tail_value),
                g=_copy_with_poisoned_tail(g, tail_value),
                beta=_copy_with_poisoned_tail(beta, tail_value),
                initial_state=initial_state.clone(),
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
            recurrent_output, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state.clone(),
                inplace_final_state=False,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
            )
            torch.accelerator.synchronize()

            assert torch.isfinite(chunk_output).all()
            assert torch.isfinite(chunk_state).all()
            assert torch.isfinite(recurrent_output).all()
            assert torch.isfinite(recurrent_state).all()
            torch.testing.assert_close(
                chunk_output, recurrent_output, rtol=1e-2, atol=5e-4
            )
            torch.testing.assert_close(
                chunk_state.float(),
                recurrent_state[-1:].float(),
                rtol=1e-2,
                atol=6e-3,
            )

            if first_chunk_output is None:
                first_chunk_output = chunk_output
                first_chunk_state = chunk_state
            else:
                torch.testing.assert_close(
                    chunk_output, first_chunk_output, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    chunk_state, first_chunk_state, rtol=0, atol=0
                )
