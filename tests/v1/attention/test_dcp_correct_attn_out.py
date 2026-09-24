# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""correct_attn_out for every DCP world size, including non-powers of two."""

import pytest
import torch

from vllm.v1.attention.ops.dcp import CPTritonContext, correct_attn_out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("world_size", [2, 3, 4, 6])
@pytest.mark.parametrize("is_base_e", [True, False])
def test_correct_attn_out_matches_reference(world_size, is_base_e):
    torch.manual_seed(world_size)
    tokens, heads, head_dim = 5, 4, 64
    out = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    lses = torch.randn(world_size, tokens, heads, device="cuda") * 4
    # A rank with no keys for a row contributes -inf.
    lses[world_size - 1, 0] = -float("inf")
    rank = world_size - 1

    expected_out = out.float().clone()
    scale = 1.0 if is_base_e else torch.log(torch.tensor(2.0)).item()
    lse = torch.logsumexp(lses * scale, dim=0) / scale
    factor = torch.exp((lses[rank] - lse) * scale)
    expected_out *= factor[..., None]

    corrected, final_lse = correct_attn_out(
        out.clone(), lses, rank, CPTritonContext(), is_lse_base_on_e=is_base_e
    )

    torch.testing.assert_close(final_lse, lse, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(corrected.float(), expected_out, rtol=2e-2, atol=2e-2)
