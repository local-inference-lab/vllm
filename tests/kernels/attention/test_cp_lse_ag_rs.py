# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
from torch.testing import assert_close

from vllm.platforms import current_platform
from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


@pytest.mark.parametrize("is_lse_base_on_e", [True, False])
def test_correct_attn_out_non_power_of_two_dcp_size(is_lse_base_on_e: bool):
    device = "cuda"
    B, H, D, N = 3, 4, 64, 10
    cp_rank = 3

    generator = torch.Generator(device=device).manual_seed(11)
    out = torch.randn((B, H, D), device=device, generator=generator)
    lses = torch.randn((N, B, H), device=device, generator=generator)

    if is_lse_base_on_e:
        expected_lse = torch.logsumexp(lses, dim=0)
        expected_out = out * torch.exp(lses[cp_rank] - expected_lse).unsqueeze(-1)
    else:
        expected_lse = torch.logsumexp(lses * math.log(2), dim=0) / math.log(2)
        expected_out = out * torch.exp2(lses[cp_rank] - expected_lse).unsqueeze(-1)

    actual_out, actual_lse = correct_attn_out(
        out.clone(),
        lses,
        cp_rank,
        CPTritonContext(),
        is_lse_base_on_e=is_lse_base_on_e,
    )

    assert_close(actual_lse, expected_lse, rtol=1e-5, atol=1e-5)
    assert_close(actual_out, expected_out, rtol=1e-5, atol=1e-5)
