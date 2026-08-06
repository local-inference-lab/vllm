# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.attn_res import attn_res

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5


@pytest.mark.parametrize(("num_blocks", "has_delta"), [(0, False), (5, True)])
@torch.inference_mode()
def test_attn_res(num_blocks: int, has_delta: bool) -> None:
    torch.manual_seed(0)
    prefix = torch.randn(3, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(prefix) if has_delta else None
    blocks = torch.randn(
        3, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = 1 + 0.1 * torch.randn_like(prefix[0])
    qk_weight = torch.randn_like(prefix[0]) / HIDDEN_SIZE**0.5
    output_norm_weight = 1 + 0.1 * torch.randn_like(prefix[0])
    original_prefix = prefix.clone()
    original_blocks = blocks.clone()

    expected_prefix = original_prefix if delta is None else original_prefix + delta
    values = torch.cat(
        (blocks[:, :num_blocks], expected_prefix.unsqueeze(1)),
        dim=1,
    )
    keys = F.rms_norm(values, (HIDDEN_SIZE,), norm_weight, EPS)
    probabilities = (keys @ qk_weight).softmax(dim=-1)
    expected = torch.matmul(probabilities.unsqueeze(1), values).squeeze(1)
    expected = F.rms_norm(expected, (HIDDEN_SIZE,), output_norm_weight, EPS)

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        num_blocks,
        EPS,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    original_blocks[:, num_blocks].copy_(expected_prefix)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
