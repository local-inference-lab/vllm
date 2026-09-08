# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 FA4 varlen forward: over-provisioned grid tiles must not touch memory.

The varlen tile scheduler sizes the grid from ``total_q + num_batch *
(tile_m - 1)`` rows, so a batch whose sequence lengths do not fill whole
query tiles (a padded zero-length request, or two requests with tile
remainders) gets extra CTAs that map to batch index ``num_batch``. The SM80
and SM120 kernels used to derive those CTAs' sequence lengths from
``cu_seqlens[num_batch + 1]``, one element past the tensor, and then loaded
K/V and stored O for the garbage length. The test plants a large value right
after ``cu_seqlens`` and a canary region right after ``out``: a kernel that
runs the extra tiles overwrites the canary (and faults when the garbage
offset leaves mapped memory); a kernel that skips them leaves it intact and
matches a per-sequence reference.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or current_platform.get_device_capability() is None
    or current_platform.get_device_capability()[0] != 12,
    reason="SM120 FA4 varlen kernel test",
)

NUM_HEADS = 11
QK_HEAD_DIM = 192
V_HEAD_DIM = 128
CANARY_ROWS = 4096
TILE_M = 128


def _reference(q, k, v, lens):
    outs = []
    start = 0
    for length in lens:
        if length == 0:
            continue
        qs = q[start : start + length].transpose(0, 1).float()
        ks = k[start : start + length].transpose(0, 1).float()
        vs = v[start : start + length].transpose(0, 1).float()
        out = torch.nn.functional.scaled_dot_product_attention(
            qs, ks, vs, is_causal=True, scale=QK_HEAD_DIM**-0.5
        )
        outs.append(out.transpose(0, 1))
        start += length
    return torch.cat(outs, dim=0)


def _wasted_tiles(lens):
    total = sum(lens)
    provisioned = (total + len(lens) * (TILE_M - 1)) // TILE_M
    used = sum((length + TILE_M - 1) // TILE_M for length in lens)
    return provisioned - used


@pytest.mark.parametrize(
    "lens",
    [
        [3615, 0],  # one request plus a zero-length padded request
        [1055, 744],  # chunked-context tail next to a fresh request
        [1057, 111],
        [744, 1055],
    ],
)
def test_sm120_varlen_extra_tiles_do_not_touch_memory(lens):
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    assert _wasted_tiles(lens) >= 1, "batch must over-provision the grid"
    torch.manual_seed(0)
    device = torch.device("cuda")
    total = sum(lens)
    dtype = torch.bfloat16
    q = torch.randn(total, NUM_HEADS, QK_HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(total, NUM_HEADS, QK_HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(total, NUM_HEADS, V_HEAD_DIM, dtype=dtype, device=device)

    # cu_seqlens is a prefix view of a buffer whose next element is huge, the
    # worst case for a kernel that reads one past the end.
    cu_buffer = torch.full(
        (len(lens) + 2,), 1 << 30, dtype=torch.int32, device=device
    )
    cu_buffer[0] = 0
    cu_buffer[1 : len(lens) + 1] = torch.cumsum(
        torch.tensor(lens, dtype=torch.int32), dim=0
    ).to(device)
    cu_seqlens = cu_buffer[: len(lens) + 1]

    canary = torch.full(
        (total + CANARY_ROWS, NUM_HEADS, V_HEAD_DIM),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    canary[:total].zero_()
    out_view = canary[:total]

    result = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=max(lens),
        cu_seqlens_q=cu_seqlens,
        max_seqlen_k=max(lens),
        cu_seqlens_k=cu_seqlens,
        causal=True,
        softmax_scale=QK_HEAD_DIM**-0.5,
        out=out_view,
        num_splits=1,
        fa_version=4,
    )
    torch.accelerator.synchronize()
    out = result[0] if isinstance(result, tuple) else result

    assert torch.isnan(canary[total:].float()).all(), (
        "extra grid tiles wrote past the end of the output"
    )
    assert cu_buffer[len(lens) + 1].item() == 1 << 30
    reference = _reference(q, k, v, lens)
    torch.testing.assert_close(out.float(), reference, atol=2e-2, rtol=2e-2)
