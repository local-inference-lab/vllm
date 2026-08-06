# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import torch

importlib.import_module("vllm.model_executor.layers.fused_moe.activation")

from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (  # noqa: E402
    align_trtllm_fp4_moe_hidden_dim_for_fi,
)


def test_align_trtllm_fp4_moe_hidden_dim_noop():
    w13 = torch.arange(2 * 8 * 256, dtype=torch.uint8).reshape(2, 8, 256)
    w13_scale = torch.arange(2 * 8 * 32, dtype=torch.uint8).reshape(2, 8, 32)
    w2 = torch.arange(2 * 512 * 4, dtype=torch.uint8).reshape(2, 512, 4)
    w2_scale = torch.arange(2 * 512 * 1, dtype=torch.uint8).reshape(2, 512, 1)

    out_w13, out_w13_scale, out_w2, out_w2_scale, padded_hidden = (
        align_trtllm_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert padded_hidden == 512
    assert out_w13 is w13
    assert out_w13_scale is w13_scale
    assert out_w2 is w2
    assert out_w2_scale is w2_scale


def test_align_trtllm_fp4_moe_hidden_dim_pads_to_256_multiple():
    hidden_dim = 2688
    padded_hidden_dim = 2816

    w13 = torch.arange(2 * 12 * (hidden_dim // 2), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 2
    )
    w13_scale = torch.arange(2 * 12 * (hidden_dim // 16), dtype=torch.uint8).reshape(
        2, 12, hidden_dim // 16
    )

    w2 = torch.arange(2 * hidden_dim * 6, dtype=torch.uint8).reshape(2, hidden_dim, 6)
    w2_scale = torch.arange(2 * hidden_dim * 2, dtype=torch.uint8).reshape(
        2, hidden_dim, 2
    )

    out_w13, out_w13_scale, out_w2, out_w2_scale, out_hidden_dim = (
        align_trtllm_fp4_moe_hidden_dim_for_fi(w13, w13_scale, w2, w2_scale)
    )

    assert out_hidden_dim == padded_hidden_dim
    assert out_w13.shape == (2, 12, padded_hidden_dim // 2)
    assert out_w13_scale.shape == (2, 12, padded_hidden_dim // 16)
    assert out_w2.shape == (2, padded_hidden_dim, 6)
    assert out_w2_scale.shape == (2, padded_hidden_dim, 2)

    torch.testing.assert_close(out_w13[:, :, : hidden_dim // 2], w13)
    torch.testing.assert_close(out_w13_scale[:, :, : hidden_dim // 16], w13_scale)
    torch.testing.assert_close(out_w2[:, :hidden_dim, :], w2)
    torch.testing.assert_close(out_w2_scale[:, :hidden_dim, :], w2_scale)

    assert torch.count_nonzero(out_w13[:, :, hidden_dim // 2 :]) == 0
    assert torch.count_nonzero(out_w13_scale[:, :, hidden_dim // 16 :]) == 0
    assert torch.count_nonzero(out_w2[:, hidden_dim:, :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, hidden_dim:, :]) == 0


def test_pad_gated_nvfp4_moe_for_swizzled_scales_splits_w13_halves():
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        _pad_gated_nvfp4_moe_for_swizzled_scales,
    )

    intermediate = 352
    padded_intermediate = 384
    w13 = torch.arange(2 * intermediate * 4, dtype=torch.uint8).reshape(
        1, 2 * intermediate, 4
    )
    w13_scale = torch.arange(2 * intermediate * 2, dtype=torch.uint8).reshape(
        1, 2 * intermediate, 2
    )
    w2 = torch.arange(8 * (intermediate // 2), dtype=torch.uint8).reshape(
        1, 8, intermediate // 2
    )
    w2_scale = torch.arange(8 * (intermediate // 16), dtype=torch.uint8).reshape(
        1, 8, intermediate // 16
    )

    out_w13, out_w13_scale, out_w2, out_w2_scale = (
        _pad_gated_nvfp4_moe_for_swizzled_scales(w13, w13_scale, w2, w2_scale)
    )

    assert out_w13.shape == (1, 2 * padded_intermediate, 4)
    assert out_w13_scale.shape == (1, 2 * padded_intermediate, 2)
    assert out_w2.shape == (1, 8, padded_intermediate // 2)
    assert out_w2_scale.shape == (1, 8, padded_intermediate // 16)

    torch.testing.assert_close(out_w13[:, :intermediate], w13[:, :intermediate])
    torch.testing.assert_close(
        out_w13[:, padded_intermediate : padded_intermediate + intermediate],
        w13[:, intermediate:],
    )
    torch.testing.assert_close(
        out_w13_scale[:, :intermediate],
        w13_scale[:, :intermediate],
    )
    torch.testing.assert_close(
        out_w13_scale[:, padded_intermediate : padded_intermediate + intermediate],
        w13_scale[:, intermediate:],
    )
    torch.testing.assert_close(out_w2[:, :, : intermediate // 2], w2)
    torch.testing.assert_close(out_w2_scale[:, :, : intermediate // 16], w2_scale)

    assert torch.count_nonzero(out_w13[:, intermediate:padded_intermediate]) == 0
    assert torch.count_nonzero(out_w13[:, padded_intermediate + intermediate :]) == 0
    assert torch.count_nonzero(
        out_w13_scale[:, intermediate:padded_intermediate]
    ) == 0
    assert (
        torch.count_nonzero(
            out_w13_scale[:, padded_intermediate + intermediate :]
        )
        == 0
    )
    assert torch.count_nonzero(out_w2[:, :, intermediate // 2 :]) == 0
    assert torch.count_nonzero(out_w2_scale[:, :, intermediate // 16 :]) == 0
