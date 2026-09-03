# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the DeepSeek-V4 vision tower."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.vision import (
    DeepseekV4Aligner,
    DeepseekV4ViT,
)


def _make_config() -> SimpleNamespace:
    return SimpleNamespace(
        vision_n_layers=2,
        vision_dim=64,
        vision_n_heads=4,
        vision_inter_dim=88,
        vision_patch_size=14,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=96,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_vit_preserves_patch_grid_shape(dist_init):
    torch.manual_seed(0)
    config = _make_config()
    vision = DeepseekV4ViT(config).cuda().eval()
    for n_vit_h, n_vit_w in ((6, 6), (7, 5), (4, 10)):
        patches = torch.randn(
            n_vit_h * n_vit_w,
            3,
            config.vision_patch_size,
            config.vision_patch_size,
            device="cuda",
        )
        output = vision(patches, n_vit_h, n_vit_w)
        assert output.shape == (n_vit_h * n_vit_w, config.vision_dim)
        assert torch.isfinite(output).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_aligner_downsamples_to_expected_grid(dist_init):
    torch.manual_seed(0)
    config = _make_config()
    aligner = DeepseekV4Aligner(config).cuda().eval()
    r = config.vision_downsample_ratio
    for n_vit_h, n_vit_w in ((6, 6), (7, 5), (4, 10)):
        n_llm_h = -(-n_vit_h // r)
        n_llm_w = -(-n_vit_w // r)
        x = torch.randn(n_vit_h * n_vit_w, config.vision_dim, device="cuda")
        output = aligner(x, n_vit_h, n_vit_w)
        assert output.shape == (n_llm_h * n_llm_w, config.hidden_size)
        assert torch.isfinite(output).all()
