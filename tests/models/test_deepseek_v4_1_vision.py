# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small native vision tower oracle; no full checkpoint or distributed fixture."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _linear(x, layer):
    bias = None if layer.bias is None else layer.bias.float()
    return F.linear(x.float(), layer.weight.float(), bias).bfloat16()


def _norm(x, layer):
    value = x.float()
    return (
        value
        * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
        * layer.weight.float()
    ).bfloat16()


def _reference(vision, aligner, patches, height, width):
    config = vision.config
    heads, dim = config.vision_n_heads, config.vision_dim // config.vision_n_heads
    inv = 1.0 / (
        config.vision_rope_theta
        ** (torch.arange(0, dim // 2, 2, device=patches.device).float() / (dim // 2))
    )
    hp = torch.arange(height, device=patches.device)[:, None].expand(height, width)
    wp = torch.arange(width, device=patches.device)[None, :].expand(height, width)
    angles = (torch.stack((hp, wp), -1).reshape(-1, 2, 1).float() * inv).flatten(1)[
        :, None
    ]

    def rotate(x):
        first, second = x.float().chunk(2, -1)
        return torch.cat(
            (
                first * angles.cos() - second * angles.sin(),
                second * angles.cos() + first * angles.sin(),
            ),
            -1,
        ).bfloat16()

    x = _linear(patches.flatten(1), vision.patch_embed.proj)
    for block in vision.blocks:
        qkv = _linear(_norm(x, block.norm1), block.attn.wqkv)
        q, k, v = [t.reshape(-1, heads, dim) for t in qkv.chunk(3, -1)]
        q, k = rotate(q), rotate(k)
        scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * dim**-0.5
        attention = torch.einsum(
            "hts,shd->thd", scores.softmax(-1), v.float()
        ).bfloat16()
        x = x + _linear(attention.reshape(-1, config.vision_dim), block.attn.wo)
        gate, up = _linear(_norm(x, block.norm2), block.mlp.w1).chunk(2, -1)
        x = x + _linear(F.silu(gate) * up, block.mlp.w2)
    x = _norm(x, vision.norm)
    ratio = aligner.downsample_ratio
    image = x.reshape(height, width, -1).permute(2, 0, 1)
    image = F.pad(image, (0, -width % ratio, 0, -height % ratio))
    merged = F.unfold(image[None].float(), ratio, stride=ratio)[0].T.bfloat16()
    aligned = _linear(
        F.gelu(_linear(merged, aligner.w1), approximate="none"), aligner.w2
    )
    return x, aligned


def test_native_small_vision_block_aligner_and_image_isolation():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x vision requires SM12x")
    from flashinfer.b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )

    from vllm.models.deepseek_v4_1.nvidia.b12x_vision import (
        DeepseekV4Aligner,
        DeepseekV4ViT,
        warmup_vision_tower,
    )

    config = SimpleNamespace(
        vision_patch_size=2,
        vision_dim=128,
        vision_n_heads=2,
        vision_inter_dim=192,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=3,
        hidden_size=80,
        vision_max_n_token=16,
    )
    torch.manual_seed(411)
    vision, aligner = DeepseekV4ViT(config).cuda(), DeepseekV4Aligner(config).cuda()
    with torch.no_grad():
        for module in (vision, aligner):
            for name, parameter in module.named_parameters():
                parameter.normal_(0.0, 0.04)
                if "norm" in name:
                    parameter.add_(1.0)
        # The post-load single-patch warmup must cover larger live images too.
        warmup_vision_tower(vision, aligner)
        warm = torch.randn(7 * 11, 3, 2, 2, device="cuda", dtype=torch.bfloat16)
        patches = torch.randn(4 * 7, 3, 2, 2, device="cuda", dtype=torch.bfloat16)
        freeze_kernel_resolution("small vision images share fixed capacity")
        try:
            aligner(vision(warm, 7, 11), 7, 11)
            encoded = vision(patches, 4, 7)
            actual = aligner(encoded, 4, 7)
            # A different image contaminates every shared capacity buffer. The
            # next call must still match this image's isolated bidirectional oracle.
            vision(warm * 4, 7, 11)
            repeated = vision(patches, 4, 7)
            torch.testing.assert_close(repeated, encoded, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            graph_encoded, graph_out = (
                torch.empty_like(encoded),
                torch.empty_like(actual),
            )
            with torch.cuda.graph(graph):
                vision(patches, 4, 7, out=graph_encoded)
                aligner(graph_encoded, 4, 7, out=graph_out)
            patches.neg_()
            graph.replay()
        finally:
            unfreeze_kernel_resolution()
        expected_encoded, expected_aligned = _reference(vision, aligner, patches, 4, 7)
        torch.testing.assert_close(
            graph_encoded, expected_encoded, atol=0.025, rtol=0.025
        )
        torch.testing.assert_close(graph_out, expected_aligned, atol=0.02, rtol=0.025)
