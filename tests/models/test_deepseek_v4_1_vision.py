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


@torch.no_grad()
def test_native_small_vision_block_aligner_and_image_isolation():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x vision requires SM12x")
    from b12x.preparation import PreparationSession
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.models.deepseek_v4_1.nvidia.b12x_vision import (
        DeepseekV4Aligner,
        DeepseekV4ViT,
    )
    from vllm.utils.b12x import B12xWorkload

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
    for module in (vision, aligner):
        for name, parameter in module.named_parameters():
            if "norm" not in name:
                parameter.normal_(std=0.04)
    workload = B12xWorkload(
        stage="weights", token_counts=(1,), fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=32,
        max_seqs=1, max_model_len=1,
    )
    session = PreparationSession(device=torch.device("cuda"), autotune=False, compile_workers=2)
    units = [
        unit
        for provider in (vision, aligner)
        for unit in _units_from_modules(provider, workload)
    ]
    requests = tuple(request for unit in units for request in unit.requests)
    result = session.prepare(requests)
    assert result is not None
    with torch.no_grad():
        warm = torch.randn(7 * 11, 3, 2, 2, device="cuda", dtype=torch.bfloat16)
        patches = torch.randn(4 * 7 + 1, 3, 2, 2, device="cuda", dtype=torch.bfloat16)[1:]
        assert patches.data_ptr() % 16 == 8
        session.freeze()
        aligner(vision(warm, 7, 11), 7, 11)
        encoded = vision(patches, 4, 7)
        actual = aligner(encoded, 4, 7)
        # A different image contaminates every shared capacity buffer. The
        # next call must still match this image's isolated bidirectional oracle.
        vision(warm * 4, 7, 11)
        repeated = vision(patches, 4, 7)
        torch.testing.assert_close(repeated, encoded, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        graph_encoded, graph_out = torch.empty_like(encoded), torch.empty_like(actual)
        with session.capture(), torch.cuda.graph(graph):
            vision(patches, 4, 7, out=graph_encoded)
            aligner(graph_encoded, 4, 7, out=graph_out)
        patches.neg_()
        graph_encoded.fill_(float("nan"))
        graph_out.fill_(float("nan"))
        pointers = (graph_encoded.data_ptr(), graph_out.data_ptr())
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        assert (graph_encoded.data_ptr(), graph_out.data_ptr()) == pointers
        assert torch.isfinite(graph_out).all() and torch.count_nonzero(graph_out) > 0
        expected_encoded, expected_aligned = _reference(vision, aligner, patches, 4, 7)
        torch.testing.assert_close(
            graph_encoded, expected_encoded, atol=0.025, rtol=0.025
        )
        torch.testing.assert_close(graph_out, expected_aligned, atol=0.02, rtol=0.025)
        graph.reset()
    session.close()


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "in_features,out_features,max_rows,offset",
    [(128, 80, 33, 0), (588, 1024, 128, 1)],
    ids=["aligned", "offset_patch"],
)
@torch.no_grad()
def test_vision_linear_prepares_bias_and_replays_into_caller_output(
    bias, in_features, out_features, max_rows, offset,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x vision requires SM12x")
    from b12x.preparation import PreparationSession
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.models.deepseek_v4_1.nvidia.b12x_vision import _Linear
    from vllm.utils.b12x import B12xWorkload

    device = torch.device("cuda", torch.cuda.current_device())
    layer = _Linear(in_features, out_features, bias=bias).to(device)
    layer.weight.normal_(std=0.125)
    if layer.bias is not None:
        layer.bias.normal_(std=0.125)
    source = torch.randn(
        max_rows + offset, in_features, device=device, dtype=torch.bfloat16,
    ).mul_(0.125)[offset:]
    if offset:
        assert source.data_ptr() % 16 == 8
    output = torch.empty(max_rows, out_features, device=device, dtype=torch.bfloat16)
    workload = B12xWorkload(
        stage="weights", token_counts=(1, 8, max_rows), fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16, max_tokens=max_rows, max_seqs=1,
        max_model_len=max_rows,
    )
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        allocated_before = torch.cuda.memory_allocated(device)
        session.prepare(tuple(
            request for unit in _units_from_modules(layer, workload)
            for request in unit.requests
        ))
        assert torch.cuda.memory_allocated(device) == allocated_before
        session.freeze()
        launch = torch.compile(layer, fullgraph=True)
        for rows in (1, 8, 17, 33):
            output.fill_(float("nan"))
            layer(source[:rows], out=output[:rows])
            torch.testing.assert_close(
                output[:rows], _linear(source[:rows], layer), rtol=0.01, atol=0.01,
            )
            assert torch.isnan(output[rows:]).all()
        launch(source, out=output)
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                launch(source, out=output)
            pointer = output.data_ptr()
            source.neg_()
            if layer.bias is not None:
                layer.bias.mul_(-0.5)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert output.data_ptr() == pointer
            assert torch.cuda.memory_allocated(device) == allocated
            assert torch.isfinite(output).all() and torch.count_nonzero(output) > 0
            torch.testing.assert_close(output, _linear(source, layer), rtol=0.01, atol=0.01)
        finally:
            graph.reset()
