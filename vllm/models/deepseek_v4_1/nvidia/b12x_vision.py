# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replicated V4.1 ViT and aligner using only native b12x compute.

One image is evaluated per call. Attention binds full planned-capacity tensors
and a device [0, live_patches] prefix, so padded/stale rows cannot participate.
Call ``prepare`` and warm up before capture; forward accepts caller-owned output.
Workspace is shared across blocks, not images concurrently on different streams.
"""

from __future__ import annotations

import itertools

import torch
from b12x.attention import varlen
from b12x.gemm import bf16_gemv
from b12x.norm import hyperconnection
from b12x.norm.vision import run_gelu, run_rope_qkv, run_spatial_merge
from torch import nn


def _capacity(config):
    capacity = getattr(config, "vision_max_tokens", None)
    if capacity is None:
        capacity = config.vision_max_n_token * config.vision_downsample_ratio**2
    if capacity <= 0:
        raise ValueError("vision token capacity must be positive")
    return int(capacity)


class _Linear(nn.Module):
    """Replicated checkpoint layout; bias is accumulated before BF16 rounding."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16), requires_grad=False
            )
            if bias
            else None
        )

    def forward(self, x, *, out):
        return bf16_gemv.mm(
            x, self.weight, bias=self.bias, out=out, output_dtype=torch.bfloat16
        )

    def prepare(self):
        bf16_gemv.precompile(
            self.weight,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            bias=self.bias,
        )


class _Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=torch.bfloat16), requires_grad=False
        )

    def forward(self, x, *, out, plan):
        binding = hyperconnection.bind(
            plan,
            normalized=out,
            bottleneck=x.view(-1)[: x.shape[0]].view(-1, 1),
            block_input=x,
            tokens=x.shape[0],
        )
        return hyperconnection.run_grouped_rmsnorm(
            x, self.weight, eps=1e-6, zero_centered=False, binding=binding
        )


class _PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = _Linear(3 * config.vision_patch_size**2, config.vision_dim)

    def forward(self, patches, *, out):
        return self.proj(patches.flatten(1), out=out)


class _Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wqkv = _Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = _Linear(config.vision_dim, config.vision_dim)

    def forward(self, x, height, width, workspace, *, out):
        rows = x.shape[0]
        qkv = self.wqkv(x, out=workspace.qkv[:rows])
        run_rope_qkv(
            qkv,
            height,
            width,
            workspace.inv_freq,
            q=workspace.q,
            k=workspace.k,
            v=workspace.v,
            cu_seqlens=workspace.cu,
        )
        attended, _ = varlen.run(binding=workspace.attention_binding)
        return self.wo(attended[:rows].reshape(rows, -1), out=out)


class _MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = _Linear(config.vision_dim, 2 * config.vision_inter_dim, False)
        self.w2 = _Linear(config.vision_inter_dim, config.vision_dim, False)

    def forward(self, x, workspace, *, out):
        rows = x.shape[0]
        gate_up = self.w1(x, out=workspace.gate_up[:rows])
        activated = hyperconnection.run_swiglu(
            gate_up, limit=float("inf"), round_silu=True, out=workspace.activated[:rows]
        )
        return self.w2(activated, out=out)


class _Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1, self.norm2 = _Norm(config.vision_dim), _Norm(config.vision_dim)
        self.attn, self.mlp = _Attention(config), _MLP(config)

    def forward(self, x, height, width, workspace):
        rows = x.shape[0]
        norm, branch = workspace.normalized[:rows], workspace.branch[:rows]
        self.norm1(x, out=norm, plan=workspace.norm_plan)
        self.attn(norm, height, width, workspace, out=branch)
        residual = hyperconnection.run_add(x, branch, out=workspace.residual[:rows])
        self.norm2(residual, out=norm, plan=workspace.norm_plan)
        self.mlp(norm, workspace, out=branch)
        return hyperconnection.run_add(residual, branch, out=x)


class _VisionWorkspace:
    def __init__(self, config, capacity, device):
        hidden, inter = config.vision_dim, config.vision_inter_dim
        heads = config.vision_n_heads
        dim = hidden // heads

        def empty(*shape):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

        self.state, self.normalized, self.branch = (
            empty(capacity, hidden) for _ in range(3)
        )
        self.residual = empty(capacity, hidden)
        self.qkv = empty(capacity, 3 * hidden)
        self.q, self.k, self.v = (empty(capacity, heads, dim) for _ in range(3))
        self.gate_up, self.activated = (
            empty(capacity, 2 * inter),
            empty(capacity, inter),
        )
        # This is static head geometry, not a table keyed by live image grids.
        inv = 1.0 / (
            config.vision_rope_theta
            ** (
                torch.arange(0, dim // 2, 2, dtype=torch.float32, device="cpu")
                / (dim // 2)
            )
        )
        self.inv_freq = inv.to(device=device)
        self.cu = torch.zeros(2, dtype=torch.int32, device=device)
        kernel_plan = varlen.create_plan(
            self.q,
            self.k,
            self.v,
            self.cu,
            max_seqlen_q=capacity,
            max_seqlen_k=capacity,
            causal=False,
        )
        scratch_plan = varlen.plan(kernel_plan)
        (spec,) = scratch_plan.scratch_specs()
        self.scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        self.attention_binding = scratch_plan.bind(
            scratch=self.scratch,
            q=self.q,
            k=self.k,
            v=self.v,
            cu_seqlens_q=self.cu,
            max_seqlen_q=capacity,
            max_seqlen_k=capacity,
            causal=False,
        )
        self.norm_plan = hyperconnection.plan(
            hyperconnection.Caps(
                device=device,
                max_tokens=capacity,
                hidden_size=hidden,
                streams=1,
                lowrank=1,
            )
        )


class DeepseekV4ViT(nn.Module):
    """Reference 2D split-half RoPE and full bidirectional attention per image."""

    def __init__(self, config):
        super().__init__()
        if (
            config.vision_dim % config.vision_n_heads
            or (config.vision_dim // config.vision_n_heads) % 4
        ):
            raise ValueError("vision head dimension must be divisible by four")
        self.config, self.capacity = config, _capacity(config)
        self.rope_dim = config.vision_dim // config.vision_n_heads // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = _PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [_Block(config) for _ in range(config.vision_n_layers)]
        )
        self.norm = _Norm(config.vision_dim)
        self._workspace = None

    def prepare(self):
        """Allocate planned storage and resolve projections/attention outside capture."""
        device = self.patch_embed.proj.weight.device
        if device.type != "cuda":
            raise ValueError("native vision requires CUDA weights")
        if self._workspace is not None:
            if self._workspace.state.device != device:
                raise ValueError("vision moved devices after preparation")
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare and warm vision before CUDA graph capture")
        self._workspace = _VisionWorkspace(self.config, self.capacity, device)
        for module in self.modules():
            if isinstance(module, _Linear):
                module.prepare()

    def forward(self, patches, n_vit_h: int, n_vit_w: int, *, out=None):
        rows = n_vit_h * n_vit_w
        if (
            n_vit_h <= 0
            or n_vit_w <= 0
            or rows != patches.shape[0]
            or rows > self.capacity
        ):
            raise ValueError(
                "vision image grid exceeds capacity or does not match patches"
            )
        if patches.dtype != torch.bfloat16 or not patches.is_contiguous():
            raise ValueError("vision patches must be contiguous BF16")
        self.prepare()
        ws = self._workspace
        if out is None:
            out = torch.empty(
                (rows, self.config.vision_dim),
                dtype=torch.bfloat16,
                device=patches.device,
            )
        x = self.patch_embed(patches, out=ws.state[:rows])
        for block in self.blocks:
            x = block(x, n_vit_h, n_vit_w, ws)
        return self.norm(x, out=out, plan=ws.norm_plan)


class DeepseekV4Aligner(nn.Module):
    """Channel-major padded spatial merge, biased projections, exact GELU."""

    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        if self.downsample_ratio <= 0:
            raise ValueError("vision downsample ratio must be positive")
        self.out_dim = config.hidden_size
        self.hidden_size, self.capacity = config.vision_dim, _capacity(config)
        self.w1 = _Linear(config.vision_dim * self.downsample_ratio**2, self.out_dim)
        self.w2 = _Linear(self.out_dim, self.out_dim)
        self._workspace = None

    def prepare(self):
        device = self.w1.weight.device
        if device.type != "cuda":
            raise ValueError("native aligner requires CUDA weights")
        if self._workspace is not None:
            if self._workspace[0].device != device:
                raise ValueError("aligner moved devices after preparation")
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare and warm aligner before CUDA graph capture")
        r = self.downsample_ratio
        capacity = (self.capacity + r - 1) // r
        self._workspace = tuple(
            torch.empty((capacity, width), device=device, dtype=torch.bfloat16)
            for width in (self.hidden_size * r * r, self.out_dim, self.out_dim)
        )
        self.w1.prepare()
        self.w2.prepare()

    def forward(self, x, n_vit_h: int, n_vit_w: int, *, out=None):
        if (
            n_vit_h <= 0
            or n_vit_w <= 0
            or x.shape != (n_vit_h * n_vit_w, self.hidden_size)
            or x.shape[0] > self.capacity
        ):
            raise ValueError(
                "aligner image grid exceeds capacity or does not match input"
            )
        self.prepare()
        r = self.downsample_ratio
        rows = ((n_vit_h + r - 1) // r) * ((n_vit_w + r - 1) // r)
        merged, hidden, activated = (t[:rows] for t in self._workspace)
        run_spatial_merge(x, n_vit_h, n_vit_w, ratio=r, out=merged)
        self.w1(merged, out=hidden)
        run_gelu(hidden, out=activated)
        if out is None:
            out = torch.empty(
                (rows, self.out_dim), dtype=torch.bfloat16, device=x.device
            )
        return self.w2(activated, out=out)


def warmup_vision_tower(vision_model: DeepseekV4ViT, aligner: DeepseekV4Aligner):
    """Resolve every native tower specialization in the root post-load hook.

    A single patch is enough: row counts and the image grid are runtime scalars,
    while attention always binds the complete planned-capacity workspace. Run
    every block so distinct checkpoint dtypes and projection geometries are
    covered, then the aligner to cover merge, biased GEMV, and erf-GELU.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("vision warmup must run after loading and before capture")
    vision_model.prepare()
    aligner.prepare()
    if getattr(vision_model, "_warmed_up", False) and getattr(
        aligner, "_warmed_up", False
    ):
        return
    patch_size = vision_model.config.vision_patch_size
    device = vision_model.patch_embed.proj.weight.device
    with torch.no_grad():
        patches = torch.zeros(
            (1, 3, patch_size, patch_size), dtype=torch.bfloat16, device=device
        )
        encoded = torch.empty(
            (1, vision_model.config.vision_dim), dtype=torch.bfloat16, device=device
        )
        aligned = torch.empty((1, aligner.out_dim), dtype=torch.bfloat16, device=device)
        vision_model(patches, 1, 1, out=encoded)
        aligner(encoded, 1, 1, out=aligned)
    vision_model._warmed_up = True
    aligner._warmed_up = True


def run_dp_sharded_vision_tower(vision_model, aligner, patches, vit_grid):
    """Shard images across replicated TP ranks and perform a real all-gather."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.model_executor.models.vision import get_load_balance_assignment

    if not vit_grid:
        if patches.shape[0] != 0:
            raise ValueError("nonempty patches require image grids")
        return []
    sizes = [h * w for h, w in vit_grid]
    if any(h <= 0 or w <= 0 for h, w in vit_grid) or sum(sizes) != patches.shape[0]:
        raise ValueError("image grids do not partition patches")
    world, rank = (
        get_tensor_model_parallel_world_size(),
        get_tensor_model_parallel_rank(),
    )
    offsets = [0, *itertools.accumulate(sizes)]
    assignments, counts, _ = get_load_balance_assignment(sizes, world)
    rank_offsets = [0, *itertools.accumulate(counts)]
    r = aligner.downsample_ratio
    output_rows = [((h + r - 1) // r) * ((w + r - 1) // r) for h, w in vit_grid]

    def images(g):
        return assignments[rank_offsets[g] : rank_offsets[g + 1]]

    max_rows = max(sum(output_rows[i] for i in images(g)) for g in range(world))
    local = torch.zeros(
        (max_rows, aligner.out_dim), dtype=patches.dtype, device=patches.device
    )
    offset = 0
    for i in images(rank):
        h, w = vit_grid[i]
        encoded = vision_model(patches[offsets[i] : offsets[i + 1]], h, w)
        aligner(encoded, h, w, out=local[offset : offset + output_rows[i]])
        offset += output_rows[i]
    gathered = tensor_model_parallel_all_gather(local, dim=0) if world > 1 else local
    result = [None] * len(vit_grid)
    for g in range(world):
        offset = g * max_rows
        for i in images(g):
            result[i] = gathered[offset : offset + output_rows[i]]
            offset += output_rows[i]
    return result
