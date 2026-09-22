# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Vision tower implementation for Kimi-K2.5 model.

This module provides the vision encoder components for Kimi-K2.5,
including 3D patch embedding, RoPE position embedding, and
temporal pooling for video chunks.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import GELUActivation

from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.models.vision import (
    is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig
from vllm.utils.torch_utils import async_tensor_h2d

logger = init_logger(__name__)


def _vision_projection_width(layer) -> int:
    """Return the physical output width for a caller-storage-capable projection."""
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )

    kernel = getattr(getattr(layer, "quant_method", None), "kernel", None)
    if (
        not isinstance(kernel, MarlinMxfp8LinearKernel)
        or not layer.disable_tp
        or layer.skip_bias_add
    ):
        return 0

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_repacked_nk,
    )

    return marlin_repacked_nk(layer.weight, num_bits=8)[0]


def _project_vision_intermediate(layer, x, workspace):
    """Use caller-owned storage until the next projection in this encoder block."""
    if workspace is None or torch.is_grad_enabled():
        return layer(x)[0]

    rows = x.numel() // x.shape[-1]
    padded_n = _vision_projection_width(layer)
    if padded_n == 0 or workspace.numel() < rows * padded_n:
        return layer(x)[0]
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        apply_mxfp8_marlin_linear,
    )

    output = workspace[: rows * padded_n].view(rows, padded_n)
    return apply_mxfp8_marlin_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=layer.output_size_per_partition,
        size_k=layer.input_size_per_partition,
        bias=layer.bias,
        output=output,
    )


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope_into_packed_qk(xq, xk, freqs_cis, *, workspace=None):
    """Rotate consumed packed Q/K views with bounded FP32 conversion storage.

    Query and key must be non-overlapping views whose unrotated contents have
    no remaining consumer. Complex multiplication preserves the checkpoint's
    FP32 rotary arithmetic before writing back into the activation dtype.
    An optional contiguous FP32 workspace is exclusively owned until return.
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)
    if workspace is not None:
        if (
            workspace.dtype != torch.float32
            or workspace.device != xq.device
            or not workspace.is_contiguous()
        ):
            raise ValueError(
                "RoPE workspace must be contiguous FP32 on the input device"
            )
        workspace = workspace.view(-1)
    frequencies = freqs_cis.unsqueeze(-2)
    if xq.ndim == 2:
        frequencies = frequencies.unsqueeze(0)
    for value in (xq, xk):
        view = value.unsqueeze(0) if value.ndim == 2 else value
        rows = view.shape[-3]
        if rows == 0:
            continue
        row_bytes = (view.numel() // rows) * 4
        capacity = 64 * 1024 * 1024 if workspace is None else workspace.numel() * 4
        if workspace is not None and capacity < row_bytes:
            raise ValueError("RoPE workspace must hold at least one sequence row")
        rows_per_chunk = max(1, capacity // max(1, row_bytes))
        for start in range(0, rows, rows_per_chunk):
            count = min(rows_per_chunk, rows - start)
            chunk = view.narrow(-3, start, count)
            if workspace is None:
                float_chunk = chunk.float()
            else:
                float_chunk = workspace[: chunk.numel()].view(chunk.shape)
                float_chunk.copy_(chunk)
            converted = torch.view_as_complex(
                float_chunk.view(*chunk.shape[:-1], -1, 2)
            )
            converted.mul_(frequencies.narrow(-3, start, count))
            chunk.copy_(torch.view_as_real(converted).flatten(-2))
            del converted, float_chunk
    return xq, xk


def get_rope_shape_decorate(func):
    _get_rope_shape_first_call_flag = set()

    def wrapper(org, interpolation_mode, shape):
        key = (org.requires_grad, torch.is_grad_enabled(), interpolation_mode)
        if key not in _get_rope_shape_first_call_flag:
            _get_rope_shape_first_call_flag.add(key)
            _ = func(org, interpolation_mode, shape=(64, 64))
        return func(org, interpolation_mode, shape)

    return wrapper


@get_rope_shape_decorate
@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    disable=current_platform.simple_compile_backend == "tpu",
)
def get_rope_shape(org, interpolation_mode, shape):
    return (
        F.interpolate(
            org.permute((2, 0, 1)).unsqueeze(0),
            size=shape,
            mode=interpolation_mode,
        )
        .squeeze(0)
        .permute((1, 2, 0))
        .flatten(end_dim=1)
    )


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sincos positional embedding from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False):
    """Generate 1D sincos positional embedding."""
    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_t)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class Learnable2DInterpPosEmbDivided_fixed(nn.Module):
    """2D learnable position embedding with temporal extension."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = "bicubic",
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode
        self.weight = nn.Parameter(torch.empty(height, width, dim))
        self.register_buffer(
            "time_weight",
            torch.from_numpy(get_1d_sincos_pos_embed(self.dim, self.num_frames))
            .float()
            .unsqueeze(1),
            persistent=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def get_pos_embeds(self, grid_thws: torch.Tensor | list[list[int]]) -> torch.Tensor:
        pos_embs = []
        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        for t, h, w in grid_thw_list:
            assert t <= self.num_frames, f"t:{t} > self.num_frames:{self.num_frames}"
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = get_rope_shape(
                    self.weight,
                    interpolation_mode=self.interpolation_mode,
                    shape=(h, w),
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = (
                    pos_emb_2d.unsqueeze(0).repeat(t, 1, 1) + self.time_weight[0:t]
                )

            pos_embs.append(pos_emb_3d.reshape(-1, pos_emb_3d.shape[-1]))

        return torch.cat(pos_embs)

    def forward(
        self, x: torch.Tensor, grid_thws: torch.Tensor | list[list[int]]
    ) -> torch.Tensor:
        return x + self.get_pos_embeds(grid_thws)


class MoonVision3dPatchEmbed(nn.Module):
    """3D patch embedding for vision tower."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | tuple[int, int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        patch_embed_proj_bias: bool = True,
        pos_emb_interpolation_mode: str = "bicubic",
    ):
        super().__init__()
        assert isinstance(patch_size, int | Sequence), (
            f"Invalid patch_size type: {type(patch_size)}"
        )
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, (
            f"Expected patch_size to be a tuple of 2, got {patch_size}"
        )
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=patch_embed_proj_bias,
        )

        if pos_emb_type == "divided_fixed":
            self.pos_emb = Learnable2DInterpPosEmbDivided_fixed(
                height=pos_emb_height,
                width=pos_emb_width,
                num_frames=pos_emb_time,
                dim=out_dim,
                interpolation_mode=pos_emb_interpolation_mode,
            )
        else:
            raise NotImplementedError(f"Not support pos_emb_type: {pos_emb_type}")

    def forward(
        self,
        x: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]] | None,
        *,
        pos_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._proj(x).view(x.size(0), -1)
        if pos_embeds is not None:
            return x + pos_embeds
        assert grid_thws is not None
        return self.pos_emb(x, grid_thws)

    def _proj(self, x: torch.Tensor) -> torch.Tensor:
        # MIOpen conv2d intermittently fails under load on ROCm; use aiter Triton.
        if current_platform.is_rocm() and x.dtype in (torch.float16, torch.bfloat16):
            from aiter.ops.triton.conv.conv2d import conv2d

            return conv2d(
                x,
                self.proj.weight,
                self.proj.bias,
                stride=self.patch_size,
                layout="nchw",
            )
        return self.proj(x)


class Rope2DPosEmbRepeated(nn.Module):
    """2D rotary position embedding with multi-resolution support."""

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def extra_repr(self):
        return (
            f"dim={self.dim}, max_height={self.max_height}, "
            f"max_width={self.max_width}, theta_base={self.theta_base}"
        )

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid."""
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor | list[list[int]], device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            grid_thws (torch.Tensor): grid time, height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis", self._precompute_freqs_cis(device), persistent=False
            )

        shapes = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis


class MLP2(nn.Module):
    """Two-layer MLP with tensor parallel support."""

    def __init__(
        self,
        dims: list[int],
        activation,
        bias: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ):
        super().__init__()
        assert len(dims) == 3
        self.use_data_parallel = use_data_parallel
        self.fc0 = ColumnParallelLinear(
            dims[0],
            dims[1],
            bias=bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc0"),
            disable_tp=self.use_data_parallel,
        )
        self.fc1 = RowParallelLinear(
            dims[1],
            dims[2],
            bias=bias,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "fc1"),
            disable_tp=self.use_data_parallel,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        *,
        projection_workspace: torch.Tensor | None = None,
        consume_input: bool = False,
    ) -> torch.Tensor:
        # Encoder normalization owns this activation independently of the
        # residual. Its contents are dead after fc0; fc1 may replace them.
        output_storage = (
            x.view(-1)
            if consume_input and not torch.is_grad_enabled() and x.is_contiguous()
            else None
        )
        x = _project_vision_intermediate(self.fc0, x, projection_workspace)
        if not torch.is_grad_enabled() and (
            self.activation is F.gelu or type(self.activation) is nn.GELU
        ):
            # fc0 owns this intermediate; no residual or other consumer needs
            # its pre-activation values. Keep ATen GELU math without a second
            # image-patch-sized allocation.
            torch.ops.aten.gelu.out(
                x, approximate=getattr(self.activation, "approximate", "none"), out=x
            )
        else:
            x = self.activation(x)
        return _project_vision_intermediate(self.fc1, x, output_storage)


def _make_vision_norm(norm_type: str, hidden_dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(hidden_dim)
    if norm_type == "rmsnorm":
        return nn.RMSNorm(hidden_dim)
    raise NotImplementedError(f"Not support norm_type: {norm_type}")


class MoonViTEncoderLayer(nn.Module):
    """Single encoder layer for MoonViT with TP/DP support."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        activation=F.gelu,
        attn_bias: bool = False,
        qkv_hidden_size: int | None = None,
        norm_type: str = "layernorm",
        mlp_type: str = "mlp2",
        linear_bias: bool = True,
    ):
        super().__init__()
        self.use_data_parallel = is_vit_use_data_parallel(num_heads)

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.qkv_hidden_size = (
            hidden_dim if qkv_hidden_size is None else qkv_hidden_size
        )
        self.hidden_size_per_attention_head = self.qkv_hidden_size // self.num_heads
        self.tp_size = (
            1 if self.use_data_parallel else get_tensor_model_parallel_world_size()
        )
        self.num_attention_heads_per_partition = divide(num_heads, self.tp_size)

        self.norm0 = _make_vision_norm(norm_type, hidden_dim)
        self.norm1 = _make_vision_norm(norm_type, hidden_dim)
        if mlp_type != "mlp2":
            raise NotImplementedError(f"Not support mlp_type: {mlp_type}")
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation,
            bias=linear_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            use_data_parallel=self.use_data_parallel,
        )
        self.wqkv = QKVParallelLinear(
            hidden_size=hidden_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wqkv",
            disable_tp=self.use_data_parallel,
        )
        self.wo = RowParallelLinear(
            self.qkv_hidden_size,
            hidden_dim,
            bias=attn_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.wo",
            disable_tp=self.use_data_parallel,
        )
        self.attn = MMEncoderAttention(
            num_heads=self.num_attention_heads_per_partition,
            head_size=self.hidden_size_per_attention_head,
            scale=self.hidden_size_per_attention_head**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(
            enforce_enable=True,
            is_neox_style=False,
            enable_fp32_compute=True,
        )

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        projection_workspace: torch.Tensor | None = None,
        output_workspace: torch.Tensor | None = None,
    ):
        """Compute self-attention with packed QKV.

        Args:
            x (torch.Tensor): (seqlen, hidden_dim)
            cu_seqlens (torch.Tensor): cumulative sequence lengths
        """
        seq_length = x.size(0)
        xqkv = _project_vision_intermediate(self.wqkv, x, projection_workspace)
        del x

        qkv_shape = xqkv.size()[:-1] + (
            3,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = torch.unbind(xqkv, dim=-3)

        if not torch.is_grad_enabled() and xq.dtype in (torch.bfloat16, torch.float16):
            from vllm.v1.worker.workspace import current_workspace_manager

            # The encoder and language model execute serially in one workspace
            # lane. Reuse its pre-reserved storage instead of raising the vision
            # peak after KV cache admission.
            (rope_scratch,) = current_workspace_manager().get_simultaneous(
                ((min(xq.numel(), 16 * 1024 * 1024),), torch.float32)
            )
            xq, xk = apply_rope_into_packed_qk(
                xq, xk, rope_freqs_cis, workspace=rope_scratch
            )
            del rope_scratch
        else:
            _apply_rope_input_validation(xq, rope_freqs_cis)
            _apply_rope_input_validation(xk, rope_freqs_cis)
            rope_cos = rope_freqs_cis.real.contiguous()
            rope_sin = rope_freqs_cis.imag.contiguous()
            xq = self.apply_rotary_emb(xq, rope_cos, rope_sin)
            xk = self.apply_rotary_emb(xk, rope_cos, rope_sin)

        if max_seqlen is None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_out = self.attn(
            xq.unsqueeze(0),
            xk.unsqueeze(0),
            xv.unsqueeze(0),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        # Attention has consumed Q/K/V. Drop every view of their packed
        # storage before allocating the output projection and its scratch.
        del xqkv, xq, xk, xv
        attn_out = attn_out.reshape(
            seq_length,
            self.num_attention_heads_per_partition
            * self.hidden_size_per_attention_head,
        )
        return _project_vision_intermediate(self.wo, attn_out, output_workspace)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rope_freqs_cis: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        projection_workspace: torch.Tensor | None = None,
    ):
        attention_output_workspace = None
        if projection_workspace is not None and not torch.is_grad_enabled():
            widths = [
                _vision_projection_width(layer)
                for layer in (self.wqkv, self.wo, getattr(self.mlp, "fc0", None))
            ]
            rows = hidden_states.shape[0]
            intermediate_size = rows * max(widths[0], widths[2])
            output_size = rows * widths[1]
            if all(widths) and projection_workspace.numel() >= (
                intermediate_size + output_size
            ):
                # The attention residual remains live during fc0. Give it a
                # disjoint tail, while QKV and fc0 share the preceding extent.
                attention_output_workspace = projection_workspace.narrow(
                    0, intermediate_size, output_size
                )
                projection_workspace = projection_workspace[:intermediate_size]
        residual = hidden_states
        hidden_states = self.attention_qkvpacked(
            self.norm0(hidden_states),
            cu_seqlens,
            rope_freqs_cis,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
            projection_workspace=projection_workspace,
            output_workspace=attention_output_workspace,
        )
        if projection_workspace is not None and not torch.is_grad_enabled():
            # Projection outputs have no other consumer. Do not overwrite the
            # residual, which can still belong to the encoder's caller.
            torch.add(residual, hidden_states, out=hidden_states)
        else:
            hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(
            self.norm1(hidden_states),
            projection_workspace=projection_workspace,
            consume_input=True,
        )
        if projection_workspace is not None and not torch.is_grad_enabled():
            torch.add(residual, hidden_states, out=hidden_states)
        else:
            hidden_states = residual + hidden_states

        return hidden_states


class MoonViT3dEncoder(nn.Module):
    """Full encoder stack for MoonViT 3D."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
        video_attn_type: str = "spatial_temporal",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        assert video_attn_type == "spatial_temporal", (
            f'video_attn_type must be "spatial_temporal", got {video_attn_type}'
        )
        self.video_attn_type = video_attn_type
        qkv_hidden_size = block_cfg.get("qkv_hidden_size") or block_cfg["hidden_dim"]
        self.rope_2d = Rope2DPosEmbRepeated(
            qkv_hidden_size // block_cfg["num_heads"], 512, 512
        )
        self.blocks = nn.ModuleList(
            [
                MoonViTEncoderLayer(
                    **block_cfg,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.final_layernorm = _make_vision_norm(
            block_cfg.get("norm_type", "layernorm"), hidden_dim
        )

    def prepare_encoder_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        device: torch.device,
        max_batch_size: int | None = None,
        max_seqlen_override: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        metadata: dict[str, torch.Tensor | None] = {}
        metadata["rope_freqs_cis"] = self.rope_2d.get_freqs_cis(
            grid_thw_list, device=device
        )

        grid_thw_np = np.array(grid_thw_list, dtype=np.int32)
        lengths = grid_thw_np[:, 0] * grid_thw_np[:, 1] * grid_thw_np[:, 2]
        cu_seqlens = np.concatenate(
            [np.zeros(1, dtype=np.int32), lengths.cumsum(dtype=np.int32)]
        )
        if max_batch_size is not None:
            num_seqs = len(cu_seqlens) - 1
            if num_seqs < max_batch_size:
                cu_seqlens = np.concatenate(
                    [
                        cu_seqlens,
                        np.full(
                            max_batch_size - num_seqs,
                            cu_seqlens[-1],
                            dtype=np.int32,
                        ),
                    ]
                )

        attn_backend = self.blocks[0].attn.attn_backend
        metadata["sequence_lengths"] = MMEncoderAttention.maybe_compute_seq_lens(
            attn_backend, cu_seqlens, device
        )
        max_seqlen = (
            max_seqlen_override
            if max_seqlen_override is not None
            else MMEncoderAttention.compute_max_seqlen(attn_backend, cu_seqlens)
        )
        metadata["max_seqlen"] = torch.tensor(max_seqlen, dtype=torch.int32)
        metadata["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
            attn_backend,
            cu_seqlens,
            self.blocks[0].hidden_dim,
            self.blocks[0].tp_size,
            device,
        )
        return metadata

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]] | None,
        *,
        encoder_metadata: dict[str, torch.Tensor | None] | None = None,
        projection_workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_metadata is None:
            assert grid_thws is not None
            grid_thw_list = (
                grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
            )
            encoder_metadata = self.prepare_encoder_metadata(
                grid_thw_list, device=hidden_states.device
            )

        rope_freqs_cis = encoder_metadata["rope_freqs_cis"]
        cu_seqlens = encoder_metadata["cu_seqlens"]
        max_seqlen = encoder_metadata["max_seqlen"]
        sequence_lengths = encoder_metadata.get("sequence_lengths")
        assert rope_freqs_cis is not None
        assert cu_seqlens is not None
        assert max_seqlen is not None

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                rope_freqs_cis=rope_freqs_cis,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
                projection_workspace=projection_workspace,
            )

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor | list[list[int]],
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> list[torch.Tensor]:
    """Temporal pooling patch merger."""
    kh, kw = merge_kernel_size
    grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
    lengths = [t * h * w for t, h, w in grid_thw_list]
    seqs = x.split(lengths, dim=0)

    outputs = []
    for seq, (t, h, w) in zip(seqs, grid_thw_list):
        nh, nw = h // kh, w // kw
        # Reshape: (t*h*w, d) -> (t, nh, kh, nw, kw, d)
        v = seq.view(t, nh, kh, nw, kw, -1)
        # Temporal pooling first (reduces tensor size before permute)
        v = v.mean(dim=0)  # (nh, kh, nw, kw, d)
        # Spatial rearrangement: (nh, kh, nw, kw, d) -> (nh, nw, kh, kw, d)
        out = v.permute(0, 2, 1, 3, 4).reshape(nh * nw, kh * kw, -1)
        outputs.append(out)

    return outputs


def build_image_merge_gather_idx(
    grid_thws: list[list[int]] | list[tuple[int, int, int]],
    merge_kernel_size: tuple[int, int],
) -> np.ndarray:
    """Build packed spatial-merge indices for image-only CUDA graphs."""
    kh, kw = merge_kernel_size
    parts: list[np.ndarray] = []
    offset = 0
    for t, h, w in grid_thws:
        if t != 1:
            raise ValueError("Image encoder CUDA graphs require grid T == 1")
        idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
        idx = idx.reshape(h // kh, kh, w // kw, kw)
        parts.append(idx.transpose(0, 2, 1, 3).reshape(-1, kh * kw) + offset)
        offset += h * w
    if not parts:
        return np.empty((0, kh * kw), dtype=np.int64)
    return np.concatenate(parts)


def tpool_patch_merger_packed(
    x: torch.Tensor,
    merge_gather_idx: torch.Tensor,
) -> torch.Tensor:
    """Apply the image-only spatial merge using precomputed tensor indices."""
    return x[merge_gather_idx]


class MoonViT3dPretrainedModel(nn.Module):
    """Main vision tower model.

    Uses KimiK25VisionConfig directly from transformers_utils/configs/kimi_k25.py.
    """

    def __init__(
        self,
        config: KimiK25VisionConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        config = deepcopy(config)
        self.config = config  # Required for run_dp_sharded_mrope_vision_model
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.merge_type = config.merge_type

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
            pos_emb_time=config.init_pos_emb_time,
            pos_emb_type=config.pos_emb_type,
            patch_embed_proj_bias=getattr(config, "patch_embed_proj_bias", True),
            pos_emb_interpolation_mode=getattr(
                config, "pos_emb_interpolation_mode", "bicubic"
            ),
        )

        self.encoder = MoonViT3dEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "qkv_hidden_size": getattr(config, "qkv_hidden_size", None),
                "mlp_dim": config.intermediate_size,
                "activation": get_act_fn(
                    getattr(config, "activation_func", "gelu_pytorch_tanh")
                ),
                "attn_bias": getattr(config, "attn_bias", True),
                "norm_type": getattr(config, "norm_type", "layernorm"),
                "mlp_type": getattr(config, "mlp_type", "mlp2"),
                "linear_bias": getattr(config, "linear_bias", True),
            },
            video_attn_type=config.video_attn_type,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "encoder"),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor | list[list[int]] | None,
        *,
        encoder_metadata: dict[str, torch.Tensor | None] | None = None,
        projection_workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values (torch.Tensor): The input pixel values.
            grid_thws (torch.Tensor): Temporal, height and width.

        Returns:
            torch.Tensor: The output tokens.
        """
        if encoder_metadata is not None and "pos_embeds" in encoder_metadata:
            hidden_states = self.patch_embed(
                pixel_values,
                None,
                pos_embeds=encoder_metadata["pos_embeds"],
            )
            hidden_states = self.encoder(
                hidden_states,
                None,
                encoder_metadata=encoder_metadata,
                projection_workspace=projection_workspace,
            )
            merge_gather_idx = encoder_metadata["merge_gather_idx"]
            assert merge_gather_idx is not None
            return tpool_patch_merger_packed(hidden_states, merge_gather_idx)

        assert grid_thws is not None
        grid_thw_list = grid_thws if isinstance(grid_thws, list) else grid_thws.tolist()
        if encoder_metadata is None:
            encoder_metadata = self.encoder.prepare_encoder_metadata(
                grid_thw_list, device=pixel_values.device
            )

        hidden_states = self.patch_embed(pixel_values, grid_thw_list)
        hidden_states = self.encoder(
            hidden_states,
            grid_thw_list,
            encoder_metadata=encoder_metadata,
            projection_workspace=projection_workspace,
        )
        if (
            self.merge_type == "sd2_tpool"
        ):  # spatial downsampling 2x with temporal pooling all
            hidden_states = tpool_patch_merger(
                hidden_states, grid_thw_list, merge_kernel_size=self.merge_kernel_size
            )
        else:
            raise NotImplementedError(f"Not support {self.merge_type}")

        return hidden_states

    def prepare_encoder_cudagraph_metadata(
        self,
        grid_thw_list: list[list[int]],
        *,
        max_batch_size: int,
        max_seqlen_override: int | None = None,
        device: torch.device,
    ) -> dict[str, torch.Tensor | None]:
        """Precompute fixed-buffer metadata for image encoder CUDA graphs."""
        grid_thw_list = [list(map(int, grid)) for grid in grid_thw_list]
        metadata = self.encoder.prepare_encoder_metadata(
            grid_thw_list,
            device=device,
            max_batch_size=max_batch_size,
            max_seqlen_override=max_seqlen_override,
        )
        metadata["pos_embeds"] = self.patch_embed.pos_emb.get_pos_embeds(
            grid_thw_list
        ).to(device=device)
        merge_gather_idx = build_image_merge_gather_idx(
            grid_thw_list, self.merge_kernel_size
        )
        metadata["merge_gather_idx"] = async_tensor_h2d(merge_gather_idx, device=device)
        return metadata


@torch.inference_mode()
def mm_projector_forward(mm_projector: torch.nn.Module, vt_output: list[torch.Tensor]):
    """Project independent images without copying their combined feature batch."""
    if not vt_output:
        raise ValueError("Kimi vision projection requires at least one image feature")
    norm = getattr(mm_projector, "pre_norm", None)
    if norm is None:
        norm = getattr(mm_projector, "post_norm", None)
    # Quantized linear storage can be FP8 or packed integers, not the
    # activation dtype required by the projector's normalization layers.
    projector_dtype = norm.weight.dtype if norm is not None else None
    outputs = []
    for features in vt_output:
        if projector_dtype is not None and features.dtype != projector_dtype:
            features = features.to(projector_dtype)
        output = mm_projector(features)
        outputs.append(output.reshape(-1, output.shape[-1]))
    return tuple(outputs)


@torch.inference_mode()
def vision_tower_forward(
    vision_tower: Any,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    mm_projector: Any,
    use_data_parallel: bool,
    projection_workspace: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """DP-sharded vision tower forward with mrope.

    Uses vLLM's standard data parallelism utility to shard the batch
    across available GPUs, enabling parallel processing of vision features.
    """
    model_kwargs = (
        {"projection_workspace": projection_workspace}
        if projection_workspace is not None
        else {}
    )
    if use_data_parallel:
        grid_thw_list = grid_thw.tolist()
        vt_outputs = run_dp_sharded_mrope_vision_model(
            vision_model=vision_tower,
            pixel_values=pixel_values,
            grid_thw_list=grid_thw_list,
            rope_type="rope_2d",
            vision_model_kwargs=model_kwargs,
        )
    else:
        grid_thw_list = grid_thw.tolist()
        encoder_metadata = vision_tower.encoder.prepare_encoder_metadata(
            grid_thw_list, device=pixel_values.device
        )
        vt_outputs = vision_tower(
            pixel_values,
            grid_thw_list,
            encoder_metadata=encoder_metadata,
            **model_kwargs,
        )
    tensors = mm_projector_forward(mm_projector, list(vt_outputs))
    return list(tensors)


class KimiK25MultiModalProjector(nn.Module):
    """Multi-modal projector with patch merging for Kimi-K2.5."""

    def __init__(
        self,
        config: KimiK25VisionConfig,
        use_data_parallel: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.use_data_parallel = use_data_parallel
        self.mm_projector_type = getattr(config, "mm_projector_type", "patchmerger")

        # Hidden size after patch merging
        merge_h, merge_w = config.merge_kernel_size
        self.hidden_size = config.hidden_size * merge_h * merge_w

        if self.mm_projector_type == "patchmergerv2":
            self.linear_1 = ReplicatedLinear(
                self.hidden_size,
                self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_1",
            )
            self.linear_2 = ReplicatedLinear(
                self.hidden_size,
                getattr(config, "text_hidden_size", config.mm_hidden_size),
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_2",
            )
            self.post_norm = torch.nn.RMSNorm(
                getattr(config, "text_hidden_size", config.mm_hidden_size),
                eps=config.projector_ln_eps,
            )
            self.act = GELUActivation()
            return

        self.pre_norm = torch.nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.linear_1 = ReplicatedLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_1",
        )
        self.linear_2 = ReplicatedLinear(
            self.hidden_size,
            config.mm_hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_2",
        )
        self.act = GELUActivation()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if self.mm_projector_type == "patchmergerv2":
            hidden_states = image_features.view(image_features.shape[0], -1)
            hidden_states, _ = self.linear_1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states, _ = self.linear_2(hidden_states)
            if not torch.is_grad_enabled() and hidden_states.shape[0] > 1024:
                # RMSNorm reduces each row independently. Normalize bounded
                # slices into the consumed projection output, preserving ATen
                # arithmetic without a full-image normalization temporary.
                for chunk in hidden_states.split(1024, dim=0):
                    chunk.copy_(self.post_norm(chunk))
                return hidden_states
            return self.post_norm(hidden_states)

        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states, _ = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states
