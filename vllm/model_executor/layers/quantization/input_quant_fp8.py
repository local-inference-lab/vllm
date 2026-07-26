# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
    group_broadcast,
    prep_scale_for_group_broadcast,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    get_tma_aligned_size,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


# --8<-- [start:quant_fp8]
@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    """
    Quantize input tensor to FP8 (per-tensor, per-token, per-channel, or per-group).
    This CustomOp supports both static and dynamic quantization.
    """

    # --8<-- [end:quant_fp8]

    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
        column_major_scales: bool = False,
        tma_aligned_scales: bool = False,
        use_ue8m0: bool | None = None,  # for Torch compile
        compile_native: bool = True,
    ):
        """
        Args:
            static: static or dynamic quantization
            group_shape: quantization group shape (PER_TOKEN, PER_TENSOR,
                PER_CHANNEL, or arbitrary block size)
            num_token_padding: Pad the token dimension of output to this
                size
            tma_aligned_scales: For group quantization, output scales in
                TMA-aligned layout
            column_major_scales: For group quantization, output scales in
                column major format
            compile_native: Manually compile forward_native if compile mode > None
        """
        super().__init__(compile_native=compile_native)
        self.static = static
        self.group_shape = group_shape
        self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
        self.num_token_padding = num_token_padding
        self.column_major_scales = column_major_scales
        self.tma_aligned_scales = tma_aligned_scales
        self.use_ue8m0 = is_deep_gemm_e8m0_used() if use_ue8m0 is None else use_ue8m0
        self.use_deep_gemm_supported = is_deep_gemm_supported()

        self.use_aiter = rocm_aiter_ops.is_linear_fp8_enabled()

        self.is_group_quant = group_shape.is_per_group()
        if self.is_group_quant:
            self.group_size = group_shape.col
        else:
            self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
            if not static:
                assert group_shape in (GroupShape.PER_TOKEN, GroupShape.PER_TENSOR), (
                    "Only per-token or per-tensor scales are supported for dynamic "
                    "non-group quantization."
                )

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils import fp8_utils

        if self.is_group_quant and self._deepgemm_wants_packed_scales():
            return fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
                x,
                group_size=self.group_size,
                use_ue8m0=True,
            )

        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"

            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                tma_aligned_scales=self.tma_aligned_scales,
                dtype=_FP8_DTYPE,
                use_ue8m0=self.use_ue8m0,
            )

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            group_shape=(self.group_shape.row, self.group_shape.col)
            if self.static
            else None,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_group_quant and use_triton:
            assert scale is None, "Dynamic group quantization does not use scale"

            return torch.ops.vllm.triton_per_token_group_quant_fp8(x, self.group_size)

        use_aiter_quant = self.use_aiter and scale_ub is None and x.is_contiguous()
        use_aiter_per_tensor_quant = (
            use_aiter_quant and self.group_shape.is_per_tensor()
        )
        use_aiter_per_token_quant = use_aiter_quant and self.group_shape.is_per_token()

        use_aiter_per_group_quant = use_aiter_quant and self.group_shape.is_per_group()

        if use_aiter_per_group_quant:
            return rocm_aiter_ops.group_fp8_quant(x, self.group_size)
        if use_aiter_per_tensor_quant:
            return rocm_aiter_ops.per_tensor_quant(x, _FP8_DTYPE, scale)
        if use_aiter_per_token_quant:
            return rocm_aiter_ops.per_token_quant(x, _FP8_DTYPE, scale)

        # Fallback to CUDA implementation
        return self.forward_cuda(x, scale, scale_ub)

    def forward_xpu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_group_quant and not self.static:
            from vllm.model_executor.layers.quantization.utils import fp8_utils

            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                dtype=_FP8_DTYPE,
                use_ue8m0=self.use_ue8m0,
            )
        return self.forward_cuda(x, scale, scale_ub, use_triton)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ):
        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"
            # Must mirror forward_cuda's choice exactly. When DeepGEMM wants
            # packed UE8M0, plain float32 scales are the wrong *format*, not
            # merely a different layout, and CustomOp.default_on() is False
            # under inductor -- so this is the path a compiled run takes and
            # DeepGEMM silently receives unpacked scales.
            if self._deepgemm_wants_packed_scales():
                return self._quantize_group_native_packed(x)
            return self._quantize_group_native(x)

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

            scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
        else:
            scale = prep_scale_for_group_broadcast(scale, x, self.group_shape)

        # Even for dynamic per-token scales,
        # reciprocal performs slightly better than division
        out = (
            x.to(torch.float32)
            * group_broadcast(scale.to(torch.float32), x.shape[-2:]).reciprocal()
        )
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale

    def _deepgemm_wants_packed_scales(self) -> bool:
        """Whether the consumer expects DeepGEMM's packed UE8M0 scale format.

        Shared by forward_cuda and forward_native so the two cannot disagree
        about the scale format they emit. The oracle only answers UE8M0 on the
        SM120 family; elsewhere it answers FLOAT32_CEIL_UE8M0 and both paths
        return plain float32 scales.
        """
        return (
            self.use_ue8m0
            and self.use_deep_gemm_supported
            and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
        )

    def _quantize_group_native_packed(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Native equivalent of per_token_group_quant_fp8_packed_for_deepgemm.

        Verified bit-identical to the C++ op (values, shape and TMA-aligned
        stride) across square, wide and ragged-row shapes. Expressed in plain
        torch ops so inductor can fuse it with neighbouring work, which a custom
        op cannot be.
        """
        hidden_dim = x.shape[-1]
        assert hidden_dim % self.group_size == 0, (
            f"hidden dim {hidden_dim} must be divisible by group {self.group_size}"
        )
        mn = x.numel() // hidden_dim
        num_groups = hidden_dim // self.group_size
        k_packed = (num_groups + 3) // 4

        xg = x.reshape(mn, num_groups, self.group_size).float()
        absmax = xg.abs().amax(dim=-1).clamp(min=_FP8_MIN_SCALING_FACTOR)

        # UE8M0 keeps only a power-of-two scale, so the exponent is the payload.
        exponent = torch.ceil(torch.log2(absmax / _FP8_MAX))
        quantized = (xg / torch.exp2(exponent).unsqueeze(-1)).clamp(_FP8_MIN, _FP8_MAX)
        x_q = quantized.to(_FP8_DTYPE).reshape(x.shape)

        # e8m0 byte is the biased exponent; four bytes pack into one int32.
        byte = (exponent + 127.0).clamp(0, 255).to(torch.int32)
        pad = k_packed * 4 - num_groups
        if pad:
            byte = F.pad(byte, (0, pad))
        b = byte.reshape(mn, k_packed, 4)
        packed = b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)

        # DeepGEMM requires size(-2) == mn, stride(-2) == 1 and
        # stride(-1) == get_tma_aligned_size(mn): the mn dim must be the fast
        # one, with the slow stride padded out to the TMA-aligned size.
        #
        # Build it by padding the transposed buffer, materializing, transposing
        # back and slicing -- NOT via empty_strided + copy_. Inductor is free to
        # ignore an empty_strided layout request and materialize row-major,
        # which trips the assertion, but it does preserve a transpose-derived
        # view, and slicing the leading dim afterwards keeps the strides intact.
        #
        # A bare transpose-of-contiguous is correct only when mn is already
        # TMA-aligned and silently yields stride(-1) == mn otherwise.
        tma_aligned_mn = get_tma_aligned_size(mn, packed.element_size())
        packed_t = F.pad(packed.transpose(0, 1), (0, tma_aligned_mn - mn))
        return x_q, packed_t.contiguous().transpose(0, 1)[:mn]

    def _quantize_group_native(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode="constant", value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
        scales_raw = absmax / _FP8_MAX
        if self.use_ue8m0:
            scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
        scales = (scales_raw).clamp(min=_FP8_MIN_SCALING_FACTOR)

        x_scaled = x_grouped / scales
        x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups,))

        if self.column_major_scales:
            scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

        return x_quant, scales
