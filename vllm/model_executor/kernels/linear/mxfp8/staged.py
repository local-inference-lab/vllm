# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared MXFP8 staging for sequential auxiliary-state producers."""

from __future__ import annotations

import torch

from vllm.utils.b12x import B12xPreparationUnit


def _scale_view(storage: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """Map caller-owned UE8M0 bytes to the documented F8_128x4 layout."""
    mt, kt = (rows + 127) // 128, columns // 128
    return storage.as_strided(
        (32, 4, mt, 4, kt, 1), (16, 4, kt * 512, 1, 512, mt * kt * 512)
    )


class B12xMxfp8InputAccumulator:
    """Assemble quantized feature slices, then perform one complete GEMM.

    Each slice ends at a K128 boundary. Quantization therefore preserves the
    same K32 groups as concatenation followed by quantization. All buffers are
    allocated before serving; each invocation passes fresh views to prepared
    kernels. No partial GEMM summation or target-model quantization is added.
    """

    def __init__(self, layer, output: torch.Tensor, slice_width: int):
        from b12x.gemm import blockscaled
        from b12x.quantization import mxfp8

        packed = getattr(layer, "b12x_mxfp8_packed_weight", None)
        if (
            packed is None
            or getattr(layer, "b12x_activation_mode", None) != "quantized"
        ):
            raise ValueError("staging requires a quantized-activation B12X MXFP8 FC")
        if output.ndim != 2 or output.dtype != torch.bfloat16:
            raise ValueError("staging output must be a BF16 [capacity,N] tensor")
        if (
            output.shape[1] != packed.out_features
            or output.device != packed.weight.values.device
        ):
            raise ValueError(
                "staging output differs from the FC weight geometry/device"
            )
        if (
            packed.in_features != packed.padded_in_features
            or slice_width <= 0
            or slice_width % 128
            or packed.in_features % slice_width
        ):
            raise ValueError("staging requires complete K128 slices without K padding")
        self.input_width = int(packed.in_features)
        self.max_tokens = int(output.shape[0])
        self.slice_width = slice_width
        self.output = output
        self.packed = packed
        self._tokens = self._column = 0

        def buffers(width):
            return (
                torch.empty(
                    (self.max_tokens, width),
                    dtype=torch.float8_e4m3fn,
                    device=output.device,
                ),
                torch.empty(
                    (self.max_tokens, width // 32),
                    dtype=torch.uint8,
                    device=output.device,
                ),
                torch.empty(
                    ((self.max_tokens + 127) // 128 * (width // 128) * 512,),
                    dtype=torch.uint8,
                    device=output.device,
                ),
            )

        self.values, self.scales, self.mma = buffers(self.input_width)
        self.slice_values, self.slice_scales, self.slice_mma = buffers(slice_width)
        self.workspace = torch.empty(0, dtype=torch.float32, device=output.device)
        self.quant_plan = mxfp8.plan(
            mxfp8.Mxfp8Query(
                rows=self.max_tokens,
                columns=slice_width,
                dtype="bfloat16",
                value_order="linear",
                scale_rows_layout="row_major",
                scale_rows_storage="uint8",
                scale_mma_layout="linear_storage",
                scale_mma_storage="uint8",
            )
        )
        lhs, rhs, out = self._operands(self.max_tokens)
        self.gemm_plan = blockscaled.plan(
            blockscaled.query_from_call(
                lhs,
                rhs,
                out=out,
                workspace=self.workspace,
                ab_dtype="float8_e4m3fn",
                sf_dtype="float8_e8m0fnu",
                sf_vec_size=32,
                expected_m=self.max_tokens,
            )
        )

    def _operands(self, rows):
        return (
            (
                self.values[:rows].unsqueeze(-1),
                _scale_view(self.mma, rows, self.input_width).view(
                    torch.float8_e8m0fnu
                ),
            ),
            (self.packed.weight.values.unsqueeze(-1), self.packed.weight.scale_mma),
            self.output[:rows].unsqueeze(-1),
        )

    def preparation_unit(self, name):
        from b12x.preparation import PreparedCall

        def quant_call(state):
            source = torch.full(
                (self.max_tokens, self.slice_width),
                0.125,
                dtype=torch.bfloat16,
                device=self.output.device,
            )
            return PreparedCall(
                run=lambda: state.run(
                    source,
                    self.slice_values,
                    self.slice_scales,
                    self.slice_mma,
                )
            )

        def gemm_call(state):
            lhs, rhs, out = self._operands(self.max_tokens)
            return PreparedCall(
                produce=lambda: (self.values.fill_(0.125), self.mma.fill_(127)),
                run=lambda: state.run(
                    *lhs, *rhs, None, out, self.workspace, self.output.dtype, None
                ),
                owners=(self.values, self.mma, self.packed, self.output),
            )

        return B12xPreparationUnit(
            name="MXFP8_STAGED_INPUT",
            key=(
                self.max_tokens,
                self.input_width,
                self.slice_width,
                self.output.shape[1],
            ),
            requests=(
                self.quant_plan.request(
                    name=f"{name}.quant",
                    prepare_call=quant_call,
                    benchmark_call=quant_call,
                ),
                self.gemm_plan.request(
                    name=f"{name}.gemm",
                    prepare_call=gemm_call,
                    benchmark_call=gemm_call,
                ),
            ),
            stage="weights",
        )

    def begin(self, tokens):
        if not 0 < tokens <= self.max_tokens:
            raise ValueError("staged input token count is outside prepared capacity")
        self._tokens, self._column = tokens, 0

    def append(self, source):
        from b12x.quantization import mxfp8

        rows, column, width = self._tokens, self._column, self.slice_width
        if (
            rows == 0
            or source.shape != (rows, width)
            or column + width > self.input_width
        ):
            raise ValueError(
                "staged input slice is out of order or has incompatible geometry"
            )
        mxfp8.quantize_rows(
            source.contiguous(),
            self.slice_values[:rows],
            self.slice_scales[:rows],
            self.slice_mma,
            plan=self.quant_plan,
        )
        self.values[:rows, column : column + width].copy_(self.slice_values[:rows])
        self.scales[:rows, column // 32 : (column + width) // 32].copy_(
            self.slice_scales[:rows]
        )
        _scale_view(self.mma, rows, self.input_width)[
            :, :, :, :, column // 128 : (column + width) // 128
        ].copy_(_scale_view(self.slice_mma, rows, width))
        self._column += width

    def finish(self):
        from b12x.gemm import blockscaled

        if self._tokens == 0 or self._column != self.input_width:
            raise RuntimeError("staged MXFP8 input is incomplete")
        lhs, rhs, out = self._operands(self._tokens)
        blockscaled.mm(
            lhs,
            rhs,
            plan=self.gemm_plan,
            out=out,
            workspace=self.workspace,
        )
        self._tokens = self._column = 0
        return out.squeeze(-1)
