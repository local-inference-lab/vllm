# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.kernels.linear.mxfp8 import b12x
from vllm.platforms import current_platform


def test_input_accumulator_fills_ordered_slices_and_reuses_output(monkeypatch):
    calls: list[tuple[Any, ...]] = []
    storage = SimpleNamespace()
    output = torch.empty(8, 4)

    class API:
        @staticmethod
        def empty_input(tokens, width, *, device):
            calls.append(("empty", tokens, width, device))
            return storage

        @staticmethod
        def quantize_input_slice(source, destination, *, destination_column):
            calls.append(
                (
                    "quantize",
                    source.clone(),
                    destination,
                    destination_column,
                )
            )

        @staticmethod
        def mm_quantized_into(
            source,
            weight,
            *,
            tokens,
            out,
            expected_m,
            stream,
        ):
            calls.append(("mm", source, weight, tokens, out, expected_m, stream))
            return out[:tokens]

    monkeypatch.setattr(b12x, "_import_b12x_mxfp8", lambda: API)
    monkeypatch.setattr(
        b12x,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream="stream"),
    )
    weight = SimpleNamespace(
        in_features=6,
        padded_in_features=6,
        out_features=4,
    )
    layer = SimpleNamespace(b12x_mxfp8_packed_weight=weight)
    accumulator = b12x.B12xMxfp8InputAccumulator(layer, output)

    accumulator.begin(3)
    accumulator.append(torch.ones(3, 2))
    accumulator.append(torch.full((3, 4), 2.0))
    result = accumulator.finish()

    assert result.data_ptr() == output.data_ptr()
    assert calls[0] == ("empty", 8, 6, output.device)
    assert calls[1][0] == "quantize" and calls[1][3] == 0
    assert calls[2][0] == "quantize" and calls[2][3] == 2
    assert calls[3] == ("mm", storage, weight, 3, output, 3, "stream")


@pytest.mark.skipif(
    not current_platform.is_cuda() or not torch.accelerator.is_available(),
    reason="requires CUDA",
)
def test_input_accumulator_runs_under_fullgraph_compile() -> None:
    mxfp8_linear = pytest.importorskip("b12x.gemm.mxfp8_linear")

    tokens = 8
    width = 128
    output_width = 64
    first = torch.randn((tokens, width), device="cuda", dtype=torch.bfloat16).div_(4)
    second = torch.randn_like(first).div_(4)
    weight = torch.randn(
        (output_width, 2 * width), device="cuda", dtype=torch.bfloat16
    ).div_(8)
    weight_values = weight.to(torch.float8_e4m3fn)
    weight_scales = torch.full(
        (output_width, 2 * width // 32),
        127,
        device="cuda",
        dtype=torch.uint8,
    )
    packed = mxfp8_linear.pack_weight(weight_values, weight_scales)
    layer = SimpleNamespace(b12x_mxfp8_packed_weight=packed)
    output = torch.empty((tokens, output_width), device="cuda", dtype=torch.bfloat16)
    accumulator = b12x.B12xMxfp8InputAccumulator(layer, output)

    @torch.compile(fullgraph=True)
    def compiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        accumulator.begin(tokens)
        accumulator.append(a)
        accumulator.append(b)
        return accumulator.finish()

    actual = compiled(first, second).clone()
    expected = mxfp8_linear.mm(torch.cat((first, second), dim=-1), packed)
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
