# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.kernels.linear.bf16_staging import Bf16InputAccumulator


def make_accumulator(device="cpu"):
    weight = torch.arange(24, device=device, dtype=torch.bfloat16).reshape(3, 8)
    layer = SimpleNamespace(
        input_size=8,
        weight=weight,
        bias=None,
        quant_method=SimpleNamespace(
            apply=lambda layer, x, bias: torch.nn.functional.linear(
                x, layer.weight, bias
            )
        ),
    )
    return Bf16InputAccumulator(
        layer, torch.empty(129, 3, device=device, dtype=torch.bfloat16), 4
    )


@pytest.mark.parametrize("rows", [1, 8, 129])
def test_bf16_staging_preserves_bits_and_releases_slices(rows):
    accumulator = make_accumulator()
    source = torch.arange(rows * 8, dtype=torch.bfloat16).reshape(rows, 8)
    expected = accumulator.layer.quant_method.apply(accumulator.layer, source, None)
    accumulator.begin(rows)
    reusable = torch.empty(rows, 4, dtype=torch.bfloat16)
    for chunk in source.split(4, dim=-1):
        reusable.copy_(chunk)
        accumulator.append(reusable)
        reusable.fill_(float("nan"))
    torch.testing.assert_close(accumulator.values[:rows], source, rtol=0, atol=0)
    torch.testing.assert_close(accumulator.finish(), expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="incomplete"):
        accumulator.finish()


def test_bf16_staging_rejects_partial_or_overfilled_inputs():
    accumulator = make_accumulator()
    source = torch.ones(8, 4, dtype=torch.bfloat16)
    for rows in (0, 130):
        with pytest.raises(ValueError, match="capacity"):
            accumulator.begin(rows)
    with pytest.raises(ValueError, match="geometry"):
        accumulator.append(source)
    accumulator.begin(8)
    with pytest.raises(ValueError, match="dtype"):
        accumulator.append(source.float())
    accumulator.append(source)
    with pytest.raises(RuntimeError, match="incomplete"):
        accumulator.finish()
    accumulator.append(source)
    with pytest.raises(ValueError, match="geometry"):
        accumulator.append(source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_staging_graph_replays_changed_inputs():
    accumulator = make_accumulator("cuda")
    source = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)

    def run():
        accumulator.begin(8)
        for chunk in source.split(4, dim=-1):
            accumulator.append(chunk)
        return accumulator.finish()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    pointer = accumulator.values.data_ptr()
    for scale in (0.125, 3.0, -0.5):
        source.mul_(scale)
        graph.replay()
        expected = accumulator.layer.quant_method.apply(accumulator.layer, source, None)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert accumulator.values.data_ptr() == pointer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows", [1, 8, 129, 4096])
@torch.inference_mode()
def test_dflash_tp16_marlin_staging_matches_bf16_concatenation(rows):
    from torch import nn

    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
        prepare_mxfp8_layer_for_marlin,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    torch.manual_seed(231)
    width, slices, local_output = 7168, 6, 448
    layer = nn.Module()
    layer.input_size = layer.input_size_per_partition = width * slices
    layer.output_size_per_partition = local_output
    layer.bias = None
    weight = (
        torch.randn(local_output, width * slices, device="cuda", dtype=torch.bfloat16)
        / 64
    )
    values, scales = mxfp8_e4m3_quantize(weight)
    layer.weight = nn.Parameter(values, requires_grad=False)
    layer.weight_scale = nn.Parameter(scales, requires_grad=False)
    prepare_mxfp8_layer_for_marlin(layer)
    kernel = object.__new__(MarlinMxfp8LinearKernel)
    layer.quant_method = SimpleNamespace(apply=kernel.apply_weights)
    accumulator = Bf16InputAccumulator(
        layer,
        torch.empty(rows, local_output, device="cuda", dtype=torch.bfloat16),
        width,
    )
    source = torch.randn(rows, width * slices, device="cuda", dtype=torch.bfloat16)

    def run():
        accumulator.begin(rows)
        for part in source.split(width, dim=-1):
            accumulator.append(part)
        return accumulator.finish()

    expected = kernel.apply_weights(layer, source)
    actual = run()
    torch.testing.assert_close(accumulator.values, source, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for scale in (-0.25, 2.0):
        source.mul_(scale)
        graph.replay()
        expected = kernel.apply_weights(layer, source)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
