# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the NVFP4 Marlin power-of-2 scale factor helper.

The helper must return the same factor as the mask-and-gather
implementation while allocating no tensor-sized temporaries.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import marlin_utils_fp4
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_permute_scales,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    _nvfp4_compute_scale_factor,
    nvfp4_marlin_process_global_scale,
    nvfp4_marlin_process_scales,
    prepare_nvfp4_moe_layer_for_marlin,
)

DTYPES = [torch.float16, torch.bfloat16]
SHAPES = [(64,), (128, 256), (8, 64, 128)]
UPPER = 448 * (2**7)


def _reference_scale_factor(
    marlin_scales: torch.Tensor, a_dtype: torch.dtype | None = None
) -> float:
    """Reference reduction over positive scales in FP32."""
    if a_dtype is not None and a_dtype == torch.half:
        return 1.0
    ws_float = marlin_scales.float() * (2**7)
    nonzero_mask = ws_float > 0
    if nonzero_mask.any():
        max_val = ws_float[nonzero_mask].max()
        if max_val < UPPER:
            sf = (UPPER / max_val).log2().floor().exp2()
            return sf.item()
    return 1.0


def _cases(shape: tuple[int, ...], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    small = torch.rand(shape, generator=gen) * 0.01
    with_zeros = small.clone()
    with_zeros.flatten()[::3] = 0.0
    with_negatives = small.clone()
    with_negatives.flatten()[::5] *= -1.0
    normalized = torch.rand(shape, generator=gen) * 400.0 + 48.0
    return {
        "small": small.to(dtype),
        "with_zeros": with_zeros.to(dtype),
        "with_negatives": with_negatives.to(dtype),
        "all_zeros": torch.zeros(shape, dtype=dtype),
        "all_negative": (-small - 0.001).to(dtype),
        "already_normalized": normalized.to(dtype),
        "single_large": torch.full(shape, 0.5, dtype=dtype),
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("a_dtype", [None, torch.float16, torch.bfloat16])
def test_scale_factor_matches_reference(
    dtype: torch.dtype, shape: tuple[int, ...], a_dtype: torch.dtype | None
):
    for name, scales in _cases(shape, dtype).items():
        expected = _reference_scale_factor(scales, a_dtype)
        actual = _nvfp4_compute_scale_factor(scales, a_dtype)
        assert actual == expected, name
        assert actual >= 1.0
        assert actual == 2.0 ** round(torch.tensor(actual).log2().item())


def test_scale_factor_rescales_into_range():
    scales = torch.full((32, 64), 2**-10, dtype=torch.bfloat16)
    sf = _nvfp4_compute_scale_factor(scales)
    rescaled = scales.float().max() * (2**7) * sf
    assert 2.0 <= rescaled < UPPER


def test_scale_factor_empty_tensor():
    assert _nvfp4_compute_scale_factor(torch.empty(0, dtype=torch.bfloat16)) == 1.0


def test_scale_factor_rejects_nan():
    scales = torch.rand(16, 16, dtype=torch.bfloat16)
    scales[3, 3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        _nvfp4_compute_scale_factor(scales)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("contiguous", [False, True])
def test_scale_factor_matches_fp8_scale_domain_on_cuda(dtype, contiguous):
    """All nonnegative finite E4M3 scales retain their factor after reduction."""
    values = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn)
    scales = values.to(dtype=dtype, device="cuda").repeat(4, 1).T
    if contiguous:
        scales = scales.contiguous()
    for multiplier in (1.0, 2**-7, 2**-10):
        sample = scales * multiplier
        assert _nvfp4_compute_scale_factor(sample, torch.bfloat16) == (
            _reference_scale_factor(sample, torch.bfloat16)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_scale_factor_allocates_no_tensor_temporaries(record_property):
    device = torch.device("cuda")
    scales = torch.rand((288, 512, 256), dtype=torch.bfloat16, device=device) * 0.01
    torch.accelerator.synchronize(device)
    torch.accelerator.reset_peak_memory_stats(device)
    baseline = torch.accelerator.memory_allocated(device)
    sf = _nvfp4_compute_scale_factor(scales)
    torch.accelerator.synchronize(device)
    transient = torch.accelerator.max_memory_allocated(device) - baseline
    record_property("transient_bytes", transient)
    record_property("input_bytes", scales.numel() * scales.element_size())
    assert sf > 1.0
    # Scalar reduction must not allocate a scale-tensor-sized temporary.
    assert transient < scales.numel() * scales.element_size() // 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("intermediate", [64, 96])
@pytest.mark.parametrize("gated", [False, True])
def test_moe_scales_match_whole_bank_reference(dtype, intermediate, gated):
    """Per-expert conversion preserves scale bytes, padding and global factors."""
    experts, hidden = 4, 128
    shards = 2 if gated else 1
    padded = (intermediate + 63) // 64 * 64
    layer = SimpleNamespace(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=dtype,
    )

    def scales(shape):
        count = 1
        for extent in shape:
            count *= extent
        # Include zero and finite positive E4M3 scales at several exponents.
        values = (
            (torch.arange(count, device="cuda") % 120)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .reshape(shape)
        )
        factors = torch.tensor([0.125, 0.25, 0.5, 1.0], device="cuda")
        return (values.float() * factors[:, None, None]).to(torch.float8_e4m3fn)

    s13 = scales((experts, shards * intermediate, hidden // 16))
    s2 = scales((experts, hidden, intermediate // 16))
    g = torch.arange(1, experts + 1, dtype=torch.float32, device="cuda")
    inputs = (s13.view(torch.uint8).clone(), s2.view(torch.uint8).clone())
    result = prepare_nvfp4_moe_layer_for_marlin(
        layer,
        torch.zeros(
            (experts, shards * intermediate, hidden // 2),
            dtype=torch.uint8,
            device="cuda",
        ),
        s13,
        g,
        torch.zeros(
            (experts, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
        ),
        s2,
        g,
        gated,
    )
    for source, actual, actual_global, is_gate in (
        (s13, result[1], result[2], True),
        (s2, result[4], result[5], False),
    ):
        source = source.to(dtype)
        if is_gate:
            source = source.reshape(experts, shards, intermediate, -1)
            source = torch.nn.functional.pad(source, (0, 0, 0, padded - intermediate))
            source = source.reshape(experts, shards * padded, -1)
            size_k, size_n = hidden, shards * padded
        else:
            source = torch.nn.functional.pad(source, (0, (padded - intermediate) // 16))
            size_k, size_n = padded, hidden
        factor = _reference_scale_factor(source, dtype)
        expected = []
        for expert in source:
            permuted = marlin_permute_scales(expert.T, size_k, size_n, 16)
            converted, _ = nvfp4_marlin_process_scales(permuted, factor, dtype)
            expected.append(converted)
        torch.testing.assert_close(
            actual.view(torch.uint8),
            torch.stack(expected).view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            actual_global,
            nvfp4_marlin_process_global_scale(g, dtype) / factor,
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(s13.view(torch.uint8), inputs[0], rtol=0, atol=0)
    torch.testing.assert_close(s2.view(torch.uint8), inputs[1], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_moe_scale_conversion_avoids_bank_temporaries(monkeypatch, record_property):
    """Additional storage contains output scales and one expert's conversion."""
    experts, hidden, intermediate = 288, 1024, 512
    layer = SimpleNamespace(
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.bfloat16,
    )
    # Weight packing is independently tested; isolate scale preparation's peak.
    monkeypatch.setattr(marlin_utils_fp4, "_repack_marlin_experts", lambda w, *a: w)
    weight = torch.zeros((), dtype=torch.uint8, device="cuda")
    s13 = torch.ones(
        (experts, 2 * intermediate, hidden // 16),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    s2 = torch.ones(
        (experts, hidden, intermediate // 16), dtype=torch.float8_e4m3fn, device="cuda"
    )
    global_scale = torch.ones(experts, dtype=torch.float32, device="cuda")
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    before = torch.accelerator.memory_allocated()
    result = prepare_nvfp4_moe_layer_for_marlin(
        layer,
        weight.expand(experts, 2 * intermediate, hidden // 2),
        s13,
        global_scale,
        weight.expand(experts, hidden, intermediate // 2),
        s2,
        global_scale,
        True,
    )
    torch.accelerator.synchronize()
    extra = torch.accelerator.max_memory_allocated() - before
    output_bytes = sum(t.numel() * t.element_size() for t in (result[1], result[4]))
    record_property("peak_extra_bytes", extra)
    record_property("output_scale_bytes", output_bytes)
    assert extra < output_bytes + 4 * 1024 * 1024
