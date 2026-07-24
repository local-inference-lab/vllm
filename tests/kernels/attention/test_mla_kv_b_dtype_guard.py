# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-quantized kv_b_proj weights must not drive a kv_c_normed upcast.

NVFP4 packs into uint8 and compressed-tensors int4/int8 packs into int32.
Both are containers, not real activation dtypes -- casting kv_c_normed to
either corrupts chunked prefill past ~4K prompts.
"""

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    is_packed_quantized_dtype,
)


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.uint8, True),  # NVFP4
        (torch.int32, True),  # compressed-tensors int4/int8
        (torch.bfloat16, False),
        (torch.float16, False),
        (torch.float8_e4m3fn, False),
    ],
)
def test_is_packed_quantized_dtype(dtype: torch.dtype, expected: bool):
    assert is_packed_quantized_dtype(dtype) is expected
