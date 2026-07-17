# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid import (
    NvFp4Nf3HybridConfig,
    _combined_tier_local_descriptors,
    _read_hybrid_keys,
    _unpack_nf3_codes,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic


@pytest.mark.parametrize(
    "config",
    [
        {
            "hybrid_bit_map": {"0": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
        },
        {
            "quantization": {
                "hybrid_bit_map": {"0": [4, 3]},
                "kept_format": "mxfp4_e8m0k32",
            }
        },
    ],
)
def test_reads_and_detects_hybrid_checkpoint(config):
    bit_map, kept_format = _read_hybrid_keys(config)

    assert bit_map == {"0": [4, 3]}
    assert kept_format == "mxfp4_e8m0k32"
    assert (
        NvFp4Nf3HybridConfig.override_quantization_method(config, None)
        == "nvfp4_nf3_hybrid"
    )
    assert NvFp4Nf3HybridConfig.override_quantization_method(config, "fp8") is None


def test_config_registration_and_parsing():
    assert get_quantization_config("nvfp4_nf3_hybrid") is NvFp4Nf3HybridConfig

    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"0": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
        }
    )

    assert config.hybrid_bit_map == {"0": [4, 3]}
    assert config.kept_format == "mxfp4_e8m0k32"


def test_config_rejects_missing_hybrid_bit_map():
    with pytest.raises(ValueError, match="hybrid_bit_map"):
        NvFp4Nf3HybridConfig.from_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
            }
        )


def test_config_accepts_dense_mxfp8_online_overlay():
    resolved = resolve_quantization_config(
        "nvfp4_nf3_hybrid",
        {
            "linear": {"weight": "mxfp8"},
            "ignore": ["re:.*kv_b_proj"],
        },
    )

    assert resolved is not None
    assert resolved.linear is not None
    assert resolved.linear.weight == kMxfp8Dynamic
    assert resolved.ignore == ["re:.*kv_b_proj"]


def test_unpack_nf3_codes():
    expected = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.int32)
    word = sum(int(code) << (index * 3) for index, code in enumerate(expected[0, 0]))
    packed = torch.tensor(
        [[[word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF]]],
        dtype=torch.uint8,
    )

    torch.testing.assert_close(_unpack_nf3_codes(packed, size_k=8), expected)


def test_grid188_tier_descriptors_encode_exact_partition():
    remap = {
        **{global_id: (0, global_id) for global_id in range(64)},
        **{global_id: (1, global_id - 64) for global_id in range(64, 256)},
    }

    descriptors = _combined_tier_local_descriptors(remap)

    assert descriptors[:64] == list(range(64))
    assert descriptors[64:] == [0x100 | local_id for local_id in range(192)]


def test_grid188_tier_descriptors_reject_incomplete_partition():
    with pytest.raises(ValueError, match="does not cover all 256"):
        _combined_tier_local_descriptors({0: (0, 0)})
