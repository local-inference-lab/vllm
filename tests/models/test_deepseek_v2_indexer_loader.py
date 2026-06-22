# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    scaled_dequantize,
)
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk


class _LoadedParam:

    def __init__(self):
        self.loaded_weight = None
        self.loaded_shard_id = None

    def weight_loader(self, _param, weight, shard_id):
        self.loaded_weight = weight
        self.loaded_shard_id = shard_id


def test_try_load_fp8_indexer_wk_consumes_mxfp8_weight_scale():
    prefix = "layers.0.self_attn.indexer"
    param = _LoadedParam()
    params_dict = {f"{prefix}.wk_weights_proj.weight": param}
    loaded_params: set[str] = set()
    pending = {}

    weight_bf16 = torch.arange(64, dtype=torch.float32).view(1, 64).to(torch.bfloat16)
    weight_fp8, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)

    assert _try_load_fp8_indexer_wk(
        f"{prefix}.wk.weight",
        weight_fp8,
        pending,
        params_dict,
        loaded_params,
        [],
    )
    assert pending
    assert _try_load_fp8_indexer_wk(
        f"{prefix}.wk.weight_scale",
        weight_scale,
        pending,
        params_dict,
        loaded_params,
        [],
    )

    assert pending == {}
    assert loaded_params == {f"{prefix}.wk_weights_proj.weight"}
    assert param.loaded_shard_id == 0
    torch.testing.assert_close(
        param.loaded_weight,
        dequant_mxfp8_to_bf16(weight_fp8, weight_scale),
    )


def test_try_load_fp8_indexer_wk_preserves_fp8_weight_scale_inv_path():
    prefix = "layers.0.self_attn.indexer"
    param = _LoadedParam()
    params_dict = {f"{prefix}.wk_weights_proj.weight": param}
    loaded_params: set[str] = set()
    pending = {}

    weight_fp8 = torch.linspace(-2.0, 2.0, 32 * 64, dtype=torch.float32).view(
        32, 64
    ).to(torch.float8_e4m3fn)
    scale_inv = torch.full((1, 2), 0.25, dtype=torch.float32)

    assert _try_load_fp8_indexer_wk(
        f"{prefix}.wk.weight_scale_inv",
        scale_inv,
        pending,
        params_dict,
        loaded_params,
        [],
    )
    assert pending
    assert _try_load_fp8_indexer_wk(
        f"{prefix}.wk.weight",
        weight_fp8,
        pending,
        params_dict,
        loaded_params,
        [],
    )

    assert pending == {}
    assert loaded_params == {f"{prefix}.wk_weights_proj.weight"}
    assert param.loaded_shard_id == 0
    torch.testing.assert_close(
        param.loaded_weight,
        scaled_dequantize(
            weight_fp8,
            scale_inv,
            group_shape=GroupShape(32, 32),
            out_dtype=torch.bfloat16,
        ),
    )


def test_try_load_fp8_indexer_wk_ignores_unrelated_mxfp8_scale():
    assert not _try_load_fp8_indexer_wk(
        "layers.0.self_attn.indexer.wq_b.weight_scale",
        torch.ones((1, 2), dtype=torch.uint8),
        {},
        {},
        set(),
        [],
    )
