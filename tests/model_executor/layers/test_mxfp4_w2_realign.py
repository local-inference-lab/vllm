# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    convert_weight_to_mxfp4_moe_kernel_format,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    compressed_tensors_moe_w4a4_mxfp4 as compressed_mxfp4,
)
from vllm.model_executor.layers.quantization.mxfp4 import (
    _ceil_div,
    _e8m0_bytes_to_float,
    _e8m0_scale_bytes_from_amax,
    _mxfp4_decode_packed,
    _mxfp4_encode_values,
    _mxfp4_realign_w2_fp4_e8m0_to_local_k32,
    _mxfp4_require_native_w2_k32,
    _mxfp4_w2_scale_cols_for_rank,
)


def _pack_codes(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2:
        pad = torch.zeros(*codes.shape[:-1], 1, dtype=torch.uint8)
        codes = torch.cat((codes, pad), dim=-1)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def _dequant_w2(
    w2: torch.Tensor,
    scale: torch.Tensor,
    *,
    logical_k: int,
    source_k_offset: int,
) -> torch.Tensor:
    flat_w2 = w2.view(-1, w2.shape[-1])
    flat_scale = scale.view(-1, scale.shape[-1])
    raw = _mxfp4_decode_packed(flat_w2, logical_k)
    cols = torch.arange(logical_k)
    source_groups = ((source_k_offset + cols) // 32).to(torch.long)
    scale_f32 = _e8m0_bytes_to_float(flat_scale.index_select(1, source_groups))
    return raw * scale_f32


def test_mxfp4_w2_scale_cols_cover_virtual_tp_alignment_8() -> None:
    assert [
        _mxfp4_w2_scale_cols_for_rank(logical_k=312, tp_rank=rank) for rank in range(10)
    ] == [10, 11, 11, 10, 10, 11, 11, 10, 10, 11]


def test_mxfp4_w2_realign_requantizes_crossing_scale_groups() -> None:
    logical_k = 40
    source_k_offset = 24
    rows = 5
    raw_scale_cols = _ceil_div(source_k_offset + logical_k, 32)
    local_scale_cols = _ceil_div(logical_k, 32)

    codes = (torch.arange(rows * logical_k, dtype=torch.uint8) % 16).view(
        rows,
        logical_k,
    )
    w2 = _pack_codes(codes).view(1, rows, logical_k // 2)
    raw_scale = torch.tensor(
        [
            [126, 129],
            [124, 127],
            [128, 126],
            [125, 130],
            [127, 128],
        ],
        dtype=torch.uint8,
    ).view(1, rows, raw_scale_cols)

    source_vals = _dequant_w2(
        w2,
        raw_scale,
        logical_k=logical_k,
        source_k_offset=source_k_offset,
    )

    _mxfp4_realign_w2_fp4_e8m0_to_local_k32(
        w2,
        raw_scale,
        logical_k=logical_k,
        source_k_offset=source_k_offset,
        row_chunk=2,
    )

    local_scale = torch.empty(rows, local_scale_cols, dtype=torch.uint8)
    expected_codes = torch.empty(rows, logical_k, dtype=torch.uint8)
    for group_idx in range(local_scale_cols):
        k_start = group_idx * 32
        k_end = min(k_start + 32, logical_k)
        group_vals = source_vals[:, k_start:k_end]
        scale_bytes = _e8m0_scale_bytes_from_amax(group_vals.abs().amax(dim=1))
        local_scale[:, group_idx] = scale_bytes
        scale = _e8m0_bytes_to_float(scale_bytes).unsqueeze(1)
        expected_codes[:, k_start:k_end] = _mxfp4_encode_values(
            group_vals / scale.clamp(min=1e-30)
        )

    expected_w2 = _pack_codes(expected_codes).view_as(w2)
    assert torch.equal(w2, expected_w2)

    dequant_after = _dequant_w2(
        w2,
        local_scale.view(1, rows, local_scale_cols),
        logical_k=logical_k,
        source_k_offset=0,
    )
    assert torch.isfinite(dequant_after).all()


def test_b12x_native_w2_k32_keeps_kimi_k3_tp8_storage() -> None:
    # Kimi K3 has N=3072; TP8 produces a local K=384 W2 shard.
    w2 = torch.empty((2, 3584, 384 // 2), dtype=torch.uint8)
    scale = torch.empty((2, 3584, 384 // 32), dtype=torch.uint8)

    result_w2, result_scale = _mxfp4_require_native_w2_k32(
        w2,
        scale,
        logical_k=384,
        source_k_offset=0,
    )

    assert result_w2 is w2
    assert result_scale is scale
    assert result_w2.untyped_storage().data_ptr() == w2.untyped_storage().data_ptr()
    assert (
        result_scale.untyped_storage().data_ptr() == scale.untyped_storage().data_ptr()
    )


def test_b12x_native_w2_k32_refuses_full_shard_requantization() -> None:
    w2 = torch.empty((1, 8, 20), dtype=torch.uint8)
    scale = torch.empty((1, 8, 2), dtype=torch.uint8)

    with pytest.raises(ValueError, match="refusing full-shard"):
        _mxfp4_require_native_w2_k32(
            w2,
            scale,
            logical_k=40,
            source_k_offset=24,
        )


def test_b12x_kimi_k3_shapes_and_checkpoint_tensors_are_not_generic_repacked() -> None:
    assert mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.B12X,
        hidden_size=3584,
        intermediate_size=3072 // 8,
    ) == (3584, 384)

    source = (
        torch.nn.Parameter(
            torch.empty((2, 768, 1792), dtype=torch.uint8), requires_grad=False
        ),
        torch.nn.Parameter(
            torch.empty((2, 3584, 192), dtype=torch.uint8), requires_grad=False
        ),
        torch.nn.Parameter(
            torch.empty((2, 768, 112), dtype=torch.uint8), requires_grad=False
        ),
        torch.nn.Parameter(
            torch.empty((2, 3584, 12), dtype=torch.uint8), requires_grad=False
        ),
    )
    converted = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.B12X,
        layer=torch.nn.Module(),
        w13_weight=source[0],
        w2_weight=source[1],
        w13_weight_scale=source[2],
        w2_weight_scale=source[3],
    )

    for original, result in zip(source, converted[:4]):
        assert isinstance(result, torch.Tensor)
        assert (
            result.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
        )


def test_compressed_mxfp4_situ_selects_b12x(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compressed_mxfp4.CutlassExpertsMxfp4,
        "_supports_current_device",
        lambda: True,
    )
    monkeypatch.setattr(
        compressed_mxfp4,
        "select_mxfp4_moe_backend",
        lambda _moe: (Mxfp4MoeBackend.B12X, B12xExperts),
    )

    method = compressed_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod(
        type("MoeConfig", (), {"activation": MoEActivation.SITU})()
    )

    assert method.mxfp4_backend == Mxfp4MoeBackend.B12X
    assert method.experts_cls is B12xExperts
    assert not method.use_cutlass_mxfp4


def test_compressed_mxfp4_b12x_keeps_checkpoint_storage_and_prepares_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = object.__new__(compressed_mxfp4.CompressedTensorsW4A4Mxfp4MoEMethod)
    method.use_cutlass_mxfp4 = False
    method.mxfp4_backend = Mxfp4MoeBackend.B12X
    method.moe = object()
    method.experts_cls = B12xExperts
    method.moe_quant_config = None
    method.moe_kernel = None

    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_weight_packed",
        torch.nn.Parameter(
            torch.empty(2, 16, 8, dtype=torch.uint8), requires_grad=False
        ),
    )
    layer.register_parameter(
        "w2_weight_packed",
        torch.nn.Parameter(
            torch.empty(2, 8, 8, dtype=torch.uint8), requires_grad=False
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.empty(2, 16, 1, dtype=torch.uint8), requires_grad=False
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.empty(2, 8, 1, dtype=torch.uint8), requires_grad=False
        ),
    )
    layer._expert_routing_tables = lambda: None

    w13_ptr = layer.w13_weight_packed.untyped_storage().data_ptr()
    w2_ptr = layer.w2_weight_packed.untyped_storage().data_ptr()
    prepared_layers = []
    fake_experts = type(
        "FakeExperts",
        (),
        {
            "process_weights_after_loading": lambda self, target: (
                prepared_layers.append(target)
            )
        },
    )()
    fake_kernel = type("FakeKernel", (), {"fused_experts": fake_experts})()

    monkeypatch.setattr(
        compressed_mxfp4,
        "make_mxfp4_moe_quant_config",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        compressed_mxfp4,
        "make_mxfp4_moe_kernel",
        lambda **_kwargs: fake_kernel,
    )
    monkeypatch.setattr(
        compressed_mxfp4,
        "prepare_moe_fp4_layer_for_marlin",
        lambda _layer: pytest.fail("B12X must not invoke the Marlin repacker"),
    )

    method.process_weights_after_loading(layer)

    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")
    assert layer.w13_weight.untyped_storage().data_ptr() == w13_ptr
    assert layer.w2_weight.untyped_storage().data_ptr() == w2_ptr
    assert prepared_layers == [layer]
