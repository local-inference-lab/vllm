# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization import kquant_kimi_k3_qsrt_tp12 as qsrt
from vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid import (
    _QSRT_X4T_W2_EXCEPTION_TASK_ROWS,
    _QSRT_X4T_W13_EXCEPTION_ROW_ROTATION,
    _QSRT_X4T_W13_EXCEPTION_TASK_ROWS,
    _QSRT_X4T_W13_LAYOUT,
    NvFp4Nf3HybridConfig,
    NvFp4Nf3HybridMoEMethod,
    _b12x_tiles_for_geometry,
    _combined_tier_local_descriptors,
    _decode_kquant_nf3_scale,
    _is_dense_layer_ignored,
    _qsrt_w4a8_requested,
    _read_hybrid_keys,
    _require_rank_local_kept_kernel,
    _stack_exl3_intermediate_rotations,
    _unpack_nf3_codes,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


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


def test_config_accepts_fixed_tp12_mixed_exl3_slab():
    descriptor = {
        "schema": "kquant_mixed_exl3_tp12_proto_v3",
        "tp_size": 12,
    }
    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"1": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
            "demoted_format": "mixed_exl3_tp12",
            "mixed_exl3_tp12": descriptor,
        }
    )

    assert config.demoted_format == "mixed_exl3_tp12"
    assert config.mixed_exl3_tp12 == descriptor


def test_config_accepts_mul1_e4m3_trellis() -> None:
    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"1": [4, 3]},
            "trellis": {
                "codebook": "mul1-e4m3",
                "mul1_mult": qsrt.MUL1_MULT,
                "reconstruction_dtype": "e4m3",
            },
        }
    )

    assert config.trellis_codebook == "mul1-e4m3"
    assert config.trellis_mul1_e4m3 & 0xFFFFFFFF == qsrt.MUL1_MULT


def test_config_accepts_qsrt_v5_sqg_external_x4t() -> None:
    descriptor = {
        "schema": "kquant_mixed_exl3_tp12_proto_v3",
        "layer_header_version": 5,
        "tp_size": 12,
        "layer_file_pattern": "mixed-exl3-tp12-layer-{layer:05d}.bin",
        "x4t_tp12_rank_file_pattern": (
            "x4t-tp12-layer-{layer:05d}-rank-{rank:02d}.safetensors"
        ),
        "x4t_tp12_version": 1,
    }
    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"1": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
            "kept_storage": "external-x4t",
            "demoted_format": "mixed_exl3_tp12",
            "mixed_exl3_tp12": descriptor,
            "trellis": {
                "bits": 3,
                "codebook": "sqg-normal-e4m3",
                "labelling": "sqg-l16-normal-r44-v1",
                "reconstruction_dtype": "e4m3",
                "rate_dependent_reconstruction": True,
                "shared_su": True,
                "pair_format": "tp12_p24_p33",
                "mode_ids": [0, 1, 2],
                "separate_r13_r2": True,
            },
        }
    )

    assert config.mixed_exl3_tp12 == descriptor
    assert config.kept_storage == "external-x4t"
    assert config.trellis_codebook == "sqg-normal-e4m3"


def test_config_accepts_qsrt_sqg_cheb_normal() -> None:
    descriptor = {
        "schema": "kquant_mixed_exl3_tp12_proto_v3",
        "layer_header_version": 5,
        "tp_size": 12,
        "layer_file_pattern": "mixed-exl3-tp12-layer-{layer:05d}.bin",
        "x4t_tp12_rank_file_pattern": (
            "x4t-tp12-layer-{layer:05d}-rank-{rank:02d}.safetensors"
        ),
        "x4t_tp12_version": 1,
    }
    config = NvFp4Nf3HybridConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "hybrid_bit_map": {"1": [4, 3]},
            "kept_format": "mxfp4_e8m0k32",
            "kept_storage": "external-x4t",
            "demoted_format": "mixed_exl3_tp12",
            "mixed_exl3_tp12": descriptor,
            "trellis": {
                "bits": 3,
                "codebook": "sqg-cheb-normal-e4m3",
                "labelling": "sqg-l16-normal-cheb-v1",
                "reconstruction_dtype": "e4m3",
                "rate_dependent_reconstruction": True,
                "shared_su": True,
                "pair_format": "tp12_p24_p33",
                "mode_ids": [0, 1, 2],
                "separate_r13_r2": True,
            },
        }
    )

    assert config.trellis_codebook == "sqg-cheb-normal-e4m3"


def test_qsrt_x4t_tp12_uses_gate_up_w4a16_abi() -> None:
    """Keep source row semantics and B12X exception scheduling explicit."""

    assert _QSRT_X4T_W13_LAYOUT == "w31"
    assert _QSRT_X4T_W13_EXCEPTION_ROW_ROTATION == 0
    assert _QSRT_X4T_W13_EXCEPTION_TASK_ROWS == 128
    assert _QSRT_X4T_W2_EXCEPTION_TASK_ROWS == 896


@pytest.mark.parametrize("disabled", ["0", "false", "no", "off"])
def test_native_trellis_w4a8_has_fail_safe_toggle(
    monkeypatch: pytest.MonkeyPatch, disabled: str
) -> None:
    monkeypatch.setenv("VLLM_KQUANT_TRELLIS_W4A8", disabled)
    assert not _qsrt_w4a8_requested()
    monkeypatch.setenv("VLLM_KQUANT_TRELLIS_W4A8", "1")
    assert _qsrt_w4a8_requested()


def test_native_trellis_w4a8_contract_is_exact_tp12(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid as hybrid

    method = object.__new__(NvFp4Nf3HybridMoEMethod)
    method.quant_config = SimpleNamespace(trellis_codebook="mul1-e4m3")
    method.moe = SimpleNamespace(activation=SimpleNamespace(value="situ"))
    state = SimpleNamespace(
        uses_mixed_tp12_slab=True,
        hidden_size=3584,
        intermediate_size=256,
        trellis_weights=SimpleNamespace(
            representation=SimpleNamespace(
                value=SimpleNamespace(
                    gate_suh=torch.empty((1, 3584)),
                    up_suh=torch.empty((1, 3584)),
                )
            )
        ),
    )
    monkeypatch.setenv("VLLM_KQUANT_TRELLIS_W4A8", "1")
    monkeypatch.setattr(hybrid, "get_tensor_model_parallel_world_size", lambda: 12)
    assert method._can_use_trellis_w4a8(state, topk=16)

    method.quant_config.trellis_codebook = "sqg-normal-e4m3"
    assert method._can_use_trellis_w4a8(state, topk=16)
    method.quant_config.trellis_codebook = "sqg-cheb-normal-e4m3"
    assert method._can_use_trellis_w4a8(state, topk=16)
    method.quant_config.trellis_codebook = "mul1-e4m3"

    state.intermediate_size = 192
    assert not method._can_use_trellis_w4a8(state, topk=16)
    state.intermediate_size = 256
    method.quant_config.trellis_codebook = "mcg"
    assert not method._can_use_trellis_w4a8(state, topk=16)
    method.quant_config.trellis_codebook = "mul1-e4m3"
    state.trellis_weights.representation.value.gate_suh = torch.empty((2, 3584))
    assert not method._can_use_trellis_w4a8(state, topk=16)


def test_native_trellis_w4a8_runner_uses_shared_scratch_and_expert_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sparkinfer.moe._shared.kernels.trellis_w4a8 as native

    calls: list[tuple] = []
    result = torch.randn((1, 8), dtype=torch.float32)

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(native, "run_trellis_w4a8_moe", fake_run)
    method = object.__new__(NvFp4Nf3HybridMoEMethod)
    scratch = object()
    method.quant_config = SimpleNamespace(
        shared_runtime=SimpleNamespace(trellis_w4a8_scratch={1: scratch})
    )
    prepared = object()
    expert_map = torch.tensor([0, -1], dtype=torch.int32)
    state = SimpleNamespace(
        trellis_weights=SimpleNamespace(representation=SimpleNamespace(value=prepared)),
        emap_nf3=expert_map,
    )
    layer = SimpleNamespace(hybrid_state=state)
    x = torch.randn((1, 8), dtype=torch.bfloat16)
    weights = torch.ones((1, 2), dtype=torch.float32)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)

    assert method._run_trellis_w4a8(layer, x, weights, ids) is result
    args, kwargs = calls.pop()
    assert args[1] is prepared
    assert args[4] is scratch
    assert kwargs == {"expert_map": expert_map, "fast_math": True}


@pytest.mark.parametrize(
    "trellis",
    [
        {"codebook": "mul1-e4m3", "mul1_mult": qsrt.MUL1_MULT},
        {"codebook": "mul1-e4m3", "reconstruction_dtype": "e4m3"},
        {"codebook": "unknown"},
    ],
)
def test_config_rejects_incomplete_mul1_e4m3_trellis(trellis) -> None:
    with pytest.raises(ValueError, match="trellis|mul1-e4m3"):
        NvFp4Nf3HybridConfig.from_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "hybrid_bit_map": {"1": [4, 3]},
                "trellis": trellis,
            }
        )


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        {"schema": "wrong", "tp_size": 12},
        {"schema": "kquant_mixed_exl3_tp12_proto_v3", "tp_size": 4},
    ],
)
def test_config_rejects_invalid_fixed_tp12_descriptor(descriptor):
    raw = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "hybrid_bit_map": {"1": [4, 3]},
        "kept_format": "mxfp4_e8m0k32",
        "demoted_format": "mixed_exl3_tp12",
    }
    if descriptor is not None:
        raw["mixed_exl3_tp12"] = descriptor

    with pytest.raises(ValueError, match="mixed_exl3_tp12"):
        NvFp4Nf3HybridConfig.from_config(raw)


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


def test_config_does_not_quantize_bf16_lm_head():
    config = NvFp4Nf3HybridConfig(
        is_checkpoint_nvfp4_serialized=True,
        hybrid_bit_map={"1": [4, 3]},
    )
    lm_head = ParallelLMHead.__new__(ParallelLMHead)

    assert config.get_quant_method(lm_head, "lm_head") is None


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("model.layers.0.self_attn.g_proj", True),
        ("model.layers.0.self_attn.b_proj", True),
        ("model.layers.0.self_attn.q_b_proj", False),
        ("model.layers.3.self_attn.kv_b_proj", True),
        ("model.vision_tower.encoder.layers.0.mlp.fc1", True),
        ("model.layers.0.self_attn.q_proj", False),
    ],
)
def test_dense_mxfp8_short_exclusions_match_path_components(prefix, expected):
    ignored = ["g_proj", "b_proj", "kv_b_proj", "vision_tower"]

    assert _is_dense_layer_ignored(prefix, ignored, {}) is expected


def test_dense_mxfp8_full_prefix_exclusion_still_matches():
    prefix = "model.layers.0.self_attn.q_proj"

    assert _is_dense_layer_ignored(prefix, [prefix], {})


def test_dense_mxfp8_rejects_partially_excluded_fused_linear():
    with pytest.raises(ValueError, match="some but not all shards"):
        _is_dense_layer_ignored(
            "model.layers.0.mlp.gate_up_proj",
            ["gate_proj"],
            {"gate_up_proj": ["gate_proj", "up_proj"]},
        )


def test_unpack_nf3_codes():
    expected = torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7]]], dtype=torch.int32)
    word = sum(int(code) << (index * 3) for index, code in enumerate(expected[0, 0]))
    packed = torch.tensor(
        [[[word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF]]],
        dtype=torch.uint8,
    )

    torch.testing.assert_close(_unpack_nf3_codes(packed, size_k=8), expected)


def test_exl3_rotation_bundle_follows_b12x_projection_order():
    w13_svh = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=torch.float16,
    )
    w2_suh = torch.tensor([[9.0, 10.0], [11.0, 12.0]], dtype=torch.float16)

    actual = _stack_exl3_intermediate_rotations(w13_svh, w2_suh)

    assert actual.tolist() == [
        [1.0, 2.0, 3.0, 4.0, 9.0, 10.0],
        [5.0, 6.0, 7.0, 8.0, 11.0, 12.0],
    ]


def test_hybrid_kept_kernel_must_return_rank_local_partial():
    class Kernel:
        def __init__(self, reduced: bool):
            self.reduced = reduced

        def output_is_reduced(self) -> bool:
            return self.reduced

    _require_rank_local_kept_kernel(Kernel(False))
    with pytest.raises(RuntimeError, match="unreduced rank-local partial"):
        _require_rank_local_kept_kernel(Kernel(True))


def test_kimi_tp16_uses_tuned_fc1_tile():
    assert _b12x_tiles_for_geometry(3584, 3072 // 16) == (128, 64, 64, 128)


def test_kquant_nf3_scale_reinterprets_raw_fp8_bits():
    biased = torch.tensor([0.5, 2.0, 8.0], dtype=torch.float8_e4m3fn)
    decoded = _decode_kquant_nf3_scale(biased.view(torch.uint8))

    assert decoded.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(decoded.float() * (2.0**-4), biased.float() / 16)


def test_grid188_tier_descriptors_encode_exact_partition():
    remap = {
        **{global_id: (0, global_id) for global_id in range(64)},
        **{global_id: (1, global_id - 64) for global_id in range(64, 256)},
    }

    descriptors = _combined_tier_local_descriptors(remap)

    assert descriptors[:64] == list(range(64))
    assert descriptors[64:] == [0x10000 | local_id for local_id in range(192)]


def test_k3_tier_descriptors_preserve_local_ids_above_255():
    num_kept, num_nf3 = 717, 179
    remap = {
        **{global_id: (0, global_id) for global_id in range(num_kept)},
        **{
            global_id: (1, global_id - num_kept)
            for global_id in range(num_kept, num_kept + num_nf3)
        },
    }

    descriptors = _combined_tier_local_descriptors(
        remap,
        num_experts=num_kept + num_nf3,
        num_kept=num_kept,
        num_nf3=num_nf3,
    )

    assert descriptors[256] == 256
    assert descriptors[716] == 716
    assert descriptors[717] == 0x10000
    assert descriptors[-1] == 0x10000 | 178


def test_grid188_tier_descriptors_reject_incomplete_partition():
    with pytest.raises(ValueError, match="does not cover all 256"):
        _combined_tier_local_descriptors({0: (0, 0)})


def _tp12_test_header(
    layer: int,
    layout: qsrt.TP12SlabLayout,
    *,
    codebook: str = qsrt.CODEBOOK_MCG,
) -> bytes:
    prefix = qsrt._HEADER.pack(
        qsrt.MAGIC,
        qsrt.HEADER_VERSION,
        qsrt.HEADER_BYTES,
        qsrt.TP_SIZE,
        layer,
        qsrt.EXPERTS,
        layout.compressed_experts,
        layout.kept_experts,
        8,
        qsrt.CODEBOOK_IDS[codebook],
        qsrt.CODEBOOK_MULTIPLIERS[codebook],
        qsrt.ALIGNMENT,
        qsrt.HEADER_BYTES,
        qsrt.HEADER_BYTES + qsrt.FORMAT_BYTES,
        layout.rank_sections_offset,
        layout.rank_stride,
        layout.disk_bytes,
    )
    return prefix + bytes(qsrt.HEADER_BYTES - len(prefix))


def _tp12_legacy_header(layer: int, layout: qsrt.TP12SlabLayout) -> bytes:
    prefix = qsrt._LEGACY_HEADER.pack(
        qsrt.LEGACY_MAGIC,
        qsrt.LEGACY_HEADER_VERSION,
        qsrt.HEADER_BYTES,
        qsrt.TP_SIZE,
        layer,
        qsrt.EXPERTS,
        layout.compressed_experts,
        layout.kept_experts,
        8,
        qsrt.MCG_MULT,
        qsrt.ALIGNMENT,
        qsrt.HEADER_BYTES,
        qsrt.HEADER_BYTES + qsrt.FORMAT_BYTES,
        layout.rank_sections_offset,
        layout.rank_stride,
        layout.disk_bytes,
    )
    return prefix + bytes(qsrt.HEADER_BYTES - len(prefix))


def _tp12_qsrt_header(
    layer: int,
    layout: qsrt.TP12SlabLayout,
    *,
    codebook: str = qsrt.CODEBOOK_SQG_NORMAL_E4M3,
) -> bytes:
    prefix = qsrt._QSRT_HEADER.pack(
        qsrt.QSRT_MAGIC,
        qsrt.QSRT_HEADER_VERSION,
        qsrt.HEADER_BYTES,
        qsrt.TP_SIZE,
        layer,
        qsrt.EXPERTS,
        layout.compressed_experts,
        layout.kept_experts,
        8,
        qsrt.CODEBOOK_IDS[codebook],
        qsrt.KEEP_STORAGE_IDS[qsrt.KEEP_STORAGE_EXTERNAL_X4T],
        qsrt.CODEBOOK_MULTIPLIERS[codebook],
        qsrt.ALIGNMENT,
        qsrt.HEADER_BYTES,
        qsrt.HEADER_BYTES + qsrt.FORMAT_BYTES,
        layout.rank_sections_offset,
        layout.rank_stride,
        layout.disk_bytes,
    )
    return prefix + bytes(qsrt.HEADER_BYTES - len(prefix))


def _write_tp12_slab_prefix(
    path,
    *,
    layer: int,
    layout: qsrt.TP12SlabLayout,
    formats: bytes,
    shared: torch.Tensor | None = None,
) -> None:
    assert len(formats) == qsrt.EXPERTS
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        os.pwrite(descriptor, _tp12_test_header(layer, layout), 0)
        os.pwrite(
            descriptor,
            formats + bytes(qsrt.FORMAT_BYTES - len(formats)),
            qsrt.HEADER_BYTES,
        )
        if shared is None:
            shared = torch.zeros((3, qsrt.HIDDEN), dtype=torch.float16)
        shared_bytes = bytes(shared.contiguous().view(torch.uint8).numpy())
        os.pwrite(
            descriptor,
            shared_bytes + bytes(qsrt.SHARED_SCALE_BYTES - len(shared_bytes)),
            qsrt.HEADER_BYTES + qsrt.FORMAT_BYTES,
        )
    finally:
        os.close(descriptor)


def test_tp12_tensor_reader_accepts_zero_sized_tiers():
    tensor = qsrt._tensor_from_bytes(
        b"",
        dtype=torch.float16,
        shape=(0, 3, qsrt.LOCAL_INTERMEDIATE),
    )

    assert tensor.shape == (0, 3, qsrt.LOCAL_INTERMEDIATE)
    assert tensor.dtype == torch.float16


def test_tp12_pair_rotation_balances_physical_ranks_without_metadata():
    counts = [0] * qsrt.TP_SIZE
    for expert in range(qsrt.EXPERTS):
        for rank in range(qsrt.TP_SIZE):
            counts[rank] += qsrt._logical_pair_index(24, expert, rank) < 5

    assert max(counts) - min(counts) <= 1


def test_tp12_slab_reader_emits_b12x_rank_contract(tmp_path):
    layer = 1
    rank = 5
    expert = 17
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    path = tmp_path / "mixed-exl3-tp12-layer-00001.bin"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.ftruncate(descriptor, layout.disk_bytes)
        os.pwrite(descriptor, _tp12_test_header(layer, layout), 0)
        formats = bytes([0x37]) * qsrt.EXPERTS
        os.pwrite(
            descriptor,
            formats + bytes(qsrt.FORMAT_BYTES - len(formats)),
            qsrt.HEADER_BYTES,
        )
        shared = torch.cat(
            [
                torch.full((qsrt.HIDDEN,), value, dtype=torch.float16)
                for value in (1.0, 2.0, 3.0)
            ]
        ).view(torch.uint8)
        shared_bytes = bytes(shared.numpy())
        os.pwrite(
            descriptor,
            shared_bytes + bytes(qsrt.SHARED_SCALE_BYTES - len(shared_bytes)),
            qsrt.HEADER_BYTES + qsrt.FORMAT_BYTES,
        )
        rank_offset = layout.rank_offset(rank)
        pair = torch.empty((3, qsrt.PAIR_WORDS), dtype=torch.int16)
        pair[0].fill_(101)
        pair[1].fill_(202)
        pair[2].fill_(303)
        os.pwrite(
            descriptor,
            bytes(pair.view(torch.uint8).numpy()),
            rank_offset + expert * qsrt.EXPERT_RANK_TRELLIS_BYTES,
        )
        local = torch.empty((3, qsrt.LOCAL_INTERMEDIATE), dtype=torch.float16)
        local[0].fill_(11.0)
        local[1].fill_(22.0)
        local[2].fill_(33.0)
        os.pwrite(
            descriptor,
            bytes(local.view(torch.uint8).numpy()),
            rank_offset
            + layout.rank_trellis_bytes
            + expert * qsrt.EXPERT_RANK_SCALE_BYTES,
        )
    finally:
        os.close(descriptor)

    payload = qsrt.read_tp12_rank_payload(
        path,
        layer=layer,
        rank=rank,
        expected_bits=[3] * qsrt.EXPERTS,
        selected_experts=[expert],
    )

    assert payload.compressed_expert_ids.tolist() == [expert]
    assert payload.kept_expert_ids.numel() == 0
    assert bool(torch.all(payload.w13_trellis[0, 0] == 101))
    assert bool(torch.all(payload.w13_trellis[1, 0] == 202))
    assert bool(torch.all(payload.w2_trellis[0] == 303))
    assert payload.fc1_pair_modes.tolist() == [0]
    assert payload.fc2_pair_modes.tolist() == [1]
    assert payload.gate_suh[0, 0].item() == 1.0
    assert payload.up_suh[0, 0].item() == 2.0
    assert payload.down_svh[0, 0].item() == 3.0
    assert payload.intermediate_rotations[0, 0].item() == 11.0
    assert payload.intermediate_rotations[0, 256].item() == 22.0
    assert payload.intermediate_rotations[0, 512].item() == 33.0


def test_tp12_slab_header_rejects_nonzero_reserved_bytes():
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    header = bytearray(_tp12_test_header(1, layout))
    header[-1] = 1

    with pytest.raises(ValueError, match="nonzero reserved"):
        qsrt.parse_tp12_slab_header(bytes(header))


def test_tp12_slab_header_identifies_mul1_and_retains_legacy_mcg() -> None:
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)

    mul1 = qsrt.parse_tp12_slab_header(
        _tp12_test_header(1, layout, codebook=qsrt.CODEBOOK_MUL1_E4M3)
    )
    legacy = qsrt.parse_tp12_slab_header(_tp12_legacy_header(1, layout))

    assert mul1.codebook == qsrt.CODEBOOK_MUL1_E4M3
    assert legacy.codebook == qsrt.CODEBOOK_MCG


def test_tp12_slab_header_identifies_qsrt_external_x4t() -> None:
    layout = qsrt.TP12SlabLayout(
        qsrt.EXPERTS - 1,
        1,
        keep_storage=qsrt.KEEP_STORAGE_EXTERNAL_X4T,
    )

    header = qsrt.parse_tp12_slab_header(_tp12_qsrt_header(24, layout))

    assert header.codebook == qsrt.CODEBOOK_SQG_NORMAL_E4M3
    assert header.keep_storage == qsrt.KEEP_STORAGE_EXTERNAL_X4T
    assert header.layout.rank_keep_bytes == 0

    cheb = qsrt.parse_tp12_slab_header(
        _tp12_qsrt_header(
            24,
            layout,
            codebook=qsrt.CODEBOOK_SQG_CHEB_NORMAL_E4M3,
        )
    )
    assert cheb.codebook == qsrt.CODEBOOK_SQG_CHEB_NORMAL_E4M3


def test_tp12_slab_reader_rejects_truncated_file(tmp_path):
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    path = tmp_path / "truncated.bin"
    path.write_bytes(_tp12_test_header(1, layout))

    with pytest.raises(ValueError, match="file size disagrees"):
        qsrt.read_tp12_rank_payload(path, layer=1, rank=0)


def test_tp12_slab_reader_rejects_invalid_format_code(tmp_path):
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    path = tmp_path / "invalid-format.bin"
    formats = bytearray(qsrt.EXPERTS)
    formats[17] = 0xD0
    _write_tp12_slab_prefix(path, layer=1, layout=layout, formats=bytes(formats))

    with pytest.raises(ValueError, match="invalid code"):
        qsrt.read_tp12_rank_payload(path, layer=1, rank=0)


def test_tp12_slab_reader_rejects_nonzero_format_padding(tmp_path):
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    path = tmp_path / "format-padding.bin"
    _write_tp12_slab_prefix(path, layer=1, layout=layout, formats=bytes(qsrt.EXPERTS))
    descriptor = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(descriptor, b"\x01", qsrt.HEADER_BYTES + qsrt.EXPERTS)
    finally:
        os.close(descriptor)

    with pytest.raises(ValueError, match="format section is malformed"):
        qsrt.read_tp12_rank_payload(path, layer=1, rank=0)


def test_tp12_slab_reader_rejects_expected_bit_map_drift(tmp_path):
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    path = tmp_path / "bit-map.bin"
    _write_tp12_slab_prefix(path, layer=1, layout=layout, formats=bytes(qsrt.EXPERTS))

    with pytest.raises(ValueError, match="bit map disagrees"):
        qsrt.read_tp12_rank_payload(
            path,
            layer=1,
            rank=0,
            expected_bits=[4] * qsrt.EXPERTS,
            selected_experts=[],
        )


@pytest.mark.parametrize("selected", [[3, 2], [2, 2], [-1], [qsrt.EXPERTS]])
def test_tp12_slab_reader_rejects_malformed_selected_experts(tmp_path, selected):
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    path = tmp_path / "selected-experts.bin"
    _write_tp12_slab_prefix(path, layer=1, layout=layout, formats=bytes(qsrt.EXPERTS))

    with pytest.raises(ValueError, match="sorted unique IDs"):
        qsrt.read_tp12_rank_payload(path, layer=1, rank=0, selected_experts=selected)


def test_tp12_slab_reader_rejects_nonfinite_shared_scales(tmp_path):
    layout = qsrt.TP12SlabLayout(qsrt.EXPERTS, 0)
    path = tmp_path / "nonfinite-shared.bin"
    shared = torch.zeros((3, qsrt.HIDDEN), dtype=torch.float16)
    shared[0, 0] = torch.nan
    _write_tp12_slab_prefix(
        path,
        layer=1,
        layout=layout,
        formats=bytes(qsrt.EXPERTS),
        shared=shared,
    )

    with pytest.raises(ValueError, match="shared scales contain non-finite"):
        qsrt.read_tp12_rank_payload(path, layer=1, rank=0, selected_experts=[])


def test_tp12_slab_reader_rejects_nonzero_rank_scale_padding(tmp_path):
    layout = qsrt.TP12SlabLayout(1, qsrt.EXPERTS - 1)
    path = tmp_path / "rank-scale-padding.bin"
    formats = bytes([0]) + bytes([qsrt.FORMAT_MXFP4]) * (qsrt.EXPERTS - 1)
    _write_tp12_slab_prefix(path, layer=1, layout=layout, formats=formats)
    descriptor = os.open(path, os.O_WRONLY)
    try:
        scale_padding = (
            layout.rank_offset(0)
            + layout.rank_trellis_bytes
            + layout.rank_scale_payload_bytes
        )
        os.pwrite(descriptor, b"\x01", scale_padding)
    finally:
        os.close(descriptor)

    with pytest.raises(ValueError, match="rank-scale section"):
        qsrt.read_tp12_rank_payload(path, layer=1, rank=0, selected_experts=[])
