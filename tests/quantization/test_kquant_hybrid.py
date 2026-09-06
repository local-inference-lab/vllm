# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.kquant_hybrid import (
    KQuantHybridConfig,
    _b12x_tiles_for_geometry,
    _is_dense_layer_ignored,
    _preplanned_w4a16_launches,
    _read_hybrid_keys,
    _require_rank_local_kept_kernel,
    _stack_exl3_intermediate_rotations,
    _w4a16_preplanned_launch_enabled,
)
from vllm.model_executor.layers.quantization.kquant_qsrt_atoms_v2 import (
    COUPLED_H308_ATOM_SLAB_BYTES,
    COUPLED_H308_PROFILE,
    PROFILE,
    PURE_K2_PROFILE,
    _atom_slot_stride_for_profile,
    _balanced_pure_k2_atom_partition,
    _coupled_h308_pair_for_rank,
    _coupled_pair_extent,
)
from vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid import (
    NvFp4Nf3HybridConfig,
)


def _base_config(**updates):
    config = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "hybrid_bit_map": {"1": [4, 3]},
        "kept_format": "mxfp4_e8m0k32",
    }
    config.update(updates)
    return config


def _qsrt_descriptor(**updates):
    descriptor = {
        "schema": "kquant_kimi_k3_qsrt_atoms_v1",
        "storage_format": "qsrt_atoms_v1",
        "encoding": "qsrt_sqg_e4m3",
        "codebook": "sqg_xor_cheb_t12",
        "artifact_manifest": "qsrt-manifest.json",
    }
    descriptor.update(updates)
    return descriptor


def test_pure_k2_tp10_atom_partition_preserves_coupled_blocks() -> None:
    extents = [_balanced_pure_k2_atom_partition(10, rank) for rank in range(10)]

    assert extents == [
        (0, 12),
        (12, 12),
        (24, 8),
        (32, 8),
        (40, 8),
        (48, 12),
        (60, 12),
        (72, 8),
        (80, 8),
        (88, 8),
    ]
    assert sum(count for _first, count in extents) == 96
    assert all(first % 4 == 0 and count % 4 == 0 for first, count in extents)
    assert all(not (first < 48 < first + count) for first, count in extents)


@pytest.mark.parametrize(
    "raw",
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
def test_reads_hybrid_checkpoint_without_claiming_legacy_nf3(raw) -> None:
    bit_map, kept_format = _read_hybrid_keys(raw)
    assert bit_map == {"0": [4, 3]}
    assert kept_format == "mxfp4_e8m0k32"
    assert KQuantHybridConfig.override_quantization_method(raw, None) is None
    assert NvFp4Nf3HybridConfig.override_quantization_method(raw, None) == (
        "nvfp4_nf3_hybrid"
    )
    assert KQuantHybridConfig.override_quantization_method(raw, "fp8") is None


def test_detects_qsrt_checkpoint() -> None:
    raw = _base_config(
        demoted_format="qsrt_sqg_e4m3",
        qsrt=_qsrt_descriptor(),
    )
    assert KQuantHybridConfig.override_quantization_method(raw, None) == (
        "kquant_hybrid"
    )
    assert KQuantHybridConfig.override_quantization_method(raw, "fp8") is None


def test_config_registration_and_generic_exl3_default() -> None:
    assert get_quantization_config("kquant_hybrid") is KQuantHybridConfig
    config = KQuantHybridConfig.from_config(_base_config())
    assert config.hybrid_bit_map == {"1": [4, 3]}
    assert config.kept_format == "mxfp4_e8m0k32"
    assert config.demoted_format == "exl3_3"
    assert config.kept_storage == "inline-mxfp4"


def test_config_accepts_tp_independent_qsrt() -> None:
    descriptor = _qsrt_descriptor()
    config = KQuantHybridConfig.from_config(
        _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
    )
    assert config.demoted_format == "qsrt_sqg_e4m3"
    assert config.kept_storage == "x4t"
    assert config.qsrt == descriptor
    assert config.trellis_codebook == "sqg_xor_cheb_t12"


def test_config_accepts_atoms_v2_revision() -> None:
    descriptor = _qsrt_descriptor(
        schema="qsrt_kimi_k3_qsrt_atoms_v2",
        storage_format="qsrt_atoms_v2",
    )
    config = KQuantHybridConfig.from_config(
        _base_config(demoted_format="qsrt_sqg_e4m3", qsrt=descriptor)
    )
    assert config.qsrt == {**descriptor, "profile": "k3x22_k4x2"}


def test_config_accepts_coupled_pure_k2_atoms_v2_profile() -> None:
    descriptor = _qsrt_descriptor(
        schema="qsrt_kimi_k3_qsrt_atoms_v2",
        storage_format="qsrt_atoms_v2",
        profile="k2_coupled_h512_h128",
    )
    config = KQuantHybridConfig.from_config(
        _base_config(
            hybrid_bit_map={"1": [2, 2]},
            demoted_format="qsrt_sqg_e4m3",
            qsrt=descriptor,
        )
    )
    assert config.hybrid_bit_map == {"1": [2, 2]}
    assert config.qsrt == descriptor


def test_atoms_v2_profiles_have_canonical_row_strides() -> None:
    assert _atom_slot_stride_for_profile(PROFILE) == 119005184
    assert _atom_slot_stride_for_profile(PURE_K2_PROFILE) == 77242368
    with pytest.raises(ValueError, match="unsupported QSRT atoms-v2 profile"):
        _atom_slot_stride_for_profile("unknown")


def test_config_accepts_coupled_high_rate_atoms_v2_profile() -> None:
    descriptor = _qsrt_descriptor(
        schema="qsrt_kimi_k3_qsrt_atoms_v2",
        storage_format="qsrt_atoms_v2",
        profile=COUPLED_H308_PROFILE,
    )
    config = KQuantHybridConfig.from_config(
        _base_config(
            hybrid_bit_map={"1": [3] * 896},
            demoted_format="qsrt_sqg_e4m3",
            qsrt=descriptor,
        )
    )
    assert config.qsrt == descriptor


def test_coupled_high_rate_atom_extents_cover_flat_slab() -> None:
    cursor = 0
    for pair in range(12):
        begin, extent, row_bytes = _coupled_pair_extent(pair)
        assert begin == cursor
        assert extent == 8 * row_bytes
        assert row_bytes % 4096 == 0
        cursor += extent
    assert cursor == COUPLED_H308_ATOM_SLAB_BYTES


def test_coupled_high_rate_rank_assignment_balances_physical_pairs() -> None:
    assignments = [
        [_coupled_h308_pair_for_rank(layer, 12, rank) for rank in range(12)]
        for layer in range(1, 93)
    ]

    assert all(sorted(layer) == list(range(12)) for layer in assignments)
    for rank in range(12):
        high_rate_layers = sum(
            assignments[layer][rank] in (5, 11) for layer in range(92)
        )
        assert high_rate_layers in (15, 16)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"demoted_format": "obsolete_private"}, "unsupported demoted_format"),
        (
            {"demoted_format": "qsrt_sqg_e4m3"},
            "requires a qsrt format descriptor",
        ),
        (
            {
                "demoted_format": "qsrt_sqg_e4m3",
                "qsrt": _qsrt_descriptor(storage_format="legacy-v1"),
            },
            "storage_format",
        ),
    ],
)
def test_config_rejects_obsolete_or_noncanonical_secondary_formats(
    updates, message
) -> None:
    with pytest.raises(ValueError, match=message):
        KQuantHybridConfig.from_config(_base_config(**updates))


def test_exl3_rotation_bundle_follows_b12x_projection_order() -> None:
    w13 = torch.arange(2 * 2 * 4, dtype=torch.float16).reshape(2, 2, 4)
    w2 = (100 + torch.arange(2 * 4, dtype=torch.float16)).reshape(2, 4)
    result = _stack_exl3_intermediate_rotations(w13, w2)
    expected = torch.cat((w13[:, 0], w13[:, 1], w2), dim=1)
    torch.testing.assert_close(result, expected)


def test_hybrid_kept_kernel_must_return_rank_local_partial() -> None:
    _require_rank_local_kept_kernel(SimpleNamespace(output_is_reduced=lambda: False))
    with pytest.raises(RuntimeError, match="unreduced rank-local partial"):
        _require_rank_local_kept_kernel(SimpleNamespace(output_is_reduced=lambda: True))


def test_b12x_tile_selection_is_geometry_driven() -> None:
    assert _b12x_tiles_for_geometry(3584, 3072) == (64, 256, 64, 256)
    assert _b12x_tiles_for_geometry(4096, 1536) == (64, 256, 64, 256)
    with pytest.raises(ValueError, match="no fixed b12x tile"):
        _b12x_tiles_for_geometry(3585, 3072)


@pytest.mark.parametrize(
    ("prefix", "ignored", "expected"),
    [
        ("model.layers.1.self_attn.q_proj", ["q_proj"], True),
        ("model.layers.1.self_attn.q_b_proj", ["b_proj"], False),
        ("model.layers.1.self_attn.q_proj", ["model.layers.1.self_attn.q_proj"], True),
    ],
)
def test_dense_short_exclusions_match_path_components(
    prefix, ignored, expected
) -> None:
    assert _is_dense_layer_ignored(prefix, ignored, {}) is expected


def test_dense_kda_precision_groups_resolve_fused_children() -> None:
    mapping = {
        "in_proj_qkv": ["q_proj", "k_proj", "v_proj"],
        "in_proj_gfab": ["g_proj", "f_a_proj", "b_proj"],
    }
    ignored = ["g_proj", "f_a_proj", "b_proj"]

    assert not _is_dense_layer_ignored(
        "model.layers.1.linear_attn.in_proj_qkv", ignored, mapping
    )
    assert _is_dense_layer_ignored(
        "model.layers.1.linear_attn.in_proj_gfab", ignored, mapping
    )


def test_preplanned_w4a16_launches_pick_capacity_and_matching_sum() -> None:
    """The prefill binding must carry the plan's capacity fused launch and
    the top-k sum variant for int32 mapped routes; other variants are for
    other route contracts."""
    fused_4608 = SimpleNamespace(size_m=4608, moe_block_size=48)
    sums = {
        (torch.int32, False): object(),
        (torch.int32, True): object(),
        (torch.int64, False): object(),
        (torch.int64, True): object(),
    }
    plan = SimpleNamespace(
        _prewarmed_fused_launches=((4608, fused_4608),),
        _prewarmed_topk_sum_launches=tuple(
            (dtype, mapped, launch) for (dtype, mapped), launch in sums.items()
        ),
    )

    resolved = _preplanned_w4a16_launches(
        plan, route_ids_dtype=torch.int32, mapped=True
    )

    assert resolved is not None
    assert resolved[0] is fused_4608
    assert resolved[1] is sums[(torch.int32, True)]
    unmapped = _preplanned_w4a16_launches(
        plan, route_ids_dtype=torch.int64, mapped=False
    )
    assert unmapped is not None and unmapped[1] is sums[(torch.int64, False)]


def test_preplanned_w4a16_launches_fall_back_without_prewarm() -> None:
    """A plan without prewarmed launches (or without the needed top-k sum
    variant) keeps the lazy per-call resolution."""
    empty = SimpleNamespace(
        _prewarmed_fused_launches=(), _prewarmed_topk_sum_launches=()
    )
    assert (
        _preplanned_w4a16_launches(empty, route_ids_dtype=torch.int32, mapped=True)
        is None
    )

    no_variant = SimpleNamespace(
        _prewarmed_fused_launches=((4608, object()),),
        _prewarmed_topk_sum_launches=((torch.int64, False, object()),),
    )
    assert (
        _preplanned_w4a16_launches(no_variant, route_ids_dtype=torch.int32, mapped=True)
        is None
    )
    assert (
        _preplanned_w4a16_launches(object(), route_ids_dtype=torch.int32, mapped=True)
        is None
    )


def test_preplanned_w4a16_launch_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_KQUANT_W4A16_PREPLANNED_LAUNCH", raising=False)
    assert _w4a16_preplanned_launch_enabled()
    monkeypatch.setenv("VLLM_KQUANT_W4A16_PREPLANNED_LAUNCH", "0")
    assert not _w4a16_preplanned_launch_enabled()


def test_b12x_w4a16_fused_moe_compile_key_is_row_count_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compiled W4A16 fused MoE kernel is keyed without the row count
    (except the single-token specialization), so the launch compiled for the
    scheduler capacity at boot is the one every prefill tail resolves to.

    Runs against the b12x planner on CPU with the kernel cache stubbed to
    always hit; only the derived cache keys are inspected.
    """
    kmod = pytest.importorskip("b12x.moe._shared.kernels.w4a16.kernel")

    class _AlwaysHit(dict):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[tuple] = []

        def get(self, key, default=None):
            self.seen.append(key)
            return placeholder

    placeholder = kmod.W4A16FusedMoeCompileResult(
        compiled=None,
        size_m=1,
        hidden_size=3584,
        intermediate_size=384,
        num_experts=896,
        top_k=16,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        element_dtype="fp16",
        fast_math=True,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        fc1_tile_n=128,
        fc1_tile_k=128,
        fc2_tile_n=128,
        fc2_tile_k=128,
        moe_block_size=48,
        max_m_blocks=1,
        blocks_per_sm=1,
    )
    cache = _AlwaysHit()
    monkeypatch.setattr(kmod, "_FUSED_CACHE", cache)
    monkeypatch.delenv("B12X_W4A16_SMALL_M_SPLITK", raising=False)

    def compile_for(size_m: int) -> None:
        # Kimi-K3 TP9 rank-0 layer with the 384-wide expert extent, the
        # served route block (48) and pinned 128x128 tiles.
        kmod.compile_w4a16_fused_moe(
            size_m=size_m,
            hidden_size=3584,
            intermediate_size=384,
            num_experts=896,
            top_k=16,
            activation="situ",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=48,
            max_m_blocks=(size_m * 16 + 896 * 47 + 47) // 48,
            element_dtype="fp16",
            fast_math=True,
            sms=188,
            max_shared_mem=101_376,
            weight_layout="trellis3_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis3_t256_proj",
            trellis_bits=2,
            force_tile_config=(128, 128, 128, 128),
            intermediate_rotation=True,
            full_rotation=True,
            coupled_hadamard=True,
            rotation_input_dtype="bf16",
        )

    keys: dict[int, tuple] = {}
    for size_m in (2, 257, 830, 1536, 4608):
        cache.seen.clear()
        compile_for(size_m)
        (keys[size_m],) = cache.seen
    assert len(set(keys.values())) == 1
    cache.seen.clear()
    compile_for(1)
    (single,) = cache.seen
    assert single != keys[4608]
