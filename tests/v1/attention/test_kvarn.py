# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import (
    KVarNConfig,
    KVarNMLAConfig,
)
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.attention.backends.kvarn_attn import (
    KVarNAttentionBackend,
    KVarNAttentionImpl,
    _build_hadamard,
)
from vllm.v1.attention.backends.mla.kvarn_mla_state import (
    KVarNMLALiveBlockTracker,
    KVarNMLAStateManager,
)
from vllm.v1.attention.ops.kvarn_decode import _unpack_lowbit, kvarn_hadamard
from vllm.v1.attention.ops.kvarn_store import (
    _pack_lowbit,
    kvarn_store_tile_k_batch_from_sinkhorn,
)
from vllm.v1.attention.ops.triton_kvarn_decode import adaptive_num_kv_splits
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    KVarNFullAttentionSpec,
    KVarNSlidingWindowSpec,
)


def test_k5_uses_dense_little_endian_bitstream() -> None:
    codes = torch.arange(8, dtype=torch.uint8)

    packed = _pack_lowbit(codes, bits=5)

    assert packed.tolist() == [32, 136, 65, 138, 57]
    assert torch.equal(_unpack_lowbit(packed, 8, bits=5), codes)


@pytest.mark.parametrize(
    ("bits", "codes", "expected"),
    (
        (2, torch.arange(8, dtype=torch.uint8) % 4, [228, 228]),
        (4, torch.arange(8, dtype=torch.uint8), [16, 50, 84, 118]),
    ),
)
def test_byte_aligned_lowbit_packers(
    bits: int, codes: torch.Tensor, expected: list[int]
) -> None:
    packed = _pack_lowbit(codes, bits=bits)

    assert packed.tolist() == expected
    assert torch.equal(_unpack_lowbit(packed, len(codes), bits=bits), codes)


def _reconstruct_batched_k(record: dict[str, torch.Tensor], group: int) -> torch.Tensor:
    q = _unpack_lowbit(record["q_packed_uint8"], group, bits=5).float()
    return (
        q * record["s_col_K"].float().unsqueeze(-1)
        + record["zp_K"].float().unsqueeze(-1)
    ) * record["s_row_K"].float().unsqueeze(-2)


def test_affine_refit_reduces_original_space_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(7)
    balanced = torch.randn(2, 16, 64, generator=generator)
    s_col = torch.linspace(0.05, 4.0, 64).repeat(2, 1)
    s_row = torch.linspace(0.25, 2.0, 16).repeat(2, 1)
    original = balanced * s_row.unsqueeze(-1) * s_col.unsqueeze(-2)

    monkeypatch.setenv("KVARN_AFFINE_REFIT", "0")
    rtn = kvarn_store_tile_k_batch_from_sinkhorn(balanced, s_col, s_row, bits=5)
    monkeypatch.setenv("KVARN_AFFINE_REFIT", "1")
    refit = kvarn_store_tile_k_batch_from_sinkhorn(balanced, s_col, s_row, bits=5)

    rtn_error = (_reconstruct_batched_k(rtn, 64) - original).square().mean()
    refit_error = (_reconstruct_batched_k(refit, 64) - original).square().mean()
    assert refit_error < rtn_error


def test_affine_refit_defaults_on_for_five_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(13)
    balanced = torch.randn(2, 16, 64, generator=generator)
    s_col = torch.linspace(0.1, 2.0, 64).repeat(2, 1)
    s_row = torch.linspace(0.5, 1.5, 16).repeat(2, 1)

    monkeypatch.delenv("KVARN_AFFINE_REFIT", raising=False)
    default = kvarn_store_tile_k_batch_from_sinkhorn(balanced, s_col, s_row, bits=5)
    monkeypatch.setenv("KVARN_AFFINE_REFIT", "1")
    explicit = kvarn_store_tile_k_batch_from_sinkhorn(balanced, s_col, s_row, bits=5)

    for name in default:
        assert torch.equal(default[name], explicit[name])


def test_k5_layout_and_exact_pool_storage_sizes() -> None:
    standard = KVarNConfig(
        head_dim=128,
        key_bits=5,
        value_bits=5,
        group=64,
    )
    mla = KVarNMLAConfig()

    assert standard.tile_bytes_aligned == 11392
    assert standard.tile_bytes_aligned // standard.group == 178
    assert standard._slot_bytes_per_layer(num_kv_heads=2) == 65536
    assert replace(standard, tail_dtype="fp8")._slot_bytes_per_layer(2) == 33280
    assert mla.tile_bytes == 30848
    assert mla.bytes_per_token == 482
    assert mla.pool_slot_bytes == 40960
    assert standard.resident_blocks_per_seq == 19
    assert mla.resident_blocks_per_seq == 3
    assert mla.pool_slots(10, 20) == 49
    assert mla.pool_slots(10, 640) == 58
    assert mla.pool_slots(10, 20, max_rollback_tokens=8) == 61


def test_standard_precision_tail_defaults_to_float16() -> None:
    assert KVarNConfig(head_dim=128).tail_dtype == "float16"


def test_k5v5_public_default_disables_retained_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_TAIL_TOKENS", raising=False)

    k5v5 = KVarNConfig.from_cache_dtype("kvarn_k5v5_g64", head_dim=128)
    k4v4 = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)

    assert k5v5.precision_tail_tokens == 0
    assert k5v5.resident_blocks_per_seq == 2
    assert k4v4.precision_tail_tokens == 1024


def test_mla_config_ignores_standard_tail_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_TAIL_TOKENS", "1024")
    monkeypatch.setenv("KVARN_SINK_TOKENS", "1024")

    config = KVarNMLAConfig.from_cache_dtype("kvarn_mla_k5_g64")

    assert config.resident_blocks_per_seq == 3
    assert config.boundary_tokens == 128
    assert not hasattr(config, "precision_tail_tokens")


def test_only_standard_kvarn_forces_v1_model_runner() -> None:
    from vllm.config import VllmConfig

    config = VllmConfig()
    config.cache_config.cache_dtype = "kvarn_k5v5_g64"
    assert "KVarN KV cache" in config._get_v2_model_runner_unsupported_features()

    config.cache_config.cache_dtype = "kvarn_mla_k5_g64"
    assert "KVarN KV cache" not in config._get_v2_model_runner_unsupported_features()


def test_k5v5_uses_128_splits_for_long_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_NUM_KV_SPLITS", raising=False)
    k5v5 = KVarNConfig.from_cache_dtype("kvarn_k5v5_g64", head_dim=128)
    k4v4 = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)

    assert adaptive_num_kv_splits(256, k5v5) == 32
    assert adaptive_num_kv_splits(257, k5v5) == 128
    assert adaptive_num_kv_splits(257, k4v4) == 64


def test_float16_precision_tail_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_TAIL_DTYPE", "float16")

    assert KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", 128).tail_dtype == "float16"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Hadacore")
@pytest.mark.parametrize("rows", (1, 3))
def test_hadacore_uses_padded_return_tensor(rows: int) -> None:
    device = torch.device("cuda")
    hadamard = _build_hadamard(128, device)
    value = torch.randn(rows, 128, device=device, dtype=torch.float16)

    expected = value.float() @ hadamard
    actual = kvarn_hadamard(value, hadamard).float()

    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_k4v2_is_a_public_standard_cache_dtype() -> None:
    cache_dtype = "kvarn_k4v2_g128"
    config = KVarNConfig.from_cache_dtype(cache_dtype, head_dim=128)

    assert (config.key_bits, config.value_bits, config.group) == (4, 2, 128)
    assert cache_dtype in KVarNAttentionBackend.supported_kv_cache_dtypes
    assert 128 in KVarNAttentionBackend.get_supported_kernel_block_sizes()
    assert get_kv_cache_torch_dtype(cache_dtype) == torch.uint8


def test_k4v4_is_a_public_standard_cache_dtype() -> None:
    cache_dtype = "kvarn_k4v4_g128"
    config = KVarNConfig.from_cache_dtype(cache_dtype, head_dim=128)

    assert (config.key_bits, config.value_bits, config.group) == (4, 4, 128)
    assert cache_dtype in KVarNAttentionBackend.supported_kv_cache_dtypes
    assert 128 in KVarNAttentionBackend.get_supported_kernel_block_sizes()
    assert get_kv_cache_torch_dtype(cache_dtype) == torch.uint8


def test_cutedsl_decode_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_CUTEDSL", raising=False)
    impl = object.__new__(KVarNAttentionImpl)

    assert not impl._can_use_cutedsl_decode(torch.device("cuda"))


def test_cutedsl_decode_rejects_k5_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_CUTEDSL", "1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    impl = object.__new__(KVarNAttentionImpl)
    impl.kvarn_config = KVarNConfig.from_cache_dtype("kvarn_k5v5_g64", 128)
    impl._activation_dtype = torch.float16
    impl.num_heads = 32
    impl.num_kv_heads = 8
    impl.sliding_window = 0
    impl._max_model_len = 32768
    impl.scale = 128**-0.5

    assert not impl._can_use_cutedsl_decode(torch.device("cuda"))


def test_kvarn_page_unification_packs_multiple_group_tiles() -> None:
    full = KVarNFullAttentionSpec(
        block_size=64,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
        tile_size=11392,
    )
    sliding = KVarNSlidingWindowSpec(
        block_size=64,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.uint8,
        sliding_window=4096,
        tile_size=11392,
    )

    unified = unify_kv_cache_spec_page_size({"full": full, "sliding": sliding})

    assert unified["full"].block_size == 128
    assert unified["sliding"].block_size == 64
    assert unified["full"].page_size_bytes == unified["sliding"].page_size_bytes


def test_mla_ownership_metadata_survives_unpadding_and_ubatching() -> None:
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.worker.ubatch_utils import UBatchSlice, split_attn_metadata

    block_fills = {0: 64, 1: 64, 3: 32}
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([10, 20], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens,
        _num_computed_tokens_cpu=torch.tensor([9, 19], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=20,
        block_table_tensor=torch.tensor([[0], [1]], dtype=torch.int32),
        slot_mapping=torch.tensor([9, 19], dtype=torch.int64),
        seq_lens_cpu_upper_bound=seq_lens,
        kvarn_mla_block_fills=block_fills,
    )

    unpadded = metadata.unpadded(1, 1)
    ubatches = split_attn_metadata(
        [
            UBatchSlice(slice(0, 1), slice(0, 1)),
            UBatchSlice(slice(1, 2), slice(1, 2)),
        ],
        metadata,
    )

    assert unpadded.kvarn_mla_block_fills is block_fills
    assert all(ubatch.kvarn_mla_block_fills is block_fills for ubatch in ubatches)


class _FakeMLAImpl:
    layer_name = "layer.0"
    _is_kvarn_mla = True
    _kvarn_group_key = None
    device = torch.device("cpu")
    _kvarn_pool_size = 32

    def __init__(self) -> None:
        self.flushed: list[int] = []

    def _flush_kvarn_mla_blocks(
        self, block_ids: torch.Tensor, pool_slots: torch.Tensor
    ) -> None:
        self.flushed.extend(block_ids.tolist())


def test_mla_preserves_bounded_live_blocks_across_unscheduled_steps() -> None:
    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()
    config = KVarNMLAConfig()
    impl = _FakeMLAImpl()
    KVarNMLAStateManager.register(impl)

    def prepare(block_fills: dict[int, int]) -> dict[int, int]:
        metadata = SimpleNamespace(kvarn_mla_block_fills=block_fills)
        KVarNMLAStateManager.prepare_step(
            ("layer.0",), ["layer.0"], metadata, config, dcp_world_size=1
        )
        return dict(KVarNMLAStateManager._groups[("layer.0",)].mapping)

    live_a = {0: 64, 1: 64, 3: 32}
    live_b = {4: 64, 5: 64, 7: 32}
    mapping_a = prepare(live_a)
    mapping_b = prepare(live_a | live_b)
    mapping_a_returned = prepare(live_a | live_b)

    assert set(mapping_a) == set(live_a)
    for block_id, slot in mapping_a.items():
        assert mapping_b[block_id] == slot
        assert mapping_a_returned[block_id] == slot
    assert impl.flushed == []

    prepare(live_b)
    assert set(KVarNMLAStateManager._groups[("layer.0",)].mapping) == set(live_b)
    assert impl.flushed == [0, 1]

    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()


def test_mla_prefill_tracks_every_written_block_without_device_readback() -> None:
    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()
    config = KVarNMLAConfig()
    impl = _FakeMLAImpl()
    KVarNMLAStateManager.register(impl)

    def prepare(block_fills: dict[int, int]) -> None:
        metadata = SimpleNamespace(kvarn_mla_block_fills=block_fills)
        KVarNMLAStateManager.prepare_step(
            ("layer.0",), ["layer.0"], metadata, config, dcp_world_size=1
        )

    prepare({0: 64, 1: 64, 2: 64, 3: 32})
    state = KVarNMLAStateManager._groups[("layer.0",)]
    assert set(state.mapping) == {0, 1, 2, 3}
    assert state.block_fill[2] == 64

    prepare({0: 64, 1: 64, 3: 64})
    assert set(state.mapping) == {0, 1, 3}
    assert impl.flushed == [2]

    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()


def test_mla_live_tracker_is_bounded_and_generation_aware() -> None:
    tracker = KVarNMLALiveBlockTracker({0: (64, 128, 1, 1, 0, 1)})
    request_a = SimpleNamespace(
        block_ids=([0, 1, 2, 3],),
        num_computed_tokens=0,
    )
    request_b = SimpleNamespace(
        block_ids=([4, 5, 6, 7],),
        num_computed_tokens=0,
    )
    requests = {"a": request_a, "b": request_b}

    tracker.update(requests, {"a": 224}, (), ())
    assert tracker.block_fills(0) == {0: 64, 1: 64, 2: 64, 3: 32}
    tracker.update(requests, {"b": 224}, (), ())
    assert tracker.block_fills(0) == {
        0: 64,
        1: 64,
        3: 32,
        4: 64,
        5: 64,
        6: 64,
        7: 32,
    }

    tracker.update(requests, {}, (), ("a",))
    assert tracker.block_fills(0) == {4: 64, 5: 64, 7: 32}

    request_a.block_ids = ([8, 9, 10, 11],)
    request_a.num_computed_tokens = 224
    tracker.update(requests, {"a": 1}, (), ())
    assert tracker.block_fills(0) == {
        4: 64,
        5: 64,
        7: 32,
        8: 64,
        9: 64,
        11: 33,
    }


def test_mla_live_tracker_unions_dbo_slices_and_shared_prefix() -> None:
    tracker = KVarNMLALiveBlockTracker({0: (64, 128, 1, 1, 0, 1)})
    request_a = SimpleNamespace(
        block_ids=([0, 1, 2, 3],),
        num_computed_tokens=0,
    )
    request_b = SimpleNamespace(
        block_ids=([0, 1, 6, 7],),
        num_computed_tokens=0,
    )
    requests = {"a": request_a, "b": request_b}

    tracker.update(requests, {"a": 224, "b": 224}, (), ())
    assert tracker.block_fills(0) == {
        0: 64,
        1: 64,
        2: 64,
        3: 32,
        6: 64,
        7: 32,
    }

    tracker.update(requests, {}, ("a",), ())
    assert tracker.block_fills(0) == {0: 64, 1: 64, 7: 32}


def test_mla_async_spec_rollback_keeps_possible_boundary_blocks() -> None:
    tracker = KVarNMLALiveBlockTracker({0: (64, 128, 1, 1, 0, 1)})
    request = SimpleNamespace(
        block_ids=([0, 1, 2, 3],),
        num_computed_tokens=194,
    )
    requests = {"a": request}

    tracker.update(
        requests,
        {"a": 2},
        (),
        (),
        rollback_tokens={"a": 4},
    )
    assert tracker.block_fills(0) == {
        0: 64,
        1: 64,
        2: 64,
        3: None,
    }

    tracker.resolve_async("a", request, actual_end_tokens=193)
    tracker.update(requests, {}, (), ())
    assert tracker.block_fills(0) == {
        0: 64,
        1: 64,
        2: 64,
        3: 1,
    }

    tracker.update(requests, {}, (), ())
    assert tracker.block_fills(0) == {0: 64, 1: 64, 3: 1}


def test_mla_unknown_rollback_block_retires_without_flush() -> None:
    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()
    config = KVarNMLAConfig()
    impl = _FakeMLAImpl()
    KVarNMLAStateManager.register(impl)
    group_key = ("layer.0",)

    metadata = SimpleNamespace(kvarn_mla_block_fills={0: 64, 1: 64, 2: 64, 3: 64})
    KVarNMLAStateManager.prepare_step(
        group_key,
        ["layer.0"],
        metadata,
        config,
        dcp_world_size=1,
    )
    state = KVarNMLAStateManager._groups[group_key]
    assert state.block_fill[3] == 64

    metadata.kvarn_mla_block_fills[3] = None
    KVarNMLAStateManager.prepare_step(
        group_key,
        ["layer.0"],
        metadata,
        config,
        dcp_world_size=1,
    )
    assert 3 in state.mapping
    assert 3 not in state.block_fill

    metadata.kvarn_mla_block_fills = {0: 64, 1: 64, 2: 64}
    KVarNMLAStateManager.prepare_step(
        group_key,
        ["layer.0"],
        metadata,
        config,
        dcp_world_size=1,
    )
    assert 3 not in state.mapping
    assert 3 not in impl.flushed

    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()


def test_mla_live_tracker_maps_hybrid_dcp_blocks() -> None:
    tracker = KVarNMLALiveBlockTracker({0: (64, 128, 4, 2, 1, 1)})
    request = SimpleNamespace(
        block_ids=([10],),
        num_computed_tokens=0,
    )

    tracker.update({"request": request}, {"request": 225}, (), ())

    assert tracker.block_fills(0) == {40: 64, 41: 48}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mla_fused_serializer_matches_reference(monkeypatch) -> None:
    from vllm.v1.attention.ops.kvarn_mla import pack_kvarn_mla_blocks
    from vllm.v1.attention.ops.triton_kvarn_sinkhorn import (
        kvarn_sinkhorn_triton,
    )

    monkeypatch.delenv("KVARN_RTN_QUANTILE", raising=False)
    monkeypatch.setenv("KVARN_AFFINE_REFIT", "1")
    config = KVarNMLAConfig()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(19)
    latent_pool = torch.randn(
        1,
        config.group,
        config.latent_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).to(torch.float8_e4m3fn)
    rope_pool = torch.randn(
        1,
        config.group,
        config.rope_dim,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    latent_tiles = latent_pool.float().transpose(1, 2).contiguous()
    balanced, s_col, s_row = kvarn_sinkhorn_triton(
        latent_tiles,
        iterations=config.sinkhorn_iters,
    )
    packed = kvarn_store_tile_k_batch_from_sinkhorn(
        balanced,
        s_col,
        s_row,
        bits=config.bits,
    )
    expected = torch.zeros(
        1,
        config.tile_bytes,
        dtype=torch.uint8,
        device=device,
    )
    expected[:, : config.latent_packed_bytes].copy_(
        packed["q_packed_uint8"].reshape(1, -1)
    )
    expected[:, config.latent_s_col_offset : config.latent_zp_offset].copy_(
        packed["s_col_K"].contiguous().view(torch.uint8)
    )
    expected[:, config.latent_zp_offset : config.latent_s_row_offset].copy_(
        packed["zp_K"].contiguous().view(torch.uint8)
    )
    expected[:, config.latent_s_row_offset : config.rope_offset].copy_(
        packed["s_row_K"].contiguous().view(torch.uint8)
    )
    expected[:, config.rope_offset :].copy_(
        rope_pool.contiguous().view(torch.uint8).reshape(1, -1)
    )

    cache = torch.zeros(
        1,
        config.group,
        config.bytes_per_token,
        dtype=torch.uint8,
        device=device,
    )
    block_ids = torch.zeros(1, dtype=torch.long, device=device)
    pack_kvarn_mla_blocks(
        cache,
        latent_pool,
        rope_pool,
        block_ids,
        block_ids,
        config,
    )

    assert torch.equal(cache.view(torch.uint8).reshape(1, -1), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bf16_sparse_attention_writes_into_query_workspace() -> None:
    from vllm.v1.attention.ops.xpu_mla_sparse import (
        triton_bf16_mla_sparse_interface,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(29)
    q = torch.randn(
        1,
        8,
        576,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    kv = torch.randn(
        16,
        1,
        576,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    indices = torch.arange(16, dtype=torch.int32, device=device).view(1, 1, -1)
    expected, _, _ = triton_bf16_mla_sparse_interface(
        q,
        kv,
        indices,
        sm_scale=576**-0.5,
    )

    q_workspace = q.clone()
    out_workspace = q_workspace[..., :512]
    actual, lse, max_logits = triton_bf16_mla_sparse_interface(
        q_workspace,
        kv,
        indices,
        sm_scale=576**-0.5,
        out=out_workspace,
        return_lse=False,
        return_max_logits=False,
    )

    assert actual.data_ptr() == q_workspace.data_ptr()
    assert lse.numel() == 1
    assert max_logits.numel() == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mla_bounded_ownership_preserves_sink_and_current_payload() -> None:
    from vllm.v1.attention.ops.kvarn_mla import materialize_selected_kvarn_mla

    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()
    device = torch.device("cuda")
    config = KVarNMLAConfig()

    class PayloadImpl:
        layer_name = "layer.0"
        _is_kvarn_mla = True
        _kvarn_group_key = None

        def __init__(self) -> None:
            pool_size = config.pool_slots(2, 256)
            self.device = device
            self._kvarn_latent_pool = torch.empty(
                pool_size,
                config.group,
                config.latent_dim,
                device=device,
                dtype=torch.float8_e4m3fn,
            )
            self._kvarn_rope_pool = torch.empty(
                pool_size,
                config.group,
                config.rope_dim,
                device=device,
                dtype=torch.bfloat16,
            )
            self._kvarn_pool_size = pool_size

        def _flush_kvarn_mla_blocks(
            self, block_ids: torch.Tensor, pool_slots: torch.Tensor
        ) -> None:
            raise AssertionError("live exact blocks must not be flushed")

    impl = PayloadImpl()
    KVarNMLAStateManager.register(impl)
    cache = torch.zeros(
        8,
        config.group,
        config.bytes_per_token,
        device=device,
        dtype=torch.uint8,
    )

    def prepare(
        block_fills: dict[int, int],
    ) -> tuple[dict[int, int], torch.Tensor]:
        metadata = SimpleNamespace(kvarn_mla_block_fills=block_fills)
        group_key = ("layer.0",)
        KVarNMLAStateManager.prepare_step(
            group_key, ["layer.0"], metadata, config, dcp_world_size=1
        )
        mirror = KVarNMLAStateManager.ensure_mirror(group_key, device, 8)
        return dict(KVarNMLAStateManager._groups[group_key].mapping), mirror

    def materialize(mirror: torch.Tensor) -> torch.Tensor:
        selected = torch.cat(
            (
                torch.arange(0, 64, device=device, dtype=torch.int32),
                torch.arange(192, 224, device=device, dtype=torch.int32),
            )
        ).view(1, -1)
        dense = torch.empty(2, config.group, 576, device=device, dtype=torch.bfloat16)
        remapped = torch.empty_like(selected)
        materialize_selected_kvarn_mla(
            selected,
            cache,
            mirror,
            impl._kvarn_latent_pool,
            impl._kvarn_rope_pool,
            dense,
            remapped,
            config,
        )
        return dense.view(-1, 576).index_select(0, remapped.flatten())

    live_a = {0: 64, 1: 64, 3: 32}
    live_b = {4: 64, 5: 64, 7: 32}
    mapping_a, mirror = prepare(live_a)
    generator = torch.Generator(device=device).manual_seed(19)
    for block_id in live_a:
        slot = mapping_a[block_id]
        impl._kvarn_latent_pool[slot].copy_(
            torch.randn(
                64, 512, device=device, dtype=torch.bfloat16, generator=generator
            )
        )
        impl._kvarn_rope_pool[slot].copy_(
            torch.randn(
                64, 64, device=device, dtype=torch.bfloat16, generator=generator
            )
        )
    before = materialize(mirror)

    mapping_b, _ = prepare(live_a | live_b)
    for block_id in live_b:
        slot = mapping_b[block_id]
        impl._kvarn_latent_pool[slot].fill_(3)
        impl._kvarn_rope_pool[slot].fill_(3)
    mapping_returned, mirror = prepare(live_a | live_b)
    after = materialize(mirror)

    assert {block_id: mapping_returned[block_id] for block_id in live_a} == mapping_a
    assert torch.equal(after, before)

    KVarNMLAStateManager._impls.clear()
    KVarNMLAStateManager.reset_cache_bindings()


def test_mla_materialization_returns_bf16_sparse_working_set(monkeypatch) -> None:
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
    from vllm.v1.attention.ops import kvarn_mla

    config = KVarNMLAConfig()
    dense_cache = torch.empty(1, config.group, config.latent_dim + config.rope_dim)
    remapped = torch.empty(1, 2, dtype=torch.int32)
    impl = SimpleNamespace(
        _kvarn_config=config,
        _kvarn_block_to_slot=torch.zeros(1, dtype=torch.int32),
        _kvarn_remapped_indices=remapped,
        _kvarn_physical_slots=torch.arange(config.group, dtype=torch.int32),
        _decode_max_rows=1,
        _kvarn_latent_pool=torch.empty(1),
        _kvarn_rope_pool=torch.empty(1),
        _kvarn_dense_cache=dense_cache,
    )

    def fake_materialize(*args) -> None:
        args[5].fill_(1)
        args[6].copy_(torch.tensor([[0, 1]], dtype=torch.int32))

    monkeypatch.setattr(kvarn_mla, "materialize_selected_kvarn_mla", fake_materialize)

    cache, selected = B12xMLASparseImpl._materialize_kvarn_mla_cache(
        impl,
        torch.tensor([[3, 4]], dtype=torch.int32),
        SimpleNamespace(),
        torch.empty(1),
    )

    assert cache is dense_cache
    assert torch.equal(selected, torch.tensor([[0, 1]], dtype=torch.int32))
    assert torch.all(cache == 1)


def test_mla_prefill_materializes_the_physical_page_arena(monkeypatch) -> None:
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
    from vllm.v1.attention.ops import kvarn_mla

    config = KVarNMLAConfig()
    num_pages = 4
    page_rows = num_pages * config.group
    dense_cache = torch.empty(
        num_pages, config.group, config.latent_dim + config.rope_dim
    )
    remapped = torch.empty(2, 2, dtype=torch.int32)
    physical_slots = torch.arange(page_rows, dtype=torch.int32)
    impl = SimpleNamespace(
        _kvarn_config=config,
        _kvarn_block_to_slot=torch.zeros(num_pages, dtype=torch.int32),
        _kvarn_remapped_indices=remapped,
        _kvarn_physical_slots=physical_slots,
        _decode_max_rows=1,
        _kvarn_latent_pool=torch.empty(1),
        _kvarn_rope_pool=torch.empty(1),
        _kvarn_dense_cache=dense_cache,
    )
    captured: dict[str, torch.Tensor] = {}

    def fake_materialize_physical(*args) -> None:
        captured["physical_slots"] = args[0]
        args[6].fill_(1)
        args[7].copy_(torch.tensor([[1, 2], [65, 66]], dtype=torch.int32))

    monkeypatch.setattr(
        kvarn_mla,
        "materialize_physical_kvarn_mla",
        fake_materialize_physical,
    )

    cache, selected = B12xMLASparseImpl._materialize_kvarn_mla_cache(
        impl,
        torch.tensor([[3, 4], [7, 8]], dtype=torch.int32),
        SimpleNamespace(),
        torch.empty(num_pages),
    )

    assert cache is dense_cache
    assert captured["physical_slots"] is physical_slots
    assert torch.equal(selected, torch.tensor([[1, 2], [65, 66]]))


def test_mla_rejects_standard_kvarn_dtype() -> None:
    with pytest.raises(ValueError, match="kvarn_mla_k5_g64"):
        KVarNMLAConfig.from_cache_dtype("kvarn_k5v5_g64")
