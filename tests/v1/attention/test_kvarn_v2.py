# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNMLAConfig
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseImpl,
    _kvarn_mla_workspace_envelope,
)
from vllm.v1.attention.ops.kvarn_mla import (
    materialize_physical_kvarn_mla,
    materialize_selected_kvarn_mla,
    remap_kvarn_mla_physical_indices,
    scatter_kvarn_mla_exact,
)


def test_mla_config_has_fixed_page_and_pool_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_TAIL_DTYPE", "invalid-legacy-value")

    config = KVarNMLAConfig.from_cache_dtype("kvarn_mla_k5_g64")

    assert "tail_dtype" not in {field.name for field in fields(config)}
    assert not hasattr(config, "tail_slot_bytes")
    assert config.group == 64
    assert config.latent_dim == 512
    assert config.rope_dim == 64
    assert config.tile_bytes == 30_848
    assert config.bytes_per_token == 482
    assert config.pool_slot_bytes == 40_960
    assert config.pool_slot_bytes == config.group * (
        config.latent_dim + config.rope_dim * 2
    )


def test_mla_config_rejects_non_mla_kvarn_dtype() -> None:
    with pytest.raises(ValueError, match="kvarn_mla_k5_g64"):
        KVarNMLAConfig.from_cache_dtype("kvarn_k5v5_g64")


def test_mla_workspace_is_bounded_by_allocated_pages() -> None:
    envelope = _kvarn_mla_workspace_envelope(
        num_kv_pages=4_097,
        group_size=64,
        latent_dim=512,
        rope_dim=64,
        max_batched_tokens=4_096,
        max_active_rows=40,
        topk_tokens=2_048,
        boundary_blocks=8,
        rollback_blocks=9,
    )

    assert envelope.dense_rows == 262_208
    assert envelope.remap_elements == 8_388_608
    assert envelope.rotation_rows == 4_096
    assert envelope.physical_slot_rows == 262_208
    assert envelope.dense_bytes == 302_063_616
    assert envelope.total_bytes == 340_861_184
    assert envelope.dense_bytes != 2_416_508_928


def test_mla_workspace_uses_stable_physical_page_rows_and_resets() -> None:
    config = KVarNMLAConfig()

    class FakeImpl:
        _is_kvarn_mla = True
        device = torch.device("cpu")
        _kvarn_config = config
        _vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=8,
                max_num_seqs=1,
                async_scheduling=False,
            ),
            speculative_config=None,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(index_topk=4),
            ),
        )
        _max_batched = 8
        _decode_max_rows = 1
        topk_tokens = 4
        _kvarn_boundary_blocks = 1
        _kvarn_rollback_blocks = 0
        _kvarn_dense_cache = None
        _kvarn_remapped_indices = None
        _kvarn_rotated_scratch = None
        _kvarn_physical_slots = None

    impl = FakeImpl()
    B12xMLASparseImpl._kvarn_instances.add(impl)
    B12xMLASparseImpl.initialize_kvarn_workspaces(3, impl.device)

    assert impl._kvarn_dense_cache is not None
    assert impl._kvarn_remapped_indices is not None
    assert impl._kvarn_rotated_scratch is not None
    assert impl._kvarn_physical_slots is not None
    assert impl._kvarn_dense_cache.shape == (3, 64, 576)
    assert impl._kvarn_remapped_indices.shape == (8, 4)
    assert impl._kvarn_rotated_scratch.shape == (8, 512)
    torch.testing.assert_close(
        impl._kvarn_physical_slots,
        torch.arange(192, dtype=torch.int32),
    )
    physical_ptr = impl._kvarn_physical_slots.data_ptr()
    B12xMLASparseImpl.initialize_kvarn_workspaces(3, impl.device)
    assert impl._kvarn_physical_slots.data_ptr() == physical_ptr

    B12xMLASparseImpl.reset_kv_cache_binding_state()
    assert impl._kvarn_dense_cache is None
    assert impl._kvarn_remapped_indices is None
    assert impl._kvarn_rotated_scratch is None
    assert impl._kvarn_physical_slots is None
    assert not B12xMLASparseImpl._kvarn_shared_dense
    assert not B12xMLASparseImpl._kvarn_shared_physical_slots


def test_mla_operation_workspaces_reject_invalid_shapes_and_types() -> None:
    config = KVarNMLAConfig()
    cache = torch.empty(1, 64, config.bytes_per_token, dtype=torch.uint8)
    block_to_slot = torch.zeros(1, dtype=torch.int32)
    latent_pool = torch.empty(1, 64, 512, dtype=torch.float8_e4m3fn)
    rope_pool = torch.empty(1, 64, 64, dtype=torch.bfloat16)
    output = torch.empty(1, 64, 576, dtype=torch.bfloat16)
    remapped = torch.empty(1, dtype=torch.int32)

    with pytest.raises(ValueError, match="selected indices"):
        materialize_selected_kvarn_mla(
            torch.zeros(1, dtype=torch.int64),
            cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            output,
            remapped,
            config,
        )
    with pytest.raises(ValueError, match="physical-slot workspace"):
        materialize_physical_kvarn_mla(
            torch.arange(63, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
            cache,
            block_to_slot,
            latent_pool,
            rope_pool,
            output,
            remapped,
            config,
        )
    with pytest.raises(ValueError, match="physical index buffers"):
        remap_kvarn_mla_physical_indices(
            torch.zeros(1, dtype=torch.int64),
            remapped,
            max_physical_slots=64,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_exact_scatter_cannot_write_through_invalid_block_or_pool_indices() -> None:
    device = torch.device("cuda")
    config = KVarNMLAConfig()
    latent = torch.ones(
        4,
        config.latent_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    rope = torch.full(
        (4, config.rope_dim),
        2,
        dtype=torch.bfloat16,
        device=device,
    )
    slot_mapping = torch.tensor([5, 64, 128, 192], dtype=torch.int32, device=device)
    block_to_slot = torch.tensor([0, 2, -1], dtype=torch.int32, device=device)
    latent_pool = torch.zeros(
        2,
        config.group,
        config.latent_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    rope_pool = torch.zeros(
        2,
        config.group,
        config.rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    scatter_kvarn_mla_exact(
        latent,
        rope,
        slot_mapping,
        block_to_slot,
        latent_pool,
        rope_pool,
    )

    torch.testing.assert_close(latent_pool[0, 5], latent[0].to(torch.float8_e4m3fn))
    torch.testing.assert_close(rope_pool[0, 5], rope[0])
    assert torch.count_nonzero(latent_pool[1]) == 0
    assert torch.count_nonzero(rope_pool[1]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_selected_materialization_bounds_pool_and_cache_indices() -> None:
    device = torch.device("cuda")
    config = KVarNMLAConfig()
    cache = torch.zeros(
        1,
        config.group,
        config.bytes_per_token,
        dtype=torch.uint8,
        device=device,
    )
    block_to_slot = torch.tensor([2], dtype=torch.int32, device=device)
    latent_pool = torch.ones(
        1,
        config.group,
        config.latent_dim,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    rope_pool = torch.ones(
        1,
        config.group,
        config.rope_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    selected = torch.tensor(
        [[0, 63, 64, -1]],
        dtype=torch.int32,
        device=device,
    )
    output = torch.full(
        (4, config.latent_dim + config.rope_dim),
        7,
        dtype=torch.bfloat16,
        device=device,
    )
    remapped = torch.empty_like(selected)

    materialize_selected_kvarn_mla(
        selected,
        cache,
        block_to_slot,
        latent_pool,
        rope_pool,
        output,
        remapped,
        config,
    )

    torch.testing.assert_close(output[:2], torch.zeros_like(output[:2]))
    torch.testing.assert_close(output[2:], torch.full_like(output[2:], 7))
    torch.testing.assert_close(
        remapped.cpu(),
        torch.tensor([[0, 1, -1, -1]], dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_physical_remap_rejects_out_of_arena_indices() -> None:
    selected = torch.tensor(
        [[0, 63, 64, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    remapped = torch.empty_like(selected)

    remap_kvarn_mla_physical_indices(
        selected,
        remapped,
        max_physical_slots=64,
    )

    torch.testing.assert_close(
        remapped.cpu(),
        torch.tensor([[0, 63, -1, -1]], dtype=torch.int32),
    )
