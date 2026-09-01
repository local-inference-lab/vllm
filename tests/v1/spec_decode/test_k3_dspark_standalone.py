# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys

import pytest
import torch

from vllm.entrypoints.k3_dspark_rpc import (
    DraftKVSlotAllocator,
    ProjectedContextCache,
    _cuda_graph_shapes,
    _encode_bfloat16_logits_frame,
    _projected_context_capacity_bytes,
    _validate_proposal_address,
)
from vllm.entrypoints.k3_dspark_standalone import (
    EMBED_TENSOR,
    LM_HEAD_TENSOR,
    _parse_args,
    _status_http_code,
    _validate_single_block_smoke_span,
    resolve_shared_weight_files,
)


def _write_index(root, weight_map):
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_resolve_shared_weight_files_requires_both_target_tensors(tmp_path):
    shard = tmp_path / "shared.safetensors"
    shard.touch()
    _write_index(tmp_path, {EMBED_TENSOR: shard.name})

    with pytest.raises(KeyError, match=LM_HEAD_TENSOR):
        resolve_shared_weight_files(tmp_path)


def test_resolve_shared_weight_files_resolves_checkpoint_shards(tmp_path):
    shard = tmp_path / "shared.safetensors"
    shard.touch()
    _write_index(
        tmp_path,
        {
            EMBED_TENSOR: shard.name,
            LM_HEAD_TENSOR: shard.name,
        },
    )

    resolved = resolve_shared_weight_files(tmp_path)

    assert resolved == {
        EMBED_TENSOR: shard.resolve(),
        LM_HEAD_TENSOR: shard.resolve(),
    }


def test_resolve_shared_weight_files_rejects_path_escape(tmp_path):
    outside = tmp_path.parent / "outside.safetensors"
    outside.touch()
    _write_index(
        tmp_path,
        {
            EMBED_TENSOR: f"../{outside.name}",
            LM_HEAD_TENSOR: f"../{outside.name}",
        },
    )

    with pytest.raises(ValueError, match="escapes"):
        resolve_shared_weight_files(tmp_path)


def test_draft_kv_slot_allocator_keeps_rolling_blocks_unique():
    allocator = DraftKVSlotAllocator(
        num_cache_blocks=11,
        block_size=4,
        window_size=16,
        max_requests=2,
    )
    first, created = allocator.get_or_allocate("first")
    second, _ = allocator.get_or_allocate("second")

    assert created
    assert allocator.physical_block_range(first) == slice(1, 6)
    assert allocator.physical_block_range(second) == slice(6, 11)
    blocks, local_len = allocator.block_table(first, 21)
    assert blocks == [2, 3, 4, 5, 1]
    assert len(blocks) == len(set(blocks))
    assert local_len == 17


def test_draft_kv_slot_allocator_reuses_freed_request_slot():
    allocator = DraftKVSlotAllocator(
        num_cache_blocks=6,
        block_size=4,
        window_size=16,
        max_requests=1,
    )
    original, _ = allocator.get_or_allocate("original")
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        allocator.get_or_allocate("other")

    assert allocator.free("original") is original
    replacement, created = allocator.get_or_allocate("replacement")
    assert created
    assert replacement.slot == original.slot


def test_draft_kv_slot_allocator_rebinds_without_changing_slot():
    allocator = DraftKVSlotAllocator(
        num_cache_blocks=6,
        block_size=4,
        window_size=16,
        max_requests=1,
    )
    original, _ = allocator.get_or_allocate("original")

    rebound = allocator.rebind("original", "replacement")

    assert rebound is original
    assert rebound.request_id == "replacement"
    assert allocator.get("original") is None
    assert allocator.get("replacement") is rebound


def test_draft_kv_slot_allocator_vectorizes_rolling_slots():
    allocator = DraftKVSlotAllocator(
        num_cache_blocks=11,
        block_size=4,
        window_size=16,
        max_requests=2,
    )
    allocator.get_or_allocate("first")
    state, _ = allocator.get_or_allocate("second")
    positions = torch.arange(2, 27, dtype=torch.int64)

    slots = allocator.cache_slots(state, positions)

    assert slots.tolist() == [
        allocator.cache_slot(state, int(position)) for position in positions
    ]


def test_projected_context_capacity_includes_chunk_straddling():
    capacity = _projected_context_capacity_bytes(
        max_requests=2,
        max_tokens=4,
        hidden_size=3,
        chunk_size=4,
    )

    assert capacity == 2 * 2 * 4 * 3 * 2


def test_cuda_graph_shapes_cover_every_accepted_request():
    assert _cuda_graph_shapes(3, 3) == [
        (3, 3),
        (3, 2),
        (3, 1),
        (2, 3),
        (2, 2),
        (2, 1),
        (1, 3),
        (1, 2),
        (1, 1),
    ]


@pytest.mark.parametrize(
    "address",
    ["tcp://127.0.0.1:8092", "tcp://[::1]:8092", "ipc:///tmp/k3.sock"],
)
def test_proposal_address_accepts_local_transports(address):
    _validate_proposal_address(address, allow_unsafe_remote=False)


@pytest.mark.parametrize(
    "address",
    ["tcp://0.0.0.0:8092", "tcp://*:8092", "tcp://192.0.2.10:8092"],
)
def test_proposal_address_rejects_unauthenticated_remote_bind(address):
    with pytest.raises(ValueError, match="no authentication"):
        _validate_proposal_address(address, allow_unsafe_remote=False)

    _validate_proposal_address(address, allow_unsafe_remote=True)


def test_dspark_smoke_span_must_fit_its_mapped_block():
    _validate_single_block_smoke_span(1, 15, 16)

    with pytest.raises(ValueError, match="single mapped KV block"):
        _validate_single_block_smoke_span(1, 16, 16)


def test_health_is_live_before_readiness():
    assert _status_http_code("/healthz", ready=False) == 200
    assert _status_http_code("/readyz", ready=False) == 503
    assert _status_http_code("/readyz", ready=True) == 200


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--num-speculative-tokens", "0"],
        ["--max-num-batched-tokens", "0"],
        ["--max-num-seqs", "0"],
        ["--draft-kv-cache-gib", "nan"],
        ["--draft-kv-window", "0"],
        ["--draft-kv-window", "17"],
        ["--cuda-graph-warmups", "0"],
        ["--port", "65536"],
    ],
)
def test_standalone_cli_rejects_invalid_numeric_values(monkeypatch, invalid_args):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "k3-draft",
            "--draft-model",
            "/draft",
            "--target-weights",
            "/target",
            "--target-config",
            "/config",
            *invalid_args,
        ],
    )

    with pytest.raises(SystemExit):
        _parse_args()


def test_projected_context_cache_rewinds_and_overwrites_exact_prefix():
    cache = ProjectedContextCache(hidden_size=2, max_tokens=8, chunk_size=4)
    initial = torch.arange(16, dtype=torch.bfloat16).view(8, 2)
    cache.append(0, initial)

    replacement = torch.tensor([[100, 101], [102, 103]], dtype=torch.bfloat16)
    cache.append(5, replacement)

    assert cache.start_position == 0
    assert cache.end_position == 7
    assert torch.equal(cache.read(0, 5), initial[:5])
    assert torch.equal(cache.read(5, 7), replacement)


def test_projected_context_cache_evicts_old_rows_without_regaining_them():
    cache = ProjectedContextCache(hidden_size=1, max_tokens=6, chunk_size=4)
    cache.append(0, torch.arange(8, dtype=torch.bfloat16).view(8, 1))

    assert cache.start_position == 2
    assert cache.has_range(2, 8)
    assert not cache.has_range(1, 8)

    cache.append(5, torch.tensor([[50], [60]], dtype=torch.bfloat16))
    assert cache.start_position == 2
    assert cache.end_position == 7
    with pytest.raises(ValueError, match="unavailable"):
        cache.read(1, 7)


def test_projected_context_cache_bounds_large_append_allocation():
    cache = ProjectedContextCache(
        hidden_size=1,
        max_tokens=4,
        chunk_size=4,
    )

    cache.append(0, torch.arange(12, dtype=torch.bfloat16).view(12, 1))

    assert cache.start_position == 8
    assert cache.end_position == 12
    assert torch.equal(
        cache.read(8, 12),
        torch.arange(8, 12, dtype=torch.bfloat16).view(4, 1),
    )
    assert cache.allocated_bytes <= _projected_context_capacity_bytes(
        max_requests=1,
        max_tokens=4,
        hidden_size=1,
        chunk_size=4,
    )


def test_projected_context_cache_tracks_configured_device():
    cache = ProjectedContextCache(
        hidden_size=2,
        max_tokens=4,
        chunk_size=2,
        device=torch.device("cpu"),
    )
    states = torch.arange(8, dtype=torch.bfloat16).view(4, 2)

    cache.append(0, states)

    assert cache.device == torch.device("cpu")
    assert cache.read(0, 4).device == cache.device
    assert cache.allocated_bytes == states.numel() * states.element_size()


def test_projected_context_cache_rejects_device_mismatch():
    cache = ProjectedContextCache(hidden_size=2, max_tokens=4)
    states = torch.empty((1, 2), dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError, match="device mismatch"):
        cache.append(0, states)


def test_encode_bfloat16_logits_frame_preserves_shape_and_bits():
    logits = torch.tensor(
        [[[1.0, -2.0], [3.5, 0.25]]],
        dtype=torch.bfloat16,
    )

    metadata, frame = _encode_bfloat16_logits_frame(logits)

    assert metadata == {
        "capability": "dflash_logits_bf16_v1",
        "dtype": "bfloat16",
        "shape": [1, 2, 2],
        "nbytes": 8,
    }
    decoded = (
        torch.frombuffer(bytearray(frame), dtype=torch.uint16)
        .view(torch.bfloat16)
        .reshape(1, 2, 2)
    )
    assert torch.equal(decoded, logits)


def test_encode_logits_rejects_non_bfloat16_input():
    with pytest.raises(ValueError, match="must be bfloat16"):
        _encode_bfloat16_logits_frame(torch.zeros(1, 2, dtype=torch.float32))
