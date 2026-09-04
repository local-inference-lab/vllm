# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.entrypoints.k3_dspark_rpc import (
    DraftKVSlotAllocator,
    ProjectedContextCache,
)
from vllm.entrypoints.k3_dspark_standalone import (
    EMBED_TENSOR,
    LM_HEAD_TENSOR,
    _effective_aux_geometry,
    _smoke_block_ids,
    resolve_shared_weight_files,
)


def test_effective_aux_geometry_uses_layer_ids_and_target_hidden_width():
    hf_config = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=(3, 7, 11),
        hidden_size=4096,
        target_hidden_size=3584,
        num_target_layers=99,
    )
    speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=hf_config)
    )

    assert _effective_aux_geometry(speculative_config) == (3, 3584)


def test_smoke_block_ids_cover_query_spillover():
    assert _smoke_block_ids(4, 1) == [1]
    assert _smoke_block_ids(4, 4) == [1]
    assert _smoke_block_ids(4, 5) == [1, 2]


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


# --- peer-memory transport: the draft side ------------------------------------


def _p2p_engine(max_requests=4, steps=3, width=16):
    import threading
    from types import SimpleNamespace

    from vllm.entrypoints.k3_dspark_rpc import K3DSparkDraftEngine

    engine = K3DSparkDraftEngine.__new__(K3DSparkDraftEngine)
    engine.method = "dflash"
    engine.raw_context_width = width
    engine.allocator = SimpleNamespace(max_requests=max_requests)
    engine.max_speculative_tokens = steps
    engine.device = torch.device("cuda")
    engine._lock = threading.Lock()
    engine._p2p = None
    return engine


def _p2p_open_header(**overrides):
    header = {
        "context_rows": 8,
        "context_width": 16,
        "context_slots": 3,
        "max_requests": 4,
        "num_speculative_tokens": 3,
        "logits_topk": 5,
        "reply_slots": 2,
    }
    header.update(overrides)
    return header


def test_open_p2p_exports_ring_and_reply_slots_and_reuses_matching_buffers(
    monkeypatch,
):
    """P2P_OPEN allocates the peer buffers for the stated geometry, exports
    IPC handles for them, and keeps the same buffers for a repeated open."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    engine = _p2p_engine()
    from vllm.entrypoints.k3_dspark_rpc import P2P_CAPABILITY

    try:
        exported = engine.open_p2p(_p2p_open_header())
    except ValueError as exc:
        if "cudaMalloc" in str(exc):
            pytest.skip("this process does not use cudaMalloc allocations")
        raise
    assert exported["ok"] and exported["capability"] == P2P_CAPABILITY
    assert exported["context"]["shape"] == [3, 8, 16]
    assert exported["values"]["shape"] == [2, 4, 3, 5]
    assert exported["indices"]["dtype"] == "int32"
    first = engine._p2p
    engine.open_p2p(_p2p_open_header())
    assert engine._p2p is first
    engine.open_p2p(_p2p_open_header(reply_slots=3))
    assert engine._p2p is not first
    with pytest.raises(ValueError, match="width"):
        engine.open_p2p(_p2p_open_header(context_width=32))
    with pytest.raises(ValueError, match="requests"):
        engine.open_p2p(_p2p_open_header(max_requests=9))


def test_stage_p2p_reply_writes_top_k_into_alternating_slots():
    """The reply slot holds exactly the top-k values and indices of the
    draft logits, narrower depths fill a prefix, and slots alternate."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from vllm.entrypoints.k3_dspark_rpc import (
        TOPK_LOGITS_P2P_CAPABILITY,
        _P2PBuffers,
    )

    engine = _p2p_engine()
    engine._p2p = _P2PBuffers(
        context=torch.zeros((3, 8, 16), dtype=torch.bfloat16, device="cuda"),
        values=torch.zeros((2, 4, 3, 5), dtype=torch.bfloat16, device="cuda"),
        indices=torch.zeros((2, 4, 3, 5), dtype=torch.int32, device="cuda"),
    )
    logits = torch.randn((2, 2, 64), device="cuda").to(torch.bfloat16)
    metadata = engine._stage_p2p_reply(logits, 5)
    assert metadata["capability"] == TOPK_LOGITS_P2P_CAPABILITY
    assert metadata["shape"] == [2, 2, 5] and metadata["p2p_reply_slot"] == 0
    values, indices = torch.topk(logits.reshape(-1, 64), 5, dim=-1, sorted=False)
    slot = engine._p2p.values[0, :2, :2].reshape(-1, 5)
    assert torch.equal(slot.sort(dim=-1).values, values.sort(dim=-1).values)
    gathered = torch.gather(
        logits.reshape(-1, 64), 1, engine._p2p.indices[0, :2, :2].reshape(-1, 5).long()
    )
    assert torch.equal(gathered, slot)
    assert engine._stage_p2p_reply(logits, 5)["p2p_reply_slot"] == 1
    assert engine._stage_p2p_reply(logits, 5)["p2p_reply_slot"] == 0
    assert engine._stage_p2p_reply(logits, 5, 1)["p2p_reply_slot"] == 1
    with pytest.raises(ValueError, match="out of range"):
        engine._stage_p2p_reply(logits, 5, 2)
    with pytest.raises(ValueError, match="top-5"):
        engine._stage_p2p_reply(logits, 4)
