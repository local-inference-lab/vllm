# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Crossing capture digest identity.

Two prompts sharing the prefix through boundary B, differing after it, each
run through the real scheduler-side capture pipeline (record -> promote ->
copy rail) against simulated GPU state tensors. The frozen capture block's
bytes at B MUST be identical across the two prompts (deterministic
recurrence over the shared prefix) and MUST differ from each prompt's
chunk-end column bytes after the next step migrates the column onward --
the exact value a live-column store would mislabel as state@B. Without
boundary capture the hand-off names the live column, so the store-time
digest is the post-migration value: this test pins the defect.
"""

import hashlib

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace


@pytest.fixture(autouse=True)
def _auto_init_hash_fn():
    init_none_hash(sha256)


BLOCK_SIZE = 4
NUM_BLOCKS = 16
PAGE_BYTES = 32


def _make_capture_manager(boundary_capture: bool):
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    boundary_capture=boundary_capture,
                ),
            ),
        ],
    )
    manager = make_kv_cache_manager(
        kv_cache_config=kv_cache_config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK_SIZE,
    )
    # Two fake "layers" of per-block state: row index == pool block id.
    kv_caches = [
        torch.zeros(NUM_BLOCKS, PAGE_BYTES, dtype=torch.uint8)
        for _ in range(2)
    ]
    return manager, kv_caches


def _write_state(kv_caches, block_id, pattern: bytes) -> None:
    for cache in kv_caches:
        cache[block_id] = torch.frombuffer(
            bytearray(pattern * (PAGE_BYTES // len(pattern))),
            dtype=torch.uint8,
        ).reshape(PAGE_BYTES)


def _digest(kv_caches, block_id) -> str:
    h = hashlib.sha256()
    for cache in kv_caches:
        h.update(cache[block_id].numpy().tobytes())
    return h.hexdigest()


def _produce(manager, kv_caches, name, token_ids, state_at_b: bytes):
    """Run one producer prompt through record -> promote -> copy rail,
    simulating the worker: the producer's forward writes state@B into the
    live column, the rail copies it, then the NEXT step's forward migrates
    (clobbers) the column with the chunk-end state past B."""
    mgr = manager.coordinator.single_type_managers[1]
    req = make_request(name, token_ids, BLOCK_SIZE, sha256)
    req.skip_reading_prefix_cache = True  # always produce, never resume
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    column = mgr.req_to_blocks[name][0]

    # Producer forward: column holds state@B exactly (chunk ends at B).
    _write_state(kv_caches, column.block_id, state_at_b)

    req.num_computed_tokens = 4
    manager.new_step_starts()  # promote: rail + offer
    ((group_id, offered_id, boundary),) = manager.take_boundary_state_offloads()[name]
    assert group_id == 1 and boundary == 4
    copies, retained = manager.take_kv_cache_block_copies()
    assert copies and offered_id == copies[0].dst_block_id

    # Worker applies the rail copy (before the next precopy/forward).
    copy_kv_cache_blocks_inplace(kv_caches, NUM_BLOCKS, copies)
    capture_digest = _digest(kv_caches, offered_id)

    # Next step's forward migrates the column: state@B is gone from it.
    _write_state(kv_caches, column.block_id, b"\xee\xff")
    column_digest_after = _digest(kv_caches, column.block_id)

    manager.block_pool.free_blocks(retained)
    return offered_id, capture_digest, column_digest_after


def test_capture_digest_identical_across_prompts_and_not_the_column():
    """The mislabel test: frozen captures at B agree across prompts and
    differ from each prompt's post-migration column bytes."""
    manager, kv_caches = _make_capture_manager(boundary_capture=True)
    # Deterministic function of the shared 4-token prefix -> same bytes.
    state_at_b = b"\x0a\x14\x1e\x28"

    id1, cap1, col1 = _produce(manager, kv_caches, "p1", [0, 1, 2, 3, 8, 9], state_at_b)
    id2, cap2, col2 = _produce(
        manager, kv_caches, "p2", [0, 1, 2, 3, 100, 101], state_at_b
    )

    assert id1 != id2  # dedicated block per producer
    assert cap1 == cap2  # same prefix -> same state@B (key-identity holds)
    # Neither capture equals either prompt's chunk-end column value after
    # migration -- the pre-capture store would have shipped exactly those.
    assert cap1 != col1
    assert cap2 != col2
    assert col1 == col2  # both columns clobbered identically past B
    # The capture survived the column clobber: bytes are still state@B.
    assert _digest(kv_caches, id1) == cap1
    assert _digest(kv_caches, id2) == cap2


def test_legacy_live_column_store_mislabels_after_migration():
    """Without capture the hand-off names the live column: at store time the
    deferred DMA would read the post-migration value under the state@B key.
    This pins the defect the capture path removes (pre-change behavior)."""
    manager, kv_caches = _make_capture_manager(boundary_capture=False)
    mgr = manager.coordinator.single_type_managers[1]

    req = make_request("x", [0, 1, 2, 3, 8, 9], BLOCK_SIZE, sha256)
    computed_blocks, num_computed, _ = manager.get_computed_blocks(req)
    assert manager.allocate_slots(req, 4, num_computed, computed_blocks) is not None
    column = mgr.req_to_blocks["x"][0]
    _write_state(kv_caches, column.block_id, b"\x0a\x14\x1e\x28")

    ((group_id, offered_id, boundary),) = manager.take_boundary_state_offloads()["x"]
    assert (group_id, offered_id, boundary) == (1, column.block_id, 4)
    # No copy rail on the legacy path.
    assert manager.take_kv_cache_block_copies() == ([], [])

    state_at_b_digest = _digest(kv_caches, offered_id)
    # The next forward migrates the column before the deferred DMA reads it.
    _write_state(kv_caches, column.block_id, b"\xee\xff")
    assert _digest(kv_caches, offered_id) != state_at_b_digest


def test_all_recurrent_groups_of_align_config_carry_capture():
    """Enablement completeness: every MambaSpec that reaches the kv-cache
    config of an align-mode + OffloadingConnector deployment carries
    boundary_capture=True, so no recurrent group's manager falls back to the
    legacy live-column offer path under a boundary-colliding key.

    Guards the single-site enablement (MambaBase.get_kv_cache_spec): a future
    spec-construction site or a get_kv_cache_spec override that rebuilds the
    spec without super() would silently split capture coverage across groups.
    """
    from types import SimpleNamespace

    from vllm.config.cache import CacheConfig
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    class _Layer(MambaBase):
        def get_state_shape(self):
            return ((1, 8), (1, 8, 4))

        mamba_type = property(lambda self: None)

        def get_state_dtype(self):
            return (torch.float32, torch.float32)

    def _spec_for(connector, offload_size, align, boundary_ckpt=False):
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(
                mamba_block_size=64,
                mamba_page_size_padded=None,
                mamba_cache_mode=align,
                use_kda_recoverssm=False,
                kv_offloading_size=offload_size,
            ),
            kv_transfer_config=connector,
            num_speculative_tokens=0,
            use_request_boundary_checkpoints=boundary_ckpt,
        )
        return _Layer().get_kv_cache_spec(vllm_config)

    connector = SimpleNamespace(
        has_connector=lambda name: name == "OffloadingConnector"
    )
    # Align + complex offloading connector: capture on.
    for mode in ("align",):
        spec = _spec_for(connector, 1.0, mode)
        assert spec.boundary_capture is True, f"{mode} group missed capture"
    # Any gate missing -> capture off: boundary stores keep the live-column
    # behavior and external hits stop at the last checkpoint.
    assert _spec_for(connector, None, "align").boundary_capture is False
    assert _spec_for(None, 1.0, "align").boundary_capture is False
    assert _spec_for(
        SimpleNamespace(has_connector=lambda name: False), 1.0, "align"
    ).boundary_capture is False
    assert _spec_for(connector, 1.0, "all").boundary_capture is False
    assert (
        _spec_for(connector, 1.0, "align", boundary_ckpt=True).boundary_capture
        is False
    )
