# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.sched.scheduler import _build_kv_connector_block_state
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)


def _hybrid_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=2048,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=512,
                    shapes=((1, 1),),
                    dtypes=(torch.float16,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )


def test_connector_block_state_has_snapshot_and_retained_boundaries() -> None:
    grouped_ids = (
        [201, 202, 203, 204],
        [0, 0, 0, 0, 0, 0, 0, 107, 0, 0, 110, 0, 0, 0, 0, 115],
    )
    manager = SimpleNamespace(get_block_ids=MagicMock(return_value=grouped_ids))

    state = _build_kv_connector_block_state(
        _hybrid_config(),
        manager,
        ["req"],
        {"req": [(1, 999, 7680)]},
        retention_interval=4096,
    )

    assert state.block_ids == {"req": grouped_ids}
    assert state.boundary_state_offloads == {
        "req": [(1, 999, 7680), (1, 107, 4096), (1, 115, 8192)]
    }
    manager.get_block_ids.assert_called_once_with("req")


def test_connector_block_state_preserves_explicit_boundary_source() -> None:
    grouped_ids = ([201, 202], [0, 0, 0, 0, 0, 0, 0, 107])
    manager = SimpleNamespace(get_block_ids=MagicMock(return_value=grouped_ids))

    state = _build_kv_connector_block_state(
        _hybrid_config(),
        manager,
        ["req"],
        {"req": [(1, 777, 4096)]},
        retention_interval=4096,
    )

    assert state.boundary_state_offloads == {"req": [(1, 777, 4096)]}


def test_connector_block_state_rejects_misaligned_retention() -> None:
    manager = SimpleNamespace(get_block_ids=MagicMock(return_value=([], [])))

    with pytest.raises(ValueError, match="must be divisible"):
        _build_kv_connector_block_state(
            _hybrid_config(),
            manager,
            ["req"],
            None,
            retention_interval=4100,
        )
