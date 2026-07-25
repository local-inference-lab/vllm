# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)


def test_block_tables_apply_staged_writes_fuses_kv_groups(monkeypatch):
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16, 32, 8],
        max_num_reqs=4,
        max_num_batched_tokens=64,
        max_num_blocks_per_group=[8, 8, 8],
        device=device,
        kernel_block_sizes=[16, 16, 8],
    )

    def fail_if_apply_write_called():
        pytest.fail("multi-group writes should use the fused apply kernel")

    for block_table in block_tables.block_tables:
        monkeypatch.setattr(block_table, "apply_write", fail_if_apply_write_called)

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2], [10, 11], []),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3], [12], [5, 6]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )
    # Group 1 has blocks_per_kv_block == 2, so each KV block expands to two
    # kernel block IDs.
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :4],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[0].gpu[1, :1],
        torch.tensor([3], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[1, :2],
        torch.tensor([24, 25], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[1, :2],
        torch.tensor([5, 6], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 2
    assert block_tables.num_blocks.np[1, 0] == 4
    assert block_tables.num_blocks.np[2, 0] == 0
    assert block_tables.num_blocks.np[0, 1] == 1
    assert block_tables.num_blocks.np[1, 1] == 2
    assert block_tables.num_blocks.np[2, 1] == 2
    assert torch.equal(
        block_tables.num_blocks.gpu[:, :2],
        torch.tensor([[2, 1], [4, 2], [0, 2]], dtype=torch.int32, device=device),
    )

    for block_table in block_tables.block_tables:
        assert not block_table._staged_write_indices
        assert not block_table._staged_write_starts
        assert not block_table._staged_write_contents
        assert not block_table._staged_write_cu_lens

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([7], [13], [8]),
        overwrite=False,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :3],
        torch.tensor([1, 2, 7], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[1].gpu[0, :6],
        torch.tensor([20, 21, 22, 23, 26, 27], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        block_tables.block_tables[2].gpu[0, :1],
        torch.tensor([8], dtype=torch.int32, device=device),
    )
    assert block_tables.num_blocks.np[0, 0] == 3
    assert block_tables.num_blocks.np[1, 0] == 6
    assert block_tables.num_blocks.np[2, 0] == 1


def test_block_tables_apply_staged_writes_single_group():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=16,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([1, 2],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()
    torch.accelerator.synchronize()

    assert torch.equal(
        block_tables.block_tables[0].gpu[0, :2],
        torch.tensor([1, 2], dtype=torch.int32, device=device),
    )


def test_compute_slot_mappings_applies_padding_mask():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[16],
        max_num_reqs=2,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[4],
        device=device,
        kernel_block_sizes=[16],
    )

    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([2],),
        overwrite=True,
    )
    block_tables.append_block_ids(
        req_index=1,
        new_block_ids=([3],),
        overwrite=True,
    )
    block_tables.apply_staged_writes()

    idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64, device=device)
    is_padding = torch.tensor(
        [False, True, False, False, True, False, False, False],
        dtype=torch.bool,
        device=device,
    )

    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded=8,
        is_padding=is_padding,
    )
    torch.accelerator.synchronize()

    assert slot_mappings.cpu().tolist() == [
        [32, PAD_SLOT_ID, 34, 48, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID]
    ]


def test_compute_slot_mappings_mixed_sharded_and_replicated_groups():
    device = torch.device("cuda")
    block_tables = BlockTables(
        block_sizes=[64, 256],
        max_num_reqs=1,
        max_num_batched_tokens=8,
        max_num_blocks_per_group=[1, 1],
        device=device,
        kernel_block_sizes=[64, 64],
        cp_size=4,
        cp_rank=2,
        group_cp_sizes=[4, 1],
    )
    block_tables.append_block_ids(
        req_index=0,
        new_block_ids=([5], [5]),
        overwrite=True,
    )
    block_tables.apply_staged_writes()

    idx_mapping = torch.tensor([0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32, device=device)
    positions = torch.arange(8, dtype=torch.int64, device=device)
    slot_mappings = block_tables.compute_slot_mappings(
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded=8,
    )
    torch.accelerator.synchronize()

    assert slot_mappings.cpu().tolist() == [
        [
            PAD_SLOT_ID,
            PAD_SLOT_ID,
            320,
            PAD_SLOT_ID,
            PAD_SLOT_ID,
            PAD_SLOT_ID,
            321,
            PAD_SLOT_ID,
        ],
        list(range(1280, 1288)),
    ]


def test_compute_slot_mappings_partial_replica_pairs_share_four_way_shard():
    device = torch.device("cuda")

    def compute_for_rank(cp_rank: int) -> list[int]:
        block_tables = BlockTables(
            block_sizes=[128],
            max_num_reqs=1,
            max_num_batched_tokens=16,
            max_num_blocks_per_group=[1],
            device=device,
            kernel_block_sizes=[64],
            cp_size=8,
            cp_rank=cp_rank,
            group_cp_sizes=[4],
        )
        block_tables.append_block_ids(
            req_index=0,
            new_block_ids=([5],),
            overwrite=True,
        )
        block_tables.apply_staged_writes()
        result = block_tables.compute_slot_mappings(
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([0, 16], dtype=torch.int32, device=device),
            torch.arange(16, dtype=torch.int64, device=device),
            num_tokens_padded=16,
        )
        torch.accelerator.synchronize()
        return result[0].cpu().tolist()

    rank0 = compute_for_rank(0)
    rank4 = compute_for_rank(4)
    assert rank0 == rank4
    assert rank0 == [
        640,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        641,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        642,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        643,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
        PAD_SLOT_ID,
    ]
