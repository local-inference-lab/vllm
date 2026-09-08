# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Snapshot the same logical attention endpoint in sharded and replicated groups."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.boundary_checkpoint import _copy_attention_tails_kernel

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")


@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("kernel_block_size", [2, 8])
@pytest.mark.parametrize("endpoint", [1, 8, 9, 16, 17, 32, 33])
def test_attention_snapshot_uses_group_virtual_page(dcp, kernel_block_size, endpoint):
    page_size = 8
    bytes_per_page = 37
    num_groups = 2
    num_captures = 3
    device = "cuda"
    sources = [
        (torch.arange(8 * bytes_per_page).reshape(8, bytes_per_page) + group)
        .to(torch.uint8)
        .to(device)
        for group in range(num_groups)
    ]
    pools = [torch.cat((source, torch.full_like(source, 165))) for source in sources]
    metadata = torch.tensor(
        [
            [pool.data_ptr(), bytes_per_page, group, page_size]
            for group, pool in enumerate(pools)
        ],
        dtype=torch.int64,
        device=device,
    )
    blocks_per_page = page_size // kernel_block_size
    tables = []
    for group in range(num_groups):
        pages = torch.arange(8, dtype=torch.int32, device=device)
        if group == 1:
            pages = pages.flip(0)
        expanded = (
            pages[:, None] * blocks_per_page
            + torch.arange(blocks_per_page, dtype=torch.int32, device=device)
        ).flatten()
        tables.append(torch.stack((torch.zeros_like(expanded), expanded)))
    table_ptrs = torch.tensor(
        [t.data_ptr() for t in tables], dtype=torch.uint64, device=device
    )
    table_strides = torch.tensor(
        [t.stride(0) for t in tables], dtype=torch.int64, device=device
    )
    kernel_sizes = torch.full(
        (num_groups,), kernel_block_size, dtype=torch.int32, device=device
    )
    cp_sizes = torch.tensor([dcp, 1], dtype=torch.int32, device=device)
    slots = torch.tensor([1], dtype=torch.int32, device=device)
    counts = torch.tensor([[endpoint, 0, 0]], dtype=torch.int32, device=device)
    destinations = torch.full(
        (2, num_captures, num_groups + 1), 9, dtype=torch.int32, device=device
    )

    _copy_attention_tails_kernel[(num_captures, num_groups, 2)](
        slots,
        counts,
        destinations,
        metadata,
        table_ptrs,
        table_strides,
        kernel_sizes,
        cp_sizes,
        NUM_GROUPS=num_groups,
        BLOCK=32,
        TILES=2,
        NUM_CAPTURES=num_captures,
    )

    for group, cp_size in enumerate((dcp, 1)):
        page = (endpoint - 1) // (page_size * cp_size)
        source_page = page if group == 0 else 7 - page
        torch.testing.assert_close(pools[group][9], sources[group][source_page])
        torch.testing.assert_close(pools[group][:8], sources[group])
        assert torch.all(pools[group][8] == 165)
        assert torch.all(pools[group][10:] == 165)
