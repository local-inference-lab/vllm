# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded rows of aligned recurrent state indices use NULL_BLOCK_ID."""

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.worker.mamba_utils import get_aligned_state_indices_multi_group_kernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_padded_rows_point_at_the_null_block():
    block_size, num_groups, num_state_slots = 16, 2, 2
    block_tables = [
        torch.arange(1, 1 + 4 * 8, dtype=torch.int32, device="cuda").view(4, 8)
        + 100 * group
        for group in range(num_groups)
    ]
    pointers = torch.tensor(
        [bt.data_ptr() for bt in block_tables], dtype=torch.int64, device="cuda"
    )
    # Rows 2 and 3 are CUDA-graph padding: no tokens.
    seq_lens = torch.tensor([17, 40, 0, 0], dtype=torch.int32, device="cuda")
    indices = torch.full(
        (num_groups, 4, num_state_slots), 12345, dtype=torch.int32, device="cuda"
    )
    get_aligned_state_indices_multi_group_kernel[(1,)](
        pointers,
        seq_lens,
        indices,
        block_tables[0].stride(0),
        seq_lens.stride(0),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        4,
        CACHE_BLOCK_SIZE=block_size,
        NUM_GROUPS=num_groups,
        BLOCK_GROUPS=triton.next_power_of_2(num_groups),
        NUM_STATE_SLOTS=num_state_slots,
        BLOCK_STATE_SLOTS=triton.next_power_of_2(num_state_slots),
        BLOCK_ROWS=4,
        num_warps=1,
    )
    for group, bt in enumerate(block_tables):
        for row, seq_len in enumerate([17, 40]):
            first = (seq_len - 1) // block_size
            expected = bt[row, first : first + num_state_slots]
            assert torch.equal(indices[group, row], expected)
        assert (indices[group, 2:] == NULL_BLOCK_ID).all()
