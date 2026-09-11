# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore endpoint state for ordinary and specialized scalar arguments."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.boundary_checkpoint import _restore_auxiliary_state_kernel

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")


@pytest.mark.parametrize("slot", [0, 1, 2])
@pytest.mark.parametrize("block", [0, 1, 2])
@pytest.mark.parametrize("size", [17, 1057])
def test_restore_accepts_specialized_scalar_indices(slot, block, size):
    width = size + 11
    pool_cpu = torch.arange(3 * width).to(torch.uint8).reshape(3, width)
    pool = pool_cpu.to("cuda")
    destination = torch.full((3, size), 165, dtype=torch.uint8, device="cuda")
    metadata = torch.tensor(
        [[destination.data_ptr(), destination.stride(0), size, 5]],
        dtype=torch.int64,
        device="cuda",
    )

    _restore_auxiliary_state_kernel[(1,)](
        metadata, pool, pool.stride(0), block, slot, BLOCK=1024
    )

    expected = torch.full((3, size), 165, dtype=torch.uint8)
    expected[slot].copy_(pool_cpu[block, 5 : 5 + size])
    torch.testing.assert_close(destination.cpu(), expected)


def test_restore_keeps_page_offsets_above_signed_int32():
    size, offset = 1057, 5
    width = size + 11
    block = (2**31 + width - 1) // width
    # Initialize only the selected page; its byte offset must exceed 2 GiB.
    pool = torch.empty((block + 1, width), dtype=torch.uint8, device="cuda")
    source = torch.arange(width).to(torch.uint8)
    pool[block].copy_(source)
    destination = torch.full((3, size), 165, dtype=torch.uint8, device="cuda")
    metadata = torch.tensor(
        [[destination.data_ptr(), destination.stride(0), size, offset]],
        dtype=torch.int64,
        device="cuda",
    )
    assert block * pool.stride(0) >= 2**31

    _restore_auxiliary_state_kernel[(1,)](
        metadata, pool, pool.stride(0), block, 1, BLOCK=1024
    )

    expected = torch.full((3, size), 165, dtype=torch.uint8)
    expected[1].copy_(source[offset : offset + size])
    torch.testing.assert_close(destination.cpu(), expected)
