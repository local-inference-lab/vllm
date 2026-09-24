# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.dcp import mask_dcp_empty_shards_


def test_mask_dcp_empty_shards_with_no_local_sequences() -> None:
    lse = torch.zeros((4, 1), dtype=torch.float32)
    seq_lens = torch.empty((0,), dtype=torch.int32)
    query_start_loc = torch.zeros((1,), dtype=torch.int32)

    mask_dcp_empty_shards_(lse, seq_lens, query_start_loc)

    assert torch.isneginf(lse).all()
