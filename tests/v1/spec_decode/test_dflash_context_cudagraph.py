# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    DFlashContextCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    _DFlashInputBatch,
    prepare_dflash_inputs,
)


def _make_dispatch_manager(*, captured: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
            max_cudagraph_capture_size=8,
        ),
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        max_num_context_tokens=12,
        _capture_descs={},
        _graphs_captured=captured,
    )


def test_context_graph_dispatch_uses_smallest_capture_bucket():
    manager = _make_dispatch_manager()

    DFlashContextCudaGraphManager._init_candidates(manager)

    assert [d.num_tokens for d in manager._capture_descs[CUDAGraphMode.FULL]] == [
        8,
        4,
        2,
        1,
    ]
    assert DFlashContextCudaGraphManager.dispatch_context(manager, 1).num_tokens == 1
    assert DFlashContextCudaGraphManager.dispatch_context(manager, 3).num_tokens == 4
    assert DFlashContextCudaGraphManager.dispatch_context(manager, 7).num_tokens == 8
    assert (
        DFlashContextCudaGraphManager.dispatch_context(manager, 9).cg_mode
        == CUDAGraphMode.NONE
    )


def test_context_graph_dispatch_falls_back_before_capture():
    manager = _make_dispatch_manager(captured=False)
    DFlashContextCudaGraphManager._init_candidates(manager)

    desc = DFlashContextCudaGraphManager.dispatch_context(manager, 3)

    assert desc.cg_mode == CUDAGraphMode.NONE
    assert desc.num_tokens == 3


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_prepare_dflash_inputs_makes_context_graph_padding_inert():
    device = torch.device("cuda")
    max_num_reqs = 4
    max_num_tokens = 16
    num_reqs = 2
    num_speculative_steps = 2
    num_query_per_req = 3
    num_context_tokens = 3
    padded_context_tokens = 8

    input_buffers = SimpleNamespace(
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
    )
    input_batch = _DFlashInputBatch(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
        positions=torch.arange(num_context_tokens, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32, device=device),
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
    )

    query_slots = torch.full((max_num_tokens,), 777, dtype=torch.int64, device=device)
    context_positions = torch.full_like(query_slots, 777)
    context_slots = torch.full_like(query_slots, 777)
    sample_rows = max_num_reqs * num_speculative_steps
    block_table = torch.arange(
        1,
        1 + max_num_reqs * 4,
        dtype=torch.int32,
        device=device,
    ).view(max_num_reqs, 4)

    prepare_dflash_inputs(
        input_buffers,
        query_slots,
        context_positions,
        context_slots,
        torch.zeros(sample_rows, dtype=torch.int64, device=device),
        torch.zeros(sample_rows, dtype=torch.int64, device=device),
        torch.full((sample_rows,), -1, dtype=torch.int32, device=device),
        input_batch,
        torch.zeros(num_reqs, dtype=torch.int32, device=device),
        torch.zeros(num_reqs, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        block_table,
        16,
        torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        1,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        1024,
        context_num_tokens_padded=padded_context_tokens,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        context_positions[:num_context_tokens],
        torch.arange(num_context_tokens, dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(
        context_slots[:num_context_tokens],
        torch.tensor([16, 17, 82], dtype=torch.int64, device=device),
    )
    torch.testing.assert_close(
        context_positions[num_context_tokens:padded_context_tokens],
        torch.zeros(
            padded_context_tokens - num_context_tokens,
            dtype=torch.int64,
            device=device,
        ),
    )
    torch.testing.assert_close(
        context_slots[num_context_tokens:padded_context_tokens],
        torch.full(
            (padded_context_tokens - num_context_tokens,),
            PAD_SLOT_ID,
            dtype=torch.int64,
            device=device,
        ),
    )
