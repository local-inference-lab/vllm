# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the V2 model runner's InputBatch (vllm.v1.worker.gpu.input_batch)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.boundary_checkpoint import BoundaryCheckpoint
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.boundary_checkpoint import BoundaryCheckpointState
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, post_update

DEVICE = current_platform.device_type


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_boundary_auxiliary_restore_survives_slot_reuse_and_virtual_attention_pages():
    """Raw selector state, saved hidden states and attention tails form one bundle."""
    device = torch.device("cuda")
    raw_state = torch.arange(6, dtype=torch.float32, device=device).reshape(2, 3)
    anchors = torch.tensor([100, 200], dtype=torch.int64, device=device)
    accepted = torch.tensor([3, 4], dtype=torch.int32, device=device)
    expected_state = raw_state[1].clone()
    cache = (
        torch.arange(16, dtype=torch.uint8, device=device)[:, None]
        .expand(16, 256)
        .clone()
    )
    model = torch.nn.Module()
    model.set_recurrent_checkpoint_anchor = lambda slot, anchor: anchors[slot].fill_(
        anchor
    )
    model_state = SimpleNamespace(
        device=device,
        max_num_reqs=2,
        model=model,
        get_recurrent_checkpoint_tensors=lambda: (raw_state, anchors, accepted),
        get_recurrent_checkpoint_acceptance=lambda: accepted,
        model_config=SimpleNamespace(max_model_len=128),
    )
    draft_state = torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4)
    draft = torch.nn.Module()
    draft.get_recurrent_checkpoint_tensors = lambda: (draft_state,)
    config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attn"],
                FullAttentionSpec(
                    block_size=8,
                    num_kv_heads=1,
                    head_size=8,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    state = BoundaryCheckpointState(
        model_state, config, {"attn": SimpleNamespace(kv_cache=cache)}, draft
    )
    state.blocks.copy_(
        torch.tensor(
            [
                [[4, 8], [5, 9], [12, 13]],
                [[6, 10], [7, 11], [14, 15]],
            ],
            device=device,
        )
    )
    idx = torch.tensor([1, 0], dtype=torch.int32, device=device)
    capture = torch.tensor(
        [
            [[7, 0, 0], [0, 11, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[0, -1, 0], [-1, 1, 0]],
        ],
        dtype=torch.int32,
        device=device,
    )
    hidden = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8)
    spec_hidden = hidden + 100
    state.capture_auxiliary(idx, capture, hidden, spec_hidden)
    draft_state.add_(10)
    expected_draft = draft_state[1].clone()
    state.capture_draft(idx, capture)
    draft_state.fill_(-1)
    tables = BlockTables(
        block_sizes=[8],
        kernel_block_sizes=[2],
        max_num_reqs=2,
        max_num_blocks_per_group=[16],
        max_num_batched_tokens=16,
        device=device,
    )
    tables.append_block_ids(0, ([2, 3],), overwrite=True)
    tables.append_block_ids(1, ([4, 5],), overwrite=True)
    tables.apply_staged_writes()
    state.capture_attention(idx, capture, tables)
    state.wait_for_copies()
    assert torch.all(cache[6] == 4)
    assert torch.all(cache[5] == 3)
    raw_state.fill_(-1)
    anchors.fill_(-1)
    request = NewRequestData(
        req_id="resume",
        prompt_token_ids=list(range(9)),
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        block_ids=([6, 7],),
        num_computed_tokens=7,
        lora_request=None,
        boundary_checkpoint=BoundaryCheckpoint(1, 7, ((6,),), (10,)),
        boundary_checkpoint_blocks=((12, 13), (14, 15)),
    )
    state.add_request(0, request)
    torch.testing.assert_close(raw_state[0], expected_state)
    assert anchors[0].item() == 6
    assert accepted[0].item() == 1
    torch.testing.assert_close(state.get_hidden_states(10), hidden[:1])
    torch.testing.assert_close(state.get_hidden_states(10, draft=True), spec_hidden[:1])
    torch.testing.assert_close(draft_state[0], expected_draft)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_boundary_capture_uses_stop_trimmed_endpoint_and_leaves_middle_steps_private():
    """GPU endpoint selection must agree with scheduler EOS/length truncation."""

    def t(data):
        return torch.tensor(data, dtype=torch.int32, device="cuda")

    state = SimpleNamespace(
        metadata=t(
            [
                [1, 7, 0, 7, 8, 1],
                [1, 7, 0, 7, 30, 1],
                [0, 7, 0, 7, 30, 1],
                [1, 7, 0, 12, 30, 1],
                [1, 9, 4, 9, 30, 1],
            ]
        ),
        stop_tokens=torch.full((5, 128), 99, dtype=torch.int32, device="cuda"),
        seen=t([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0, 0]]),
    )
    sampled = t(
        [
            [10, 0, 0, 0],
            [10, 99, 12, 13],
            [10, 99, 12, 13],
            [99, 10, 99, 11],
            [0, 0, 0, 0],
        ]
    )
    computed = t([0, 8, 8, 8, 0])
    total = t([7, 9, 9, 9, 9])
    num_sampled = t([1, 4, 4, 4, 0])
    rejected = t([0, 0, 0, 0, 0])
    last_sampled = t([0, 0, 0, 0, 0])
    all_tokens = torch.zeros((5, 32), dtype=torch.int32, device="cuda")
    capture = torch.empty((3, 5, 3), dtype=torch.int32, device="cuda")
    post_update(
        t([0, 1, 2, 3, 4]),
        computed,
        last_sampled,
        None,
        sampled,
        num_sampled,
        rejected,
        t([0, 7, 11, 15, 19, 23]),
        all_tokens,
        total,
        state,
        capture,
    )
    assert capture[0].tolist() == [
        [7, 7, 0],
        [0, 10, 0],
        [0, 0, 0],
        [0, 11, 0],
        [0, 0, 4],
    ]
    assert capture[1].tolist() == [
        [0, 0, 0],
        [0, 1, 0],
        [0, 3, 0],
        [0, 2, 0],
        [0, 0, 0],
    ]
    assert num_sampled.tolist() == [1, 2, 4, 3, 0]
    assert computed.tolist() == [7, 10, 12, 11, 4]
    assert total.tolist() == [8, 11, 13, 12, 9]
    assert last_sampled.tolist() == [10, 99, 13, 99, 0]
    assert capture[2][0].tolist() == [6, 6, -1]
    assert capture[2][1, 1].item() == 8
    assert capture[2][3, 1].item() == 17
    assert capture[2][4, 2].item() == 22


@pytest.mark.parametrize(
    "num_reqs,num_tokens",
    [
        (256, 496),  # remainder 240: previously gave the last request 241 tokens
        (128, 512),  # no remainder
        (3, 8),
        (1, 7),
    ],
)
def test_make_dummy_distributes_remainder(num_reqs: int, num_tokens: int):
    """No dummy request may exceed ceil(num_tokens / num_reqs) tokens.

    Dumping the remainder on a single request can produce a dummy request with
    seq_len > max_model_len, which the block tables cannot back; attention
    kernels running on the dummy batch during cudagraph capture then read
    block-table entries out of bounds (https://github.com/vllm-project/vllm/pull/49364
    CI failure).
    """
    buffers = InputBuffers(
        max_num_reqs=num_reqs, max_num_tokens=num_tokens, device=torch.device(DEVICE)
    )
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)

    max_per_req = -(-num_tokens // num_reqs)
    assert batch.num_scheduled_tokens.sum() == num_tokens
    assert batch.num_scheduled_tokens.max() == max_per_req
    assert batch.num_scheduled_tokens.min() >= num_tokens // num_reqs
    # Requests with an extra token are placed at the end of the batch.
    assert (batch.num_scheduled_tokens[:-1] <= batch.num_scheduled_tokens[1:]).all()

    # seq_len == query_len for the dummy prefill-shaped batch, on GPU and CPU.
    query_lens = batch.query_start_loc_np[1:] - batch.query_start_loc_np[:-1]
    assert (query_lens == batch.num_scheduled_tokens).all()
    assert torch.equal(
        batch.seq_lens, torch.from_numpy(batch.num_scheduled_tokens).to(DEVICE)
    )
    assert batch.query_start_loc_np[-1] == num_tokens
    assert torch.equal(
        batch.query_start_loc.cpu(), torch.from_numpy(batch.query_start_loc_np)
    )
