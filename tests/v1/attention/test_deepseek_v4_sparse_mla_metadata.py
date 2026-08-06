# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadataBuilder,
    build_c128a_topk_metadata,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("dcp_rank", "valid_offset", "expected_slot"),
    [
        (0, 7, 17 * 64),
        (1, 3, 11 * 64 + 63),
    ],
)
def test_compressed_slot_mapping_uses_dcp_local_pages(
    monkeypatch: pytest.MonkeyPatch,
    dcp_rank: int,
    valid_offset: int,
    expected_slot: int,
) -> None:
    from vllm.distributed import parallel_state

    monkeypatch.setattr(
        parallel_state,
        "get_dcp_group",
        lambda: SimpleNamespace(rank_in_group=dcp_rank),
    )
    device = torch.device("cuda")
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=512),
            max_model_len=1024,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        speculative_config=None,
    )
    builder = DeepseekV4FlashMLAMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
    )

    query_start_loc = torch.tensor([0, 8], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([516], dtype=torch.int32, device=device)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=8,
        max_query_len=8,
        max_seq_len=516,
        block_table_tensor=torch.tensor([[11, 17]], dtype=torch.int32, device=device),
        slot_mapping=torch.full((8,), -123, dtype=torch.int64, device=device),
        causal=True,
        dcp_local_seq_lens=torch.tensor([258], dtype=torch.int32, device=device),
    )

    metadata = builder.build(0, common)

    expected = torch.full((8,), -1, dtype=torch.int64, device=device)
    expected[valid_offset] = expected_slot
    torch.testing.assert_close(metadata.slot_mapping, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("dcp_rank", "expected_decode"),
    [
        (0, [22, 23, 34]),
        (1, [22, 23]),
    ],
)
def test_c128a_metadata_compacts_dcp_local_decode_slots(
    dcp_rank: int,
    expected_decode: list[int],
) -> None:
    device = torch.device("cuda")
    max_compressed_tokens = 8
    global_decode, decode_lens, prefill = build_c128a_topk_metadata(
        positions=torch.tensor([639, 639, 639], dtype=torch.int64, device=device),
        compress_ratio=128,
        num_decode_tokens=1,
        actual_num_query_tokens=2,
        token_to_req_indices=torch.zeros(3, dtype=torch.int32, device=device),
        block_table=torch.tensor([[11, 17]], dtype=torch.int32, device=device),
        block_size=2,
        # Rank-local write ownership must not invalidate the decode query.
        slot_mapping=torch.full((3,), -1, dtype=torch.int64, device=device),
        global_decode_buffer=torch.empty(
            (3, max_compressed_tokens), dtype=torch.int32, device=device
        ),
        decode_lens_buffer=torch.empty(3, dtype=torch.int32, device=device),
        prefill_buffer=torch.empty(
            (3, max_compressed_tokens), dtype=torch.int32, device=device
        ),
        max_compressed_tokens=max_compressed_tokens,
        dcp_world_size=2,
        dcp_rank=dcp_rank,
        cp_kv_cache_interleave_size=1,
    )

    expected_decode_tensor = torch.full(
        (max_compressed_tokens,), -1, dtype=torch.int32, device=device
    )
    expected_decode_tensor[: len(expected_decode)] = torch.tensor(
        expected_decode, dtype=torch.int32, device=device
    )
    torch.testing.assert_close(global_decode[0], expected_decode_tensor)
    assert decode_lens.tolist() == [len(expected_decode)]

    expected_prefill = torch.full(
        (2, max_compressed_tokens), -1, dtype=torch.int32, device=device
    )
    expected_prefill[0, :5] = torch.arange(5, dtype=torch.int32, device=device)
    torch.testing.assert_close(prefill, expected_prefill)
