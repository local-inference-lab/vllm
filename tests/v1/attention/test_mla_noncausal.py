# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec


class _NonCausalMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    supports_non_causal_multi_token_decode = True


def _metadata(
    query_start_loc: list[int],
    num_tokens: int | None = None,
    causal: bool = False,
) -> CommonAttentionMetadata:
    num_reqs = len(query_start_loc) - 1
    num_tokens = query_start_loc[-1] if num_tokens is None else num_tokens
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.arange(1, num_reqs + 1, dtype=torch.int32) * 100 + 8,
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(
            end - start for start, end in zip(query_start_loc, query_start_loc[1:])
        ),
        max_seq_len=num_reqs * 100 + 8,
        block_table_tensor=torch.arange(num_reqs * 3, dtype=torch.int32).view(
            num_reqs, 3
        ),
        slot_mapping=torch.arange(num_tokens),
        causal=causal,
        seq_lens_cpu_upper_bound=None,
    )


def _builder(marked: bool = True) -> _NonCausalMLAMetadataBuilder:
    builder = object.__new__(_NonCausalMLAMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.reorder_batch_threshold = 1
    builder.query_len_support = QueryLenSupport.SINGLE_ONLY
    builder.non_causal_multi_token_decode = marked
    builder.dcp_world_size = 1
    builder.use_pcp = False
    builder.metadata_cls = MLACommonMetadata
    builder.model_config = SimpleNamespace(
        dtype=torch.bfloat16, get_head_size=lambda: 576
    )
    return builder


def test_group_capability_keeps_runtime_causality_per_step():
    builder = _builder()

    target_metadata = builder.build(0, _metadata([0, 1, 2], causal=True))
    assert (
        target_metadata.num_decodes,
        target_metadata.num_decode_tokens,
        target_metadata.num_prefills,
        target_metadata.causal,
    ) == (2, 2, 0, True)

    common_metadata = _metadata([0, 8, 16])
    metadata = builder.build(0, common_metadata)

    assert metadata.num_decodes == 2
    assert metadata.num_decode_tokens == 16
    assert metadata.num_prefills == 0
    assert metadata.prefill is None
    assert metadata.decode is not None
    assert metadata.decode.block_table.shape == (2, 3)
    assert metadata.decode.seq_lens.shape == (2,)
    assert torch.equal(metadata.decode.block_table, common_metadata.block_table_tensor)
    assert torch.equal(metadata.decode.seq_lens, common_metadata.seq_lens)
    assert not metadata.causal


def test_noncausal_support_is_explicit_and_uniform():
    with pytest.raises(ValueError, match="explicitly supported"):
        _builder(marked=False).build(0, _metadata([0, 8, 16]))
    with pytest.raises(ValueError, match="uniform query block"):
        _builder().build(0, _metadata([0, 3, 8]))


def test_noncausal_block_allows_trailing_cudagraph_padding():
    common_metadata = _metadata([0, 8, 16, 16], num_tokens=24)
    common_metadata.seq_lens[-1] = 0

    metadata = _builder().build(0, common_metadata)

    assert metadata.num_decodes == 3
    assert metadata.num_decode_tokens == 24
    assert metadata.decode is not None
    assert metadata.decode.seq_lens.tolist() == [108, 208, 0]


def test_noncausal_block_rejects_non_trailing_padding():
    with pytest.raises(ValueError, match="uniform query block"):
        _builder().build(0, _metadata([0, 8, 8, 16], num_tokens=24))


def test_noncausal_decode_metadata_keeps_live_request_buffers():
    common_metadata = _metadata([0, 8, 16])
    metadata = _builder().build(0, common_metadata)

    assert metadata.decode is not None
    assert metadata.decode.seq_lens.data_ptr() == common_metadata.seq_lens.data_ptr()
    assert (
        metadata.decode.block_table.data_ptr()
        == common_metadata.block_table_tensor.data_ptr()
    )


def test_mla_cache_marker_is_promoted_to_group_capability():
    kwargs = {
        "block_size": 64,
        "num_kv_heads": 1,
        "head_size": 576,
        "dtype": torch.bfloat16,
    }
    marked = MLAAttentionSpec(**kwargs, non_causal_multi_token_decode=True)
    unmarked = MLAAttentionSpec(**kwargs)

    assert MLAAttentionSpec.merge([marked, marked]).non_causal_multi_token_decode
    assert not MLAAttentionSpec.merge(
        [unmarked, unmarked]
    ).non_causal_multi_token_decode
    assert MLAAttentionSpec.merge([unmarked, marked]).non_causal_multi_token_decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata kernel")
@pytest.mark.parametrize("dcp_size,rank", [(1, 0), (8, 7), (10, 3), (16, 0), (16, 15)])
@pytest.mark.parametrize("causal", [True, False])
def test_b12x_dense_decode_flattens_local_visibility_and_graph_padding(
    dcp_size, rank, causal
):
    from vllm.v1.attention.backends.mla.b12x_mla import _flatten_decode_metadata

    device = torch.device("cuda")
    interleave = 16
    starts = [0, 3, 5, 5]
    global_lengths = [dcp_size * interleave + 1, interleave + 2, 0]

    def local(length):
        cycle = dcp_size * interleave
        return length // cycle * interleave + min(
            max(length % cycle - rank * interleave, 0), interleave
        )

    local_lengths = [local(length) for length in global_lengths]
    cu = torch.tensor(starts, dtype=torch.int32, device=device)
    global_seq = torch.tensor(global_lengths, dtype=torch.int32, device=device)
    local_seq = torch.tensor(local_lengths, dtype=torch.int32, device=device)
    table = torch.arange(3 * 11, dtype=torch.int32, device=device).reshape(3, 11)[:, :7]
    lengths = torch.empty(8, dtype=torch.int32, device=device)
    pages = torch.empty((8, 9), dtype=torch.int32, device=device)

    def run():
        _flatten_decode_metadata[(8, 1)](
            cu,
            local_seq,
            global_seq,
            table,
            lengths,
            pages,
            3,
            table.shape[1],
            table.stride(0),
            pages.shape[1],
            dcp_size,
            rank,
            interleave,
            causal,
            128,
        )

    def check():
        expected = []
        for request, (start, end) in enumerate(zip(starts, starts[1:])):
            for row in range(start, end):
                expected.append(
                    local(global_lengths[request] + row + 1 - end)
                    if causal
                    else local_lengths[request]
                )
        assert lengths.tolist() == expected + [0, 0, 0]
        torch.testing.assert_close(pages[:3, :7], table[0].expand(3, -1))
        torch.testing.assert_close(pages[3:5, :7], table[1].expand(2, -1))
        assert torch.count_nonzero(pages[:, 7:]) == 0
        assert torch.count_nonzero(pages[5:]) == 0

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    pages.fill_(-77)
    lengths.fill_(-77)
    graph.replay()
    check()
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata kernel")
def test_bounded_draft_metadata_keeps_source_tables_and_replays_live_window():
    """Windowing must not mutate shared target metadata or read evicted pages."""
    from vllm.v1.attention.backends.mla.b12x_mla import _flatten_decode_metadata

    device = torch.device("cuda")
    window, page_size, rows = 4096, 64, 9
    cu = torch.tensor([0, 4, 8, 8], dtype=torch.int32, device=device)
    lengths = torch.tensor([window - 1, 524289, 0], dtype=torch.int32, device=device)
    table = torch.arange(3 * 16385, dtype=torch.int32, device=device).view(3, -1)
    source = table.clone()
    flat_lengths = torch.empty(rows, dtype=torch.int32, device=device)
    pages = torch.empty((rows, 65), dtype=torch.int32, device=device)

    def run():
        _flatten_decode_metadata[(rows, 1)](
            cu,
            lengths,
            None,
            table,
            flat_lengths,
            pages,
            3,
            table.shape[1],
            table.stride(0),
            pages.shape[1],
            1,
            0,
            1,
            False,
            128,
            WINDOW=window,
            PAGE_SIZE=page_size,
        )

    def check():
        expected_lengths = []
        for request, length in enumerate(lengths.tolist()[:2]):
            offset = max(length - window, 0) // page_size
            expected_lengths.extend([length - offset * page_size] * 4)
            torch.testing.assert_close(
                pages[request * 4 : (request + 1) * 4],
                source[request, offset : offset + 65].expand(4, -1),
            )
        assert flat_lengths.tolist() == expected_lengths + [0]
        assert torch.count_nonzero(pages[-1]) == 0
        torch.testing.assert_close(table, source)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for live_lengths in ([window, window + 1, 0], [window + 63, 1048000, 0]):
        lengths.copy_(torch.tensor(live_lengths, dtype=torch.int32, device=device))
        pages.fill_(-77)
        flat_lengths.fill_(-77)
        graph.replay()
        check()
    graph.reset()
