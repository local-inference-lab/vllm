# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4.nvidia.b12x_indexer import _flatten_index_cache
from vllm.models.glm5next.nvidia import pooled_indexer as indexer_module
from vllm.models.glm5next.nvidia.ops import glm_kpool
from vllm.models.glm5next.nvidia.ops.glm_kpool import (
    expand_c4_block_table,
    expand_pool_ids,
    gather_c4_block_table_rows,
    pool_seq_lens,
    prepare_c4_decode_metadata,
    update_decode_pools,
)
from vllm.models.glm5next.nvidia.pooled_indexer import (
    Glm5NextIndexerScratch,
    Glm5NextPooledIndexer,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xGLM5NextMLASparseMetadataBuilder,
    B12xMLASparseMetadata,
)
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def _require_glm_gpu() -> torch.device:
    if os.environ.get("B12X_GLM53_GPU_TEST") != "1":
        pytest.skip("set B12X_GLM53_GPU_TEST=1 to run GLM-5.3 GPU tests")
    if not torch.accelerator.is_available():
        pytest.skip("GLM-5.3 GPU tests require CUDA")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    if current_platform.get_device_capability(device.index or 0) not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("GLM-5.3 GPU tests require SM120 or SM121")
    return device


def _hadamard128(x: torch.Tensor) -> torch.Tensor:
    for stride in (1, 2, 4, 8, 16, 32, 64):
        x = x.reshape(-1, 2, stride)
        a, b = x[:, 0], x[:, 1]
        x = torch.stack((a + b, a - b), dim=1).reshape(128)
    return x / (128**0.5)


@pytest.mark.parametrize("rows", [1, 4, 5, 31, 32, 33, 128])
def test_glm53_fused_fwht_weight_scaling_is_bitwise(rows: int) -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(53 + rows)
    query = torch.randn(
        (rows, 128), generator=generator, dtype=torch.bfloat16, device=device
    )
    expected_query = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    expected_scales = torch.empty(rows, dtype=torch.float32, device=device)
    actual_query = torch.empty_like(expected_query)
    actual_scales = torch.empty_like(expected_scales)
    legacy_query = torch.empty_like(expected_query)
    legacy_scales = torch.empty_like(expected_scales)
    initial_weights = torch.randn(
        rows, generator=generator, dtype=torch.float32, device=device
    )
    expected_weights = initial_weights.clone()
    actual_weights = initial_weights.clone()
    legacy_weights = initial_weights.clone()

    glm_kpool.fwht128_quant_fp8(query, expected_query, expected_scales)
    expected_weights.mul_(expected_scales)
    expected_weights.mul_((128 * 32) ** -0.5)
    glm_kpool.fwht128_quant_fp8(
        query,
        actual_query,
        actual_scales,
        weights=actual_weights,
    )
    glm_kpool._fwht_quant_kernel[(triton.cdiv(rows, 32),)](
        query,
        legacy_query,
        legacy_scales,
        legacy_weights,
        rows,
        HEAD_DIM=128,
        FP8_MAX=448.0,
        BLOCK_R=32,
        SCALE_WEIGHTS=True,
        WEIGHT_NORM=(128 * 32) ** -0.5,
        num_warps=2,
    )

    torch.testing.assert_close(actual_query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_query, legacy_query, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, legacy_scales, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, legacy_weights, rtol=0, atol=0)


def test_glm53_fused_fwht_weight_scaling_graph_replays_live_inputs() -> None:
    device = _require_glm_gpu()
    rows = 33
    generator = torch.Generator(device=device).manual_seed(5300)
    query = torch.randn(
        (rows, 128), generator=generator, dtype=torch.bfloat16, device=device
    )
    weights = torch.randn(rows, generator=generator, dtype=torch.float32, device=device)
    query_out = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows, dtype=torch.float32, device=device)

    def transform() -> None:
        glm_kpool.fwht128_quant_fp8(query, query_out, scales, weights=weights)

    transform()
    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        transform()

    query.copy_(
        torch.randn(
            query.shape,
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
    )
    input_weights = torch.randn(
        rows, generator=generator, dtype=torch.float32, device=device
    )
    weights.copy_(input_weights)
    query_out.zero_()
    scales.zero_()
    graph.replay()
    torch.accelerator.synchronize()

    expected_query = torch.empty_like(query_out)
    expected_scales = torch.empty_like(scales)
    expected_weights = input_weights.clone()
    glm_kpool.fwht128_quant_fp8(query, expected_query, expected_scales)
    expected_weights.mul_(expected_scales)
    expected_weights.mul_((128 * 32) ** -0.5)
    torch.testing.assert_close(query_out, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)

    allocated = torch.accelerator.memory_allocated()
    weights.copy_(input_weights)
    graph.replay()
    weights.copy_(input_weights)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated


@pytest.mark.parametrize("mixed_prefill", [False, True])
@pytest.mark.parametrize("dcp_size,dcp_rank", [(1, 0), (4, 0), (4, 3)])
def test_glm53_adaptive_sparse_metadata_replays_device_boundaries(
    mixed_prefill, dcp_size, dcp_rank
) -> None:
    device = _require_glm_gpu()
    builder = object.__new__(B12xGLM5NextMLASparseMetadataBuilder)
    builder.use_pcp = False
    builder.reorder_batch_threshold = 128
    builder._prefill_backend = None
    builder.topk_tokens = 2048
    builder.cp_kv_cache_interleave_size = 4
    builder.kv_cache_spec = SimpleNamespace(block_size=256)
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    builder.requires_glm_next_selector_metadata = True
    builder._ckv_gather_requested = False
    builder.dcp_world_size = dcp_size
    builder.dcp_rank = dcp_rank
    builder._max_speculative_decode_query_len = 8
    builder.req_id_per_token_buffer = torch.empty(12, dtype=torch.int32, device=device)
    builder.cache_seq_lens_per_token_buffer = torch.empty_like(
        builder.req_id_per_token_buffer
    )
    builder._capture_default_state_slot_ids = torch.arange(
        4, dtype=torch.int32, device=device
    )
    builder._capture_state_slot_ids = torch.empty(4, dtype=torch.int32, device=device)
    builder._capture_state_is_fresh = torch.empty(4, dtype=torch.bool, device=device)
    builder._capture_num_accepted_tokens = torch.empty_like(
        builder._capture_state_slot_ids
    )
    builder._capture_is_prefilling = torch.empty_like(builder._capture_state_is_fresh)
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=12,
        max_query_len=8,
        max_logits_per_req=None,
        max_seq_len=1024,
        query_start_loc=torch.tensor(
            [0, 4, 8, 12, 12], dtype=torch.int32, device=device
        ),
        query_start_loc_cpu=torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32),
        seq_lens=torch.tensor([264, 524, 784, 0], dtype=torch.int32, device=device),
        seq_lens_cpu_upper_bound=torch.tensor([268, 528, 784, 0], dtype=torch.int32),
        block_table_tensor=torch.tensor(
            [[5], [7], [9], [0]], dtype=torch.int32, device=device
        ),
        slot_mapping=torch.full((12,), -1, dtype=torch.int64, device=device),
        dcp_local_seq_lens=None,
        positions=None,
        is_prefilling=torch.tensor([False, False, mixed_prefill, False]),
    )
    captured = builder.build_for_cudagraph_capture(common)
    observed_ids = torch.empty_like(captured.req_id_per_token)
    observed_lens = torch.empty_like(captured.cache_seq_lens_per_token)
    if dcp_size == 1:
        from b12x.attention import sparse_mla

        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index,
        )

        cache = torch.empty((10, 256, 528), dtype=torch.uint8, device=device)
        kv_c = torch.randn((2560, 512), dtype=torch.bfloat16, device=device)
        slots = torch.arange(2560, dtype=torch.int64, device=device)
        sparse_mla.concat_and_cache_glm_next_mla_fp8(
            kv_c, cache, slots, plan=sparse_mla.plan_cache_writer(kv_c, cache, slots)
        )
        query = torch.randn((12, 16, 512), dtype=torch.bfloat16, device=device)
        indices = torch.full((12, 2051), -1, dtype=torch.int32, device=device)
        indices[:, :4] = torch.arange(4, dtype=torch.int32, device=device)
        plan = sparse_mla.plan(
            sparse_mla.Caps(
                device=device,
                num_q_heads=16,
                max_q_rows=12,
                max_width=2051,
                softmax_scale=256**-0.5,
                kv_dtype=torch.uint8,
                head_dim=512,
                v_head_dim=512,
                model_type=int(sparse_mla.ModelType.GLM_NEXT),
                mode="extend" if mixed_prefill else "decode",
                max_batch=12,
                page_size=256,
            )
        )
        (spec,) = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

        def attention(request_ids, lengths):
            physical, counts = triton_convert_req_index_to_global_index(
                request_ids,
                common.block_table_tensor,
                indices,
                BLOCK_SIZE=256,
                NUM_TOPK_TOKENS=2051,
                return_valid_counts=True,
            )
            result = sparse_mla.run(
                sparse_mla.bind(
                    plan,
                    scratch=scratch,
                    q=query,
                    kv_cache=cache,
                    selected_indices=physical,
                    cache_lengths=lengths,
                    selected_lengths=counts,
                )
            )
            # The extend mode also returns the log-sum-exp rows.
            return result[0] if isinstance(result, tuple) else result

        observed_attention = attention(
            captured.req_id_per_token, captured.cache_seq_lens_per_token
        ).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observed_ids.copy_(captured.req_id_per_token)
        observed_lens.copy_(captured.cache_seq_lens_per_token)
        if dcp_size == 1:
            observed_attention.copy_(
                attention(captured.req_id_per_token, captured.cache_seq_lens_per_token)
            )
    query_splits = [[1, 7, 4, 0], [7, 1, 4, 0]]
    if not mixed_prefill:
        query_splits += [[4, 1, 7, 0], [1, 1, 1, 0]]
    for query_lens in query_splits:
        starts = torch.tensor(
            [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
        )
        common.query_start_loc.copy_(starts)
        if sum(query_lens) == 3:
            common.query_start_loc_cpu.copy_(starts)
        seq_lens = torch.tensor(
            [260 + query_lens[0], 520 + query_lens[1], 780 + query_lens[2], 0],
            dtype=torch.int32,
        )
        common.seq_lens.copy_(seq_lens)
        metadata = builder.build(
            0,
            common,
            selector_state_slot_ids=torch.tensor(
                [2, 0, 1, -1], dtype=torch.int32, device=device
            ),
            selector_state_is_fresh=torch.zeros(4, dtype=torch.bool, device=device),
            selector_num_accepted_tokens=torch.tensor(
                [3, 2, 1, 1], dtype=torch.int32, device=device
            ),
            selector_is_prefilling=common.is_prefilling.to(device),
        )
        assert (
            metadata.req_id_per_token.data_ptr() == captured.req_id_per_token.data_ptr()
        )
        assert (
            metadata.cache_seq_lens_per_token.data_ptr()
            == captured.cache_seq_lens_per_token.data_ptr()
        )
        if mixed_prefill:
            assert metadata.num_decodes == 2
            assert metadata.num_decode_tokens == 8
            assert metadata.prefill_query_lens_cpu.tolist() == [4, 0]
        torch.accelerator.synchronize()
        allocations = torch.accelerator.memory_stats()["allocation.all.allocated"]
        graph.replay()
        graph.replay()
        torch.accelerator.synchronize()
        assert (
            torch.accelerator.memory_stats()["allocation.all.allocated"] == allocations
        )
        expected_ids = torch.zeros(12, dtype=torch.int32)
        expected_lens = torch.zeros(12, dtype=torch.int32)
        for request, length in enumerate(query_lens):
            start = int(starts[request])
            expected_ids[start : start + length] = request
            expected_lens[start : start + length] = torch.arange(
                int(seq_lens[request]) - length + 1, int(seq_lens[request]) + 1
            )
        expected_lens = get_dcp_local_seq_lens(expected_lens, dcp_size, dcp_rank, 4)
        torch.testing.assert_close(observed_ids.cpu(), expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(observed_lens.cpu(), expected_lens, rtol=0, atol=0)
        if dcp_size == 1:
            expected_attention = attention(
                expected_ids.to(device), expected_lens.to(device)
            )
            torch.testing.assert_close(
                observed_attention, expected_attention, rtol=0, atol=0
            )


def _pool_reference(
    key: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(gate.float() + ape.float(), dim=0)
    pooled = (key.float() * weights).sum(dim=0).to(torch.bfloat16).float()
    rotated = _hadamard128(pooled).to(torch.bfloat16).float()
    scale = torch.exp2(
        torch.ceil(torch.log2(rotated.abs().max().clamp_min(1e-4) / 448))
    )
    quantized = (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale


def _read_cache_entry(
    cache: torch.Tensor, physical_page: int, page_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    page_stride = int(cache.stride(0))
    byte_view = cache.view(torch.uint8).reshape(-1)
    key_begin = physical_page * page_stride + page_offset * 128
    scale_begin = physical_page * page_stride + 64 * 128 + page_offset * 4
    key = byte_view[key_begin : key_begin + 128].view(torch.float8_e4m3fn)
    scale = byte_view[scale_begin : scale_begin + 4].view(torch.float32)
    return key, scale


def test_glm53_selector_lazily_caches_fp32_head_projection() -> None:
    hidden_size = 8
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.weights_proj = nn.Linear(hidden_size, 32, bias=False, dtype=torch.bfloat16)
    indexer._weights_proj_fp32 = None
    hidden = torch.randn(3, hidden_size, dtype=torch.bfloat16)

    expected = torch.nn.functional.linear(
        hidden.float(), indexer.weights_proj.weight.float()
    )
    actual = indexer._project_head_weights(hidden)

    torch.testing.assert_close(actual, expected)
    assert indexer._weights_proj_fp32 is not None
    pointer = indexer._weights_proj_fp32.data_ptr()
    indexer._project_head_weights(hidden)
    assert indexer._weights_proj_fp32.data_ptr() == pointer


def test_glm53_mla_spec_scales_fp8_index_tail_with_manager_block() -> None:
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=528,
        page_tail_bytes_per_token=33,
        model_version="glm5_next",
    )

    assert spec.unpadded_page_size_bytes == 256 * 528
    assert spec.page_size_bytes == 256 * (528 + 33)
    promoted = spec.copy_with_new_block_size(2304)
    assert promoted.unpadded_page_size_bytes == 2304 * 528
    assert promoted.page_size_bytes == 2304 * (528 + 33)


def test_glm53_selector_prefill_lengths_do_not_require_attention_backend() -> None:
    metadata = B12xMLASparseMetadata(
        num_reqs=2,
        max_query_len=3,
        max_seq_len=8,
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        slot_mapping=torch.empty(4, dtype=torch.int64),
        block_table=torch.empty((2, 1), dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 1, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4, 8], dtype=torch.int32),
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        prefill=None,
        prefill_query_lens_cpu=torch.tensor([3], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
    )

    assert metadata.prefill is None
    assert metadata.prefill_query_lens_cpu.tolist() == [3]
    assert metadata.prefill_seq_lens_cpu is not None
    assert metadata.prefill_seq_lens_cpu.tolist() == [8]


@pytest.mark.parametrize(
    ("seq_len", "expected_pages"),
    [(0, 1), (3, 1), (4, 1), (259, 1), (260, 2), (32768, 128)],
)
def test_glm53_active_index_pages_cover_completed_pools(
    seq_len: int, expected_pages: int
) -> None:
    assert Glm5NextPooledIndexer._active_index_page_count(seq_len) == expected_pages


def test_glm53_packed_c4_metadata_uses_parent_stride() -> None:
    device = _require_glm_gpu()
    source = torch.tensor([[3, 1], [7, -1]], dtype=torch.int32, device=device)
    expanded = torch.empty((2, 4), dtype=torch.int32, device=device)
    expand_c4_block_table(
        source,
        expanded,
        rows=2,
        subpages_per_parent=2,
        parent_stride_pages=153,
    )
    torch.testing.assert_close(
        expanded.cpu(),
        torch.tensor([[459, 460, 153, 154], [1071, 1072, -1, -1]], dtype=torch.int32),
    )

    gathered = torch.empty((3, 4), dtype=torch.int32, device=device)
    gather_c4_block_table_rows(
        expanded,
        torch.tensor([1, 0, 1], dtype=torch.int32, device=device),
        gathered,
    )
    torch.testing.assert_close(gathered.cpu(), expanded[[1, 0, 1]].cpu())

    positions = torch.tensor([0, 3, 4, 7, 8], dtype=torch.int64, device=device)
    lengths = torch.empty(5, dtype=torch.int32, device=device)
    pool_seq_lens(positions, lengths)
    torch.testing.assert_close(
        lengths.cpu(), torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32)
    )

    dcp_positions = torch.tensor(
        [0, 3, 7, 11, 15, 19, 31], dtype=torch.int64, device=device
    )
    expected_by_rank = (
        [0, 1, 1, 1, 1, 2, 2],
        [0, 0, 1, 1, 1, 1, 2],
        [0, 0, 0, 1, 1, 1, 2],
        [0, 0, 0, 0, 1, 1, 2],
    )
    for rank, expected in enumerate(expected_by_rank):
        local_lengths = torch.empty(7, dtype=torch.int32, device=device)
        pool_seq_lens(
            dcp_positions,
            local_lengths,
            dcp_size=4,
            dcp_rank=rank,
            pool_interleave=1,
        )
        torch.testing.assert_close(
            local_lengths.cpu(), torch.tensor(expected, dtype=torch.int32)
        )


@pytest.mark.parametrize(("rows", "requests"), [(1, 1), (7, 4), (32, 32)])
@pytest.mark.parametrize(
    ("dcp_size", "dcp_rank", "pool_interleave"),
    [(1, 0, 1), (4, 2, 1), (4, 3, 2)],
)
def test_glm53_c4_decode_metadata_matches_reference(
    rows: int,
    requests: int,
    dcp_size: int,
    dcp_rank: int,
    pool_interleave: int,
) -> None:
    device = _require_glm_gpu()
    source_width = 5
    subpages_per_parent = 9
    parent_stride_pages = 37
    source = torch.arange(
        requests * source_width, dtype=torch.int32, device=device
    ).reshape(requests, source_width)
    source[0, -1] = -1
    source[-1, 0] = 58_000_000
    request_ids = torch.arange(rows, dtype=torch.int32, device=device) % requests
    positions = torch.arange(rows, dtype=torch.int64, device=device) * 257 + 3

    expanded = torch.empty(
        (requests, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    expected_table = torch.empty(
        (rows, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    expected_seq_lens = torch.empty(rows, dtype=torch.int32, device=device)
    actual_table = torch.empty_like(expected_table)
    actual_seq_lens = torch.empty_like(expected_seq_lens)

    expand_c4_block_table(
        source,
        expanded,
        rows=requests,
        subpages_per_parent=subpages_per_parent,
        parent_stride_pages=parent_stride_pages,
    )
    gather_c4_block_table_rows(expanded, request_ids, expected_table)
    pool_seq_lens(
        positions,
        expected_seq_lens,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        pool_interleave=pool_interleave,
    )
    prepare_c4_decode_metadata(
        source,
        request_ids,
        positions,
        actual_table,
        actual_seq_lens,
        subpages_per_parent=subpages_per_parent,
        parent_stride_pages=parent_stride_pages,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        pool_interleave=pool_interleave,
    )

    torch.testing.assert_close(actual_table, expected_table, rtol=0, atol=0)
    torch.testing.assert_close(actual_seq_lens, expected_seq_lens, rtol=0, atol=0)


def test_glm53_c4_decode_metadata_graph_replays_live_inputs() -> None:
    device = _require_glm_gpu()
    rows = 7
    requests = 4
    source_width = 5
    subpages_per_parent = 9
    parent_stride_pages = 37
    source = torch.arange(
        requests * source_width, dtype=torch.int32, device=device
    ).reshape(requests, source_width)
    request_ids = torch.arange(rows, dtype=torch.int32, device=device) % requests
    positions = torch.arange(rows, dtype=torch.int64, device=device) * 4 + 3
    output_table = torch.empty(
        (rows, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    output_seq_lens = torch.empty(rows, dtype=torch.int32, device=device)

    def prepare() -> None:
        prepare_c4_decode_metadata(
            source,
            request_ids,
            positions,
            output_table,
            output_seq_lens,
            subpages_per_parent=subpages_per_parent,
            parent_stride_pages=parent_stride_pages,
            dcp_size=4,
            dcp_rank=2,
            pool_interleave=2,
        )

    prepare()
    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        prepare()

    source.add_(100)
    source[1, -1] = -1
    request_ids.copy_(
        torch.tensor([3, 1, 2, 0, 3, 2, 1], dtype=torch.int32, device=device)
    )
    positions.add_(4096)
    output_table.fill_(37)
    output_seq_lens.fill_(37)
    graph.replay()
    torch.accelerator.synchronize()

    expanded = torch.empty(
        (requests, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    expected_table = torch.empty_like(output_table)
    expected_seq_lens = torch.empty_like(output_seq_lens)
    expand_c4_block_table(
        source,
        expanded,
        rows=requests,
        subpages_per_parent=subpages_per_parent,
        parent_stride_pages=parent_stride_pages,
    )
    gather_c4_block_table_rows(expanded, request_ids, expected_table)
    pool_seq_lens(
        positions,
        expected_seq_lens,
        dcp_size=4,
        dcp_rank=2,
        pool_interleave=2,
    )
    torch.testing.assert_close(output_table, expected_table, rtol=0, atol=0)
    torch.testing.assert_close(output_seq_lens, expected_seq_lens, rtol=0, atol=0)

    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated


def test_glm53_physical_selection_provider_is_explicit() -> None:
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.dcp_world_size = 1
    indexer._emit_physical_selection = True
    indexer.topk_indices_buffer = torch.empty((8, 2051), dtype=torch.int32)
    indexer.scratch = Glm5NextIndexerScratch(8, 1, torch.device("cpu"))

    selected = indexer.get_b12x_physical_selection(
        num_tokens=3,
        num_prefills=0,
        num_decode_tokens=3,
    )
    assert selected is not None
    assert selected[0].shape == (3, 2051)
    assert selected[1].shape == (3,)
    assert (
        indexer.get_b12x_physical_selection(
            num_tokens=3,
            num_prefills=1,
            num_decode_tokens=2,
        )
        is None
    )

    indexer._emit_physical_selection = False
    assert (
        indexer.get_b12x_physical_selection(
            num_tokens=3,
            num_prefills=0,
            num_decode_tokens=3,
        )
        is None
    )


@pytest.mark.parametrize(
    "main_blocks, error",
    [(0, "main cache is not bound"), (1, "physical selection has not been declared")],
)
def test_glm53_missing_physical_plan_does_not_mutate_recurrent_cache(
    monkeypatch, main_blocks, error
) -> None:
    from vllm.models.glm5next.nvidia import pooled_indexer as module

    hidden = torch.zeros((1, 128))
    metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        num_decode_tokens=1,
        num_decodes=1,
        max_query_len=1,
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        selector_state_slot_ids=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
    )
    indexer = SimpleNamespace(
        max_tokens=1,
        wq_b=lambda value: (torch.zeros((1, 32 * 128)), None),
        wk=lambda value: (hidden, None),
        k_norm=lambda value: value,
        index_kpool_compress_gate=torch.zeros((128, 128)),
        index_kpool_compress_ape=torch.zeros((4, 128)),
        _project_head_weights=lambda value: torch.zeros((1, 32)),
        scratch=Glm5NextIndexerScratch(1, 1, torch.device("cpu")),
        main_layer_name="attention",
        _index_cache=torch.zeros((1, 64, 132), dtype=torch.uint8),
        _tail=torch.zeros((1, 2, 4, 128)),
        _state_slots=Glm5NextPooledIndexer._state_slots,
        _parent_table_width=1,
        _parent_stride_pages=1,
        block_size=256,
        dcp_world_size=1,
        _emit_physical_selection=True,
        _main_cache_num_blocks=main_blocks,
        _physical_selection_plan=None,
    )
    monkeypatch.setattr(module, "fwht128_quant_fp8", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"attention": metadata}),
    )

    def reject_cache_update(*args, **kwargs):
        pytest.fail("Missing physical-selection resources must not update the cache")

    monkeypatch.setattr(module, "update_decode_pools", reject_cache_update)
    with pytest.raises(RuntimeError, match=error):
        Glm5NextPooledIndexer.forward(
            indexer, hidden, hidden, torch.zeros(1, dtype=torch.int64), None
        )


def _packed_main_cache(
    *,
    device: torch.device,
    blocks: int,
    layers: int,
    block_size: int,
    layer: int,
    record_bytes: int = 528,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic_page_bytes = block_size * record_bytes
    content_page_bytes = ((semantic_page_bytes + 8447) // 8448) * 8448
    page_bytes = content_page_bytes + block_size * 33
    raw = torch.zeros(blocks * layers * page_bytes, dtype=torch.uint8, device=device)
    main = torch.as_strided(
        raw,
        size=(blocks, block_size, record_bytes),
        stride=(layers * page_bytes, record_bytes, 1),
        storage_offset=layer * page_bytes,
    )
    return raw, main


@pytest.mark.parametrize("release_first_pool", [False, True])
def test_glm53_selector_constructs_with_matching_preparation_heads(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config, release_first_pool
) -> None:
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    dsa_indexer = pytest.importorskip("b12x.attention.dsa_indexer")
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dsa_indexer, "is_supported", lambda: True)
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=2),
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        ),
        speculative_config=None,
    )
    indexer = Glm5NextPooledIndexer(
        config,
        SimpleNamespace(
            index_topk=2048,
            index_n_heads=32,
            index_head_dim=128,
            index_kpool=4,
            qk_rope_head_dim=0,
        ),
        hidden_size=128,
        q_lora_rank=128,
        quant_config=None,
        cache_config=SimpleNamespace(block_size=256),
        topk_indices_buffer=torch.empty(8, 2051, dtype=torch.int32),
        pool_topk_indices_buffer=torch.empty(8, 512, dtype=torch.int32),
        main_layer_name="model.layers.0.self_attn",
        prefix="model.layers.0.self_attn.indexer",
    )
    if release_first_pool:
        _, initial = _packed_main_cache(
            device=torch.device("cpu"), blocks=1, layers=1, block_size=256, layer=0
        )
        indexer.bind_main_kv_cache(initial)
        indexer.indexer_op._plans[("decode", 8)] = object()
        indexer.unbind_main_kv_cache()
        assert indexer._index_cache is None
        assert indexer._main_cache_num_blocks == 0
        assert indexer.indexer_op._index_cache is None
        assert indexer.indexer_op._plans == {}
    _, main = _packed_main_cache(
        device=torch.device("cpu"), blocks=2, layers=1, block_size=256, layer=0
    )
    assert indexer._physical_selection_plan is None
    indexer._physical_selection_plan = object()
    indexer.bind_main_kv_cache(main)

    assert indexer.indexer_op._num_q_heads == indexer.scratch.q_fp8.shape[1] == 32
    assert indexer._physical_selection_plan is None
    indexer._physical_selection_plan = object()
    indexer.unbind_main_kv_cache()
    assert indexer._physical_selection_plan is None
    assert indexer._index_cache is None
    assert indexer._main_cache_num_blocks == 0


def test_glm53_packed_tail_accepts_nvfp4_main_record() -> None:
    _, main = _packed_main_cache(
        device=torch.device("cpu"),
        blocks=2,
        layers=3,
        block_size=3328,
        layer=1,
        record_bytes=304,
    )

    index_cache, subpages, parent_stride_pages = (
        Glm5NextPooledIndexer._index_cache_view(main)
    )

    assert subpages == 13
    assert parent_stride_pages == 399
    assert index_cache.stride() == (8448, 132, 1)


def test_glm53_decode_table_capacity_uses_batched_token_limit() -> None:
    device = _require_glm_gpu()
    _, main = _packed_main_cache(
        device=device, blocks=2, layers=3, block_size=256, layer=1
    )
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.max_tokens = 128
    indexer.max_seqs = 16
    indexer.max_model_len = 4096
    indexer.dcp_world_size = 1
    published = {}

    def publish(*args, **kwargs):
        published.update(kwargs)

    indexer.indexer_op = SimpleNamespace(
        max_model_len=4096 // 4, set_b12x_index_cache=publish
    )
    indexer.scratch = Glm5NextIndexerScratch(
        indexer.max_tokens, indexer.max_seqs, device
    )

    indexer.bind_main_kv_cache(main)

    assert indexer.scratch.decode_block_table.shape[0] == indexer.max_tokens
    assert indexer.scratch.decode_block_table.shape[0] > indexer.max_seqs * (5 + 1)
    assert indexer.indexer_op.max_model_len == 4096 // 4
    assert published["num_q_heads"] == 32
    assert (
        published["max_page_table_width"] == indexer.scratch.pool_block_table.shape[1]
    )


def test_b12x_c4_declaration_matches_glm_query_and_page_table_layout() -> None:
    from vllm.models.deepseek_v4.nvidia import b12x_indexer as c4

    indexer = c4.B12xC4SparseIndexer.__new__(c4.B12xC4SparseIndexer)
    nn.Module.__init__(indexer)
    indexer._index_num_q_heads = 32
    indexer._index_max_page_table_width = 4104
    indexer._index_cache = torch.empty((1, 64, 132), dtype=torch.uint8)
    indexer.topk_tokens = 512
    fake_module = SimpleNamespace(
        invocation_from_descriptors=lambda caps, operands: operands
    )
    indexer._b12x_indexer = fake_module
    caps = SimpleNamespace(
        max_q_rows=8192, max_page_table_width=4104, num_q_heads=32, mode="prefill"
    )

    invocation = indexer._invocation(caps, scores=False)

    assert invocation["q_fp8"]["shape"][1:] == (32, 128)
    assert invocation["page_table"]["shape"] == (8192, 4104)
    assert invocation["page_table"]["strides"] == (0, 1)


@pytest.mark.parametrize("page_width", [1, 4, 32])
def test_c4_preparation_respects_bound_page_capacity(monkeypatch, page_width):
    """A DCP-local table can address fewer keys than the backing cache holds."""
    from b12x.attention import dsa_indexer

    from vllm.models.deepseek_v4.nvidia import b12x_indexer as c4
    from vllm.utils.b12x import B12xWorkload

    monkeypatch.setattr(c4, "_require_b12x_indexer", lambda: dsa_indexer)
    owner = c4.B12xC4SparseIndexer(
        None,
        128,
        "ue8m0",
        512,
        128,
        1024,
        1024,
        torch.empty((8, 512), dtype=torch.int32),
        skip_k_cache_insert=True,
        compress_ratio=4,
    )
    cache = torch.full((16, 64, 132), 0x5A, dtype=torch.uint8)
    owner.set_b12x_index_cache(cache, num_q_heads=32, max_page_table_width=page_width)
    workload = B12xWorkload(
        stage="state",
        token_counts=(1, 8),
        fixed_token_counts=(1,),
        output_dtype=torch.bfloat16,
        max_tokens=8,
        max_seqs=1,
        max_model_len=1024,
    )
    bindings = []

    def bind(**kwargs):
        bindings.append(kwargs)
        return kwargs

    state = SimpleNamespace(layout=SimpleNamespace(scratch_specs=lambda: ()), bind=bind)
    (unit,) = owner.get_b12x_preparation_units(owner, workload)
    for request in unit.requests:
        assert callable(request.prepare_call)
        call = request.prepare_call(state)
        metadata = bindings[-1]
        pages = metadata["real_page_table"]
        live_keys = min(1024, page_width * 64)
        live_pages = live_keys // 64
        assert pages.shape[1] == page_width
        assert metadata["cache_seqlens_int32"].tolist() == [live_keys] * pages.shape[0]
        assert metadata["active_width"].item() == live_keys
        assert pages[0, :live_pages].tolist() == list(range(live_pages))
        assert pages[0, live_pages:].count_nonzero().item() == 0
        if metadata["shared_page_table"]:
            assert pages.stride(0) == 0
        assert call.restore is not None
        call.restore()
    assert len(bindings) == 2  # Decode and prefill use the same capacity bound.
    assert torch.all(cache == 0x5A)


def test_b12x_c4_replans_after_page_table_width_changes() -> None:
    from vllm.models.deepseek_v4.nvidia import b12x_indexer as c4

    plans = []
    indexer = c4.B12xC4SparseIndexer.__new__(c4.B12xC4SparseIndexer)
    nn.Module.__init__(indexer)
    indexer._index_num_q_heads = 32
    indexer._index_max_page_table_width = 4
    indexer._score_output = False
    indexer._index_cache = torch.empty((1, 64, 132), dtype=torch.uint8)
    indexer.topk_tokens = 512
    indexer.max_model_len = 256
    indexer.topk_indices_buffer = torch.empty((1, 512), dtype=torch.int32)
    indexer._plans = {}

    def make_plan(caps, invocation):
        plans.append(caps)
        return caps

    indexer._b12x_indexer = SimpleNamespace(
        Caps=lambda **caps: SimpleNamespace(**caps),
        plan=make_plan,
        invocation_from_descriptors=lambda caps, operands: operands,
    )

    old_plan = indexer._plan_for("decode", 1)
    indexer.set_b12x_index_cache(
        indexer._index_cache, num_q_heads=32, max_page_table_width=8
    )
    new_plan = indexer._plan_for("decode", 1)

    assert old_plan is not new_plan
    assert new_plan.max_page_table_width == 8
    assert len(plans) == 2


def test_deepseek_c4_default_page_table_width_is_unchanged() -> None:
    from vllm.models.deepseek_v4.nvidia import b12x_indexer as c4

    indexer = c4.B12xC4SparseIndexer.__new__(c4.B12xC4SparseIndexer)
    indexer.max_model_len = 262_144
    indexer._index_max_page_table_width = None

    assert indexer._max_page_table_width == 4096


@pytest.mark.parametrize("autotune", [False, True])
def test_glm53_physical_selection_prepares_before_resolution_freeze(
    autotune: bool, monkeypatch
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.attention.sparse_mla import (
        expand_pooled_topk_to_physical_slots,
        pooled_selection,
    )
    from b12x.preparation import PreparationSession

    device = _require_glm_gpu()
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.block_size = 2048
    indexer._parent_table_width = 512
    indexer._main_cache_num_blocks = 1024
    indexer._expand_pooled_topk_to_physical_slots = expand_pooled_topk_to_physical_slots
    output = torch.empty((32, 2051), dtype=torch.int32, device=device)
    counts = torch.empty(32, dtype=torch.int32, device=device)
    indexer.max_tokens = 32
    indexer.prefix = "model.layers.3.indexer"
    indexer.topk_indices_buffer = output
    indexer.scratch = Glm5NextIndexerScratch(32, 1, device)
    indexer.scratch.physical_active_counts = counts
    request = indexer.get_b12x_physical_selection_preparation_request()
    session = PreparationSession(device=device, autotune=autotune, compile_workers=0)
    session.prepare((request,))
    assert request.plan.prepared is not None

    def unexpected_resolution(*args, **kwargs):
        pytest.fail("pooled selection resolved a kernel after preparation")

    monkeypatch.setattr(
        pooled_selection._expand_pooled_topk_to_physical_slots_kernel,
        "run",
        unexpected_resolution,
    )
    with kernel_resolution_guard("prepared physical-selection replay"):
        for rows in (0, 1, 4, 32):
            call = indexer.make_b12x_physical_selection_prepare_call(
                output[:rows], counts[:rows]
            )
            call.run()
            assert torch.all(output[:rows, 0] == 0)
            assert torch.all(output[:rows, 1:] == -1)
            assert torch.all(counts[:rows] == 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call.run()
        output.fill_(37)
        counts.fill_(37)
        graph.replay()
        assert torch.all(output[:, 0] == 0)
        assert torch.all(output[:, 1:] == -1)
        assert torch.all(counts == 1)


def test_glm53_selector_capacity_tracks_auto_fit_max_model_len() -> None:
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.max_model_len = 1_048_576
    indexer.indexer_op = SimpleNamespace(max_model_len=262_144)

    assert indexer._aligned_max_seq_len == 1_048_576

    indexer.update_max_model_len(1_985)
    assert indexer._aligned_max_seq_len == 1_988
    assert indexer.indexer_op.max_model_len == 497


def test_glm53_indexer_scratch_rebind_preserves_same_geometry_storage() -> None:
    scratch = Glm5NextIndexerScratch(128, 4, torch.device("cpu"))
    scratch.bind_tables(32, torch.device("cpu"))
    pointers = {name: buffer.data_ptr() for name, buffer in scratch.named_buffers()}
    scratch.decode_block_table.fill_(53)
    scratch.bind_tables(32, torch.device("cpu"))
    assert {
        name: buffer.data_ptr() for name, buffer in scratch.named_buffers()
    } == pointers
    assert torch.all(scratch.decode_block_table == 53)
    scratch.bind_tables(16, torch.device("cpu"))
    assert scratch.decode_block_table.shape == (128, 16)
    assert scratch.pool_block_table.shape == (4, 16)
    assert scratch.q_fp8.data_ptr() == pointers["q_fp8"]
    assert not scratch.state_dict()


def test_glm53_shared_scratch_keeps_layer_checkpoints_and_draft_selection_private(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        indexer_module, "ReplicatedLinear", lambda *a, **k: nn.Identity()
    )
    monkeypatch.setattr(
        indexer_module, "B12xC4SparseIndexer", lambda *a, **k: SimpleNamespace()
    )
    monkeypatch.setattr(
        indexer_module,
        "get_b12x_sparse_mla",
        lambda: SimpleNamespace(expand_pooled_topk_to_physical_slots=lambda *a: None),
    )
    config = SimpleNamespace(
        index_topk=2048,
        index_n_heads=32,
        index_head_dim=128,
        index_kpool=4,
        qk_rope_head_dim=0,
    )
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=128, max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=4096),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=4
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    scratch = Glm5NextIndexerScratch(128, 4, torch.device("cpu"))
    target_ids = torch.zeros((128, 2051), dtype=torch.int32)
    target_pools = torch.zeros((128, 512), dtype=torch.int32)

    def make(name, shared):
        return Glm5NextPooledIndexer(
            vllm_config,
            config,
            8,
            8,
            None,
            SimpleNamespace(block_size=2048),
            target_ids if shared else torch.full_like(target_ids, 42),
            target_pools if shared else torch.full_like(target_pools, 42),
            main_layer_name=name,
            prefix=name,
            scratch=scratch if shared else None,
        )

    first, second, draft = (
        make("target.0", True),
        make("target.1", True),
        make("mtp", False),
    )
    assert first.scratch is second.scratch
    assert draft.scratch is not scratch
    for indexer, value in ((first, 1), (second, 2), (draft, 3)):
        indexer._tail.fill_(value)
        indexer.snapshot_speculative_interval_starts()
    scratch.q_fp8.view(torch.uint8).fill_(255)
    target_ids.fill_(-1)
    first._tail.zero_()
    first.restore_speculative_interval_starts()
    for indexer, value in ((first, 1), (second, 2), (draft, 3)):
        assert torch.all(indexer._tail == value)
        assert torch.all(indexer._tail_snapshot == value)
    assert torch.all(draft.topk_indices_buffer == 42)
    assert torch.all(draft.pool_topk_indices_buffer == 42)


def test_glm53_shared_indexer_scratch_graph_consumes_each_live_query() -> None:
    device = _require_glm_gpu()
    scratch = Glm5NextIndexerScratch(8, 1, device)
    inputs = [
        torch.randn((8, 32, 128), device=device, dtype=torch.bfloat16) for _ in range(2)
    ]
    outputs = [torch.empty_like(scratch.q_fp8) for _ in range(2)]
    scales = [torch.empty_like(scratch.q_scale) for _ in range(2)]

    def run() -> None:
        for query, output, scale in zip(inputs, outputs, scales):
            glm_kpool.fwht128_quant_fp8(
                query.view(-1, 128),
                scratch.q_fp8.view(-1, 128),
                scratch.q_scale.view(-1),
            )
            output.copy_(scratch.q_fp8)
            scale.copy_(scratch.q_scale)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        for query in inputs:
            query.normal_()
        scratch.q_fp8.view(torch.uint8).fill_(255)
        scratch.q_scale.fill_(float("nan"))
        graph.replay()
        for query, output, scale in zip(inputs, outputs, scales):
            reference_q = torch.empty_like(output)
            reference_scale = torch.empty_like(scale)
            glm_kpool.fwht128_quant_fp8(
                query.view(-1, 128),
                reference_q.view(-1, 128),
                reference_scale.view(-1),
            )
            torch.testing.assert_close(output, reference_q, rtol=0, atol=0)
            torch.testing.assert_close(scale, reference_scale, rtol=0, atol=0)


def test_glm53_parent_table_width_tracks_dcp_sharding() -> None:
    max_model_len = 524288
    block_size = 2304

    assert Glm5NextPooledIndexer._max_parent_table_width(
        max_model_len,
        block_size,
        dcp_world_size=1,
    ) == math.ceil(max_model_len / block_size)
    assert Glm5NextPooledIndexer._max_parent_table_width(
        max_model_len,
        block_size,
        dcp_world_size=4,
    ) == math.ceil(max_model_len / (block_size * 4))
    assert (
        Glm5NextPooledIndexer._max_parent_table_width(
            block_size,
            block_size,
            dcp_world_size=1,
        )
        == 1
    )


def test_glm53_packed_tail_reuses_c4_page_contract() -> None:
    device = _require_glm_gpu()
    _, main = _packed_main_cache(
        device=device, blocks=2, layers=3, block_size=512, layer=1
    )
    index_cache, subpages, parent_stride_pages = (
        Glm5NextPooledIndexer._index_cache_view(main)
    )
    assert subpages == 2
    assert parent_stride_pages == 102
    assert index_cache.stride() == (8448, 132, 1)

    generator = torch.Generator(device=device).manual_seed(55)
    key = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    update_decode_pools(
        index_cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        torch.arange(512, 516, dtype=torch.int64, device=device),
        torch.arange(4, dtype=torch.int64, device=device),
        1,
        model_block_size=512,
        parent_stride_pages=parent_stride_pages,
    )
    actual_key, actual_scale = _read_cache_entry(index_cache, parent_stride_pages, 0)
    expected_key, expected_scale = _pool_reference(key, gate, ape)
    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


def test_glm53_packed_tail_scores_through_existing_c4_indexer() -> None:
    device = _require_glm_gpu()
    _, main = _packed_main_cache(
        device=device, blocks=2, layers=3, block_size=512, layer=1
    )
    index_cache, _, parent_stride_pages = Glm5NextPooledIndexer._index_cache_view(main)
    virtual_page = parent_stride_pages
    page = index_cache[virtual_page]
    quant = page.as_strided(
        (64, 128), (128, 1), storage_offset=page.storage_offset()
    ).view(torch.float8_e4m3fn)
    scales = page.as_strided(
        (64 * 4,),
        (1,),
        storage_offset=page.storage_offset() + 64 * 128,
    ).view(torch.float32)
    quant.zero_()
    scales.fill_(1.0)
    quant[0].fill_(1.0)
    quant[1].fill_(2.0)

    from vllm.utils.b12x import get_b12x_dsa_indexer

    module = get_b12x_dsa_indexer()
    assert module is not None
    q = torch.ones((1, 32, 128), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.ones((1, 32), dtype=torch.float32, device=device)
    block_table = torch.tensor([[virtual_page]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([2], dtype=torch.int32, device=device)
    caps = module.Caps(
        device=device,
        num_q_heads=32,
        max_q_rows=1,
        max_page_table_width=1,
        topk=512,
        mode="decode",
    )
    output = torch.empty((1, 512), dtype=torch.int32, device=device)
    operands = dict(
        q_fp8=q,
        query_weights=weights,
        index_k_cache=_flatten_index_cache(index_cache),
        page_table=block_table,
        cache_lengths=seq_lens,
        active_width=torch.ones(1, dtype=torch.int32, device=device),
        output_indices=output,
    )
    plan = module.plan(
        caps, invocation=module.invocation_from_tensors(caps, **operands)
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in plan.scratch_specs()
    )
    binding = module.bind(plan, scratch=scratch, **operands)
    module.run(binding)
    torch.accelerator.synchronize()
    assert set(output[0, :2].tolist()) == {0, 1}
    assert torch.all(output[0, 2:] == -1)

    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        module.run(binding)
    graph.replay()
    torch.accelerator.synchronize()
    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated
    assert set(output[0, :2].tolist()) == {0, 1}


def test_b12x_c4_indexers_name_their_preparation_requests_per_layer(
    monkeypatch,
) -> None:
    """GLM builds one C4 indexer per sparse layer without a prefixed k_cache.

    Startup preparation rejects the whole model when two requests share a name,
    so each indexer must carry its layer's prefix into its request names.
    """
    from vllm.models.deepseek_v4.nvidia import b12x_indexer as c4

    monkeypatch.setattr(c4, "_require_b12x_indexer", lambda: SimpleNamespace())

    def make(k_cache, prefix=None):
        return c4.B12xC4SparseIndexer(
            k_cache,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=4,
            head_dim=128,
            max_model_len=64,
            max_total_seq_len=64,
            topk_indices_buffer=torch.empty((8, 4), dtype=torch.int32),
            skip_k_cache_insert=True,
            compress_ratio=4,
            prefix=prefix,
        )

    first = make(None, prefix="model.layers.3.self_attn.indexer")
    second = make(None, prefix="model.layers.7.self_attn.indexer")

    assert first._request_name("decode", 1) == (
        "model.layers.3.self_attn.indexer.c4_indexer.decode.m1"
    )
    assert first._request_name("decode", 1) != second._request_name("decode", 1)

    prefixed_cache = make(SimpleNamespace(prefix="layers.1.attn", kv_cache=None))
    assert prefixed_cache._request_name("prefill", 8) == (
        "layers.1.attn.c4_indexer.prefill.m8"
    )
    anonymous = [make(None), make(None)]
    assert anonymous[0]._request_name("decode", 1) != anonymous[1]._request_name(
        "decode", 1
    )


def test_c4_preparation_names_do_not_collide_without_cache_owner(monkeypatch):
    from b12x.attention import dsa_indexer

    from vllm.models.deepseek_v4.nvidia import b12x_indexer
    from vllm.utils.b12x import B12xWorkload

    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: dsa_indexer)
    workload = B12xWorkload(
        stage="state",
        token_counts=(1, 8),
        fixed_token_counts=(1,),
        output_dtype=torch.bfloat16,
        max_tokens=8,
        max_seqs=1,
        max_model_len=1024,
    )
    owners, names = [], []
    for _ in range(2):
        owner = b12x_indexer.B12xC4SparseIndexer(
            None,
            128,
            "ue8m0",
            512,
            128,
            1024,
            1024,
            torch.empty((8, 512), dtype=torch.int32),
            skip_k_cache_insert=True,
            compress_ratio=4,
        )
        owners.append(owner)
        owner.set_b12x_index_cache(
            torch.empty((16, 64, 132), dtype=torch.uint8), num_q_heads=32
        )
        (unit,) = owner.get_b12x_preparation_units(owner, workload)
        request_names = [request.name for request in unit.requests]
        assert request_names
        names.extend(request_names)
    assert len(names) == len(set(names))


def test_glm53_pool_write_matches_fp8_reference() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(53)
    key = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    slots = torch.tensor([-1, -1, -1, 0], dtype=torch.int64, device=device)

    update_decode_pools(
        cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        slots,
        torch.arange(4, dtype=torch.int64, device=device),
        1,
    )
    actual_key, actual_scale = _read_cache_entry(cache, 0, 0)
    expected_key, expected_scale = _pool_reference(key, gate, ape)

    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


def test_glm53_decode_tail_completes_the_same_pool_as_prefill() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(54)
    key = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    state_slots = torch.zeros((1,), dtype=torch.int32, device=device)

    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 3], dtype=torch.int32, device=device),
        key[:3],
        gate[:3],
        ape,
        torch.full((3,), -1, dtype=torch.int64, device=device),
        torch.arange(3, dtype=torch.int64, device=device),
        1,
    )
    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        key[3:],
        gate[3:],
        ape,
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.tensor([3], dtype=torch.int64, device=device),
        1,
    )

    actual_key, actual_scale = _read_cache_entry(cache, 0, 0)
    expected_key, expected_scale = _pool_reference(key, gate, ape)
    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


@pytest.mark.parametrize("tail_capacity", [4, 12])
def test_glm53_decode_writer_matches_parallel_prefill_writer(
    tail_capacity: int,
) -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(56)
    key = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    prefill_cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    decode_cache = torch.zeros_like(prefill_cache)
    prefill_tail = torch.zeros(
        (1, 2, tail_capacity, 128), dtype=torch.bfloat16, device=device
    )
    decode_tail = torch.zeros_like(prefill_tail)
    state_slots = torch.zeros(1, dtype=torch.int32, device=device)

    update_decode_pools(
        prefill_cache,
        prefill_tail,
        state_slots,
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        torch.tensor([-1, -1, -1, 3, -1, -1, -1, 7], device=device),
        torch.arange(8, dtype=torch.int64, device=device),
        1,
        num_decode_requests=0,
        max_query_len=8,
        model_block_size=256,
        parent_stride_pages=1,
    )
    for position in range(8):
        update_decode_pools(
            decode_cache,
            decode_tail,
            state_slots,
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            key[position : position + 1],
            gate[position : position + 1],
            ape,
            torch.tensor(
                [position if position % 4 == 3 else -1],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor([position], dtype=torch.int64, device=device),
            1,
            model_block_size=256,
            parent_stride_pages=1,
        )

    assert torch.equal(decode_cache, prefill_cache)
    assert torch.equal(decode_tail, prefill_tail)


@pytest.mark.parametrize("tail_capacity", [4, 12])
def test_glm53_parallel_prefill_preserves_boundary_tail_and_state_slots(
    tail_capacity: int,
) -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(5304)
    key = torch.randn(
        (10, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (10, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    initial_tail = torch.randn(
        (2, 2, tail_capacity, 128),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    sequential_tail = initial_tail.clone()
    parallel_tail = initial_tail.clone()
    sequential_cache = torch.zeros((2, 64, 132), dtype=torch.uint8, device=device)
    parallel_cache = torch.zeros_like(sequential_cache)
    state_slots = torch.tensor([1, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [4099, 4100, 4101, 4102, 4103, 4099, 4100, 4101, 4102, 4103],
        dtype=torch.int64,
        device=device,
    )
    slot_mapping = torch.tensor(
        [3, -1, -1, -1, 7, 259, -1, -1, -1, 263],
        dtype=torch.int64,
        device=device,
    )
    common = dict(model_block_size=256, parent_stride_pages=1)

    update_decode_pools(
        sequential_cache,
        sequential_tail,
        state_slots,
        query_start_loc,
        key,
        gate,
        ape,
        slot_mapping,
        positions,
        2,
        **common,
    )
    update_decode_pools(
        parallel_cache,
        parallel_tail,
        state_slots,
        query_start_loc,
        key,
        gate,
        ape,
        slot_mapping,
        positions,
        2,
        num_decode_requests=0,
        max_query_len=5,
        **common,
    )

    assert torch.equal(parallel_cache, sequential_cache)
    assert torch.equal(parallel_tail, sequential_tail)


@pytest.mark.parametrize("resume_prefill", [False, True])
def test_glm53_selector_preserves_committed_pool_after_speculative_rejection(
    resume_prefill: bool,
) -> None:
    """Rejected rows must not overwrite the raw keys of the committed C4 tail."""
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(5306)
    key = torch.randn(
        (33, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (33, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.zeros((1, 2, 12, 128), dtype=torch.bfloat16, device=device)
    state_slots = torch.zeros(1, dtype=torch.int32, device=device)

    for start, end in ((0, 25), (25, 31)):
        positions = torch.arange(start, end, dtype=torch.int64, device=device)
        update_decode_pools(
            cache,
            tail,
            state_slots,
            torch.tensor([0, end - start], dtype=torch.int32, device=device),
            key[start:end],
            gate[start:end],
            ape,
            positions,
            positions,
            1,
            num_decode_requests=int(start != 0),
            max_query_len=end - start,
            model_block_size=256,
            parent_stride_pages=1,
        )

    # Only position 25 committed. Replace the rejected positions 26 and 27.
    positions = torch.tensor([26, 27], dtype=torch.int64, device=device)
    update_decode_pools(
        cache,
        tail.clone(),
        state_slots,
        torch.tensor([0, 2], dtype=torch.int32, device=device),
        key[31:],
        gate[31:],
        ape,
        positions,
        positions,
        1,
        num_decode_requests=int(not resume_prefill),
        max_query_len=2,
        model_block_size=256,
        parent_stride_pages=1,
    )
    actual_key, actual_scale = _read_cache_entry(cache, 0, 6)
    expected_key, expected_scale = _pool_reference(
        torch.cat((key[24:26], key[31:])),
        torch.cat((gate[24:26], gate[31:])),
        ape,
    )
    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


def test_glm53_parallel_prefill_ignores_invalid_dummy_slots() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(5305)
    key = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    initial_tail = torch.randn(
        (1, 2, 4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    tail = initial_tail.clone()

    update_decode_pools(
        cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        torch.full((8,), -1, dtype=torch.int64, device=device),
        torch.zeros(8, dtype=torch.int64, device=device),
        1,
        num_decode_requests=0,
        max_query_len=8,
        model_block_size=256,
        parent_stride_pages=1,
    )

    assert torch.count_nonzero(cache).item() == 0
    torch.testing.assert_close(tail[0, 0, 0], key[-1])
    torch.testing.assert_close(tail[0, 1, 0], gate[-1])
    assert torch.equal(tail[:, :, 1:], initial_tail[:, :, 1:])


def test_glm53_tail_state_isolated_between_requests() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(57)
    keys = torch.randn(
        (2, 4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gates = torch.randn(
        (2, 4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.full((2, 2, 4, 128), float("nan"), dtype=torch.bfloat16, device=device)
    state_slots = torch.tensor([1, 0], dtype=torch.int32, device=device)

    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
        torch.cat((keys[0, :3], keys[1, :3])),
        torch.cat((gates[0, :3], gates[1, :3])),
        ape,
        torch.full((6,), -1, dtype=torch.int64, device=device),
        torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device),
        2,
    )
    torch.testing.assert_close(tail[1, 0, :3], keys[0, :3])
    torch.testing.assert_close(tail[1, 1, :3], gates[0, :3])
    torch.testing.assert_close(tail[0, 0, :3], keys[1, :3])
    torch.testing.assert_close(tail[0, 1, :3], gates[1, :3])
    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        keys[:, 3],
        gates[:, 3],
        ape,
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        torch.tensor([3, 3], dtype=torch.int64, device=device),
        2,
    )

    for request in range(2):
        actual_key, actual_scale = _read_cache_entry(cache, 0, request)
        expected_key, expected_scale = _pool_reference(
            keys[request], gates[request], ape
        )
        assert torch.equal(actual_key, expected_key)
        torch.testing.assert_close(
            actual_scale, expected_scale.reshape(1), rtol=0, atol=0
        )


def test_glm53_pool_expansion_appends_only_the_incomplete_tail() -> None:
    device = _require_glm_gpu()
    pool_ids = torch.full((3, 512), -1, dtype=torch.int32, device=device)
    pool_ids[1, :2] = torch.tensor([1, 0], dtype=torch.int32, device=device)
    pool_ids[2] = torch.arange(512, dtype=torch.int32, device=device)
    positions = torch.tensor([2, 7, 2052], dtype=torch.int64, device=device)
    output = torch.empty((3, 2051), dtype=torch.int32, device=device)

    expand_pool_ids(pool_ids, positions, output)

    assert torch.all(output[0, 3:] == -1)
    assert torch.equal(output[0, :3].cpu(), torch.tensor([0, 1, 2], dtype=torch.int32))
    assert torch.equal(output[1, :8].cpu(), torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))
    assert torch.all(output[1, 8:] == -1)
    assert torch.equal(output[2, :2048].cpu(), torch.arange(2048, dtype=torch.int32))
    assert int(output[2, 2048]) == 2052
    assert torch.all(output[2, 2049:] == -1)


def test_glm53_pool_write_uses_int64_for_live_high_page() -> None:
    device = _require_glm_gpu()
    block_size = 256
    parent_page_bytes = block_size * (528 + 33)
    high_page = 2**31 // parent_page_bytes + 1
    raw = torch.empty(
        (high_page + 1) * parent_page_bytes, dtype=torch.uint8, device=device
    )
    main = torch.as_strided(
        raw,
        size=(high_page + 1, block_size, 528),
        stride=(parent_page_bytes, 528, 1),
    )
    cache, _, parent_stride_pages = Glm5NextPooledIndexer._index_cache_view(main)
    key = torch.ones((4, 128), dtype=torch.bfloat16, device=device)
    gate = torch.zeros_like(key)
    ape = torch.zeros_like(key)
    slots = high_page * block_size + torch.arange(4, dtype=torch.int64, device=device)
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    update_decode_pools(
        cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        slots,
        torch.arange(4, dtype=torch.int64, device=device),
        1,
        model_block_size=block_size,
        parent_stride_pages=parent_stride_pages,
    )
    written_key, written_scale = _read_cache_entry(
        cache, high_page * parent_stride_pages, 0
    )

    assert torch.count_nonzero(written_key).item() == 1
    assert torch.isfinite(written_scale).all()
    assert float(written_scale[0]) > 0


def test_glm53_pool_expansion_replays_without_allocation() -> None:
    device = _require_glm_gpu()
    pool_ids = torch.arange(512, dtype=torch.int32, device=device).repeat(2, 1)
    positions = torch.tensor([2048, 2049], dtype=torch.int64, device=device)
    output = torch.empty((2, 2051), dtype=torch.int32, device=device)
    expand_pool_ids(pool_ids, positions, output)
    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        expand_pool_ids(pool_ids, positions, output)
    graph.replay()
    torch.accelerator.synchronize()
    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated
