# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4.nvidia import b12x_indexer
from vllm.models.deepseek_v4.nvidia.b12x_indexer import _flatten_index_cache
from vllm.models.glm5next.nvidia.ops.glm_kpool import (
    expand_c4_block_table,
    expand_pool_ids,
    gather_c4_block_table_rows,
    pool_seq_lens,
    update_decode_pools,
)
from vllm.models.glm5next.nvidia.pooled_indexer import Glm5NextPooledIndexer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseMetadata
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
    assert metadata.prefill_seq_lens_cpu.tolist() == [8]


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


def _packed_main_cache(
    *, device: torch.device, blocks: int, layers: int, block_size: int, layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    page_bytes = block_size * (528 + 33)
    raw = torch.zeros(blocks * layers * page_bytes, dtype=torch.uint8, device=device)
    main = torch.as_strided(
        raw,
        size=(blocks, block_size, 528),
        stride=(layers * page_bytes, 528, 1),
        storage_offset=layer * page_bytes,
    )
    return raw, main


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
    indexer.max_speculative_tokens = 5

    indexer.bind_main_kv_cache(main)

    assert indexer._decode_block_table.shape[0] == indexer.max_tokens
    assert indexer._decode_block_table.shape[0] > (
        indexer.max_seqs * (indexer.max_speculative_tokens + 1)
    )


def test_glm53_parent_table_width_tracks_dcp_sharding() -> None:
    max_model_len = 524288
    block_size = 2304

    assert Glm5NextPooledIndexer._max_parent_table_width(
        max_model_len,
        0,
        block_size,
        dcp_world_size=1,
    ) == math.ceil(max_model_len / block_size)
    assert Glm5NextPooledIndexer._max_parent_table_width(
        max_model_len,
        0,
        block_size,
        dcp_world_size=4,
    ) == math.ceil(max_model_len / (block_size * 4))
    assert (
        Glm5NextPooledIndexer._max_parent_table_width(
            block_size,
            1,
            block_size,
            dcp_world_size=1,
        )
        == 2
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
    plan = module.plan(
        module.Caps(
            device=device,
            source_layout=module.SOURCE_LAYOUT_PAGED,
            num_q_heads=32,
            max_q_rows=1,
            max_page_table_width=1,
            topk=512,
            mode="decode",
            shared_page_table=False,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table,
        cache_seqlens_int32=seq_lens,
        expected_num_q_heads=32,
        shared_page_table=False,
        output_physical_slots=False,
    )
    output = torch.empty((1, 512), dtype=torch.int32, device=device)
    module.index_topk_fp8(
        q_fp8=q,
        weights=weights,
        index_k_cache=_flatten_index_cache(index_cache),
        binding=binding,
        page_size=64,
        expected_num_q_heads=32,
        out_indices=output,
    )
    torch.accelerator.synchronize()
    assert set(output[0, :2].tolist()) == {0, 1}
    assert torch.all(output[0, 2:] == -1)

    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        module.index_topk_fp8(
            q_fp8=q,
            weights=weights,
            index_k_cache=_flatten_index_cache(index_cache),
            binding=binding,
            page_size=64,
            expected_num_q_heads=32,
            out_indices=output,
        )
    graph.replay()
    torch.accelerator.synchronize()
    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated
    assert set(output[0, :2].tolist()) == {0, 1}


def test_glm53_deepgemm_prefill_matches_b12x_on_packed_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    generator = torch.Generator(device=device).manual_seed(5303)
    quant.copy_(
        torch.randn((64, 128), generator=generator, device=device).to(
            torch.float8_e4m3fn
        )
    )
    scales.copy_(
        torch.exp2(
            torch.randint(-3, 3, (64,), generator=generator, device=device).float()
        )
    )
    q = torch.randn(
        (3, 32, 128), generator=generator, device=device, dtype=torch.float32
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(
        (3, 32), generator=generator, device=device, dtype=torch.float32
    )
    seq_lens = torch.tensor([16, 37, 64], dtype=torch.int32, device=device)
    block_table = torch.tensor([[virtual_page]], dtype=torch.int32, device=device)
    topk = 512

    from vllm.utils.b12x import get_b12x_dsa_indexer

    module = get_b12x_dsa_indexer()
    assert module is not None
    plan = module.plan(
        module.Caps(
            device=device,
            source_layout=module.SOURCE_LAYOUT_PAGED,
            num_q_heads=32,
            max_q_rows=3,
            max_page_table_width=1,
            topk=topk,
            mode="prefill",
            shared_page_table=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    binding = plan.bind(
        scratch=scratch,
        real_page_table=block_table.expand(3, 1),
        cache_seqlens_int32=seq_lens,
        expected_num_q_heads=32,
        shared_page_table=True,
        output_physical_slots=False,
    )
    expected = torch.empty((3, topk), dtype=torch.int32, device=device)
    module.index_topk_fp8(
        q_fp8=q,
        weights=weights,
        index_k_cache=_flatten_index_cache(index_cache),
        binding=binding,
        page_size=64,
        expected_num_q_heads=32,
        out_indices=expected,
    )

    class Workspace:
        def get_simultaneous(self, *specs):
            return tuple(
                torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs
            )

    monkeypatch.setattr(
        b12x_indexer,
        "current_workspace_manager",
        lambda: Workspace(),
    )
    actual = torch.empty_like(expected)
    b12x_indexer._run_deepgemm_prefill_topk(
        q=q,
        weights=weights,
        kv_cache=index_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        gather_cu_seq_lens=torch.tensor([0, 64], dtype=torch.int32, device=device),
        row_starts=torch.zeros(3, dtype=torch.int32, device=device),
        output=actual,
        topk=topk,
        total_k_rows=64,
        workspace_k_rows=64,
    )
    torch.accelerator.synchronize()

    assert torch.equal(
        torch.sort(actual, dim=-1).values,
        torch.sort(expected, dim=-1).values,
    )


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


def test_glm53_decode_writer_matches_batched_prefill_writer() -> None:
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
    prefill_tail = torch.full(
        (1, 2, 4, 128), float("nan"), dtype=torch.bfloat16, device=device
    )
    decode_tail = torch.full_like(prefill_tail, float("nan"))
    state_slots = torch.zeros(1, dtype=torch.int32, device=device)

    update_decode_pools(
        prefill_cache,
        prefill_tail,
        state_slots,
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        torch.tensor([-1, -1, -1, 0, -1, -1, -1, 1], device=device),
        torch.arange(8, dtype=torch.int64, device=device),
        1,
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
                [position // 4 if position % 4 == 3 else -1],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor([position], dtype=torch.int64, device=device),
            1,
        )

    assert torch.equal(decode_cache, prefill_cache)
    assert torch.equal(decode_tail, prefill_tail)


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

    assert torch.all(output[0, :2048] == -1)
    assert torch.equal(
        output[0, 2048:].cpu(), torch.tensor([0, 1, 2], dtype=torch.int32)
    )
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
