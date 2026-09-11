# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the active prefix consumed by sparse attention after pool expansion."""

import os

import pytest
import torch

from vllm.models.glm5next.nvidia.ops.glm_kpool import expand_pool_ids
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
    triton_filter_and_convert_dcp_index,
)

LENGTHS = (1, 2, 3, 4, 5, 7, 8, 15, 2047, 2048, 2049, 2051, 262173)


def _device():
    if os.environ.get("B12X_GLM53_GPU_TEST") != "1":
        pytest.skip("set B12X_GLM53_GPU_TEST=1 to run GPU tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")


def _selection(length, changed=False):
    complete = length // 4
    count = min(complete, 512)
    # Dense complete prefixes below top-k; distinct selected pools above top-k.
    first = complete - count if changed else 0
    pools = list(range(first, first + count))
    tokens = [4 * pool + offset for pool in pools for offset in range(4)]
    return pools, tokens + list(range(4 * complete, length))


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("route", ("dcp1_cache", "dcp1_prefill", "dcp4"))
@pytest.mark.parametrize("block_size,interleave", ((64, 4), (64, 64), (2048, 2048)))
@pytest.mark.parametrize("replay", (False, True), ids=("eager", "graph"))
def test_pool_tail_active_prefix(length, route, block_size, interleave, replay):
    device = _device()
    # Include a padding row and change request order and page mapping on replay.
    pools = torch.full((2, 512), -1, dtype=torch.int32, device=device)
    positions = torch.full((2,), -1, dtype=torch.int64, device=device)
    expanded = torch.empty((2, 2051), dtype=torch.int32, device=device)
    requests = torch.tensor([1, 0], dtype=torch.int32, device=device)
    pages = (max(LENGTHS) + block_size - 1) // block_size + 1
    table = torch.empty((2, pages), dtype=torch.int32, device=device)
    stride = block_size + 16
    workspace_ids = torch.tensor([1, -1], dtype=torch.int32, device=device)
    starts = torch.tensor([31, 4099], dtype=torch.int32, device=device)

    def stage(n, changed):
        selected, expected = _selection(n, changed)
        pools.fill_(-1)
        pools[0, : len(selected)] = torch.tensor(
            selected, dtype=torch.int32, device=device
        )
        positions.copy_(torch.tensor([n - 1, -1], device=device))
        requests.copy_(torch.tensor([int(not changed), int(changed)], device=device))
        mapping = torch.arange(2 * pages, dtype=torch.int32, device=device)
        if changed:
            mapping = mapping.flip(0)
        table.copy_(mapping.reshape(2, pages) + 7)
        starts.copy_(torch.tensor([31, 8197 if changed else 4099], device=device))
        return expected

    def run():
        expand_pool_ids(pools, positions, expanded)
        common = dict(
            BLOCK_SIZE=block_size,
            BLOCK_STRIDE_ROWS=stride,
            NUM_TOPK_TOKENS=2051,
            return_valid_counts=True,
        )
        if route == "dcp4":
            return [
                triton_filter_and_convert_dcp_index(
                    requests, table, expanded, 4, rank, interleave, **common
                )
                for rank in range(4)
            ]
        return [
            triton_convert_req_index_to_global_index(
                requests,
                table,
                expanded,
                HAS_PREFILL_WORKSPACE=route == "dcp1_prefill",
                prefill_workspace_request_ids=workspace_ids,
                prefill_workspace_starts=starts,
                **common
            )
        ]

    def check(results, expected):
        logical = expanded[0].cpu().tolist()
        assert sorted(t for t in logical if t >= 0) == sorted(expected)
        assert torch.all(expanded[1] == -1)
        mapping = table[requests[0].item()].cpu().tolist()
        inverse = {page: i for i, page in enumerate(mapping)}
        union = []
        for rank, (output, counts) in enumerate(results):
            count = counts[0].item()
            active = output[0, :count].cpu().tolist()
            assert all(slot >= 0 for slot in active), (length, route, rank, active)
            assert torch.all(output[0, count:] == -1)
            assert counts[1].item() == 0
            assert torch.all(output[1] == -1)
            decoded = []
            for slot in active:
                if route == "dcp1_prefill":
                    token = slot - starts[1].item()
                else:
                    page, offset = divmod(slot, stride)
                    assert offset < block_size
                    local = inverse[page] * block_size + offset
                    token = local
                    if route == "dcp4":
                        token = ((local // interleave) * 4 + rank) * interleave
                        token += local % interleave
                decoded.append(token)
            owned = [
                t for t in expected if route != "dcp4" or (t // interleave) % 4 == rank
            ]
            assert count == len(owned)
            assert sorted(decoded) == sorted(owned)
            assert len(decoded) == len(set(decoded))
            union.extend(decoded)
        assert sorted(union) == sorted(expected)
        assert len(union) == len(set(union))

    expected = stage(length, False)
    results = run()
    if not replay:
        check(results, expected)
        return
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = run()
    # Start with a changed input so a short baseline failure tests live replay.
    for n, changed in ((length, True), (7, False), (5, True), (length, False)):
        expected = stage(n, changed)
        graph.replay()
        graph.replay()
        check(results, expected)


@torch.inference_mode()
def test_five_token_tail_changes_b12x_attention():
    device = _device()
    from b12x.attention import sparse_mla
    from b12x.attention._shared.mla.reference import (
        sparse_mla_reference,
    )

    from b12x.attention._shared.mla.traits import ModelType, ScaleFormat

    pools = torch.full((1, 512), -1, dtype=torch.int32, device=device)
    pools[0, 0] = 0
    expanded = torch.empty((1, 2051), dtype=torch.int32, device=device)
    expand_pool_ids(pools, torch.tensor([4], device=device), expanded)
    selected, counts = triton_convert_req_index_to_global_index(
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        expanded,
        NUM_TOPK_TOKENS=2051,
        return_valid_counts=True,
    )
    # MLA shares the latent key/value. Only the final token has nonzero values.
    keys = torch.zeros((64, 512), dtype=torch.bfloat16, device=device)
    keys[4] = 2
    cache = torch.empty((1, 64, 528), dtype=torch.uint8, device=device)
    sparse_mla.concat_and_cache_glm_next_mla(
        keys, cache, torch.arange(64, dtype=torch.int64, device=device)
    )
    query = torch.zeros((1, 16, 512), dtype=torch.bfloat16, device=device)
    query[..., 0] = 1
    scale = 256**-0.5
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=16,
            max_q_rows=1,
            max_width=2051,
            kv_dtype=torch.uint8,
            max_batch=1,
            softmax_scale=scale,
            max_kv_rows=5,
            head_dim=512,
            v_head_dim=512,
            page_size=64,
            model_type=ModelType.GLM_NEXT,
            scale_format=ScaleFormat.ARBITRARY_FP32,
            cache_record_bytes=528,
            fp8_rope=False,
            latent_scale_per_token=False,
            mode="decode",
        )
    )
    spec = plan.scratch_specs()[0]
    binding = sparse_mla.bind(
        plan,
        scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device),
        q=query,
        selected_indices=selected,
        kv_cache=cache,
        cache_lengths=torch.tensor([5], dtype=torch.int32, device=device),
        selected_lengths=counts,
    )
    actual = sparse_mla.run(binding)
    oracle_indices = torch.full_like(selected, -1)
    oracle_indices[0, :5] = torch.arange(5, dtype=torch.int32, device=device)
    expected = sparse_mla_reference(
        q_all=query,
        kv_cache=cache.view(64, 1, 528),
        page_table_1=oracle_indices,
        active_token_counts=torch.tensor([5], dtype=torch.int32, device=device),
        sm_scale=scale,
        v_head_dim=512,
    )
    assert expected.float().min().item() > 0.3
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.05, atol=0.05)
