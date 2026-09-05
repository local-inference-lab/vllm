# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-weight identity for uneven DFlash QKV, O, and aux projections."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.models.dflash_tp9 import (
    DFlashTP9QKVParallelLinear,
    DFlashTP9RowParallelLinear,
    aligned_tp9_extent,
    dflash_tp9_head_extent,
)


@pytest.fixture(autouse=True)
def local_projection_parameter_rank(monkeypatch):
    # The projections disable the inherited TP slicer and load explicit extents.
    # Parameter construction still queries the ambient rank before reconciliation.
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 8
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 9
    )


def test_tp9_qkv_loader_keeps_source_queries_and_replicates_actual_kv():
    hidden, head_dim = 8, 4
    q = torch.arange(32 * head_dim * hidden).reshape(-1, hidden).float()
    k = torch.arange(8 * head_dim * hidden).reshape(-1, hidden).float() + 10000
    v = k + 10000
    with set_current_vllm_config(VllmConfig()):
        for layer in range(6):
            gathered_queries = {}
            for rank in range(9):
                extent = dflash_tp9_head_extent(layer, rank)
                projection = DFlashTP9QKVParallelLinear(
                    hidden, head_dim, extent, quant_config=None, prefix="draft.qkv"
                )
                for name, tensor in (("q", q), ("k", k), ("v", v)):
                    projection.weight.weight_loader(projection.weight, tensor, name)
                qrows = extent.query_count * head_dim
                got_q, got_k, got_v = projection.weight.split(
                    (qrows, head_dim, head_dim), dim=0
                )
                source_q = q[
                    extent.first_query * head_dim : (
                        extent.first_query + extent.query_count
                    )
                    * head_dim
                ]
                torch.testing.assert_close(got_q, source_q, rtol=0, atol=0)
                torch.testing.assert_close(
                    got_k,
                    k[extent.kv_head * head_dim : (extent.kv_head + 1) * head_dim],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    got_v,
                    v[extent.kv_head * head_dim : (extent.kv_head + 1) * head_dim],
                    rtol=0,
                    atol=0,
                )
                for head in range(
                    extent.first_query, extent.first_query + extent.query_count
                ):
                    assert head not in gathered_queries
                    gathered_queries[head] = extent.kv_head
            assert gathered_queries == {head: head // 4 for head in range(32)}


def test_tp9_aux_row_projection_preserves_full_matrix_product():
    source_width, hidden = 192, 8
    weight = (torch.arange(source_width * hidden) % 7 - 3).reshape(hidden, -1).float()
    x = (torch.arange(source_width * 2) % 5 - 2).reshape(2, -1).float()
    partials = []
    with set_current_vllm_config(VllmConfig()):
        for rank in range(9):
            first, count = aligned_tp9_extent(source_width, rank, 4)
            projection = DFlashTP9RowParallelLinear(
                source_width,
                hidden,
                first,
                count,
                input_is_full=True,
                params_dtype=torch.float32,
                prefix="draft.fc",
                return_bias=False,
            )
            projection.weight.weight_loader(projection.weight, weight)
            partials.append(
                torch.nn.functional.linear(
                    x[:, first : first + count], projection.weight
                )
            )
    torch.testing.assert_close(
        sum(partials), torch.nn.functional.linear(x, weight), atol=0, rtol=0
    )
