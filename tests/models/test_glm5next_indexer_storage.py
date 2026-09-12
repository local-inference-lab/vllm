# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ownership checks for serial GLM target/MTP indexer storage."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.glm5next.nvidia.mtp import Glm5NextMTP
from vllm.models.glm5next.nvidia.pooled_indexer import (
    Glm5NextIndexerScratch,
    Glm5NextPooledIndexer,
)


def make_indexer(device="cpu", shared=None):
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    for name, value in dict(
        max_tokens=16,
        max_seqs=4,
        max_model_len=1048576,
        block_size=2048,
        dcp_world_size=2,
        dcp_rank=0,
        pool_interleave=1,
    ).items():
        setattr(indexer, name, value)
    indexer.scratch = (
        shared.scratch
        if shared
        else Glm5NextIndexerScratch(16, 4, torch.device(device))
    )
    for name, width in (
        ("topk_indices_buffer", 2051),
        ("pool_topk_indices_buffer", 512),
    ):
        value = (
            getattr(shared, name)
            if shared
            else torch.empty((16, width), dtype=torch.int32, device=device)
        )
        setattr(indexer, name, value)
    indexer.indexer_op = nn.Module()
    indexer.indexer_op.topk_indices_buffer = indexer.pool_topk_indices_buffer
    indexer._index_cache = torch.ones(32, dtype=torch.uint8, device=device)
    indexer.register_buffer("_tail", torch.ones(32, device=device), persistent=False)
    indexer.register_parameter("weight", nn.Parameter(torch.ones(4, device=device)))
    return indexer


def make_pair(device="cpu"):
    target = nn.Module()
    first = make_indexer(device)
    target.layers = nn.ModuleList([first, make_indexer(device, shared=first)])
    draft = Glm5NextMTP.__new__(Glm5NextMTP)
    nn.Module.__init__(draft)
    draft.model = nn.Module()
    draft.model.num_mtp_layers = 1
    draft.model.indexer = make_indexer(device)
    draft.model.attention = nn.Module()
    draft.model.attention.topk_indices_buffer = draft.model.indexer.topk_indices_buffer
    draft.model.attention.impl = SimpleNamespace(
        topk_indices_buffer=draft.model.indexer.topk_indices_buffer
    )
    return target, draft


def test_share_indexer_storage_preserves_roles_and_persistent_state():
    target, draft = make_pair()
    source, indexer = target.layers[0], draft.model.indexer
    scratch_ref = weakref.ref(indexer.scratch)
    token_ref = weakref.ref(indexer.topk_indices_buffer)
    pool_ref = weakref.ref(indexer.pool_topk_indices_buffer)
    cache, tail, weight = indexer._index_cache, indexer._tail, indexer.weight
    assert draft.share_target_indexer_storage(target)
    assert scratch_ref() is token_ref() is pool_ref() is None
    assert indexer.scratch is source.scratch
    assert indexer.topk_indices_buffer is source.topk_indices_buffer
    assert indexer.pool_topk_indices_buffer is source.pool_topk_indices_buffer
    assert indexer.indexer_op.topk_indices_buffer is source.pool_topk_indices_buffer
    assert draft.model.attention.topk_indices_buffer is source.topk_indices_buffer
    assert draft.model.attention.impl.topk_indices_buffer is source.topk_indices_buffer
    assert indexer._index_cache is cache
    assert indexer._tail is tail
    assert indexer.weight is weight
    source.scratch.bind_tables(2048, torch.device("cpu"))
    pointer = source.scratch.decode_block_table.data_ptr()
    indexer.scratch.bind_tables(2048, torch.device("cpu"))
    assert indexer.scratch.decode_block_table.data_ptr() == pointer
    assert draft.share_target_indexer_storage(target)  # Idempotent ownership.


@pytest.mark.parametrize(
    "difference",
    [
        "layers",
        "geometry",
        "capacity",
        "dtype",
        "scratch",
        "target_storage",
    ],
)
def test_incompatible_indexer_storage_remains_independent(difference):
    target, draft = make_pair()
    indexer = draft.model.indexer
    if difference == "layers":
        draft.model.num_mtp_layers = 2
    elif difference == "geometry":
        indexer.dcp_world_size = 1
    elif difference == "capacity":
        indexer.topk_indices_buffer = indexer.topk_indices_buffer[:8]
    elif difference == "dtype":
        indexer.pool_topk_indices_buffer = indexer.pool_topk_indices_buffer.long()
    elif difference == "scratch":
        indexer.scratch.bind_tables(2, torch.device("cpu"))
    else:
        target.layers[1].scratch = Glm5NextIndexerScratch(16, 4, torch.device("cpu"))
    scratch, tokens, pools = (
        indexer.scratch,
        indexer.topk_indices_buffer,
        indexer.pool_topk_indices_buffer,
    )
    assert not draft.share_target_indexer_storage(target)
    assert indexer.scratch is scratch
    assert indexer.topk_indices_buffer is tokens
    assert indexer.pool_topk_indices_buffer is pools


@pytest.mark.parametrize(
    "pp,dbo,expected", [(1, False, True), (2, False, False), (1, True, False)]
)
def test_loader_shares_only_serial_indexers(monkeypatch, pp, dbo, expected):
    from vllm.v1.worker.gpu.spec_decode.eagle import utils

    inner, draft = make_pair()
    target = nn.Module()
    target.model = inner
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=None),
        parallel_config=SimpleNamespace(enable_dbo=dbo),
    )
    monkeypatch.setattr(utils, "_make_eagle_draft_vllm_config", lambda value: value)
    monkeypatch.setattr(utils, "get_model", lambda **_: draft)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=pp))
    assert utils.load_eagle_model(target, config) is draft
    assert (draft.model.indexer.scratch is inner.layers[0].scratch) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph ownership")
def test_shared_indexer_storage_serial_graph_replay():
    target, draft = make_pair("cuda")
    assert draft.share_target_indexer_storage(target)
    source, indexer = target.layers[0], draft.model.indexer
    value = torch.ones((), device="cuda")

    def run(owner, factor):
        owner.scratch.pool_scores.copy_(value * factor)
        owner.topk_indices_buffer.fill_(factor)
        owner.pool_topk_indices_buffer.fill_(factor + 1)
        return (
            owner.scratch.pool_scores.clone(),
            owner.topk_indices_buffer.clone(),
            owner.pool_topk_indices_buffer.clone(),
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(source, 2)
        run(indexer, 3)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        target_result = run(source, 2)
        draft_result = run(indexer, 3)
    for scalar in (1, 5, 9):
        value.fill_(scalar)
        graph.replay()
        for result, factor in ((target_result, 2), (draft_result, 3)):
            assert torch.equal(result[0], torch.full_like(result[0], scalar * factor))
            assert torch.equal(result[1], torch.full_like(result[1], factor))
            assert torch.equal(result[2], torch.full_like(result[2], factor + 1))
