# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU host-policy checks; kernels are recorded, never executed.

The standalone lab base dispatches multi-query requests to 2D. Selection tests
exercise eligibility separately, without pretending the wrapper has PR839.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.v1.attention.ops import triton_unified_attention_diffkv as ops

pytestmark = pytest.mark.cpu_test


class Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))

        return launch


@pytest.fixture
def platform(monkeypatch):
    platform = SimpleNamespace(
        is_cuda=Mock(return_value=True), is_device_capability=Mock(return_value=True)
    )
    monkeypatch.setattr(ops, "current_platform", platform)
    monkeypatch.setattr(ops, "is_batch_invariant", False)
    return platform


def tensors(
    tokens=128,
    heads=16,
    kv_heads=1,
    qk=192,
    hv=128,
    block=16,
    columns=8,
    dtype=torch.bfloat16,
    cache_dtype=torch.bfloat16,
):
    return (
        torch.empty(tokens, heads, qk, dtype=dtype),
        torch.empty(1, block, kv_heads, qk, dtype=cache_dtype),
        torch.empty(1, block, kv_heads, hv, dtype=cache_dtype),
        torch.empty(8, columns, dtype=torch.int32),
    )


def select(args, **changes):
    controls = dict(use_3d=True, max_seqlen_q=8, sliding_window=0, num_segments=16)
    controls.update(changes)
    return ops._select_diffkv_tiling(*args, **controls)


@pytest.mark.parametrize("tokens", [128, 256])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_selected_multiquery_geometry(platform, tokens, cache_dtype):
    assert select(tensors(tokens=tokens, cache_dtype=cache_dtype)) == (32, 16)
    platform.is_device_capability.assert_called_once_with(120, 0)


@pytest.mark.parametrize(
    "change",
    [
        dict(tokens=127),
        dict(heads=8),
        dict(kv_heads=2),
        dict(qk=128),
        dict(hv=64),
        dict(dtype=torch.float16),
        dict(columns=5),
    ],
)
def test_other_geometries_keep_original_tiles(platform, change):
    assert select(tensors(**change)) == (16, 16)


@pytest.mark.parametrize(
    "change,expected",
    [
        (dict(use_3d=False), (16, 32)),
        (dict(max_seqlen_q=1), (16, 16)),
        (dict(sliding_window=128), (16, 16)),
        (dict(num_segments=8), (16, 16)),
    ],
)
def test_dispatch_window_and_split_count_are_not_expanded(platform, change, expected):
    assert select(tensors(), **change) == expected


@pytest.mark.parametrize("tokens,max_query", [(512, 512), (2048, 2048)])
def test_2d_prefill_geometry_remains_unchanged(platform, tokens, max_query):
    assert select(
        tensors(tokens=tokens), use_3d=False, max_seqlen_q=max_query, num_segments=None
    ) == (16, 32)


@pytest.mark.parametrize(
    "tokens,max_query", [(511, 512), (512, 511), (256, 256), (2048, 8)]
)
def test_unmeasured_2d_queries_keep_default(platform, tokens, max_query):
    actual = select(tensors(tokens=tokens), use_3d=False, max_seqlen_q=max_query)
    assert actual == (16, 32)


@pytest.mark.parametrize("use_3d,tokens,max_query", [(True, 128, 8), (False, 512, 512)])
def test_batch_invariant_and_swa_never_retiled(
    platform, monkeypatch, use_3d, tokens, max_query
):
    expected = (16, 16 if use_3d else 32)
    assert (
        select(
            tensors(tokens=tokens),
            use_3d=use_3d,
            max_seqlen_q=max_query,
            sliding_window=128,
        )
        == expected
    )
    monkeypatch.setattr(ops, "is_batch_invariant", True)
    actual = select(tensors(tokens=tokens), use_3d=use_3d, max_seqlen_q=max_query)
    assert actual == expected


def test_hardware_gate_and_query_device(platform):
    args = list(tensors())
    q = args[0]
    args[0] = SimpleNamespace(
        shape=q.shape,
        dtype=q.dtype,
        element_size=q.element_size,
        device=SimpleNamespace(index=3),
    )
    assert select(args) == (32, 16)
    platform.is_device_capability.assert_called_with(120, 3)
    platform.is_device_capability.return_value = False
    assert select(args) == (16, 16)
    platform.is_cuda.return_value = False
    platform.is_device_capability.reset_mock()
    assert select(args) == (16, 16)
    platform.is_device_capability.assert_not_called()


def call_wrapper(monkeypatch, qlen, *, batch=8, force_tile_for_grid_test=False):
    q, k, v, table = tensors(tokens=batch * qlen, columns=max(8, (qlen + 15) // 16))
    table = torch.empty(batch, table.shape[1], dtype=torch.int32)
    main, reduce = Recorder(), Recorder()
    monkeypatch.setattr(ops, "kernel_unified_attention_diffkv", main)
    monkeypatch.setattr(ops, "kernel_reduce_segments_diffkv", reduce)
    if force_tile_for_grid_test:
        # Test grid plumbing in the real wrapper, not multi-query eligibility.
        monkeypatch.setattr(ops, "_select_diffkv_tiling", lambda *a, **kw: (32, 16))
    ops.unified_attention_diffkv(
        q=q,
        k=k,
        v=v,
        out=torch.empty(batch * qlen, 16, 128),
        cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32) * qlen,
        seqused_k=torch.full((batch,), max(128, qlen), dtype=torch.int32),
        softmax_scale=192**-0.5,
        causal=True,
        window_size=(-1, -1),
        block_table=table,
        softcap=0,
        max_seqlen_q=qlen,
        seq_threshold_3D=128,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.empty(128, 16, 16, 128),
        softmax_segm_max=torch.empty(128, 16, 16),
        softmax_segm_expsum=torch.empty(128, 16, 16),
    )
    return main.calls, reduce.calls


@pytest.mark.parametrize("batch", [1, 8, 16])
def test_multiquery_geometry_matches_selected_dispatch(
    platform, monkeypatch, record_property, batch
):
    main, reduce = call_wrapper(monkeypatch, 8, batch=batch)
    assert len(main) == 1
    grid, kw = main[0]
    record_property("diffkv_dispatch", "3d" if kw["IS_3D"] else "2d")
    if kw["IS_3D"]:
        block_m, block_q = (32, 2) if batch == 16 else (16, 1)
        assert (kw["BLOCK_M"], kw["BLOCK_Q"], kw["TILE_SIZE"]) == (block_m, block_q, 16)
        assert grid == (batch * 8 // block_q + batch, 1, 16)
        assert kw["NUM_SEGMENTS_PER_SEQ"] == 16
        assert len(reduce) == 1
        assert reduce[0][0] == (batch * 8, 16)
        assert reduce[0][1]["BLOCK_Q"] == block_q
        assert reduce[0][1]["TILE_SIZE"] == 16
        assert reduce[0][1]["NUM_SEGMENTS_PER_SEQ"] == 16
    else:
        assert (kw["BLOCK_M"], kw["BLOCK_Q"], kw["TILE_SIZE"]) == (16, 1, 32)
        assert grid == (batch * 9, 1)
        assert not reduce


def test_single_query_3d_remains_unchanged(platform, monkeypatch):
    main, reduce = call_wrapper(monkeypatch, 1)
    grid, kw = main[0]
    assert kw["IS_3D"]
    assert (kw["BLOCK_M"], kw["BLOCK_Q"], kw["TILE_SIZE"]) == (16, 1, 16)
    assert grid == (16, 1, 16)
    assert reduce[0][0] == (8, 16)
    assert reduce[0][1]["NUM_SEGMENTS_PER_SEQ"] == 16


def test_selected_tile_precedes_main_and_reduction_grid_derivation(
    platform, monkeypatch
):
    main, reduce = call_wrapper(monkeypatch, 1, force_tile_for_grid_test=True)
    grid, kw = main[0]
    assert grid == (12, 1, 16)
    assert (kw["BLOCK_M"], kw["BLOCK_Q"], kw["TILE_SIZE"]) == (32, 2, 16)
    assert reduce[0][0] == (8, 16)
    assert reduce[0][1]["BLOCK_Q"] == 2
    assert reduce[0][1]["TILE_SIZE"] == 16
    assert reduce[0][1]["NUM_SEGMENTS_PER_SEQ"] == 16


def test_real_wrapper_2d_prefill_keeps_original_grid(platform, monkeypatch):
    # 8 * 512 query tokens cannot fit this wrapper fixture's 128-token workspace.
    main, reduce = call_wrapper(monkeypatch, 512)
    grid, kw = main[0]
    assert not kw["IS_3D"]
    assert (kw["BLOCK_M"], kw["BLOCK_Q"], kw["TILE_SIZE"]) == (16, 1, 32)
    assert grid == (4104, 1)
    assert not reduce


@pytest.mark.parametrize("block,columns", [(16, 4), (64, 1)])
def test_exact_aligned_table_coverage_is_eligible(platform, block, columns):
    assert select(tensors(block=block, columns=columns)) == (32, 16)


@pytest.mark.parametrize("use_3d", [False, True])
@pytest.mark.parametrize(
    "key_dtype,value_dtype",
    [
        (torch.float32, torch.float32),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz),
    ],
)
def test_unmeasured_or_mixed_cache_types_keep_original_tiles(
    platform, use_3d, key_dtype, value_dtype
):
    args = list(tensors(tokens=512, cache_dtype=key_dtype))
    args[2] = torch.empty(args[2].shape, dtype=value_dtype)
    assert select(args, use_3d=use_3d, max_seqlen_q=512) == (16, 16 if use_3d else 32)
    platform.is_device_capability.assert_not_called()


@pytest.mark.parametrize("tokens", [8, 64, 127])
def test_c1_c8_and_subthreshold_query_batches_keep_original_tiles(platform, tokens):
    assert select(tensors(tokens=tokens)) == (16, 16)
    platform.is_device_capability.assert_not_called()
