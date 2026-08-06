# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch

from vllm.model_executor.layers import sparse_attn_indexer as indexer_mod


def _profile_forward_context():
    return types.SimpleNamespace(
        attn_metadata=None,
        cudagraph_runtime_mode=indexer_mod.CUDAGraphMode.NONE,
        batch_descriptor=object(),
    )


class _FakeWorkspaceManager:
    def __init__(self, *, device: str | None = None) -> None:
        self.device = device
        self.specs: tuple[tuple[tuple[int, ...], torch.dtype], ...] | None = None

    def get_simultaneous(
        self, *shapes_and_dtypes: tuple[tuple[int, ...], torch.dtype]
    ) -> list[torch.Tensor]:
        self.specs = shapes_and_dtypes
        tensors = []
        for shape, dtype in shapes_and_dtypes:
            kwargs = {"dtype": dtype}
            if self.device is not None:
                kwargs["device"] = self.device
            tensors.append(torch.empty(shape, **kwargs))
        return tensors


def _install_fake_b12x_indexer(
    monkeypatch,
    calls: list[tuple],
    *,
    prefill_route: str = "packed_contiguous",
):
    sparkinfer_mod = types.ModuleType("sparkinfer")
    sparkinfer_mod.__path__ = []
    attention_mod = types.ModuleType("sparkinfer.attention")
    attention_mod.__path__ = []
    indexer_mod = types.ModuleType("sparkinfer.attention.nsa_indexer")
    indexer_mod.__path__ = []

    class _Caps:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    class _Plan:
        def __init__(self, specs, bind_impl, *, route: str) -> None:
            self._specs = specs
            self._bind_impl = bind_impl
            self.layout = types.SimpleNamespace(route=route)

        def shapes_and_dtypes(self):
            return self._specs

        def bind(self, scratch, **kwargs):
            return self._bind_impl(scratch, kwargs)

    def plan_indexer_scratch(caps):
        route = (
            prefill_route
            if caps.shared_page_table or caps.mode == "prefill"
            else "paged_fused"
        )
        calls.append(
            (
                "indexer_plan",
                caps.source_layout,
                caps.mode,
                caps.shared_page_table,
                caps.max_q_rows,
                caps.max_page_table_width,
                caps.topk,
                route,
            )
        )
        specs = (
            ((caps.max_q_rows, caps.topk), torch.int32),
            ((caps.max_q_rows,), torch.int32),
        )

        def bind_impl(scratch, kwargs):
            calls.append(
                (
                    "paged_bind",
                    tuple(kwargs["real_page_table"].shape),
                    tuple(kwargs["cache_seqlens_int32"].shape),
                    kwargs["expected_num_q_heads"],
                    kwargs["shared_page_table"],
                    kwargs["active_width"] is not None,
                    kwargs["output_physical_slots"],
                )
            )
            return types.SimpleNamespace(scratch=scratch, route=route, **kwargs)

        return _Plan(specs, bind_impl, route=route)

    def index_topk_fp8(**kwargs):
        calls.append(
            (
                "paged_index_topk",
                tuple(kwargs["q_fp8"].shape),
                tuple(kwargs["index_k_cache"].shape),
                tuple(kwargs["index_k_cache"].stride()),
                kwargs["page_size"],
                kwargs["expected_num_q_heads"],
                kwargs["out_scores"] is not None,
            )
        )
        kwargs["out_indices"].fill_(123)
        if kwargs["out_scores"] is not None:
            kwargs["out_scores"].fill_(456)
        return kwargs["out_indices"]

    def uses_paged_mqa_schedule(*, q_rows: int, max_pages: int) -> bool:
        calls.append(("uses_schedule", q_rows, max_pages))
        return True

    def build_paged_mqa_schedule_metadata(seq_lens, block_size, num_sms, *, out):
        calls.append(
            (
                "build_schedule",
                tuple(seq_lens.tolist()),
                block_size,
                num_sms,
                out is not None,
            )
        )
        out.fill_(7)
        return out

    indexer_mod.PAGED_INDEX_PAGE_SIZE = 64
    indexer_mod.Caps = _Caps
    indexer_mod.SOURCE_LAYOUT_PAGED = "paged"
    indexer_mod.plan = plan_indexer_scratch
    indexer_mod.index_topk_fp8 = index_topk_fp8
    indexer_mod.uses_paged_schedule = uses_paged_mqa_schedule
    indexer_mod.plan_paged_schedule = build_paged_mqa_schedule_metadata

    sparkinfer_mod.attention = attention_mod
    attention_mod.nsa_indexer = indexer_mod
    monkeypatch.setitem(sys.modules, "sparkinfer", sparkinfer_mod)
    monkeypatch.setitem(sys.modules, "sparkinfer.attention", attention_mod)
    monkeypatch.setitem(
        sys.modules,
        "sparkinfer.attention.nsa_indexer",
        indexer_mod,
    )


def _install_fake_b12x_dcp_merge(monkeypatch, run_row_topk, *, world_size: int):
    tiled_topk_mod = types.ModuleType("sparkinfer.attention.nsa_indexer.tiled_topk")
    tiled_topk_mod.run_row_topk = run_row_topk
    monkeypatch.setitem(
        sys.modules,
        "sparkinfer.attention.nsa_indexer.tiled_topk",
        tiled_topk_mod,
    )

    class FakeDCPGroup:
        def __init__(self) -> None:
            self.world_size = world_size

        def all_gather(self, tensor, dim):
            assert dim == 0
            return torch.cat([tensor.clone() for _ in range(world_size)], dim=dim)

    import vllm.distributed.parallel_state as parallel_state
    import vllm.v1.attention.backends.mla.sparse_utils as sparse_utils

    monkeypatch.setattr(parallel_state, "get_dcp_group", lambda: FakeDCPGroup())
    monkeypatch.setattr(
        sparse_utils,
        "triton_convert_dcp_local_topk_to_global",
        lambda *args, **kwargs: None,
    )

    def gather_topk_ids_by_position(candidate_ids, positions, out):
        gathered = torch.gather(candidate_ids, 1, positions.to(torch.int64))
        out.copy_(gathered.to(out.dtype))

    monkeypatch.setattr(
        sparse_utils,
        "triton_gather_topk_ids_by_position",
        gather_topk_ids_by_position,
    )


@pytest.mark.parametrize(
    "page_stride0",
    [
        64 * 132,
        64 * 576,
    ],
)
def test_b12x_decode_indexer_is_non_shared_for_fused_route(
    monkeypatch,
    page_stride0,
):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )

    q_rows = 2
    num_heads = 1
    topk = 4
    page_table_width = 10
    q_fp8 = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads, 1), dtype=torch.float32)
    kv_cache = torch.empty_strided(
        (page_table_width, 64, 132),
        (page_stride0, 132, 1),
        dtype=torch.uint8,
    )
    seq_lens = torch.tensor([600, 640], dtype=torch.int32)
    block_table = torch.arange(q_rows * page_table_width, dtype=torch.int32).reshape(
        q_rows, page_table_width
    )
    topk_indices = torch.empty((q_rows, topk), dtype=torch.int32)
    active_width = torch.tensor([8, 10], dtype=torch.int32)

    result = indexer_mod._run_b12x_paged_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_cache=kv_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        schedule_metadata=None,
        active_width=active_width,
        topk_indices=topk_indices,
        topk_tokens=topk,
    )

    assert result is topk_indices
    assert topk_indices.tolist() == [[123] * topk, [123] * topk]
    assert workspace_manager.specs == (
        ((q_rows, topk), torch.int32),
        ((q_rows,), torch.int32),
    )
    assert calls == [
        (
            "indexer_plan",
            "paged",
            "decode",
            False,
            q_rows,
            page_table_width,
            topk,
            "paged_fused",
        ),
        (
            "paged_bind",
            tuple(block_table.shape),
            tuple(seq_lens.shape),
            num_heads,
            False,
            True,
            False,
        ),
        (
            "paged_index_topk",
            tuple(q_fp8.shape),
            (page_table_width, 64 * 132),
            (page_stride0, 1),
            64,
            num_heads,
            False,
        ),
    ]


def test_b12x_prefill_indexer_marks_shared_page_table(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )

    q_rows = 2
    num_heads = 1
    topk = 4
    page_table_width = 10
    q_fp8 = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads, 1), dtype=torch.float32)
    kv_cache = torch.empty((page_table_width, 64, 132), dtype=torch.uint8)
    seq_lens = torch.tensor([600, 640], dtype=torch.int32)
    base_block_table = torch.arange(page_table_width, dtype=torch.int32).reshape(
        1, page_table_width
    )
    block_table = base_block_table.expand(q_rows, page_table_width)
    topk_indices = torch.empty((q_rows, topk), dtype=torch.int32)

    result = indexer_mod._run_b12x_paged_topk(
        q_fp8=q_fp8,
        weights=weights,
        kv_cache=kv_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        schedule_metadata=None,
        topk_indices=topk_indices,
        topk_tokens=topk,
        shared_page_table=True,
        output_physical_slots=True,
    )

    assert result is topk_indices
    assert workspace_manager.specs == (
        ((q_rows, topk), torch.int32),
        ((q_rows,), torch.int32),
    )
    assert calls == [
        (
            "indexer_plan",
            "paged",
            "prefill",
            True,
            q_rows,
            page_table_width,
            topk,
            "packed_contiguous",
        ),
        (
            "paged_bind",
            tuple(block_table.shape),
            tuple(seq_lens.shape),
            num_heads,
            True,
            False,
            True,
        ),
        (
            "paged_index_topk",
            tuple(q_fp8.shape),
            (page_table_width, 64 * 132),
            (64 * 132, 1),
            64,
            num_heads,
            False,
        ),
    ]


def test_b12x_prefill_indexer_requires_packed_contiguous_route(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls, prefill_route="paged_tiled")
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )

    q_rows = 2
    num_heads = 1
    topk = 4
    page_table_width = 10
    q_fp8 = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads, 1), dtype=torch.float32)
    kv_cache = torch.empty((page_table_width, 64, 132), dtype=torch.uint8)
    seq_lens = torch.tensor([600, 640], dtype=torch.int32)
    block_table = (
        torch.arange(page_table_width, dtype=torch.int32)
        .reshape(1, page_table_width)
        .expand(q_rows, page_table_width)
    )
    topk_indices = torch.empty((q_rows, topk), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="packed_contiguous.*scratch plan"):
        indexer_mod._run_b12x_paged_topk(
            q_fp8=q_fp8,
            weights=weights,
            kv_cache=kv_cache,
            seq_lens=seq_lens,
            block_table=block_table,
            schedule_metadata=None,
            topk_indices=topk_indices,
            topk_tokens=topk,
            shared_page_table=True,
        )

    assert workspace_manager.specs is None
    assert calls == [
        (
            "indexer_plan",
            "paged",
            "prefill",
            True,
            q_rows,
            page_table_width,
            topk,
            "paged_tiled",
        )
    ]


@pytest.mark.parametrize(
    "page_stride0",
    [
        64 * 132,
        64 * 576,
    ],
)
def test_sparse_attn_indexer_decode_uses_non_shared_b12x_binding(
    monkeypatch,
    page_stride0,
):
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(
        indexer_mod,
        "_ensure_b12x_sparse_indexer_supported",
        lambda: None,
    )
    monkeypatch.setattr(
        indexer_mod.current_platform,
        "fp8_dtype",
        lambda: torch.uint8,
        raising=False,
    )

    q_rows = 2
    num_heads = 1
    topk = 4
    page_table_width = 10
    seq_lens = torch.tensor([600, 640], dtype=torch.int32)
    block_table = torch.arange(q_rows * page_table_width, dtype=torch.int32).reshape(
        q_rows, page_table_width
    )
    active_width = torch.tensor([10], dtype=torch.int32)
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens,
        max_seq_len=640,
        slot_mapping=torch.arange(q_rows, dtype=torch.int32),
        num_decodes=q_rows,
        num_decode_tokens=q_rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=640,
            decode_lens=seq_lens,
            requires_padding=False,
            schedule_metadata=None,
            active_width=active_width,
        ),
    )
    layer_name = "layers.0.attn"
    metadata_key = indexer_mod._resolve_layer_name(layer_name)
    monkeypatch.setattr(
        indexer_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={metadata_key: metadata}),
    )

    hidden_states = torch.empty((q_rows, 128), dtype=torch.bfloat16)
    kv_cache = torch.empty_strided(
        (page_table_width, 64, 132),
        (page_stride0, 132, 1),
        dtype=torch.uint8,
    )
    q_quant = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    k = torch.empty((q_rows, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads, 1), dtype=torch.float32)
    topk_indices_buffer = torch.empty((q_rows, topk), dtype=torch.int32)

    result = indexer_mod.sparse_attn_indexer(
        hidden_states,
        layer_name,
        kv_cache,
        q_quant,
        None,
        k,
        weights,
        128,
        None,
        topk_tokens=topk,
        head_dim=128,
        max_model_len=4096,
        total_seq_lens=1024,
        topk_indices_buffer=topk_indices_buffer,
        skip_k_cache_insert=True,
        use_fp4_cache=False,
        use_b12x_sparse_indexer=True,
    )

    assert result is topk_indices_buffer
    assert topk_indices_buffer.tolist() == [[123] * topk, [123] * topk]
    assert calls == [
        (
            "indexer_plan",
            "paged",
            "decode",
            False,
            q_rows,
            page_table_width,
            topk,
            "paged_fused",
        ),
        (
            "paged_bind",
            tuple(block_table.shape),
            tuple(seq_lens.shape),
            num_heads,
            False,
            True,
            False,
        ),
        (
            "paged_index_topk",
            tuple(q_quant.shape),
            (page_table_width, 64 * 132),
            (page_stride0, 1),
            64,
            num_heads,
            False,
        ),
    ]


def test_b12x_dcp_decode_requests_score_output(monkeypatch):
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )

    calls: list[tuple] = []
    merge_calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(
        indexer_mod,
        "_ensure_b12x_sparse_indexer_supported",
        lambda: None,
    )

    def fake_merge(**kwargs):
        merge_calls.append(
            (
                kwargs["topk_indices"].data_ptr(),
                kwargs["topk_scores"].data_ptr(),
                kwargs["topk_tokens"],
                kwargs["dcp_world_size"],
                kwargs["dcp_rank"],
                kwargs["cp_kv_cache_interleave_size"],
            )
        )

    monkeypatch.setattr(indexer_mod, "_merge_b12x_dcp_topk", fake_merge)
    monkeypatch.setattr(
        indexer_mod.current_platform,
        "fp8_dtype",
        lambda: torch.uint8,
        raising=False,
    )

    q_rows = 2
    num_heads = 1
    topk = 4
    page_table_width = 10
    seq_lens = torch.tensor([600, 640], dtype=torch.int32)
    block_table = torch.arange(q_rows * page_table_width, dtype=torch.int32).reshape(
        q_rows, page_table_width
    )
    metadata = DeepseekV32IndexerMetadata(
        seq_lens=seq_lens,
        max_seq_len=640,
        slot_mapping=torch.arange(q_rows, dtype=torch.int32),
        num_decodes=q_rows,
        num_decode_tokens=q_rows,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=640,
            decode_lens=seq_lens,
            requires_padding=False,
            schedule_metadata=None,
            active_width=None,
        ),
    )
    layer_name = "layers.0.attn"
    metadata_key = indexer_mod._resolve_layer_name(layer_name)
    monkeypatch.setattr(
        indexer_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata={metadata_key: metadata}),
    )

    hidden_states = torch.empty((q_rows, 128), dtype=torch.bfloat16)
    kv_cache = torch.empty((page_table_width, 64, 132), dtype=torch.uint8)
    q_quant = torch.empty((q_rows, num_heads, 128), dtype=torch.uint8)
    k = torch.empty((q_rows, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, num_heads), dtype=torch.float32)
    topk_indices_buffer = torch.empty((q_rows, topk), dtype=torch.int32)
    topk_scores_buffer = torch.empty((q_rows, topk), dtype=torch.float32)

    result = indexer_mod.sparse_attn_indexer(
        hidden_states,
        layer_name,
        kv_cache,
        q_quant,
        None,
        k,
        weights,
        128,
        None,
        topk_tokens=topk,
        head_dim=128,
        max_model_len=4096,
        total_seq_lens=1024,
        topk_indices_buffer=topk_indices_buffer,
        skip_k_cache_insert=True,
        use_fp4_cache=False,
        use_b12x_sparse_indexer=True,
        topk_scores_buffer=topk_scores_buffer,
        dcp_world_size=2,
        dcp_rank=1,
        cp_kv_cache_interleave_size=16,
    )

    assert result is topk_indices_buffer
    assert topk_scores_buffer.tolist() == [[456] * topk, [456] * topk]
    assert calls[-1] == (
        "paged_index_topk",
        tuple(q_quant.shape),
        (page_table_width, 64 * 132),
        (64 * 132, 1),
        64,
        num_heads,
        True,
    )
    assert merge_calls == [
        (
            topk_indices_buffer.data_ptr(),
            topk_scores_buffer.data_ptr(),
            topk,
            2,
            1,
            16,
        )
    ]


def test_b12x_dcp_merge_passes_contiguous_scores_to_topk(monkeypatch):
    run_row_topk_calls: list[tuple[bool, tuple[int, ...]]] = []

    def run_row_topk(*, row_logits, lengths, topk, output_values, output_indices):
        run_row_topk_calls.append((row_logits.is_contiguous(), tuple(row_logits.shape)))
        assert row_logits.is_contiguous()
        assert lengths.tolist() == [row_logits.shape[1]] * row_logits.shape[0]
        positions = torch.arange(topk, dtype=output_indices.dtype).expand_as(
            output_indices
        )
        output_indices.copy_(positions)
        output_values.copy_(row_logits[:, :topk])

    _install_fake_b12x_dcp_merge(monkeypatch, run_row_topk, world_size=2)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )

    topk_indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    topk_scores = torch.tensor(
        [[0.1, 0.4, 0.3, 0.2], [0.8, 0.5, 0.7, 0.6]],
        dtype=torch.float32,
    )

    indexer_mod._merge_b12x_dcp_topk(
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        topk_tokens=4,
        dcp_world_size=2,
        dcp_rank=0,
        cp_kv_cache_interleave_size=16,
    )

    assert run_row_topk_calls == [(True, (2, 8))]


def test_b12x_dcp_merge_warmup_reserves_workspace(monkeypatch):
    def run_row_topk(*, row_logits, lengths, topk, output_values, output_indices):
        positions = torch.arange(topk, dtype=output_indices.dtype).expand_as(
            output_indices
        )
        output_indices.copy_(positions)
        output_values.copy_(row_logits[:, :topk])

    _install_fake_b12x_dcp_merge(monkeypatch, run_row_topk, world_size=4)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(indexer_mod, "_use_triton_dcp_remap", lambda _: False)

    indexer_mod._prewarm_b12x_dcp_topk_merge(
        q_rows=3,
        topk_tokens=4,
        dcp_world_size=4,
        dcp_rank=1,
        cp_kv_cache_interleave_size=1,
        device=torch.device("cpu"),
    )

    assert workspace_manager.specs == (
        ((3, 2, 4), torch.int32),
        ((12, 2, 4), torch.int32),
        ((3, 16), torch.int32),
        ((3, 16), torch.int32),
        ((3,), torch.int32),
    )


def test_dcp_warmup_params_use_group_when_config_missing(monkeypatch):
    import vllm.config as vllm_config
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: None,
    )
    monkeypatch.setattr(
        parallel_state,
        "get_dcp_group",
        lambda: types.SimpleNamespace(world_size=4, rank_in_group=2),
    )

    assert indexer_mod._get_dcp_warmup_params() == (4, 2, 1)


def test_b12x_profile_skips_legacy_logits_dummy_allocation(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(
        indexer_mod,
        "get_forward_context",
        _profile_forward_context,
    )
    monkeypatch.setattr(
        indexer_mod,
        "_ensure_b12x_sparse_indexer_supported",
        lambda: None,
    )
    monkeypatch.setattr(
        indexer_mod.current_platform,
        "fp8_dtype",
        lambda: torch.uint8,
        raising=False,
    )
    monkeypatch.setattr(indexer_mod.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)

    q_rows = 2
    topk = 4
    total_seq_lens = 1024
    hidden_states = torch.empty((q_rows, 128), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 132), dtype=torch.uint8)
    q_quant = torch.empty((q_rows, 1, 128), dtype=torch.uint8)
    k = torch.empty((total_seq_lens, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, 1), dtype=torch.float32)
    topk_indices_buffer = torch.empty((q_rows, topk), dtype=torch.int32)

    legacy_logits_elems = 1024 * 1024
    torch_empty = torch.empty

    def guarded_empty(*args, **kwargs):
        shape = args[0] if args else kwargs.get("size")
        if shape == legacy_logits_elems:
            raise AssertionError("B12X profile path allocated legacy logits dummy")
        if isinstance(shape, tuple) and shape == (legacy_logits_elems,):
            raise AssertionError("B12X profile path allocated legacy logits dummy")
        return torch_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", guarded_empty)

    result = indexer_mod.sparse_attn_indexer(
        hidden_states,
        "layers.0.attn",
        kv_cache,
        q_quant,
        None,
        k,
        weights,
        128,
        None,
        topk_tokens=topk,
        head_dim=128,
        max_model_len=total_seq_lens,
        total_seq_lens=total_seq_lens,
        topk_indices_buffer=topk_indices_buffer,
        skip_k_cache_insert=False,
        use_fp4_cache=False,
        use_b12x_sparse_indexer=True,
    )

    assert result is topk_indices_buffer
    assert workspace_manager.specs == (
        ((q_rows, topk), torch.int32),
        ((q_rows,), torch.int32),
    )
    assert calls == [
        (
            "indexer_plan",
            "paged",
            "decode",
            False,
            q_rows,
            total_seq_lens // 64,
            topk,
            "paged_fused",
        ),
        (
            "indexer_plan",
            "paged",
            "prefill",
            True,
            q_rows,
            total_seq_lens // 64,
            topk,
            "packed_contiguous",
        ),
        (
            "indexer_plan",
            "paged",
            "prefill",
            True,
            q_rows,
            total_seq_lens // 64,
            topk,
            "packed_contiguous",
        ),
        (
            "paged_bind",
            (q_rows, total_seq_lens // 64),
            (q_rows,),
            1,
            True,
            False,
            False,
        ),
        (
            "paged_index_topk",
            tuple(q_quant.shape),
            (1, 64 * 132),
            (64 * 132, 1),
            64,
            1,
            False,
        ),
    ]


def test_b12x_profile_work_skips_piecewise_capture(monkeypatch):
    def fail_profile_work(*args, **kwargs):
        raise AssertionError("profile work should not run during capture")

    monkeypatch.setattr(indexer_mod, "current_workspace_manager", fail_profile_work)
    monkeypatch.setattr(
        indexer_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            attn_metadata=None,
            cudagraph_runtime_mode=indexer_mod.CUDAGraphMode.PIECEWISE,
            batch_descriptor=object(),
        ),
    )
    monkeypatch.setattr(
        indexer_mod.current_platform,
        "fp8_dtype",
        lambda: torch.uint8,
        raising=False,
    )

    topk = 4
    hidden_states = torch.empty((128, 128), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 132), dtype=torch.uint8)
    q_quant = torch.empty((128, 1, 128), dtype=torch.uint8)
    k = torch.empty((128, 128), dtype=torch.uint8)
    weights = torch.empty((128, 1), dtype=torch.float32)
    topk_indices_buffer = torch.empty((128, topk), dtype=torch.int32)

    result = indexer_mod.sparse_attn_indexer(
        hidden_states,
        "layers.0.attn",
        kv_cache,
        q_quant,
        None,
        k,
        weights,
        128,
        None,
        topk_tokens=topk,
        head_dim=128,
        max_model_len=32768,
        total_seq_lens=128,
        topk_indices_buffer=topk_indices_buffer,
        skip_k_cache_insert=False,
        use_fp4_cache=False,
        use_b12x_sparse_indexer=True,
    )

    assert result is topk_indices_buffer


def test_b12x_dcp_profile_uses_planner_page_table_width(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(
        indexer_mod, "current_workspace_manager", lambda: workspace_manager
    )
    monkeypatch.setattr(
        indexer_mod,
        "get_forward_context",
        _profile_forward_context,
    )
    monkeypatch.setattr(
        indexer_mod,
        "_ensure_b12x_sparse_indexer_supported",
        lambda: None,
    )
    monkeypatch.setattr(
        indexer_mod.current_platform,
        "fp8_dtype",
        lambda: torch.uint8,
        raising=False,
    )
    monkeypatch.setattr(indexer_mod.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
    monkeypatch.setattr(indexer_mod, "_get_dcp_warmup_params", lambda: (2, 0, 1))
    monkeypatch.setattr(indexer_mod, "_prewarm_b12x_dcp_topk_merge", lambda **_: None)

    q_rows = 8192
    profile_q_rows = 4096
    topk = 4
    max_model_len = 32768
    # This is an aggregate prefill-buffer capacity, not a per-request K width.
    total_seq_lens = 40 * max_model_len
    page_table_width = max_model_len // 64
    hidden_states = torch.empty((q_rows, 128), dtype=torch.bfloat16)
    kv_cache = torch.empty((1, 64, 132), dtype=torch.uint8)
    q_quant = torch.empty((q_rows, 1, 128), dtype=torch.uint8)
    k = torch.empty((q_rows, 128), dtype=torch.uint8)
    weights = torch.empty((q_rows, 1), dtype=torch.float32)
    topk_indices_buffer = torch.empty((q_rows, topk), dtype=torch.int32)

    result = indexer_mod.sparse_attn_indexer(
        hidden_states,
        "layers.0.attn",
        kv_cache,
        q_quant,
        None,
        k,
        weights,
        128,
        None,
        topk_tokens=topk,
        head_dim=128,
        max_model_len=max_model_len,
        total_seq_lens=total_seq_lens,
        topk_indices_buffer=topk_indices_buffer,
        skip_k_cache_insert=False,
        use_fp4_cache=False,
        use_b12x_sparse_indexer=True,
    )

    assert result is topk_indices_buffer
    assert workspace_manager.specs == (
        ((profile_q_rows, topk), torch.int32),
        ((profile_q_rows,), torch.int32),
    )
    assert calls == [
        (
            "indexer_plan",
            "paged",
            "decode",
            False,
            profile_q_rows,
            page_table_width,
            topk,
            "paged_fused",
        ),
        (
            "indexer_plan",
            "paged",
            "prefill",
            True,
            profile_q_rows,
            page_table_width,
            topk,
            "packed_contiguous",
        ),
        (
            "indexer_plan",
            "paged",
            "prefill",
            True,
            profile_q_rows,
            page_table_width,
            topk,
            "packed_contiguous",
        ),
        (
            "paged_bind",
            (profile_q_rows, page_table_width),
            (profile_q_rows,),
            1,
            True,
            False,
            False,
        ),
        (
            "paged_index_topk",
            (profile_q_rows, 1, 128),
            (1, 64 * 132),
            (64 * 132, 1),
            64,
            1,
            False,
        ),
    ]


def test_b12x_paged_profile_rows_follow_logits_budget(monkeypatch):
    monkeypatch.setattr(indexer_mod.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 512)
    monkeypatch.delenv("B12X_PAGED_INDEX_SUPERTILE_K", raising=False)

    assert indexer_mod._get_b12x_paged_indexer_profile_q_rows(q_rows=65536) == 4096

    assert indexer_mod._get_b12x_paged_indexer_profile_q_rows(q_rows=1024) == 1024


def test_b12x_schedule_metadata_uses_canonical_indexer_import(monkeypatch):
    calls: list[tuple] = []
    _install_fake_b12x_indexer(monkeypatch, calls)

    from vllm.v1.attention.backends.mla import indexer as mla_indexer_mod

    monkeypatch.setattr(
        mla_indexer_mod.envs,
        "VLLM_USE_B12X_SPARSE_INDEXER",
        True,
    )
    builder = object.__new__(mla_indexer_mod.DeepseekV32IndexerMetadataBuilder)
    builder.scheduler_metadata_buffer = torch.zeros((5, 2), dtype=torch.int32)
    builder.kv_cache_spec = types.SimpleNamespace(storage_block_size=64)
    builder.num_sms = 4

    seq_lens = torch.tensor([64, 128], dtype=torch.int32)
    block_table = torch.zeros((2, 3), dtype=torch.int32)

    result = builder._maybe_build_b12x_schedule_metadata(
        seq_lens=seq_lens,
        block_table=block_table,
        num_decode_tokens=2,
        requires_padding=False,
    )

    assert result is builder.scheduler_metadata_buffer
    assert result.tolist() == [[7, 7]] * 5
    assert calls == [
        ("uses_schedule", 2, 3),
        ("build_schedule", (64, 128), 64, 4, True),
    ]


def test_mtp_variable_decode_preserves_block_table_alignment_padding():
    from vllm.v1.attention.backends.mla import indexer as mla_indexer_mod

    assert mla_indexer_mod._align_block_table_width(1875, 64) == 1876
    assert mla_indexer_mod._align_block_table_width(1874, 64) == 1874

    builder = object.__new__(mla_indexer_mod.DeepseekV32IndexerMetadataBuilder)
    # max_model_len=480000, block_size=64, DCP=4 requires 1875 logical
    # blocks. The model runner aligns that row to 1876 entries.
    aligned_width = 1876
    buffer_rows = 16
    builder.decode_seq_lens_buffer = torch.zeros(buffer_rows, dtype=torch.int32)
    builder.expanded_block_table_buffer = torch.zeros(
        (buffer_rows, aligned_width), dtype=torch.int32
    )
    builder.decode_lens_buffer = torch.zeros(buffer_rows, dtype=torch.int32)
    builder.arange_buffer = torch.arange(buffer_rows, dtype=torch.int32)
    builder.compress_ratio = 1

    block_table = torch.arange(3 * aligned_width, dtype=torch.int32).view(
        3, aligned_width
    )
    seq_lens = torch.tensor([100, 200, 300], dtype=torch.int32)
    decode_lens = torch.tensor([3, 1, 3], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 4], dtype=torch.int32)

    _, expanded, _, batch_size, requires_padding = builder._prepare_decode_tensors(
        seq_lens=seq_lens,
        block_table=block_table,
        decode_lens=decode_lens,
        decode_lens_cpu=decode_lens,
        query_start_loc=query_start_loc,
        num_decodes=3,
        num_decode_tokens=7,
        use_native=False,
        next_n=4,
        max_decode_len=3,
    )

    expected = torch.repeat_interleave(block_table, decode_lens, dim=0, output_size=7)
    assert expanded.shape == (7, aligned_width)
    torch.testing.assert_close(expanded, expected)
    assert batch_size == 7
    assert not requires_padding

    # Future allocation drift must fail before either the Triton copy or a
    # paged indexer kernel can consume mismatched row strides.
    builder.expanded_block_table_buffer = torch.zeros(
        (buffer_rows, 1875), dtype=torch.int32
    )
    with pytest.raises(ValueError, match="block-table row width"):
        builder._prepare_decode_tensors(
            seq_lens=seq_lens,
            block_table=block_table,
            decode_lens=decode_lens,
            decode_lens_cpu=decode_lens,
            query_start_loc=query_start_loc,
            num_decodes=3,
            num_decode_tokens=7,
            use_native=False,
            next_n=4,
            max_decode_len=3,
        )
