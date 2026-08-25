# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import torch

from vllm.models.deepseek_v4.nvidia import b12x as b12x_mla
from vllm.models.deepseek_v4.nvidia import b12x_indexer


class _Workspace:
    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes]


def test_b12x_dsv4_backend_preserves_cache_contract() -> None:
    backend = b12x_mla.DeepseekV4B12xSparseMLABackend

    assert backend.get_name() == "B12X"
    assert "auto" in backend.supported_kv_cache_dtypes
    assert not backend.supports_pcp()
    assert not b12x_indexer.DeepseekV4B12xIndexerBackend.supports_pcp()

    storage = torch.empty((2, 600), dtype=torch.uint8)
    page_view = b12x_mla._cache_page_view(storage, page_size=1, name="cache")

    assert page_view.shape == (2, 584)
    assert page_view.stride() == (600, 1)
    assert (
        page_view.untyped_storage().data_ptr() == storage.untyped_storage().data_ptr()
    )


def test_b12x_compressed_sparse_mla_uses_plan_bind_run(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def make_caps(**kwargs):
        calls["caps"] = kwargs
        return SimpleNamespace(**kwargs)

    def bind(**kwargs):
        calls["bind"] = kwargs
        return SimpleNamespace(scratch=SimpleNamespace(mode=None))

    plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((32,), torch.uint8),),
        bind=bind,
    )

    def run(**kwargs):
        calls["run"] = kwargs
        kwargs["out"].fill_(3)

    module = SimpleNamespace(
        Caps=make_caps,
        plan=lambda caps: plan,
        run=run,
        split_chunks_for_contract=lambda **kwargs: 5,
    )
    monkeypatch.setattr(b12x_mla, "_require_b12x_compressed_sparse_mla", lambda: module)
    monkeypatch.setattr(b12x_mla, "current_workspace_manager", lambda: _Workspace())

    q = torch.empty((2, 16, 512), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    b12x_mla._run_compressed_sparse_mla(
        q=q,
        output=output,
        attn_sink=torch.zeros((32,), dtype=torch.float32),
        scale=0.125,
        swa_k_cache=torch.empty((1, 584), dtype=torch.uint8),
        swa_indices=torch.zeros((2, 3), dtype=torch.int32),
        swa_lens=torch.full((2,), 3, dtype=torch.int32),
        swa_page_size=1,
        indexed_k_cache=torch.empty((1, 584), dtype=torch.uint8),
        indexed_indices=torch.zeros((2, 4), dtype=torch.int32),
        indexed_lens=torch.full((2,), 4, dtype=torch.int32),
        indexed_page_size=1,
        mode="decode",
        decode_row_capacity=8,
    )

    assert calls["caps"]["max_width"] == 7
    assert calls["caps"]["max_chunks_per_row"] == 5
    assert calls["bind"]["scratch"][0].dtype == torch.uint8
    assert calls["run"]["binding"].scratch.mode == "decode"
    assert calls["run"]["attn_sink"].shape == (16,)
    assert calls["run"]["out"] is output
    assert torch.count_nonzero(output != 3) == 0


def test_b12x_wo_projection_packs_and_runs(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def pack_weights(*args, **kwargs):
        calls["pack"] = (args, kwargs)
        return object()

    def run_inv_rope(*args, **kwargs):
        calls["run"] = (args, kwargs)
        return torch.full((args[0].shape[0], 256), 7, dtype=torch.bfloat16)

    module = SimpleNamespace(
        is_supported=lambda: True,
        pack_weights=pack_weights,
        run_inv_rope=run_inv_rope,
    )
    monkeypatch.setattr(b12x_mla, "get_b12x_wo_projection", lambda: module)
    monkeypatch.setattr(
        b12x_mla,
        "current_stream",
        lambda: SimpleNamespace(cuda_stream=123),
    )

    layer = object.__new__(b12x_mla.DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.n_local_groups = 2
    layer.n_local_heads = 4
    layer.head_dim = 128
    layer.nope_head_dim = 96
    layer.rope_head_dim = 32
    layer.o_lora_rank = 128
    layer.hidden_size = 256
    layer.rotary_emb = SimpleNamespace(cos_sin_cache=torch.empty((1, 64)))
    layer.wo_a = SimpleNamespace(
        weight=torch.empty((256, 256), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.empty((2, 2), dtype=torch.float32),
        b12x_warmup_provider=object(),
    )
    layer.wo_b = SimpleNamespace(
        weight=torch.empty((256, 256), dtype=torch.float8_e4m3fn),
        weight_scale_inv=torch.empty((2, 2), dtype=torch.float32),
        b12x_warmup_provider=object(),
        reduce_results=False,
        tp_size=2,
    )
    layer._b12x_wo_projection_weights = None

    layer.setup_b12x_wo_projection()
    output = layer._o_proj(
        torch.empty((3, 4, 128), dtype=torch.bfloat16),
        torch.arange(3),
    )

    assert calls["pack"][1] == {
        "groups": 2,
        "group_width": 256,
        "rank": 128,
        "hidden": 256,
    }
    assert calls["run"][1]["heads_per_group"] == 2
    assert calls["run"][1]["stream"] == 123
    assert layer.wo_a.b12x_warmup_provider is None
    assert layer.wo_b.b12x_warmup_provider is None
    assert output.shape == (3, 256)
    assert torch.count_nonzero(output != 7) == 0


def test_b12x_mhc_uses_plan_bind_run(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def make_caps(**kwargs):
        calls["caps"] = kwargs
        return SimpleNamespace(**kwargs)

    def bind(plan, **kwargs):
        calls["bind"] = (plan, kwargs)
        return SimpleNamespace(**kwargs)

    plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
    )

    def run_pre(*args, **kwargs):
        calls["pre"] = (args, kwargs)
        binding = kwargs["binding"]
        return binding.out, binding.post, binding.comb, binding.y

    def run_post_pre(*args, **kwargs):
        calls["post_pre"] = (args, kwargs)
        binding = kwargs["binding"]
        return binding.out, binding.post, binding.comb, binding.y

    def run_post(*args):
        calls["post"] = args
        return args[1]

    module = SimpleNamespace(
        Caps=make_caps,
        DEFAULT_BLOCK_K=128,
        MULT=4,
        bind=bind,
        plan=lambda caps: plan,
        run_post=run_post,
        run_post_pre=run_post_pre,
        run_pre=run_pre,
    )
    monkeypatch.setattr(b12x_mla, "_require_b12x_mhc", lambda: module)
    monkeypatch.setattr(b12x_mla, "current_workspace_manager", lambda: _Workspace())

    mhc = b12x_mla.B12xMHCResidual(
        hidden_size=256,
        hc_mult=4,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    residual = torch.empty((3, 256), dtype=torch.bfloat16)
    hc_fn = torch.empty((24, 256), dtype=torch.float32)
    hc_scale = torch.empty((3,), dtype=torch.float32)
    hc_base = torch.empty((24,), dtype=torch.float32)
    norm_weight = torch.empty((256,), dtype=torch.bfloat16)

    residual_out, post, comb, layer_input = mhc.run_pre(
        residual,
        hc_fn,
        hc_scale,
        hc_base,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    next_outputs = mhc.run_post_pre(
        layer_input,
        residual_out,
        post,
        comb,
        torch.empty((24, 1024), dtype=torch.float32),
        hc_scale,
        hc_base,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    final = mhc.run_post(layer_input, *next_outputs[:3])

    assert calls["caps"]["hidden_size"] == 256
    assert calls["caps"]["split_k"] == 8
    assert calls["bind"][1]["scratch"].dtype == torch.uint8
    assert calls["pre"][1]["binding"].expected_m == 3
    assert calls["post_pre"][1]["expected_m"] == 3
    assert residual_out.shape == (3, 4, 256)
    assert layer_input.shape == (3, 256)
    assert final is next_outputs[0]


def test_b12x_dsv4_indexer_uses_logical_slots(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def make_caps(**kwargs):
        calls["caps"] = kwargs
        return SimpleNamespace(**kwargs)

    def bind(**kwargs):
        calls["bind"] = kwargs
        return SimpleNamespace(route="packed_contiguous")

    plan = SimpleNamespace(
        route="packed_contiguous",
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
        bind=bind,
    )

    def index_topk_fp8(**kwargs):
        calls["run"] = kwargs
        kwargs["out_indices"].fill_(11)

    module = SimpleNamespace(
        Caps=make_caps,
        SOURCE_LAYOUT_PAGED="paged",
        PAGED_INDEX_PAGE_SIZE=64,
        plan=lambda caps: plan,
        index_topk_fp8=index_topk_fp8,
    )
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    monkeypatch.setattr(b12x_indexer, "current_workspace_manager", lambda: _Workspace())

    output = torch.empty((3, 4), dtype=torch.int32)
    b12x_indexer._run_paged_topk(
        q=torch.empty((3, 16, 128), dtype=torch.float8_e4m3fn),
        weights=torch.empty((3, 16, 1), dtype=torch.float32),
        kv_cache=torch.empty((4, 64, 132), dtype=torch.uint8),
        seq_lens=torch.full((3,), 128, dtype=torch.int32),
        block_table=torch.zeros((3, 2), dtype=torch.int32),
        schedule_metadata=None,
        active_width=None,
        output=output,
        topk=4,
        shared_page_table=True,
    )

    builder = object.__new__(b12x_indexer.DeepseekV4B12xIndexerMetadataBuilder)
    builder.prefill_k_rows = 32768
    builder.max_prefill_buffer_size = 1 << 30
    assert builder._supports_native_decode(8)
    assert builder._split_prefill_chunks(
        torch.tensor([64, 65536, 131072]),
        torch.tensor([1, 1]),
        num_decodes=1,
        max_logits_bytes=1 << 30,
    ) == [
        (slice(1, 2), slice(0, 1)),
        (slice(2, 3), slice(0, 1)),
    ]
    assert calls["caps"]["source_layout"] == "paged"
    assert calls["bind"]["output_physical_slots"] is False
    assert calls["run"]["out_indices"] is output
    assert torch.count_nonzero(output != 11) == 0

    indexer = b12x_indexer.DeepseekV4B12xSparseIndexer(
        SimpleNamespace(),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=512,
        head_dim=128,
        max_model_len=65536,
        max_total_seq_len=65536,
        topk_indices_buffer=torch.empty((2, 512), dtype=torch.int32),
        skip_k_cache_insert=True,
        compress_ratio=4,
    )
    indexer._reserve_profile_workspace(
        torch.empty((2, 64, 128), dtype=torch.float8_e4m3fn)
    )
    assert calls["caps"]["max_page_table_width"] == 1024
