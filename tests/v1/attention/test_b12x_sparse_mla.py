# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import torch

from vllm.config import AttentionConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import (
    _canonicalize_sparse_mla_kv_cache_dtype,
)
from vllm.models.deepseek_v32.attention import (
    DeepseekV32Indexer,
    _select_sparse_components,
)
from vllm.models.deepseek_v32.b12x import B12xDeepseekV32Indexer
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.b12x import B12xPagedAttentionBackend
from vllm.v1.attention.backends.mla import b12x_indexer, b12x_mla_sparse
from vllm.v1.attention.backends.mla.b12x_indexer import B12xIndexerBackend
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class _Workspace:
    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes]


def test_b12x_selector_routes_non_compressed_sparse_mla() -> None:
    assert AttentionConfig(backend="b12x").backend == AttentionBackendEnum.B12X
    assert AttentionBackendEnum.B12X.get_class() is B12xPagedAttentionBackend
    assert B12xMLASparseBackend.get_name() == "B12X"
    assert not B12xIndexerBackend.supports_device_cpu_query_lens_mismatch()
    assert not B12xMLASparseBackend.supports_device_cpu_query_lens_mismatch()

    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.B12X)
    )
    indexer_cls, backend_cls = _select_sparse_components(
        config, None, DeepseekV32Indexer
    )
    assert indexer_cls is B12xDeepseekV32Indexer
    assert backend_cls is B12xMLASparseBackend


def test_b12x_sparse_mla_accepts_glm_dsa_contract(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="glm_moe_dsa",
                index_topk=2048,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                qk_nope_head_dim=192,
                v_head_dim=256,
            )
        )
    )

    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []
    assert (
        _canonicalize_sparse_mla_kv_cache_dtype(B12xMLASparseBackend, "auto")
        == "fp8_ds_mla"
    )


def test_b12x_indexer_exposes_scores_for_dcp(monkeypatch) -> None:
    calls: dict[str, Any] = {}

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
        kwargs["out_indices"].fill_(7)
        kwargs["out_scores"].fill_(0.5)

    module = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        SOURCE_LAYOUT_PAGED="paged",
        PAGED_INDEX_PAGE_SIZE=64,
        plan=lambda caps: plan,
        index_topk_fp8=index_topk_fp8,
    )
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    monkeypatch.setattr(
        b12x_indexer,
        "current_workspace_manager",
        lambda: _Workspace(),
    )

    output = torch.empty((2, 4), dtype=torch.int32)
    scores = torch.empty((2, 4), dtype=torch.float32)
    b12x_indexer._run_paged_topk(
        q=torch.empty((2, 32, 128), dtype=torch.float8_e4m3fn),
        weights=torch.empty((2, 32), dtype=torch.float32),
        kv_cache=torch.empty((4, 64, 132), dtype=torch.uint8),
        seq_lens=torch.full((2,), 128, dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        schedule_metadata=None,
        active_width=None,
        output=output,
        scores=scores,
        topk=4,
        shared_page_table=True,
    )

    assert calls["bind"]["output_physical_slots"] is False
    assert calls["run"]["out_scores"] is scores
    assert torch.count_nonzero(output != 7) == 0
    assert torch.count_nonzero(scores != 0.5) == 0
