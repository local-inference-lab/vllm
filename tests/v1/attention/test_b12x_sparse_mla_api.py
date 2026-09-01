# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for the B12x sparse MLA adapters."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config import AttentionConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    _canonicalize_sparse_mla_kv_cache_dtype,
    _maybe_view_mla_cache_as_fp8,
    _uses_packed_sparse_mla_workspace,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadataBuilder,
)
from vllm.models.deepseek_v4.nvidia import b12x as b12x_mla
from vllm.models.deepseek_v4.nvidia import b12x_indexer
from vllm.models.deepseek_v32.attention import (
    DeepseekV32Indexer,
    _select_sparse_components,
)
from vllm.models.deepseek_v32.b12x import B12xDeepseekV32Indexer
from vllm.platforms.interface import DeviceCapability, Platform
from vllm.v1.attention.backends.b12x import B12xPagedAttentionBackend
from vllm.v1.attention.backends.mla import b12x_indexer as generic_b12x_indexer
from vllm.v1.attention.backends.mla import b12x_mla_sparse
from vllm.v1.attention.backends.mla.b12x_indexer import B12xIndexerBackend
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xGLM5NextMLASparseBackend,
    B12xGLM5NextMLASparseMetadataBuilder,
    B12xGLMDSAMLASparseBackend,
    B12xMLASparseBackend,
    B12xMLASparseImpl,
    B12xMLASparseMetadata,
    B12xMLASparseMetadataBuilder,
    _global_causal_lens_for_ckv_gather,
    _is_glm_next_ckv_source_layout,
    _is_speculative_decode_batch,
    _max_speculative_decode_query_len,
    _selected_index_block_stride_rows,
    _use_b12x_full_ckv_gather,
)
from vllm.v1.attention.backends.mla.sparse_utils import _remap_tiling
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import select_common_block_size


class _Workspace:
    def get_simultaneous(self, *shapes_and_dtypes):
        return [torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes]


def test_b12x_selector_routes_supported_attention_families() -> None:
    assert AttentionConfig(backend="b12x").backend == AttentionBackendEnum.B12X
    assert AttentionBackendEnum.B12X.get_class() is B12xPagedAttentionBackend
    assert B12xMLASparseBackend.get_name() == "B12X"
    assert b12x_mla.DeepseekV4B12xSparseMLABackend.get_name() == "B12X"
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

    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
    )
    indexer_cls, backend_cls = _select_sparse_components(
        config, None, DeepseekV32Indexer
    )
    assert indexer_cls is B12xDeepseekV32Indexer
    assert backend_cls is B12xGLMDSAMLASparseBackend


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

    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="nvfp4_ds_mla",
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


def test_b12x_glm_dsa_nvfp4_cache_spec(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
        )
    )
    probe = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.uint8,
        cache_dtype_str="nvfp4_ds_mla",
    )

    with set_current_vllm_config(config):
        packed = B12xMLASparseBackend.customize_spec(probe)

    assert packed.state_content_bytes == 368
    assert packed.page_size_bytes == 64 * 368
    assert packed.model_version == "glm_moe_dsa"
    assert B12xMLASparseBackend.customize_spec(packed) == packed

    packed_without_config = B12xGLMDSAMLASparseBackend.customize_spec(probe)
    assert packed_without_config == packed


def test_b12x_nvfp4_rejects_non_glm_dsa_architecture(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="deepseek_v32",
                index_topk=2048,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            )
        )
    )

    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="nvfp4_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == [
        ("B12X nvfp4_ds_mla requires GLM5Next or the GLM-5.2/5.3 DSA architecture")
    ]


def test_b12x_dsa_requires_layer_compact_cache_layout() -> None:
    assert B12xMLASparseBackend.supported_kv_cache_layouts() == (KVCacheLayout.LBNHC,)


def test_b12x_glm5_next_requires_block_outermost_cache_layout() -> None:
    assert B12xGLM5NextMLASparseBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
    )


def _glm5_next_config(
    *,
    dcp_size: int = 1,
    cp_interleave: int = 1,
    speculative: bool = False,
    prefix_caching: bool = False,
    **overrides: int,
) -> SimpleNamespace:
    recipe = dict(
        model_type="glm5_next_text",
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
    )
    recipe.update(overrides)
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(**recipe)),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=cp_interleave,
        ),
        speculative_config=object() if speculative else None,
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
    )


def test_b12x_glm5_next_cache_spec_and_layout(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    config = _glm5_next_config()
    probe = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=656,
    )
    unidentified = B12xMLASparseBackend.customize_spec(probe)
    packed_by_glm_backend = B12xGLM5NextMLASparseBackend.customize_spec(probe)
    with set_current_vllm_config(config):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
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
        packed = B12xMLASparseBackend.customize_spec(probe)
        layouts = B12xGLM5NextMLASparseBackend.supported_kv_cache_layouts()
    packed_without_config_context = B12xMLASparseBackend.customize_spec(packed)

    assert invalid_reasons == []
    assert unidentified == probe
    assert packed_by_glm_backend.state_content_bytes == 528
    assert packed_by_glm_backend.page_size_padded is None
    assert packed_by_glm_backend.page_tail_bytes_per_token == 132 // 4
    assert packed_by_glm_backend.page_size_bytes == 64 * (528 + 132 // 4)
    assert packed_by_glm_backend.model_version == "glm5_next"
    assert packed.state_content_bytes == 528
    assert packed.page_size_padded is None
    assert packed.page_tail_bytes_per_token == 132 // 4
    assert packed.page_size_bytes == 64 * (528 + 132 // 4)
    assert packed.model_version == "glm5_next"
    assert packed_without_config_context == packed
    assert layouts == (KVCacheLayout.BLHNC,)
    assert "nvfp4_ds_mla" in B12xMLASparseBackend.supported_kv_cache_dtypes


def test_b12x_glm5_next_nvfp4_cache_spec() -> None:
    probe = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="nvfp4_ds_mla",
        state_content_bytes=656,
    )

    with set_current_vllm_config(_glm5_next_config()):
        packed = B12xMLASparseBackend.customize_spec(probe)

    assert packed.state_content_bytes == 304
    assert packed.page_tail_bytes_per_token == 33
    assert packed.page_size_padded == 3 * 64 * 132
    assert packed.page_size_bytes == 3 * 64 * 132 + 64 * 33
    assert packed.model_version == "glm5_next"


def test_b12x_glm5_next_binds_nvfp4_record_width(monkeypatch) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._cache_record_bytes = 304
    impl._ckv_gather_enabled = False
    planned: list[int] = []
    monkeypatch.setattr(impl, "_set_kernel_page_size", planned.append)

    impl.bind_kv_cache(torch.empty((2, 64, 304), dtype=torch.uint8))

    assert planned == [64]
    with pytest.raises(ValueError, match="page_size, 304"):
        impl.bind_kv_cache(torch.empty((2, 64, 528), dtype=torch.uint8))


def test_b12x_glm_dsa_binds_nvfp4_fp8_rope_record() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._uses_glm_dsa_nvfp4_cache = True

    impl.bind_kv_cache(torch.empty((2, 64, 368), dtype=torch.uint8))

    with pytest.raises(ValueError, match="page_size, 368"):
        impl.bind_kv_cache(torch.empty((2, 64, 432), dtype=torch.uint8))


def test_b12x_nvfp4_run_options_match_each_glm_record_abi() -> None:
    assert b12x_mla_sparse._nvfp4_run_options(is_glm_next=False) == {
        "scale_format": 2,
        "fp8_rope": True,
    }
    assert b12x_mla_sparse._nvfp4_run_options(is_glm_next=True) == {
        "scale_format": 2,
        "fp8_rope": False,
        "latent_scale_per_token": True,
    }


def test_packed_nvfp4_mla_dtype_bypasses_generic_layout_guard() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        cache_config=SimpleNamespace(cache_dtype="nvfp4_ds_mla"),
    )

    assert VllmConfig.validate_nvfp4_kv_cache_with_mla(config) is config

    config.cache_config.cache_dtype = "nvfp4"
    with pytest.raises(ValueError, match="not supported with MLA"):
        VllmConfig.validate_nvfp4_kv_cache_with_mla(config)


@pytest.mark.parametrize("cache_dtype", ["fp8_ds_mla", "nvfp4_ds_mla"])
def test_packed_mla_cache_keeps_uint8_forward_view(cache_dtype: str) -> None:
    cache = torch.empty((2, 64, 304), dtype=torch.uint8)

    forwarded = _maybe_view_mla_cache_as_fp8(cache, cache_dtype)

    assert forwarded is cache
    assert forwarded.dtype == torch.uint8


def test_plain_fp8_mla_cache_uses_native_fp8_forward_view(monkeypatch) -> None:
    cache = torch.empty((2, 64, 512), dtype=torch.uint8)
    monkeypatch.setattr(
        "vllm.model_executor.layers.attention.mla_attention.current_platform.fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )

    forwarded = _maybe_view_mla_cache_as_fp8(cache, "fp8")

    assert forwarded.data_ptr() == cache.data_ptr()
    assert forwarded.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize(
    ("resolved_cache_dtype", "expected"),
    [
        ("fp8_ds_mla", True),
        ("nvfp4_ds_mla", True),
        ("auto", False),
        (None, False),
    ],
)
def test_packed_workspace_uses_resolved_layer_cache_spec(
    resolved_cache_dtype: str | None,
    expected: bool,
) -> None:
    spec = SimpleNamespace(cache_dtype_str=resolved_cache_dtype)

    assert _uses_packed_sparse_mla_workspace(spec) is expected


def test_b12x_glm5_next_keeps_hybrid_manager_page_unsplit() -> None:
    supported = B12xGLM5NextMLASparseBackend.get_supported_kernel_block_sizes()

    assert len(supported) == 1
    assert supported[0].base == 64
    assert select_common_block_size(2304, [B12xGLM5NextMLASparseBackend]) == 2304
    assert B12xGLM5NextMLASparseBackend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
    )


def test_glm5_next_split_cache_auto_aligns_to_dcp_retention(monkeypatch) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Glm5NextForConditionalGeneration",
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(
            block_size=256,
            mamba_block_size=None,
            mamba_cache_mode="align",
            mamba_page_size_padded=1234,
            prefix_cache_retention_interval=4096,
        ),
    )
    monkeypatch.setenv("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")
    monkeypatch.setenv("VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE", "auto")

    Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)

    assert config.cache_config.block_size == 1024
    assert config.cache_config.mamba_block_size == 1024
    assert config.cache_config.mamba_page_size_padded is None


def test_glm5_next_split_cache_auto_requires_dcp_aligned_retention(
    monkeypatch,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Glm5NextForConditionalGeneration",
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            prefix_cache_retention_interval=4097,
        ),
    )
    monkeypatch.setenv("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")

    with pytest.raises(ValueError, match="divisible by decode_context_parallel_size"):
        Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)


def test_b12x_glm5_next_nvfp4_aligns_hybrid_page_to_packed_record(
    monkeypatch,
) -> None:
    mamba_page_size = 1_085_440
    config = _glm5_next_config(dcp_size=4, cp_interleave=4)
    config.model_config.is_hybrid = True
    config.model_config.use_mla = True
    config.model_config.architecture = "Glm5NextForConditionalGeneration"
    config.model_config.dtype = torch.bfloat16
    config.model_config.get_num_kv_heads = lambda parallel_config: 1
    config.model_config.get_head_size = lambda: 512
    config.cache_config.cache_dtype = "nvfp4_ds_mla"
    config.cache_config.block_size = 256
    config.cache_config.mamba_block_size = None
    config.cache_config.user_specified_mamba_block_size = False
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.mamba_page_size_padded = None

    model_cls = SimpleNamespace(
        get_mamba_state_shape_from_config=lambda vllm_config: ((mamba_page_size,),),
        get_mamba_state_dtype_from_config=lambda vllm_config: (torch.uint8,),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.ModelRegistry.resolve_model_cls",
        lambda *args, **kwargs: (model_cls, None),
    )

    Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)

    materialized_probe = MLAAttentionSpec(
        block_size=config.cache_config.block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="nvfp4_ds_mla",
        state_content_bytes=656,
    )
    with set_current_vllm_config(config):
        materialized = B12xGLM5NextMLASparseBackend.customize_spec(materialized_probe)

    assert config.cache_config.block_size == 3328
    assert config.cache_config.mamba_block_size == 3328
    assert materialized.page_size_bytes == 1_123_584
    assert config.cache_config.mamba_page_size_padded == materialized.page_size_bytes


def test_b12x_glm5_next_rejects_unaligned_dcp(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config(dcp_size=2)):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
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

    assert invalid_reasons == [
        "B12X GLM5Next C4 DCP requires cp_kv_cache_interleave_size divisible by 4"
    ]


def test_b12x_glm5_next_accepts_pool_aligned_dcp_without_speculation(
    monkeypatch,
) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config(dcp_size=4, cp_interleave=4)):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
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


def test_b12x_glm5_next_accepts_dcp_with_speculation(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(
        _glm5_next_config(dcp_size=4, cp_interleave=4, speculative=True)
    ):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
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


@pytest.mark.parametrize(
    ("max_query_len", "num_decode_tokens", "num_tokens", "expected"),
    [
        (1, 0, 32, False),
        (6, 192, 192, False),
        (6, 0, 192, True),
        (128, 0, 8192, True),
        (128, 0, 600000, False),
    ],
)
def test_b12x_full_ckv_gather_excludes_decode_and_mtp_batches(
    max_query_len: int,
    num_decode_tokens: int,
    num_tokens: int,
    expected: bool,
) -> None:
    assert (
        _use_b12x_full_ckv_gather(
            enabled=True,
            is_glm_next=True,
            dcp_world_size=4,
            max_query_len=max_query_len,
            num_tokens=num_tokens,
            num_decode_tokens=num_decode_tokens,
            min_tokens=16,
            max_tokens=524288,
        )
        is expected
    )


def test_b12x_full_ckv_gather_uses_global_causal_lengths() -> None:
    global_seq_lens = torch.tensor([5, 12], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    req_id_per_token = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)

    actual = _global_causal_lens_for_ckv_gather(
        global_seq_lens,
        query_start_loc,
        req_id_per_token,
        num_actual_tokens=5,
    )

    assert actual.tolist() == [4, 5, 10, 11, 12]


def test_b12x_glm5_next_accepts_dcp_with_prefix_caching(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(
        _glm5_next_config(
            dcp_size=4,
            cp_interleave=4,
            prefix_caching=True,
        )
    ):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
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


def test_b12x_glm5_next_rejects_dsv4_head_size(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config()):
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

    assert invalid_reasons == ["B12X GLM5Next sparse MLA requires head_size=512"]


def test_b12x_glm5_next_rejects_recipe_drift(monkeypatch) -> None:
    monkeypatch.setattr(b12x_mla_sparse, "get_b12x_sparse_mla", lambda: object())
    with set_current_vllm_config(_glm5_next_config(index_kpool=8)):
        invalid_reasons = B12xMLASparseBackend.validate_configuration(
            head_size=512,
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

    assert invalid_reasons == [
        "B12X GLM5Next sparse MLA requires index_kpool=8 (expected 4)"
    ]


def test_b12x_glm5_next_ckv_source_layout() -> None:
    storage = torch.empty((2 * 37888,), dtype=torch.uint8)
    cache = torch.as_strided(
        storage,
        size=(2, 64, 528),
        stride=(37888, 528, 1),
    )
    assert _is_glm_next_ckv_source_layout(cache, page_size=64, record_bytes=528)
    assert not _is_glm_next_ckv_source_layout(
        cache[:, :, ::2], page_size=64, record_bytes=528
    )


@pytest.mark.parametrize("record_bytes", [528, 304])
def test_b12x_glm5_next_full_ckv_workspaces_follow_cache_format(
    record_bytes: int,
) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._max_tokens = 32
    impl._q_head_dim = 512
    impl._ckv_local_capacity = 128
    impl.dcp_world_size = 4
    impl._cache_record_bytes = record_bytes
    plan = SimpleNamespace(shapes_and_dtypes=lambda: (((16,), torch.uint8),))

    specs = impl._workspace_specs(plan, input_num_heads=8, include_ckv=True)

    assert specs[-2:] == (
        ((128, record_bytes), torch.uint8),
        ((512, record_bytes), torch.uint8),
    )


@pytest.mark.parametrize("record_bytes", [528, 304])
def test_b12x_glm5_next_full_ckv_gather_preserves_native_records(
    monkeypatch: pytest.MonkeyPatch,
    record_bytes: int,
) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._kernel_page_size = 2
    impl._ckv_local_capacity = 4
    impl._cache_record_bytes = record_bytes
    impl.dcp_world_size = 2
    impl.uses_full_ckv_dcp = lambda *_: True

    kv_cache = (
        torch.arange(4 * record_bytes, dtype=torch.int64)
        .to(torch.uint8)
        .view(2, 2, record_bytes)
    )
    local_buffer = torch.full((4, record_bytes), 255, dtype=torch.uint8)
    gathered_buffer = torch.empty((8, record_bytes), dtype=torch.uint8)
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        dcp_local_total_tokens=3,
        dcp_padded_total_tokens=4,
        dcp_local_cu_seq_lens=torch.tensor([0, 3], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        num_reqs=1,
    )

    def fake_cp_gather_cache(**kwargs: Any) -> None:
        kwargs["dst"].copy_(kwargs["src_cache"].view(-1, record_bytes)[:3])

    def fake_all_gather(_group: Any, src: torch.Tensor, dst: torch.Tensor) -> None:
        dst.copy_(src.repeat(2))

    monkeypatch.setattr(b12x_mla_sparse.ops, "cp_gather_cache", fake_cp_gather_cache)
    monkeypatch.setattr(
        b12x_mla_sparse, "_dcp_all_gather_current_stream", fake_all_gather
    )
    monkeypatch.setattr(b12x_mla_sparse, "get_dcp_group", lambda: object())

    gathered = impl._gather_full_ckv(kv_cache, metadata, local_buffer, gathered_buffer)

    expected_rank = torch.cat(
        (kv_cache.view(-1, record_bytes)[:3], torch.zeros((1, record_bytes))),
        dim=0,
    )
    assert gathered.shape == (4, 2, record_bytes)
    assert torch.equal(gathered.view(-1, record_bytes), expected_rank.repeat(2, 1))


def test_b12x_glm5_next_full_ckv_gather_rejects_wrong_record_width() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._kernel_page_size = 2
    impl._ckv_local_capacity = 4
    impl._cache_record_bytes = 304
    impl.dcp_world_size = 2
    impl.uses_full_ckv_dcp = lambda *_: True
    metadata = SimpleNamespace(num_actual_tokens=1)

    with pytest.raises(ValueError, match="requires native 304-byte records"):
        impl._gather_full_ckv(
            torch.empty((2, 2, 528), dtype=torch.uint8),
            metadata,
            torch.empty((4, 304), dtype=torch.uint8),
            torch.empty((8, 304), dtype=torch.uint8),
        )


@pytest.mark.parametrize("record_bytes", [528, 656])
def test_b12x_selected_indices_use_physical_slots(record_bytes: int) -> None:
    storage = torch.empty((2, 2, 64, record_bytes), dtype=torch.uint8)
    cache = storage[:, 0]

    assert cache.stride(0) // record_bytes == 128
    assert _selected_index_block_stride_rows(cache, block_size=64) == 64


def test_sparse_index_remap_tiling_covers_glm5_next_width() -> None:
    assert _remap_tiling(2048, 128, True) == (True, 2048, 1, 8)
    assert _remap_tiling(2051, 128, True) == (False, 128, 17, 4)


def test_b12x_glm5_next_cache_writer_ignores_empty_rope() -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._concat_and_cache_glm_next_mla = lambda *args: calls.append(args)
    kv_c = torch.empty((3, 512), dtype=torch.bfloat16)
    kv_cache = torch.empty((2, 64, 528), dtype=torch.uint8)
    slots = torch.tensor([0, 64, -1], dtype=torch.int64)

    impl.do_kv_cache_update(
        kv_c,
        torch.empty((3, 1, 0), dtype=torch.bfloat16),
        kv_cache,
        slots,
        "fp8_ds_mla",
        torch.ones((), dtype=torch.float32),
    )

    assert calls == [(kv_c, kv_cache, slots)]


def test_b12x_glm_dsa_nvfp4_cache_writer_keeps_rope() -> None:
    calls: list[tuple[torch.Tensor, ...]] = []
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._uses_glm_dsa_nvfp4_cache = True
    impl._concat_and_cache_nvfp4_mla_fp8_rope = lambda *args: calls.append(args)
    kv_c = torch.empty((3, 512), dtype=torch.bfloat16)
    k_pe = torch.empty((3, 1, 64), dtype=torch.bfloat16)
    kv_cache = torch.empty((2, 64, 368), dtype=torch.uint8)
    slots = torch.tensor([0, 64, -1], dtype=torch.int64)
    scale = torch.ones((), dtype=torch.float32)

    impl.do_kv_cache_update(
        kv_c,
        k_pe,
        kv_cache,
        slots,
        "nvfp4_ds_mla",
        scale,
    )

    assert len(calls) == 1
    actual_kv_c, actual_k_pe, actual_cache, actual_slots, actual_scale = calls[0]
    assert actual_kv_c is kv_c
    assert torch.equal(actual_k_pe, k_pe.squeeze(1))
    assert actual_cache is kv_cache
    assert torch.equal(actual_slots, slots)
    assert actual_scale is scale


def test_b12x_glm5_next_cache_geometry_is_finalized_before_bind(monkeypatch) -> None:
    planned: list[SimpleNamespace] = []
    reservations: list[tuple[tuple[tuple[int, ...], torch.dtype], ...]] = []
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        b12x_mla_sparse,
        "is_workspace_manager_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        b12x_mla_sparse,
        "current_workspace_manager",
        lambda: SimpleNamespace(reserve_all=lambda *specs: reservations.append(specs)),
    )

    class FakeCaps(SimpleNamespace):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @staticmethod
        def shapes_and_dtypes():
            return ()

    class FakeModule:
        Caps = FakeCaps

        @staticmethod
        def plan(caps):
            planned.append(caps)
            return caps

    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._uses_nvfp4_cache = False
    impl._cache_record_bytes = 528
    impl._module = FakeModule
    impl._kernel_page_size = 64
    impl._kernel_page_size_finalized = False
    impl._input_num_heads = 64
    impl.num_heads = 16
    impl.dcp_world_size = 4
    impl._max_tokens = 4096
    impl._max_seqs = 4
    impl._max_speculative_decode_query_len = 6
    impl._decode_max_rows = 24
    impl._topk_tokens = 2051
    impl._kv_dtype = torch.uint8
    impl._q_head_dim = 512
    impl.kv_lora_rank = 512
    impl._model_type = 1
    impl._ckv_gather_enabled = True
    impl._ckv_capacity_tokens = 131200
    impl._ckv_local_capacity = 131200
    impl._decode_plan = SimpleNamespace()
    impl._extend_plan = SimpleNamespace()
    impl._ckv_extend_plan = SimpleNamespace()
    owner = SimpleNamespace(impl=impl, indexer=None)
    cache = torch.empty((2, 1, 2304, 528), dtype=torch.uint8)

    MLAAttention.finalize_kv_cache_geometry(
        owner,
        SimpleNamespace(cache_config=SimpleNamespace(block_size=2304)),
    )
    MLAAttention.bind_kv_cache(owner, cache)

    assert owner.kv_cache.shape == (2, 2304, 528)
    assert impl._kernel_page_size == 2304
    assert impl._kernel_page_size_finalized
    assert [(caps.mode, caps.page_size) for caps in planned] == [
        ("decode", 2304),
        ("extend", 2304),
        ("extend", 2304),
    ]
    plan_geometry = [
        (caps.num_q_heads, caps.max_q_rows, caps.max_batch) for caps in planned
    ]
    assert plan_geometry == [
        (64, 24, 24),
        (64, 4096, 4096),
        (16, 4096, 4096),
    ]
    assert len(reservations) == 3
    assert reservations[0] == (((4096, 64, 512), torch.bfloat16),)
    assert reservations[1] == (((4096, 64, 512), torch.bfloat16),)
    assert reservations[2] == (
        ((4096, 16, 512), torch.bfloat16),
        ((131328, 528), torch.uint8),
        ((525312, 528), torch.uint8),
    )

    with pytest.raises(RuntimeError, match="immutable after finalization"):
        impl.finalize_kv_cache_geometry(64)
    with pytest.raises(RuntimeError, match="does not match the finalized"):
        impl.bind_kv_cache(torch.empty((2, 64, 528), dtype=torch.uint8))


def test_b12x_glm5_next_full_ckv_bind_requires_geometry_finalization() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._cache_record_bytes = 528
    impl._ckv_gather_enabled = True
    impl._kernel_page_size_finalized = False

    with pytest.raises(RuntimeError, match="before KV-cache memory profiling"):
        impl.bind_kv_cache(torch.empty((2, 2304, 528), dtype=torch.uint8))


@pytest.mark.parametrize(
    ("parallel_drafting", "expected_query_len"),
    [(False, 6), (True, 11)],
)
def test_b12x_sparse_mla_bounds_speculative_decode_query_len(
    parallel_drafting: bool,
    expected_query_len: int,
) -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            num_speculative_tokens=5,
            parallel_drafting=parallel_drafting,
        )
    )

    assert _max_speculative_decode_query_len(config) == expected_query_len


@pytest.mark.parametrize(
    ("max_query_len", "is_prefilling", "expected"),
    [
        (1, [False, False], False),
        (6, [False, False], True),
        (6, [False, True], False),
        (7, [False, False], False),
    ],
)
def test_b12x_sparse_mla_identifies_only_speculative_verifier_batches(
    max_query_len: int,
    is_prefilling: list[bool],
    expected: bool,
) -> None:
    common = SimpleNamespace(
        num_reqs=2,
        max_query_len=max_query_len,
        is_prefilling=torch.tensor(is_prefilling),
    )

    assert _is_speculative_decode_batch(common, 6) is expected


@pytest.mark.parametrize(
    ("max_query_len", "is_spec_decode", "num_tokens", "expected_decode"),
    [
        (1, False, 4, True),
        (6, True, 24, True),
        (6, False, 24, False),
        (6, True, 25, False),
        (7, True, 24, False),
    ],
)
def test_b12x_sparse_mla_routes_only_planned_decode_rows(
    max_query_len: int,
    is_spec_decode: bool,
    num_tokens: int,
    expected_decode: bool,
) -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._max_speculative_decode_query_len = 6
    impl._decode_max_rows = 24
    metadata = SimpleNamespace(
        num_reqs=4,
        max_query_len=max_query_len,
        is_spec_decode=is_spec_decode,
    )

    assert impl._use_decode_plan(metadata, num_tokens) is expected_decode


def test_b12x_sparse_mla_reserves_largest_planned_workspace(monkeypatch) -> None:
    reservations: list[tuple[tuple[tuple[int, ...], torch.dtype], ...]] = []
    manager = SimpleNamespace(
        get_simultaneous=lambda *specs: reservations.append(specs)
    )
    monkeypatch.setattr(
        b12x_mla_sparse,
        "is_workspace_manager_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        b12x_mla_sparse,
        "current_workspace_manager",
        lambda: manager,
    )

    impl = object.__new__(B12xMLASparseImpl)
    impl._max_tokens = 64
    impl._input_num_heads = 8
    impl._q_head_dim = 512
    impl._decode_plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((32,), torch.uint8),)
    )
    impl._extend_plan = SimpleNamespace(
        shapes_and_dtypes=lambda: (((512,), torch.uint8),)
    )

    impl._reserve_planned_workspaces()

    assert reservations == [
        (
            ((64, 8, 512), torch.bfloat16),
            ((512,), torch.uint8),
        )
    ]


def _bare_glm_selector_metadata_builder() -> B12xMLASparseMetadataBuilder:
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = True
    builder.supports_draft_decode_metadata_update = True
    builder._ckv_gather_requested = False
    builder.dcp_world_size = 1
    builder._max_speculative_decode_query_len = 6
    builder._capture_default_state_slot_ids = torch.arange(4, dtype=torch.int32)
    builder._capture_state_slot_ids = torch.empty(4, dtype=torch.int32)
    builder._capture_state_is_fresh = torch.ones(4, dtype=torch.bool)
    builder._capture_num_accepted_tokens = torch.ones(4, dtype=torch.int32)
    builder._capture_is_prefilling = torch.zeros(4, dtype=torch.bool)
    return builder


def _build_short_packed_metadata(
    builder_cls: type[B12xMLASparseMetadataBuilder],
    *,
    seq_lens: list[int],
    query_lens: list[int],
    is_prefilling: list[bool],
) -> B12xMLASparseMetadata:
    builder = object.__new__(builder_cls)
    builder.metadata_cls = B12xMLASparseMetadata
    builder.require_uniform_decodes = False
    builder.use_pcp = False
    builder.reorder_batch_threshold = 128
    builder._prefill_backend = None
    builder.topk_tokens = 2048
    builder.cp_kv_cache_interleave_size = 1
    builder.kv_cache_spec = SimpleNamespace(block_size=64)
    builder.model_config = SimpleNamespace(dtype=torch.bfloat16)
    rows = sum(query_lens)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    request_ids = torch.repeat_interleave(
        torch.arange(len(query_lens), dtype=torch.int32),
        torch.tensor(query_lens),
    )
    builder._build_req_id_per_token = lambda common: request_ids
    positions = torch.cat(
        [torch.arange(length, dtype=torch.int64) for length in query_lens]
    )
    common = SimpleNamespace(
        num_reqs=len(seq_lens),
        num_actual_tokens=rows,
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table_tensor=torch.arange(len(seq_lens), dtype=torch.int32).view(-1, 1),
        slot_mapping=torch.arange(rows, dtype=torch.int64),
        positions=positions,
        is_prefilling=torch.tensor(is_prefilling),
    )
    return SparseMLACommonMetadataBuilder.build(builder, 0, common)


def test_glm_short_packed_prefills_do_not_use_selector_decode_transactions() -> None:
    fresh = _build_short_packed_metadata(
        B12xGLM5NextMLASparseMetadataBuilder,
        seq_lens=[2, 3],
        query_lens=[2, 3],
        is_prefilling=[True, True],
    )
    assert fresh.num_decodes == 0
    assert fresh.num_prefills == 2
    assert fresh.num_decode_tokens == 0
    assert fresh.req_id_per_token.tolist() == [0, 0, 1, 1, 1]
    assert fresh.query_start_loc.tolist() == [0, 2, 5]

    mixed = _build_short_packed_metadata(
        B12xGLM5NextMLASparseMetadataBuilder,
        seq_lens=[4, 2],
        query_lens=[1, 2],
        is_prefilling=[False, True],
    )
    assert mixed.num_decodes == 1
    assert mixed.num_prefills == 1
    assert mixed.num_decode_tokens == 1
    assert mixed.req_id_per_token.tolist() == [0, 1, 1]
    assert mixed.query_start_loc.tolist() == [0, 1, 3]

    dsv4 = _build_short_packed_metadata(
        B12xMLASparseMetadataBuilder,
        seq_lens=[2, 3],
        query_lens=[2, 3],
        is_prefilling=[True, True],
    )
    assert dsv4.num_decodes == 2
    assert dsv4.num_prefills == 0
    assert dsv4.num_decode_tokens == 5


def test_glm_selector_metadata_builder_stages_padded_rows_and_capture(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=0,
            num_decode_tokens=0,
        ),
    )
    builder = _bare_glm_selector_metadata_builder()
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=4,
        max_query_len=1,
        seq_lens=torch.tensor([8, 9, 0, 0], dtype=torch.int32),
        dcp_local_seq_lens=None,
    )

    captured = builder.build_for_cudagraph_capture(common)
    pointers = tuple(
        tensor.data_ptr()
        for tensor in (
            captured.selector_state_slot_ids,
            captured.selector_state_is_fresh,
            captured.selector_num_accepted_tokens,
            captured.selector_is_prefilling,
        )
    )
    assert torch.equal(
        captured.selector_state_slot_ids,
        torch.arange(4, dtype=torch.int32),
    )
    assert captured.selector_state_is_fresh.all()
    assert torch.equal(
        captured.selector_num_accepted_tokens,
        torch.ones(4, dtype=torch.int32),
    )
    assert not captured.selector_is_prefilling.any()

    runtime = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        selector_state_slot_ids=torch.tensor([7, 3, -1, -1], dtype=torch.int32),
        selector_state_is_fresh=torch.tensor([False, True, True, True]),
        selector_num_accepted_tokens=torch.tensor([4, 2, 1, 1], dtype=torch.int32),
        selector_is_prefilling=torch.tensor([False, True, False, False]),
    )
    assert (
        tuple(
            tensor.data_ptr()
            for tensor in (
                runtime.selector_state_slot_ids,
                runtime.selector_state_is_fresh,
                runtime.selector_num_accepted_tokens,
                runtime.selector_is_prefilling,
            )
        )
        == pointers
    )
    assert torch.equal(
        runtime.selector_state_slot_ids,
        torch.tensor([7, 3, -1, -1], dtype=torch.int32),
    )
    assert torch.equal(
        runtime.selector_state_is_fresh,
        torch.tensor([False, True, True, True]),
    )
    assert torch.equal(
        runtime.selector_num_accepted_tokens,
        torch.tensor([4, 2, 1, 1], dtype=torch.int32),
    )
    assert torch.equal(
        runtime.selector_is_prefilling,
        torch.tensor([False, True, False, False]),
    )


def test_glm_selector_metadata_builder_requires_complete_runtime_state(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=0,
            num_decode_tokens=0,
        ),
    )
    builder = _bare_glm_selector_metadata_builder()
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=1,
        max_query_len=1,
        seq_lens=torch.ones(1, dtype=torch.int32),
        dcp_local_seq_lens=None,
    )

    with pytest.raises(RuntimeError, match="requires selector state slots"):
        builder.build(common_prefix_len=0, common_attn_metadata=common)


def test_glm_selector_metadata_builder_updates_draft_acceptance() -> None:
    builder = _bare_glm_selector_metadata_builder()
    accepted = torch.tensor([4, 2, 1, 1], dtype=torch.int32)
    metadata = SimpleNamespace(selector_num_accepted_tokens=accepted)

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(accepted, torch.ones(4, dtype=torch.int32))


def test_dsv4_metadata_builder_does_not_claim_glm_selector_state() -> None:
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False

    assert builder._stage_glm_next_selector_metadata(
        num_reqs=2,
        for_cudagraph_capture=False,
        selector_state_slot_ids=None,
        selector_state_is_fresh=None,
        selector_num_accepted_tokens=None,
        selector_is_prefilling=None,
    ) == (None, None, None, None)
    with pytest.raises(TypeError, match="non-GLM"):
        builder._stage_glm_next_selector_metadata(
            num_reqs=2,
            for_cudagraph_capture=False,
            selector_state_slot_ids=torch.arange(2, dtype=torch.int32),
            selector_state_is_fresh=None,
            selector_num_accepted_tokens=None,
            selector_is_prefilling=None,
        )


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


def test_b12x_non_compressed_indexer_exposes_scores_for_dcp(monkeypatch) -> None:
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
    monkeypatch.setattr(generic_b12x_indexer, "_require_b12x_indexer", lambda: module)
    monkeypatch.setattr(
        generic_b12x_indexer,
        "current_workspace_manager",
        lambda: _Workspace(),
    )

    output = torch.empty((2, 4), dtype=torch.int32)
    scores = torch.empty((2, 4), dtype=torch.float32)
    generic_b12x_indexer._run_paged_topk(
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


def test_b12x_compressed_sparse_mla_uses_public_plan_bind_run(
    monkeypatch,
) -> None:
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


def test_b12x_wo_projection_packs_and_runs_public_api(monkeypatch) -> None:
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


def test_b12x_mhc_uses_public_plan_bind_run(monkeypatch) -> None:
    calls: dict[str, Any] = {}
    retained_bindings: list[Any] = []

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
    monkeypatch.setattr(
        b12x_mla,
        "retain_cuda_graph_capture_resource",
        retained_bindings.append,
    )

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
    assert retained_bindings == [
        calls["pre"][1]["binding"],
        calls["post_pre"][1]["binding"],
    ]
    assert residual_out.shape == (3, 4, 256)
    assert layer_input.shape == (3, 256)
    assert final is next_outputs[0]


def test_b12x_dsa_indexer_uses_logical_slot_contract(monkeypatch) -> None:
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
        calls["output_before_run"] = kwargs["out_indices"].clone()
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

    output = torch.full((3, 4), 37, dtype=torch.int32)
    scores = torch.empty((3, 4), dtype=torch.float32)
    b12x_indexer._run_paged_topk(
        module=module,
        plan=plan,
        q=torch.empty((3, 16, 128), dtype=torch.float8_e4m3fn),
        weights=torch.empty((3, 16, 1), dtype=torch.float32),
        kv_cache=torch.empty((4, 64, 132), dtype=torch.uint8),
        seq_lens=torch.full((3,), 128, dtype=torch.int32),
        block_table=torch.zeros((3, 2), dtype=torch.int32),
        schedule_metadata=None,
        output=output,
        scores=scores,
        topk=4,
        shared_page_table=True,
    )

    assert calls["run"]["out_scores"] is scores
    assert torch.count_nonzero(calls["output_before_run"] != 37) == 0
    assert "active_width" not in calls["bind"]

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
    assert calls["caps"]["source_layout"] == "paged"
    assert calls["caps"]["max_page_table_width"] == 1024


def test_b12x_dsa_indexer_reuses_plans_and_rebinds_shared_workspace(
    monkeypatch,
) -> None:
    calls = {"plan": 0, "workspace": 0, "bind": 0, "run": 0}

    def bind(**kwargs):
        calls["bind"] += 1
        return SimpleNamespace(route="packed_contiguous")

    plan = SimpleNamespace(
        route="packed_contiguous",
        shapes_and_dtypes=lambda: (((64,), torch.uint8),),
        bind=bind,
    )

    def make_plan(_caps):
        calls["plan"] += 1
        return plan

    def index_topk_fp8(**kwargs):
        calls["run"] += 1
        kwargs["out_indices"].fill_(7)

    module = SimpleNamespace(
        Caps=lambda **kwargs: SimpleNamespace(**kwargs),
        SOURCE_LAYOUT_PAGED="paged",
        PAGED_INDEX_PAGE_SIZE=64,
        plan=make_plan,
        index_topk_fp8=index_topk_fp8,
    )

    class Workspace:
        def get_simultaneous(self, *shapes_and_dtypes):
            calls["workspace"] += 1
            return [
                torch.empty(shape, dtype=dtype) for shape, dtype in shapes_and_dtypes
            ]

    workspace = Workspace()
    monkeypatch.setattr(b12x_indexer, "_require_b12x_indexer", lambda: module)
    monkeypatch.setattr(
        b12x_indexer,
        "current_workspace_manager",
        lambda: workspace,
    )

    indexer = b12x_indexer.B12xC4SparseIndexer(
        SimpleNamespace(),
        quant_block_size=128,
        scale_fmt="ue8m0",
        topk_tokens=4,
        head_dim=128,
        max_model_len=128,
        max_total_seq_len=128,
        topk_indices_buffer=torch.empty((3, 4), dtype=torch.int32),
        skip_k_cache_insert=True,
        compress_ratio=4,
    )
    inputs = {
        "q": torch.empty((3, 16, 128), dtype=torch.float8_e4m3fn),
        "weights": torch.empty((3, 16, 1), dtype=torch.float32),
        "kv_cache": torch.empty((4, 64, 132), dtype=torch.uint8),
        "seq_lens": torch.full((3,), 128, dtype=torch.int32),
        "block_table": torch.zeros((3, 2), dtype=torch.int32),
        "output": torch.empty((3, 4), dtype=torch.int32),
        "shared_page_table": True,
    }

    indexer.run_paged_topk(**inputs)
    indexer.run_paged_topk(**inputs)

    assert calls == {"plan": 1, "workspace": 2, "bind": 2, "run": 2}
    assert torch.count_nonzero(inputs["output"] != 7) == 0
