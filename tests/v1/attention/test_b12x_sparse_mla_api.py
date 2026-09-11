# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for the B12x sparse MLA adapters."""

import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention.mla_attention import (
    _canonicalize_sparse_mla_kv_cache_dtype,
    _maybe_view_mla_cache_as_fp8,
    _uses_packed_sparse_mla_workspace,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonMetadataBuilder,
)
from vllm.model_executor.models.config import DeepseekV41ForCausalLMConfig
from vllm.models.deepseek_v4.nvidia import b12x as b12x_mla
from vllm.models.deepseek_v4.nvidia import b12x_indexer
from vllm.models.deepseek_v4_1.nvidia.b12x_attention import (
    DeepseekV41B12xAttention,
)
from vllm.models.deepseek_v4_1.nvidia.model import _select_dsv4_attn_cls
from vllm.models.deepseek_v4_1.sparse_mla import DeepseekV41B12xBackend
from vllm.models.deepseek_v32.nvidia.b12x import (
    B12xDSAIndexer,
    DeepseekV32B12xAttention,
    _get_sparse_mla_backend,
)
from vllm.models.deepseek_v32.nvidia.model import _get_attention_cls
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
    _ckv_rank_token_alignment,
    _global_causal_lens_for_ckv_gather,
    _is_glm_next_ckv_source_layout,
    _is_speculative_decode_batch,
    _max_speculative_decode_query_len,
    _round_up_ckv_rank_tokens,
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
    assert _get_attention_cls(config) is DeepseekV32B12xAttention
    assert DeepseekV32B12xAttention.indexer_cls is B12xDSAIndexer

    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")
    )
    assert _get_sparse_mla_backend(config) is B12xGLMDSAMLASparseBackend


def test_b12x_selector_routes_deepseek_v41() -> None:
    config = SimpleNamespace(attention_config=SimpleNamespace(backend=None))

    DeepseekV41ForCausalLMConfig.verify_and_update_config(config)

    assert config.attention_config.backend is AttentionBackendEnum.B12X
    assert _select_dsv4_attn_cls(config) is DeepseekV41B12xAttention
    assert DeepseekV41B12xBackend.get_name() == "B12X"

    config.attention_config.backend = AttentionBackendEnum.FLASH_ATTN
    with pytest.raises(ValueError, match="requires B12X"):
        DeepseekV41ForCausalLMConfig.verify_and_update_config(config)


def test_deepseek_v41_tp3_padding_uses_generic_parallel_hook() -> None:
    text_config = SimpleNamespace(num_attention_heads=64, o_groups=8)
    model_config = SimpleNamespace(
        architecture="DSparkV41DraftModel",
        hf_config=text_config,
        hf_text_config=text_config,
        model_arch_config=SimpleNamespace(total_num_attention_heads=64),
    )
    model_config.get_model_arch_config = lambda: SimpleNamespace(
        total_num_attention_heads=text_config.num_attention_heads
    )
    parallel_config = SimpleNamespace(tensor_parallel_size=3)

    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)
    ModelConfig._update_model_config_for_parallelism(model_config, parallel_config)

    assert text_config.original_num_attention_heads == 64
    assert text_config.original_o_groups == 8
    assert text_config.num_attention_heads == 72
    assert text_config.o_groups == 9
    assert model_config.model_arch_config.total_num_attention_heads == 72


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
    impl._uses_glm_dsa_nvfp4_cache = False
    impl._kernel_page_size_finalized = False
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


@pytest.mark.parametrize(
    (
        "dcp",
        "retention_interval",
        "scheduled_tokens",
        "batched_tokens",
        "expected_block_size",
    ),
    [
        (1, None, None, 4096, 4096),
        (2, None, None, 4096, 2048),
        (4, None, None, 4096, 1024),
        (8, None, None, 4096, 512),
        (4, 0, None, 4096, 1024),
        (4, 0, 4096, 4352, 1024),
    ],
)
def test_glm5_next_split_cache_auto_falls_back_to_scheduler_budget(
    monkeypatch,
    dcp: int,
    retention_interval: int | None,
    scheduled_tokens: int | None,
    batched_tokens: int,
    expected_block_size: int,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Glm5NextForConditionalGeneration",
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        scheduler_config=SimpleNamespace(
            max_num_scheduled_tokens=scheduled_tokens,
            max_num_batched_tokens=batched_tokens,
        ),
        cache_config=SimpleNamespace(
            block_size=256,
            mamba_block_size=None,
            mamba_cache_mode="align",
            mamba_page_size_padded=1234,
            prefix_cache_retention_interval=retention_interval,
        ),
    )
    monkeypatch.setenv("VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE", "auto")
    monkeypatch.setenv("VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE", "auto")

    Platform._align_hybrid_block_size(config, B12xGLM5NextMLASparseBackend)

    assert config.cache_config.block_size == expected_block_size
    assert config.cache_config.mamba_block_size == expected_block_size
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
    impl._scratch_nbytes = 16
    impl._ckv_local_capacity = 128
    impl.dcp_world_size = 4
    impl._cache_record_bytes = record_bytes
    specs = impl._workspace_specs(input_num_heads=8, include_ckv=True)

    assert len(specs) == 3
    assert specs[-1] == ((512, record_bytes), torch.uint8)


@pytest.mark.parametrize("record_bytes", [528, 304])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("padded_tokens", [2, 4])
def test_b12x_glm5_next_full_ckv_gather_preserves_native_records(
    monkeypatch: pytest.MonkeyPatch,
    record_bytes: int,
    rank: int,
    padded_tokens: int,
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
    gathered_buffer = torch.full((8, record_bytes), 255, dtype=torch.uint8)
    local_tokens = padded_tokens - 1
    metadata = SimpleNamespace(
        num_actual_tokens=local_tokens,
        dcp_local_total_tokens=local_tokens,
        dcp_padded_total_tokens=padded_tokens,
        dcp_local_cu_seq_lens=torch.tensor([0, local_tokens], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        num_reqs=1,
    )

    def fake_cp_gather_cache(**kwargs: Any) -> None:
        kwargs["dst"].copy_(kwargs["src_cache"].view(-1, record_bytes)[:local_tokens])

    def fake_all_gather(_group: Any, src: torch.Tensor, dst: torch.Tensor) -> None:
        assert src.data_ptr() == dst.data_ptr() + rank * padded_tokens * record_bytes
        dst.copy_(src.repeat(2))

    monkeypatch.setattr(b12x_mla_sparse.ops, "cp_gather_cache", fake_cp_gather_cache)
    monkeypatch.setattr(
        b12x_mla_sparse, "_dcp_all_gather_current_stream", fake_all_gather
    )
    monkeypatch.setattr(
        b12x_mla_sparse,
        "get_dcp_group",
        lambda: SimpleNamespace(rank_in_group=rank, world_size=2),
    )

    gathered = impl._gather_full_ckv(kv_cache, metadata, gathered_buffer)

    expected_rank = torch.cat(
        (
            kv_cache.view(-1, record_bytes)[:local_tokens],
            torch.zeros((1, record_bytes), dtype=torch.uint8),
        ),
        dim=0,
    )
    assert gathered.shape == (4, 2, record_bytes)
    assert torch.equal(
        gathered.view(-1, record_bytes)[: 2 * padded_tokens], expected_rank.repeat(2, 1)
    )
    assert torch.all(gathered.view(-1, record_bytes)[2 * padded_tokens :] == 255)


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
            torch.empty((8, 304), dtype=torch.uint8),
        )


@pytest.mark.parametrize(
    ("page_size", "dcp_world_size", "alignment"),
    [(2048, 4, 512), (2048, 2, 1024), (512, 4, 128), (512, 3, 512)],
)
def test_full_ckv_rank_alignment_only_pads_the_concatenated_cache_to_pages(
    page_size: int,
    dcp_world_size: int,
    alignment: int,
) -> None:
    assert _ckv_rank_token_alignment(page_size, dcp_world_size) == alignment
    padded = _round_up_ckv_rank_tokens(
        1025,
        page_size=page_size,
        dcp_world_size=dcp_world_size,
    )
    assert padded >= 1025
    assert padded % alignment == 0
    assert padded * dcp_world_size % page_size == 0


@pytest.mark.parametrize("record_bytes", [528, 656])
def test_b12x_selected_indices_use_physical_slots(record_bytes: int) -> None:
    storage = torch.empty((2, 2, 64, record_bytes), dtype=torch.uint8)
    cache = storage[:, 0]

    assert cache.stride(0) // record_bytes == 128
    assert _selected_index_block_stride_rows(cache, block_size=64) == 64


def test_sparse_index_remap_tiling_covers_glm5_next_width() -> None:
    assert _remap_tiling(2048, 128, True) == (True, 2048, 1, 8)
    assert _remap_tiling(2051, 128, True) == (True, 4096, 1, 8)
    assert _remap_tiling(384, 128, True) == (False, 128, 3, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA metadata kernel")
@pytest.mark.parametrize("gathered", [False, True])
@pytest.mark.parametrize("dcp_rank", range(4))
def test_glm_dcp_compaction_preserves_tail_order_on_graph_replay(
    gathered: bool, dcp_rank: int
) -> None:
    """A C4 tail cannot race the history into a different attention order."""
    from vllm.v1.attention.backends.mla.sparse_utils import (
        triton_filter_and_convert_dcp_index,
    )

    rows, width, page_size = 4, 2051, 2048
    # Remapping only reads metadata. This physical page would lie above 2 GiB
    # in a 528-byte packed-record pool; no KV payload is allocated here.
    page_id = 4097
    req_ids = torch.zeros(rows, dtype=torch.int32, device="cuda")
    block_table = torch.full((1, 1), page_id, dtype=torch.int32, device="cuda")
    starts = torch.zeros((4, 1), dtype=torch.int32, device="cuda")
    lengths = torch.full_like(starts, page_size)
    indices = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    out, counts = torch.empty_like(indices), torch.empty_like(req_ids)

    def run():
        if gathered:
            b12x_mla_sparse._map_global_topk_to_gathered_ckv(
                req_ids,
                indices,
                starts,
                lengths,
                out,
                counts,
                dcp_size=4,
                cp_kv_cache_interleave_size=4,
                padded_rank_tokens=page_size,
            )
            return out, counts
        return triton_filter_and_convert_dcp_index(
            req_ids,
            block_table,
            indices,
            dcp_size=4,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=4,
            BLOCK_SIZE=page_size,
            NUM_TOPK_TOKENS=width,
            return_valid_counts=True,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result, valid_counts = run()

    for tail_count in (3, 0, 1):
        source = torch.full((rows, width), -1, dtype=torch.int32)
        source[1, 2048 : 2048 + tail_count] = torch.arange(tail_count)
        source[2, :24] = torch.arange(24)
        source[2, 2048 : 2048 + tail_count] = torch.arange(24, 24 + tail_count)
        source[3] = torch.arange(width)
        expected = torch.full_like(source, -1)
        expected_counts = torch.zeros(rows, dtype=torch.int32)
        for row in range(rows):
            token = source[row].to(torch.int64)
            owner = token // 4 % 4
            local = token // 16 * 4 + token % 4
            valid = (token >= 0) & (local < page_size)
            if gathered:
                slots = owner * page_size + local
            else:
                valid &= owner == dcp_rank
                slots = page_id * page_size + local
            selected = slots[valid].to(torch.int32)
            expected_counts[row] = selected.numel()
            expected[row, : selected.numel()] = selected
        indices.copy_(source)
        for _ in range(4):
            result.fill_(123456)
            valid_counts.fill_(-123456)
            graph.replay()
            assert torch.equal(result.cpu(), expected)
            assert torch.equal(valid_counts.cpu(), expected_counts)


def test_b12x_glm_dsa_nvfp4_cache_writer_keeps_rope() -> None:
    calls: list[tuple[torch.Tensor, ...]] = []
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._uses_glm_dsa_nvfp4_cache = True
    impl._concat_and_cache_nvfp4_mla_fp8_rope = lambda *args: calls.append(args)
    kv_c = torch.zeros((3, 512), dtype=torch.bfloat16)
    k_pe = torch.zeros((3, 1, 64), dtype=torch.bfloat16)
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


def test_b12x_glm5_next_full_ckv_bind_requires_geometry_finalization() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = True
    impl._uses_glm_dsa_nvfp4_cache = False
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

    assert impl._use_decode_execution(metadata, num_tokens) is expected_decode


@pytest.mark.parametrize("selection_dcp_size", [None, 1, 4])
def test_b12x_sparse_mla_declares_plans_once_and_prepares_in_place(
    monkeypatch,
    selection_dcp_size,
) -> None:
    """A layer's plans are declared once; preparation fills them in place."""
    from vllm.utils.b12x import B12xWorkload

    class _FakePlan:
        def __init__(self) -> None:
            self.prepared = None

        def request(self, *, name, prepare_call, benchmark_call=None, **_kwargs):
            return SimpleNamespace(name=name, plan=self)

    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._decode_max_rows = 4
    impl._plans = {("decode", 4): _FakePlan(), ("extend", 8): _FakePlan()}
    impl._plan_caps = {("decode", 4): object(), ("extend", 8): object()}
    impl._bound_kv_cache = torch.empty((2, 64, 656), dtype=torch.uint8)
    impl._preparation_prefix = lambda: "test.sparse-mla"
    impl.dcp_world_size = selection_dcp_size or 1
    selection_plan = _FakePlan()
    impl._physical_selection_provider = (
        SimpleNamespace(
            get_b12x_physical_selection_preparation_request=lambda: SimpleNamespace(
                name="test.physical_selection", plan=selection_plan
            )
        )
        if selection_dcp_size is not None
        else None
    )
    declared = []

    def declare(mode, rows):
        declared.append((mode, rows))
        return object(), _FakePlan()

    monkeypatch.setattr(impl, "_declare_plan", declare)
    workload = B12xWorkload(
        stage="state",
        token_counts=(2, 4, 8),
        fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=8,
        max_seqs=1,
        max_model_len=8,
    )

    (unit,) = impl.get_b12x_preparation_units(impl, workload)
    first = {request.name: request.plan for request in unit.requests}
    # Decode plans are exact in rows: every planned count that can route to
    # decode is declared; a count above the decode capacity is not.
    assert declared == [("decode", 2)]
    expected = {
        "test.sparse-mla.decode.m2": impl._plans[("decode", 2)],
        "test.sparse-mla.decode.m4": impl._plans[("decode", 4)],
        "test.sparse-mla.extend.m8": impl._plans[("extend", 8)],
    }
    if selection_dcp_size == 1:
        expected["test.physical_selection"] = selection_plan
    assert first == expected
    assert all(plan.prepared is None for plan in impl._plans.values())

    # A second declaration reuses the exact same plan objects: plans are
    # declared once and never rebuilt by preparation itself.
    (unit_again,) = impl.get_b12x_preparation_units(impl, workload)
    second = {request.name: request.plan for request in unit_again.requests}
    assert second == first
    assert declared == [("decode", 2)]

    for plan in impl._plans.values():
        plan.prepared = object()
    assert all(plan.prepared is not None for plan in impl._plans.values())


@pytest.mark.parametrize(
    ("kernel_returns_lse", "consumer_needs_lse"),
    [(False, False), (True, False), (True, True)],
)
def test_sparse_mla_forward_normalizes_decode_and_extend_results(
    monkeypatch, kernel_returns_lse, consumer_needs_lse
):
    impl = object.__new__(B12xMLASparseImpl)
    impl._is_glm_next = False
    impl._input_num_heads = 2
    impl._plan_key = lambda metadata, tokens: ("extend", 4)
    impl._plan = lambda key: object()
    impl._workspace_specs = lambda **kwargs: (
        ((4, 2, 8), torch.bfloat16),
        ((512,), torch.uint8),
    )
    impl.topk_indices_buffer = torch.zeros((4, 8), dtype=torch.int32)
    impl._physical_selection_provider = None
    impl.dcp_world_size = 1
    impl.need_to_return_lse_for_decode = consumer_needs_lse
    impl._bind = lambda plan, **kwargs: kwargs
    output = torch.ones((4, 2, 8), dtype=torch.bfloat16)
    lse = torch.zeros((4, 2), dtype=torch.float32)
    impl._run = lambda binding: (output, lse) if kernel_returns_lse else output
    monkeypatch.setattr(
        b12x_mla_sparse, "current_workspace_manager", lambda: _Workspace()
    )
    metadata = SimpleNamespace(
        block_size=64,
        seq_lens=torch.ones(4, dtype=torch.int32),
        cache_seq_lens_per_token=torch.ones(4, dtype=torch.int32),
        num_reqs=4,
    )
    actual, actual_lse = impl.forward_mqa(
        torch.ones((4, 2, 8), dtype=torch.bfloat16),
        torch.empty((1, 64, 528), dtype=torch.uint8),
        metadata,
        None,
    )
    assert actual is output
    assert actual_lse is (lse if consumer_needs_lse else None)


def test_sparse_mla_preparation_borrows_declared_scratch_shape(monkeypatch):
    from b12x._lib.scratch import scratch_buffer_spec

    impl = object.__new__(B12xMLASparseImpl)
    impl._bound_kv_cache = torch.zeros((2, 64, 528), dtype=torch.uint8)
    caps = SimpleNamespace(
        max_q_rows=4,
        num_q_heads=2,
        head_dim=8,
        max_width=16,
        max_batch=2,
        page_size=64,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    spec = scratch_buffer_spec("workspace", nbytes=512, device=caps.device)
    impl._scratch_nbytes = spec.nbytes
    borrowed = torch.empty(spec.shape, dtype=spec.dtype)
    reservations = []

    def get_simultaneous(*specs):
        assert specs == ((spec.shape, spec.dtype),)
        return (borrowed,)

    monkeypatch.setattr(
        "vllm.v1.attention.backends.mla.b12x_mla_sparse.current_workspace_manager",
        lambda: SimpleNamespace(
            reserve_all=lambda *specs: reservations.append(specs),
            get_simultaneous=get_simultaneous,
        ),
    )
    bound = {}
    calls = []

    def bind(**kwargs):
        bound.update(kwargs)
        assert kwargs["scratch"].shape == spec.shape
        assert kwargs["scratch"].dtype == spec.dtype
        assert kwargs["scratch"] is borrowed
        return kwargs

    state = SimpleNamespace(
        scratch_specs=lambda: (spec,),
        bind=bind,
        prime=lambda binding, **kwargs: calls.append("prime"),
        run=lambda binding, **kwargs: calls.append("run"),
    )
    call = impl._make_prepare_call(state, caps)
    call.produce()
    call.run()
    assert calls == ["prime", "run"]
    assert torch.all(bound["q"] == 1)
    assert bound["kv_cache"] is impl._bound_kv_cache
    second_call = impl._make_prepare_call(state, caps)
    second_call.produce()
    second_call.run()
    assert calls == ["prime", "run", "prime", "run"]
    assert reservations == [(((spec.nbytes,), torch.uint8),)] * 2


@pytest.mark.parametrize("query_projection", [False, True])
def test_mla_layer_discovers_backend_and_optional_query_plans(
    monkeypatch, query_projection
):
    from vllm.model_executor.layers.attention import mla_attention
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.utils.b12x import (
        B12xPreparationUnit,
        B12xWorkload,
        set_b12x_preparation_provider,
    )

    unit = B12xPreparationUnit(
        name="sparse MLA", key="cache-owner", requests=(), stage="state"
    )

    class Backend:
        def get_b12x_preparation_units(self, owner, workload):
            assert owner is self
            assert workload.stage == "state"
            return (unit,)

    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.impl = Backend()
    set_b12x_preparation_provider(layer.impl, layer.impl)
    set_b12x_preparation_provider(layer, layer)
    if query_projection:
        layer.W_UK_T = torch.empty((2, 16, 32), dtype=torch.bfloat16)
        layer._b12x_query_plans = {}
        layer._b12x_query_prefix = "test.query"
        layer._b12x_query_call = lambda tokens: None
        layer._declare_b12x_query_plan = lambda tokens: SimpleNamespace(
            request=lambda name, **kwargs: SimpleNamespace(name=name)
        )
        monkeypatch.setattr(
            mla_attention, "can_implement_bf16_mla_query", lambda **kwargs: True
        )
    workload = B12xWorkload(
        stage="state",
        token_counts=(8,),
        fixed_token_counts=(),
        output_dtype=torch.bfloat16,
        max_tokens=8,
        max_seqs=1,
        max_model_len=64,
    )
    units = tuple(_units_from_modules(layer, workload))
    assert units[0] is unit
    assert len(units) == (2 if query_projection else 1)
    if query_projection:
        assert units[1].name == "MLA_QUERY"
        assert [request.name for request in units[1].requests] == ["test.query.m8"]


def test_b12x_sparse_mla_plan_lookup_declares_unplanned_decode_rows_once(
    monkeypatch,
) -> None:
    """A planned key is served as declared; an unplanned decode row count is
    declared once and materializes its default on first use, never prepared."""
    import b12x.preparation as preparation

    class _FakePlan:
        def request(self, *, name, prepare_call, **_kwargs):
            return SimpleNamespace(name=name, plan=self, prepare_call=prepare_call)

    impl = object.__new__(B12xMLASparseImpl)
    impl._preparation_prefix = lambda: "test.sparse-mla"
    planned = _FakePlan()
    impl._plans = {("decode", 8): planned}
    impl._plan_caps = {("decode", 8): object()}
    declared = []
    prepared = []
    prepare_calls = []

    def declare(mode, rows):
        declared.append((mode, rows))
        return f"caps:{mode}:{rows}", _FakePlan()

    def make_prepare_call(state, caps):
        prepare_calls.append((state, caps))
        return "layer-call"

    monkeypatch.setattr(impl, "_declare_plan", declare)
    monkeypatch.setattr(impl, "_make_prepare_call", make_prepare_call)
    monkeypatch.setattr(
        preparation, "prepare_default", lambda request: prepared.append(request)
    )

    assert impl._plan(("decode", 8)) is planned
    assert declared == [] and prepared == []

    plan = impl._plan(("decode", 3))
    assert declared == [("decode", 3)]
    assert prepared == [] and prepare_calls == []
    assert impl._plans[("decode", 3)] is plan
    assert impl._plan_caps[("decode", 3)] == "caps:decode:3"

    assert impl._plan(("decode", 3)) is plan
    assert declared == [("decode", 3)]
    assert prepared == []


def test_b12x_sparse_mla_plan_key_is_exact_for_decode_rows() -> None:
    impl = object.__new__(B12xMLASparseImpl)
    impl._max_speculative_decode_query_len = 1
    impl._decode_max_rows = 24
    impl._max_tokens = 128
    impl.uses_full_ckv_dcp = lambda attn_metadata, num_tokens: False
    decode = SimpleNamespace(num_reqs=11, max_query_len=1, is_spec_decode=False)
    prefill = SimpleNamespace(num_reqs=1, max_query_len=37, is_spec_decode=False)

    assert impl._plan_key(decode, 11) == ("decode", 11)
    assert impl._plan_key(prefill, 37) == ("extend", 128)


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
        max_logits_per_req=None,
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


def test_b12x_sparse_mla_spec_decode_lengths_stay_in_builder_buffer(
    monkeypatch,
) -> None:
    """Multi-row decode lengths must live in the builder buffer.

    A FULL CUDA graph binds the tensor address at capture and replays against
    whatever a later build wrote there, so a fresh tensor per build leaves the
    replayed kernel reading stale lengths.
    """
    monkeypatch.setattr(
        SparseMLACommonMetadataBuilder,
        "build",
        lambda *args, **kwargs: SimpleNamespace(
            num_prefills=0,
            num_decodes=2,
            num_decode_tokens=8,
        ),
    )
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False
    builder._ckv_gather_requested = False
    builder.dcp_world_size = 1
    builder._max_speculative_decode_query_len = 4
    builder.cache_seq_lens_per_token_buffer = torch.zeros(16, dtype=torch.int32)
    positions = torch.tensor([28, 29, 30, 31, 36, 37, 38, 39], dtype=torch.int64)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=8,
        max_query_len=4,
        seq_lens=torch.tensor([32, 40], dtype=torch.int32),
        dcp_local_seq_lens=None,
        positions=positions,
        is_prefilling=torch.zeros(2, dtype=torch.bool),
    )

    first = builder.build(common_prefix_len=0, common_attn_metadata=common)
    lengths = first.cache_seq_lens_per_token
    assert first.is_spec_decode
    assert lengths.data_ptr() == builder.cache_seq_lens_per_token_buffer.data_ptr()
    assert lengths.tolist() == [29, 30, 31, 32, 37, 38, 39, 40]

    common.positions = positions + 8
    second = builder.build(common_prefix_len=0, common_attn_metadata=common)
    assert second.cache_seq_lens_per_token.data_ptr() == lengths.data_ptr()
    assert lengths.tolist() == [37, 38, 39, 40, 45, 46, 47, 48]


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


def test_dsa_builder_refreshes_fused_dcp_lengths(monkeypatch) -> None:
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False
    builder.supports_draft_decode_metadata_update = True
    builder.dcp_world_size = 4
    builder.dcp_rank = 2
    builder.cp_kv_cache_interleave_size = 1
    global_seq_lens = torch.tensor([17, 9], dtype=torch.int32)
    local_seq_lens = torch.zeros(2, dtype=torch.int32)
    calls = []

    def refresh(*args) -> None:
        calls.append(args)

    monkeypatch.setattr(b12x_mla_sparse, "refresh_dcp_local_seq_lens_", refresh)
    metadata = SimpleNamespace(
        dcp_global_seq_lens=global_seq_lens,
        seq_lens=local_seq_lens,
        num_reqs=2,
        selector_num_accepted_tokens=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert calls == [
        (local_seq_lens, global_seq_lens, 2, 4, 2, 1),
    ]


def test_dsa_builder_rejects_missing_fused_dcp_lengths() -> None:
    builder = B12xMLASparseMetadataBuilder.__new__(B12xMLASparseMetadataBuilder)
    builder.requires_glm_next_selector_metadata = False
    builder.supports_draft_decode_metadata_update = True
    builder.dcp_world_size = 4
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    metadata = SimpleNamespace(
        dcp_global_seq_lens=None,
        seq_lens=torch.zeros(1, dtype=torch.int32),
        num_reqs=1,
        selector_num_accepted_tokens=None,
    )

    with pytest.raises(RuntimeError, match="global sequence lengths"):
        builder.update_draft_decode_metadata(metadata)


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
    plan = object()

    def bind(bound_plan, **kwargs):
        calls["plan"] = bound_plan
        calls["bind"] = kwargs
        return SimpleNamespace(
            output=kwargs["output_indices"],
            scores=kwargs["output_scores"],
        )

    def run(binding):
        calls["run"] = binding
        binding.output.fill_(7)
        binding.scores.fill_(0.5)

    module = SimpleNamespace(
        bind=bind,
        run=run,
        scratch_specs=lambda bound_plan, *, device: (
            SimpleNamespace(shape=(64,), dtype=torch.uint8),
        ),
    )
    monkeypatch.setattr(
        generic_b12x_indexer,
        "current_workspace_manager",
        lambda: _Workspace(),
    )

    output = torch.empty((2, 4), dtype=torch.int32)
    scores = torch.empty((2, 4), dtype=torch.float32)
    generic_b12x_indexer._run_paged_topk(
        module=module,
        plan=plan,
        q=torch.empty((2, 32, 128), dtype=torch.float8_e4m3fn),
        weights=torch.empty((2, 32), dtype=torch.float32),
        kv_cache=torch.empty((4, 64, 132), dtype=torch.uint8),
        seq_lens=torch.full((2,), 128, dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        active_width=torch.full((1,), 128, dtype=torch.int32),
        output=output,
        scores=scores,
    )

    assert calls["plan"] is plan
    assert calls["bind"]["output_scores"] is scores
    assert calls["run"].scores is scores
    assert torch.count_nonzero(output != 7) == 0
    assert torch.count_nonzero(scores != 0.5) == 0


def test_b12x_dsa_indexer_uses_logical_slot_contract(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def bind(bound_plan, **kwargs):
        calls["bind_plan"] = bound_plan
        calls["bind"] = kwargs
        return SimpleNamespace(
            runtime=SimpleNamespace(route="packed_contiguous"),
            output=kwargs["output_indices"],
        )

    plan = object()

    def run(binding):
        calls["run"] = binding
        calls["output_before_run"] = binding.output.clone()
        binding.output.fill_(11)

    module = SimpleNamespace(
        PAGED_INDEX_PAGE_SIZE=64,
        bind=bind,
        run=run,
        scratch_specs=lambda bound_plan, *, device: (
            SimpleNamespace(shape=(64,), dtype=torch.uint8),
        ),
    )
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
        active_width=torch.full((1,), 128, dtype=torch.int32),
        output=output,
        scores=scores,
        shared_page_table=True,
    )

    assert calls["bind_plan"] is plan
    assert calls["bind"]["output_scores"] is scores

    builder = object.__new__(b12x_indexer.DeepseekV4B12xIndexerMetadataBuilder)
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


def _deepseek_v4_wo_layer(device, groups=2, heads_per_group=8, rank=128, hidden=256):
    layer = b12x_mla.DeepseekV4B12xAttention.__new__(b12x_mla.DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "test.deepseek_v4.wo"
    layer.indexer = None
    layer.n_local_groups = groups
    layer.n_local_heads = groups * heads_per_group
    layer.head_dim, layer.nope_head_dim, layer.rope_head_dim = 512, 448, 64
    layer.hidden_size, layer.o_lora_rank = hidden, rank
    layer._b12x_wo_plans = {}
    layer._b12x_wo_projection_weights = None
    layer.compress_ratio = 1
    layer.swa_cache_layer = SimpleNamespace(kv_cache=torch.empty(0))
    layer.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.empty(32, 64, dtype=torch.bfloat16, device=device)
    )
    group_width = heads_per_group * layer.head_dim
    layer.wo_a = SimpleNamespace(
        weight=torch.empty(
            groups * rank, group_width, dtype=torch.float8_e4m3fn, device=device
        ),
        weight_scale_inv=torch.ones(
            groups * (rank // 128), group_width // 128, device=device
        ),
    )
    layer.wo_b = SimpleNamespace(
        weight=torch.empty(
            hidden, groups * rank, dtype=torch.float8_e4m3fn, device=device
        ),
        weight_scale_inv=torch.ones(hidden // 128, groups * rank // 128, device=device),
        reduce_results=False,
        tp_size=1,
    )
    return layer


@pytest.mark.parametrize("eager_only", [False, True])
def test_deepseek_v4_wo_declares_exact_rows_before_profiling(monkeypatch, eager_only):
    from dataclasses import replace

    from b12x.gemm import wo_projection

    from vllm.utils.b12x import B12xWorkload

    layer = _deepseek_v4_wo_layer(torch.device("cpu"))
    layer._b12x_wo_projection_weights = SimpleNamespace(
        groups=2, group_width=4096, rank=128, hidden=256
    )
    monkeypatch.setattr(b12x_mla, "_require_b12x_wo_projection", lambda: wo_projection)
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 4, 8, 17),
        fixed_token_counts=(1, 4, 8),
        output_dtype=torch.bfloat16,
        max_tokens=17,
        max_seqs=4,
        max_model_len=32,
        eager_only=eager_only,
    )
    (unit,) = layer.get_b12x_preparation_units(layer, workload)
    assert unit.stage == "weights" and unit.autotune is True
    assert tuple(request.plan.query.max_tokens for request in unit.requests) == (
        1,
        4,
        8,
        17,
    )
    assert all(request.plan.prepared is None for request in unit.requests)
    for request in unit.requests:
        query = request.plan.query
        assert query.operation == "inv_rope" and not query.dynamic_tokens
        assert (query.heads_per_group, query.nope_dim, query.rope_dim) == (8, 448, 64)
        assert query.positions_dtype == "int64" and query.cos_sin_dtype == "bfloat16"
    (repeated,) = layer.get_b12x_preparation_units(layer, workload)
    assert all(a.plan is b.plan for a, b in zip(unit.requests, repeated.requests))
    assert (
        layer.get_b12x_preparation_units(layer, replace(workload, stage="state")) == ()
    )


def test_deepseek_v4_wo_fake_execution_does_not_materialize_native_plans():
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.utils.torch_utils import _encode_layer_name

    layer = _deepseek_v4_wo_layer(torch.device("cpu"))
    layer._b12x_wo_projection_weights = object()
    layer._b12x_wo_layer_name = _encode_layer_name(layer.prefix)
    with FakeTensorMode(allow_non_fake_inputs=True):
        source = torch.empty(17, 32, 512, dtype=torch.bfloat16)[:, :16]
        positions = torch.empty(17, dtype=torch.int64)
        output = layer._o_proj(source, positions)
    assert output.shape == (17, 256) and output.dtype == torch.bfloat16
    assert layer._b12x_wo_plans == {}


@pytest.mark.parametrize("groups,heads_per_group", [(1, 1), (2, 8)])
@torch.no_grad()
def test_deepseek_v4_wo_preparation_runs_and_replays_native_projection(
    groups,
    heads_per_group,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x WO projection requires SM12x")
    from b12x.gemm import wo_projection
    from b12x.preparation import PreparationSession

    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.utils.b12x import B12xWorkload, get_b12x_scratch_buffers
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        reset_workspace_manager,
    )

    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.manual_seed(419)
    layer = _deepseek_v4_wo_layer(device, groups, heads_per_group)
    for projection in (layer.wo_a, layer.wo_b):
        weight = torch.randn(projection.weight.shape, device=device)
        projection.weight.copy_(
            (weight / weight.shape[1] ** 0.5).to(projection.weight.dtype)
        )
    table = layer.rotary_emb.cos_sin_cache
    angles = torch.randn(32, 32, device=device)
    table.copy_(torch.cat((angles.cos(), angles.sin()), dim=-1))
    layer.setup_b12x_wo_projection()
    counts = (1, 4, 8, 17)
    workload = B12xWorkload(
        stage="weights",
        token_counts=counts,
        fixed_token_counts=counts[:-1],
        output_dtype=torch.bfloat16,
        max_tokens=counts[-1],
        max_seqs=4,
        max_model_len=32,
    )
    allocated = torch.accelerator.memory_allocated(device)
    units = tuple(_units_from_modules(layer, workload))
    assert torch.accelerator.memory_allocated(device) == allocated
    assert len(units) == 1 and len(units[0].requests) == len(counts)
    source = torch.randn(
        17, max(32, layer.n_local_heads), 512, device=device, dtype=torch.bfloat16
    )
    source = source[:, : layer.n_local_heads]
    positions = torch.arange(17, dtype=torch.int64, device=device)
    init_workspace_manager(device)
    try:
        with PreparationSession(
            device=device, autotune=False, compile_workers=2
        ) as session:
            session.prepare(units[0].requests)
            for plan in layer._b12x_wo_plans.values():
                get_b12x_scratch_buffers(plan)
            session.freeze()
            current_workspace_manager().lock()
            for rows in counts:
                plan = layer._b12x_wo_plans[rows]
                assert plan.prepared is not None
                scratch = tuple(
                    torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    for spec in plan.scratch_specs()
                )
                binding = wo_projection.bind_inv_rope(
                    plan,
                    scratch=scratch,
                    o=source[:rows],
                    positions=positions[:rows],
                    cos_sin_cache=table,
                    weights=layer._b12x_wo_projection_weights,
                    heads_per_group=heads_per_group,
                    nope_dim=448,
                    rope_dim=64,
                )
                expected = wo_projection.run_inv_rope(
                    binding=binding, plan=plan
                ).clone()
                actual = layer._o_proj(source[:rows], positions[:rows])
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for buffer in get_b12x_scratch_buffers(plan):
                    buffer.fill_(213)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                try:
                    with session.capture(), torch.cuda.graph(graph):
                        replayed = layer._o_proj(source[:rows], positions[:rows])
                    pointer = replayed.data_ptr()
                    source[:rows].neg_()
                    positions[:rows].add_(1).remainder_(table.shape[0])
                    replayed.fill_(float("nan"))
                    allocated = torch.accelerator.memory_allocated(device)
                    graph.replay()
                    torch.accelerator.synchronize(device)
                    assert torch.accelerator.memory_allocated(device) == allocated
                    assert replayed.data_ptr() == pointer
                    expected = wo_projection.run_inv_rope(binding=binding, plan=plan)
                    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
                    assert (
                        torch.isfinite(replayed).all()
                        and torch.count_nonzero(replayed) > 0
                    )
                finally:
                    graph.reset()
            with pytest.raises(RuntimeError, match="frozen"):
                layer._o_proj(source[:3], positions[:3])
    finally:
        reset_workspace_manager()


def _deepseek_v4_mhc_layer(first_layer):
    hidden = 4096
    layer = torch.nn.Module()
    for name in ("hc_attn_fn", "hc_ffn_fn"):
        setattr(layer, name, torch.empty(24, 4 * hidden, dtype=torch.float32))
    layer.hc_ffn_fn_bf16 = torch.empty(24, 4 * hidden, dtype=torch.bfloat16)
    layer.hc_attn_fn_broadcast = (
        torch.empty(24, hidden, dtype=torch.float32) if first_layer else None
    )
    for name in ("hc_attn_scale", "hc_ffn_scale"):
        setattr(layer, name, torch.empty(3, dtype=torch.float32))
    for name in ("hc_attn_base", "hc_ffn_base"):
        setattr(layer, name, torch.empty(24, dtype=torch.float32))
    for name in ("attn_norm", "ffn_norm"):
        setattr(
            layer,
            name,
            SimpleNamespace(
                weight=torch.empty(hidden, dtype=torch.bfloat16), variance_epsilon=1e-6
            ),
        )
    return layer


@pytest.mark.parametrize("first_layer", [False, True])
@pytest.mark.parametrize("glm_operands", [False, True])
def test_mhc_prepares_model_operands_and_first_layer_broadcast(
    monkeypatch, first_layer, glm_operands
):
    from b12x.norm import mhc

    from vllm.utils.b12x import B12xWorkload

    monkeypatch.setattr(b12x_mla, "_require_b12x_mhc", lambda: mhc)
    layer = _deepseek_v4_mhc_layer(first_layer)
    hidden = layer.attn_norm.weight.numel()
    operands = None
    operations = ("post_pre", "post_pre_bf16", "post")
    if glm_operands:
        layer.input_layernorm = layer.attn_norm
        layer.post_attention_layernorm = layer.ffn_norm
        del layer.attn_norm, layer.ffn_norm, layer.hc_ffn_fn_bf16
        operands = b12x_mla.MHCOperands(
            attn_norm="input_layernorm",
            ffn_norm="post_attention_layernorm",
            ffn_fn_bf16=None,
        )
        operations = ("post_pre", "post")
    owner = b12x_mla.B12xMHCResidual(
        hidden_size=hidden,
        hc_mult=4,
        rms_eps=1e-6,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        operands=operands,
    )
    workload = B12xWorkload(
        stage="weights",
        token_counts=(1, 8, 4096),
        fixed_token_counts=(1, 8),
        output_dtype=torch.bfloat16,
        max_tokens=4096,
        max_seqs=8,
        max_model_len=8192,
    )
    (unit,) = owner.get_b12x_preparation_units(layer, workload)
    plans = [request.plan for request in unit.requests]
    for rows in workload.token_counts:
        for operation in operations:
            plan = owner._plan_for(operation, rows)
            assert any(plan is candidate for candidate in plans)
            assert plan.prepared is None
            assert plan.query.max_tokens == rows
            assert plan.query.operation == (
                "post_pre" if operation == "post_pre_bf16" else operation
            )
            assert plan.query.has_fn_bf16 == (operation == "post_pre_bf16")
        if first_layer:
            assert owner._plan_for("pre", rows).query.operation == "pre"
    assert any(plan.query.operation == "pre" for plan in plans) == first_layer
    assert any(plan.query.has_fn_bf16 for plan in plans) != glm_operands


@pytest.mark.parametrize("operation", ["pre", "post_pre", "post_pre_bf16", "post"])
def test_mhc_preparation_releases_unused_trial_outputs(monkeypatch, operation):
    from b12x.norm import mhc
    from b12x.norm.mhc import _impl

    monkeypatch.setattr(b12x_mla, "_require_b12x_mhc", lambda: mhc)
    layer = _deepseek_v4_mhc_layer(first_layer=True)
    hidden = layer.attn_norm.weight.numel()
    owner = b12x_mla.B12xMHCResidual(
        hidden_size=hidden, hc_mult=4, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20
    )
    outputs = []

    def kernel(*args, **kwargs):
        result = torch.empty(8, 4, hidden, dtype=torch.bfloat16)
        outputs.append(weakref.ref(result))
        return result if operation == "post" else (result,)

    for name in ("pre", "post_pre", "post"):
        monkeypatch.setattr(_impl, f"_b12x_mhc_{name}_impl", kernel)
    call = owner._prepare_call(layer, operation, 8)(object())
    call.produce()
    call.invoke()
    call.invoke()
    assert len(outputs) == 2
    assert all(output() is None for output in outputs)


def _deepseek_v4_mla_layer(device, compress_ratio):
    from vllm.utils.b12x import B12xWorkload

    layer = b12x_mla.DeepseekV4B12xAttention.__new__(b12x_mla.DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "test.deepseek_v4.mla"
    layer.indexer = None
    layer._b12x_cache_page_views = {}
    layer._b12x_mla_plans = {}
    layer.compress_ratio = compress_ratio
    layer.padded_heads = 16
    layer.window_size = 128
    layer.max_image_tokens = 64
    layer.max_model_len = 32768
    layer.scale = 512**-0.5
    layer.attn_sink = torch.zeros(16, dtype=torch.float32, device=device)
    layer.swa_cache_layer = SimpleNamespace(
        block_size=64,
        kv_cache=torch.empty((2, 64 * 584 + 128), dtype=torch.uint8, device=device),
    )
    layer.kv_cache = torch.empty(
        (2, (256 // compress_ratio) * 584 + 128), dtype=torch.uint8, device=device
    )
    layer.topk_indices_buffer = torch.empty(17, 32, dtype=torch.int32, device=device)
    layer.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        speculative_config=SimpleNamespace(
            use_dspark=lambda: True, num_speculative_tokens=3
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=17, max_num_seqs=2),
    )
    workload = B12xWorkload(
        stage="state",
        token_counts=(1, 4, 8, 17),
        fixed_token_counts=(1, 4, 8),
        output_dtype=torch.bfloat16,
        max_tokens=17,
        max_seqs=2,
        max_model_len=32768,
    )
    return layer, workload


@pytest.mark.parametrize("compress_ratio", [1, 4, 128])
def test_deepseek_v4_mla_declares_pool_layout_and_metadata_variants(
    monkeypatch, compress_ratio
):
    from b12x.attention import compressed_sparse_mla as mla

    monkeypatch.setattr(b12x_mla, "_require_b12x_compressed_sparse_mla", lambda: mla)
    layer, workload = _deepseek_v4_mla_layer("cpu", compress_ratio)
    (unit,) = layer.get_b12x_preparation_units(layer, workload)
    assert unit.stage == "state"
    queries = [request.plan.query for request in unit.requests]
    assert {q.query_rows for q in queries if q.mode == "decode"} == set(range(1, 9))
    assert {q.query_rows for q in queries if q.mode == "extend"} == {17}
    expected_index_widths = (
        {0} if compress_ratio == 1 else {32} if compress_ratio == 4 else {128, 256}
    )
    assert {q.indexed_width for q in queries} == expected_index_widths
    for query in queries:
        assert query.swa_cache_stride == layer.swa_cache_layer.kv_cache.stride()
        assert query.swa_cache_shape == (2, 64 * 584)
        assert query.swa_page_size == 64
        assert query.attn_sink_present and query.output_mode == "provided"
        assert query.q_shape == (query.query_rows, 16, 512)
        assert query.indexed_cache_present == (compress_ratio > 1)
        if compress_ratio > 1:
            assert query.indexed_cache_stride == layer.kv_cache.stride()
            assert query.indexed_page_size == 256 // compress_ratio
    assert all(request.plan.prepared is None for request in unit.requests)
    (again,) = layer.get_b12x_preparation_units(layer, workload)
    assert all(a.plan is b.plan for a, b in zip(unit.requests, again.requests))
    layer.swa_cache_layer.kv_cache = layer.swa_cache_layer.kv_cache.clone()
    (replaced,) = layer.get_b12x_preparation_units(layer, workload)
    assert all(a.plan is not b.plan for a, b in zip(unit.requests, replaced.requests))


def test_deepseek_v4_mla_prepares_autofit_context_capacity(monkeypatch):
    """Cache admission can shrink the context after attention construction."""
    from dataclasses import replace

    from b12x.attention import compressed_sparse_mla as mla

    monkeypatch.setattr(b12x_mla, "_require_b12x_compressed_sparse_mla", lambda: mla)
    layer, workload = _deepseek_v4_mla_layer("cpu", 128)
    layer.max_model_len = 1048576
    workload = replace(
        workload,
        max_tokens=4096,
        max_seqs=8,
        token_counts=(1, 48, 4096),
        fixed_token_counts=(1, 48),
        max_model_len=1048576,
    )
    (profile,) = layer.get_b12x_preparation_units(layer, workload)
    (serving,) = layer.get_b12x_preparation_units(
        layer, replace(workload, max_model_len=869888)
    )
    plan = layer._b12x_mla_plan(
        "decode",
        48,
        torch.empty(48, 128, dtype=torch.int32),
        torch.empty(48, 6912, dtype=torch.int32),
    )
    assert any(request.plan is plan for request in serving.requests)
    assert serving.key != profile.key
    assert {request.plan.query.indexed_width for request in serving.requests} == {
        128,
        256,
        512,
        1024,
        2048,
        4096,
        6912,
    }


@pytest.mark.parametrize(
    "compress_ratio,high_pages", [(1, False), (4, False), (128, True)]
)
@pytest.mark.parametrize("autotune", [False, True])
@torch.no_grad()
def test_deepseek_v4_mla_preparation_restores_cache_and_replays(
    compress_ratio, high_pages, autotune, tmp_path
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x compressed MLA requires SM12x")
    from b12x.attention._shared.mla.compressed_reference import (
        compressed_sparse_mla_page_nbytes,
        compressed_sparse_mla_reference,
    )
    from b12x.preparation import PreparationSession

    from vllm.models.deepseek_v4.common.ops import quantize_and_insert_k_cache
    from vllm.utils.b12x import get_b12x_scratch_buffers
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        reset_workspace_manager,
    )

    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.manual_seed(614)
    layer, workload = _deepseek_v4_mla_layer(device, compress_ratio)
    cache_data = []
    for page_size in (64,) if compress_ratio == 1 else (64, 256 // compress_ratio):
        stride = compressed_sparse_mla_page_nbytes(page_size) + 128
        base = 2**31 // stride + 1 if high_pages else 0
        pool = torch.empty((base + 2, stride), dtype=torch.uint8, device=device)
        touched = min(pool.shape[0], (256 + page_size - 1) // page_size)
        pool[:touched].fill_(37)
        snapshot = pool[:touched].clone()
        count = min(32, 2 * page_size)
        source = torch.randn(count, 512, dtype=torch.bfloat16, device=device) * 0.25
        slots = torch.arange(count, dtype=torch.int64, device=device) + base * page_size
        cache_data.append((pool, snapshot, page_size, base, source, slots))
    layer.swa_cache_layer.kv_cache = cache_data[0][0]
    if compress_ratio > 1:
        layer.kv_cache = cache_data[1][0]
    allocated = torch.accelerator.memory_allocated(device)
    (unit,) = layer.get_b12x_preparation_units(layer, workload)
    assert torch.accelerator.memory_allocated(device) == allocated
    init_workspace_manager(device)
    try:
        with PreparationSession(
            device=device, autotune=autotune, compile_workers=2, cache_dir=tmp_path
        ) as session:
            session.prepare(unit.requests)
            for pool, saved, _, _, _, _ in cache_data:
                torch.testing.assert_close(
                    pool[: saved.shape[0]], saved, rtol=0, atol=0
                )
            references = []
            for pool, _, page_size, base, source, slots in cache_data:
                quantize_and_insert_k_cache(source, pool, slots, page_size)
                reference = torch.zeros(
                    (2, compressed_sparse_mla_page_nbytes(page_size)),
                    dtype=torch.uint8,
                    device=device,
                )
                reference[:, : page_size * 584].copy_(
                    pool[base : base + 2, : page_size * 584]
                )
                references.append(reference)
            for plan in layer._b12x_mla_plans.values():
                get_b12x_scratch_buffers(plan)
            session.freeze()
            current_workspace_manager().lock()
            indexed_widths = (
                (0,)
                if compress_ratio == 1
                else (32,)
                if compress_ratio == 4
                else (128, 256)
            )
            cases = [
                ("decode", 3, 128, indexed_widths[0]),
                (
                    "decode",
                    8,
                    b12x_mla.get_dspark_swa_index_width(layer.window_size, 3),
                    indexed_widths[-1],
                ),
                ("extend", 7, 192, indexed_widths[0]),
                ("extend", 17, 192, indexed_widths[-1]),
            ]
            for mode, rows, swa_width, indexed_width in cases:
                query = (
                    torch.randn(rows, 16, 512, dtype=torch.bfloat16, device=device)
                    * 0.25
                )
                output = torch.empty_like(query)
                indices, lengths = [], []
                for width, (_, _, _, _, _, slots) in zip(
                    (swa_width, indexed_width), cache_data
                ):
                    ids = torch.full(
                        (rows, width), -1, dtype=torch.int32, device=device
                    )
                    ids[:, : slots.numel()] = slots.to(torch.int32)
                    indices.append(ids)
                    lengths.append(
                        torch.full(
                            (rows,), slots.numel(), dtype=torch.int32, device=device
                        )
                    )
                indexed_indices = indices[1] if compress_ratio > 1 else None
                plan = layer._b12x_mla_plan(mode, rows, indices[0], indexed_indices)

                def run(
                    query=query,
                    output=output,
                    indices=indices,
                    lengths=lengths,
                    indexed_indices=indexed_indices,
                    plan=plan,
                ):
                    b12x_mla._run_compressed_sparse_mla(
                        q=query,
                        output=output,
                        attn_sink=layer.attn_sink,
                        scale=layer.scale,
                        swa_k_cache=layer._get_cache_page_view(
                            cache_data[0][0], 64, "swa_k_cache"
                        ),
                        swa_indices=indices[0],
                        swa_lens=lengths[0],
                        swa_page_size=64,
                        indexed_k_cache=(
                            layer._get_cache_page_view(
                                cache_data[1][0],
                                256 // compress_ratio,
                                "indexed_k_cache",
                            )
                            if compress_ratio > 1
                            else None
                        ),
                        indexed_indices=indexed_indices,
                        indexed_lens=lengths[1] if compress_ratio > 1 else None,
                        indexed_page_size=256 // compress_ratio
                        if compress_ratio > 1
                        else None,
                        plan=plan,
                    )

                def reference(query=query, indices=indices, lengths=lengths):
                    relative = [
                        torch.where(ids >= 0, ids - base * page_size, ids)
                        for ids, (_, _, page_size, base, _, _) in zip(
                            indices, cache_data
                        )
                    ]
                    return compressed_sparse_mla_reference(
                        query,
                        references[0],
                        relative[0],
                        lengths[0],
                        sm_scale=layer.scale,
                        attn_sink=layer.attn_sink,
                        swa_page_size=64,
                        extra_k_cache=references[1] if compress_ratio > 1 else None,
                        extra_indices=relative[1] if compress_ratio > 1 else None,
                        extra_topk_lengths=lengths[1] if compress_ratio > 1 else None,
                        extra_page_size=256 // compress_ratio
                        if compress_ratio > 1
                        else None,
                    )

                run()
                expected = reference()
                torch.testing.assert_close(
                    output.float(), expected.float(), rtol=0.02, atol=0.01
                )
                assert torch.isfinite(output).all() and torch.count_nonzero(output) > 0
                graph = torch.cuda.CUDAGraph()
                try:
                    with session.capture(), torch.cuda.graph(graph):
                        run()
                    query.neg_()
                    for ids, lens in zip(indices, lengths):
                        ids[:, 0].add_(1)
                        lens[0] = 1
                    expected = reference()
                    output.fill_(float("nan"))
                    pointer = output.data_ptr()
                    allocated = torch.accelerator.memory_allocated(device)
                    graph.replay()
                    torch.accelerator.synchronize(device)
                    assert output.data_ptr() == pointer
                    assert torch.accelerator.memory_allocated(device) == allocated
                    torch.testing.assert_close(
                        output.float(), expected.float(), rtol=0.02, atol=0.01
                    )
                finally:
                    graph.reset()
    finally:
        reset_workspace_manager()


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("autotune", [False, True])
@torch.no_grad()
def test_deepseek_v4_indexer_prepares_replicated_heads_and_live_cache(
    mode, autotune, tmp_path
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x DSA indexing requires SM12x")
    from dataclasses import replace

    from b12x.attention.dsa_indexer.reference import (
        pack_index_k_cache_reference,
        unpack_index_k_cache_reference,
    )
    from b12x.preparation import PreparationSession

    from vllm.model_executor.warmup.b12x_prepare import _units_from_modules
    from vllm.utils.b12x import set_b12x_preparation_provider
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        init_workspace_manager,
        reset_workspace_manager,
    )

    device = torch.device("cuda", torch.accelerator.current_device_index())
    torch.manual_seed(413)
    layer, workload = _deepseek_v4_mla_layer(device, 4)
    workload = replace(
        workload,
        max_tokens=128,
        max_seqs=8,
        max_model_len=1048576,
        speculative_tokens=7,
        token_counts=(1, 8, 64, 128),
        fixed_token_counts=(1, 8, 64),
    )
    layer.indexer = torch.nn.Module()
    layer.indexer.n_head = 64
    layer.indexer.k_cache = torch.nn.Module()
    layer.indexer.k_cache.prefix = "test.deepseek_v4.indexer"
    layer.indexer.k_cache.kv_cache = torch.empty(0)
    indexer = b12x_indexer.B12xC4SparseIndexer(
        layer.indexer.k_cache,
        128,
        "ue8m0",
        512,
        128,
        262144,
        262144,
        torch.empty((64, 512), dtype=torch.int32, device=device),
        skip_k_cache_insert=True,
        compress_ratio=4,
    )
    layer.indexer.indexer_op = indexer
    set_b12x_preparation_provider(layer, layer)
    stride = 64 * 132 + 192
    high_page = 2**31 // stride + 1
    pool = torch.empty((high_page + 16, stride), dtype=torch.uint8, device=device)
    cache = pool[:, : 64 * 132].view(-1, 64, 132)
    pool[:16].fill_(37)
    saved_pages = pool[:16].clone()
    packed = pack_index_k_cache_reference(torch.randn(1024, 128, device=device))
    decoded = unpack_index_k_cache_reference(packed, num_tokens=1024).float()
    cache[high_page:].copy_(packed.reshape(16, 64, 132))
    layer.indexer.k_cache.kv_cache = cache
    allocated = torch.accelerator.memory_allocated(device)
    units = tuple(_units_from_modules(layer, workload))
    assert torch.accelerator.memory_allocated(device) == allocated
    (unit,) = (unit for unit in units if unit.name == "B12xC4SparseIndexer")
    for request in unit.requests:
        query = request.plan.query
        assert query.num_q_heads == 64
        assert query.max_k_rows == 262144
        assert query.operands["index_k_cache"]["strides"] == (stride, 1)
        assert request.plan.prepared is None
    rows = 64 if mode == "decode" else 17
    q = torch.randn(rows, 64, 128, device=device).to(torch.float8_e4m3fn)
    weights = torch.rand(rows, 64, device=device)
    lengths = torch.full((rows,), 1024, dtype=torch.int32, device=device)
    base_table = torch.arange(
        high_page, high_page + 16, dtype=torch.int32, device=device
    )[None].repeat(rows, 1)
    pages = base_table[:1].expand(rows, 16) if mode == "prefill" else base_table
    output = torch.empty((rows, 512), dtype=torch.int32, device=device)

    def run(live_rows):
        indexer.run_paged_topk(
            q=q[:live_rows],
            weights=weights[:live_rows],
            kv_cache=cache,
            seq_lens=lengths[:live_rows],
            block_table=pages[:live_rows],
            output=output[:live_rows],
            shared_page_table=mode == "prefill",
        )

    def expected(live_rows):
        slots = (pages[:live_rows] - high_page).to(torch.int64)[
            :, :, None
        ] * 64 + torch.arange(64, device=device)
        keys = decoded[slots.reshape(live_rows, 1024)]
        logits = torch.einsum("rhd,rkd->rhk", q[:live_rows].float(), keys)
        scores = (logits.relu_() * weights[:live_rows, :, None]).sum(dim=1)
        scores.masked_fill_(
            torch.arange(1024, device=device)[None] >= lengths[:live_rows, None],
            -float("inf"),
        )
        return scores.topk(512, dim=1).indices.to(torch.int32).sort(dim=1).values

    init_workspace_manager(device)
    try:
        with PreparationSession(
            device=device, autotune=autotune, cache_dir=tmp_path
        ) as session:
            session.prepare(unit.requests)
            torch.testing.assert_close(pool[:16], saved_pages, rtol=0, atol=0)
            for plan in indexer._plans.values():
                current_workspace_manager().get_simultaneous(
                    *((spec.shape, spec.dtype) for spec in plan.scratch_specs())
                )
            current_workspace_manager().lock()
            session.freeze()
            for live_rows in (1, 3, rows):
                run(live_rows)
                torch.testing.assert_close(
                    output[:live_rows].sort(dim=1).values,
                    expected(live_rows),
                    rtol=0,
                    atol=0,
                )
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    run(3)
                q.copy_((-q.float()).to(q.dtype))
                lengths[0] = 768
                base_table[:, [0, 1]] = base_table[:, [1, 0]]
                reference = expected(3)
                output.fill_(-1)
                allocated = torch.accelerator.memory_allocated(device)
                graph.replay()
                torch.accelerator.synchronize(device)
                assert torch.accelerator.memory_allocated(device) == allocated
                torch.testing.assert_close(
                    output[:3].sort(dim=1).values, reference, rtol=0, atol=0
                )
            finally:
                graph.reset()
    finally:
        reset_workspace_manager()
