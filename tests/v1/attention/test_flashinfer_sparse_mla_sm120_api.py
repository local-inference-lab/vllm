# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120_mod
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    FlashInferMLASparseMetadataBuilder,
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import select_common_block_size


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_glm5next_sm120_backend_accepts_512_head_size(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm5_next")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_glm5next_sm120_backend_publishes_packed_cache_geometry() -> None:
    probe = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=656,
    )

    with set_current_vllm_config(_fake_vllm_config("glm5_next")):
        packed = FlashInferMLASparseSM120Backend.customize_spec(probe)

    assert packed.state_content_bytes == 656
    assert packed.page_tail_bytes_per_token == 37
    assert packed.model_version == "glm5_next"
    assert packed.page_size_bytes == 256 * (656 + 37)
    assert FlashInferMLASparseSM120Backend.supported_kv_cache_layouts() == (
        KVCacheLayout.BLHNC,
    )


def test_glm5next_sm120_backend_publishes_geometry_without_config_context() -> None:
    probe = MLAAttentionSpec(
        block_size=2304,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=656,
        model_version="glm5_next",
    )

    packed = FlashInferMLASparseSM120Backend.customize_spec(probe)

    assert packed.state_content_bytes == 656
    assert packed.page_tail_bytes_per_token == 37
    assert packed.model_version == "glm5_next"
    assert packed.page_size_bytes == 2304 * (656 + 37)


def test_sm120_backend_does_not_reclassify_deepseek_v4_as_glm() -> None:
    probe = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        state_content_bytes=584,
        model_version="deepseek_v4",
    )

    assert FlashInferMLASparseSM120Backend.customize_spec(probe) is probe


def test_glm5next_sm120_backend_keeps_packed_manager_page_intact() -> None:
    with set_current_vllm_config(_fake_vllm_config("glm5_next")):
        supported = FlashInferMLASparseSM120Backend.get_supported_kernel_block_sizes()
        selected = select_common_block_size(
            2304,
            [FlashInferMLASparseSM120Backend],
        )

    assert len(supported) == 1
    assert isinstance(supported[0], MultipleOf)
    assert supported[0].base == 64
    assert selected == 2304


@pytest.mark.parametrize("qk_rope_head_dim", [0, 64])
@pytest.mark.parametrize("page_block_size", [64, 256, 2304])
def test_sm120_forward_passes_active_topk_lengths(
    monkeypatch, qk_rope_head_dim: int, page_block_size: int
) -> None:
    impl = sm120_mod.FlashInferMLASparseSM120Impl.__new__(
        sm120_mod.FlashInferMLASparseSM120Impl
    )
    impl.num_heads = 2
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256 if qk_rope_head_dim == 0 else 128
    impl.qk_rope_head_dim = qk_rope_head_dim
    impl.rope_pad = 64 if qk_rope_head_dim == 0 else 0
    impl.kernel_qk_rope_head_dim = qk_rope_head_dim + impl.rope_pad
    impl.scale = 0.125
    impl.kv_scale_format = "arbitrary_fp32"
    if qk_rope_head_dim == 0:
        impl.topk_indices_buffer = torch.tensor(
            [[4, 3, 2, 1, 9, 8, 7], [7, 6, 5, 4, 3, 2, 1]],
            dtype=torch.int32,
        )
    else:
        impl.topk_indices_buffer = torch.tensor(
            [[4, 3, -1, -1], [7, 6, 5, -1]], dtype=torch.int32
        )
    impl._workspace_buffer = torch.empty(1, dtype=torch.uint8)

    physical = torch.tensor([[20, 19, -1, -1], [23, 22, 21, -1]], dtype=torch.int32)
    active_lens = torch.tensor([2, 3], dtype=torch.int32)
    convert_kwargs = {}

    converted = {}

    def fake_convert(*args, **kwargs):
        converted["indices"] = args[2].clone()
        convert_kwargs.update(kwargs)
        return physical, active_lens

    call_kwargs = {}

    def fake_flashinfer(**kwargs):
        call_kwargs.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(sm120_mod, "triton_filter_and_convert_dcp_index", fake_convert)
    monkeypatch.setattr(
        fi_utils, "flashinfer_trtllm_batch_decode_with_kv_cache_mla", fake_flashinfer
    )

    head_dim = 512 + qk_rope_head_dim
    q = torch.empty(2, 2, head_dim, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        block_size=64,
        topk_tokens=4,
    )
    out, lse = impl.forward_mqa(
        q,
        torch.empty(
            1,
            page_block_size,
            656,
            dtype=torch.uint8,
        ),
        metadata,
        SimpleNamespace(),
    )

    assert convert_kwargs["return_valid_counts"] is True
    assert convert_kwargs["dcp_size"] == 1
    assert convert_kwargs["dcp_rank"] == 0
    assert call_kwargs["backend"] == "sparse"
    expected_kernel_rows = 2 if page_block_size == 64 else 65
    assert call_kwargs["query"].shape[0] == expected_kernel_rows
    assert call_kwargs["block_tables"].shape[0] == expected_kernel_rows
    assert call_kwargs["seq_lens"].shape[0] == expected_kernel_rows
    torch.testing.assert_close(call_kwargs["seq_lens"][:2], active_lens)
    if page_block_size != 64:
        assert torch.count_nonzero(call_kwargs["query"][2:]) == 0
        assert torch.all(call_kwargs["block_tables"][2:] == -1)
        assert torch.count_nonzero(call_kwargs["seq_lens"][2:]) == 0
    if qk_rope_head_dim == 0:
        torch.testing.assert_close(
            converted["indices"],
            torch.tensor([[4, 9, 8, 7], [7, 3, 2, 1]], dtype=torch.int32),
        )
        assert call_kwargs["query"].shape[-1] == 576
        assert call_kwargs["qk_rope_head_dim"] == 64
    else:
        assert call_kwargs["query"].shape[-1] == 576
        assert call_kwargs["qk_rope_head_dim"] == 64
    assert out.shape == (2, 2, 512)
    assert lse is None


def test_sm120_nope_cache_update_zero_pads_rope(monkeypatch) -> None:
    impl = sm120_mod.FlashInferMLASparseSM120Impl.__new__(
        sm120_mod.FlashInferMLASparseSM120Impl
    )
    impl.rope_pad = 64
    captured = {}

    def fake_update(self, kv_c, k_pe, cache, slots, cache_dtype, scale):
        captured["k_pe"] = k_pe

    monkeypatch.setattr(sm120_mod.MLAAttentionImpl, "do_kv_cache_update", fake_update)
    impl.do_kv_cache_update(
        torch.empty(2, 512),
        torch.empty(2, 1, 0),
        torch.empty(1, 2, 656, dtype=torch.uint8),
        torch.tensor([0, 1]),
        "fp8_ds_mla",
        torch.ones(1),
    )

    assert captured["k_pe"].shape == (2, 1, 64)
    assert torch.count_nonzero(captured["k_pe"]) == 0


def _selector_metadata_builder(capacity: int = 4):
    builder = FlashInferMLASparseMetadataBuilder.__new__(
        FlashInferMLASparseMetadataBuilder
    )
    builder.requires_glm_next_selector_metadata = True
    builder._capture_default_state_slot_ids = torch.arange(capacity, dtype=torch.int32)
    builder._capture_state_slot_ids = torch.empty(capacity, dtype=torch.int32)
    builder._capture_state_is_fresh = torch.ones(capacity, dtype=torch.bool)
    builder._capture_num_accepted_tokens = torch.ones(capacity, dtype=torch.int32)
    builder._capture_is_prefilling = torch.zeros(capacity, dtype=torch.bool)
    return builder


def test_glm5next_flashinfer_stages_runtime_selector_metadata() -> None:
    builder = _selector_metadata_builder()
    staged = builder._stage_glm_next_selector_metadata(
        num_reqs=2,
        for_cudagraph_capture=False,
        selector_state_slot_ids=torch.tensor([7, 3], dtype=torch.int32),
        selector_state_is_fresh=torch.tensor([False, True]),
        selector_num_accepted_tokens=torch.tensor([4, 1], dtype=torch.int32),
        selector_is_prefilling=torch.tensor([False, True]),
    )

    torch.testing.assert_close(staged[0], torch.tensor([7, 3], dtype=torch.int32))
    torch.testing.assert_close(staged[1], torch.tensor([False, True]))
    torch.testing.assert_close(staged[2], torch.tensor([4, 1], dtype=torch.int32))
    torch.testing.assert_close(staged[3], torch.tensor([False, True]))


def test_glm5next_flashinfer_routes_short_extends_through_prefill() -> None:
    builder = _selector_metadata_builder()
    assert not builder._should_treat_short_extends_as_decodes()

    builder.requires_glm_next_selector_metadata = False
    assert builder._should_treat_short_extends_as_decodes()


def test_glm5next_flashinfer_stages_capture_selector_metadata() -> None:
    builder = _selector_metadata_builder()
    staged = builder._stage_glm_next_selector_metadata(
        num_reqs=3,
        for_cudagraph_capture=True,
        selector_state_slot_ids=None,
        selector_state_is_fresh=None,
        selector_num_accepted_tokens=None,
        selector_is_prefilling=None,
    )

    torch.testing.assert_close(staged[0], torch.arange(3, dtype=torch.int32))
    assert staged[1].all()
    torch.testing.assert_close(staged[2], torch.ones(3, dtype=torch.int32))
    assert not staged[3].any()


def test_glm5next_flashinfer_rejects_missing_runtime_selector_metadata() -> None:
    builder = _selector_metadata_builder()
    with pytest.raises(RuntimeError, match="requires selector state slots"):
        builder._stage_glm_next_selector_metadata(
            num_reqs=1,
            for_cudagraph_capture=False,
            selector_state_slot_ids=None,
            selector_state_is_fresh=None,
            selector_num_accepted_tokens=None,
            selector_is_prefilling=None,
        )


def test_glm5next_flashinfer_metadata_exposes_pooled_selector_fields() -> None:
    assert FlashInferMLASparseMetadata.prefill_query_lens_cpu is None
    assert FlashInferMLASparseMetadata.selector_state_slot_ids is None
    assert FlashInferMLASparseMetadata.selector_state_is_fresh is None
    assert FlashInferMLASparseMetadata.selector_num_accepted_tokens is None
    assert FlashInferMLASparseMetadata.selector_is_prefilling is None


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192
