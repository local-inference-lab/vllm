# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer GQA builder: reorder threshold under DCP with spec decode."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("FlashInfer backend requires a CUDA platform.", allow_module_level=True)

import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.v1.attention.backends import flashinfer as flashinfer_backend
from vllm.v1.attention.backends.flashinfer import (
    BatchDCPPrefillWrapper,
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
    _get_dcp_local_kv_page_metadata,
)
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_flashinfer_gqa_dcp_spec_decode_clamps_reorder_threshold(monkeypatch):
    """trtllm-gen decode receives no cp_rank/global-seq-len information, so its
    end-aligned causal mask is wrong for q_len > 1 over the DCP-interleaved
    local KV shard. The builder must keep reorder_batch_threshold at 1 under
    DCP so spec queries take the (DCP-aware) prefill path instead.
    """
    vllm_config = create_vllm_config(max_model_len=1024)
    vllm_config.parallel_config.decode_context_parallel_size = 2
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram", num_speculative_tokens=3
    )

    monkeypatch.setattr(
        flashinfer_backend, "can_use_trtllm_attention", lambda *args, **kwargs: True
    )
    monkeypatch.setattr(
        FlashInferMetadataBuilder,
        "_get_flashinfer_trtllm_api_decode_kernel",
        staticmethod(lambda: FlashInferDecodeKernel.TRTLLM_GEN),
    )
    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        lambda *args, **kwargs: {
            "layer.0": PerLayerParameters(
                window_left=-1, logits_soft_cap=None, sm_scale=0.1, has_sinks=False
            )
        },
    )

    kv_cache_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        ),
        head_size=vllm_config.model_config.get_head_size(),
        dtype=vllm_config.model_config.dtype,
    )
    with set_current_vllm_config(vllm_config):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec,
            ["layer.0"],
            vllm_config,
            torch.device("cpu"),
        )

    # Guard against passing vacuously with the kernel disabled.
    assert (
        builder.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.TRTLLM_GEN
    )
    assert builder.reorder_batch_threshold == 1


def test_dcp_prefill_wrapper_preserves_noncausal_draft_mask() -> None:
    wrapper = BatchDCPPrefillWrapper.__new__(BatchDCPPrefillWrapper)
    wrapper._context = Mock()
    wrapper._new_tokens = Mock()
    indptr = torch.tensor([0, 3], dtype=torch.int32)

    wrapper.plan(
        qo_indptr_cpu=indptr,
        paged_kv_indptr_cpu=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len_cpu=torch.tensor([3], dtype=torch.int32),
        page_size=16,
        num_qo_heads=4,
        dcp_world_size=2,
        num_kv_heads=1,
        head_dim=128,
        sm_scale=0.1,
        window_left=-1,
        logits_soft_cap=None,
        q_data_type=torch.bfloat16,
        kv_cache_dtype=torch.float8_e4m3fn,
        prefill_fixed_split_size=-1,
        disable_split_kv=False,
        causal=False,
    )

    assert wrapper._context.plan.call_args.kwargs["causal"] is False
    assert wrapper._new_tokens.plan.call_args.kwargs["causal"] is False


def test_flashinfer_selects_dcp_wrapper_for_noncausal_prefill(monkeypatch) -> None:
    expected = object()
    factory = Mock(return_value=expected)
    monkeypatch.setattr(flashinfer_backend, "BatchDCPPrefillWrapper", factory)
    monkeypatch.setattr(
        flashinfer_backend,
        "get_flashinfer_layout_string",
        lambda _: "NHD",
    )

    builder = FlashInferMetadataBuilder.__new__(FlashInferMetadataBuilder)
    builder.use_dcp = True
    builder.dcp_a2a = False
    builder._prefill_wrapper = None
    builder.cache_config = SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: object()
    )
    builder._get_workspace_buffer = Mock(return_value=torch.empty(0))

    assert builder._get_prefill_wrapper(causal=False) is expected
    factory.assert_called_once()


@pytest.mark.parametrize(
    ("rank", "expected_lengths", "expected_pages"),
    [
        (0, [0, 12, 17], [0, 1, 2]),
        (1, [0, 12, 16], [0, 1, 1]),
        (2, [0, 12, 16], [0, 1, 1]),
        (3, [0, 8, 16], [0, 1, 1]),
    ],
)
def test_dcp_flashinfer_page_metadata_is_rank_local(
    rank: int,
    expected_lengths: list[int],
    expected_pages: list[int],
) -> None:
    # Global lengths 44 and 65 would incorrectly produce 3 and 5 native
    # pages at page_size=16. DCP4 stores only the rank-local lengths below.
    local_lens, local_lens_np, local_pages_np = _get_dcp_local_kv_page_metadata(
        torch.tensor([0, 44, 65], dtype=torch.int32),
        dcp_world_size=4,
        dcp_rank=rank,
        dcp_kv_cache_interleave_size=4,
        page_size=16,
    )

    assert local_lens.tolist() == expected_lengths
    assert local_lens_np.tolist() == expected_lengths
    assert local_pages_np.tolist() == expected_pages
