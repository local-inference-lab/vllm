# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for mixed speculative and non-speculative GDN metadata."""

from dataclasses import dataclass

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import MambaSpec

BLOCK_SIZE = 16
DEVICE = torch.device("cpu")


@dataclass
class GDNBuildTestCase:
    """Specification for a GDN metadata builder classification test."""

    seq_lens: list[int]
    query_lens: list[int]
    num_decode_draft_tokens: list[int] | None  # None = no spec config
    num_speculative_tokens: int
    expected_num_decodes: int
    expected_num_prefills: int
    expected_num_prefill_tokens: int
    expected_num_spec_decodes: int


GDN_BUILD_TEST_CASES = {
    # The original #34845 crash: non-spec query_len=1 + spec decode
    "mixed_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[65, 20],
        query_lens=[1, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
    # All requests are spec decodes — no reclassification needed
    "pure_spec_decode": GDNBuildTestCase(
        seq_lens=[50, 30],
        query_lens=[3, 3],
        num_decode_draft_tokens=[2, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=2,
    ),
    # No speculative config at all — standard decode path
    "pure_regular_decode": GDNBuildTestCase(
        seq_lens=[40, 30, 20],
        query_lens=[1, 1, 1],
        num_decode_draft_tokens=None,
        num_speculative_tokens=0,
        expected_num_decodes=3,
        expected_num_prefills=0,
        expected_num_prefill_tokens=0,
        expected_num_spec_decodes=0,
    ),
    # Multi-token prefill alongside spec decode — no decode to reclassify
    "spec_decode_with_real_prefill": GDNBuildTestCase(
        seq_lens=[100, 20],
        query_lens=[50, 3],
        num_decode_draft_tokens=[-1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=50,
        expected_num_spec_decodes=1,
    ),
    # All three types in one batch — decode gets reclassified
    "prefill_decode_and_spec_decode": GDNBuildTestCase(
        seq_lens=[100, 65, 20],
        query_lens=[50, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=2,
        expected_num_prefill_tokens=51,
        expected_num_spec_decodes=1,
    ),
    # Multiple non-spec query_len=1 requests all reclassified
    "multiple_decodes_reclassified": GDNBuildTestCase(
        seq_lens=[40, 50, 60, 20],
        query_lens=[1, 1, 1, 3],
        num_decode_draft_tokens=[-1, -1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=3,
        expected_num_prefill_tokens=3,
        expected_num_spec_decodes=1,
    ),
    # Zero-length padded sequence excluded from counts
    "zero_length_padding_with_spec": GDNBuildTestCase(
        seq_lens=[16, 65, 20],
        query_lens=[0, 1, 3],
        num_decode_draft_tokens=[-1, -1, 2],
        num_speculative_tokens=2,
        expected_num_decodes=0,
        expected_num_prefills=1,
        expected_num_prefill_tokens=1,
        expected_num_spec_decodes=1,
    ),
}


def _create_gdn_builder(
    num_speculative_tokens: int = 0,
    full_cuda_graph: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
) -> GDNAttentionMetadataBuilder:
    """Create a GDNAttentionMetadataBuilder with minimal config."""
    vllm_config = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        max_num_batched_tokens=4096,
    )
    if full_cuda_graph:
        vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    if num_speculative_tokens > 0:
        vllm_config.speculative_config = SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=num_speculative_tokens,
        )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
    )
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=mamba_spec,
        layer_names=["layer.0"],
        vllm_config=vllm_config,
        device=DEVICE,
    )


def _build(
    builder: GDNAttentionMetadataBuilder,
    batch_spec: BatchSpec,
    num_decode_draft_tokens: list[int] | None = None,
    is_prefilling: list[bool] | None = None,
) -> GDNAttentionMetadata:
    """Build GDN attention metadata, optionally with spec-decode kwargs."""
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    if is_prefilling is None:
        is_prefilling = [False] * batch_spec.batch_size
    common = common.replace(is_prefilling=torch.tensor(is_prefilling))
    kwargs: dict = {}
    if num_decode_draft_tokens is not None:
        kwargs["num_decode_draft_tokens_cpu"] = torch.tensor(
            num_decode_draft_tokens, dtype=torch.int32
        )
        kwargs["num_accepted_tokens"] = torch.ones(
            batch_spec.batch_size, dtype=torch.int32, device=DEVICE
        )
    return builder.build(common_prefix_len=0, common_attn_metadata=common, **kwargs)


@pytest.mark.parametrize(
    "test_case", GDN_BUILD_TEST_CASES.values(), ids=GDN_BUILD_TEST_CASES.keys()
)
def test_gdn_build_classification(test_case: GDNBuildTestCase):
    """Test that GDN metadata builder classifies requests correctly."""
    builder = _create_gdn_builder(test_case.num_speculative_tokens)
    batch = BatchSpec(seq_lens=test_case.seq_lens, query_lens=test_case.query_lens)
    meta = _build(builder, batch, test_case.num_decode_draft_tokens)

    assert meta.num_decodes == test_case.expected_num_decodes
    assert meta.num_prefills == test_case.expected_num_prefills
    assert meta.num_prefill_tokens == test_case.expected_num_prefill_tokens
    assert meta.num_spec_decodes == test_case.expected_num_spec_decodes


def test_fresh_single_token_prompt_uses_prefill_state_initialization() -> None:
    builder = _create_gdn_builder()
    batch = BatchSpec(seq_lens=[1], query_lens=[1])

    meta = _build(builder, batch, is_prefilling=[True])

    assert meta.num_decodes == 0
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 1
    assert meta.has_initial_state is not None
    assert meta.has_initial_state.tolist() == [False]
    assert meta.prefill_query_start_loc is not None
    assert meta.prefill_query_start_loc.tolist() == [0, 1]


def test_fresh_two_token_prompt_uses_prefill_state_initialization() -> None:
    builder = _create_gdn_builder()
    batch = BatchSpec(seq_lens=[2], query_lens=[2])

    meta = _build(builder, batch, is_prefilling=[True])

    assert meta.num_decodes == 0
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == 2
    assert meta.has_initial_state is not None
    assert meta.has_initial_state.tolist() == [False]
    assert meta.prefill_query_start_loc is not None
    assert meta.prefill_query_start_loc.tolist() == [0, 2]


def test_has_initial_state_after_reclassification():
    """After reclassification, num_prefills > 0 so the prefill kernel path
    should compute has_initial_state. For the reclassified request with
    context_lens > 0, the corresponding entry must be True."""
    builder = _create_gdn_builder(num_speculative_tokens=2)
    batch = BatchSpec(seq_lens=[65, 20], query_lens=[1, 3])
    meta = _build(builder, batch, num_decode_draft_tokens=[-1, 2])

    assert meta.num_prefills > 0, "reclassification should produce prefills"
    assert meta.has_initial_state is not None
    # req0 has context_lens = 65 - 1 = 64 > 0, so has_initial_state[0] = True
    assert meta.has_initial_state[0].item() is True


def test_full_cudagraph_spec_metadata_uses_request_count():
    """FULL cudagraph token padding must not pad request-indexed metadata."""
    num_speculative_tokens = 3
    builder = _create_gdn_builder(
        num_speculative_tokens=num_speculative_tokens,
        full_cuda_graph=True,
    )
    batch = BatchSpec(seq_lens=[80, 96], query_lens=[4, 4])
    meta = _build(builder, batch, num_decode_draft_tokens=[3, 3])

    assert meta.num_spec_decodes == batch.batch_size
    assert meta.num_spec_decode_tokens == batch.compute_num_tokens()
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_state_indices_tensor.shape == (
        batch.batch_size,
        num_speculative_tokens + 1,
    )
    assert meta.spec_sequence_masks is not None
    assert meta.spec_sequence_masks.shape == (batch.batch_size,)
    assert meta.spec_query_start_loc is not None
    assert meta.spec_query_start_loc.shape == (batch.batch_size + 1,)
    assert meta.num_accepted_tokens is not None
    assert meta.num_accepted_tokens.shape == (batch.batch_size,)


def test_gdn_block_table_reuse_supports_regular_and_spec_decode() -> None:
    assert _create_gdn_builder().supports_update_block_table
    assert _create_gdn_builder(num_speculative_tokens=2).supports_update_block_table


def test_gdn_prefill_checkpoint_targets_crossed_cache_boundary() -> None:
    builder = _create_gdn_builder(num_prefill_checkpoint_blocks=1)
    builder.vllm_config.cache_config.mamba_cache_mode = "align"
    batch = BatchSpec(seq_lens=[33], query_lens=[33])
    common = create_common_attn_metadata(
        batch,
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    ).replace(is_prefilling=torch.tensor([True]))

    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    assert metadata.prefill_checkpoint is not None
    torch.testing.assert_close(
        metadata.prefill_checkpoint.checkpoint_offsets,
        torch.tensor([32], dtype=torch.int32),
    )
    torch.testing.assert_close(
        metadata.prefill_checkpoint.request_rows,
        torch.tensor([0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        metadata.prefill_checkpoint.block_table_columns,
        torch.tensor([1], dtype=torch.int64),
    )
    torch.testing.assert_close(
        metadata.prefill_checkpoint.state_indices,
        torch.tensor([1], dtype=torch.int32),
    )


def test_gdn_prefill_checkpoint_refreshes_reused_block_table() -> None:
    builder = _create_gdn_builder(num_prefill_checkpoint_blocks=1)
    builder.vllm_config.cache_config.mamba_cache_mode = "align"
    batch = BatchSpec(seq_lens=[33], query_lens=[33])
    common = create_common_attn_metadata(
        batch,
        BLOCK_SIZE,
        DEVICE,
        arange_block_indices=True,
    ).replace(is_prefilling=torch.tensor([True]))
    metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
    replacement_block_table = torch.tensor(
        [[101, 102, 103]],
        dtype=torch.int32,
    )

    updated = builder.update_block_table(
        metadata,
        replacement_block_table,
        torch.zeros(33, dtype=torch.int64),
    )

    assert updated.prefill_checkpoint is not None
    torch.testing.assert_close(
        updated.prefill_checkpoint.state_indices,
        torch.tensor([102], dtype=torch.int32),
    )


def test_gdn_update_block_table_uses_current_builders_graph_buffers() -> None:
    builder_a = _create_gdn_builder(full_cuda_graph=True)
    builder_b = _create_gdn_builder(full_cuda_graph=True)
    batch = BatchSpec(seq_lens=[40, 30, 20], query_lens=[1, 1, 1])
    metadata_a = _build(builder_a, batch)
    block_table_b = torch.tensor(
        [[11, 12, 13], [21, 22, 23], [31, 32, 33]],
        dtype=torch.int32,
    )

    metadata_b = builder_b.update_block_table(
        metadata_a,
        block_table_b,
        torch.zeros(3, dtype=torch.int64),
    )

    assert metadata_b.non_spec_state_indices_tensor is not None
    assert (
        metadata_b.non_spec_state_indices_tensor.data_ptr()
        == builder_b.non_spec_state_indices_tensor.data_ptr()
    )
    assert metadata_b.non_spec_query_start_loc is not None
    assert (
        metadata_b.non_spec_query_start_loc.data_ptr()
        == builder_b.non_spec_query_start_loc.data_ptr()
    )
    expected_state_indices = mamba_get_block_table_tensor(
        block_table_b,
        metadata_a.seq_lens,
        builder_b.kv_cache_spec,
        builder_b.vllm_config.cache_config.mamba_cache_mode,
    )
    torch.testing.assert_close(
        metadata_b.non_spec_state_indices_tensor,
        expected_state_indices[:, 0],
    )
    torch.testing.assert_close(
        metadata_b.non_spec_query_start_loc,
        metadata_a.non_spec_query_start_loc,
    )


def test_gdn_build_uses_precomputed_aligned_state_indices(monkeypatch) -> None:
    builder = _create_gdn_builder()
    builder.vllm_config.cache_config.mamba_cache_mode = "align"
    aligned_state_indices = torch.tensor(
        [[101, 102], [201, 202], [301, 302]],
        dtype=torch.int32,
    )
    builder.mamba_aligned_state_indices = aligned_state_indices

    def fail_fallback(*_args, **_kwargs):
        raise AssertionError("per-group aligned state-index fallback was used")

    monkeypatch.setattr(
        "vllm.v1.attention.backends.gdn_attn.mamba_get_block_table_tensor",
        fail_fallback,
    )
    metadata = _build(
        builder,
        BatchSpec(seq_lens=[40, 30, 20], query_lens=[1, 1, 1]),
    )

    torch.testing.assert_close(
        metadata.non_spec_state_indices_tensor,
        aligned_state_indices[:, 0],
    )


def test_gdn_spec_update_uses_current_builders_graph_buffers() -> None:
    builder_a = _create_gdn_builder(
        num_speculative_tokens=2,
        full_cuda_graph=True,
    )
    builder_b = _create_gdn_builder(
        num_speculative_tokens=2,
        full_cuda_graph=True,
    )
    builder_a.mamba_aligned_state_indices = torch.tensor(
        [[101, 102, 103], [201, 202, 203]],
        dtype=torch.int32,
    )
    builder_b.mamba_aligned_state_indices = torch.tensor(
        [[111, 112, 113], [211, 212, 213]],
        dtype=torch.int32,
    )
    batch = BatchSpec(seq_lens=[40, 30], query_lens=[3, 3])
    metadata_a = _build(builder_a, batch, num_decode_draft_tokens=[2, 2])
    block_table_b = torch.tensor(
        [[11, 12, 13], [21, 22, 23]],
        dtype=torch.int32,
    )

    metadata_b = builder_b.update_block_table(
        metadata_a,
        block_table_b,
        torch.zeros(6, dtype=torch.int64),
    )

    assert metadata_b.spec_state_indices_tensor is not None
    assert (
        metadata_b.spec_state_indices_tensor.data_ptr()
        == builder_b.spec_state_indices_tensor.data_ptr()
    )
    assert metadata_b.spec_sequence_masks is not None
    assert (
        metadata_b.spec_sequence_masks.data_ptr()
        == builder_b.spec_sequence_masks.data_ptr()
    )
    assert metadata_b.spec_query_start_loc is not None
    assert (
        metadata_b.spec_query_start_loc.data_ptr()
        == builder_b.spec_query_start_loc.data_ptr()
    )
    assert metadata_b.num_accepted_tokens is not None
    assert (
        metadata_b.num_accepted_tokens.data_ptr()
        == builder_b.num_accepted_tokens.data_ptr()
    )
    torch.testing.assert_close(
        metadata_b.spec_state_indices_tensor,
        builder_b.mamba_aligned_state_indices,
    )
    torch.testing.assert_close(
        metadata_b.spec_sequence_masks,
        metadata_a.spec_sequence_masks,
    )
    torch.testing.assert_close(
        metadata_b.spec_query_start_loc,
        metadata_a.spec_query_start_loc,
    )
    torch.testing.assert_close(
        metadata_b.num_accepted_tokens,
        metadata_a.num_accepted_tokens,
    )


def test_gdn_mixed_spec_update_selects_group_specific_state_indices() -> None:
    builder_a = _create_gdn_builder(num_speculative_tokens=2)
    builder_b = _create_gdn_builder(num_speculative_tokens=2)
    builder_a.mamba_aligned_state_indices = torch.tensor(
        [[101, 102, 103], [201, 202, 203]],
        dtype=torch.int32,
    )
    builder_b.mamba_aligned_state_indices = torch.tensor(
        [[111, 112, 113], [211, 212, 213]],
        dtype=torch.int32,
    )
    metadata_a = _build(
        builder_a,
        BatchSpec(seq_lens=[65, 20], query_lens=[1, 3]),
        num_decode_draft_tokens=[-1, 2],
    )

    metadata_b = builder_b.update_block_table(
        metadata_a,
        torch.zeros((2, 3), dtype=torch.int32),
        torch.zeros(4, dtype=torch.int64),
    )

    assert metadata_b.spec_state_indices_tensor is not None
    assert metadata_b.non_spec_state_indices_tensor is not None
    assert metadata_b.prefill_state_indices is not None
    torch.testing.assert_close(
        metadata_b.spec_state_indices_tensor,
        builder_b.mamba_aligned_state_indices[1:2],
    )
    torch.testing.assert_close(
        metadata_b.non_spec_state_indices_tensor,
        builder_b.mamba_aligned_state_indices[0:1, 0],
    )
    torch.testing.assert_close(
        metadata_b.prefill_state_indices,
        builder_b.mamba_aligned_state_indices[0:1, 0],
    )
