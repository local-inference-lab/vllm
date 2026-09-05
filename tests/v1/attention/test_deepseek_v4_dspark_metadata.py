# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.models.deepseek_v4.nvidia.b12x import _get_dspark_decode_row_capacity
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLAMetadataBuilder
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def _dspark_config(max_num_seqs: int = 8) -> SimpleNamespace:
    speculative_config = SimpleNamespace(
        num_speculative_tokens=5,
        parallel_drafting=True,
        use_dspark=lambda: True,
    )
    hf_config = SimpleNamespace(
        sliding_window=128, compress_ratios=[1, 4, 128], index_topk=2048
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=4096, hf_config=hf_config),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_seqs=max_num_seqs,
        ),
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )


def test_dspark_swa_decode_threshold_matches_target_verification() -> None:
    builder = DeepseekSparseSWAMetadataBuilder(
        MLAAttentionSpec(
            block_size=256,
            num_kv_heads=1,
            head_size=512,
            dtype=torch.bfloat16,
        ),
        ["placeholder"],
        _dspark_config(),
        torch.device("cpu"),
    )

    assert builder.decode_threshold == 6


def test_dspark_sparse_mla_split_matches_swa_split() -> None:
    config = _dspark_config()
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
    )
    swa_builder = DeepseekSparseSWAMetadataBuilder(
        spec, ["placeholder"], config, torch.device("cpu")
    )
    sparse_builder = DeepseekV4SparseMLAMetadataBuilder(
        spec, ["placeholder"], config, torch.device("cpu")
    )

    assert sparse_builder.reorder_batch_threshold == swa_builder.decode_threshold == 6


def test_dspark_decode_capacity_excludes_prefill_warmup_rows() -> None:
    assert _get_dspark_decode_row_capacity(_dspark_config()) == 48
