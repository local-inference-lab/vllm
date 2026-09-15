# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import b12x as adapter


@pytest.mark.skipif(not torch.cuda.is_available(), reason="B12X device plan required")
@pytest.mark.parametrize("max_length", [786_688, 1_048_576])
@pytest.mark.parametrize("drafts", [0, 5])
def test_profile_covers_shorter_compressed_prefixes(monkeypatch, max_length, drafts):
    """Shorter prefixes can select more split chunks than the longest prefix."""
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._scratch import (
        plan_compressed_sparse_mla_scratch,
    )

    spec = SimpleNamespace(use_dspark=lambda: True, num_speculative_tokens=drafts)
    layer = SimpleNamespace(
        compress_ratio=128,
        max_model_len=max_length,
        max_num_batched_tokens=4096,
        max_image_tokens=0,
        window_size=128,
        vllm_config=SimpleNamespace(
            speculative_config=spec if drafts else None,
            scheduler_config=SimpleNamespace(
                max_num_seqs=8, max_num_batched_tokens=4096
            ),
        ),
        swa_cache_layer=SimpleNamespace(block_size=256),
    )
    reserved = []
    monkeypatch.setattr(
        adapter,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *items: reserved.append(
                sum(prod(shape) * dtype.itemsize for shape, dtype in items)
            )
        ),
    )
    adapter.DeepseekV4B12xAttention._reserve_profile_workspace(
        layer, torch.empty((0, 32, 512), device="cuda")
    )
    for rows in (1, 48, 128, 240, 256, 257, 1024, 4096):
        for index_width in (128, 256, 512, 1024, 2048, 4096):
            width = 128 + index_width
            capacity = 8 * (1 + drafts) if drafts else None
            plan = plan_compressed_sparse_mla_scratch(
                mla.Caps(
                    device=torch.device("cuda"),
                    num_q_heads=32,
                    max_q_rows=rows,
                    max_width=width,
                    head_dim=512,
                    v_head_dim=512,
                    page_size=256,
                    decode_row_capacity=capacity,
                    max_chunks_per_row=mla.split_chunks_for_contract(
                        rows=rows,
                        width=width,
                        decode_row_capacity=capacity,
                    ),
                )
            )
            required = sum(
                prod(shape) * dtype.itemsize
                for shape, dtype in plan.shapes_and_dtypes()
            )
            assert max(reserved) >= required, (rows, width, reserved, required)
