# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup import sparse_mla_triton_warmup


def test_b12x_prefill_metadata_warmup_uses_runtime_dcp_shape(monkeypatch) -> None:
    calls = []

    class B12xBackend:
        @staticmethod
        def get_name() -> str:
            return "B12X_MLA_SPARSE"

    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cpu"),
        dcp_rank=2,
        dcp_size=4,
        cp_interleave=64,
        attn_groups=[[SimpleNamespace(backend=B12xBackend())]],
    )
    worker = SimpleNamespace(
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=3072),
    )

    monkeypatch.setattr(
        sparse_mla_triton_warmup,
        "_warm_prefill_chunk_metadata_kernel",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    sparse_mla_triton_warmup.sparse_mla_triton_warmup_if_needed(worker)

    assert calls == [
        (
            (torch.device("cpu"),),
            {
                "compress_ratio": 1,
                "query_len": 8,
                "dcp_rank": 2,
                "dcp_world_size": 4,
                "cp_kv_cache_interleave_size": 64,
            },
        )
    ]
