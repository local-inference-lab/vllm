# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.warmup import b12x_sparse_indexer_warmup as warmup


@pytest.mark.parametrize("runner_kind", ["current", "legacy"])
def test_block_table_widths_by_layer(runner_kind):
    table = torch.empty((4, 37), dtype=torch.int32)
    group = SimpleNamespace(layer_names=["model.layers.0.indexer.k_cache"])
    runner = SimpleNamespace(kv_cache_config=SimpleNamespace(kv_cache_groups=[group]))
    if runner_kind == "current":
        runner.block_tables = SimpleNamespace(input_block_tables=[table])
    else:
        runner.input_batch = SimpleNamespace(
            block_table=SimpleNamespace(
                block_tables=[SimpleNamespace(block_table=SimpleNamespace(gpu=table))]
            )
        )
    worker = SimpleNamespace(model_runner=runner)

    assert warmup._block_table_widths_by_layer(worker) == {
        "model.layers.0.indexer.k_cache": 37
    }


def test_warmup_compiles_each_distinct_fused_decode_policy_once(monkeypatch):
    class FakeIndexer(nn.Module):
        def __init__(self, prefix: str):
            super().__init__()
            self.use_b12x_sparse_indexer = True
            self.num_q_heads = 2
            self.topk_tokens = 4
            self.head_dim = 128
            self.output_physical_slots = False
            self.topk_scores_buffer = None
            self.k_cache = SimpleNamespace(
                prefix=prefix,
                kv_cache=torch.empty((2, 64, 132), dtype=torch.uint8),
            )

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.indexers = nn.ModuleList(
                [FakeIndexer("indexer.0"), FakeIndexer("indexer.1")]
            )

    groups = [SimpleNamespace(layer_names=["indexer.0", "indexer.1"])]
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        block_tables=SimpleNamespace(
            input_block_tables=[torch.empty((4, 11), dtype=torch.int32)]
        ),
    )
    model = FakeModel()
    worker = SimpleNamespace(model_runner=runner, get_model=lambda: model)

    calls = []
    monkeypatch.setattr(warmup, "SparseAttnIndexer", FakeIndexer)
    monkeypatch.setattr(
        warmup,
        "_fused_decode_warmup_rows",
        lambda **kwargs: (1, 3, 16),
    )
    monkeypatch.setattr(warmup.current_platform, "fp8_dtype", lambda: torch.uint8)

    def run_paged_topk(**kwargs):
        calls.append(kwargs)
        assert not bool(kwargs["seq_lens"].any())
        return kwargs["topk_indices"]

    monkeypatch.setattr(warmup, "_run_b12x_paged_topk", run_paged_topk)

    assert warmup.warmup_b12x_sparse_indexer(worker) == 3
    assert [call["q_fp8"].shape[0] for call in calls] == [16, 1, 3]
    assert all(call["block_table"].shape[1] == 11 for call in calls)
    assert all(call["schedule_metadata"] is None for call in calls)
