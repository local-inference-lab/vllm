# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Order of the latent query projection against the sparse indexer.

On the bf16 query path the attention module issues the W_UK projection of
the query after the indexer launch, so the selection sort that the B12X
indexer runs on a side stream between the indexer and the attention kernels
has main-stream work to overlap. On the fp8 query path the projection must
precede ``fused_q``, which packs it.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v32 import attention as attention_module
from vllm.models.deepseek_v32.attention import DeepseekV32Attention


def _module(events: list[str], *, fp8_query: bool) -> DeepseekV32Attention:
    module = DeepseekV32Attention.__new__(DeepseekV32Attention)
    module.skip_topk = False
    module.use_pcp = False
    module.dcp_manager = None
    module._vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1
        )
    )
    module._dense_mha_metadata_layer_name = "dense"
    module.layer_name = "layer"
    module._fp8_query = fp8_query
    module.W_UK_T = torch.ones(2, 3, 4)

    def run_indexer(*args, **kwargs):
        events.append("indexer")

    module.indexer = SimpleNamespace(run_indexer=run_indexer)

    def latent_query(q_nope):
        events.append("latent")
        return torch.zeros(q_nope.shape[0], q_nope.shape[1], 4)

    module._latent_query = latent_query  # type: ignore[method-assign]
    return module


@pytest.mark.parametrize("fp8_query", [False, True])
def test_latent_query_runs_after_the_indexer_on_the_bf16_path(
    monkeypatch: pytest.MonkeyPatch, fp8_query: bool
) -> None:
    events: list[str] = []
    module = _module(events, fp8_query=fp8_query)
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda layer_name: (None, None, None, None),
    )
    q_nope = torch.zeros(5, 2, 3)
    output = torch.ones(5, 8)
    ql_nope = module._latent_query(q_nope) if fp8_query else None
    events.clear()
    module._sparse_indexer_and_attn(
        torch.zeros(5, 6),
        torch.zeros(5, 1, 1),
        None,
        torch.zeros(5, 1),
        None,
        None,
        ql_nope,
        q_nope,
        torch.zeros(5, 2, 4),
        output,
    )
    if fp8_query:
        assert events == ["indexer"]
    else:
        assert events == ["indexer", "latent"]
    assert torch.count_nonzero(output) == 0
