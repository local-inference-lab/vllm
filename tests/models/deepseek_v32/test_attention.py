# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v32 import attention as deepseek_v32_attention


@pytest.mark.parametrize("full_ckv_dcp", [False, True])
def test_sparse_dcp_collectives_match_cache_mode(monkeypatch, full_ckv_dcp):
    attention = object.__new__(deepseek_v32_attention.DeepseekV32Attention)
    nn.Module.__init__(attention)
    attention.indexer = None
    attention.skip_topk = True
    attention.use_pcp = False
    attention.layer_name = "model.layers.0.self_attn.attn"
    attention._fp8_kv_needs_view = False
    attention._fp8_query = False
    attention.num_local_heads = 1
    attention.kv_lora_rank = 2
    attention.v_head_dim = 2
    attention.W_UV = torch.eye(2).unsqueeze(0)

    seq_lens = torch.tensor([1, 2], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        num_decodes=1,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        decode=SimpleNamespace(seq_lens=seq_lens[:1]),
    )
    attn_out = torch.arange(6, dtype=torch.float32).view(3, 1, 2)
    gathered_query = torch.empty((3, 2, 2))
    query_gather = Mock(return_value=gathered_query)
    combine = Mock(return_value=attn_out)
    attention.dcp_manager = SimpleNamespace(
        query_gather=query_gather,
        combine=combine,
    )
    attention.impl = SimpleNamespace(
        dcp_world_size=2,
        pcp_world_size=1,
        uses_full_ckv_dcp=Mock(return_value=full_ckv_dcp),
        forward_mqa=Mock(return_value=(attn_out, torch.zeros(3, 1))),
    )
    monkeypatch.setattr(
        deepseek_v32_attention,
        "get_attention_context",
        lambda layer_name: (metadata, None, torch.empty(0), torch.empty(0)),
    )

    output = torch.empty((3, 2))
    attention._sparse_indexer_and_attn(
        q_c=torch.empty((3, 2)),
        index_q_fp8=None,
        index_k=None,
        index_weights_out=None,
        kv_c=None,
        k_pe=None,
        ql_nope=torch.empty((3, 1, 1)),
        mqa_q=torch.empty((3, 1, 1)),
        output=output,
    )

    if full_ckv_dcp:
        query_gather.assert_not_called()
        combine.assert_not_called()
    else:
        query_gather.assert_called_once()
        assert attention.impl.forward_mqa.call_args.args[0] is gathered_query
        assert combine.call_args.kwargs["seq_lens"] is seq_lens
        assert combine.call_args.kwargs["query_start_loc"] is query_start_loc
    torch.testing.assert_close(output, attn_out.flatten(1))
