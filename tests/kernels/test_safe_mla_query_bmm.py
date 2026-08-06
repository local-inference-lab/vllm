# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _require_safe_mla_query_bmm():
    if not hasattr(torch.ops._C, "safe_mla_query_bmm"):
        pytest.skip("safe_mla_query_bmm is not built")


@pytest.mark.parametrize(
    "heads,tokens,q_dim",
    [
        (8, 1, 512),
        (8, 6, 512),
        (8, 11, 512),
        (11, 6, 512),
        (16, 6, 512),
        (8, 8192, 192),
    ],
)
def test_safe_mla_query_bmm_matches_torch_bmm(heads: int, tokens: int, q_dim: int):
    _require_safe_mla_query_bmm()
    device = torch.device("cuda")
    rope_dim = 64
    latent_dim = 512
    torch.manual_seed(0)

    query_storage = torch.randn(
        tokens, heads, q_dim + rope_dim, dtype=torch.bfloat16, device=device
    )
    query = query_storage[..., :q_dim].transpose(0, 1)
    weight = torch.randn(heads, q_dim, latent_dim, dtype=torch.bfloat16, device=device)
    output = torch.empty(heads, tokens, latent_dim, dtype=torch.bfloat16, device=device)

    assert not query.is_contiguous()
    torch.ops._C.safe_mla_query_bmm(query, weight, output)
    expected = torch.bmm(query.contiguous(), weight)

    torch.testing.assert_close(output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_safe_mla_query_bmm_cuda_graph_replay():
    _require_safe_mla_query_bmm()
    device = torch.device("cuda")
    heads = 8
    tokens = 6
    q_dim = 512
    latent_dim = 512
    torch.manual_seed(1)

    query_storage = torch.randn(
        tokens, heads, q_dim + 64, dtype=torch.bfloat16, device=device
    )
    query = query_storage[..., :q_dim].transpose(0, 1)
    weight = torch.randn(heads, q_dim, latent_dim, dtype=torch.bfloat16, device=device)
    output = torch.empty(heads, tokens, latent_dim, dtype=torch.bfloat16, device=device)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C.safe_mla_query_bmm(query, weight, output)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.safe_mla_query_bmm(query, weight, output)

    graph.replay()
    graph.replay()
    expected = torch.bmm(query.contiguous(), weight)

    torch.testing.assert_close(output.float(), expected.float(), rtol=5e-2, atol=5e-2)
