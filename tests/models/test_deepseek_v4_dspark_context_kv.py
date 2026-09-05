# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "module_name",
    [
        "vllm.models.deepseek_v4.nvidia.dspark",
        "vllm.models.deepseek_v4.amd.dspark",
    ],
)
def test_fp8_context_kv_insert_uses_caller_owned_q_output(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
):
    """DSpark context insertion must follow the fused operator's mutation contract."""
    if ".amd." in module_name:
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _device: SimpleNamespace(gcnArchName="gfx950"),
        )
    dspark = import_module(module_name)
    num_tokens = 3
    head_dim = 8
    padded_heads = 4
    block_size = 2
    kv = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    cache = torch.empty(2, block_size, 16, dtype=torch.uint8)
    cos_sin_cache = torch.randn(16, 4, dtype=torch.bfloat16)
    q_out = torch.empty(num_tokens, padded_heads, head_dim, dtype=torch.bfloat16)
    scratch_inputs: list[torch.Tensor] = []

    def get_q_padded_scratch(q: torch.Tensor) -> torch.Tensor:
        scratch_inputs.append(q)
        return q_out

    attn = SimpleNamespace(
        swa_cache_layer=SimpleNamespace(kv_cache=cache, block_size=block_size),
        rotary_emb=SimpleNamespace(cos_sin_cache=cos_sin_cache),
        n_local_heads=2,
        padded_heads=padded_heads,
        head_dim=head_dim,
        eps=1e-6,
        _get_q_padded_scratch=get_q_padded_scratch,
    )
    op_calls: list[tuple[object, ...]] = []

    def fused_insert(*args: object) -> None:
        op_calls.append(args)

    monkeypatch.setattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        fused_insert,
        raising=False,
    )

    dspark._insert_context_kv(attn, kv, positions, slot_mapping)

    assert len(scratch_inputs) == 1
    dummy_q = scratch_inputs[0]
    assert dummy_q.shape == (num_tokens, attn.n_local_heads, head_dim)
    assert torch.count_nonzero(dummy_q) == 0
    assert len(op_calls) == 1
    (
        actual_q,
        actual_kv,
        actual_q_out,
        actual_cache,
        actual_slots,
        actual_positions,
        actual_cos_sin,
        actual_eps,
        actual_block_size,
    ) = op_calls[0]
    assert actual_q is dummy_q
    assert actual_kv is kv
    assert actual_q_out is q_out
    assert isinstance(actual_cache, torch.Tensor)
    assert actual_cache.data_ptr() == cache.data_ptr()
    assert actual_cache.shape == (cache.shape[0], cache[0].numel())
    assert actual_slots is slot_mapping
    assert actual_positions is positions
    assert actual_cos_sin is cos_sin_cache
    assert actual_eps == attn.eps
    assert actual_block_size == block_size
