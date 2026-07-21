# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    _can_use_b12x_dcp_prefill_workspace,
)
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
from vllm.v1.attention.ops import common


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("enabled", False),
        ("project_before_merge", False),
        ("dcp_use_b12x", True),
        ("num_tokens", 1024),
        ("max_num_tokens", 1024),
        ("non_dbo_workspace", False),
        ("is_sparse_impl", False),
        ("backend_name", "FLASHINFER_MLA"),
        ("is_capturing", True),
    ],
)
def test_dcp_workspace_gate_rejects_unsupported_profiles(override, value):
    profile = {
        "enabled": True,
        "project_before_merge": True,
        "dcp_use_b12x": False,
        "num_tokens": 1025,
        "max_num_tokens": 3072,
        "non_dbo_workspace": True,
        "is_sparse_impl": True,
        "backend_name": "B12X_MLA_SPARSE",
        "is_capturing": False,
    }
    profile[override] = value

    assert not _can_use_b12x_dcp_prefill_workspace(**profile)


@pytest.mark.parametrize(
    ("num_tokens", "max_num_tokens"),
    [(1025, 2048), (2048, 2048), (3072, 8192), (8192, 8192)],
)
def test_dcp_workspace_gate_accepts_valid_rows(num_tokens, max_num_tokens):
    assert _can_use_b12x_dcp_prefill_workspace(
        enabled=True,
        project_before_merge=True,
        dcp_use_b12x=False,
        num_tokens=num_tokens,
        max_num_tokens=max_num_tokens,
        non_dbo_workspace=True,
        is_sparse_impl=True,
        backend_name="B12X_MLA_SPARSE",
        is_capturing=False,
    )


@pytest.mark.parametrize("world_size", [2, 3, 4, 6, 8])
def test_cp_lse_ag_out_rs_into_preserves_borrowed_output(monkeypatch, world_size):
    rank = world_size - 1
    corrected = torch.arange(world_size * 16, dtype=torch.bfloat16).view(
        1, world_size, 16
    )
    corrected_lse = torch.arange(world_size, dtype=torch.float32).view(1, world_size)
    borrowed = torch.empty((1, 1, 16), dtype=torch.bfloat16)

    monkeypatch.setattr(
        common,
        "_cp_lse_common",
        lambda *args, **kwargs: (corrected, corrected_lse),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    class FakeGroup:
        rank_in_group = rank
        world_size: int

        def reduce_scatter_into(self, input_, output, dim):
            assert input_ is corrected
            assert output is borrowed
            assert dim == 1
            output.copy_(input_[:, rank : rank + 1])
            return output

    group = FakeGroup()
    group.world_size = world_size
    output, lse = common.cp_lse_ag_out_rs_into(
        torch.empty_like(corrected),
        torch.empty_like(corrected_lse),
        group,
        output_provider=lambda value: borrowed,
        return_lse=True,
    )

    assert output is borrowed
    assert torch.equal(output, corrected[:, rank : rank + 1])
    assert torch.equal(lse, corrected_lse[:, rank : rank + 1])


def test_cp_lse_ag_out_rs_requests_head_major_output(monkeypatch):
    corrected_storage = torch.arange(8 * 3 * 16, dtype=torch.bfloat16).view(
        8, 3, 16
    )
    corrected = corrected_storage.movedim(0, 1)
    corrected_lse = torch.zeros(3, 8, dtype=torch.float32)

    monkeypatch.setattr(
        common,
        "_cp_lse_common",
        lambda *args, **kwargs: (corrected, corrected_lse),
    )

    class FakeGroup:
        rank_in_group = 0
        world_size = 2

        def reduce_scatter_head_major(self, input_, dim):
            assert input_ is corrected
            assert dim == 1
            storage = torch.empty(4, 3, 16, dtype=input_.dtype)
            output = storage.movedim(0, 1)
            output.copy_(input_[:, :4])
            return output

    output = common.cp_lse_ag_out_rs(
        corrected,
        corrected_lse,
        FakeGroup(),
        head_major_output=True,
    )

    assert output.stride() == (16, 3 * 16, 1)
    torch.testing.assert_close(output, corrected[:, :4])


@pytest.mark.parametrize(
    ("tp_size", "dcp_size", "local_heads", "input_heads", "kernel_heads"),
    [
        (4, 4, 16, 64, 64),
        (6, 2, 11, 22, 24),
        (6, 3, 11, 33, 40),
        (6, 6, 11, 66, 72),
        (8, 2, 8, 16, 16),
        (8, 4, 8, 32, 32),
        (8, 8, 8, 64, 64),
    ],
)
def test_dcp_workspace_contract_accepts_validated_topologies(
    monkeypatch,
    tp_size,
    dcp_size,
    local_heads,
    input_heads,
    kernel_heads,
):
    impl = object.__new__(B12xMLASparseImpl)
    impl._max_batched = 8192
    impl.dcp_workspace_non_dbo = True
    impl.tp_world_size = tp_size
    impl.dcp_world_size = dcp_size
    impl.num_heads = local_heads
    impl._input_num_heads = input_heads
    impl._kernel_num_heads = kernel_heads
    impl._pad_heads = kernel_heads != input_heads
    impl.q_head_dim = 576
    impl.kv_lora_rank = 512
    impl.v_head_dim = 256
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    impl._validate_dcp_prefill_workspace_contract(2048)


def test_dcp_workspace_contract_rejects_unvalidated_topology(monkeypatch):
    impl = object.__new__(B12xMLASparseImpl)
    impl._max_batched = 8192
    impl.dcp_workspace_non_dbo = True
    impl.tp_world_size = 6
    impl.dcp_world_size = 4
    impl.num_heads = 11
    impl._input_num_heads = 44
    impl._kernel_num_heads = 48
    impl._pad_heads = True
    impl.q_head_dim = 576
    impl.kv_lora_rank = 512
    impl.v_head_dim = 256
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with pytest.raises(RuntimeError, match="unsupported topology or geometry"):
        impl._validate_dcp_prefill_workspace_contract(2048)
