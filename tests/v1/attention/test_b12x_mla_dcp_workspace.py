# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import envs
from vllm.model_executor.layers.attention.mla_attention import (
    MLAAttention,
    _can_use_b12x_dcp_prefill_workspace,
    _estimate_dcp_ag_rs_transient_bytes,
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


def _make_profile_attention(*, workspace_enabled: bool, pure_a2a: bool = False):
    class Backend:
        @staticmethod
        def get_name():
            return "B12X_MLA_SPARSE"

    class Impl:
        dcp_world_size = 6
        dcp_workspace_non_dbo = True
        is_sparse = True
        _max_batched = 4096

    attn = object.__new__(MLAAttention)
    attn.attn_backend = Backend
    attn.impl = Impl()
    attn.num_heads = 11
    attn.kv_lora_rank = 512
    attn.qk_rope_head_dim = 64
    attn.v_head_dim = 256
    attn.dcp_project_before_merge = True
    attn.dcp_project_before_merge_min_prefill_tokens = 1024
    attn.dcp_a2a = True
    attn.dcp_a2a_max_tokens = 0 if pure_a2a else 256
    attn.dcp_a2a_large_backend = "ag_rs"
    return attn, workspace_enabled


def test_sparse_profile_reserves_largest_non_workspace_ag_rs_batch(monkeypatch):
    attn, workspace_enabled = _make_profile_attention(workspace_enabled=True)
    monkeypatch.setattr(envs, "VLLM_MEMORY_PROFILE_INCLUDE_ATTN", True)
    monkeypatch.setattr(
        envs, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", workspace_enabled
    )

    expected = _estimate_dcp_ag_rs_transient_bytes(
        num_tokens=1024,
        local_heads=11,
        dcp_world_size=6,
        q_head_dim=576,
        output_head_dim=512,
        kv_lora_rank=512,
        v_head_dim=256,
        project_before_merge=False,
    )
    assert attn._get_sparse_memory_profile_bytes() == expected


def test_sparse_profile_accounts_for_projected_fallback_without_workspace(monkeypatch):
    attn, workspace_enabled = _make_profile_attention(workspace_enabled=False)
    monkeypatch.setattr(envs, "VLLM_MEMORY_PROFILE_INCLUDE_ATTN", True)
    monkeypatch.setattr(
        envs, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", workspace_enabled
    )

    unprojected = _estimate_dcp_ag_rs_transient_bytes(
        num_tokens=1024,
        local_heads=11,
        dcp_world_size=6,
        q_head_dim=576,
        output_head_dim=512,
        kv_lora_rank=512,
        v_head_dim=256,
        project_before_merge=False,
    )
    projected = _estimate_dcp_ag_rs_transient_bytes(
        num_tokens=4096,
        local_heads=11,
        dcp_world_size=6,
        q_head_dim=576,
        output_head_dim=256,
        kv_lora_rank=512,
        v_head_dim=256,
        project_before_merge=True,
    )
    assert attn._get_sparse_memory_profile_bytes() == max(unprojected, projected)


def test_sparse_profile_accounts_for_unprojected_full_batch(monkeypatch):
    attn, _ = _make_profile_attention(workspace_enabled=False)
    attn.dcp_project_before_merge = False
    monkeypatch.setattr(envs, "VLLM_MEMORY_PROFILE_INCLUDE_ATTN", True)
    monkeypatch.setattr(envs, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", False)

    expected = _estimate_dcp_ag_rs_transient_bytes(
        num_tokens=4096,
        local_heads=11,
        dcp_world_size=6,
        q_head_dim=576,
        output_head_dim=512,
        kv_lora_rank=512,
        v_head_dim=256,
        project_before_merge=False,
    )
    assert attn._get_sparse_memory_profile_bytes() == expected


def test_sparse_profile_skips_pure_a2a(monkeypatch):
    attn, _ = _make_profile_attention(workspace_enabled=True, pure_a2a=True)
    monkeypatch.setattr(envs, "VLLM_MEMORY_PROFILE_INCLUDE_ATTN", True)
    monkeypatch.setattr(envs, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", True)

    assert attn._get_sparse_memory_profile_bytes() == 0


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
