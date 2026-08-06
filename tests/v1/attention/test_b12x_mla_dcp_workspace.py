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


def test_ckv_prefetch_reset_drops_old_cache_generation(monkeypatch):
    old_cache = torch.zeros((1, 64, 368), dtype=torch.uint8)
    old_event = object()
    monkeypatch.setattr(B12xMLASparseImpl, "_all_layer_kv_caches", [old_cache])
    monkeypatch.setattr(B12xMLASparseImpl, "_shared_gather_event", old_event)
    monkeypatch.setattr(B12xMLASparseImpl, "_shared_gather_buf_idx", 1)

    B12xMLASparseImpl.reset_kv_cache_binding_state()

    assert B12xMLASparseImpl._all_layer_kv_caches == []
    assert B12xMLASparseImpl._shared_gather_event is None
    assert B12xMLASparseImpl._shared_gather_buf_idx == 0


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


def test_sparse_profile_accounts_for_projected_fallback_before_workspace(monkeypatch):
    attn, workspace_enabled = _make_profile_attention(workspace_enabled=True)
    attn.dcp_project_before_merge_min_prefill_tokens = 512
    monkeypatch.setattr(envs, "VLLM_MEMORY_PROFILE_INCLUDE_ATTN", True)
    monkeypatch.setattr(
        envs, "VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE", workspace_enabled
    )

    unprojected = _estimate_dcp_ag_rs_transient_bytes(
        num_tokens=512,
        local_heads=11,
        dcp_world_size=6,
        q_head_dim=576,
        output_head_dim=512,
        kv_lora_rank=512,
        v_head_dim=256,
        project_before_merge=False,
    )
    projected = _estimate_dcp_ag_rs_transient_bytes(
        num_tokens=1024,
        local_heads=11,
        dcp_world_size=6,
        q_head_dim=576,
        output_head_dim=256,
        kv_lora_rank=512,
        v_head_dim=256,
        project_before_merge=True,
    )
    assert attn._get_sparse_memory_profile_bytes() == max(unprojected, projected)


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


def test_dcp_workspace_projection_accepts_partial_pitched_head_major_output(
    monkeypatch,
):
    max_batched = 8
    num_tokens = 5
    input_heads = 3
    kernel_heads = 4
    kv_lora_rank = 4
    v_head_dim = 2

    impl = object.__new__(B12xMLASparseImpl)
    impl._max_batched = max_batched
    impl._input_num_heads = input_heads
    impl._kernel_num_heads = kernel_heads
    impl._pad_heads = True
    impl.kv_lora_rank = kv_lora_rank
    impl.v_head_dim = v_head_dim

    q_workspace = torch.empty(
        max_batched, kernel_heads, kv_lora_rank, dtype=torch.bfloat16
    )
    dense_storage = torch.arange(
        input_heads * max_batched * kv_lora_rank, dtype=torch.float32
    ).to(torch.bfloat16)
    dense_workspace = dense_storage.view(
        input_heads, max_batched, kv_lora_rank
    ).transpose(0, 1)
    projected_nbytes = input_heads * num_tokens * v_head_dim * 2
    scratch_storage = torch.empty(projected_nbytes, dtype=torch.uint8)

    monkeypatch.setattr(
        impl, "_validate_dcp_prefill_workspace_contract", lambda _: None
    )
    monkeypatch.setattr(
        impl,
        "_borrow_workspace_parts",
        lambda: (q_workspace, dense_workspace, scratch_storage),
    )

    attn_out = dense_workspace[:num_tokens]
    lse = torch.zeros(num_tokens, input_heads, dtype=torch.float32)
    w_uv = torch.randn(input_heads, kv_lora_rank, v_head_dim, dtype=torch.bfloat16)
    expected = torch.bmm(attn_out.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)

    actual = impl.dcp_project_before_merge_in_workspace(attn_out, lse, w_uv)

    assert actual.movedim(0, 1).is_contiguous()
    torch.testing.assert_close(actual, expected)


def test_dcp_workspace_projection_accepts_aligned_head_capacity_pitch(monkeypatch):
    max_batched = 8
    num_tokens = 5
    input_heads = kernel_heads = 4
    kv_lora_rank = 4
    v_head_dim = 2

    impl = object.__new__(B12xMLASparseImpl)
    impl._max_batched = max_batched
    impl._input_num_heads = input_heads
    impl._kernel_num_heads = kernel_heads
    impl._pad_heads = False
    impl.kv_lora_rank = kv_lora_rank
    impl.v_head_dim = v_head_dim

    q_workspace = torch.empty(
        max_batched, kernel_heads, kv_lora_rank, dtype=torch.bfloat16
    )
    scratch_nbytes = input_heads * max_batched * kv_lora_rank * 2
    scratch_storage = torch.empty(scratch_nbytes, dtype=torch.uint8)
    full_output = (
        scratch_storage.view(torch.bfloat16)
        .view(input_heads, max_batched, kv_lora_rank)
        .transpose(0, 1)
    )
    full_output.copy_(
        torch.arange(full_output.numel(), dtype=torch.float32)
        .to(torch.bfloat16)
        .view_as(full_output)
    )

    monkeypatch.setattr(
        impl, "_validate_dcp_prefill_workspace_contract", lambda _: None
    )
    monkeypatch.setattr(
        impl,
        "_borrow_workspace_parts",
        lambda: (q_workspace, None, scratch_storage),
    )

    attn_out = full_output[:num_tokens]
    lse = torch.zeros(num_tokens, input_heads, dtype=torch.float32)
    w_uv = torch.randn(input_heads, kv_lora_rank, v_head_dim, dtype=torch.bfloat16)
    expected = torch.bmm(attn_out.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)

    actual = impl.dcp_project_before_merge_in_workspace(attn_out, lse, w_uv)

    assert attn_out.stride() == (kv_lora_rank, max_batched * kv_lora_rank, 1)
    assert actual.movedim(0, 1).is_contiguous()
    torch.testing.assert_close(actual, expected)
