# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.device_communicators.custom_all_reduce as custom_all_reduce
import vllm.v1.attention.backends.mla.b12x_mla_sparse as b12x_mla_sparse
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.layers.b12x_contract import (
    b12x_backend_active_for_config,
    b12x_sparse_indexer_active_for_config,
    b12x_sparse_mla_active_for_config,
)
from vllm.model_executor.layers.fused_moe import b12x_moe
from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseMetadataBuilder,
    B12xMLASparseImpl,
    _default_decode_q_per_req,
    clear_b12x_mla_workspace_cache,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_b12x_backend_active_for_sparse_mla_attention():
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(backend="B12X_MLA_SPARSE"),
    )

    assert b12x_backend_active_for_config(vllm_config)
    assert b12x_sparse_mla_active_for_config(vllm_config)
    assert b12x_sparse_indexer_active_for_config(vllm_config)


def test_b12x_backend_active_for_moe_backend():
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="B12X"),
    )

    assert b12x_backend_active_for_config(vllm_config)
    assert not b12x_sparse_mla_active_for_config(vllm_config)
    assert not b12x_sparse_indexer_active_for_config(vllm_config)


def test_b12x_backend_active_for_glm51_sparse_mla_contract():
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        index_topk=2048,
        index_n_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )

    assert b12x_backend_active_for_config(vllm_config)
    assert b12x_sparse_mla_active_for_config(vllm_config)
    assert b12x_sparse_indexer_active_for_config(vllm_config)


def test_sparse_mla_implies_sparse_indexer():
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        index_topk=2048,
        index_n_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )

    assert b12x_sparse_mla_active_for_config(vllm_config)
    assert b12x_sparse_indexer_active_for_config(vllm_config)


def test_explicit_non_b12x_attention_backend_disables_sparse_mla_auto_contract():
    hf_config = SimpleNamespace(
        model_type="glm_moe_dsa",
        index_topk=2048,
        index_n_heads=32,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=192,
    )
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(backend="FLASHINFER_MLA_SPARSE"),
        model_config=SimpleNamespace(hf_text_config=hf_config),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )

    assert not b12x_sparse_mla_active_for_config(vllm_config)
    assert not b12x_sparse_indexer_active_for_config(vllm_config)


def test_b12x_decode_workspace_covers_short_extend_as_decode():
    assert (
        _default_decode_q_per_req(
            3,
            spec_extend_as_decode=True,
            spec_decode_max_q=8,
        )
        == 8
    )
    assert (
        _default_decode_q_per_req(
            3,
            spec_extend_as_decode=False,
            spec_decode_max_q=8,
        )
        == 4
    )


def test_b12x_joint_arena_preinstall_normalizes_moe_backend():
    calls = []

    class FakeModule:

        impl = SimpleNamespace(
            _preinstall_b12x_joint_arena=lambda: calls.append("preinstall")
        )

    class FakeModel:

        def named_modules(self):
            return [("model.layers.0.self_attn", FakeModule())]

    runner = object.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="B12X"),
    )
    runner.model = FakeModel()

    GPUModelRunner._preinstall_b12x_joint_attention_arenas(runner)

    assert calls == ["preinstall"]


def test_b12x_backend_inactive_for_non_b12x_config():
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(backend="FLASHINFER"),
        kernel_config=SimpleNamespace(moe_backend="flashinfer"),
    )

    assert not b12x_backend_active_for_config(vllm_config)
    assert not b12x_sparse_mla_active_for_config(vllm_config)
    assert not b12x_sparse_indexer_active_for_config(vllm_config)


def test_b12x_moe_requires_joint_pool_when_sparse_mla_is_active(monkeypatch):

    def failing_shared_pool(device):
        raise RuntimeError("no active shared arena")

    monkeypatch.setattr(b12x_moe, "_b12x_shared_arena_pool", failing_shared_pool)
    monkeypatch.setattr(b12x_moe, "_requires_b12x_joint_moe_pool", lambda: True)

    with pytest.raises(RuntimeError, match="cannot fall back"):
        b12x_moe._get_b12x_workspace_pool(SimpleNamespace(index=0))


def test_b12x_moe_requires_shared_arena_api_when_sparse_mla_is_active(
    monkeypatch,
):
    monkeypatch.setattr(b12x_moe, "_b12x_shared_arena_pool", None)
    monkeypatch.setattr(b12x_moe, "_requires_b12x_joint_moe_pool", lambda: True)

    with pytest.raises(RuntimeError, match="shared execution-lane arena"):
        b12x_moe._get_b12x_workspace_pool(SimpleNamespace(index=0))


def test_b12x_mla_requires_joint_attention_arena(monkeypatch):
    fake_b12x = types.ModuleType("b12x")
    fake_integration = types.ModuleType("b12x.integration")
    fake_mla = types.ModuleType("b12x.integration.mla")

    class FakeAttentionArena:

        @staticmethod
        def allocate(_attention_caps):
            raise AssertionError("standalone attention arena must not be used")

    class FakeJointArenaSpec:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fail_to_create_lane(_spec):
        raise RuntimeError("no shared lane")

    fake_mla.B12XAttentionArena = FakeAttentionArena
    fake_integration.B12XJointArenaSpec = FakeJointArenaSpec
    fake_integration.ensure_b12x_execution_lane_arena = fail_to_create_lane
    fake_b12x.integration = fake_integration

    monkeypatch.setitem(sys.modules, "b12x", fake_b12x)
    monkeypatch.setitem(sys.modules, "b12x.integration", fake_integration)
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)

    impl = object.__new__(B12xMLASparseImpl)
    impl.use_joint_arena = True

    with pytest.raises(RuntimeError, match="cannot fall back"):
        impl._allocate_b12x_attention_arena(
            attention_caps=SimpleNamespace(device="cuda"),
            moe_caps=SimpleNamespace(),
        )


def test_b12x_mla_requires_moe_arena_caps_for_joint_arena(monkeypatch):
    impl = object.__new__(B12xMLASparseImpl)
    impl.use_joint_arena = True
    impl.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    impl.hf_config = SimpleNamespace(
        n_routed_experts=128,
        hidden_size=8192,
        num_experts_per_tok=8,
    )
    impl._moe_backend_name = lambda _vllm_config: "b12x"

    with pytest.raises(RuntimeError, match="missing moe_intermediate_size"):
        impl._build_b12x_moe_arena_caps(
            device=SimpleNamespace(),
            dtype=SimpleNamespace(),
        )


def test_b12x_mla_rejects_non_arena_workspace(monkeypatch):
    clear_b12x_mla_workspace_cache()
    fake_mla = types.ModuleType("b12x.integration.mla")

    class FakeAttentionArenaCaps:
        pass

    class FakeWorkspaceContract:
        pass

    fake_mla.B12XAttentionArenaCaps = FakeAttentionArenaCaps
    fake_mla.B12XAttentionWorkspaceContract = FakeWorkspaceContract

    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)

    impl = object.__new__(B12xMLASparseImpl)
    impl.use_arena = False
    impl.use_joint_arena = False
    impl.vllm_config = SimpleNamespace()
    impl.decode_max_total_q = 1
    impl.arena_max_running_requests = 2
    impl.arena_extend_max_total_q = 8
    impl.arena_extend_max_batch = 2
    impl.arena_extend_max_kv_rows = 16
    impl.indexer_num_q_heads = 32
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.topk_indices_buffer = torch.empty((1, 4), dtype=torch.int32)
    impl.extend_use_cuda_graph = True
    impl.decode_use_cuda_graph = True
    impl.extend_max_chunks_per_row = 4
    impl.decode_workspace_per_layer = False
    impl.decode_workspace_ring = 1
    impl._moe_backend_name = lambda _vllm_config: ""

    q = torch.empty((2, 3, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 64, 576), dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="attention arena/workspace contract"):
        impl._get_workspace(
            "extend",
            q,
            kv_cache,
            max_kv_rows=4,
            page_table_width=4,
        )


def test_b12x_decode_graph_lse_requests_natural_scale(monkeypatch):
    calls = []

    def fake_decode_forward(**kwargs):
        calls.append(kwargs)
        out = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
        lse = torch.full((1, 1), 7.0, dtype=torch.float32)
        return out, lse

    fake_b12x = types.ModuleType("b12x")
    fake_integration = types.ModuleType("b12x.integration")
    fake_mla = types.ModuleType("b12x.integration.mla")
    fake_mla.sparse_mla_decode_forward = fake_decode_forward
    fake_b12x.integration = fake_integration
    fake_integration.mla = fake_mla
    monkeypatch.setitem(sys.modules, "b12x", fake_b12x)
    monkeypatch.setitem(sys.modules, "b12x.integration", fake_integration)
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)

    workspace = SimpleNamespace(use_cuda_graph=True)
    workspace.prepare_decode = lambda *_args: pytest.fail(
        "decode graph capture must not call prepare_decode"
    )
    metadata = SimpleNamespace(
        page_table_1=torch.zeros((1, 1), dtype=torch.int32),
        cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
        nsa_cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
    )

    _, lse = b12x_mla_sparse._sparse_mla_decode_forward_with_lse_vllm_metadata(
        q_all=torch.zeros((1, 1, 576), dtype=torch.bfloat16),
        kv_cache=torch.zeros((1, 1, 576), dtype=torch.uint8),
        metadata=metadata,
        workspace=workspace,
        sm_scale=1.0,
        v_head_dim=512,
    )

    assert calls[0]["workspace"] is workspace
    assert calls[0]["page_table_1"] is metadata.page_table_1
    assert calls[0]["cache_seqlens_int32"] is metadata.cache_seqlens_int32
    assert calls[0]["nsa_cache_seqlens_int32"] is metadata.nsa_cache_seqlens_int32
    assert calls[0]["return_lse"] is True
    assert calls[0]["lse_scale"] == "natural"
    assert lse.item() == 7.0


def test_b12x_decode_graph_passes_metadata_to_forward(monkeypatch):
    calls = []

    def fake_decode_forward(**kwargs):
        calls.append(kwargs)
        return torch.zeros((1, 1, 512), dtype=torch.bfloat16)

    fake_b12x = types.ModuleType("b12x")
    fake_integration = types.ModuleType("b12x.integration")
    fake_mla = types.ModuleType("b12x.integration.mla")
    fake_mla.sparse_mla_decode_forward = fake_decode_forward
    fake_b12x.integration = fake_integration
    fake_integration.mla = fake_mla
    monkeypatch.setitem(sys.modules, "b12x", fake_b12x)
    monkeypatch.setitem(sys.modules, "b12x.integration", fake_integration)
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)

    workspace = SimpleNamespace(use_cuda_graph=True)
    workspace.prepare_decode = lambda *_args: pytest.fail(
        "decode graph capture must not call prepare_decode"
    )
    metadata = SimpleNamespace(
        page_table_1=torch.zeros((1, 1), dtype=torch.int32),
        cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
        nsa_cache_seqlens_int32=torch.ones(1, dtype=torch.int32),
    )

    out = b12x_mla_sparse._sparse_mla_decode_forward_vllm_metadata(
        q_all=torch.zeros((1, 1, 576), dtype=torch.bfloat16),
        kv_cache=torch.zeros((1, 1, 576), dtype=torch.uint8),
        metadata=metadata,
        workspace=workspace,
        sm_scale=1.0,
        v_head_dim=512,
    )

    assert calls[0]["workspace"] is workspace
    assert calls[0]["page_table_1"] is metadata.page_table_1
    assert calls[0]["cache_seqlens_int32"] is metadata.cache_seqlens_int32
    assert calls[0]["nsa_cache_seqlens_int32"] is metadata.nsa_cache_seqlens_int32
    assert out.shape == (1, 1, 512)


def test_b12x_mla_extend_uses_attention_arena(monkeypatch):
    clear_b12x_mla_workspace_cache()
    fake_b12x = types.ModuleType("b12x")
    fake_integration = types.ModuleType("b12x.integration")
    fake_mla = types.ModuleType("b12x.integration.mla")
    caps_calls = []
    make_workspace_calls = []

    class FakeAttentionArenaCaps:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            caps_calls.append(kwargs)

    class FakeWorkspaceContract:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeWorkspace:

        @staticmethod
        def for_fixed_capacity(**_kwargs):
            raise AssertionError("fixed-capacity workspace must not be used")

    class FakeArena:

        def make_workspace(self, contract, *, use_cuda_graph):
            make_workspace_calls.append((contract.mode, use_cuda_graph))
            return SimpleNamespace()

    class FakeAttentionArena:

        @staticmethod
        def allocate(_attention_caps):
            return FakeArena()

    fake_mla.B12XAttentionArena = FakeAttentionArena
    fake_mla.B12XAttentionArenaCaps = FakeAttentionArenaCaps
    fake_mla.B12XAttentionWorkspace = FakeWorkspace
    fake_mla.B12XAttentionWorkspaceContract = FakeWorkspaceContract

    monkeypatch.setitem(sys.modules, "b12x", fake_b12x)
    monkeypatch.setitem(sys.modules, "b12x.integration", fake_integration)
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)

    impl = object.__new__(B12xMLASparseImpl)
    impl.use_arena = True
    impl.use_joint_arena = False
    impl.vllm_config = SimpleNamespace()
    impl.decode_max_total_q = 1
    impl.arena_max_running_requests = 2
    impl.arena_extend_max_total_q = 8
    impl.arena_extend_max_batch = 2
    impl.arena_extend_max_kv_rows = 16
    impl.indexer_num_q_heads = 32
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.topk_indices_buffer = torch.empty((1, 4), dtype=torch.int32)
    impl.extend_use_cuda_graph = True
    impl.decode_use_cuda_graph = True
    impl.extend_max_chunks_per_row = 4
    impl.extend_indexer_tile_logits_k_rows = 16384
    impl.scale = 0.7
    impl._moe_backend_name = lambda _vllm_config: ""

    q = torch.empty((2, 3, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 64, 576), dtype=torch.uint8)

    impl._get_workspace(
        "extend",
        q,
        kv_cache,
        max_kv_rows=4,
        page_table_width=4,
    )

    assert make_workspace_calls == [("extend", True)]
    assert caps_calls[0]["reserve_extend_indexer_logits"] is False
    assert caps_calls[0]["extend_indexer_tile_logits_k_rows"] == 16384
    assert caps_calls[0]["max_chunks_per_row"] == 4


def test_b12x_mla_decode_arena_sizes_mtp_rows(monkeypatch):
    clear_b12x_mla_workspace_cache()
    fake_b12x = types.ModuleType("b12x")
    fake_integration = types.ModuleType("b12x.integration")
    fake_mla = types.ModuleType("b12x.integration.mla")
    caps_calls = []
    contract_calls = []

    class FakeAttentionArenaCaps:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            caps_calls.append(kwargs)

    class FakeWorkspaceContract:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            contract_calls.append(kwargs)

    class FakeWorkspace:

        @staticmethod
        def for_fixed_capacity(**_kwargs):
            raise AssertionError("fixed-capacity workspace must not be used")

    class FakeArena:

        def make_workspace(self, contract, *, use_cuda_graph):
            assert use_cuda_graph
            return SimpleNamespace(contract=contract)

    class FakeAttentionArena:

        @staticmethod
        def allocate(_attention_caps):
            return FakeArena()

    fake_mla.B12XAttentionArena = FakeAttentionArena
    fake_mla.B12XAttentionArenaCaps = FakeAttentionArenaCaps
    fake_mla.B12XAttentionWorkspace = FakeWorkspace
    fake_mla.B12XAttentionWorkspaceContract = FakeWorkspaceContract

    monkeypatch.setitem(sys.modules, "b12x", fake_b12x)
    monkeypatch.setitem(sys.modules, "b12x.integration", fake_integration)
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)

    impl = object.__new__(B12xMLASparseImpl)
    impl.use_arena = True
    impl.use_joint_arena = False
    impl.vllm_config = SimpleNamespace()
    impl.decode_max_total_q = 4
    impl.arena_max_running_requests = 1
    impl.arena_extend_max_total_q = 8
    impl.arena_extend_max_batch = 1
    impl.arena_extend_max_kv_rows = 16
    impl.indexer_num_q_heads = 32
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.topk_indices_buffer = torch.empty((1, 4), dtype=torch.int32)
    impl.extend_use_cuda_graph = True
    impl.decode_use_cuda_graph = True
    impl.extend_max_chunks_per_row = 4
    impl.decode_workspace_per_layer = False
    impl.decode_workspace_ring = 1
    impl.scale = 0.7
    impl._moe_backend_name = lambda _vllm_config: ""

    q = torch.empty((4, 3, 576), dtype=torch.bfloat16)
    kv_cache = torch.empty((4, 64, 576), dtype=torch.uint8)

    impl._get_workspace(
        "decode",
        q,
        kv_cache,
        page_table_width=4,
    )

    assert caps_calls[0]["paged_max_q_rows"] == 4
    assert caps_calls[0]["paged_max_batch"] == 4
    assert caps_calls[0]["reserve_extend_indexer_logits"] is False
    assert contract_calls[0]["max_total_q"] == 4
    assert contract_calls[0]["max_batch"] == 4
    assert contract_calls[0]["max_paged_q_rows"] == 4


def test_b12x_joint_arena_preinstall_uses_shared_lane_capacity(monkeypatch):
    clear_b12x_mla_workspace_cache()
    fake_mla = types.ModuleType("b12x.integration.mla")
    caps_calls = []
    contract_calls = []

    class FakeWorkspaceContract:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            contract_calls.append(kwargs)

    class FakeArena:

        def make_workspace(self, contract, *, use_cuda_graph):
            assert use_cuda_graph
            return SimpleNamespace(contract=contract)

    fake_mla.B12XAttentionWorkspaceContract = FakeWorkspaceContract
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)
    monkeypatch.setattr(
        b12x_mla_sparse.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        b12x_mla_sparse,
        "_prime_b12x_sm_scale",
        lambda *_args, **_kwargs: None,
    )

    impl = object.__new__(B12xMLASparseImpl)
    impl.use_arena = True
    impl.use_joint_arena = True
    impl.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    impl.num_heads = 64
    impl.indexer_num_q_heads = 32
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.topk_indices_buffer = torch.empty((1, 4), dtype=torch.int32)
    impl.decode_max_total_q = 4
    impl.arena_max_running_requests = 1
    impl.arena_extend_max_total_q = 128
    impl.arena_extend_max_batch = 32
    impl.arena_extend_max_kv_rows = 202752
    impl.extend_max_chunks_per_row = 1
    impl.decode_use_cuda_graph = True
    impl.scale = 0.7
    impl._moe_backend_name = lambda _vllm_config: "b12x"
    impl._build_b12x_attention_arena_caps = (
        lambda **kwargs: caps_calls.append(kwargs) or SimpleNamespace()
    )
    impl._build_b12x_moe_arena_caps = lambda *_args: SimpleNamespace()
    impl._allocate_b12x_attention_arena = lambda *_args: FakeArena()
    impl._ensure_decode_split_chunk_config = lambda *_args, **_kwargs: None

    impl._preinstall_b12x_joint_arena()

    assert caps_calls[0]["caps_extend_max_total_q"] == 128
    assert caps_calls[0]["caps_extend_max_batch"] == 32
    assert caps_calls[0]["caps_extend_max_kv_rows"] == 202752
    assert caps_calls[0]["reserve_extend_indexer_logits"] is False
    assert caps_calls[0]["extend_indexer_tile_logits_k_rows"] == 32768
    assert caps_calls[0]["max_chunks_per_row"] == 1
    assert contract_calls[0]["max_total_q"] == 4
    assert contract_calls[0]["max_batch"] == 4
    assert contract_calls[0]["max_paged_q_rows"] == 4


def test_b12x_joint_arena_preinstall_caps_dcp_chunks(monkeypatch):
    clear_b12x_mla_workspace_cache()
    fake_mla = types.ModuleType("b12x.integration.mla")
    caps_calls = []

    class FakeWorkspaceContract:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeArena:

        def make_workspace(self, contract, *, use_cuda_graph):
            assert use_cuda_graph
            return SimpleNamespace(contract=contract)

    fake_mla.B12XAttentionWorkspaceContract = FakeWorkspaceContract
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)
    monkeypatch.setattr(
        b12x_mla_sparse.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        b12x_mla_sparse,
        "_prime_b12x_sm_scale",
        lambda *_args, **_kwargs: None,
    )

    impl = object.__new__(B12xMLASparseImpl)
    impl.use_arena = True
    impl.use_joint_arena = True
    impl.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=8),
    )
    impl.num_heads = 8
    impl.indexer_num_q_heads = 32
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.topk_indices_buffer = torch.empty((1, 4), dtype=torch.int32)
    impl.decode_max_total_q = 4
    impl.arena_max_running_requests = 1
    impl.arena_extend_max_total_q = 128
    impl.arena_extend_max_batch = 32
    impl.arena_extend_max_kv_rows = 202752
    impl.extend_max_chunks_per_row = 4
    impl.decode_use_cuda_graph = True
    impl.scale = 0.7
    impl._moe_backend_name = lambda _vllm_config: "b12x"
    impl._build_b12x_attention_arena_caps = (
        lambda **kwargs: caps_calls.append(kwargs) or SimpleNamespace()
    )
    impl._build_b12x_moe_arena_caps = lambda *_args: SimpleNamespace()
    impl._allocate_b12x_attention_arena = lambda *_args: FakeArena()
    impl._ensure_decode_split_chunk_config = lambda *_args, **_kwargs: None

    impl._preinstall_b12x_joint_arena()

    assert caps_calls[0]["num_q_heads"] == 64
    assert caps_calls[0]["max_chunks_per_row"] == 4


def test_b12x_joint_arena_preinstall_fails_when_required(monkeypatch):
    clear_b12x_mla_workspace_cache()
    fake_mla = types.ModuleType("b12x.integration.mla")

    class FakeWorkspaceContract:

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    fake_mla.B12XAttentionWorkspaceContract = FakeWorkspaceContract
    monkeypatch.setitem(sys.modules, "b12x.integration.mla", fake_mla)
    monkeypatch.setattr(
        b12x_mla_sparse.current_platform,
        "is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    impl = object.__new__(B12xMLASparseImpl)
    impl.use_arena = True
    impl.use_joint_arena = True
    impl.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    impl.num_heads = 64
    impl.indexer_num_q_heads = 32
    impl.head_size = 576
    impl.kv_lora_rank = 512
    impl.topk_indices_buffer = torch.empty((1, 4), dtype=torch.int32)
    impl.decode_max_total_q = 4
    impl.arena_extend_max_total_q = 128
    impl.arena_extend_max_batch = 32
    impl.arena_extend_max_kv_rows = 202752
    impl.decode_use_cuda_graph = True
    impl._moe_backend_name = lambda _vllm_config: "b12x"
    impl._build_b12x_attention_arena_caps = lambda **_kwargs: SimpleNamespace()
    impl._build_b12x_moe_arena_caps = lambda *_args: SimpleNamespace()
    impl._allocate_b12x_attention_arena = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("joint arena unavailable")
    )

    with pytest.raises(RuntimeError, match="joint arena unavailable"):
        impl._preinstall_b12x_joint_arena()


def test_b12x_mla_advertises_uniform_batch_cudagraph_support():
    assert (
        B12xMLASparseMetadataBuilder.get_cudagraph_support(
            SimpleNamespace(), SimpleNamespace()
        )
        == AttentionCGSupport.UNIFORM_BATCH
    )


def test_compilation_does_not_disable_b12x_cudagraph_by_backend_name():
    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        mode=CompilationMode.VLLM_COMPILE,
        use_inductor_graph_partition=True,
    )

    assert (
        compilation_config.resolve_cudagraph_mode_and_sizes(
            AttentionCGSupport.NEVER,
            "B12xMLASparseBackend",
        )
        == CUDAGraphMode.PIECEWISE
    )


def test_b12x_pcie_oneshot_skips_full_cudagraph_capture(monkeypatch):
    communicator = object.__new__(custom_all_reduce.CustomAllreduce)
    communicator.disabled = False
    communicator._fused_pcie_ops = None
    communicator._fused_pcie_ptr = 0
    communicator._ptr = 0
    communicator._pcie_runtime = SimpleNamespace(
        close=lambda: None,
        should_allreduce=lambda _inp: pytest.fail(
            "full cudagraph capture must use the NCCL fallback"
        )
    )

    monkeypatch.setattr(custom_all_reduce, "_is_full_cudagraph_runtime", lambda: True)

    assert not communicator.should_custom_ar(torch.empty(16, device="cpu"))


@pytest.mark.parametrize(
    "parallel_config",
    [
        SimpleNamespace(
            use_ep=True,
            ep_size=1,
            use_all2all_kernels=False,
            enable_eplb=False,
        ),
        SimpleNamespace(
            use_ep=False,
            ep_size=2,
            use_all2all_kernels=False,
            enable_eplb=False,
        ),
        SimpleNamespace(
            use_ep=False,
            ep_size=1,
            use_all2all_kernels=True,
            enable_eplb=False,
        ),
        SimpleNamespace(
            use_ep=False,
            ep_size=1,
            use_all2all_kernels=False,
            enable_eplb=True,
        ),
    ],
)
def test_b12x_moe_rejects_ep_all2all_and_eplb(parallel_config):
    assert not B12xExperts._supports_parallel_config(parallel_config)


def test_b12x_moe_decode_a16_env_keeps_prefill_nvfp4(monkeypatch):
    expert = object.__new__(B12xExperts)
    expert.layer_idx = 0
    expert.layer_name = "model.layers.0.mlp.experts"
    expert._warned_global_a16 = False

    monkeypatch.setenv("VLLM_B12X_MOE_DECODE_A16", "1")
    monkeypatch.setattr(b12x_moe, "is_forward_context_available", lambda: True)

    def set_metadata(attn_metadata):
        monkeypatch.setattr(
            b12x_moe,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=attn_metadata),
        )

    set_metadata(
        {"model.layers.0.self_attn": SimpleNamespace(num_prefills=0, num_decodes=4)}
    )
    assert expert._select_quant_mode() == "w4a16"

    set_metadata(
        {"model.layers.0.self_attn": SimpleNamespace(num_prefills=1, num_decodes=0)}
    )
    assert expert._select_quant_mode() == "nvfp4"

    set_metadata(
        {"model.layers.0.self_attn": SimpleNamespace(num_prefills=1, num_decodes=4)}
    )
    assert expert._select_quant_mode() == "nvfp4"

    monkeypatch.setattr(
        b12x_moe,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=None,
            cudagraph_runtime_mode=b12x_moe.CUDAGraphMode.PIECEWISE,
            batch_descriptor=SimpleNamespace(uniform=True),
        ),
    )
    assert expert._select_quant_mode() == "w4a16"
