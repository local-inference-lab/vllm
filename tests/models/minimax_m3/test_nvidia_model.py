# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.fused_allreduce_gemma_rms_norm as fused_ar_norm
import vllm.model_executor.parameter as parameter_module
import vllm.models.minimax_m3.nvidia.model as minimax_model
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.config.virtual_tp import VIRTUAL_TP_PLAN_ATTR
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MinimaxM3IndexerQKParallelLinear,
    MinimaxM3QKVParallelLinear,
    MinimaxM3QKVParallelLinearWithIndexer,
)
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.models.minimax_m3.common.indexer import _get_minimax_m3_indexer_num_heads
from vllm.models.minimax_m3.nvidia import sparse_attention_b12x
from vllm.models.minimax_m3.nvidia.model import (
    MiniMAXGemmaRMSNorm,
    MiniMaxM3Model,
    MiniMaxM3SparseAttention,
    MiniMaxM3SparseForCausalLM,
    _enable_minimax_m3_torch_compile,
    _run_minimax_m3_qknorm_rope_kv_insert,
    _should_split_mxfp8_indexer_projection,
    minimax_m3_sparse_attention_with_output,
    minimax_m3_sparse_kv_cache_update,
)


class _FakeParam:
    def __init__(self, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.calls: list[tuple[torch.Tensor, object | None]] = []

    def weight_loader(
        self,
        param: "_FakeParam",
        loaded_weight: torch.Tensor,
        shard_id: object | None = None,
    ) -> None:
        assert param is self
        self.calls.append((loaded_weight, shard_id))


def _fake_minimax_model(params: dict[str, _FakeParam]) -> MiniMaxM3Model:
    model = object.__new__(MiniMaxM3Model)
    nn.Module.__init__(model)
    model.get_expert_mapping = lambda: []  # type: ignore[method-assign]
    model.named_parameters = lambda: iter(params.items())  # type: ignore[method-assign]
    return model


class _RecordingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[
            tuple[
                torch.Tensor | None,
                torch.Tensor,
                object | None,
                torch.Tensor | None,
            ]
        ] = []

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: object | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append((input_ids, positions, intermediate_tensors, inputs_embeds))
        if inputs_embeds is None:
            return torch.empty(0)
        return inputs_embeds


def _gemma_rms_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    weight_fp32 = weight.float() + 1.0
    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return (x_fp32 * torch.rsqrt(variance + eps) * weight_fp32).to(x.dtype)


def test_minimax_m3_causallm_forwards_stable_embed_contract() -> None:
    lm = object.__new__(MiniMaxM3SparseForCausalLM)
    nn.Module.__init__(lm)
    backbone = _RecordingBackbone()
    lm.model = backbone

    input_ids = torch.tensor([3, 1, 4], dtype=torch.long)
    positions = torch.arange(input_ids.numel(), dtype=torch.long)

    output = MiniMaxM3SparseForCausalLM.forward(lm, input_ids, positions)

    assert len(backbone.calls) == 1
    seen_input_ids, seen_positions, seen_intermediate, seen_embeds = backbone.calls[0]
    assert seen_input_ids is None
    assert seen_intermediate is None
    assert seen_embeds is not None
    torch.testing.assert_close(seen_positions, positions)
    torch.testing.assert_close(seen_embeds, input_ids.to(torch.float32).unsqueeze(-1))
    torch.testing.assert_close(output, seen_embeds)


def test_minimax_m3_causallm_preserves_supplied_inputs_embeds() -> None:
    lm = object.__new__(MiniMaxM3SparseForCausalLM)
    nn.Module.__init__(lm)
    backbone = _RecordingBackbone()
    lm.model = backbone

    positions = torch.arange(3, dtype=torch.long)
    inputs_embeds = torch.randn(3, 8)

    output = MiniMaxM3SparseForCausalLM.forward(
        lm,
        input_ids=torch.tensor([3, 1, 4], dtype=torch.long),
        positions=positions,
        inputs_embeds=inputs_embeds,
    )

    assert len(backbone.calls) == 1
    seen_input_ids, seen_positions, seen_intermediate, seen_embeds = backbone.calls[0]
    assert seen_input_ids is None
    assert seen_intermediate is None
    assert seen_embeds is inputs_embeds
    torch.testing.assert_close(seen_positions, positions)
    torch.testing.assert_close(output, inputs_embeds)


def test_minimax_m3_torch_compile_is_env_gated(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    monkeypatch.delenv("VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE", raising=False)
    assert _enable_minimax_m3_torch_compile(VllmConfig()) is False

    monkeypatch.setenv("VLLM_MINIMAX_M3_ENABLE_TORCH_COMPILE", "1")
    assert _enable_minimax_m3_torch_compile(VllmConfig()) is True

    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1")
    assert _enable_minimax_m3_torch_compile(VllmConfig()) is True

    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    assert _enable_minimax_m3_torch_compile(VllmConfig()) is False


@torch.inference_mode()
def test_minimax_m3_norm_uses_vllm_gemma_semantics(default_vllm_config) -> None:
    assert issubclass(MiniMAXGemmaRMSNorm, GemmaRMSNorm)

    layer = MiniMAXGemmaRMSNorm(16, eps=1e-6)
    layer.weight.data.normal_(mean=0.0, std=0.1)

    x = torch.randn(4, 16)
    out = layer(x)
    torch.testing.assert_close(
        out, _gemma_rms_norm_ref(x, layer.weight, layer.variance_epsilon)
    )

    residual = torch.randn_like(x)
    out_with_residual, new_residual = layer(x.clone(), residual.clone())
    ref_residual = x + residual
    torch.testing.assert_close(new_residual, ref_residual)
    torch.testing.assert_close(
        out_with_residual,
        _gemma_rms_norm_ref(ref_residual, layer.weight, layer.variance_epsilon),
    )


@torch.inference_mode()
def test_minimax_m3_norm_traces_as_fullgraph(default_vllm_config) -> None:
    layer = MiniMAXGemmaRMSNorm(16, eps=1e-6)
    layer.weight.data.normal_(mean=0.0, std=0.1)
    x = torch.randn(4, 16)
    residual = torch.randn_like(x)

    def fn(
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return layer(hidden_states, residual)

    compiled = torch.compile(
        fn,
        fullgraph=True,
        dynamic=False,
        backend="eager",
    )
    out, new_residual = compiled(x, residual)
    ref_residual = x + residual

    torch.testing.assert_close(new_residual, ref_residual)
    torch.testing.assert_close(
        out,
        _gemma_rms_norm_ref(ref_residual, layer.weight, layer.variance_epsilon),
    )


@torch.inference_mode()
def test_fused_allreduce_norm_compile_path_skips_flashinfer_probe(
    default_vllm_config,
    monkeypatch,
) -> None:
    layer = MiniMAXGemmaRMSNorm(16, eps=1e-6)
    x = torch.randn(4, 16)
    residual = torch.randn_like(x)

    def fail_flashinfer_probe(*args, **kwargs):
        raise AssertionError("FlashInfer probe should be outside compile path")

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        fused_ar_norm, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        fused_ar_norm, "tensor_model_parallel_all_reduce", lambda tensor: tensor + 1
    )
    monkeypatch.setattr(fused_ar_norm, "_can_use_flashinfer", fail_flashinfer_probe)

    out, new_residual = fused_ar_norm.fused_allreduce_gemma_rms_norm(x, residual, layer)
    ref_residual = x + 1 + residual

    torch.testing.assert_close(new_residual, ref_residual)
    torch.testing.assert_close(
        out,
        _gemma_rms_norm_ref(ref_residual, layer.weight, layer.variance_epsilon),
    )


def test_minimax_m3_qknorm_rope_uses_custom_op_while_compiling(
    monkeypatch,
) -> None:
    calls = []

    def record_custom_op(*args):
        calls.append(args)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        torch.ops.vllm,
        "minimax_m3_qknorm_rope_kv_insert",
        record_custom_op,
    )

    qkv = torch.empty(2, 64)
    q_norm_weight = torch.empty(16)
    k_norm_weight = torch.empty(16)
    cos_sin_cache = torch.empty(4, 32)
    positions = torch.arange(2)
    index_q_norm_weight = torch.empty(16)
    index_k_norm_weight = torch.empty(16)
    slot_mapping = torch.arange(2, dtype=torch.int32)
    index_slot_mapping = torch.arange(2, dtype=torch.int32)
    kv_cache = torch.empty(1, 2, 2, 1, 16)
    index_cache = torch.empty(1, 2, 16)
    q_out = torch.empty(2, 16)
    index_q_out = torch.empty(2, 16)

    _run_minimax_m3_qknorm_rope_kv_insert(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        1,
        1,
        16,
        1e-6,
        index_q_norm_weight,
        index_k_norm_weight,
        1,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        16,
        q_out,
        index_q_out,
    )

    assert len(calls) == 1
    seen_args = calls[0]
    assert seen_args[0] is qkv
    assert seen_args[14] is kv_cache
    assert seen_args[15] is index_cache
    assert seen_args[17] is q_out
    assert seen_args[18] is index_q_out


def test_minimax_m3_sparse_attention_custom_op_is_default_split() -> None:
    assert (
        "vllm::minimax_m3_sparse_attention_with_output"
        in CompilationConfig()._attention_ops
    )


def test_minimax_m3_sparse_kv_cache_update_is_default_split() -> None:
    config = CompilationConfig(mode=CompilationMode.VLLM_COMPILE)

    config.set_splitting_ops_for_v1(all2all_backend="")

    assert config.splitting_ops is not None
    assert "vllm::minimax_m3_sparse_kv_cache_update" in config.splitting_ops
    assert config.splitting_ops_contain_kv_cache_update()


def test_minimax_m3_sparse_kv_cache_update_uses_forward_context() -> None:
    calls = []
    layer = object.__new__(MiniMaxM3SparseAttention)
    nn.Module.__init__(layer)
    layer.layer_name = "layers.3.self_attn"
    main_kv_cache = torch.empty(1, 2, 2, 1, 4)
    index_kv_cache = torch.empty(1, 2, 3)
    layer.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(
            prefix="layers.3.self_attn.index_cache",
            kv_cache=index_kv_cache,
        )
    )
    layer.kv_cache = main_kv_cache

    def record_insert_kv(
        key: torch.Tensor,
        value: torch.Tensor,
        index_key: torch.Tensor,
        main_kv_cache: torch.Tensor,
        index_kv_cache: torch.Tensor,
        main_slot_mapping: torch.Tensor,
        index_slot_mapping: torch.Tensor,
    ) -> None:
        calls.append(
            (
                key,
                value,
                index_key,
                main_kv_cache,
                index_kv_cache,
                main_slot_mapping,
                index_slot_mapping,
            )
        )

    layer._insert_kv_into_caches = record_insert_kv  # type: ignore[method-assign]
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context[layer.layer_name] = layer
    key = torch.randn(2, 4)
    value = torch.randn(2, 4)
    index_key = torch.randn(2, 3)
    main_slot_mapping = torch.tensor([0, 1], dtype=torch.int32)
    index_slot_mapping = torch.tensor([2, 3], dtype=torch.int32)
    slot_mapping = {
        layer.layer_name: main_slot_mapping,
        layer.indexer.index_cache.prefix: index_slot_mapping,
    }

    with set_forward_context({}, vllm_config, slot_mapping=slot_mapping):
        dummy = minimax_m3_sparse_kv_cache_update(
            key,
            value,
            index_key,
            layer.layer_name,
        )

    assert dummy.shape == (0,)
    assert dummy.dtype == main_kv_cache.dtype
    assert dummy.device == main_kv_cache.device
    assert calls == [
        (
            key,
            value,
            index_key,
            main_kv_cache,
            index_kv_cache,
            main_slot_mapping,
            index_slot_mapping,
        )
    ]


def test_minimax_m3_sparse_attention_compile_path_avoids_python_slot_mapping(
    monkeypatch,
) -> None:
    class FakeQKVProj(nn.Module):
        def __init__(self, qkv: torch.Tensor) -> None:
            super().__init__()
            self.qkv = qkv

        def forward(self, hidden_states: torch.Tensor):
            return self.qkv[: hidden_states.shape[0]].clone(), None

    class FakeOProj(nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            return hidden_states, None

    def fail_forward_context():
        raise AssertionError("compile path must not read slot_mapping in Python")

    qknorm_calls = []
    kv_update_calls = []
    sparse_attn_calls = []

    def record_qknorm_rope_kv_insert(*args) -> None:
        qkv = args[0]
        q_out = args[17]
        index_q_out = args[18]
        q_out.copy_(qkv[:, : q_out.shape[1]])
        index_q_out.copy_(qkv[:, -index_q_out.shape[1] :])
        qknorm_calls.append(args)

    def record_kv_cache_update(*args) -> torch.Tensor:
        kv_update_calls.append(args)
        return torch.empty(0, device=args[0].device, dtype=args[0].dtype)

    def record_sparse_attention_with_output(*args) -> None:
        query = args[0]
        output = args[4]
        output.copy_(query)
        sparse_attn_calls.append(args)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(minimax_model, "get_forward_context", fail_forward_context)
    monkeypatch.setattr(
        torch.ops.vllm,
        "minimax_m3_qknorm_rope_kv_insert",
        record_qknorm_rope_kv_insert,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "minimax_m3_sparse_kv_cache_update",
        record_kv_cache_update,
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "minimax_m3_sparse_attention_with_output",
        record_sparse_attention_with_output,
    )

    layer = object.__new__(MiniMaxM3SparseAttention)
    nn.Module.__init__(layer)
    layer.layer_name = "layers.3.self_attn"
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.num_idx_heads = 1
    layer.head_dim = 2
    layer.idx_head_dim = 3
    layer.q_size = 4
    layer.index_q_size = 3
    layer.hidden_size = 4
    layer._fp8_kv = False
    layer.q_norm = SimpleNamespace(
        weight=torch.empty(layer.head_dim),
        variance_epsilon=1e-6,
    )
    layer.k_norm = SimpleNamespace(weight=torch.empty(layer.head_dim))
    layer.index_q_norm = SimpleNamespace(weight=torch.empty(layer.idx_head_dim))
    layer.index_k_norm = SimpleNamespace(weight=torch.empty(layer.idx_head_dim))
    layer.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.empty(8, layer.head_dim * 2),
        rotary_dim=layer.head_dim,
    )
    layer.indexer = SimpleNamespace(
        index_cache=SimpleNamespace(
            prefix="layers.3.self_attn.index_cache",
            kv_cache=torch.empty(1, 2, layer.idx_head_dim),
        )
    )
    layer.kv_cache = torch.empty(1, 2, 2, layer.num_kv_heads, layer.head_dim)
    total_qkv_size = (
        layer.q_size
        + 2 * layer.num_kv_heads * layer.head_dim
        + layer.index_q_size
        + layer.num_idx_heads * layer.idx_head_dim
    )
    layer.qkv_proj = FakeQKVProj(torch.arange(2 * total_qkv_size).view(2, -1).float())
    layer.indexer_qk_proj = None
    layer.o_proj = FakeOProj()

    positions = torch.arange(2)
    hidden_states = torch.empty(2, layer.hidden_size)

    output = MiniMaxM3SparseAttention.forward(layer, positions, hidden_states)

    assert len(qknorm_calls) == 1
    qknorm_args = qknorm_calls[0]
    assert qknorm_args[12] is None
    assert qknorm_args[13] is None
    assert qknorm_args[14] is None
    assert qknorm_args[15] is None
    assert qknorm_args[16] == 0
    assert len(kv_update_calls) == 1
    assert minimax_model._resolve_layer_name(kv_update_calls[0][3]) == layer.layer_name
    assert len(sparse_attn_calls) == 1
    assert sparse_attn_calls[0][6] is not None
    torch.testing.assert_close(output, qknorm_args[17])


def test_minimax_m3_sparse_attention_custom_op_uses_forward_context() -> None:
    class RecordingIndexer:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def forward_with_cache(
            self,
            index_query: torch.Tensor,
            kv_cache: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.calls.append((index_query, kv_cache))
            topk = torch.zeros((1, 1), dtype=torch.int32)
            return topk, topk

    class RecordingImpl:
        def __init__(self) -> None:
            self.calls: list[
                tuple[
                    MiniMaxM3SparseAttention,
                    torch.Tensor,
                    torch.Tensor,
                    tuple[torch.Tensor, torch.Tensor],
                    torch.Tensor,
                ]
            ] = []

        def forward(
            self,
            layer: MiniMaxM3SparseAttention,
            query: torch.Tensor,
            kv_cache: torch.Tensor,
            topk_idx: tuple[torch.Tensor, torch.Tensor],
            output: torch.Tensor,
        ) -> torch.Tensor:
            self.calls.append((layer, query, kv_cache, topk_idx, output))
            output.copy_(query)
            return output

    layer = object.__new__(MiniMaxM3SparseAttention)
    nn.Module.__init__(layer)
    layer.layer_name = "layers.3.self_attn"
    layer.indexer = RecordingIndexer()
    layer.impl = RecordingImpl()
    layer.kv_cache = torch.empty(0)

    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context[layer.layer_name] = layer
    query = torch.randn(2, 4)
    index_query = torch.randn(2, 3)
    main_kv_cache = torch.empty(1, 2, 2, 1, 4)
    index_kv_cache = torch.empty(1, 2, 3)
    output = torch.empty_like(query)

    with set_forward_context({}, vllm_config):
        result = minimax_m3_sparse_attention_with_output(
            query,
            index_query,
            main_kv_cache,
            index_kv_cache,
            output,
            layer.layer_name,
            torch.empty(0),
        )

    assert result is None
    assert len(layer.indexer.calls) == 1
    seen_index_query, seen_index_cache = layer.indexer.calls[0]
    assert seen_index_query is index_query
    assert seen_index_cache is index_kv_cache
    assert len(layer.impl.calls) == 1
    seen_layer, seen_query, seen_cache, seen_topk, seen_output = layer.impl.calls[0]
    assert seen_layer is layer
    assert seen_query is query
    assert seen_cache is main_kv_cache
    assert seen_topk[0] is seen_topk[1]
    assert seen_output is output
    torch.testing.assert_close(output, query)


def test_b12x_msa_triton_compare_budget_is_process_global(monkeypatch) -> None:
    monkeypatch.setattr(sparse_attention_b12x, "_B12X_MSA_TRITON_COMPARE_REPORTS", 0)

    assert sparse_attention_b12x._claim_triton_compare_report(2)
    assert sparse_attention_b12x._claim_triton_compare_report(2)
    assert not sparse_attention_b12x._claim_triton_compare_report(2)


def test_minimax_m3_loads_indexer_projection_into_index_q_shard() -> None:
    qkv_weight = _FakeParam(torch.bfloat16)
    params = {
        "layers.3.self_attn.qkv_proj.weight": qkv_weight,
    }
    model = _fake_minimax_model(params)

    loaded = MiniMaxM3Model.load_weights(
        model,
        [
            (
                "layers.3.self_attn.indexer.q_proj.weight",
                torch.ones((4, 64), dtype=torch.bfloat16),
            )
        ],
    )

    assert loaded == {"layers.3.self_attn.qkv_proj.weight"}
    assert len(qkv_weight.calls) == 1
    loaded_weight, shard_id = qkv_weight.calls[0]
    assert loaded_weight.dtype == torch.bfloat16
    assert shard_id == "index_q"


def test_minimax_m3_prefers_split_indexer_projection_when_present() -> None:
    qkv_weight = _FakeParam(torch.float8_e4m3fn)
    indexer_qk_weight = _FakeParam(torch.bfloat16)
    params = {
        "layers.3.self_attn.qkv_proj.weight": qkv_weight,
        "layers.3.self_attn.indexer_qk_proj.weight": indexer_qk_weight,
    }
    model = _fake_minimax_model(params)

    MiniMaxM3Model.load_weights(
        model,
        [
            (
                "layers.3.self_attn.indexer.q_proj.weight",
                torch.randn((4, 64), dtype=torch.bfloat16),
            )
        ],
    )

    assert not qkv_weight.calls
    assert len(indexer_qk_weight.calls) == 1
    loaded_weight, shard_id = indexer_qk_weight.calls[0]
    assert loaded_weight.dtype == torch.bfloat16
    assert shard_id == "index_q"


def test_minimax_m3_splits_unquantized_indexer_for_mixed_mxfp8() -> None:
    class QuantConfig:
        def _resolve_quant_algo(self, prefix: str) -> str | None:
            if prefix.endswith(".qkv_proj"):
                return "MXFP8"
            return None

    assert _should_split_mxfp8_indexer_projection(
        QuantConfig(), "language_model.model.layers.3.self_attn"
    )


def test_minimax_m3_keeps_fused_projection_for_quantized_indexer() -> None:
    class QuantConfig:
        def _resolve_quant_algo(self, prefix: str) -> str | None:
            if prefix.endswith((".qkv_proj", ".indexer.q_proj", ".indexer.k_proj")):
                return "MXFP8"
            return None

    assert not _should_split_mxfp8_indexer_projection(
        QuantConfig(), "language_model.model.layers.3.self_attn"
    )


_MINIMAX_M3_TP3_VIRTUAL_TP_PLAN = {
    "sharding": "b12x-padded",
    "model_type": "minimax_m3",
    "attention_heads": {
        "original_size": 64,
        "padded_size": 96,
        "tp_size": 3,
        "local_size": 32,
    },
    "kv_heads": {
        "original_size": 4,
        "padded_size": 6,
        "tp_size": 3,
        "local_size": 2,
        "q_heads_per_kv": 16,
    },
    "index_heads": {
        "original_size": 4,
        "padded_size": 6,
        "tp_size": 3,
        "local_size": 2,
    },
}


def _fake_current_config_with_virtual_tp_plan():
    text_config = SimpleNamespace()
    setattr(text_config, VIRTUAL_TP_PLAN_ATTR, _MINIMAX_M3_TP3_VIRTUAL_TP_PLAN)
    return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=text_config))


def _fake_vllm_config_for_indexer(virtual_tp_plan: dict | None = None):
    text_config = SimpleNamespace()
    if virtual_tp_plan is not None:
        setattr(text_config, VIRTUAL_TP_PLAN_ATTR, virtual_tp_plan)
    hf_config = SimpleNamespace(text_config=text_config)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            hf_text_config=text_config,
        )
    )


def test_minimax_m3_indexer_metadata_uses_virtual_tp3_index_heads() -> None:
    vllm_config = _fake_vllm_config_for_indexer(_MINIMAX_M3_TP3_VIRTUAL_TP_PLAN)

    assert _get_minimax_m3_indexer_num_heads(vllm_config, 4, 3) == 2


def test_minimax_m3_indexer_metadata_keeps_regular_tp_head_counts() -> None:
    vllm_config = _fake_vllm_config_for_indexer()

    assert _get_minimax_m3_indexer_num_heads(vllm_config, 4, 2) == 2
    assert _get_minimax_m3_indexer_num_heads(vllm_config, 4, 4) == 1
    assert _get_minimax_m3_indexer_num_heads(vllm_config, 4, 8) == 1
    with pytest.raises(AssertionError):
        _get_minimax_m3_indexer_num_heads(vllm_config, 4, 3)


def _fake_qkv_layer(
    tp_rank: int,
    tp_size: int = 4,
    virtual_tp_plan: dict | None = None,
) -> MinimaxM3QKVParallelLinear:
    layer = object.__new__(MinimaxM3QKVParallelLinear)
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.head_size = 128
    layer.v_head_size = 128
    layer.num_heads = 32 if virtual_tp_plan is not None else 16
    layer.num_kv_heads = 2 if virtual_tp_plan is not None else 1
    layer.num_kv_head_replicas = 1
    layer._minimax_m3_virtual_tp_plan = virtual_tp_plan
    return layer


def _fake_qkv_indexer_layer(
    tp_rank: int,
    tp_size: int = 4,
    virtual_tp_plan: dict | None = None,
) -> MinimaxM3QKVParallelLinearWithIndexer:
    layer = object.__new__(MinimaxM3QKVParallelLinearWithIndexer)
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.head_size = 128
    layer.index_head_size = 128
    layer.num_heads = 32 if virtual_tp_plan is not None else 16
    layer.num_kv_heads = 2 if virtual_tp_plan is not None else 1
    layer.num_kv_head_replicas = 1
    layer.num_index_heads = 2 if virtual_tp_plan is not None else 1
    layer._minimax_m3_virtual_tp_plan = virtual_tp_plan
    return layer


def _fake_indexer_qk_layer(
    tp_rank: int,
    tp_size: int = 4,
    virtual_tp_plan: dict | None = None,
) -> MinimaxM3IndexerQKParallelLinear:
    layer = object.__new__(MinimaxM3IndexerQKParallelLinear)
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.index_head_size = 128
    layer.num_index_heads = 2 if virtual_tp_plan is not None else 1
    layer.num_index_head_replicas = 1
    layer._minimax_m3_virtual_tp_plan = virtual_tp_plan
    return layer


def _make_real_model_weight_parameter(
    tp_rank: int,
    data: torch.Tensor,
    monkeypatch,
    tp_size: int = 4,
) -> ModelWeightParameter:
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_rank", lambda: tp_rank
    )
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    return ModelWeightParameter(
        data=data,
        input_dim=1,
        output_dim=0,
        weight_loader=lambda *args, **kwargs: None,
    )


def _rows(rows: int, cols: int, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(rows, cols)
    if dtype == torch.uint8:
        return values.remainder(251).to(torch.uint8)
    return values.to(dtype)


def _expected_padded_rows(
    rows: int,
    cols: int,
    dtype: torch.dtype,
    start: int,
    size: int,
) -> torch.Tensor:
    expected = torch.zeros((size, cols), dtype=dtype)
    available = rows - start
    if available > 0:
        valid = min(size, available)
        expected[:valid].copy_(_rows(rows, cols, dtype).narrow(0, start, valid))
    return expected


def test_minimax_m3_qkv_indexer_loader_places_real_tp_shards(monkeypatch) -> None:
    local_rows = 2048 + 128 + 128 + 128 + 128
    cols = 4

    for tp_rank in range(4):
        layer = _fake_qkv_indexer_layer(tp_rank)
        weight = _make_real_model_weight_parameter(
            tp_rank,
            torch.zeros((local_rows, cols), dtype=torch.bfloat16),
            monkeypatch,
        )
        scale = _make_real_model_weight_parameter(
            tp_rank,
            torch.zeros((local_rows, cols), dtype=torch.uint8),
            monkeypatch,
        )

        layer.weight_loader_v2(weight, _rows(8192, cols, torch.bfloat16), "q")
        layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "k")
        layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "v")
        layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "index_q")
        layer.weight_loader_v2(weight, _rows(128, cols, torch.bfloat16), "index_k")

        layer.weight_loader_v2(scale, _rows(512, cols, torch.uint8), "index_q")
        layer.weight_loader_v2(scale, _rows(128, cols, torch.uint8), "index_k")

        torch.testing.assert_close(
            weight.data[:2048],
            _rows(8192, cols, torch.bfloat16).narrow(0, tp_rank * 2048, 2048),
        )
        torch.testing.assert_close(
            weight.data[2048:2176],
            _rows(512, cols, torch.bfloat16).narrow(0, tp_rank * 128, 128),
        )
        torch.testing.assert_close(
            weight.data[2176:2304],
            _rows(512, cols, torch.bfloat16).narrow(0, tp_rank * 128, 128),
        )
        torch.testing.assert_close(
            weight.data[2304:2432],
            _rows(512, cols, torch.bfloat16).narrow(0, tp_rank * 128, 128),
        )
        torch.testing.assert_close(
            weight.data[2432:2560], _rows(128, cols, torch.bfloat16)
        )
        torch.testing.assert_close(
            scale.data[2304:2432],
            _rows(512, cols, torch.uint8).narrow(0, tp_rank * 128, 128),
        )
        torch.testing.assert_close(scale.data[2432:2560], _rows(128, cols, torch.uint8))


def test_minimax_m3_qkv_loader_zero_fills_virtual_tp3_tail(monkeypatch) -> None:
    local_rows = 4096 + 256 + 256
    cols = 4
    current_config = _fake_current_config_with_virtual_tp_plan()

    for tp_rank in range(3):
        layer = _fake_qkv_layer(
            tp_rank,
            tp_size=3,
            virtual_tp_plan=_MINIMAX_M3_TP3_VIRTUAL_TP_PLAN,
        )
        weight = _make_real_model_weight_parameter(
            tp_rank,
            torch.zeros((local_rows, cols), dtype=torch.bfloat16),
            monkeypatch,
            tp_size=3,
        )

        with set_current_vllm_config(current_config):
            layer.weight_loader_v2(weight, _rows(8192, cols, torch.bfloat16), "q")
            layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "k")
            layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "v")

        torch.testing.assert_close(
            weight.data[:4096],
            _expected_padded_rows(
                8192, cols, torch.bfloat16, tp_rank * 4096, 4096
            ),
        )
        torch.testing.assert_close(
            weight.data[4096:4352],
            _expected_padded_rows(512, cols, torch.bfloat16, tp_rank * 256, 256),
        )
        torch.testing.assert_close(
            weight.data[4352:4608],
            _expected_padded_rows(512, cols, torch.bfloat16, tp_rank * 256, 256),
        )


def test_minimax_m3_qkv_indexer_loader_zero_fills_virtual_tp3_tail(
    monkeypatch,
) -> None:
    local_rows = 4096 + 256 + 256 + 256 + 128
    cols = 4
    current_config = _fake_current_config_with_virtual_tp_plan()

    for tp_rank in range(3):
        layer = _fake_qkv_indexer_layer(
            tp_rank,
            tp_size=3,
            virtual_tp_plan=_MINIMAX_M3_TP3_VIRTUAL_TP_PLAN,
        )
        weight = _make_real_model_weight_parameter(
            tp_rank,
            torch.zeros((local_rows, cols), dtype=torch.bfloat16),
            monkeypatch,
            tp_size=3,
        )

        with set_current_vllm_config(current_config):
            layer.weight_loader_v2(weight, _rows(8192, cols, torch.bfloat16), "q")
            layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "k")
            layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "v")
            layer.weight_loader_v2(
                weight, _rows(512, cols, torch.bfloat16), "index_q"
            )
            layer.weight_loader_v2(
                weight, _rows(128, cols, torch.bfloat16), "index_k"
            )

        torch.testing.assert_close(
            weight.data[:4096],
            _expected_padded_rows(
                8192, cols, torch.bfloat16, tp_rank * 4096, 4096
            ),
        )
        torch.testing.assert_close(
            weight.data[4096:4352],
            _expected_padded_rows(512, cols, torch.bfloat16, tp_rank * 256, 256),
        )
        torch.testing.assert_close(
            weight.data[4352:4608],
            _expected_padded_rows(512, cols, torch.bfloat16, tp_rank * 256, 256),
        )
        torch.testing.assert_close(
            weight.data[4608:4864],
            _expected_padded_rows(512, cols, torch.bfloat16, tp_rank * 256, 256),
        )
        torch.testing.assert_close(
            weight.data[4864:4992], _rows(128, cols, torch.bfloat16)
        )


def test_minimax_m3_split_indexer_loader_places_real_tp_shards(
    monkeypatch,
) -> None:
    local_rows = 128 + 128
    cols = 4

    for tp_rank in range(4):
        layer = _fake_indexer_qk_layer(tp_rank)
        weight = _make_real_model_weight_parameter(
            tp_rank,
            torch.zeros((local_rows, cols), dtype=torch.bfloat16),
            monkeypatch,
        )

        layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "index_q")
        layer.weight_loader_v2(weight, _rows(128, cols, torch.bfloat16), "index_k")

        torch.testing.assert_close(
            weight.data[:128],
            _rows(512, cols, torch.bfloat16).narrow(0, tp_rank * 128, 128),
        )
        torch.testing.assert_close(
            weight.data[128:256], _rows(128, cols, torch.bfloat16)
        )


def test_minimax_m3_split_indexer_loader_zero_fills_virtual_tp3_tail(
    monkeypatch,
) -> None:
    local_rows = 256 + 128
    cols = 4
    current_config = _fake_current_config_with_virtual_tp_plan()

    for tp_rank in range(3):
        layer = _fake_indexer_qk_layer(
            tp_rank,
            tp_size=3,
            virtual_tp_plan=_MINIMAX_M3_TP3_VIRTUAL_TP_PLAN,
        )
        weight = _make_real_model_weight_parameter(
            tp_rank,
            torch.zeros((local_rows, cols), dtype=torch.bfloat16),
            monkeypatch,
            tp_size=3,
        )

        with set_current_vllm_config(current_config):
            layer.weight_loader_v2(weight, _rows(512, cols, torch.bfloat16), "index_q")
            layer.weight_loader_v2(weight, _rows(128, cols, torch.bfloat16), "index_k")

        torch.testing.assert_close(
            weight.data[:256],
            _expected_padded_rows(512, cols, torch.bfloat16, tp_rank * 256, 256),
        )
        torch.testing.assert_close(
            weight.data[256:384], _rows(128, cols, torch.bfloat16)
        )


def test_minimax_m3_loads_indexer_norms() -> None:
    index_q_norm = _FakeParam(torch.bfloat16)
    params = {
        "layers.3.self_attn.index_q_norm.weight": index_q_norm,
    }
    model = _fake_minimax_model(params)

    loaded = MiniMaxM3Model.load_weights(
        model,
        [
            (
                "layers.3.self_attn.indexer.q_norm.weight",
                torch.ones((128,), dtype=torch.bfloat16),
            )
        ],
    )

    assert loaded == {"layers.3.self_attn.index_q_norm.weight"}
    assert len(index_q_norm.calls) == 1
    loaded_weight, shard_id = index_q_norm.calls[0]
    assert loaded_weight.shape == (128,)
    assert shard_id is None
