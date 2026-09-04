# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from vllm.config import AttentionConfig, ParallelConfig
from vllm.config.compilation import CompilationMode
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor import custom_op
from vllm.model_executor.layers import vocab_parallel_embedding
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models import qwen3_dflash
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3Model,
    _get_dflash_draft_vocab_size,
    _get_glm53_tp3_head_geometry,
    _get_glm53_tp3_vocab_kwargs,
)
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def _disable_custom_ops(monkeypatch) -> None:
    compilation_config = SimpleNamespace(
        mode=CompilationMode.NONE,
        custom_ops=["none"],
        enabled_custom_ops=set(),
        disabled_custom_ops=set(),
    )
    monkeypatch.setattr(
        custom_op,
        "get_cached_compilation_config",
        lambda: compilation_config,
    )


def _tp3_dflash_config(**overrides):
    values = {
        "glm53_tp3_padding": True,
        "glm53_tp3_vocab_padding_size": 192,
        "glm53_tp3_vocab_storage_size": 154944,
        "num_attention_heads": 36,
        "num_key_value_heads": 9,
        "original_num_attention_heads": 32,
        "original_num_key_value_heads": 8,
        "original_vocab_size": 154880,
        "draft_vocab_size": 154880,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "vocab_size": 154880,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _capture_attention_projection_kwargs(monkeypatch, config, tp_size):
    _disable_custom_ops(monkeypatch)
    calls = {}

    class FakeProjection(torch.nn.Module):
        def __init__(self, kind, args, kwargs):
            super().__init__()
            calls[kind] = (args, kwargs)

    monkeypatch.setattr(
        qwen3_dflash,
        "QKVParallelLinear",
        lambda *args, **kwargs: FakeProjection("qkv", args, kwargs),
    )
    monkeypatch.setattr(
        qwen3_dflash,
        "RowParallelLinear",
        lambda *args, **kwargs: FakeProjection("o", args, kwargs),
    )
    monkeypatch.setattr(
        qwen3_dflash,
        "DFlashAttention",
        lambda *args, **kwargs: FakeProjection("attention", args, kwargs),
    )
    monkeypatch.setattr(qwen3_dflash, "get_rope", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    DFlashQwen3Attention(
        hidden_size=64,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        config=config,
        rope_parameters={},
        head_dim=2,
        prefix="model.layers.0.self_attn",
    )
    return calls


def test_dflash_tp3_wires_logical_checkpoint_projection_sizes(monkeypatch) -> None:
    calls = _capture_attention_projection_kwargs(
        monkeypatch, _tp3_dflash_config(), tp_size=3
    )
    qkv_args, qkv_kwargs = calls["qkv"]
    o_args, o_kwargs = calls["o"]

    assert qkv_args[:4] == (64, 2, 36, 9)
    assert qkv_kwargs["loaded_total_num_heads"] == 32
    assert qkv_kwargs["loaded_total_num_kv_heads"] == 8
    assert qkv_kwargs["prefix"] == "model.layers.0.self_attn.qkv_proj"
    assert o_args[:2] == (72, 64)
    assert o_kwargs["loaded_input_size"] == 64
    assert o_kwargs["prefix"] == "model.layers.0.self_attn.o_proj"


def test_dflash_tp3_geometry_and_vocab_storage(monkeypatch) -> None:
    _disable_custom_ops(monkeypatch)
    config = _tp3_dflash_config()

    assert _get_glm53_tp3_head_geometry(config) == (32, 8)
    assert _get_dflash_draft_vocab_size(config) == 154880
    vocab_kwargs = _get_glm53_tp3_vocab_kwargs(config)
    assert vocab_kwargs == {"padding_size": 192}
    assert config.intermediate_size % 3 == 0
    assert config.intermediate_size // 3 == 4096
    assert config.intermediate_size % 4 == 0

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 3,
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_rank",
        lambda: 2,
    )
    embedding = VocabParallelEmbedding(
        _get_dflash_draft_vocab_size(config),
        8,
        **vocab_kwargs,
    )
    assert embedding.num_embeddings == 154880
    assert embedding.num_embeddings_padded == 154944
    assert embedding.num_embeddings_per_partition == 51648
    assert embedding.weight.shape == (51648, 8)


def test_dflash_tp3_sink_bias_pads_only_rank_local_tail(monkeypatch) -> None:
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 2)
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model.config = _tp3_dflash_config()
    loaded = torch.arange(32, dtype=torch.float32)

    [(name, local)] = model._preprocess([("attention_sink_bias", loaded)])

    assert name == "attention_sink_bias"
    assert local.shape == (12,)
    torch.testing.assert_close(local[:8], loaded[24:32])
    torch.testing.assert_close(local[8:], torch.zeros(4))
    assert local.untyped_storage().data_ptr() != loaded.untyped_storage().data_ptr()


def test_zero_padded_qkv_o_projection_matches_logical_reference() -> None:
    generator = torch.Generator().manual_seed(17)
    hidden_states = torch.randn(3, 64, generator=generator)
    logical_q_weight = torch.randn(64, 64, generator=generator)
    logical_kv_weight = torch.randn(16, 64, generator=generator)

    logical_q = torch.nn.functional.linear(hidden_states, logical_q_weight)
    logical_k = torch.nn.functional.linear(hidden_states, logical_kv_weight)
    physical_q = torch.zeros(3, 72)
    physical_k = torch.zeros(3, 18)
    physical_q[:, :64] = logical_q
    physical_k[:, :16] = logical_k

    torch.testing.assert_close(physical_q[:, 64:], torch.zeros(3, 8))
    torch.testing.assert_close(physical_k[:, 16:], torch.zeros(3, 2))

    logical_o_weight = torch.randn(64, 64, generator=generator)
    physical_o_weight = torch.zeros(64, 72)
    physical_o_weight[:, :64] = logical_o_weight
    logical_output = torch.nn.functional.linear(logical_q, logical_o_weight)
    padded_output = torch.nn.functional.linear(physical_q, physical_o_weight)
    torch.testing.assert_close(padded_output, logical_output)


def test_dflash_logits_and_selector_exclude_physical_vocab_tail() -> None:
    logical_vocab_size = _get_dflash_draft_vocab_size(_tp3_dflash_config())
    storage_logits = torch.zeros(1, 154944)
    storage_logits[:, logical_vocab_size - 1] = 2
    storage_logits[:, logical_vocab_size:] = 100

    processor = LogitsProcessor.__new__(LogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.org_vocab_size = logical_vocab_size
    processor.scale = 1.0
    processor.soft_cap = None
    processor._apply_head = lambda *args, **kwargs: storage_logits.clone()
    lm_head = SimpleNamespace(
        tp_size=1,
        shard_indices=SimpleNamespace(
            num_org_vocab_padding=154944 - logical_vocab_size,
            org_vocab_start_index=0,
        ),
    )

    logits = processor._get_logits(torch.empty(1, 1), lm_head, None)
    assert logits is not None
    assert logits.shape == (1, logical_vocab_size)
    token_ids, values = processor.get_top_k_tokens(lm_head, torch.empty(1, 1), k=1)
    assert token_ids.item() == logical_vocab_size - 1
    assert values.item() == 2


def test_dflash_tp4_paths_are_exact_noops(monkeypatch) -> None:
    config = SimpleNamespace(
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=154880,
    )
    original_fields = vars(config).copy()
    assert _get_glm53_tp3_head_geometry(config) is None
    assert _get_glm53_tp3_vocab_kwargs(config) == {}
    assert _get_dflash_draft_vocab_size(config) == 154880
    assert vars(config) == original_fields

    calls = _capture_attention_projection_kwargs(monkeypatch, config, tp_size=4)
    qkv_args, qkv_kwargs = calls["qkv"]
    o_args, o_kwargs = calls["o"]
    assert qkv_args[:4] == (64, 2, 32, 8)
    assert "loaded_total_num_heads" not in qkv_kwargs
    assert "loaded_total_num_kv_heads" not in qkv_kwargs
    assert o_args[:2] == (64, 64)
    assert "loaded_input_size" not in o_kwargs

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_rank",
        lambda: 2,
    )
    embedding = VocabParallelEmbedding(config.vocab_size, 8)
    assert embedding.num_embeddings == 154880
    assert embedding.num_embeddings_padded == 154880
    assert embedding.num_embeddings_per_partition == 38720

    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(qwen3_dflash, "get_tensor_model_parallel_rank", lambda: 2)
    model = DFlashQwen3Model.__new__(DFlashQwen3Model)
    torch.nn.Module.__init__(model)
    model.config = config
    loaded = torch.arange(32, dtype=torch.float32)
    [(_, local)] = model._preprocess([("attention_sink_bias", loaded)])
    torch.testing.assert_close(local, loaded[16:24])
    assert local.untyped_storage().data_ptr() == loaded.untyped_storage().data_ptr()


def test_dense_dflash_tp3_drops_target_expert_parallel(monkeypatch) -> None:
    from vllm.transformers_utils.configs import glm53_tp3

    monkeypatch.setattr(glm53_tp3, "is_glm53_config", lambda _: True)
    applied = []
    monkeypatch.setattr(
        glm53_tp3,
        "apply_glm53_tp3_draft_geometry",
        lambda *args: applied.append(args),
    )
    placement = {
        "prefill_context_parallel_size": 2,
        "decode_context_parallel_size": 2,
        "dcp_kv_cache_interleave_size": 4,
        "dcp_comm_backend": "a2a",
        "dcp_q_replicate": True,
        "cp_kv_cache_interleave_size": 4,
        "data_parallel_size": 2,
        "data_parallel_size_local": 2,
        "data_parallel_rank": 1,
        "data_parallel_rank_local": 1,
        "data_parallel_index": 1,
        "data_parallel_master_ip": "127.0.0.1",
        "data_parallel_rpc_port": 1234,
        "data_parallel_master_port": 4321,
        "data_parallel_backend": "mp",
        "data_parallel_external_lb": False,
        "data_parallel_hybrid_lb": False,
    }
    target_parallel = SimpleNamespace(
        tensor_parallel_size=3,
        enable_expert_parallel=True,
        rank=2,
        **placement,
    )
    draft_parallel = ParallelConfig(
        tensor_parallel_size=3,
        enable_expert_parallel=True,
    )
    spec = object.__new__(SpeculativeConfig)
    object.__setattr__(spec, "method", "dflash")
    object.__setattr__(spec, "target_parallel_config", target_parallel)
    target_hf_config = SimpleNamespace(architectures=["Glm5NextForCausalLM"])
    object.__setattr__(
        spec,
        "target_model_config",
        SimpleNamespace(
            hf_config=target_hf_config,
            hf_text_config=target_hf_config,
        ),
    )
    object.__setattr__(spec, "draft_model_config", object())
    object.__setattr__(spec, "draft_parallel_config", draft_parallel)

    spec._apply_glm53_tp3_draft_geometry()

    assert draft_parallel.enable_expert_parallel is False
    assert draft_parallel.world_size == 6
    for name, value in placement.items():
        assert getattr(draft_parallel, name) == value
    assert len(applied) == 1

    @dataclass
    class DraftVllmConfig:
        model_config: Any
        parallel_config: Any
        attention_config: AttentionConfig

    base = DraftVllmConfig(
        model_config=SimpleNamespace(
            model_arch_config=SimpleNamespace(is_mm_prefix_lm=False)
        ),
        parallel_config=target_parallel,
        attention_config=AttentionConfig(),
    )
    proposer = object.__new__(DFlashProposer)
    proposer.speculative_config = spec
    proposer.vllm_config = base
    proposer.dflash_causal = True
    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        lambda _: base,
    )

    draft_vllm_config = proposer._create_draft_vllm_config()

    assert draft_vllm_config.parallel_config is not draft_parallel
    assert draft_vllm_config.parallel_config.tensor_parallel_size == 3
    assert not draft_vllm_config.parallel_config.enable_expert_parallel
    assert draft_vllm_config.parallel_config.rank == target_parallel.rank
    assert draft_parallel.rank == 0
    assert target_parallel.tensor_parallel_size == 3
    assert target_parallel.enable_expert_parallel

    target_parallel.tensor_parallel_size = 4
    tp4_vllm_config = proposer._create_draft_vllm_config()
    assert tp4_vllm_config.parallel_config is target_parallel


def test_engine_dp_identity_reaches_speculative_draft() -> None:
    from vllm.v1.engine.core import EngineCoreProc

    target = SimpleNamespace(
        data_parallel_index=3,
        data_parallel_rank=3,
        data_parallel_rank_local=1,
    )
    draft = SimpleNamespace(
        data_parallel_index=0,
        data_parallel_rank=0,
        data_parallel_rank_local=0,
    )
    vllm_config = SimpleNamespace(
        parallel_config=target,
        speculative_config=SimpleNamespace(draft_parallel_config=draft),
    )

    EngineCoreProc._sync_speculative_draft_dp_identity(vllm_config)

    assert draft.data_parallel_index == 3
    assert draft.data_parallel_rank == 3
    assert draft.data_parallel_rank_local == 1


def test_dflash7_uses_eight_target_kda_state_columns(monkeypatch) -> None:
    def fake_base_init(self, vllm_config, device) -> None:
        self.max_num_tokens = 16
        self.hidden_size = 4
        self.dtype = torch.float32
        self.num_speculative_steps = 7
        self.max_num_reqs = 2
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                num_hidden_layers=0,
                dflash_config={"mask_token_id": 0},
            )
        )

    monkeypatch.setattr(DraftModelSpeculator, "__init__", fake_base_init)
    speculator = DFlashSpeculator(SimpleNamespace(), torch.device("cpu"))

    # The target KDA keeps one initial-state column plus one per draft token.
    assert speculator.num_speculative_steps == 7
    assert speculator.num_query_per_req == 8
    assert speculator.sample_col.shape == (14,)
