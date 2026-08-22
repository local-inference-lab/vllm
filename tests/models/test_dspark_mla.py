# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.warmup.kimi_k3_triton_warmup import _get_kda_layer
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia import kda as kimi_k3_kda
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_exposes_packed_quantization_mapping() -> None:
    assert K3DSparkForCausalLM.packed_modules_mapping == {
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    assert set(K3DSparkForCausalLM.checkpoint_skip_substrs) == {
        "confidence_head",
        "embed_tokens",
        "lm_head",
    }


@pytest.mark.cpu_test
def test_dspark_quant_config_aliases_runtime_layer_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = {
        "layers.0.mlp.down_proj": {"stored_tensors": {"down": {}}},
        "model.layers.1.self_attn.q_b_proj": {"stored_tensors": {"q_b": {}}},
        "context_proj": {"stored_tensors": {"context": {}}},
    }
    quant_config = SimpleNamespace(tensor_storage=entries)
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(num_hidden_layers=2)
            )
        ),
    )

    def fail_reparse(*args, **kwargs):
        raise AssertionError("configured draft quantization must be reused")

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", fail_reparse)

    result = dspark_mla._prepare_dspark_quant_config(vllm_config, start_layer_id=5)

    assert result is quant_config
    assert entries["layers.5.mlp.down_proj"] is entries["layers.0.mlp.down_proj"]
    assert (
        entries["model.layers.6.self_attn.q_b_proj"]
        is entries["model.layers.1.self_attn.q_b_proj"]
    )
    assert set(entries) == {
        "layers.0.mlp.down_proj",
        "model.layers.1.self_attn.q_b_proj",
        "context_proj",
        "layers.5.mlp.down_proj",
        "model.layers.6.self_attn.q_b_proj",
    }

    dspark_mla._prepare_dspark_quant_config(vllm_config, start_layer_id=5)
    assert len(entries) == 5


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )
    monkeypatch.delenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", raising=False)
    monkeypatch.delenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", raising=False)

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
@pytest.mark.parametrize("replicate_w1", [False, True])
def test_dspark_markov_head_shards_vocabulary_weights(
    monkeypatch: pytest.MonkeyPatch,
    replicate_w1: bool,
) -> None:
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.setenv(
        "VLLM_DSPARK_REPLICATE_MARKOV_W1",
        "1" if replicate_w1 else "0",
    )
    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")

    assert head.shard_across_tp
    assert head.replicate_w1 is replicate_w1
    assert head.markov_w2.tp_size == 8
    assert head.markov_w2.weight.shape == (16, 8)
    if replicate_w1:
        assert isinstance(head.markov_w1, nn.Embedding)
        assert head.markov_w1.weight.shape == (128, 8)
    else:
        assert isinstance(
            head.markov_w1,
            vocab_parallel_embedding.VocabParallelEmbedding,
        )
        assert head.markov_w1.weight.shape == (16, 8)

    processor = LogitsProcessor(128)
    local_bias = head.local_bias(torch.ones((2, 8)), processor)
    assert local_bias.shape == (2, 16)


@pytest.mark.cpu_test
def test_replicated_markov_w1_requires_sharded_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", raising=False)
    monkeypatch.setenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", "1")

    with pytest.raises(ValueError, match="requires VLLM_DSPARK_SHARD_MARKOV_HEAD"):
        DSparkMarkovHead(128, 128, 8, prefix="markov_head")


@pytest.mark.cpu_test
def test_k3_dspark_model_reuses_prepared_quant_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quant_config = object()
    projection_configs = []
    decoder_configs = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(0))

    def make_projection(*args, **kwargs):
        projection_configs.append(kwargs.get("quant_config"))
        return DummyModule()

    def make_decoder(*args, **kwargs):
        decoder_configs.append(kwargs.get("quant_config"))
        return DummyModule()

    monkeypatch.setattr(
        dspark_mla,
        "_prepare_dspark_quant_config",
        lambda vllm_config, start_layer_id: quant_config,
    )
    monkeypatch.setattr(
        dspark_mla,
        "get_draft_quant_config",
        lambda _: (_ for _ in ()).throw(
            AssertionError("decoder must reuse prepared quantization")
        ),
    )
    monkeypatch.setattr(dspark_mla, "ColumnParallelLinear", make_projection)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", make_projection)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_projection)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", make_decoder)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", DummyModule)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=5, prefix="model")

    assert decoder_configs == [quant_config]
    assert projection_configs and all(
        candidate is quant_config for candidate in projection_configs
    )


@pytest.mark.cpu_test
def test_k3_dspark_quantized_context_projection_needs_no_dense_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class QuantizedProjection(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.register_parameter(
                "trellis",
                nn.Parameter(torch.empty(0, dtype=torch.int16), requires_grad=False),
            )

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(0))

    monkeypatch.setattr(
        dspark_mla,
        "_prepare_dspark_quant_config",
        lambda vllm_config, start_layer_id: object(),
    )
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", QuantizedProjection)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", QuantizedProjection)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", DummyModule)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
        target_layer_ids=(),
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )

    model = K3DSparkModel(vllm_config=vllm_config, start_layer_id=5, prefix="model")

    assert not hasattr(model.context_proj, "weight")
    assert model._streamed_context_states.dtype == torch.get_default_dtype()
    assert model._streamed_context_states.device == model.context_proj.trellis.device


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []
    context_kv_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(0))

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    def make_context_kv_proj(*args, **kwargs):
        context_kv_proj_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
    assert context_kv_proj_calls == [
        (
            (8, [4]),
            {
                "bias": False,
                "return_bias": False,
                "quant_config": None,
                "prefix": "model.layers.0.self_attn.fused_qkv_a_proj",
                "disable_tp": True,
            },
        )
    ]


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("tp_size", "expect_sharded"),
    [(1, False), (8, True), (12, False), (16, True)],
)
def test_k3_dspark_context_projection_uses_divisible_tp_geometry(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    expect_sharded: bool,
) -> None:
    context_projection_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(0))

    def make_sharded_projection(*args, **kwargs):
        context_projection_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ColumnParallelLinear", make_sharded_projection)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", DummyModule)

    config = SimpleNamespace(
        target_hidden_size=7168,
        num_target_layers=5,
        hidden_size=7168,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=163840,
        draft_vocab_size=163840,
        markov_rank=256,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    model = K3DSparkModel(
        vllm_config=vllm_config,
        start_layer_id=0,
        prefix="model",
    )

    assert model.context_proj_sharded is expect_sharded
    assert model._streamed_context_states.dtype == model.context_proj.weight.dtype
    assert model._streamed_context_states.device == model.context_proj.weight.device
    assert bool(context_projection_calls) is expect_sharded
    if expect_sharded:
        args, kwargs = context_projection_calls[0]
        assert args == (7168 * 5, 7168)
        assert kwargs["gather_output"] is True
        assert kwargs["quant_config"] is None


@pytest.mark.cpu_test
def test_k3_dspark_streams_auxiliary_projection_into_tp_local_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        target_hidden_size=4,
        hidden_size=2,
        target_layer_ids=(0, 1),
    )
    model.context_proj_sharded = True
    model.context_proj = nn.Linear(8, 2, bias=False, dtype=torch.bfloat16)
    model.context_norm = SimpleNamespace(
        weight=torch.tensor([0.75, 1.25], dtype=torch.bfloat16),
        variance_epsilon=1e-5,
    )
    model.context_kv_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model._max_num_context_tokens = 1024
    model._streamed_aux_layer_ids = (1, 2)
    model._streamed_aux_scratch = None
    model._streamed_aux_tokens = 0
    model._streamed_aux_index = 0
    model._streamed_aux_generation = 0
    model._completed_stream_generation = 0
    model._consumed_stream_generation = 0
    model._completed_stream_result = None
    model._context_local_width = 2
    model._context_local_start = 0
    model.register_buffer(
        "_streamed_context_states",
        torch.empty(1024, 2, dtype=torch.bfloat16),
        persistent=False,
    )
    scratch = torch.empty(1024, 4, dtype=torch.bfloat16)
    model.bind_auxiliary_stream_scratch(scratch)
    monkeypatch.setattr(
        dspark_mla,
        "tensor_model_parallel_all_reduce_in_place",
        lambda tensor: tensor,
    )

    first = torch.randn(1024, 4, dtype=torch.bfloat16)
    second = torch.randn(1024, 4, dtype=torch.bfloat16)
    residual = torch.randn(1024, 4, dtype=torch.bfloat16)
    combined = torch.cat((first, second + residual), dim=-1)
    projected = torch.nn.functional.linear(combined, model.context_proj.weight)
    expected = projected * torch.rsqrt(
        projected.float().square().mean(dim=-1, keepdim=True) + 1e-5
    ).to(projected.dtype)
    expected *= model.context_norm.weight

    assert model.can_stream_auxiliary_states((1, 2), first)
    with torch.inference_mode():
        model.begin_auxiliary_stream(first)
        model.accumulate_auxiliary_state(first, None)
        model.accumulate_auxiliary_state(second, residual)
        output = model.finish_auxiliary_stream()

    # Streaming replaces one BF16 GEMM with an ordered sum of BF16 GEMMs, so
    # cancellation can move individual low-magnitude elements by several
    # ulps. Preserve the projected vector rather than requiring elementwise
    # equality between the two legal BF16 accumulation orders.
    mean_absolute_error = (output.float() - expected.float()).abs().mean()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        output.float().flatten(), expected.float().flatten(), dim=0
    )
    assert mean_absolute_error.item() < 2e-3
    assert cosine_similarity.item() > 0.9999
    assert model.is_streamed_context_states([output])
    assert not model.is_streamed_context_states([output])
    assert output.data_ptr() == model._streamed_context_states.data_ptr()

    with torch.inference_mode():
        model.begin_auxiliary_stream(first)
        model.accumulate_auxiliary_state(first, None)
        model.accumulate_auxiliary_state(second, residual)
        next_output = model.finish_auxiliary_stream()

    assert not model.is_streamed_context_states([output])
    assert model.is_streamed_context_states([next_output])


@pytest.mark.cpu_test
def test_k3_dspark_streamed_auxiliary_normalization_uses_global_hidden_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        target_hidden_size=4,
        hidden_size=4,
        target_layer_ids=(0, 1),
    )
    model.context_proj_sharded = True
    model.context_proj = nn.Linear(8, 2, bias=False, dtype=torch.bfloat16)
    model.context_norm = SimpleNamespace(
        weight=torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.bfloat16),
        variance_epsilon=1e-5,
    )
    model.context_kv_proj = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model._max_num_context_tokens = 1024
    model._streamed_aux_layer_ids = (1, 2)
    model._streamed_aux_scratch = None
    model._streamed_aux_tokens = 0
    model._streamed_aux_index = 0
    model._streamed_aux_generation = 0
    model._completed_stream_generation = 0
    model._consumed_stream_generation = 0
    model._completed_stream_result = None
    model._context_local_width = 2
    model._context_local_start = 2
    model.register_buffer(
        "_streamed_context_states",
        torch.empty(1024, 2, dtype=torch.bfloat16),
        persistent=False,
    )
    model.bind_auxiliary_stream_scratch(torch.empty(1024, 4, dtype=torch.bfloat16))

    first = torch.randn(1024, 4, dtype=torch.bfloat16)
    second = torch.randn(1024, 4, dtype=torch.bfloat16)
    combined = torch.cat((first, second), dim=-1)
    local_projection = torch.nn.functional.linear(combined, model.context_proj.weight)
    other_rank_projection = torch.randn_like(local_projection)

    def add_other_rank_squared_norm(tensor: torch.Tensor) -> torch.Tensor:
        tensor.add_(other_rank_projection.float().square().sum(dim=-1, keepdim=True))
        return tensor

    monkeypatch.setattr(
        dspark_mla,
        "tensor_model_parallel_all_reduce_in_place",
        add_other_rank_squared_norm,
    )
    expected_scale = torch.rsqrt(
        (
            local_projection.float().square().sum(dim=-1, keepdim=True)
            + other_rank_projection.float().square().sum(dim=-1, keepdim=True)
        )
        / model.config.hidden_size
        + model.context_norm.variance_epsilon
    ).to(local_projection.dtype)
    expected = local_projection * expected_scale * model.context_norm.weight[2:4]

    with torch.inference_mode():
        model.begin_auxiliary_stream(first)
        model.accumulate_auxiliary_state(first, None)
        model.accumulate_auxiliary_state(second, None)
        output = model.finish_auxiliary_stream()

    torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.cpu_test
def test_k3_dspark_disables_streaming_for_incompatible_scratch_width() -> None:
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(target_hidden_size=8)
    model._max_num_context_tokens = 16
    model._streamed_aux_scratch = torch.empty(16, 8)

    assert not model.bind_auxiliary_stream_scratch(torch.empty(16, 4))
    assert model._streamed_aux_scratch is None


@pytest.mark.cpu_test
def test_k3_dspark_preserves_fallback_when_target_has_no_stream_hook() -> None:
    wrapper = object.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(wrapper)
    wrapper.model = SimpleNamespace(bind_auxiliary_stream_scratch=Mock())
    target_model = SimpleNamespace(
        get_language_model=lambda: SimpleNamespace(model=object())
    )

    wrapper.bind_target_auxiliary_stream(
        target_model,
        torch.empty(16, 8),
    )

    wrapper.model.bind_auxiliary_stream_scratch.assert_not_called()


@pytest.mark.cpu_test
def test_k3_dspark_projects_streamed_context_one_layer_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object.__new__(K3DSparkModel)
    nn.Module.__init__(model)
    model._context_local_start = 2
    model._context_local_width = 2
    model._context_kv_width = 3
    model._context_kv_lora_rank = 2
    model._context_rope_dim = 1
    model._context_rms_norm_eps = 1e-5
    model._context_kv_norm_weights = torch.ones(2, 2)
    model.context_kv_proj = nn.Linear(4, 6, bias=False)
    model.context_kv_proj.weight.data.copy_(torch.arange(24).view(6, 4))
    cache_updates = []

    class Attention:
        def __init__(self):
            self.rotary_emb = SimpleNamespace(head_size=1, is_neox_style=True)
            self.impl = SimpleNamespace(
                do_kv_cache_update=lambda *args: cache_updates.append(args)
            )
            self.kv_cache = object()
            self.kv_cache_dtype = "bf16"
            self._k_scale = 1.0

    model.layers = [
        SimpleNamespace(self_attn=Attention()),
        SimpleNamespace(self_attn=Attention()),
    ]
    model._get_rope_inputs = lambda positions: (positions, torch.empty(0))
    reduced = []

    def record_all_reduce(tensor):
        reduced.append(tensor.clone())
        return tensor

    monkeypatch.setattr(
        dspark_mla,
        "tensor_model_parallel_all_reduce_in_place",
        record_all_reduce,
    )
    rms_norm_io = []

    def record_rms_norm(output, input_, weight, eps):
        rms_norm_io.append((output, input_))
        output.copy_(input_)

    monkeypatch.setattr(dspark_mla.ops, "rms_norm", record_rms_norm)
    monkeypatch.setattr(dspark_mla.ops, "rotary_embedding", lambda *args: None)
    context = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positions = torch.tensor([5, 6])
    first_slots = torch.tensor([0, 1])

    model._precompute_streamed_context_kv(
        context,
        positions,
        [first_slots, None],
    )

    expected = []
    for layer_index in range(2):
        weight = model.context_kv_proj.weight[
            layer_index * 3 : (layer_index + 1) * 3, 2:4
        ]
        expected.append(torch.nn.functional.linear(context, weight))
    torch.testing.assert_close(reduced[0], expected[0])
    torch.testing.assert_close(reduced[1], expected[1])
    assert all(output.data_ptr() != input_.data_ptr() for output, input_ in rms_norm_io)
    assert len(cache_updates) == 1
    torch.testing.assert_close(cache_updates[0][0], expected[0][:, :2])
    torch.testing.assert_close(cache_updates[0][1], expected[0][:, 2:].view(2, 1, 1))
    assert cache_updates[0][3] is first_slots


def test_context_kv_weights_are_loaded_as_merged_linear_shards():
    weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight_packed",
            torch.arange(4),
        ),
        (
            "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale",
            torch.tensor(0.5),
        ),
    ]

    duplicated = dspark_mla._duplicate_context_kv_weights(weights, 2)
    mapped = list(K3DSparkForCausalLM.hf_to_vllm_mapper.apply(duplicated))

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.fused_qkv_a_proj.weight_packed",
        "model.context_kv_proj.weight_packed",
        "model.layers.1.self_attn.fused_qkv_a_proj.weight_scale",
        "model.context_kv_proj.weight_scale",
    ]
    assert [weight.shard_id for _, weight in mapped] == [1, 0, 1, 1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr()
    assert mapped[2][1].data_ptr() == mapped[3][1].data_ptr()


def test_kda_warmup_ignores_cacheless_dspark_layers(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeKDA:
        kv_cache: list[object]

    draft_layer = FakeKDA()
    target_layer = FakeKDA()
    target_layer.kv_cache = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            compilation_config=SimpleNamespace(
                static_forward_context={
                    "draft.layers.0": draft_layer,
                    "target.layers.0": target_layer,
                }
            )
        )
    )
    monkeypatch.setattr(kimi_k3_kda, "KimiK3DeltaAttention", FakeKDA)

    assert _get_kda_layer(worker) is target_layer


def _make_local_argmax_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tp_size: int = 8,
    tp_rank: int = 0,
    vocab_size: int = 128,
) -> K3DSparkForCausalLM:
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "1")
    monkeypatch.delenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", raising=False)
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_rank",
        lambda: tp_rank,
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    model = object.__new__(K3DSparkForCausalLM)
    nn.Module.__init__(model)
    model.lm_head = ParallelLMHead(vocab_size, 8, prefix="lm_head")
    model.model = nn.Module()
    model.model.markov_head = DSparkMarkovHead(
        vocab_size, vocab_size, 8, prefix="markov_head"
    )
    model.logits_processor = LogitsProcessor(vocab_size)
    model._b12x_dspark_argmax_enabled = False
    model._b12x_dspark_argmax_cls = None
    model._b12x_dspark_argmax_runtime = None
    model._b12x_dspark_argmax_output = None
    model._b12x_dspark_argmax_max_batch = 8
    return model


@pytest.mark.cpu_test
def test_k3_dspark_local_argmax_requires_matching_shards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_local_argmax_model(monkeypatch)
    assert model.supports_local_draft_argmax()

    model.model.markov_head.markov_w2.num_embeddings_padded += 64
    assert not model.supports_local_draft_argmax()


@pytest.mark.cpu_test
def test_k3_dspark_cutedsl_argmax_masks_tp12_vocab_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_local_argmax_model(
        monkeypatch,
        tp_size=12,
        tp_rank=11,
        vocab_size=190,
    )
    assert model.lm_head.num_embeddings_padded == 192
    assert model.supports_local_draft_argmax()
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    class FakeArgmax:
        @classmethod
        def from_exchange_group(cls, **kwargs):
            return cls()

        def fused_add_argmax(self, base, bias, out):
            calls.append((base.clone(), bias.clone()))
            return out.fill_(7)

    monkeypatch.setattr(
        dspark_mla,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group="tp-metadata-group"),
    )
    model._b12x_dspark_argmax_cls = FakeArgmax
    model._b12x_dspark_argmax_enabled = True
    base = torch.zeros((1, 16), dtype=torch.bfloat16)
    bias = torch.zeros_like(base)
    base[:, -2:] = 1000
    bias[:, -2:] = 1000

    sampled = model.sample_local_draft_logits(base, bias)

    assert sampled.tolist() == [7]
    assert torch.isneginf(calls[0][0][:, -2:]).all()
    assert torch.isneginf(calls[0][1][:, -2:]).all()


@pytest.mark.cpu_test
def test_k3_dspark_cutedsl_argmax_uses_tp8_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_local_argmax_model(monkeypatch)
    construct_calls: list[dict[str, object]] = []
    sample_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    class FakeArgmax:
        @classmethod
        def from_exchange_group(cls, **kwargs):
            construct_calls.append(kwargs)
            return cls()

        def fused_add_argmax(self, base, bias, out):
            sample_calls.append((base.clone(), bias.clone(), out))
            return out.fill_(7)

    monkeypatch.setattr(
        dspark_mla,
        "get_tp_group",
        lambda: SimpleNamespace(cpu_group="tp-metadata-group"),
    )
    model._b12x_dspark_argmax_cls = FakeArgmax
    model._b12x_dspark_argmax_enabled = True
    base = torch.randn((3, 16), dtype=torch.bfloat16)
    bias = torch.randn_like(base)

    sampled = model.sample_local_draft_logits(base, bias)

    assert sampled.tolist() == [7, 7, 7]
    assert construct_calls[0] == {
        "exchange_group": "tp-metadata-group",
        "device": torch.device("cpu"),
        "local_vocab_size": 16,
        "max_batch_size": 8,
    }
    assert sample_calls[0][2].shape == (3,)


@pytest.mark.cpu_test
def test_k3_dspark_local_argmax_falls_back_to_exact_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_local_argmax_model(monkeypatch)
    gathered = []

    def fake_all_gather(logits: torch.Tensor, dim: int) -> torch.Tensor:
        gathered.append((logits.clone(), dim))
        return logits

    monkeypatch.setattr(dspark_mla, "tensor_model_parallel_all_gather", fake_all_gather)
    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.bfloat16)
    bias = torch.tensor([[0.0, 4.0, 0.0, 0.0]], dtype=torch.bfloat16)

    sampled = model.sample_local_draft_logits(base, bias)

    assert sampled.tolist() == [1]
    torch.testing.assert_close(gathered[0][0], base + bias)
    assert gathered[0][1] == -1
