# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
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


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    mapper = K3DSparkForCausalLM.hf_to_vllm_mapper
    for name in ("confidence_head.weight", "embed_tokens.weight", "lm_head.weight"):
        assert mapper._map_name(name) is None


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    from vllm.model_executor.layers import vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
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


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_v41_dspark_retains_each_markov_embedding_during_graph_replay(
    default_vllm_config,
    monkeypatch,
    id_dtype,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native b12x Markov embedding requires SM12x")
    monkeypatch.setenv("VLLM_MXFP8_LM_HEAD", "0")
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.models.deepseek_v4_1.nvidia.dspark import (
        DSparkMarkovHead as V41DSparkMarkovHead,
    )
    from vllm.v1.worker import workspace

    device = torch.device("cuda")
    vocab, rank = 131, 128
    with torch.device(device), torch.no_grad():
        head = V41DSparkMarkovHead(vocab, vocab, rank, prefix="markov_head").bfloat16()
        checkpoint = (
            torch.arange(vocab, device=device)[:, None] / 16
            + torch.arange(rank, device=device)[None, :] / 128
        ).bfloat16()
        default_weight_loader(head.markov_w1.weight, checkpoint)
        token_ids = [
            torch.tensor(values, device=device, dtype=id_dtype)
            for values in ([0, 1, 130], [64, 95, 96], [127, 17, 33])
        ]
        eager = [head.embed(ids) for ids in token_ids]
        # A later Markov step must not overwrite an earlier retained result.
        head.embed(token_ids[-1].roll(1))
        for rows, ids in zip(eager, token_ids):
            torch.testing.assert_close(rows, checkpoint[ids.long()], rtol=0, atol=0)

        freeze_kernel_resolution("V4.1 retained DSpark Markov rows")
        try:
            graph = torch.cuda.CUDAGraph()
            with workspace.collect_cuda_graph_capture_resources() as retained:
                with torch.cuda.graph(graph):
                    captured = [head.embed(ids) for ids in token_ids]
            for offset in (1, 19):
                for step, ids in enumerate(token_ids):
                    ids.add_(offset + step).remainder_(vocab)
                graph.replay()
                torch.cuda.synchronize()
                for rows, ids in zip(captured, token_ids):
                    torch.testing.assert_close(
                        rows,
                        checkpoint[ids.long()],
                        rtol=0,
                        atol=0,
                    )
            del retained
        finally:
            unfreeze_kernel_resolution()


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []
    context_kv_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

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


@pytest.mark.parametrize("layer_groups", [None, [0, 1, 0]])
@torch.inference_mode()
def test_v41_context_graph_replay_matches_checkpoint_projection(
    default_vllm_config, workspace_init, monkeypatch, layer_groups
):
    """Real native projections, rotary and cache writes across shrinking batches."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native V4.1 context preparation requires SM12x")
    from contextlib import nullcontext

    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution
    from b12x.attention import compressed_sparse_mla

    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
    from vllm.models.deepseek_v4_1.b12x_layers import (
        B12xFP8LinearMethod,
        B12xRMSNorm,
    )
    from vllm.models.deepseek_v4_1.nvidia.dspark import (
        DSparkContextCudaGraphs,
        DSparkDeepseekV4Model,
        _ContextKVProjection,
    )
    from vllm.v1.worker import workspace
    from vllm.v1.worker.gpu import cudagraph_utils

    # The exercised context path is replicated and has no collectives. Supply
    # only the distributed capture envelope, not mock model/compute operations.
    monkeypatch.setattr(
        cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(cudagraph_utils, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(cudagraph_utils, "graph_capture", lambda device: nullcontext())
    config = default_vllm_config
    config.scheduler_config.max_num_batched_tokens = 8
    config.scheduler_config.max_num_seqs = 8
    config.compilation_config.max_cudagraph_capture_size = 8
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(1831)

    class NativeLinear(nn.Module):
        def __init__(self, n, k):
            super().__init__()
            self.weight = nn.Parameter(
                torch.randn(n, k, device=device, generator=generator).to(
                    torch.float8_e4m3fn
                ),
                requires_grad=False,
            )
            self.weight_scale_inv = torch.randint(
                122,
                128,
                (n // 32, k // 32),
                dtype=torch.uint8,
                device=device,
                generator=generator,
            ).view(torch.float8_e8m0fnu)
            self.method = B12xFP8LinearMethod(
                SimpleNamespace(weight_block_size=[32, 32])
            )
            self.method.process_weights_after_loading(self)

        def forward(self, x):
            return self.method.apply(self, x)

    with torch.device(device):
        model = DSparkDeepseekV4Model.__new__(DSparkDeepseekV4Model)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=128, dspark_target_layer_ids=(0, 1))
        model.main_proj = NativeLinear(128, 256)
        model.main_norm = B12xRMSNorm(128)
        model.layers = nn.ModuleList()
        model._context_kv_projections = []
        page_bytes = compressed_sparse_mla.page_nbytes(
            32, cache_format="deepseek_v41", cache_kind="swa"
        )
        phases = (
            torch.arange(128, dtype=torch.float32)[:, None]
            * (torch.arange(32, dtype=torch.float32)[None, :] + 1)
            / 128
        )
        for _ in range(3):
            attn = DeepseekV4Attention.__new__(DeepseekV4Attention)
            nn.Module.__init__(attn)
            # Context insertion needs no query-attention plan.
            attn._ready = True
            attn.q_lora_rank = 256
            attn.fused_wqa_wkv = NativeLinear(768, 128)
            attn.kv_norm = B12xRMSNorm(512)
            attn.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.cat((phases.cos(), phases.sin()), dim=-1)
            )
            attn.swa_cache_layer = SimpleNamespace(
                kv_cache=torch.full((4, page_bytes), 91, dtype=torch.uint8)
            )
            layer = nn.Module()
            layer.attn = attn
            model.layers.append(layer)
            model._context_kv_projections.append(_ContextKVProjection(attn, 8))

        hidden = torch.zeros(8, 128, dtype=torch.bfloat16)
        positions = torch.zeros(8, dtype=torch.int64)
        slots = torch.full((2, 8), -1, dtype=torch.int64)
        context = DSparkContextCudaGraphs(
            model, config, hidden, positions, slots, layer_groups, 8
        )
        # Compare KV-only checkpoint packing against the original fused Q|KV
        # projection (nonuniform UE8M0 scales distinguish every block).
        x = torch.randn(8, 128, dtype=torch.bfloat16, generator=generator)
        for layer, projection in zip(model.layers, model._context_kv_projections):
            full = layer.attn.fused_wqa_wkv(x)
            torch.testing.assert_close(projection(x), full[:, 256:], rtol=0, atol=0)
        for capacity in context.manager.compilation_config.cudagraph_capture_sizes:
            context._forward(capacity)
        workspace.lock_workspace()
        freeze_kernel_resolution("DSpark context capacities are prewarmed")
        try:
            context.capture()
            for rows in (7, 3, 6, 1):
                # New allocations on every call ensure graphs never bind the
                # target's transient auxiliary outputs.
                aux = [
                    torch.randn(rows, 128, dtype=torch.bfloat16, generator=generator)
                    for _ in range(2)
                ]
                positions.fill_(1000000)  # stale padding must not read this RoPE row
                positions[:rows].copy_(torch.arange(rows, dtype=torch.int64) + rows)
                slots[0].copy_(torch.arange(8, dtype=torch.int64) + 32)
                slots[1].copy_(torch.arange(8, dtype=torch.int64) + 64)
                slots[:, rows - 1] = -1  # rejected suffix
                if rows > 1:
                    slots[1, 0] = -1  # group-specific nonresident/PAD row
                main_x = model.combine_hidden_states(torch.cat(aux, dim=-1))
                expected = []
                for i, layer in enumerate(model.layers):
                    attn = layer.attn
                    attn.swa_cache_layer.kv_cache.fill_(91)
                    # Oracle uses the original fused checkpoint projection.
                    kv = attn.kv_norm(attn.fused_wqa_wkv(main_x)[:, 256:])
                    group = 0 if layer_groups is None else layer_groups[i]
                    attn.insert_context_kv(kv, positions[:rows], slots[group, :rows])
                    expected.append(attn.swa_cache_layer.kv_cache.clone())
                    attn.swa_cache_layer.kv_cache.fill_(91)
                context.run(aux, rows)
                torch.cuda.synchronize()
                torch.testing.assert_close(hidden[:rows], main_x, rtol=0, atol=0)
                for layer, cache in zip(model.layers, expected):
                    torch.testing.assert_close(
                        layer.attn.swa_cache_layer.kv_cache, cache, rtol=0, atol=0
                    )
                # A restored cache must not consume changed sources or metadata.
                for source in aux:
                    source.fill_(float("nan"))
                slots.fill_(0)
                context.run(aux, rows, context_kv_is_restored=True)
                for layer, cache in zip(model.layers, expected):
                    torch.testing.assert_close(
                        layer.attn.swa_cache_layer.kv_cache, cache, rtol=0, atol=0
                    )
            assert not context.can_run(9)
            with pytest.raises(ValueError):
                context.run(aux, 9)
        finally:
            unfreeze_kernel_resolution()
            workspace.unlock_workspace()
