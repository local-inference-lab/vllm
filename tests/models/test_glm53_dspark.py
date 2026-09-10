# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.glm53_dspark import (
    Afterburner,
    Glm53DSparkForCausalLM,
    Glm53DSparkModel,
)
from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4Model
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.glm53_dspark import Glm53DSparkConfig
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)


@pytest.fixture
def single_rank_linears(monkeypatch):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)


def _reference_afterburner(module, hidden):
    hidden = F.linear(hidden, module.projection.weight, module.projection.bias)
    for block in module.blocks:
        normalized = hidden.float()
        normalized = (
            normalized
            * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + 1e-6)
            * block.norm.weight.float()
        )
        gate, up = F.linear(normalized.to(hidden.dtype), block.gate_up.weight).chunk(
            2, -1
        )
        output = F.linear(F.silu(gate) * up, block.down.weight)
        hidden = hidden + output if block.residual else output
    return hidden


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("residual", [False, True])
def test_afterburner_matches_equations_and_cuda_graph(
    default_vllm_config, single_rank_linears, device, residual
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = SimpleNamespace(
        afterburner_depth=4,
        afterburner_intermediate_size=64,
        afterburner_residual=residual,
    )
    torch.manual_seed(23)
    module = Afterburner(32, 32, config, "afterburner").to(
        device=device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(std=0.1)
        for layer in module.modules():
            if isinstance(layer, RMSNorm):
                layer.float()
                layer.weight.fill_(1.001)
        hidden = torch.randn(14, 32, dtype=torch.bfloat16, device=device)
        expected = _reference_afterburner(module, hidden)
        actual = module(hidden)
        torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.004)
        if device == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    module(hidden)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = module(hidden)
            for _ in range(3):
                hidden.normal_()
                expected = module(hidden)
                graph.replay()
                torch.testing.assert_close(captured, expected, atol=0, rtol=0)


def test_context_afterburners_keep_taps_separate_and_ordered(
    default_vllm_config, single_rank_linears
):
    model = Glm53DSparkModel.__new__(Glm53DSparkModel)
    nn.Module.__init__(model)
    config = SimpleNamespace(
        afterburner_depth=1, afterburner_intermediate_size=8, afterburner_residual=True
    )
    model.context_afterburners = nn.ModuleList(
        [Afterburner(8, 8, config, f"tap{i}") for i in range(3)]
    )
    model.main_proj = nn.Linear(24, 8, bias=False)
    model.main_norm = nn.Identity()
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(std=0.1)
        hidden = torch.randn(5, 24)
        expected = model.main_proj(
            torch.cat(
                [
                    _reference_afterburner(module, tap)
                    for module, tap in zip(
                        model.context_afterburners, hidden.chunk(3, -1)
                    )
                ],
                -1,
            )
        )
        torch.testing.assert_close(model.combine_hidden_states(hidden), expected)


@pytest.mark.parametrize("num_drafts", [1, 4, 7, 12])
def test_noise_positions_follow_runtime_depth_and_not_anchor_token_id(
    monkeypatch, default_vllm_config, single_rank_linears, num_drafts
):
    def init_trunk(self, *, vllm_config, prefix):
        nn.Module.__init__(self)
        self.config = SimpleNamespace(
            hidden_size=4,
            target_hidden_size=4,
            dspark_block_size=7,
            afterburner_depth=1,
            afterburner_intermediate_size=8,
            afterburner_residual=True,
        )
        self.target_layer_ids = (42, 43, 44)
        self.embed_tokens = nn.Embedding(16, 4)

    monkeypatch.setattr(DSparkDeepseekV4Model, "__init__", init_trunk)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=num_drafts),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1, max_num_seqs=2),
    )
    model = Glm53DSparkModel(vllm_config=config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.1)
    monkeypatch.setattr(
        DSparkDeepseekV4Model,
        "forward",
        lambda self, ids, positions, inputs_embeds: inputs_embeds,
    )
    ids = torch.full((2 * num_drafts,), 15)
    actual = model(ids, torch.arange(ids.numel()))
    expected_anchor = model.token_afterburner(model.embed_tokens(ids[[0, num_drafts]]))
    torch.testing.assert_close(actual[[0, num_drafts]], expected_anchor)
    torch.testing.assert_close(
        actual[model._noise_positions.squeeze(-1)],
        model.noise_embedding.expand(2 * (num_drafts - 1), -1),
    )


def test_output_afterburner_runs_after_norm_and_before_head():
    model = Glm53DSparkForCausalLM.__new__(Glm53DSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.norm = nn.LayerNorm(4)
    model.model.output_afterburner = nn.Linear(4, 4)
    model.lm_head = nn.Linear(4, 16, bias=False)
    model.logits_processor = lambda head, hidden: head(hidden)
    hidden = torch.randn(3, 4)
    expected = model.lm_head(model.model.output_afterburner(model.model.norm(hidden)))
    torch.testing.assert_close(model.compute_draft_logits(hidden), expected)


def test_draft_constructor_uses_own_config_without_changing_verifier(
    monkeypatch, default_vllm_config
):
    from vllm.config import get_current_vllm_config
    from vllm.models.deepseek_v4.nvidia.dspark import DSparkDeepseekV4ForCausalLM

    draft = SimpleNamespace(hf_config=Glm53DSparkConfig(o_lora_rank=1024))
    default_vllm_config.speculative_config = SimpleNamespace(draft_model_config=draft)
    observed = []

    def construct(self, *, vllm_config, prefix):
        nn.Module.__init__(self)
        observed.append(vllm_config.model_config)
        assert get_current_vllm_config() is vllm_config
        assert vllm_config.model_config.hf_config.o_lora_rank == 1024

    monkeypatch.setattr(DSparkDeepseekV4ForCausalLM, "__init__", construct)
    Glm53DSparkForCausalLM(vllm_config=default_vllm_config)
    assert observed == [draft]
    assert default_vllm_config.model_config is None
    assert get_current_vllm_config() is default_vllm_config


@pytest.mark.parametrize(
    ("method", "model_type", "contract_streams"),
    [
        ("dspark", "glm53_dspark", True),
        ("dspark", "deepseek_v4", False),
        ("dflash", "qwen3", True),
        ("eagle3", "qwen3", False),
    ],
)
def test_target_contracts_mhc_streams_only_for_compatible_drafters(
    monkeypatch, default_vllm_config, method, model_type, contract_streams
):
    from vllm.models.glm5next.nvidia import model as target

    monkeypatch.setattr(
        target,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(target, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        target, "VocabParallelEmbedding", lambda n, d, **kwargs: nn.Embedding(n, d)
    )
    monkeypatch.setattr(target, "make_layers", lambda *a, **k: (0, 0, nn.ModuleList()))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=16,
                hidden_size=4,
                num_hidden_layers=0,
                num_attention_heads=1,
                rms_norm_eps=1e-6,
            )
        ),
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        speculative_config=SimpleNamespace(
            use_dflash=lambda: method == "dflash",
            use_dspark=lambda: method == "dspark",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=model_type)
            ),
        ),
    )
    model = target.Glm5NextModel(vllm_config=config)
    assert model.dflash_capture is contract_streams


@pytest.mark.parametrize("num_drafts", [4, 7, 12])
@pytest.mark.parametrize("adaptive", [False, True])
def test_glm53_dspark_config_keeps_own_architecture_precision_and_taps(
    tmp_path, num_drafts, adaptive
):
    draft = tmp_path / "draft"
    target = tmp_path / "target"
    draft.mkdir()
    target.mkdir()
    config = {
        "model_type": "glm53_dspark",
        "architectures": ["Glm53DSparkForCausalLM"],
        "hidden_size": 4096,
        "target_hidden_size": 4096,
        "vocab_size": 154880,
        "num_hidden_layers": 45,
        "n_mtp_layers": 3,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "qk_rope_head_dim": 64,
        "compress_ratios": [0] * 48,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "max_position_embeddings": 8192,
        "dspark_target_layer_ids": [42, 43, 44],
        "dspark_block_size": 7,
        "torch_dtype": "bfloat16",
        "expert_dtype": "bfloat16",
    }
    (draft / "config.json").write_text(json.dumps(config))
    target_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 8192,
        "num_hidden_layers": 45,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "vocab_size": 154880,
        "max_position_embeddings": 8192,
        "torch_dtype": "bfloat16",
    }
    (target / "config.json").write_text(json.dumps(target_config))
    loaded = get_config(draft, trust_remote_code=False)
    assert isinstance(loaded, Glm53DSparkConfig)
    spec = SpeculativeConfig(
        model=str(draft),
        method="dspark",
        num_speculative_tokens=num_drafts,
        enable_adaptive_verification=adaptive,
        target_model_config=ModelConfig(
            model=str(target), tokenizer_mode="skip", max_model_len=8192
        ),
        target_parallel_config=ParallelConfig(),
    )
    assert spec.draft_model_config.architectures == ["Glm53DSparkForCausalLM"]
    assert spec.num_speculative_tokens == num_drafts
    assert spec.enable_adaptive_verification is adaptive
    assert spec.draft_model_config.quantization is None
    assert spec.draft_model_config.use_mla
    assert spec.draft_model_config.get_head_size() == 512
    assert spec.draft_model_config.hf_config.router_dtype == "float32"
    assert get_eagle3_aux_layers_from_config(spec) == (43, 44, 45)


def test_draft_attention_metadata_uses_draft_geometry(default_vllm_config):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.vllm_config = default_vllm_config
    speculator.requires_non_causal = True
    speculator.draft_model_config = SimpleNamespace(
        hf_config=Glm53DSparkConfig(sliding_window=128, head_dim=512)
    )
    speculator.speculative_config = SimpleNamespace(
        attention_backend=AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
        kv_cache_dtype="fp8",
    )
    config = speculator.attn_vllm_config
    assert config.model_config is speculator.draft_model_config
    assert config.attention_config.use_non_causal
    assert (
        config.attention_config.backend
        is speculator.speculative_config.attention_backend
    )
    assert config.cache_config.cache_dtype == "fp8"
    assert config.quant_config is None
    assert default_vllm_config.model_config is None
    assert not default_vllm_config.attention_config.use_non_causal
    assert default_vllm_config.cache_config.cache_dtype == "auto"


@pytest.mark.parametrize("name", ["afterburner_depth", "afterburner_intermediate_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_afterburner_config_rejects_invalid_dimensions(name, value):
    with pytest.raises(ValueError, match="positive integer"):
        Glm53DSparkConfig(**{name: value})
