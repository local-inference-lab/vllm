# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import vllm.config.virtual_tp as virtual_tp
import vllm.model_executor.layers.linear as linear
import vllm.model_executor.parameter as parameter
from vllm.config import ParallelConfig, set_current_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.config.virtual_tp import (
    VIRTUAL_TP_PLAN_ATTR,
    maybe_apply_b12x_virtual_tp_padding,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.virtual_tp import (
    get_virtual_tp_axis_local_size,
    get_virtual_tp_axis_original_size,
    get_virtual_tp_axis_padded_size,
    get_virtual_tp_axis_shard_size,
    get_virtual_tp_vocab_padding_size,
    pad_or_narrow_weight,
)
from vllm.transformers_utils.configs.qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class FakeModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="deepseek_v4",
            num_attention_heads=128,
            o_groups=16,
            moe_intermediate_size=3072,
            n_routed_experts=384,
            n_shared_experts=1,
            vocab_size=129280,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


class FakeDeepSeekV4MTPModelConfig(FakeModelConfig):
    def __init__(self):
        super().__init__()
        self.verified_attention_heads = None
        self.hf_text_config.model_type = "deepseek_mtp"
        self.hf_text_config.architectures = ["DeepSeekV4MTPModel"]
        self.hf_text_config.num_attention_heads = 64
        self.hf_text_config.moe_intermediate_size = 2048
        self.model_arch_config = self.get_model_arch_config()

    def verify_with_parallel_config(self, parallel_config):
        self.verified_attention_heads = self.hf_text_config.num_attention_heads
        assert self.verified_attention_heads % parallel_config.tensor_parallel_size == 0


class FakeDeepSeekV4DSparkModelConfig(FakeDeepSeekV4MTPModelConfig):
    def __init__(self):
        super().__init__()
        self.hf_text_config.model_type = "deepseek_v4"
        self.hf_text_config.architectures = ["DSparkDraftModel"]


class FakeNvfp4DeepSeekV4ModelConfig(FakeModelConfig):
    def is_nvfp4_quantized(self) -> bool:
        return True


class FakeGlmDsaModelConfig:
    def __init__(self):
        self.verified_attention_heads = None
        self.hf_text_config = SimpleNamespace(
            model_type="glm_moe_dsa",
            architectures=["GlmMoeDsaForCausalLM"],
            hidden_size=8192,
            num_attention_heads=64,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            kv_lora_rank=512,
            index_topk=256,
            moe_intermediate_size=2048,
            n_routed_experts=256,
            n_shared_experts=1,
            vocab_size=129280,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )

    def verify_with_parallel_config(self, parallel_config):
        self.verified_attention_heads = self.hf_text_config.num_attention_heads
        assert self.verified_attention_heads % parallel_config.tensor_parallel_size == 0


class FakeWrappedGlmDsaModelConfig(FakeGlmDsaModelConfig):
    def __init__(self):
        super().__init__()
        self.hf_config = SimpleNamespace(
            model_type="wrapper",
            architectures=["GlmMoeDsaForCausalLM"],
            text_config=self.hf_text_config,
            num_attention_heads=self.hf_text_config.num_attention_heads,
            moe_intermediate_size=self.hf_text_config.moe_intermediate_size,
            vocab_size=self.hf_text_config.vocab_size,
        )
        self.model_arch_config = self.get_model_arch_config()


class FakeUnsupportedModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="llama",
            num_attention_heads=64,
            moe_intermediate_size=2048,
            n_routed_experts=8,
            vocab_size=32000,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


class FakeAlignedGlmDsaModelConfig(FakeGlmDsaModelConfig):
    def __init__(self):
        super().__init__()
        self.hf_text_config.num_attention_heads = 96
        self.hf_text_config.moe_intermediate_size = 2112
        self.hf_text_config.vocab_size = 129408
        self.model_arch_config = self.get_model_arch_config()


class FakeMiniMaxM3ModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="minimax_m3_text",
            architectures=["MiniMaxM3SparseForCausalLM"],
            vocab_size=200064,
            hidden_size=6144,
            intermediate_size=3072,
            dense_intermediate_size=12288,
            shared_intermediate_size=3072,
            n_shared_experts=1,
            num_attention_heads=64,
            num_key_value_heads=4,
            sparse_attention_config={
                "sparse_num_index_heads": 4,
                "sparse_index_dim": 128,
            },
        )
        self.hf_config = SimpleNamespace(
            model_type="minimax_m3_vl",
            text_config=self.hf_text_config,
        )
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


class FakeQwen35MoeModelConfig:
    def __init__(
        self,
        *,
        moe_intermediate_size: int = 1024,
        mm_encoder_tp_mode: str = "weights",
    ):
        self.hf_text_config = Qwen3_5MoeTextConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=2,
            linear_num_key_heads=16,
            linear_num_value_heads=64,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=1024,
            vocab_size=248320,
            num_experts=512,
        )
        self.hf_config = SimpleNamespace(
            model_type="qwen3_5_moe",
            text_config=self.hf_text_config,
            vision_config=Qwen3_5MoeVisionConfig(
                hidden_size=1152,
                num_heads=16,
                intermediate_size=4304,
            ),
        )
        self.multimodal_config = SimpleNamespace(
            mm_encoder_tp_mode=mm_encoder_tp_mode,
        )
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )

    def is_nvfp4_quantized(self) -> bool:
        return True


class FakeQwen3NextModelConfig:
    def __init__(self):
        self.hf_text_config = Qwen3NextConfig()
        self.hf_config = self.hf_text_config
        self.multimodal_config = None
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )

    def is_nvfp4_quantized(self) -> bool:
        return True


class FakeDenseGQADraftModelConfig:
    def __init__(self):
        self.hf_text_config = SimpleNamespace(
            model_type="qwen3",
            architectures=["Qwen3DSparkModel"],
            vllm_virtual_tp_profile="dense-gqa",
            hidden_size=7168,
            intermediate_size=14336,
            num_attention_heads=64,
            num_key_value_heads=16,
            vocab_size=163840,
        )
        self.hf_config = self.hf_text_config
        self.model_arch_config = self.get_model_arch_config()

    def get_model_arch_config(self):
        return SimpleNamespace(
            total_num_attention_heads=self.hf_text_config.num_attention_heads,
        )


def _fake_vllm_config(
    *,
    model_config: Any | None = None,
    moe_backend: str = "b12x",
    tensor_parallel_size: int = 10,
    attention_backend: AttentionBackendEnum = AttentionBackendEnum.B12X_MLA_SPARSE,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=model_config or FakeModelConfig(),
        parallel_config=ParallelConfig(
            tensor_parallel_size=tensor_parallel_size,
        ),
        kernel_config=SimpleNamespace(moe_backend=moe_backend),
        attention_config=SimpleNamespace(
            backend=attention_backend,
        ),
    )


def test_b12x_virtual_tp_padding_deepseek_v4_pro_tp10():
    vllm_config = _fake_vllm_config()

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.num_attention_heads == 160
    assert text_config.o_groups == 20
    assert text_config.moe_intermediate_size == 3200
    assert text_config.vocab_size == 129280
    assert vllm_config.model_config.model_arch_config.total_num_attention_heads == 160

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["attention_heads"] == {
        "original_size": 128,
        "padded_size": 160,
        "tp_size": 10,
        "local_size": 16,
    }
    assert plan["output_groups"] == {
        "original_size": 16,
        "padded_size": 20,
        "tp_size": 10,
        "local_size": 2,
        "heads_per_group": 8,
    }
    assert plan["moe_intermediate_size"] == {
        "original_size": 3072,
        "padded_size": 3200,
        "tp_size": 10,
        "local_size": 320,
    }
    assert plan["moe_intermediate_size"]["local_size"] % 32 == 0
    assert plan["shared_expert_intermediate_size"] == {
        "original_size": 3072,
        "padded_size": 3840,
        "tp_size": 10,
        "local_size": 384,
    }
    assert plan["shared_expert_intermediate_size"]["local_size"] % 128 == 0
    assert plan["vocab_size"] == {
        "original_size": 129280,
        "padded_size": 129280,
        "tp_size": 10,
        "local_size": 12928,
        "padding_size": 320,
    }


def test_b12x_virtual_tp_padding_logs_when_triggered(
    monkeypatch: pytest.MonkeyPatch,
):
    vllm_config = _fake_vllm_config()
    logs: list[str] = []

    def warning(message: str, *args: Any) -> None:
        logs.append(message % args)

    monkeypatch.setattr(virtual_tp.logger, "warning", warning)
    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    assert any("Automatically enabled B12X virtual TP padding" in log for log in logs)
    assert any("attention heads 128 -> 160" in log for log in logs)
    assert any("MoE intermediate size 3072 -> 3200" in log for log in logs)


def test_b12x_virtual_tp_vocab_padding_deepseek_v4_pro_tp3():
    vllm_config = _fake_vllm_config(tensor_parallel_size=3)

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.num_attention_heads == 144
    assert text_config.vocab_size == 129280

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["attention_heads"] == {
        "original_size": 128,
        "padded_size": 144,
        "tp_size": 3,
        "local_size": 48,
    }
    assert plan["vocab_size"] == {
        "original_size": 129280,
        "padded_size": 129408,
        "tp_size": 3,
        "local_size": 43136,
        "padding_size": 192,
    }
    assert plan["output_groups"] == {
        "original_size": 16,
        "padded_size": 18,
        "tp_size": 3,
        "local_size": 6,
        "heads_per_group": 8,
    }


def test_b12x_virtual_tp_moe_padding_deepseek_v4_flash_tp3():
    vllm_config = _fake_vllm_config(tensor_parallel_size=3)
    vllm_config.model_config.hf_text_config.moe_intermediate_size = 2048

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.moe_intermediate_size == 2112

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["moe_intermediate_size"] == {
        "original_size": 2048,
        "padded_size": 2112,
        "tp_size": 3,
        "local_size": 704,
    }


def test_b12x_virtual_tp_deepseek_v4_uses_reduced_nvfp4_alignment():
    model_config = FakeNvfp4DeepSeekV4ModelConfig()
    model_config.hf_text_config.moe_intermediate_size = 2048
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=3,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    plan = getattr(model_config.hf_text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["moe_intermediate_size"] == {
        "original_size": 2048,
        "padded_size": 2064,
        "tp_size": 3,
        "local_size": 688,
    }


def test_b12x_virtual_tp_padding_deepseek_v4_mtp_precedes_validation():
    target_model_config = FakeModelConfig()
    draft_model_config = FakeDeepSeekV4MTPModelConfig()
    spec_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
    )
    spec_config.method = "mtp"
    spec_config.target_model_config = target_model_config
    spec_config.draft_model_config = draft_model_config
    spec_config.draft_parallel_config = ParallelConfig(tensor_parallel_size=3)

    spec_config._verify_args()

    assert draft_model_config.verified_attention_heads == 72


def test_b12x_virtual_tp_padding_deepseek_v4_dspark_precedes_validation():
    target_model_config = FakeModelConfig()
    draft_model_config = FakeDeepSeekV4DSparkModelConfig()
    spec_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
    )
    spec_config.method = "dspark"
    spec_config.target_model_config = target_model_config
    spec_config.draft_model_config = draft_model_config
    spec_config.draft_parallel_config = ParallelConfig(tensor_parallel_size=3)

    spec_config._verify_args()

    assert draft_model_config.verified_attention_heads == 72


def test_b12x_virtual_tp_padding_glm_dsa_tp6():
    vllm_config = _fake_vllm_config(
        model_config=FakeGlmDsaModelConfig(),
        tensor_parallel_size=6,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.num_attention_heads == 66
    assert text_config.moe_intermediate_size == 2112
    assert text_config.vocab_size == 129280
    assert text_config.original_num_attention_heads == 64
    assert text_config.original_moe_intermediate_size == 2048
    assert vllm_config.model_config.model_arch_config.total_num_attention_heads == 66

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert "output_groups" not in plan
    assert "shared_expert_intermediate_size" not in plan
    assert plan["attention_heads"] == {
        "original_size": 64,
        "padded_size": 66,
        "tp_size": 6,
        "local_size": 11,
    }
    assert plan["moe_intermediate_size"] == {
        "original_size": 2048,
        "padded_size": 2112,
        "tp_size": 6,
        "local_size": 352,
    }
    assert plan["vocab_size"] == {
        "original_size": 129280,
        "padded_size": 129408,
        "tp_size": 6,
        "local_size": 21568,
        "padding_size": 192,
    }


def test_b12x_virtual_tp_padding_glm_dsa_draft_tp6():
    target_model_config = FakeGlmDsaModelConfig()
    draft_model_config = FakeGlmDsaModelConfig()
    spec_config = SimpleNamespace(
        method="mtp",
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        draft_parallel_config=ParallelConfig(
            tensor_parallel_size=6,
        ),
    )

    SpeculativeConfig._maybe_apply_virtual_tp_to_draft(cast(Any, spec_config))

    text_config = draft_model_config.hf_text_config
    assert text_config.num_attention_heads == 66
    assert text_config.moe_intermediate_size == 2112
    assert draft_model_config.model_arch_config.total_num_attention_heads == 66


def test_b12x_virtual_tp_padding_glm_dsa_draft_precedes_validation():
    target_model_config = FakeGlmDsaModelConfig()
    draft_model_config = FakeGlmDsaModelConfig()
    spec_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=1,
    )
    spec_config.method = "mtp"
    spec_config.target_model_config = target_model_config
    spec_config.draft_model_config = draft_model_config
    spec_config.draft_parallel_config = ParallelConfig(tensor_parallel_size=6)

    spec_config._verify_args()

    assert draft_model_config.verified_attention_heads == 66


def test_dense_gqa_dspark_draft_virtual_tp12_padding():
    draft_model_config = FakeDenseGQADraftModelConfig()
    spec_config = SimpleNamespace(
        method="dspark",
        draft_model_config=draft_model_config,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=12),
        target_model_config=object(),
    )

    SpeculativeConfig._maybe_apply_virtual_tp_to_draft(cast(Any, spec_config))

    config = draft_model_config.hf_text_config
    plan = getattr(config, VIRTUAL_TP_PLAN_ATTR)
    assert config.num_attention_heads == 96
    assert config.num_key_value_heads == 24
    assert config.intermediate_size == 14592
    assert config.original_num_attention_heads == 64
    assert config.original_num_key_value_heads == 16
    assert config.original_intermediate_size == 14336
    assert plan["attention_heads"]["local_size"] == 8
    assert plan["kv_heads"]["local_size"] == 2
    assert plan["dense_intermediate_size"]["local_size"] == 1216
    assert plan["vocab_size"] == {
        "original_size": 163840,
        "padded_size": 163968,
        "tp_size": 12,
        "local_size": 13664,
        "padding_size": 192,
    }


def test_dense_gqa_dspark_draft_skips_natively_aligned_tp16():
    draft_model_config = FakeDenseGQADraftModelConfig()
    spec_config = SimpleNamespace(
        method="dspark",
        draft_model_config=draft_model_config,
        draft_parallel_config=SimpleNamespace(tensor_parallel_size=16),
        target_model_config=object(),
    )

    SpeculativeConfig._maybe_apply_virtual_tp_to_draft(cast(Any, spec_config))

    config = draft_model_config.hf_text_config
    assert not hasattr(config, VIRTUAL_TP_PLAN_ATTR)
    assert config.num_attention_heads == 64
    assert config.num_key_value_heads == 16
    assert config.intermediate_size == 14336


def test_b12x_virtual_tp_padding_minimax_m3_tp3_only():
    vllm_config = _fake_vllm_config(
        model_config=FakeMiniMaxM3ModelConfig(),
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.B12X_ATTN,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert text_config.num_attention_heads == 96
    assert text_config.num_key_value_heads == 4
    assert text_config.intermediate_size == 3072
    assert text_config.dense_intermediate_size == 12288
    assert vllm_config.model_config.model_arch_config.total_num_attention_heads == 96

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert getattr(vllm_config.model_config.hf_config, VIRTUAL_TP_PLAN_ATTR) is plan
    assert plan["model_type"] == "minimax_m3"
    assert plan["attention_heads"] == {
        "original_size": 64,
        "padded_size": 96,
        "tp_size": 3,
        "local_size": 32,
    }
    assert plan["kv_heads"] == {
        "original_size": 4,
        "padded_size": 6,
        "tp_size": 3,
        "local_size": 2,
        "q_heads_per_kv": 16,
    }
    assert plan["index_heads"] == {
        "original_size": 4,
        "padded_size": 6,
        "tp_size": 3,
        "local_size": 2,
    }
    assert plan["moe_intermediate_size"] == {
        "original_size": 3072,
        "padded_size": 3072,
        "tp_size": 3,
        "local_size": 1024,
    }
    assert plan["dense_intermediate_size"] == {
        "original_size": 12288,
        "padded_size": 12288,
        "tp_size": 3,
        "local_size": 4096,
    }
    assert plan["vocab_size"] == {
        "original_size": 200064,
        "padded_size": 200064,
        "tp_size": 3,
        "local_size": 66688,
        "padding_size": 192,
    }


def test_b12x_virtual_tp_padding_qwen35_moe_tp3():
    model_config = FakeQwen35MoeModelConfig()
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = model_config.hf_text_config
    vision_config = model_config.hf_config.vision_config
    assert text_config.num_attention_heads == 48
    assert text_config.num_key_value_heads == 3
    assert text_config.linear_num_key_heads == 18
    assert text_config.linear_num_value_heads == 72
    assert text_config.moe_intermediate_size == 1056
    assert text_config.shared_expert_intermediate_size == 1056
    assert text_config.hidden_size == 4096
    assert text_config.vocab_size == 248320
    assert vision_config.num_heads == 18
    assert vision_config.hidden_size == 1152
    assert vision_config.intermediate_size == 4320
    assert model_config.model_arch_config.total_num_attention_heads == 48

    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert getattr(model_config.hf_config, VIRTUAL_TP_PLAN_ATTR) is plan
    assert getattr(vision_config, VIRTUAL_TP_PLAN_ATTR) is plan
    assert plan["model_type"] == "gqa-gdn-moe"
    assert plan["attention_heads"] == {
        "original_size": 32,
        "padded_size": 48,
        "tp_size": 3,
        "local_size": 16,
    }
    assert plan["kv_heads"] == {
        "original_size": 2,
        "padded_size": 3,
        "tp_size": 3,
        "local_size": 1,
        "q_heads_per_kv": 16,
    }
    assert plan["linear_attention_key_heads"] == {
        "original_size": 16,
        "padded_size": 18,
        "tp_size": 3,
        "local_size": 6,
        "value_heads_per_key": 4,
    }
    assert plan["linear_attention_value_heads"] == {
        "original_size": 64,
        "padded_size": 72,
        "tp_size": 3,
        "local_size": 24,
    }
    assert plan["moe_intermediate_size"] == {
        "original_size": 1024,
        "padded_size": 1056,
        "tp_size": 3,
        "local_size": 352,
    }
    assert plan["shared_expert_intermediate_size"] == {
        "original_size": 1024,
        "padded_size": 1056,
        "tp_size": 3,
        "local_size": 352,
    }
    assert plan["mtp_projection_size"] == {
        "original_size": 4096,
        "padded_size": 4098,
        "tp_size": 3,
        "local_size": 1366,
    }
    assert plan["vocab_size"] == {
        "original_size": 248320,
        "padded_size": 248448,
        "tp_size": 3,
        "local_size": 82816,
        "padding_size": 192,
    }
    assert plan["vision_attention_heads"] == {
        "original_size": 16,
        "padded_size": 18,
        "tp_size": 3,
        "local_size": 6,
    }
    assert plan["vision_attention_projection_size"] == {
        "original_size": 1152,
        "padded_size": 1296,
        "tp_size": 3,
        "local_size": 432,
        "head_size": 72,
    }
    assert plan["vision_intermediate_size"] == {
        "original_size": 4304,
        "padded_size": 4320,
        "tp_size": 3,
        "local_size": 1440,
    }
    assert (
        get_virtual_tp_axis_original_size(
            "linear_attention_key_heads", -1, config=text_config
        )
        == 16
    )
    assert (
        get_virtual_tp_axis_padded_size(
            "vision_attention_projection_size", -1, config=vision_config
        )
        == 1296
    )


def test_b12x_virtual_tp_profile_is_reused_by_qwen3_next():
    model_config = FakeQwen3NextModelConfig()
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    config = model_config.hf_text_config
    plan = getattr(config, VIRTUAL_TP_PLAN_ATTR)
    assert config.num_attention_heads == 24
    assert config.num_key_value_heads == 3
    assert config.linear_num_key_heads == 18
    assert config.linear_num_value_heads == 36
    assert config.moe_intermediate_size == 528
    assert config.shared_expert_intermediate_size == 528
    assert config.intermediate_size == 5664
    assert plan["mtp_projection_size"]["padded_size"] == 2049
    assert plan["vocab_size"]["padded_size"] == 152064


def test_b12x_virtual_tp_qwen35_uses_reduced_nvfp4_alignment(
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = FakeQwen35MoeModelConfig(moe_intermediate_size=1060)
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    plan = getattr(model_config.hf_text_config, VIRTUAL_TP_PLAN_ATTR)
    assert plan["moe_intermediate_size"]["local_size"] == 368

    monkeypatch.setenv("B12X_FORCE_MOE_A8", "1")
    a8_model_config = FakeQwen35MoeModelConfig(moe_intermediate_size=1060)
    a8_vllm_config = _fake_vllm_config(
        model_config=a8_model_config,
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )
    maybe_apply_b12x_virtual_tp_padding(cast(Any, a8_vllm_config))

    a8_plan = getattr(a8_model_config.hf_text_config, VIRTUAL_TP_PLAN_ATTR)
    assert a8_plan["moe_intermediate_size"]["local_size"] == 384


def test_b12x_virtual_tp_qwen35_keeps_data_parallel_vision_shapes():
    model_config = FakeQwen35MoeModelConfig(mm_encoder_tp_mode="data")
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    vision_config = model_config.hf_config.vision_config
    plan = getattr(vision_config, VIRTUAL_TP_PLAN_ATTR)
    assert vision_config.num_heads == 16
    assert vision_config.intermediate_size == 4304
    assert plan["vision_attention_projection_size"]["padded_size"] == 1152


def test_b12x_virtual_tp_qwen35_preserves_kv_head_replication():
    model_config = FakeQwen35MoeModelConfig()
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=4,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = model_config.hf_text_config
    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert text_config.num_attention_heads == 32
    assert text_config.num_key_value_heads == 2
    assert plan["kv_heads"] == {
        "original_size": 2,
        "padded_size": 2,
        "tp_size": 4,
        "local_size": 1,
        "q_heads_per_kv": 16,
    }
    assert plan["vision_intermediate_size"]["padded_size"] == 4352


@pytest.mark.parametrize(
    ("output_sizes", "loaded_output_sizes", "loaded_shard_id", "expected"),
    [
        (
            [6, 6, 12],
            [5, 5, 10],
            (0, 1, 2),
            [4, 0, 9, 0, 18, 19, 0, 0],
        ),
        ([24], [20], None, [16, 17, 18, 19, 0, 0, 0, 0]),
    ],
)
def test_merged_column_uses_logical_checkpoint_segment_sizes(
    monkeypatch: pytest.MonkeyPatch,
    output_sizes: list[int],
    loaded_output_sizes: list[int],
    loaded_shard_id: tuple[int, ...] | None,
    expected: list[int],
):
    model_config = FakeQwen35MoeModelConfig()
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )
    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 2)

    with set_current_vllm_config(cast(Any, vllm_config)):
        layer = MergedColumnParallelLinear(
            input_size=1,
            output_sizes=output_sizes,
            loaded_output_sizes=loaded_output_sizes,
            bias=False,
        )
        loaded_weight = torch.arange(
            sum(loaded_output_sizes), dtype=layer.weight.dtype
        ).unsqueeze(1)
        layer.weight.weight_loader(
            layer.weight,
            loaded_weight,
            loaded_shard_id,
        )

    torch.testing.assert_close(
        layer.weight[:, 0],
        torch.tensor(expected, dtype=layer.weight.dtype),
    )


def test_qkv_parallel_uses_logical_checkpoint_head_counts(
    monkeypatch: pytest.MonkeyPatch,
):
    model_config = FakeQwen35MoeModelConfig()
    vllm_config = _fake_vllm_config(
        model_config=model_config,
        tensor_parallel_size=3,
        attention_backend=AttentionBackendEnum.FLASH_ATTN,
    )
    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))
    monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 3)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 1)

    with set_current_vllm_config(cast(Any, vllm_config)):
        layer = QKVParallelLinear(
            hidden_size=1,
            head_size=1,
            total_num_heads=6,
            total_num_kv_heads=3,
            loaded_total_num_heads=4,
            loaded_total_num_kv_heads=2,
            bias=False,
        )
        loaded_weight = torch.arange(8, dtype=layer.weight.dtype).unsqueeze(1)
        layer.weight.weight_loader(layer.weight, loaded_weight)

    torch.testing.assert_close(
        layer.weight[:, 0],
        torch.tensor([2, 3, 5, 7], dtype=layer.weight.dtype),
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_b12x_virtual_tp_padding_minimax_m3_skips_working_tp(tp_size: int):
    vllm_config = _fake_vllm_config(
        model_config=FakeMiniMaxM3ModelConfig(),
        tensor_parallel_size=tp_size,
        attention_backend=AttentionBackendEnum.B12X_ATTN,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert not hasattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert text_config.num_attention_heads == 64
    assert text_config.num_key_value_heads == 4
    assert text_config.intermediate_size == 3072
    assert text_config.dense_intermediate_size == 12288
    assert vllm_config.model_config.model_arch_config.total_num_attention_heads == 64


def test_b12x_virtual_tp_padding_updates_distinct_hf_configs():
    vllm_config = _fake_vllm_config(
        model_config=FakeWrappedGlmDsaModelConfig(),
        tensor_parallel_size=6,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    root_config = vllm_config.model_config.hf_config
    text_config = vllm_config.model_config.hf_text_config
    plan = getattr(text_config, VIRTUAL_TP_PLAN_ATTR)

    assert root_config.num_attention_heads == 66
    assert root_config.moe_intermediate_size == 2112
    assert root_config.vocab_size == 129280
    assert getattr(root_config, VIRTUAL_TP_PLAN_ATTR) is plan
    assert getattr(root_config.text_config, VIRTUAL_TP_PLAN_ATTR) is plan


def test_b12x_virtual_tp_padding_skips_aligned_config():
    vllm_config = _fake_vllm_config(
        model_config=FakeAlignedGlmDsaModelConfig(),
        tensor_parallel_size=6,
    )

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert not hasattr(text_config, VIRTUAL_TP_PLAN_ATTR)
    assert text_config.num_attention_heads == 96
    assert text_config.moe_intermediate_size == 2112
    assert text_config.vocab_size == 129408


def test_b12x_virtual_tp_padding_rejects_flashinfer_moe():
    vllm_config = _fake_vllm_config(moe_backend="flashinfer_b12x")

    with pytest.raises(ValueError, match="native B12X MoE"):
        maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))


def test_b12x_virtual_tp_padding_skips_unsupported_models():
    vllm_config = _fake_vllm_config(model_config=FakeUnsupportedModelConfig())

    maybe_apply_b12x_virtual_tp_padding(cast(Any, vllm_config))

    text_config = vllm_config.model_config.hf_text_config
    assert not hasattr(text_config, VIRTUAL_TP_PLAN_ATTR)


def test_virtual_tp_pad_or_narrow_weight_zero_fills_tail():
    current_config = _fake_vllm_config()
    maybe_apply_b12x_virtual_tp_padding(cast(Any, current_config))
    loaded_weight = torch.arange(6).reshape(3, 2)

    with set_current_vllm_config(cast(Any, current_config)):
        padded = pad_or_narrow_weight(loaded_weight, 0, 2, 3)
        local_moe_size = get_virtual_tp_axis_local_size("moe_intermediate_size", -1)
        vocab_padding_size = get_virtual_tp_vocab_padding_size(-1)

    expected = torch.tensor([[4, 5], [0, 0], [0, 0]])
    assert torch.equal(padded, expected)
    assert local_moe_size == 320
    assert vocab_padding_size == 320


def test_virtual_tp_axis_shard_size_uses_stored_tensor_units():
    current_config = _fake_vllm_config()
    maybe_apply_b12x_virtual_tp_padding(cast(Any, current_config))

    with set_current_vllm_config(cast(Any, current_config)):
        assert get_virtual_tp_axis_shard_size("moe_intermediate_size", 320) == 320
        assert get_virtual_tp_axis_shard_size("moe_intermediate_size", 160) == 160
        assert get_virtual_tp_axis_shard_size("moe_intermediate_size", 512) == 320


def test_virtual_tp_pad_or_narrow_weight_is_strict_without_plan():
    loaded_weight = torch.arange(6).reshape(3, 2)

    with pytest.raises(RuntimeError):
        pad_or_narrow_weight(loaded_weight, 0, 2, 3)
