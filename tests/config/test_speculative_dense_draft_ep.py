# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense draft models must not inherit the MoE target's expert parallelism."""

import pytest
from transformers import LlamaConfig, Qwen2MoeConfig

from vllm.config import ModelConfig, ParallelConfig
from vllm.config.speculative import SpeculativeConfig

pytestmark = pytest.mark.cpu_test


def checkpoint(path, moe):
    config_type = Qwen2MoeConfig if moe else LlamaConfig
    config = config_type(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=64,
        max_position_embeddings=4096,
        architectures=["Qwen2MoeForCausalLM" if moe else "LlamaForCausalLM"],
        **(
            dict(
                num_experts=8,
                num_experts_per_tok=2,
                moe_intermediate_size=64,
                shared_expert_intermediate_size=64,
            )
            if moe
            else {}
        ),
    )
    config.save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("target_ep", [False, True])
@pytest.mark.parametrize("draft_moe", [False, True])
def test_draft_parallelism_matches_its_own_architecture(
    tmp_path, tp, target_ep, draft_moe
):
    target_path = checkpoint(tmp_path / "target", True)
    draft_path = checkpoint(tmp_path / "draft", draft_moe)
    target = ModelConfig(
        model=target_path,
        dtype="bfloat16",
        max_model_len=4096,
        skip_tokenizer_init=True,
    )
    parallel = ParallelConfig(
        tensor_parallel_size=tp,
        enable_expert_parallel=target_ep,
        distributed_executor_backend="mp",
    )
    config = SpeculativeConfig(
        target_model_config=target,
        target_parallel_config=parallel,
        model=draft_path,
        method="draft_model",
        num_speculative_tokens=3,
    )
    assert config.draft_model_config.is_moe is draft_moe
    assert config.draft_parallel_config.enable_expert_parallel is (
        target_ep and draft_moe
    )
    assert config.draft_parallel_config.tensor_parallel_size == tp
    assert parallel.enable_expert_parallel is target_ep
    assert config.draft_parallel_config is not parallel
    config.draft_model_config.verify_with_parallel_config(config.draft_parallel_config)
    target.verify_with_parallel_config(parallel)
