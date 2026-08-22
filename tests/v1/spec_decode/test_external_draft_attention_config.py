# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.lora.layers.base import BaseLayerWithLoRA
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils


def test_dspark_loader_uses_external_draft_parallel_geometry(monkeypatch) -> None:
    """The external draft model constructor receives its DCP1 configuration."""
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    target_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=draft_model_config,
            attention_backend=AttentionBackendEnum.B12X_MLA,
        )
    )
    draft_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(backend=None, use_non_causal=False),
        quant_config=object(),
        lora_config=object(),
    )
    draft_quant_config = object()
    captured = {}

    class ModelCaptured(RuntimeError):
        pass

    def fake_get_model(*, vllm_config, model_config):
        captured["vllm_config"] = vllm_config
        captured["model_config"] = model_config
        raise ModelCaptured

    monkeypatch.setattr(
        dspark_utils,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )
    monkeypatch.setattr(dspark_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _config: True,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.utils.get_draft_quant_config",
        lambda config: draft_quant_config if config is target_config else None,
    )

    with pytest.raises(ModelCaptured):
        dspark_utils.load_dspark_model(object(), target_config)

    assert captured["model_config"] is draft_model_config
    loaded_config = captured["vllm_config"]
    assert loaded_config.parallel_config.decode_context_parallel_size == 1
    assert loaded_config.attention_config.backend == AttentionBackendEnum.B12X_MLA
    assert loaded_config.attention_config.use_non_causal
    assert loaded_config.quant_config is draft_quant_config
    assert loaded_config.lora_config is None


def test_dspark_loader_shares_unwrapped_lora_embed_and_lm_head(monkeypatch) -> None:
    """The external draft shares base layers, never target LoRA wrappers."""

    target_embed_base = nn.Identity()
    target_lm_head_base = nn.Identity()
    target_embed = BaseLayerWithLoRA()
    target_embed.base_layer = target_embed_base
    target_lm_head = BaseLayerWithLoRA()
    target_lm_head.base_layer = target_lm_head_base
    target_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=target_embed),
        lm_head=target_lm_head,
    )
    draft_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=nn.Identity()),
        lm_head=nn.Identity(),
    )
    draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    target_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=draft_model_config,
            attention_backend=AttentionBackendEnum.B12X_MLA,
        )
    )
    draft_config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None, use_non_causal=False),
        quant_config=object(),
        lora_config=object(),
    )

    monkeypatch.setattr(
        dspark_utils,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )
    monkeypatch.setattr(dspark_utils, "get_model", lambda **_kwargs: draft_model)
    monkeypatch.setattr(
        dspark_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(dspark_utils, "_should_share", lambda *_args: True)
    monkeypatch.setattr(
        "vllm.compilation.backends.set_model_tag",
        lambda _tag: nullcontext(),
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_dflash.dflash_has_any_non_causal",
        lambda _config: False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.utils.get_draft_quant_config",
        lambda _config: None,
    )

    loaded = dspark_utils.load_dspark_model(target_model, target_config)

    assert loaded is draft_model
    assert loaded.model.embed_tokens is target_embed_base
    assert loaded.lm_head is target_lm_head_base
    assert loaded.model.embed_tokens is not target_embed
    assert loaded.lm_head is not target_lm_head
    assert draft_config.lora_config is None


def test_dflash_attention_metadata_uses_external_draft_geometry(monkeypatch) -> None:
    """Attention metadata retains the external draft's DCP1 geometry."""
    target_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=16)
    )
    draft_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        attention_config=SimpleNamespace(
            backend=AttentionBackendEnum.B12X_MLA,
            use_non_causal=False,
        ),
    )
    monkeypatch.setattr(
        dflash_speculator,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )
    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator.requires_non_causal = True
    speculator.draft_kv_window = None
    speculator.draft_kv_window_block_size = None

    attention_config = speculator.attn_vllm_config

    assert attention_config.parallel_config.decode_context_parallel_size == 1
    assert attention_config.attention_config.use_non_causal
    assert not draft_config.attention_config.use_non_causal


def test_bounded_draft_attention_preserves_derived_model_attributes(
    monkeypatch,
) -> None:
    """Bounded plan sizing copies runtime objects without reconstructing them."""
    draft_model_config = SimpleNamespace(
        max_model_len=1_048_576,
        model_arch_config=object(),
    )
    draft_config = SimpleNamespace(
        model_config=draft_model_config,
        attention_config=SimpleNamespace(use_non_causal=False),
    )
    target_config = object()
    monkeypatch.setattr(
        dflash_speculator,
        "_create_draft_vllm_config",
        lambda config: draft_config if config is target_config else None,
    )
    speculator = object.__new__(dflash_speculator.DFlashSpeculator)
    speculator.vllm_config = target_config
    speculator.requires_non_causal = True
    speculator.draft_kv_window = 65_536
    speculator.draft_kv_window_block_size = 768

    attention_config = speculator.attn_vllm_config

    assert draft_model_config.max_model_len == 1_048_576
    assert attention_config.model_config.max_model_len == 65_536 + 768 - 1
    assert attention_config.model_config.model_arch_config is not None
    assert attention_config.attention_config.use_non_causal
