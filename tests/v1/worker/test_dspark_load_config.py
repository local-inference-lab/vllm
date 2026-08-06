# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from types import SimpleNamespace

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils


def test_dspark_loader_forwards_draft_load_config(monkeypatch):
    draft_load_config = object()
    draft_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            dflash_config={"causal": True},
        )
    )
    speculative_config = SimpleNamespace(
        attention_backend=AttentionBackendEnum.B12X_ATTN,
        draft_load_config=draft_load_config,
        draft_model_config=draft_model_config,
        kv_cache_dtype="fp8",
    )
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
        speculative_config=speculative_config,
    )

    def fake_replace(instance, **changes):
        result = copy.copy(instance)
        for name, value in changes.items():
            setattr(result, name, value)
        return result

    loaded_model = SimpleNamespace(
        model=SimpleNamespace(embed_tokens=None),
        lm_head=None,
    )
    captured = {}

    def fake_get_model(**kwargs):
        captured.update(kwargs)
        return loaded_model

    monkeypatch.setattr(dspark_utils, "replace", fake_replace)
    monkeypatch.setattr(dspark_utils, "get_model", fake_get_model)
    monkeypatch.setattr(
        dspark_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    target_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=None))
    result = dspark_utils.load_dspark_model(target_model, vllm_config)

    assert result is loaded_model
    assert captured["model_config"] is draft_model_config
    assert captured["load_config"] is draft_load_config
    assert (
        captured["vllm_config"].attention_config.backend
        is AttentionBackendEnum.B12X_ATTN
    )
    assert captured["vllm_config"].cache_config.cache_dtype == "fp8"
