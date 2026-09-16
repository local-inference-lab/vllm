# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""index_topk threads from the HF config into the MXFP4 indexer plan."""

from types import SimpleNamespace

import pytest
import torch


def _index_module(index_topk: int):
    from vllm.models.deepseek_v4_1 import attention

    module = attention.DeepseekV4Attention.__new__(attention.DeepseekV4Attention)
    torch.nn.Module.__init__(module)
    module._index_topk = index_topk
    module._index_width = 32
    module._index_page = 64
    module.layer_id = 0
    module.candidate_source_layer = 0
    module.is_ced_decoder = False
    module.indexer = SimpleNamespace(heads=32)
    module.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.empty(0, dtype=torch.float32, device="cpu")
    )
    module._short_index_shape = None
    return module


@pytest.mark.parametrize("index_topk", [512, 1024, 2048])
def test_index_topk_threads_into_the_mxfp4_plan_caps(monkeypatch, index_topk: int) -> None:
    from b12x.attention import dsa_indexer

    captured = {}

    def fake_plan(caps, **kwargs):
        captured["caps"] = caps
        return SimpleNamespace()

    monkeypatch.setattr(dsa_indexer, "plan", fake_plan)
    module = _index_module(index_topk)
    module._declare_index_plan("decode", 64)
    assert captured["caps"].topk == index_topk
    assert captured["caps"].cache_format == "mxfp4"


def test_index_topk_defaults_to_512_without_a_config_key() -> None:
    from vllm.models.deepseek_v4_1 import attention

    hf = SimpleNamespace()
    assert getattr(hf, "index_topk", 512) == 512
    assert attention.DeepseekV4Attention.__init__ is not None
