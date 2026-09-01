# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config-only DFlash behavior.

``dflash_has_any_non_causal`` decides pre-build whether the draft needs a
non-causal-capable backend, so its branch table (explicit override, SWA-derived
per-layer causality, and the no-``layer_types`` fallback) is worth pinning.
"""

from types import SimpleNamespace

import pytest
import torch.nn as nn

from vllm.model_executor.models.qwen3_dflash import (
    _dflash_layer_causal,
    _get_dflash_fc_input_size,
    dflash_has_any_non_causal,
    dflash_target_rope_is_neox_style,
)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)


def _config(num_hidden_layers, layer_types=None, causal_override=None):
    dflash_config = None if causal_override is None else {"causal": causal_override}
    return SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        dflash_config=dflash_config,
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        # Override forces causality on every layer, ignoring layer_types.
        (_config(2, layer_types=["full_attention"] * 2, causal_override=True), False),
        # Override forces non-causal on every layer.
        (
            _config(2, layer_types=["sliding_attention"] * 2, causal_override=False),
            True,
        ),
        # SWA-derived: full-attention layers are non-causal.
        (_config(2, layer_types=["sliding_attention", "full_attention"]), True),
        # SWA-derived: all-sliding is fully causal.
        (_config(2, layer_types=["sliding_attention", "sliding_attention"]), False),
        # No layer_types -> non-causal fallback.
        (_config(2, layer_types=None), True),
        (_config(2, layer_types=[]), True),
    ],
)
def test_dflash_has_any_non_causal(config, expected):
    assert dflash_has_any_non_causal(config) is expected


def test_dflash_layer_causal_is_per_layer():
    config = _config(2, layer_types=["sliding_attention", "full_attention"])
    assert _dflash_layer_causal(config, 0) is True
    assert _dflash_layer_causal(config, 1) is False


def _vllm_config(**draft_config):
    config = SimpleNamespace(**draft_config)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        )
    )


def test_dflash_fc_uses_aux_layer_count():
    vllm_config = _vllm_config(
        num_hidden_layers=5,
        hidden_size=4096,
        target_hidden_size=None,
        target_layer_ids=[1, 17, 32],
    )

    assert _get_dflash_fc_input_size(vllm_config) == 3 * 4096


@pytest.mark.parametrize("config_name", ["dflash_config", "eagle_config"])
def test_eagle_aux_layers_preserves_legacy_layer_ids(config_name):
    layer_ids = [1, 17, 32]
    vllm_config = _vllm_config(
        **{config_name: {"layer_ids": layer_ids}},
    )

    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == tuple(
        layer_ids
    )


class _TargetRotaryModule(nn.Module):
    def __init__(self, is_neox_style: bool):
        super().__init__()
        self.is_neox_style = is_neox_style


class _TargetModel(nn.Module):
    def __init__(self, is_neox_style: bool):
        super().__init__()
        self.rotary = _TargetRotaryModule(is_neox_style)


@pytest.mark.parametrize("is_neox_style", [False, True])
def test_dflash_target_rope_layout_is_discovered(is_neox_style):
    target = _TargetModel(is_neox_style)

    assert dflash_target_rope_is_neox_style(target) is is_neox_style


def test_dflash_loader_propagates_target_rope_layout(monkeypatch):
    """DFlash configures the target rotary layout before draft construction."""
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as loader_module

    draft_hf_config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        dflash_config={"causal": True},
    )
    speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        attention_backend=None,
        kv_cache_dtype=None,
        draft_load_config=None,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        attention_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
        quant_config=None,
    )

    def fake_replace(obj, **changes):
        values = vars(obj).copy()
        values.update(changes)
        return SimpleNamespace(**values)

    class DraftConstructionObserved(Exception):
        pass

    def fake_get_model(**_kwargs):
        assert draft_hf_config.is_neox_style is False
        raise DraftConstructionObserved

    monkeypatch.setattr(loader_module, "replace", fake_replace)
    monkeypatch.setattr(loader_module, "get_model", fake_get_model)
    with pytest.raises(DraftConstructionObserved):
        loader_module.load_dflash_model(_TargetModel(is_neox_style=False), vllm_config)


def test_dspark_loader_preserves_checkpoint_rope_layout(monkeypatch):
    """DSpark inference uses the rotary layout encoded by its training model."""
    from vllm.model_executor.models import utils as model_utils
    from vllm.v1.worker.gpu.spec_decode.dspark import utils as loader_module

    draft_hf_config = SimpleNamespace(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        dflash_config={"causal": True},
        is_neox_style=True,
    )
    speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=draft_hf_config),
        attention_backend=None,
        kv_cache_dtype=None,
        draft_load_config=None,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        attention_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
        quant_config=None,
    )

    class DraftConstructionObserved(Exception):
        pass

    def fake_get_model(**_kwargs):
        assert draft_hf_config.is_neox_style is True
        raise DraftConstructionObserved

    monkeypatch.setattr(loader_module, "get_model", fake_get_model)
    monkeypatch.setattr(
        loader_module,
        "_create_draft_vllm_config",
        lambda _config: vllm_config,
    )
    monkeypatch.setattr(model_utils, "get_draft_quant_config", lambda _config: None)

    with pytest.raises(DraftConstructionObserved):
        loader_module.load_dspark_model(_TargetModel(is_neox_style=False), vllm_config)
