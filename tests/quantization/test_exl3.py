# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module
import vllm.model_executor.parameter as parameter_module
from vllm.config import CompilationMode
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import (
    Exl3Config,
    Exl3MoEParameter,
)
from vllm.model_executor.models import glm4_moe


def _rank_sliced_metadata(**overrides):
    metadata = {
        "format": "exl3-trellis",
        "bits": 3.0,
        "codebook": "mcg",
        "experts_per_layer": 256,
        "moe_layers": [3, 77],
        "tensor_schema": (
            "model.layers.{L}.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}"
        ),
        "tp": 4,
    }
    metadata.update(overrides)
    return metadata


def test_rank_sliced_checkpoint_selects_exl3_override():
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata())

    assert get_quantization_config("exl3") is Exl3Config
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, None, hf_config
        )
        == "exl3"
    )
    assert (
        Exl3Config.override_quantization_method(
            {"quant_method": "modelopt"}, "fp8", hf_config
        )
        is None
    )


def test_glm_model_retains_quant_config_for_weight_loading(monkeypatch):
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
    monkeypatch.setattr(glm4_moe, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        glm4_moe,
        "make_layers",
        lambda *args, **kwargs: (0, 0, torch.nn.ModuleList()),
    )
    monkeypatch.setattr(
        glm4_moe,
        "make_empty_intermediate_tensors_factory",
        lambda *args, **kwargs: object(),
    )
    quant_config = object()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                vocab_size=1,
                hidden_size=8,
                num_hidden_layers=0,
            )
        ),
        cache_config=object(),
        quant_config=quant_config,
        parallel_config=SimpleNamespace(enable_eplb=False),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )

    model = glm4_moe.Glm4MoeModel(vllm_config=vllm_config)

    assert model.quant_config is quant_config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"codebook": "mul1"}, "MCG codebook"),
        ({"moe_layers": [77, 3]}, "moe_layers"),
        ({"tensor_schema": "unsupported"}, "tensor schema"),
    ],
)
def test_rank_sliced_metadata_fails_closed(overrides, message):
    config = Exl3Config()
    hf_config = SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata(**overrides))

    with pytest.raises(ValueError, match=message):
        config.maybe_update_config("unused", hf_config)


def test_rank_sliced_metadata_admits_only_declared_moe_layers():
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )

    assert config._moe_prefix_is_exl3("model.layers.3.mlp.experts")
    assert config._moe_prefix_is_exl3("model.layers.77.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.2.mlp.experts")
    assert not config._moe_prefix_is_exl3("model.layers.78.mlp.experts")
    assert (
        config.codebook_for_prefix("model.layers.10.mlp.experts.0.gate_proj") == "mcg"
    )


def test_rank_sliced_weight_name_keeps_only_local_tp_rank(monkeypatch):
    config = Exl3Config()
    config.maybe_update_config(
        "unused",
        SimpleNamespace(hybrid_tr3_tail=_rank_sliced_metadata()),
    )
    monkeypatch.setattr(exl3_module, "get_tensor_model_parallel_rank", lambda: 2)
    prefix = "model.layers.3.mlp.experts.17.gate_proj"

    assert (
        config.normalize_rank_sliced_weight_name(f"{prefix}.rank2.trellis")
        == f"{prefix}.trellis"
    )
    assert config.normalize_rank_sliced_weight_name(f"{prefix}.rank1.trellis") is None
    assert (
        config.normalize_rank_sliced_weight_name("model.embed_tokens.weight")
        == "model.embed_tokens.weight"
    )


def test_rank_sliced_parameter_preallocates_projection_major_slab(monkeypatch):
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    param = Exl3MoEParameter(
        weight_loader=lambda *args, **kwargs: None,
        num_experts=3,
        shard_ids=("w1", "w3"),
        preallocate=True,
    )
    w1 = torch.arange(8, dtype=torch.int16).reshape(2, 2, 2)
    w3 = w1 + 20

    param.load_exl3_weight(w1, expert_id=1, shard_id="w1")
    param.load_exl3_weight(w3, expert_id=2, shard_id="w3")

    assert param.exl3_backing is not None
    assert tuple(param.exl3_backing.shape) == (2, 3, 2, 2, 2)
    assert (
        param.exl3_tensors[(1, "w1")].data_ptr() == param.exl3_backing[0, 1].data_ptr()
    )
    assert (
        param.exl3_tensors[(2, "w3")].data_ptr() == param.exl3_backing[1, 2].data_ptr()
    )
    torch.testing.assert_close(param.exl3_tensors[(1, "w1")], w1)
    torch.testing.assert_close(param.exl3_tensors[(2, "w3")], w3)


def test_rank_sliced_runtime_scope_is_per_owning_model():
    """Target and rank-sliced MTP draft layers must not share a cached runtime.

    The rank-sliced runtime cache stores mutable Trellis/prefill scratch and
    parity staging buffers. A target MoE layer and an MTP draft layer of the same
    model have identical shapes, topk and planner settings, so a shape-only key
    would hand the draft the target's scratch and break the target/draft
    isolation their independently captured CUDA graphs depend on.
    """
    target_config = SimpleNamespace()
    draft_config = SimpleNamespace()

    target_scope = exl3_module._runtime_scope_id(target_config)
    draft_scope = exl3_module._runtime_scope_id(draft_config)

    # Distinct owning configs must never collide...
    assert target_scope != draft_scope
    # ...and the scope must be stable, so every layer of one model keeps sharing
    # a single runtime (the prefill arena is ~1 GiB; per-layer runtimes would not
    # fit on a 75+ layer model).
    assert exl3_module._runtime_scope_id(target_config) == target_scope
    assert exl3_module._runtime_scope_id(draft_config) == draft_scope


def test_rank_sliced_runtime_key_differs_across_models_with_same_shape():
    """Two same-shape layers owned by different models get different cache keys."""

    def _key(quant_config):
        # Mirrors the scope-prefixed key built in Exl3MoEMethod._rank_sliced_runtime
        # for two layers whose shape/planner components are byte-for-byte equal.
        return (
            exl3_module._runtime_scope_id(quant_config),
            0,  # device index
            torch.bfloat16,
            5120,  # hidden size
            768,  # intermediate size per partition
            64,  # local experts
            8,  # topk
            3072,  # max batched tokens
        )

    target_config = SimpleNamespace()
    draft_config = SimpleNamespace()

    target_key = _key(target_config)
    draft_key = _key(draft_config)

    assert target_key != draft_key
    # Everything except the leading scope is identical, proving the scope is the
    # only thing preventing the collision.
    assert target_key[1:] == draft_key[1:]
    # Same owner -> same key, so target layers still share one runtime.
    assert _key(target_config) == target_key


def test_draft_layer_window_defaults_to_min_capturable_m(monkeypatch) -> None:
    """A rank-sliced draft layer must be capturable without an env workaround.

    Regression test for the boot failure reported in vLLM #183: with the Trellis
    window left at its historical default of 4, CUDA-graph capture of an EXL3
    rank-sliced MTP draft reaches the eager parity path at m=1,2,3 and the engine
    cannot start:

        RuntimeError: EXL3 eager parity path entered during CUDA graph capture
        (m=3); capture sizes must lie inside the Trellis window [4, 32]

    It was invariant to num_speculative_tokens and to cudagraph_capture_sizes,
    because m here is the draft's row count per step, not a target batch size.
    The backend now declares MIN_CAPTURABLE_TRELLIS_M and defaults draft layers to
    it, so no operator has to set VLLM_EXL3_TRELLIS_MIN_M by hand.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import exl3 as exl3_mod

    monkeypatch.delenv("VLLM_EXL3_TRELLIS_MIN_M", raising=False)

    # The GLM-5.2 MTP head is named exactly like a target layer, so the role
    # comes from the exl3_is_draft stamp applied by load_eagle_model -- name
    # inspection alone cannot classify it.
    draft = SimpleNamespace(
        layer_name="model.layers.78.mlp.experts", exl3_is_draft=True
    )
    target = SimpleNamespace(layer_name="model.layers.30.mlp.experts")

    assert exl3_mod._is_draft_layer(draft)
    assert not exl3_mod._is_draft_layer(target)
    # Unstamped draft with a distinctive prefix still classifies via fallback.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.mlp.experts")
    )
    # A stamp always wins over the name, in both directions.
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.experts", exl3_is_draft=False)
    )

    def resolved(layer):
        default = (
            exl3_mod.MIN_CAPTURABLE_TRELLIS_M
            if exl3_mod._is_draft_layer(layer)
            else exl3_mod._DEFAULT_TRELLIS_MIN_M
        )
        return exl3_mod._positive_env_int("VLLM_EXL3_TRELLIS_MIN_M", default)

    # The draft must admit m=1 so capture at m=1,2,3 stays on the Trellis path.
    assert resolved(draft) <= exl3_mod.MIN_CAPTURABLE_TRELLIS_M == 1
    # The target keeps its historical default.
    assert resolved(target) == exl3_mod._DEFAULT_TRELLIS_MIN_M == 4

    # An explicit value remains authoritative in both directions (kill switch).
    monkeypatch.setenv("VLLM_EXL3_TRELLIS_MIN_M", "4")
    assert resolved(draft) == 4


def test_draft_role_stamp_wins_over_name() -> None:
    """The exl3_is_draft stamp set in create_weights is authoritative.

    Forward/plan/capture time has no current vllm config, so the role cannot be
    inferred there; create_weights stamps it from runner_type while the
    construction context is live. Stamped values must win over any name
    heuristic in both directions.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization import exl3 as exl3_mod

    # GLM-5.2 MTP head: named like a target, stamped draft.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.78.mlp.experts", exl3_is_draft=True)
    )
    # Target stamped False keeps its role even with a suspicious name.
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.experts", exl3_is_draft=False)
    )
    # Unstamped layers fall back to the name heuristic.
    assert exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.0.mtp.mlp.experts")
    )
    assert not exl3_mod._is_draft_layer(
        SimpleNamespace(layer_name="model.layers.30.mlp.experts")
    )
