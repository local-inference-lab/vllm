# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model
from vllm.models.deepseek_v32.nvidia import mtp as deepseek_v32_mtp


def test_rank_sliced_exl3_names_are_normalized_before_loading(monkeypatch):
    model = object.__new__(deepseek_v32_model.DeepseekV32Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=0)
    model.num_redundant_experts = 0
    model.register_parameter(
        "local_weight",
        nn.Parameter(torch.zeros(1), requires_grad=False),
    )

    def normalize(name: str) -> str | None:
        if ".rank1." in name:
            return None
        return name.replace(".rank0.mcg", "")

    model.quant_config = SimpleNamespace(
        normalize_rank_sliced_weight_name=normalize,
    )
    monkeypatch.setattr(
        deepseek_v32_model,
        "fused_moe_make_expert_params_mapping",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        deepseek_v32_model,
        "get_pp_missing_layer_names",
        lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(
        deepseek_v32_model,
        "get_spec_layer_idx_from_weight_name",
        lambda *args, **kwargs: None,
    )

    loaded = model.load_weights(
        iter(
            [
                ("local_weight.rank1.mcg", torch.tensor([1.0])),
                ("local_weight.rank0.mcg", torch.tensor([2.0])),
            ]
        )
    )

    assert loaded == {"local_weight"}
    torch.testing.assert_close(model.local_weight, torch.tensor([2.0]))


def test_rank_sliced_exl3_mtp_skips_peer_rank_payloads(monkeypatch):
    model = object.__new__(deepseek_v32_mtp.DeepseekV32MTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(n_routed_experts=0)
    model.model = SimpleNamespace(mtp_start_layer_idx=78, num_mtp_layers=1)
    model.quant_config = SimpleNamespace(
        normalize_rank_sliced_weight_name=lambda name: (
            None if ".rank1." in name else name
        ),
    )
    monkeypatch.setattr(
        deepseek_v32_mtp,
        "fused_moe_make_expert_params_mapping",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        deepseek_v32_mtp,
        "get_pp_missing_layer_names",
        lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(
        deepseek_v32_mtp,
        "is_mtp_completeness_check_enabled",
        lambda: False,
    )

    loaded = model.load_weights(
        iter(
            [
                (
                    "model.layers.78.mlp.experts.0.down_proj.rank1.mcg",
                    torch.tensor([1.0]),
                ),
            ]
        )
    )

    assert loaded == set()
