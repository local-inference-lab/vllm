# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GLM DSA model declares its fused all-reduce + RMSNorm sites.

The model is not torch.compiled, so fused_allreduce_rms_norm calls the B12X
fused operation directly, and the communicator runs it only for declared
(shape, norm weight, epsilon) sites.
"""

from types import SimpleNamespace

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.models.deepseek_v32.nvidia import model as deepseek_v32_model


def _layer():
    return SimpleNamespace(
        input_layernorm=RMSNorm(8), post_attention_layernorm=RMSNorm(8)
    )


def test_every_all_reduce_followed_by_a_norm_is_declared(
    monkeypatch, default_vllm_config
):
    declared = {}

    def declare(owner, norms, hidden_size, name_prefix):
        declared.update(
            owner=owner, norms=list(norms), hidden_size=hidden_size, prefix=name_prefix
        )
        return len(declared["norms"])

    monkeypatch.setattr(
        deepseek_v32_model, "declare_b12x_fused_allreduce_rms_norm_sites", declare
    )
    layers = [_layer() for _ in range(5)]
    model = SimpleNamespace(
        layers=layers,
        start_layer=2,
        end_layer=5,
        norm=RMSNorm(8),
        config=SimpleNamespace(hidden_size=8),
    )

    deepseek_v32_model.DeepseekV32Model._declare_fused_allreduce_rms_norm_sites(model)

    names = [name for name, _ in declared["norms"]]
    # The first local layer's input norm follows no all-reduce on this rank.
    assert names == [
        "layers.2.post_attention_layernorm",
        "layers.3.input_layernorm",
        "layers.3.post_attention_layernorm",
        "layers.4.input_layernorm",
        "layers.4.post_attention_layernorm",
        "norm",
    ]
    assert declared["norms"][1][1] is layers[3].input_layernorm
    assert declared["norms"][-1][1] is model.norm
    assert declared["owner"] is model and declared["hidden_size"] == 8
