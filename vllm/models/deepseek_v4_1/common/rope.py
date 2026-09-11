# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 adjacent-pair rotary table construction (no runtime foreign compute)."""

from vllm.model_executor.layers.rotary_embedding import get_rope


def build_deepseek_v4_rope(
    config, *, head_dim, rope_head_dim, max_position_embeddings, compress_ratio
):
    parameters = config.rope_parameters
    if isinstance(parameters.get("main"), dict):
        parameters = parameters["compress" if compress_ratio else "main"]
    parameters = dict(parameters)
    parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio else config.rope_theta
    )
    if compress_ratio and parameters["rope_type"] != "default":
        parameters["rope_type"] = (
            "deepseek_yarn"
            if parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    else:
        parameters["rope_type"] = "deepseek_yarn"
        parameters["factor"] = 1.0
        parameters["original_max_position_embeddings"] = max_position_embeddings
    parameters.update(
        mscale=0, mscale_all_dim=0, is_deepseek_v4=True, rope_dim=rope_head_dim
    )
    return get_rope(
        head_dim,
        max_position=max_position_embeddings,
        rope_parameters=parameters,
        is_neox_style=False,
    )
