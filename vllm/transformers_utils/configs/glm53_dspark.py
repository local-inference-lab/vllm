# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for GLM-5.3 DSpark with Afterburner feature conditioning."""

from transformers import PretrainedConfig


class Glm53DSparkConfig(PretrainedConfig):
    model_type = "glm53_dspark"

    def __init__(
        self,
        afterburner_depth: int = 4,
        afterburner_intermediate_size: int = 4096,
        afterburner_residual: bool = True,
        rope_parameters: dict | None = None,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.afterburner_depth = afterburner_depth
        self.afterburner_intermediate_size = afterburner_intermediate_size
        self.afterburner_residual = afterburner_residual
        self.rope_theta = rope_theta
        self.router_dtype = "float32"
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": rope_theta,
        }
        for name, value in (
            ("afterburner_depth", afterburner_depth),
            ("afterburner_intermediate_size", afterburner_intermediate_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(afterburner_residual, bool):
            raise ValueError("afterburner_residual must be boolean")
        super().__init__(**kwargs)
