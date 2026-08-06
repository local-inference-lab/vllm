# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    @property
    def model_returns_tuple(self) -> bool:
        # DeepSeek MTP recycles the post-final-norm hidden state between
        # draft steps, so forward() returns (logit_hidden, recycle_hidden).
        return "DeepSeekMTPModel" in (
            self.draft_model_config.hf_config.architectures or []
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        return load_eagle_model(target_model, self.vllm_config)
