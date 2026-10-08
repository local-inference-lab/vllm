# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 (744B) MTP drafts read only their own checkpoint tensors."""

from types import SimpleNamespace

from vllm.models.deepseek_v32.nvidia.mtp import DeepseekV32MTP


def test_glm53_mtp_selects_only_draft_checkpoint_weights() -> None:
    mtp = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=1)
    )

    assert DeepseekV32MTP._checkpoint_weight_name_prefixes(mtp) == (
        "model.layers.78.",
        "model.embed_tokens.",
    )
