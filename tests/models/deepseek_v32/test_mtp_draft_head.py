# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 (744B) MTP drafts with the configured draft-only vocabulary head."""

import pytest
import torch

from vllm.models.deepseek_v32.nvidia import mtp as deepseek_v32_mtp
from vllm.models.deepseek_v32.nvidia.mtp import DeepseekV32MultiTokenPredictor


def _predictor(target_head: torch.nn.Module) -> DeepseekV32MultiTokenPredictor:
    predictor = DeepseekV32MultiTokenPredictor.__new__(DeepseekV32MultiTokenPredictor)
    torch.nn.Module.__init__(predictor)
    layer = torch.nn.Module()
    layer.shared_head = torch.nn.Module()
    layer.shared_head.head = target_head
    predictor.layers = torch.nn.ModuleDict({"78": layer})
    predictor.mtp_start_layer_idx = 78
    predictor.num_mtp_layers = 1
    predictor.logits_processor = lambda head, hidden_states: head(hidden_states)
    predictor.quantized_draft_head = None
    return predictor


def test_glm53_mtp_drafts_with_configured_head(monkeypatch) -> None:
    target_head = torch.nn.Linear(4, 8, bias=False)
    draft_head = torch.nn.Linear(4, 8, bias=False)
    monkeypatch.setattr(
        deepseek_v32_mtp,
        "make_quantized_draft_head",
        lambda source: draft_head if source is target_head else None,
    )
    predictor = _predictor(target_head)
    hidden_states = torch.randn(3, 4)

    predictor.prepare_draft_lm_head(target_head)

    torch.testing.assert_close(
        predictor.compute_logits(hidden_states), draft_head(hidden_states)
    )


@pytest.mark.parametrize("runtime_quantization", ["nvfp4", "mxfp8"])
def test_glm53_mtp_drafts_with_runtime_quantized_target_head(
    monkeypatch, runtime_quantization: str
) -> None:
    target_head = torch.nn.Linear(4, 8, bias=False)
    target_head.runtime_lm_head_quantization = runtime_quantization

    def reject_requantization(_source):
        raise AssertionError("a packed target head was quantized again")

    monkeypatch.setattr(
        deepseek_v32_mtp, "make_quantized_draft_head", reject_requantization
    )
    predictor = _predictor(target_head)
    hidden_states = torch.randn(3, 4)

    predictor.prepare_draft_lm_head(target_head)

    assert predictor.quantized_draft_head is None
    torch.testing.assert_close(
        predictor.compute_logits(hidden_states), target_head(hidden_states)
    )
