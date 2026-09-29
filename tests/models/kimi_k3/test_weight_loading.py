# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.kimi_k3.nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
)

pytestmark = pytest.mark.cpu_test


class _FakeKimiLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tensor_a = nn.Parameter(torch.zeros(1))
        self.tensor_c = nn.Parameter(torch.zeros(1))
        self.finalized_values: list[tuple[float, float]] = []
        self.workspace_reservations = 0

    def reserve_attn_res_workspace(self) -> None:
        assert self.finalized_values == [(1.0, 3.0)]
        self.workspace_reservations += 1

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            params[name].data.copy_(value)
            loaded.add(name)
        return loaded

    def finalize_mega_moe_weights(self) -> None:
        self.finalized_values.append((self.tensor_a.item(), self.tensor_c.item()))
        # MegaMoE finalization replaces its original weight parameters.
        self.tensor_a = None
        self.tensor_c = None


def test_interleaved_composite_weights_finalize_kimi_once_after_loading() -> None:
    language_model = object.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(language_model)
    language_model.config = SimpleNamespace(tie_word_embeddings=False)
    language_model.model = _FakeKimiLinearModel()

    model = object.__new__(KimiK3ForConditionalGeneration)
    nn.Module.__init__(model)
    model.language_model = language_model
    model.vision_tower = nn.Module()
    model.vision_tower.tensor_b = nn.Parameter(torch.zeros(1))

    loaded = model.load_weights(
        iter(
            [
                ("language_model.model.tensor_a", torch.tensor([1.0])),
                ("vision_tower.tensor_b", torch.tensor([2.0])),
                ("language_model.model.tensor_c", torch.tensor([3.0])),
            ]
        )
    )

    assert loaded == {
        "language_model.model.tensor_a",
        "vision_tower.tensor_b",
        "language_model.model.tensor_c",
    }
    assert language_model.model.finalized_values == []

    model.process_weights_after_loading()

    assert language_model.model.finalized_values == [(1.0, 3.0)]
    assert language_model.model.workspace_reservations == 1
    assert model.vision_tower.tensor_b.item() == 2.0


@pytest.mark.parametrize("source_ndim", [1, 4])
@pytest.mark.parametrize("rank", [0, 9])
def test_kda_head_parameters_zero_only_checkpoint_absent_tp_tail(
    monkeypatch, source_ndim, rank
):
    from vllm.models.kimi_k3.nvidia import kda

    monkeypatch.setattr(kda, "get_tensor_model_parallel_rank", lambda: rank)
    source = torch.arange(96, dtype=torch.float32) + 1
    param = nn.Parameter(torch.full((10,), float("nan")), requires_grad=False)
    param.allow_tp_padding = True
    loaded = source if source_ndim == 1 else source.view(1, 1, 96, 1)
    kda.a_log_weight_loader(0)(param, loaded)
    expected = torch.nn.functional.pad(source, (0, 4))[rank * 10 : (rank + 1) * 10]
    torch.testing.assert_close(param, expected, rtol=0, atol=0)


@pytest.mark.parametrize("with_decode_copy", [False, True])
def test_kda_packed_conv_loading_preserves_qkv_boundaries_with_tp_tail(
    with_decode_copy,
):
    from vllm.models.kimi_k3.nvidia.kda import _make_decode_conv1d_weight_loader

    source = torch.arange(96 * 3, dtype=torch.float32).view(96, 1, 3)
    param = nn.Parameter(torch.full((30, 1, 3), float("nan")), requires_grad=False)
    param.allow_tp_padding = True
    decode = torch.full((3, 3, 10), float("nan")) if with_decode_copy else None
    loader = _make_decode_conv1d_weight_loader([100] * 3, 10, 9, decode)
    for shard in (2, 0, 1):
        weight = source + 1000 * shard
        loader(param, weight, shard)
        expected = torch.cat((weight[90:], torch.zeros(4, 1, 3)))
        torch.testing.assert_close(
            param[shard * 10 : (shard + 1) * 10], expected, rtol=0, atol=0
        )
        if decode is not None:
            torch.testing.assert_close(
                decode[shard], expected.squeeze(1).T, rtol=0, atol=0
            )
