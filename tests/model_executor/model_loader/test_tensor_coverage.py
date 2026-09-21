# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch

from vllm.model_executor.model_loader.coverage import tensor_coverage


def test_actual_calls_distinguish_loaded_views_and_ignored_tensors(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    path = tmp_path / "coverage.json"
    values = [("target", torch.ones(2, 2)), ("optional", torch.zeros(2))]
    with tensor_coverage(model, values, path) as weights:
        for name, value in weights:
            if name == "target":
                model.weight.weight_loader(model.weight, value.unsqueeze(0)[0])
    result = json.loads(path.read_text())
    assert result["status"] == "completed"
    assert result["tensors"]["target"]["destinations"][0]["parameter"] == "weight"
    assert not result["tensors"]["optional"]["destinations"]
    assert not hasattr(model.weight, "weight_loader")
    torch.testing.assert_close(model.weight, values[0][1])


def test_unattributed_input_fails_and_restores_loader(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    path = tmp_path / "coverage.json"
    with (
        pytest.raises(ValueError, match="Cannot attribute"),
        tensor_coverage(model, [("target", torch.ones(2, 2))], path) as weights,
    ):
        for _, value in weights:
            model.weight.weight_loader(model.weight, value.clone())
    assert json.loads(path.read_text())["status"] == "failed"
    assert not hasattr(model.weight, "weight_loader")
