# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.model_loader.weight_utils as weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


def test_instanttensor_requests_owned_tensors(monkeypatch):
    tensor = torch.ones(4)
    observed: dict[str, object] = {}

    class FakeReader:
        total_tensor_size = tensor.numel() * tensor.element_size()

        def tensors(self):
            yield "weight", tensor

    @contextmanager
    def fake_safe_open(files, *, framework, device, process_group, copy):
        observed.update(
            files=files,
            framework=framework,
            device=device,
            process_group=process_group,
            copy=copy,
        )
        yield FakeReader()

    def no_world_group():
        raise AssertionError

    monkeypatch.setattr(
        weight_utils,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True, current_device=lambda: 0),
    )
    monkeypatch.setattr(weight_utils, "get_world_group", no_world_group)
    monkeypatch.setitem(
        sys.modules, "instanttensor", SimpleNamespace(safe_open=fake_safe_open)
    )

    loaded = list(instanttensor_weights_iterator(["model.safetensors"], False))

    assert loaded == [("weight", tensor)]
    assert observed == {
        "files": ["model.safetensors"],
        "framework": "pt",
        "device": 0,
        "process_group": None,
        "copy": True,
    }


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    model_dir = download_weights_from_hf(
        "openai-community/gpt2", cache_dir=None, allow_patterns=["*.safetensors"]
    )
    safetensors = glob.glob(f"{model_dir}/*.safetensors")
    assert len(safetensors) > 0

    instanttensor_tensors = {}
    hf_safetensors_tensors = {}

    for name, tensor in instanttensor_weights_iterator(safetensors, True):
        instanttensor_tensors[name] = tensor.to("cpu")

    for name, tensor in safetensors_weights_iterator(safetensors, True):
        hf_safetensors_tensors[name] = tensor

    assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

    for name, instanttensor_tensor in instanttensor_tensors.items():
        assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
        assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
        assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


if __name__ == "__main__":
    test_instanttensor_model_loader()
