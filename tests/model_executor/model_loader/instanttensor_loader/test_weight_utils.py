# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import tempfile

import huggingface_hub.constants
import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_model_loader():
    with tempfile.TemporaryDirectory() as tmpdir:
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "openai-community/gpt2", allow_patterns=["*.safetensors"], cache_dir=tmpdir
        )
        safetensors = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
        assert len(safetensors) > 0

        instanttensor_tensors = {}
        hf_safetensors_tensors = {}

        for name, tensor in instanttensor_weights_iterator(safetensors, True):
            # Copy the tensor immediately as it is a reference to the internal
            # buffer of instanttensor.
            instanttensor_tensors[name] = tensor.to("cpu")

        for name, tensor in safetensors_weights_iterator(safetensors, True):
            hf_safetensors_tensors[name] = tensor

        assert len(instanttensor_tensors) == len(hf_safetensors_tensors)

        for name, instanttensor_tensor in instanttensor_tensors.items():
            assert instanttensor_tensor.dtype == hf_safetensors_tensors[name].dtype
            assert instanttensor_tensor.shape == hf_safetensors_tensors[name].shape
            assert torch.all(instanttensor_tensor.eq(hf_safetensors_tensors[name]))


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="InstantTensor requires NVIDIA GPUs",
)
def test_instanttensor_honors_tensor_to_shard_index(tmp_path):
    base_shard = tmp_path / "model-00001-of-00002.safetensors"
    overlay_shard = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "model.dense.weight": torch.tensor([2.0]),
            "model.expert.weight": torch.tensor([1.0, 1.0]),
        },
        base_shard,
    )
    save_file(
        {"model.expert.weight": torch.tensor([3.0, 3.0, 3.0])},
        overlay_shard,
    )
    indexed_tensor_files = {
        "model.dense.weight": str(base_shard.resolve()),
        "model.expert.weight": str(overlay_shard.resolve()),
    }

    weights = {
        name: tensor.cpu()
        for name, tensor in instanttensor_weights_iterator(
            [str(base_shard), str(overlay_shard)],
            use_tqdm_on_load=False,
            indexed_tensor_files=indexed_tensor_files,
        )
    }

    assert set(weights) == {"model.dense.weight", "model.expert.weight"}
    assert weights["model.dense.weight"].tolist() == [2.0]
    assert weights["model.expert.weight"].tolist() == [3.0, 3.0, 3.0]


if __name__ == "__main__":
    test_instanttensor_model_loader()
