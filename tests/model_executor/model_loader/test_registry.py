# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


@pytest.mark.parametrize("option", ["instanttensor_copy", "instanttensor_distributed"])
@pytest.mark.parametrize("load_format", ["instanttensor", "fastsafetensors"])
def test_default_loader_rejects_removed_instanttensor_options(option, load_format):
    with pytest.raises(ValueError, match="Unexpected extra config keys"):
        DefaultModelLoader(
            LoadConfig(
                load_format=load_format,
                model_loader_extra_config={option: False},
            )
        )


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)


@pytest.mark.parametrize(
    "load_format", ["safetensors", "fastsafetensors", "instanttensor"]
)
def test_default_loader_restricts_safetensors_shards_by_weight_prefix(
    tmp_path, load_format
):
    base_shard = tmp_path / "model-00001-of-00002.safetensors"
    mtp_shard = tmp_path / "model-00002-of-00002.safetensors"
    base_shard.write_bytes(b"")
    mtp_shard.write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map": {'
        f'"model.layers.0.weight": "{base_shard.name}",'
        f'"model.layers.45.weight": "{mtp_shard.name}"'
        "}}"
    )
    loader = DefaultModelLoader(LoadConfig(load_format=load_format))

    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=False,
        allow_patterns_overrides=None,
        weight_name_prefixes=("model.layers.45.",),
    )

    assert use_safetensors
    assert files == [str(mtp_shard)]
