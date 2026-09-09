# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from contextlib import contextmanager

import huggingface_hub.constants
import pytest
import torch
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import default_loader, weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.weight_transfer import get_file_tensor_source


@pytest.mark.parametrize(
    "load_format", ["safetensors", "fastsafetensors", "instanttensor"]
)
def test_mmap_ranges_bypass_payload_loading_in_mixed_files(
    tmp_path, monkeypatch, load_format
):
    """Only ordinary weights may reach a payload reader, even in mixed files."""
    table_name = "keep.table.weight"
    table = torch.arange(48, dtype=torch.uint8).reshape(6, 8)
    scale = torch.tensor([0.5], dtype=torch.bfloat16)
    mixed = tmp_path / "part-1.safetensors"
    ordinary = tmp_path / "part-0.safetensors"
    save_file(
        {table_name: table, "keep.scale": scale, "ignore.weight": table + 1}, str(mixed)
    )
    save_file({"keep.ordinary": scale + 1}, str(ordinary))
    real_open = weight_utils.safe_open
    real_iterator = weight_utils.safetensors_weights_iterator

    @contextmanager
    def guarded_open(*args, **kwargs):
        with real_open(*args, **kwargs) as archive:

            class GuardedArchive:
                def __getattr__(self, name):
                    return getattr(archive, name)

                def get_tensor(self, name):
                    assert name != table_name, "mmap payload was materialized"
                    return archive.get_tensor(name)

            yield GuardedArchive()

    def ordinary_iterator(paths, *args, **kwargs):
        assert str(mixed) not in paths, "mixed file reached bulk payload loading"
        yield from real_iterator(paths, False, weight_name_prefixes=("keep.",))

    monkeypatch.setattr(weight_utils, "safe_open", guarded_open)
    monkeypatch.setattr(
        default_loader, f"{load_format}_weights_iterator", ordinary_iterator
    )
    loader = default_loader.DefaultModelLoader(
        LoadConfig(load_format=load_format, use_tqdm_on_load=False)
    )
    source = loader.Source(
        str(tmp_path),
        None,
        prefix="wrapped.",
        weight_name_prefixes=("keep.",),
        mmap_weight_filter=lambda name: name == table_name,
    )
    loaded = dict(loader._get_weights_iterator(source))
    assert set(loaded) == {
        "wrapped.keep.ordinary",
        "wrapped.keep.scale",
        "wrapped.keep.table.weight",
    }
    torch.testing.assert_close(loaded["wrapped.keep.ordinary"], scale + 1)
    torch.testing.assert_close(loaded["wrapped.keep.scale"], scale)
    mapped = loaded["wrapped.keep.table.weight"]
    assert mapped.is_meta
    descriptor = get_file_tensor_source(mapped)
    assert descriptor is not None
    with open(descriptor.path, "rb", buffering=0) as file:
        file.seek(descriptor.offset)
        assert file.read(table.numel()) == bytes(table.flatten().tolist())


def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        # assert LocalEntryNotFoundError error is thrown
        # if offline is set and model is not cached
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )

        # download the model
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "facebook/opt-125m",
            allow_patterns=["*.safetensors", "*.bin"],
            cache_dir=tmpdir,
        )

        # now it should work offline
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert (
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )
            is not None
        )


class TestMaybeRemapKvScaleName:
    """Tests for maybe_remap_kv_scale_name covering all checkpoint formats."""

    PARAMS_DICT = {
        "model.layers.0.self_attn.attn.k_scale": None,
        "model.layers.0.self_attn.attn.v_scale": None,
        "model.layers.0.self_attn.attn.q_scale": None,
        "model.layers.0.self_attn.qkv_proj.weight": None,
    }

    def test_qkv_proj_k_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.k_scale -> attn.k_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qkv_proj_v_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.v_scale -> attn.v_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_modelopt_k_proj_k_scale(self):
        """ModelOpt format: k_proj.k_scale -> attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_modelopt_v_proj_v_scale(self):
        """ModelOpt format: v_proj.v_scale -> attn.v_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_deprecated_kv_scale(self):
        """Old format: kv_scale -> attn.k_scale (deprecated)"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.kv_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_default_bare_k_scale(self):
        """Default format: .k_scale -> .attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_non_scale_name_unchanged(self):
        """Non-scale names should be returned unchanged."""
        name = "model.layers.0.self_attn.qkv_proj.weight"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_modelopt_k_proj_k_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        k_proj.k_scale -> attn.k_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_nvfp4_modelopt_v_proj_v_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        v_proj.v_scale -> attn.v_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_qwen3_vl_moe_qkv_proj_k_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qwen3_vl_moe_qkv_proj_v_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_nvfp4_weight_scale_not_remapped(self):
        """NVFP4 weight_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.weight_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_input_scale_not_remapped(self):
        """NVFP4 input_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.input_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_missing_target_returns_none(self):
        """If remapped name not in params_dict, return None."""
        empty_params: dict[str, None] = {}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", empty_params
        )
        assert result is None


class TestKvCacheScaleMapper:
    """The `WeightsMapper` returned by `get_cache_scale_mapper` replaces the
    per-model `maybe_remap_kv_scale_name` calls. It must remap the same set of
    checkpoint formats (the non-`params_dict`-dependent ones) and be idempotent
    so it composes safely with a model's own qkv/gate_up `hf_to_vllm_mapper`."""

    def _mapper(self):
        # `get_cache_scale_mapper` does not use `self`; call it on the base
        # class to get the default (non-config-specific) mapper.
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig,
        )

        return QuantizationConfig.get_cache_scale_mapper()

    def _map(self, name: str) -> str | None:
        return self._mapper()._map_name(name)

    @pytest.mark.parametrize(
        "name,expected",
        [
            # Qwen3-MoE / llm-compressor fused qkv_proj
            (
                "model.layers.0.self_attn.qkv_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.qkv_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # ModelOpt / NVFP4 k_proj/v_proj
            (
                "model.layers.0.self_attn.k_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.v_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # deprecated fused kv_scale and bare scales
            (
                "model.layers.0.self_attn.kv_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # NemotronH mixer
            (
                "model.layers.0.mixer.k_proj.k_scale",
                "model.layers.0.mixer.attn.k_scale",
            ),
            # already in vLLM form -> unchanged (idempotent)
            (
                "model.layers.0.self_attn.attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # non-kv scales must not be touched
            (
                "model.layers.0.self_attn.k_proj.weight_scale",
                "model.layers.0.self_attn.k_proj.weight_scale",
            ),
            (
                "model.layers.0.self_attn.k_proj.input_scale",
                "model.layers.0.self_attn.k_proj.input_scale",
            ),
            # regular weights untouched
            (
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ),
        ],
    )
    def test_remap(self, name, expected):
        assert self._map(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.k_proj.k_scale",
            "model.layers.0.self_attn.qkv_proj.v_scale",
            "model.layers.0.mixer.k_proj.k_scale",
        ],
    )
    def test_idempotent(self, name):
        once = self._map(name)
        assert once is not None
        assert self._map(once) == once

    def test_composes_with_qkv_mapper(self):
        """Applied together with a model's qkv/gate_up mapper, the regex scale
        rules run before the substr rename, so scales are normalized to `.attn.`
        and regular projections are still fused correctly."""
        from vllm.model_executor.models.utils import WeightsMapper

        model_mapper = WeightsMapper(
            orig_to_new_substr={
                ".q_proj": ".qkv_proj.q",
                ".k_proj": ".qkv_proj.k",
                ".v_proj": ".qkv_proj.v",
            }
        )
        # AutoWeightsLoader does `mapper |= cache_scale_mapper`
        combined = model_mapper | self._mapper()

        assert (
            combined._map_name("model.layers.0.self_attn.q_proj.weight")
            == "model.layers.0.self_attn.qkv_proj.q.weight"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_proj.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )


if __name__ == "__main__":
    test_download_weights_from_hf()
