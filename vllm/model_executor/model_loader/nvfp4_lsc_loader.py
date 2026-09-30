# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load retained tensors while the quantization method owns compressed experts."""

import time
from pathlib import Path

import regex as re
from safetensors import safe_open

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


class Nvfp4LscModelLoader(DefaultModelLoader):
    def _root(self, model_config):
        from b12x.moe.checkpoints.nvfp4_lsc import checkpoint_contract

        quant = getattr(model_config.hf_config, "quantization_config", None)
        quant = quant or model_config.hf_text_config.quantization_config
        if quant.get("quant_method") != "nvfp4_lsc":
            raise ValueError("NVFP4-LSC loader requires quant_method=nvfp4_lsc")
        root = Path(quant["checkpoint_root"])
        if not root.is_absolute():
            raise ValueError("NVFP4-LSC checkpoint_root must be an absolute local path")
        return root, checkpoint_contract(str(root.resolve()))

    def download_model(self, model_config):
        self._root(model_config)

    def get_all_weights(self, model_config, model):
        root, contract = self._root(model_config)
        if getattr(model, "secondary_weights", ()):
            raise NotImplementedError(
                "NVFP4-LSC does not support secondary weight sources"
            )
        prefixes = getattr(model, "checkpoint_weight_name_prefixes", None)
        layers = model_config.hf_text_config.num_hidden_layers
        self.counter_before_loading_weights = time.perf_counter()
        for filename in sorted(set(contract["source_names"].values())):
            with safe_open(
                root / "tensors" / filename, framework="pt", device="cpu"
            ) as handle:
                for name in sorted(handle.keys()):
                    if prefixes is not None and not name.startswith(prefixes):
                        continue
                    match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.", name)
                    if match and int(match.group(1)) < layers:
                        continue
                    yield name, handle.get_tensor(name)
