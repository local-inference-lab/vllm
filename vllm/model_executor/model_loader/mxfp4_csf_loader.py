# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read retained native tensors from a lossless MXFP4 container.

Compressed routed experts are owned by the model's quantization method.
Engram tensors retain their immutable file-range descriptors, avoiding full
CPU/GPU staging allocations for the host-mapped embedding tables.
"""

import time
from pathlib import Path

import regex as re
from safetensors import safe_open

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    file_source_tensor,
    safetensors_file_sources,
)


class Mxfp4CsfModelLoader(DefaultModelLoader):
    def _root(self, model_config):
        from b12x.moe.checkpoints.mxfp4_csf import checkpoint_contract

        text_config = getattr(model_config, "hf_text_config", model_config.hf_config)
        quant = getattr(model_config.hf_config, "quantization_config", None)
        quant = quant or text_config.quantization_config
        if quant.get("quant_method") not in ("mxfp4_csf",):
            raise ValueError(
                "MXFP4-CSF loading requires a compressed-scale model config"
            )
        root = Path(quant["checkpoint_root"])
        if not root.is_absolute():
            raise ValueError("MXFP4-CSF checkpoint_root must be an absolute local path")
        return root, checkpoint_contract(str(root.resolve()))

    def download_model(self, model_config):
        self._root(model_config)

    def get_all_weights(self, model_config, model):
        root, contract = self._root(model_config)
        if getattr(model, "secondary_weights", ()):
            raise NotImplementedError(
                "MXFP4-CSF does not support secondary weight sources"
            )
        file_filter = getattr(model, "checkpoint_file_weight_filter", None)
        prefixes = getattr(model, "checkpoint_weight_name_prefixes", None)
        text_config = getattr(model_config, "hf_text_config", model_config.hf_config)
        layers = text_config.num_hidden_layers
        self.counter_before_loading_weights = time.perf_counter()
        for filename in sorted(set(contract["source_names"].values())):
            path = str(root / "tensors" / filename)
            descriptors = safetensors_file_sources(path)
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in sorted(descriptors):
                    if prefixes is not None and not name.startswith(prefixes):
                        continue
                    match = re.search(
                        r"(?:^|\.)layers\.(\d+)\.(?:ffn|block_sparse_moe)\.experts\.",
                        name,
                    )
                    if match and int(match.group(1)) < layers:
                        continue
                    if callable(file_filter) and file_filter(name):
                        yield name, file_source_tensor(descriptors[name])
                    else:
                        yield name, handle.get_tensor(name)
