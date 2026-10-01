# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read retained native tensors from a lossless MXFP4 container.

Compressed routed experts are owned by the model's quantization method.
Engram tensors retain their immutable file-range descriptors, avoiding full
CPU/GPU staging allocations for the host-mapped embedding tables.
"""

import time
from functools import lru_cache
from pathlib import Path

import regex as re
from safetensors import safe_open

from vllm.model_executor.model_loader.csf_utils import (
    CsfTensorReader,
    read_csf_contract,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    file_source_tensor,
    safetensors_file_sources,
)

SCHEMA = "lil-mxfp4-csf-checkpoint/1"
CODEC = "row-base-offset1-u24-exceptions/1"
FAMILIES = {
    "deepseek_v41": (384, 5120, 2304),
    "kimi_k3": (896, 3584, 3072),
}


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the MXFP4-CSF container before loading model tensors."""
    return read_csf_contract(root, schema=SCHEMA, codec=CODEC, families=FAMILIES)


def read_mxfp4_csf_layer(
    root,
    layer_index,
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
):
    """Resolve model tensor names and pass gate/up/down sources to B12X."""
    from b12x.moe.checkpoints.mxfp4_csf import load_mxfp4_csf_weights

    contract = checkpoint_contract(str(Path(root).resolve()))
    family = contract["family"]
    if (num_experts, hidden_size, intermediate_size) != FAMILIES[family]:
        raise ValueError("MXFP4-CSF expert geometry differs from the checkpoint family")
    supported_tp = (1, 2, 4, 8) if family == "deepseek_v41" else (1, 2, 4, 8, 12, 16)
    if tp_size not in supported_tp or not 0 <= tp_rank < tp_size:
        raise ValueError(
            f"MXFP4-CSF {family} supports TP {supported_tp} with a valid rank"
        )
    with CsfTensorReader(root, contract["source_names"], "mxfp4") as reader:

        def experts():
            for expert in range(num_experts):
                if family == "kimi_k3":
                    prefix = (
                        f"language_model.model.layers.{layer_index}."
                        f"block_sparse_moe.experts.{expert}"
                    )
                    weight, scale = "weight_packed", "weight_scale"
                else:
                    prefix = f"layers.{layer_index}.ffn.experts.{expert}"
                    weight, scale = "weight", "scale"
                yield tuple(
                    reader.matrix(f"{prefix}.{p}.{weight}", f"{prefix}.{p}.{scale}")
                    for p in ("w1", "w3", "w2")
                )

        return load_mxfp4_csf_weights(
            experts(),
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            device=device,
            w13_scale_scratch=w13_scale_scratch,
            w2_scale_scratch=w2_scale_scratch,
        )


class Mxfp4CsfModelLoader(DefaultModelLoader):
    def _root(self, model_config):
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
