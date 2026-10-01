# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load retained tensors while the quantization method owns compressed experts."""

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

SCHEMA = "lil-nvfp4-csf-checkpoint/1"
CODEC = "byte-window4-fixed-stream-u24-exceptions/1"
FAMILIES = {
    "glm53_nvfp4": (288, 4096, 2048, range(3, 45)),
    "qwen38_flash_next_nvfp4": (512, 2560, 640, range(48)),
}


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the NVFP4-CSF container before loading model tensors."""
    return read_csf_contract(root, schema=SCHEMA, codec=CODEC, families=FAMILIES)


def read_nvfp4_csf_layer(
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
    """Resolve model tensor names and pass up/gate/down sources to B12X."""
    from b12x.moe.checkpoints.nvfp4_csf import load_nvfp4_csf_weights

    contract = checkpoint_contract(str(Path(root).resolve()))
    e, h, n, layers = FAMILIES[contract["family"]]
    if (num_experts, hidden_size, intermediate_size) != (e, h, n):
        raise ValueError("NVFP4-CSF expert geometry differs from the checkpoint family")
    if layer_index not in layers:
        raise ValueError("NVFP4-CSF layer is outside the compressed expert inventory")
    with CsfTensorReader(root, contract["source_names"], "nvfp4") as reader:

        def experts():
            for expert in range(num_experts):
                prefix = (
                    f"model.language_model.layers.{layer_index}.mlp.experts.{expert}"
                )
                yield tuple(
                    reader.matrix(
                        f"{prefix}.{p}.weight",
                        f"{prefix}.{p}.weight_scale",
                        global_scale=f"{prefix}.{p}.weight_scale_2",
                        input_scale=f"{prefix}.{p}.input_scale",
                    )
                    for p in ("up_proj", "gate_proj", "down_proj")
                )

        return load_nvfp4_csf_weights(
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


class Nvfp4CsfModelLoader(DefaultModelLoader):
    def _root(self, model_config):
        quant = getattr(model_config.hf_config, "quantization_config", None)
        quant = quant or model_config.hf_text_config.quantization_config
        if quant.get("quant_method") != "nvfp4_csf":
            raise ValueError("NVFP4-CSF loader requires quant_method=nvfp4_csf")
        root = Path(quant["checkpoint_root"])
        if not root.is_absolute():
            raise ValueError("NVFP4-CSF checkpoint_root must be an absolute local path")
        return root, checkpoint_contract(str(root.resolve()))

    def download_model(self, model_config):
        self._root(model_config)

    def get_all_weights(self, model_config, model):
        root, contract = self._root(model_config)
        if getattr(model, "secondary_weights", ()):
            raise NotImplementedError(
                "NVFP4-CSF does not support secondary weight sources"
            )
        prefixes = getattr(model, "checkpoint_weight_name_prefixes", None)
        file_filter = getattr(model, "checkpoint_file_weight_filter", None)
        layers = model_config.hf_text_config.num_hidden_layers
        self.counter_before_loading_weights = time.perf_counter()
        for filename in sorted(set(contract["source_names"].values())):
            descriptors = safetensors_file_sources(str(root / "tensors" / filename))
            with safe_open(
                root / "tensors" / filename, framework="pt", device="cpu"
            ) as handle:
                for name in sorted(handle.keys()):
                    if prefixes is not None and not name.startswith(prefixes):
                        continue
                    match = re.search(
                        r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.", name
                    )
                    if match and int(match.group(1)) < layers:
                        continue
                    if callable(file_filter) and file_filter(name):
                        yield name, file_source_tensor(descriptors[name])
                    else:
                        yield name, handle.get_tensor(name)
