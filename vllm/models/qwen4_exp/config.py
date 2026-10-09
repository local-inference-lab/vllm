# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint storage metadata for Qwen4Exp serving."""

from pathlib import Path
from typing import Any

from transformers import PreTrainedConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


def resolve_ple_embedding_dtype(
    config: PreTrainedConfig,
    config_dict: dict[str, Any],
    model: str | Path,
    revision: str | None,
) -> None:
    """Infer omitted PLE storage dtype from checkpoint headers before allocation."""
    text_config = config.get_text_config()
    text_config_dict = config_dict.get("text_config", config_dict)
    if not getattr(text_config, "ple_embedding_dtype", None):
        text_config.ple_embedding_dtype = "bfloat16"
    if not text_config.ple_layer_ids or text_config_dict.get("ple_embedding_dtype"):
        return

    from vllm.transformers_utils.config import get_safetensors_params_metadata

    source = str(model)
    quant_config = (
        config_dict.get("quantization_config")
        or text_config_dict.get("quantization_config")
        or {}
    )
    if quant_config.get("quant_method") in {"nvfp4_csf", "mxfp4_csf"} and (
        root := quant_config.get("checkpoint_root")
    ):
        # An FP4-CSF serving directory holds metadata only; the checkpoint's
        # tensors, PLE shards included, live under its root.
        source = str(Path(root) / "tensors")
    metadata = get_safetensors_params_metadata(source, revision=revision)
    dtypes = {
        info["dtype"]
        for name, info in metadata.items()
        if ".ple.ple_embedding.ngram_embedding.shard_" in name
        and name.endswith(".weight")
    }
    if not dtypes:
        return
    storage_dtypes = {
        "BF16": "bfloat16",
        "F8_E4M3": "float8_e4m3fn",
        "U8": "nvfp4",
    }
    if len(dtypes) != 1 or not dtypes.issubset(storage_dtypes):
        raise ValueError(f"Unsupported PLE checkpoint storage dtypes: {sorted(dtypes)}")
    text_config.ple_embedding_dtype = storage_dtypes[dtypes.pop()]
    logger.info(
        "Resolved PLE embedding storage dtype from checkpoint: %s",
        text_config.ple_embedding_dtype,
    )
