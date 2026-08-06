# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def maybe_load_mask_embedding(
    model: nn.Module,
    model_path: str,
    mask_token_id: int,
) -> None:
    """Load a checkpoint-provided mask token embedding into the embed table.

    DFlash FP4 exports ship the trained mask embedding as
    ``mask_embedding.pt`` next to the draft weights because the target
    checkpoint's embedding row for ``mask_token_id`` is zeroed/untrained.
    Without it, every masked draft slot is embedded as ~zero and the
    drafter's per-position acceptance collapses after the first position.
    """
    mask_path = os.path.join(model_path, "mask_embedding.pt")
    if not os.path.exists(mask_path):
        return
    data = torch.load(mask_path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        file_token_id = data.get("mask_token_id")
        embedding = data.get("embedding")
    else:
        file_token_id = None
        embedding = data
    if embedding is None:
        logger.warning("Ignoring %s: no 'embedding' entry found.", mask_path)
        return
    if file_token_id is not None and int(file_token_id) != int(mask_token_id):
        logger.warning(
            "mask_embedding.pt token id %s differs from configured "
            "mask_token_id %s; using the file's token id.",
            file_token_id,
            mask_token_id,
        )
    token_id = int(file_token_id if file_token_id is not None else mask_token_id)
    embedding = embedding.reshape(-1)

    embed_tokens = model.model.embed_tokens
    weight = embed_tokens.weight
    shard_indices = getattr(embed_tokens, "shard_indices", None)
    if shard_indices is not None:
        start = shard_indices.org_vocab_start_index
        end = shard_indices.org_vocab_end_index
    else:
        start, end = 0, weight.shape[0]
    if start <= token_id < end:
        row = weight.data[token_id - start]
        embedding = embedding.to(device=row.device, dtype=row.dtype)
        if embedding.shape != row.shape:
            raise ValueError(
                "mask_embedding.pt shape "
                f"{tuple(embedding.shape)} does not match embedding row "
                f"shape {tuple(row.shape)}."
            )
        row.copy_(embedding)
    logger.info_once(
        "Loaded parallel-drafting mask embedding for token %d from %s.",
        token_id,
        mask_path,
        scope="local",
    )


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            # Honor the speculative-config attention backend for the draft
            # (matches llm_base_proposer): otherwise auto-select picks
            # FlashInfer, which downgrades the spec-decode cudagraph to
            # PIECEWISE and cannot do non-causal prefill under DCP.
            backend=speculative_config.attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = dflash_model.model

    # Skip embedding sharing under PP — each rank owns its own embedding.
    if get_pp_group().world_size == 1:
        target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
            target_inner, "embedding", None
        )
        draft_embed = getattr(draft_inner, "embed_tokens", None)
        if target_embed is not None and _should_share(
            dflash_model, "has_own_embed_tokens", draft_embed, target_embed
        ):
            if draft_embed is not None:
                del draft_inner.embed_tokens
            draft_inner.embed_tokens = target_embed

    # Share lm_head with the target when the draft has no own copy. DFlash
    # may expose draft_id_to_target_id for sampled-token remapping, but its
    # logits still need to be scored by the target-vocab head for acceptance.
    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
