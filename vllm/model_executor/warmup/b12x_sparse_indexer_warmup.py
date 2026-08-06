# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    _run_b12x_paged_topk,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _block_table_widths_by_layer(worker: Worker) -> dict[str, int]:
    runner = worker.model_runner
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    if kv_cache_config is None:
        return {}

    block_tables = getattr(runner, "block_tables", None)
    if block_tables is not None:
        tables = getattr(block_tables, "input_block_tables", ())
    else:
        input_batch = getattr(runner, "input_batch", None)
        multi_group_table = getattr(input_batch, "block_table", None)
        groups = getattr(multi_group_table, "block_tables", ())
        tables = tuple(group.block_table.gpu for group in groups)

    widths: dict[str, int] = {}
    for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
        if group_id >= len(tables):
            break
        width = int(tables[group_id].shape[1])
        for layer_name in group.layer_names:
            widths[layer_name] = width
    return widths


def _fused_decode_warmup_rows(
    *,
    topk: int,
    num_heads: int,
    max_pages: int,
    device: torch.device,
) -> tuple[int, ...]:
    from sparkinfer.attention.nsa_indexer.fused_indexer import (
        fused_indexer_decode_warmup_rows,
    )

    return fused_indexer_decode_warmup_rows(
        topk=topk,
        num_heads=num_heads,
        max_pages=max_pages,
        device=device,
    )


@torch.inference_mode()
def warmup_b12x_sparse_indexer(worker: Worker) -> int:
    """Compile all row-specialized B12X fused decode variants."""
    widths_by_layer = _block_table_widths_by_layer(worker)
    if not widths_by_layer:
        return 0

    warmed_variants = 0
    seen_signatures: set[tuple[object, ...]] = set()
    for module in worker.get_model().modules():
        if not isinstance(module, SparseAttnIndexer):
            continue
        if not module.use_b12x_sparse_indexer or module.num_q_heads is None:
            continue

        kv_cache = module.k_cache.kv_cache
        layer_name = module.k_cache.prefix
        page_table_width = widths_by_layer.get(layer_name)
        if page_table_width is None or kv_cache.numel() == 0:
            logger.warning_once(
                "Skipping B12X sparse-indexer warmup for %s because its finalized "
                "cache or block-table geometry is unavailable.",
                layer_name,
            )
            continue

        num_heads = int(module.num_q_heads)
        topk = int(module.topk_tokens)
        warmup_rows = _fused_decode_warmup_rows(
            topk=topk,
            num_heads=num_heads,
            max_pages=page_table_width,
            device=kv_cache.device,
        )
        if not warmup_rows:
            continue

        signature = (
            kv_cache.device,
            tuple(kv_cache.shape),
            tuple(kv_cache.stride()),
            page_table_width,
            num_heads,
            topk,
            bool(module.output_physical_slots),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        max_rows = max(warmup_rows)
        q_fp8 = torch.empty(
            (max_rows, num_heads, int(module.head_dim)),
            dtype=current_platform.fp8_dtype(),
            device=kv_cache.device,
        )
        weights = torch.empty(
            (max_rows, num_heads),
            dtype=torch.float32,
            device=kv_cache.device,
        )
        seq_lens = torch.zeros((max_rows,), dtype=torch.int32, device=kv_cache.device)
        block_table = torch.zeros(
            (max_rows, page_table_width),
            dtype=torch.int32,
            device=kv_cache.device,
        )
        topk_indices = torch.empty(
            (max_rows, topk), dtype=torch.int32, device=kv_cache.device
        )
        topk_scores = (
            torch.empty((max_rows, topk), dtype=torch.float32, device=kv_cache.device)
            if module.topk_scores_buffer is not None
            else None
        )

        # Launch the largest row shape first so the shared workspace reaches
        # its final capacity before any smaller policy launch is queued.
        ordered_rows = (max_rows, *(rows for rows in warmup_rows if rows != max_rows))
        for rows in ordered_rows:
            _run_b12x_paged_topk(
                q_fp8=q_fp8[:rows],
                weights=weights[:rows],
                kv_cache=kv_cache,
                seq_lens=seq_lens[:rows],
                block_table=block_table[:rows],
                schedule_metadata=None,
                topk_indices=topk_indices[:rows],
                topk_tokens=topk,
                topk_scores=(topk_scores[:rows] if topk_scores is not None else None),
                output_physical_slots=bool(module.output_physical_slots),
            )
            warmed_variants += 1

    if warmed_variants and current_platform.is_cuda():
        torch.accelerator.synchronize()
    return warmed_variants
