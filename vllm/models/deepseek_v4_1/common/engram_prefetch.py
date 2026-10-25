# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side Engram row-id prediction for advisory disk row prefetch."""

from __future__ import annotations

import functools
from collections.abc import Callable
from itertools import islice
from typing import Any

import torch

from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID
from vllm.utils.b12x_row_prefetch import PrefetchTarget


def compress_engram_tokens(raw: torch.Tensor, token_map: torch.Tensor) -> torch.Tensor:
    """Raw ids -> compressed ids with -1 for dead tokens, as _prepare_metadata does."""
    v = token_map.numel()
    dead = (raw < 0) | (raw >= v) | (raw == IMAGE_SENTINEL_BASE_ID)
    return torch.where(dead, -1, token_map[raw.clamp(0, v - 1)])


def engram_prefetch_targets(model) -> tuple[PrefetchTarget, ...]:
    """One target per Engram layer of a disk-backed ``DeepseekV4Model``.

    The row ids are exactly the ``hash_ids`` that ``stage_disk`` binds and
    ``run_lookups`` passes to ``read_rows``, predicted from raw token ids.
    """
    from b12x.sequence.engram.host import query_row_ids

    layout = model.engram_layout
    token_map = torch.tensor(layout.token_map, dtype=torch.int64)
    pad = layout.token_map[2]
    targets = []
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        engram = getattr(layer, "engram", None)
        if engram is None:
            continue
        embed = engram.embed_tokens

        def resolve_table(embed=embed):
            # Built lazily by _ensure_disk_table, so absent until the first
            # real step.
            return embed.disk_table

        def row_ids(
            compressed: torch.Tensor, layer_id: int = embed.caps.layer_id
        ) -> torch.Tensor:
            return query_row_ids(
                compress_engram_tokens(compressed, token_map),
                geometry=layout.geometry,
                layer_id=layer_id,
                pad=pad,
            )

        targets.append(PrefetchTarget(resolve_table=resolve_table, row_ids=row_ids))
    return tuple(targets)


def _resolve_engram_disk_table(embed: Any):
    """The disk table of one Engram embedding, or None while lazily built."""
    # Built lazily by _ensure_disk_table, so absent until the first real step.
    return embed.disk_table


def engram_disk_row_tables(
    model,
) -> tuple[tuple[str, Callable[[], Any | None]], ...]:
    """One ``(kind, disk table resolver)`` pair per Engram layer.

    The row-cache metrics sampler needs only the tables, so this stays free of
    the token-map and geometry copies ``engram_prefetch_targets`` makes: it is
    built whether or not advisory prefetch is enabled.
    """
    tables = []
    for layer in islice(model.layers, model.start_layer, model.end_layer):
        engram = getattr(layer, "engram", None)
        if engram is None:
            continue
        tables.append(
            (
                "engram",
                functools.partial(_resolve_engram_disk_table, engram.embed_tokens),
            )
        )
    return tuple(tables)
