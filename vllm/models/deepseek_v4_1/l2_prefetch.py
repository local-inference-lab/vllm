# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for DeepSeek V4.1 decode on GB10.

Decode projections stream their weights from LPDDR5X, while each layer also
runs serial latency-bound work (sparse-MLA preparation, TP all-reduces, mHC,
routing) that leaves device memory idle. GB10 has a 24 MB L2, enough for one
layer's pair of attention projections. Each window issues bulk L2 prefetches
for the next projections on a side stream (the GLM-5.3 prefetcher), so those
GEMMs read L2 instead of DRAM. Cache hints only; numerics are unchanged.

At a fixed eight-row verifier, TP4 decode cycles fall from 49.36 to 47.45 ms
(4 prefetch CTAs). Issuing FFN/NEXT after their all-reduces loses the window
(49.13 ms); WO alone is nearly neutral (49.12 ms) because the fill slows the
latency-bound sparse-MLA preparation kernels it overlaps.

Windows per decoder layer:
  WO    after the query up-projection: this layer's WO-A and WO-B
  FFN   before the attention all-reduce: router, mHC FFN mix, shared expert
  NEXT  before the MoE all-reduce: next layer's mHC attention mix, fused
        Q-A/KV and Q-B projections, and indexer projections

Environment:
  VLLM_DS41_L2_PREFETCH=0/1           default on for SM121 (GB10)
  VLLM_DS41_L2_PREFETCH_{WO,FFN,NEXT}_MB   per-window fill budgets
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable

import torch

from vllm.logger import init_logger
from vllm.models.glm5next.nvidia import l2_prefetch as _l2pf

logger = init_logger(__name__)

Segment = _l2pf.Segment


def _enabled() -> bool:
    if not torch.cuda.is_available():
        return False
    raw = os.getenv("VLLM_DS41_L2_PREFETCH")
    if raw is not None:
        return raw != "0"
    return torch.cuda.get_device_capability() == (12, 1)


ENABLED = _enabled()
BUDGET_WO = _l2pf._mb("VLLM_DS41_L2_PREFETCH_WO_MB", "20")
BUDGET_FFN = _l2pf._mb("VLLM_DS41_L2_PREFETCH_FFN_MB", "16")
BUDGET_NEXT = _l2pf._mb("VLLM_DS41_L2_PREFETCH_NEXT_MB", "20")


def object_segments(name: str, obj: object, depth: int = 4) -> list[Segment]:
    """CUDA tensor ranges reachable from a packed-weight object, deduplicated.

    Packed b12x weights are dataclasses or plain objects holding the device
    tensors their kernels read; walking them prefetches exactly those bytes.
    """
    out: list[Segment] = []
    seen: set[int] = set()

    def visit(label: str, value: object, level: int) -> None:
        if isinstance(value, torch.Tensor):
            segment = _l2pf.tensor_segment(label, value)
            if segment is not None and segment[1] not in seen:
                seen.add(segment[1])
                out.append(segment)
            return
        if level <= 0 or value is None or isinstance(value, (str, bytes, int, float)):
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                visit(f"{label}[{i}]", item, level - 1)
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{label}.{key}", item, level - 1)
            return
        items: Iterable[tuple[str, object]]
        if dataclasses.is_dataclass(value):
            items = (
                (f.name, getattr(value, f.name, None))
                for f in dataclasses.fields(value)
            )
        elif hasattr(value, "__dict__"):
            items = vars(value).items()
        else:
            items = (
                (s, getattr(value, s, None))
                for s in getattr(type(value), "__slots__", ())
            )
        for key, item in items:
            if not key.startswith("__"):
                visit(f"{label}.{key}", item, level - 1)

    visit(name, obj, depth)
    return out


def linear_segments(name: str, linear: torch.nn.Module | None) -> list[Segment]:
    """Bytes read by a V4.1 linear: its packed b12x weight, else its weight."""
    if linear is None:
        return []
    packed = getattr(linear, "b12x_weight", None)
    if packed is not None:
        return object_segments(f"{name}.b12x_weight", packed)
    segment = _l2pf.tensor_segment(f"{name}.weight", getattr(linear, "weight", None))
    return [segment] if segment is not None else []


def param_segments(name: str, owner: object, attrs: tuple[str, ...]) -> list[Segment]:
    out = []
    for attr in attrs:
        segment = _l2pf.tensor_segment(
            f"{name}.{attr}", getattr(owner, attr, None), min_bytes=0
        )
        if segment is not None:
            out.append(segment)
    return out


def make_plan(segments: list[Segment], budget: int, device: torch.device):
    seen: set[int] = set()
    unique = []
    for segment in segments:
        if segment[1] not in seen:
            seen.add(segment[1])
            unique.append(segment)
    plan, _ = _l2pf.make_plan(unique, budget, device)
    return plan


def warmup() -> bool:
    return ENABLED and _l2pf._get_launcher() is not None


def issue(plan: _l2pf.L2PrefetchPlan | None, num_tokens: int) -> None:
    if ENABLED and plan is not None and plan.segs is not None:
        _l2pf.L2Prefetcher.get(plan.segs.device).issue(plan, num_tokens)


def join() -> None:
    if not ENABLED:
        return
    for prefetcher in list(_l2pf.L2Prefetcher._instances.values()):
        prefetcher.join()
