# SPDX-License-Identifier: Apache-2.0
"""Project the progressive boot's weight footprint BEFORE loading anything.

A mixed-K checkpoint's parameter footprint is fully determined by the tier
bitmap, which is written before the engine starts. So "will this policy fit?"
is arithmetic, not an experiment -- yet the failure mode we actually hit was
a 62-minute load followed by::

    Available KV cache memory: -3.1 GiB
    ValueError: No available memory for the cache blocks.

Sixty-two minutes to learn something computable in five seconds. This module
does that arithmetic, and :func:`check_or_raise` runs it at the top of the
progressive loader so an oversized policy is rejected while the operator can
still do something about it.

The projection is deliberately anchored on MEASURED per-K segment sizes rather
than a bits-per-weight model: EXL3 experts carry fixed-size ``suh``/``svh``/
``mcg`` companions alongside the K-proportional trellis, so "K4 is 4/3 of K3"
is wrong by a few percent -- and a few percent of 64 GiB is the entire KV
cache.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

GIB = 1 << 30


class BudgetExceeded(RuntimeError):
    """The policy's weights cannot leave room for a usable KV cache."""


def project_expert_bytes(
    bits_by_layer: Mapping[int, Iterable[int]],
    expert_bytes_by_k: Mapping[int, int],
) -> tuple[int, dict[int, int]]:
    """Total expert bytes for one rank, plus the per-K expert census.

    ``bits_by_layer`` maps layer -> per-expert K. Unknown Ks raise rather than
    silently contributing zero: a policy referencing a K we cannot size is
    exactly the case where a quiet under-estimate would green-light an OOM.
    """
    census: dict[int, int] = {}
    total = 0
    for layer, bits in bits_by_layer.items():
        for k in bits:
            k = int(k)
            try:
                total += expert_bytes_by_k[k]
            except KeyError:
                raise KeyError(
                    f"layer {layer}: no measured expert size for K{k} "
                    f"(have {sorted(expert_bytes_by_k)}) -- cannot project "
                    f"the footprint, so cannot certify it fits"
                ) from None
            census[k] = census.get(k, 0) + 1
    return total, census


def project(
    bits_by_layer: Mapping[int, Iterable[int]],
    expert_bytes_by_k: Mapping[int, int],
    dense_bytes: int,
    *,
    device_total_bytes: int,
    gpu_memory_utilization: float,
    runtime_overhead_bytes: int,
    min_kv_bytes: int,
) -> dict:
    """Return the full projection. Pure arithmetic; no CUDA, no I/O."""
    expert_bytes, census = project_expert_bytes(bits_by_layer, expert_bytes_by_k)
    weights = expert_bytes + dense_bytes
    budget = int(device_total_bytes * gpu_memory_utilization)
    kv = budget - weights - runtime_overhead_bytes
    return {
        "expert_bytes": expert_bytes,
        "dense_bytes": dense_bytes,
        "weight_bytes": weights,
        "budget_bytes": budget,
        "runtime_overhead_bytes": runtime_overhead_bytes,
        "projected_kv_bytes": kv,
        "min_kv_bytes": min_kv_bytes,
        "fits": kv >= min_kv_bytes,
        "shortfall_bytes": max(0, min_kv_bytes - kv),
        "census": census,
        "experts": sum(census.values()),
    }


def headroom_in_promotions(
    proj: dict, expert_bytes_by_k: Mapping[int, int], low_k: int, high_k: int
) -> int:
    """How many low_k->high_k promotions the shortfall is worth.

    An operator staring at "3.4 GiB over" cannot act on it. "demote 3,094
    experts from K4 to K3" is an instruction.
    """
    step = expert_bytes_by_k[high_k] - expert_bytes_by_k[low_k]
    if step <= 0:
        return 0
    return -(-proj["shortfall_bytes"] // step)  # ceil


def render(proj: dict, expert_bytes_by_k: Mapping[int, int]) -> list[str]:
    g = lambda b: f"{b / GIB:8.2f} GiB"  # noqa: E731
    census = ", ".join(
        f"K{k}={n:,}" for k, n in sorted(proj["census"].items())
    )
    lines = [
        "FQ memory preflight (per rank, projected from the tier bitmap):",
        f"    experts        {g(proj['expert_bytes'])}   "
        f"({proj['experts']:,} experts: {census})",
        f"    non-expert     {g(proj['dense_bytes'])}",
        f"    weights total  {g(proj['weight_bytes'])}",
        f"    runtime overhd {g(proj['runtime_overhead_bytes'])}   "
        f"(allocator + activations + graphs)",
        f"    device budget  {g(proj['budget_bytes'])}",
        f"    => KV cache    {g(proj['projected_kv_bytes'])}   "
        f"(need >= {g(proj['min_kv_bytes'])})",
    ]
    if not proj["fits"]:
        ks = sorted(expert_bytes_by_k)
        lines.append(
            f"    SHORT BY       {g(proj['shortfall_bytes'])}"
        )
        for hi in reversed(ks[1:]):
            lo = ks[ks.index(hi) - 1]
            if proj["census"].get(hi):
                n = headroom_in_promotions(proj, expert_bytes_by_k, lo, hi)
                lines.append(
                    f"    remedy         demote {n:,} experts K{hi}->K{lo} "
                    f"(or raise --gpu-memory-utilization)"
                )
                break
    return lines


def _env_bytes(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    raw = raw.strip().lower()
    mult = 1
    if raw.endswith("g") or raw.endswith("gib"):
        mult, raw = GIB, raw.rstrip("bi").rstrip("g")
    elif raw.endswith("m") or raw.endswith("mib"):
        mult, raw = 1 << 20, raw.rstrip("bi").rstrip("m")
    return int(float(raw) * mult)


def check_or_raise(proj: dict, expert_bytes_by_k: Mapping[int, int], log) -> None:
    """Log the projection; raise :class:`BudgetExceeded` if it cannot fit.

    ``VLLM_FQ_BUDGET_ENFORCE=0`` downgrades the raise to a warning -- for the
    case where the operator knows the overhead estimate is pessimistic and
    wants the engine's own profiler to have the final say.
    """
    for line in render(proj, expert_bytes_by_k):
        log(line)
    if proj["fits"]:
        return
    msg = (
        f"FQ memory preflight FAILED: projected KV cache "
        f"{proj['projected_kv_bytes'] / GIB:.2f} GiB is below the "
        f"{proj['min_kv_bytes'] / GIB:.2f} GiB floor "
        f"(short by {proj['shortfall_bytes'] / GIB:.2f} GiB). "
        f"Refusing to spend a full model load on a policy that cannot serve. "
        f"Set VLLM_FQ_BUDGET_ENFORCE=0 to try anyway."
    )
    if os.environ.get("VLLM_FQ_BUDGET_ENFORCE", "1") == "0":
        log("WARNING: " + msg + " [enforcement disabled]")
        return
    raise BudgetExceeded(msg)
