# SPDX-License-Identifier: Apache-2.0
"""Human-readable expert-composition matrix for the engine log.

An operator watching a fungible-quant serve needs to answer one question at a
glance: *which experts are currently held at which bitrate, and what changed
since last time?* The Prometheus gauge ``fq_tier_occupancy{layer,tier}``
carries the same numbers, but a scrape target is not something you can eyeball
in a terminal at 3am, and it does not remember what the previous shape was.

So: a layer x K-tier table, printed once at startup (the composition the
checkpoint booted with) and periodically thereafter, with the delta against
the previous print called out explicitly.

Periodic prints are DIFF-ONLY by default. GLM-5.2 has 90+ MoE layers, and
dumping a 90-row table every interval would bury the log — the thing an
operator actually wants is "these three layers moved, everything else is
where you left it".
"""
from __future__ import annotations

from collections.abc import Mapping

# Tier columns we render. Fixed and explicit rather than derived from the
# data, so a tier dropping to zero experts still shows as a 0 column instead
# of silently vanishing from the table.
TIERS: tuple[int, ...] = (2, 3, 4, 5)


def format_bytes(n: int | float | None) -> str:
    """Binary-unit byte size, matching how the budget knob is spelled."""
    if n is None:
        return "unbounded"
    neg = "-" if n < 0 else ""
    n = abs(float(n))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            return (f"{neg}{int(n)} {unit}" if unit == "B"
                    else f"{neg}{n:.2f} {unit}")
        n /= 1024.0
    return f"{neg}{n:.2f} TiB"  # pragma: no cover - loop always returns


def budget_lines(budget: Mapping | None) -> list[str]:
    """Footer for the composition table: ceiling, actual use, headroom.

    ``budget`` is :meth:`policy.MemoryBudget.summary` output. The
    per-expert byte figures and — importantly — WHERE THEY CAME FROM are
    printed with it: a headroom number is only as trustworthy as the
    geometry it was computed from, so the source never travels separately
    from the number.
    """
    if not budget:
        return []
    used = int(budget.get("used_bytes", 0))
    limit = budget.get("limit_bytes")
    out: list[str] = []
    if limit is None:
        out.append(f"  memory: {format_bytes(used)} used "
                   f"(no byte budget set — fixed cardinality only)")
    else:
        limit = int(limit)
        pct = (100.0 * used / limit) if limit else 0.0
        head = budget.get("headroom_bytes")
        promos = budget.get("headroom_promotions")
        if head is not None and head < 0:
            out.append(
                f"  memory budget: {format_bytes(limit)} | used "
                f"{format_bytes(used)} ({pct:.1f}%) | OVER BUDGET by "
                f"{format_bytes(-head)}")
        else:
            out.append(
                f"  memory budget: {format_bytes(limit)} | used "
                f"{format_bytes(used)} ({pct:.1f}%) | headroom "
                f"{format_bytes(head)}"
                + ("" if promos is None
                   else f" = {promos} more K3->K4 promotions"))
    per_k = budget.get("per_k_bytes") or {}
    if per_k:
        cells = " ".join(f"K{k}={int(v)}" for k, v in sorted(
            (int(a), b) for a, b in per_k.items()))
        out.append(f"  bytes/expert/rank: {cells}")
    src = budget.get("source")
    if src:
        flag = "REFERENCE (not this checkpoint!) " if budget.get(
            "is_reference") else ""
        out.append(f"  byte model: {flag}{src}")
    return out


def _budget_oneline(budget: Mapping | None) -> str:
    """Compact form for the paths that return a single line."""
    if not budget:
        return ""
    limit = budget.get("limit_bytes")
    used = format_bytes(int(budget.get("used_bytes", 0)))
    if limit is None:
        return f", {used} used, no byte budget"
    promos = budget.get("headroom_promotions")
    return (f", {used}/{format_bytes(int(limit))} used"
            + ("" if promos is None else f", headroom {promos} promotions"))


def _fmt_delta(cur: Mapping[int, int], prev: Mapping[int, int] | None) -> str:
    if prev is None:
        return ""
    parts = []
    for k in TIERS:
        d = int(cur.get(k, 0)) - int(prev.get(k, 0))
        if d:
            parts.append(f"{d:+d} K{k}")
    return " / ".join(parts)


def render(
    occupancy: Mapping[int, Mapping[int, int]],
    previous: Mapping[int, Mapping[int, int]] | None = None,
    *,
    num_experts: int | None = None,
    title: str = "expert composition",
    diff_only: bool = False,
    max_rows: int = 128,
    budget: Mapping | None = None,
) -> str:
    """Render ``{layer: {k: count}}`` as a table.

    ``previous`` enables the change column. ``diff_only`` restricts rows to
    layers whose composition actually moved (the totals line always covers
    every layer, changed or not, so the summary stays truthful).
    ``budget`` (``policy.MemoryBudget.summary`` output) appends the memory
    ceiling, the footprint the shown occupancy actually costs, and the
    remaining promotion headroom.
    """
    if not occupancy:
        return (f"FQ {title}: no MoE layers instrumented"
                + _budget_oneline(budget))

    layers = sorted(int(x) for x in occupancy.keys())
    norm: dict[int, dict[int, int]] = {
        int(l): {int(k): int(v) for k, v in occupancy[l].items()}
        for l in occupancy
    }
    pnorm: dict[int, dict[int, int]] | None = None
    if previous:
        pnorm = {int(l): {int(k): int(v) for k, v in previous[l].items()}
                 for l in previous}

    changed = set()
    if pnorm is not None:
        for l in layers:
            if norm.get(l, {}) != pnorm.get(l, {}):
                changed.add(l)

    shown = [l for l in layers if l in changed] if diff_only else layers
    omitted_note = ""
    if diff_only:
        if not shown:
            # Say it plainly. A silent "no table" reads as "telemetry broke".
            return (f"FQ {title}: no tier changes across {len(layers)} layers "
                    f"(totals {_totals_str(norm)})" + _budget_oneline(budget))
        omitted_note = (f"  ({len(layers) - len(shown)} unchanged layers "
                        f"omitted)")
    truncated = 0
    if len(shown) > max_rows:
        truncated = len(shown) - max_rows
        shown = shown[:max_rows]

    head = f"FQ {title}: {len(layers)} MoE layers"
    if num_experts:
        head += f" x {num_experts} experts"
    head += omitted_note

    lines = [head]
    cols = "".join(f"{'K'+str(k):>7}" for k in TIERS)
    lines.append(f"  {'layer':>5} |{cols} |  change")
    lines.append(f"  {'-'*5}-+{'-'*(7*len(TIERS))}-+{'-'*20}")
    for l in shown:
        row = norm.get(l, {})
        cells = "".join(f"{row.get(k, 0):>7}" for k in TIERS)
        mark = _fmt_delta(row, pnorm.get(l) if pnorm else None)
        flag = "*" if l in changed else " "
        lines.append(f"  {l:>5}{flag}|{cells} | {mark}")
    if truncated:
        lines.append(f"  ... {truncated} more rows suppressed "
                     f"(max_rows={max_rows})")
    lines.append(f"  {'-'*5}-+{'-'*(7*len(TIERS))}-+{'-'*20}")
    tot = {k: sum(norm[l].get(k, 0) for l in layers) for k in TIERS}
    ptot = None
    if pnorm is not None:
        ptot = {k: sum(pnorm.get(l, {}).get(k, 0) for l in layers)
                for k in TIERS}
    cells = "".join(f"{tot[k]:>7}" for k in TIERS)
    lines.append(f"  {'total':>5} |{cells} | {_fmt_delta(tot, ptot)}")

    # Mean bits/expert is the one number that says whether the fleet got more
    # accurate or cheaper overall; it is what the fixed-cardinality invariant
    # is supposed to hold roughly constant.
    n = sum(tot.values())
    if n:
        mean_bits = sum(k * tot[k] for k in TIERS) / n
        extra = ""
        if ptot and sum(ptot.values()):
            pm = sum(k * ptot[k] for k in TIERS) / sum(ptot.values())
            extra = f" ({mean_bits - pm:+.4f})"
        lines.append(f"  mean bits/expert: {mean_bits:.4f}{extra}")
    lines.extend(budget_lines(budget))
    return "\n".join(lines)


def _totals_str(norm: Mapping[int, Mapping[int, int]]) -> str:
    tot = {k: sum(int(r.get(k, 0)) for r in norm.values()) for k in TIERS}
    return " ".join(f"K{k}={tot[k]}" for k in TIERS if tot[k])


def from_membership(membership, layer_ids, bits=(3, 4)) -> dict:
    """Build ``{layer: {k: count}}`` from a boolean tier-1 membership array.

    ``membership[i][e]`` true means expert ``e`` of ``layer_ids[i]`` sits in
    the HIGH tier (``bits[1]``), false the low tier (``bits[0]``). This is the
    representation the policy engine works in.
    """
    lo, hi = bits
    out: dict[int, dict[int, int]] = {}
    for i, layer in enumerate(layer_ids):
        row = membership[i]
        n_hi = int(sum(1 for v in row if v))
        out[int(layer)] = {lo: int(len(row)) - n_hi, hi: n_hi}
    return out
