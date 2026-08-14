# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator-facing expert-composition table."""
from __future__ import annotations

import pytest

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        occupancy_table as ot,
    )
except ImportError:  # standalone: load by path (no compiled vllm._C needed)
    import importlib.util
    import sys
    from pathlib import Path as _P

    _dir = (_P(__file__).resolve().parents[2] / "vllm" / "model_executor"
            / "layers" / "quantization" / "exl3_fungible")
    _spec = importlib.util.spec_from_file_location(
        "fq_occupancy_table_standalone", _dir / "occupancy_table.py")
    ot = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = ot
    _spec.loader.exec_module(ot)


def occ(**layers):
    return {int(k[1:]): v for k, v in layers.items()}


def test_renders_all_four_tier_columns_even_when_empty():
    out = ot.render(occ(L3={3: 256}))
    for k in (2, 3, 4, 5):
        assert f"K{k}" in out
    # a tier with no experts must show as 0, not vanish
    assert "      0" in out


def test_totals_row_sums_every_layer():
    out = ot.render(occ(L3={3: 192, 5: 64}, L4={3: 200, 5: 56}))
    assert "total" in out
    assert "    392" in out  # 192+200
    assert "    120" in out  # 64+56


def test_mean_bits_is_computed_and_correct():
    # 128 experts at K3, 128 at K5 -> mean exactly 4
    out = ot.render(occ(L0={3: 128, 5: 128}))
    assert "mean bits/expert: 4.0000" in out


def test_change_column_shows_signed_per_tier_delta():
    prev = occ(L3={3: 200, 5: 56})
    cur = occ(L3={3: 192, 5: 64})
    out = ot.render(cur, prev)
    assert "-8 K3" in out and "+8 K5" in out


def test_changed_layers_are_flagged():
    prev = occ(L3={3: 256}, L4={3: 256})
    cur = occ(L3={3: 250, 5: 6}, L4={3: 256})
    out = ot.render(cur, prev)
    rows = [l for l in out.splitlines() if l.strip().startswith("3")]
    assert any("*" in r for r in rows)


def test_diff_only_omits_unchanged_layers_but_totals_stay_complete():
    prev = occ(L1={3: 256}, L2={3: 256}, L3={3: 256})
    cur = occ(L1={3: 256}, L2={3: 256}, L3={3: 250, 5: 6})
    out = ot.render(cur, prev, diff_only=True)
    assert "2 unchanged layers omitted" in out
    # totals must cover ALL layers, not just the shown one
    assert "    762" in out  # 256+256+250


def test_diff_only_with_no_changes_says_so_explicitly():
    same = occ(L1={3: 256})
    out = ot.render(same, same, diff_only=True)
    assert "no tier changes" in out
    # and still reports where things stand, so it is not mistaken for a fault
    assert "K3=256" in out


def test_mean_bits_delta_shown_against_previous():
    prev = occ(L0={3: 256})
    cur = occ(L0={3: 128, 5: 128})
    out = ot.render(cur, prev)
    assert "mean bits/expert: 4.0000 (+1.0000)" in out


def test_empty_occupancy_does_not_crash():
    assert "no MoE layers" in ot.render({})


def test_row_cap_is_reported_not_silent():
    big = {i: {3: 256} for i in range(200)}
    out = ot.render(big, max_rows=10)
    assert "more rows suppressed" in out
    # totals still cover all 200 layers
    assert str(200 * 256) in out


def test_from_membership_counts_high_tier():
    membership = [[True, False, False, True], [False, False, False, False]]
    got = ot.from_membership(membership, [7, 8], bits=(3, 4))
    assert got == {7: {3: 2, 4: 2}, 8: {3: 4, 4: 0}}


def test_string_keys_are_accepted():
    """Prometheus-derived dicts arrive with string layer/tier keys."""
    out = ot.render({"3": {"3": 192, "5": 64}})
    assert "192" in out and "64" in out


@pytest.mark.parametrize("n", [1, 2, 3])
def test_table_is_stable_width(n):
    out = ot.render({i: {3: 256} for i in range(n)})
    body = [l for l in out.splitlines() if l.startswith("  ") and "|" in l]
    assert len({len(l.split("|")[1]) for l in body}) == 1
