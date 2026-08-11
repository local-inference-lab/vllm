# SPDX-License-Identifier: Apache-2.0
"""Known-answer tests for the progressive-boot memory projection.

The numbers below are the MEASURED ones from the GLM-5.2 TP4 boot that this
module exists to prevent, so a regression here is a regression against
reality rather than against a made-up fixture:

    K3 expert  3,578,892 B/rank      flat-K3 weights  76.14 GiB/rank
    K4 expert  4,758,540 B/rank      device           97,887 MiB
    promotion  1,179,648 B/rank      observed KV      -3.1 GiB at util 0.92
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from vllm.model_executor.layers.quantization.exl3_fungible import (
        memory_preflight as MP,
    )
except ImportError:  # standalone: load by path (no torch needed -- this
    # module is deliberately pure arithmetic so it can be tested on CPU)
    import importlib.util

    _p = (Path(__file__).resolve().parents[2] / "vllm" / "model_executor"
          / "layers" / "quantization" / "exl3_fungible"
          / "memory_preflight.py")
    _spec = importlib.util.spec_from_file_location("fq_preflight_standalone", _p)
    MP = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(MP)

GIB = MP.GIB
BudgetExceeded = MP.BudgetExceeded
check_or_raise = MP.check_or_raise
headroom_in_promotions = MP.headroom_in_promotions
project = MP.project
project_expert_bytes = MP.project_expert_bytes
render = MP.render

K3_BYTES = 3_578_892
K4_BYTES = 4_758_540
SIZES = {3: K3_BYTES, 4: K4_BYTES}
DEVICE = 97_887 * (1 << 20)
# 76.14 GiB flat-K3 weights minus the 64.00 GiB of K3 experts.
DENSE = int(76.14 * GIB) - K3_BYTES * 256 * 75


def _bits(n_k4: int, layers: int = 75, experts: int = 256):
    """Tier bitmap with the first ``n_k4`` experts promoted to K4."""
    flat = [3] * (layers * experts)
    for i in range(n_k4):
        flat[i] = 4
    return {
        layer: flat[layer * experts:(layer + 1) * experts]
        for layer in range(layers)
    }


def _project(n_k4, util=0.92, overhead=5.27, min_kv=4.0):  # noqa: E501
    return project(
        _bits(n_k4), SIZES, DENSE,
        device_total_bytes=DEVICE,
        gpu_memory_utilization=util,
        runtime_overhead_bytes=int(overhead * GIB),
        min_kv_bytes=int(min_kv * GIB),
    )


def test_flat_k3_matches_the_measured_boot():
    """Zero promotions must reproduce the 76.14 GiB / +6.54 GiB KV boot."""
    p = _project(0)
    assert p["weight_bytes"] / GIB == pytest.approx(76.14, abs=0.01)
    assert p["projected_kv_bytes"] / GIB == pytest.approx(6.54, abs=0.05)
    assert p["fits"]


# The failing boot carried 9.19 GiB of non-weight residue, not the 5.27 GiB a
# flat load leaves: progressive staging left 3.92 GiB stranded in the caching
# allocator. That is a separate defect (fixed by the post-load reclaim), and
# these two tests keep the two causes from being confused for each other.
OVERHEAD_WITH_RESIDUE = 9.19
OVERHEAD_AFTER_RECLAIM = 5.27


def test_the_policy_that_actually_failed_is_rejected():
    """5,206 K4 promotions is the attempt-11 policy: -3.1 GiB of KV."""
    p = _project(5_206, overhead=OVERHEAD_WITH_RESIDUE)
    assert p["weight_bytes"] / GIB == pytest.approx(81.86, abs=0.02)
    assert p["projected_kv_bytes"] / GIB == pytest.approx(-3.1, abs=0.05)
    assert not p["fits"]


def test_util_095_does_not_rescue_the_failing_policy():
    """The auto-retry at 0.95 was doomed too -- prove the check knows it,
    in the five seconds it takes rather than the 62 minutes it took."""
    p = _project(5_206, util=0.95, overhead=OVERHEAD_WITH_RESIDUE)
    assert p["projected_kv_bytes"] / GIB == pytest.approx(-0.23, abs=0.05)
    assert not p["fits"]


def test_reclaiming_the_residue_alone_does_not_rescue_it_either():
    """Freeing all 3.92 GiB of allocator residue lifts KV to +0.82 GiB --
    real progress, still far under a usable floor. Both fixes are needed;
    neither substitutes for the other."""
    p = _project(5_206, overhead=OVERHEAD_AFTER_RECLAIM)
    assert p["projected_kv_bytes"] / GIB == pytest.approx(0.82, abs=0.05)
    assert not p["fits"]


def test_both_fixes_together_admit_a_trimmed_policy():
    """With the residue reclaimed and util at 0.95, the envelope that fits is
    ~3,000 promotions, not 5,206 -- the number the demo must be sized to."""
    p = _project(3_000, util=0.95, overhead=OVERHEAD_AFTER_RECLAIM)
    assert p["fits"]
    assert p["projected_kv_bytes"] / GIB > 4.0


def test_remedy_is_expressed_in_experts_not_bytes():
    p = _project(5_206)
    n = headroom_in_promotions(p, SIZES, 3, 4)
    # Demoting n experts must actually clear the shortfall, and n-1 must not:
    # an off-by-one here hands the operator an instruction that still OOMs.
    assert _project(5_206 - n)["fits"]
    assert not _project(5_206 - n + 1)["fits"]
    assert any(f"demote {n:,}" in line for line in render(p, SIZES))


def test_census_counts_every_expert_exactly_once():
    total, census = project_expert_bytes(_bits(1_000), SIZES)
    assert census == {3: 75 * 256 - 1_000, 4: 1_000}
    assert sum(census.values()) == 75 * 256
    assert total == census[3] * K3_BYTES + census[4] * K4_BYTES


def test_unsizeable_k_raises_rather_than_undercounting():
    """A K we cannot size must not silently contribute zero bytes -- that is
    precisely the direction that green-lights an OOM."""
    with pytest.raises(KeyError, match="K5"):
        project_expert_bytes({0: [3, 5]}, SIZES)


def test_check_or_raise_blocks_by_default(caplog):
    with pytest.raises(BudgetExceeded, match="short by"):
        check_or_raise(_project(5_206), SIZES, lambda m: None)


def test_enforcement_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_FQ_BUDGET_ENFORCE", "0")
    seen: list[str] = []
    check_or_raise(_project(5_206), SIZES, seen.append)
    assert any("enforcement disabled" in m for m in seen)


def test_fitting_policy_passes_silently():
    check_or_raise(_project(2_000), SIZES, lambda m: None)


@pytest.mark.parametrize("n", [0, 1, 1_000, 19_200])
def test_projection_is_monotonic_in_promotions(n):
    """More K4 experts must never project MORE KV cache."""
    p = _project(n)
    if n < 19_200:
        assert _project(n + 1)["projected_kv_bytes"] < p["projected_kv_bytes"]
