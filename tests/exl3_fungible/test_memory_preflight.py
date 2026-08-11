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


# RETRACTED HYPOTHESIS, kept as a cautionary constant.
#
# The failing boot carried 9.19 GiB of non-weight overhead against the 5.27 GiB
# implied by an earlier flat-K3 run, and I attributed the 3.92 GiB gap to
# progressive staging stranded in the caching allocator. Direct measurement
# refuted it. With the reclaim moved to the correct hook -- after
# process_weights_after_loading, where the footprint really is 79.17 GiB --
# gc + empty_cache freed EXACTLY 0.00 GiB:
#
#   FQ reclaim: reserved 79.39 -> 79.39 GiB, freed 0.00 GiB;
#               weight footprint 79.17 GiB
#
# reserved exceeds allocated by 0.22 GiB, so there is no residue to reclaim.
# The two runs differed in configuration (max_model_len 32768 vs 8192, graph
# capture on vs off), not in allocator behaviour, and 5.27 was never a
# constant of this system.
#
# MEASURED, at util 0.95 / eager / max_model_len 8192:
#   budget 90.81 - weights 79.08 - KV 3.67 = 8.06 GiB of real overhead.
OVERHEAD_MEASURED = 8.06
# Retained only so the tests below still describe the boot that failed.
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


def test_reclaiming_the_residue_alone_would_not_have_rescued_it_either():
    """Hypothetical, and now known to be unavailable: even if all 3.92 GiB had
    been reclaimable, KV reaches only +0.82 GiB -- still under any usable
    floor. The policy was always the binding constraint; the residue theory
    was never going to save it. Measurement later showed there is no residue
    at all (see OVERHEAD_MEASURED)."""
    p = _project(5_206, overhead=OVERHEAD_AFTER_RECLAIM)
    assert p["projected_kv_bytes"] / GIB == pytest.approx(0.82, abs=0.05)
    assert not p["fits"]


def test_measured_overhead_reproduces_the_observed_kv():
    """Known-answer against the boot that actually served: the fitted policy
    (2,658 K4) at util 0.95 with the MEASURED 8.06 GiB overhead must land on
    the 3.67 GiB of KV the engine reported."""
    p = _project(2_658, util=0.95, overhead=OVERHEAD_MEASURED)
    assert p["weight_bytes"] / GIB == pytest.approx(79.08, abs=0.05)
    assert p["projected_kv_bytes"] / GIB == pytest.approx(3.67, abs=0.10)


def test_the_envelope_that_actually_fits_is_far_below_the_seeded_one():
    """Under MEASURED overhead, clearing a 4 GiB KV floor needs ~2,350
    promotions -- not the 5,206 the seeded policy asked for. The demo envelope
    is set by the card, not by the 3.42bpw coder quant's shape."""
    assert not _project(2_658, util=0.95, overhead=OVERHEAD_MEASURED)["fits"]
    assert _project(2_300, util=0.95, overhead=OVERHEAD_MEASURED)["fits"]


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
