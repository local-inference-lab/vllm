# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

import pytest

from vllm.v1.core.sched.compute_fairness import (
    PrefillComputeHalfLife,
    PrefillComputeShareController,
)


def _run_controller(
    controller: PrefillComputeShareController,
    *,
    decode_cost: float,
    prefill_cost: float,
    steps: int,
) -> dict[str, float]:
    totals = {"decode": 0.0, "prefill": 0.0}
    for _ in range(steps):
        service_class = controller.select(decode_runnable=True, prefill_runnable=True)
        assert service_class is not None
        elapsed = prefill_cost if service_class == "prefill" else decode_cost
        controller.dispatch(service_class, contended=True)
        totals[service_class] += elapsed
        controller.record(service_class, elapsed, contended=True)
    return totals


@pytest.mark.parametrize(
    ("decode_cost", "prefill_cost"), [(1.0, 1.0), (0.1, 1.0), (2.0, 0.25)]
)
def test_controller_converges_for_equal_and_unequal_costs(
    decode_cost: float, prefill_cost: float
):
    controller = PrefillComputeShareController(0.5)

    totals = _run_controller(
        controller,
        decode_cost=decode_cost,
        prefill_cost=prefill_cost,
        steps=1000,
    )

    prefill_fraction = totals["prefill"] / sum(totals.values())
    assert prefill_fraction == pytest.approx(0.5, abs=0.01)


def test_runtime_cost_change_changes_selected_cadence():
    controller = PrefillComputeShareController(0.5)

    first = _run_controller(controller, decode_cost=0.1, prefill_cost=1.0, steps=200)
    first_prefill_steps = first["prefill"] / 1.0

    second = _run_controller(controller, decode_cost=0.1, prefill_cost=0.2, steps=200)
    second_prefill_steps = second["prefill"] / 0.2

    assert second_prefill_steps > first_prefill_steps


def test_contention_transitions_reset_credit():
    controller = PrefillComputeShareController(0.5)
    controller.select(decode_runnable=True, prefill_runnable=True)
    controller.dispatch("decode", contended=True)
    controller.record("decode", 1.0, contended=True)
    assert controller.decode_virtual_runtime > 0.0

    assert controller.select(decode_runnable=True, prefill_runnable=False) == "decode"
    assert controller.decode_virtual_runtime == 0.0
    assert controller.prefill_virtual_runtime == 0.0
    assert not controller.contention_active

    assert controller.select(decode_runnable=True, prefill_runnable=True) == "decode"


def test_single_runnable_class_receives_full_service():
    controller = PrefillComputeShareController(0.5)

    assert controller.select(decode_runnable=True, prefill_runnable=False) == "decode"
    assert controller.select(decode_runnable=False, prefill_runnable=True) == "prefill"
    assert controller.select(decode_runnable=False, prefill_runnable=False) is None


@pytest.mark.parametrize("prefill_share", [0.2, 0.8, 0.9])
def test_neither_class_starves_at_asymmetric_share(prefill_share: float):
    controller = PrefillComputeShareController(prefill_share)
    totals = _run_controller(controller, decode_cost=0.1, prefill_cost=1.0, steps=1000)

    assert totals["decode"] > 0.0
    assert totals["prefill"] > 0.0
    assert totals["prefill"] / sum(totals.values()) == pytest.approx(
        prefill_share, abs=0.02
    )


def test_outlier_lead_is_bounded_to_one_service_quantum():
    controller = PrefillComputeShareController(0.5)
    controller.select(decode_runnable=True, prefill_runnable=True)
    controller.dispatch("prefill", contended=True)
    controller.record("prefill", 100.0, contended=True)

    normalized_quantum = 100.0 / controller.prefill_compute_share
    lead = controller.prefill_virtual_runtime - controller.decode_virtual_runtime
    assert 0.0 <= lead <= normalized_quantum


def test_ties_favor_decode():
    controller = PrefillComputeShareController(0.5)

    assert controller.select(decode_runnable=True, prefill_runnable=True) == "decode"


def test_two_deep_async_queue_converges_with_unequal_costs():
    controller = PrefillComputeShareController(0.5)
    pending: deque[str] = deque()
    totals = {"decode": 0.0, "prefill": 0.0}

    for _ in range(10_000):
        while len(pending) < 2:
            service_class = controller.select(
                decode_runnable=True, prefill_runnable=True
            )
            assert service_class is not None
            controller.dispatch(service_class, contended=True)
            pending.append(service_class)

        service_class = pending.popleft()
        elapsed = 0.7 if service_class == "prefill" else 0.01
        totals[service_class] += elapsed
        controller.record(service_class, elapsed, contended=True)

    prefill_fraction = totals["prefill"] / sum(totals.values())
    assert prefill_fraction == pytest.approx(0.5, abs=0.01)


def test_zero_elapsed_completion_releases_reservation():
    controller = PrefillComputeShareController(0.5, last_decode_seconds=0.1)
    service_class = controller.select(decode_runnable=True, prefill_runnable=True)
    assert service_class == "decode"
    controller.dispatch(service_class, contended=True)

    assert controller.decode_reservations
    controller.record(service_class, 0.0, contended=True)

    assert not controller.decode_reservations
    assert controller.decode_virtual_runtime == 0.0


def _primed_auto_controller(
    half_life: PrefillComputeHalfLife = None,
) -> PrefillComputeShareController:
    controller = PrefillComputeShareController("auto", half_life)
    controller.last_decode_seconds = 0.1
    controller.last_decode_completed_at = 0.0
    controller.prefill_seconds_per_token = 0.001
    return controller


def test_auto_mode_starts_at_default_and_waits_for_service_feedback():
    controller = PrefillComputeShareController("auto")

    controller.observe_demand(
        prefill_pressure=10.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=0.0,
    )
    controller.observe_demand(
        prefill_pressure=10.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=1.0,
    )

    assert controller.effective_prefill_compute_share == pytest.approx(0.4)


def test_auto_mode_does_not_treat_idle_as_decode_starvation():
    controller = _primed_auto_controller()
    controller.observe_demand(
        prefill_pressure=1.0,
        local_prefill_backlog_tokens=0,
        decode_runnable=True,
        prefill_runnable=False,
        now=100.0,
    )
    controller.observe_demand(
        prefill_pressure=1.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        contention_started=True,
        now=101.0,
    )

    assert controller.decode_pressure == pytest.approx(1.0)
    assert controller.effective_prefill_compute_share == pytest.approx(0.4)


def test_auto_mode_moves_toward_the_class_with_more_pressure():
    controller = _primed_auto_controller()
    controller.observe_demand(
        prefill_pressure=1.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=0.0,
    )
    controller.last_decode_completed_at = 1.0
    controller.observe_demand(
        prefill_pressure=4.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=1.0,
    )
    assert 0.4 < controller.effective_prefill_compute_share <= 0.45

    controller.observe_demand(
        prefill_pressure=1.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=2.0,
    )
    assert controller.effective_prefill_compute_share < 0.45


def test_auto_mode_is_bounded_and_slew_limited():
    controller = _primed_auto_controller()
    controller.observe_demand(
        prefill_pressure=1.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=0.0,
    )

    previous = controller.effective_prefill_compute_share
    for now in range(1, 101):
        controller.last_decode_completed_at = float(now)
        controller.observe_demand(
            prefill_pressure=1000.0,
            local_prefill_backlog_tokens=1000,
            decode_runnable=True,
            prefill_runnable=True,
            now=float(now),
        )
        assert controller.effective_prefill_compute_share - previous <= 0.0500001
        previous = controller.effective_prefill_compute_share

    assert controller.effective_prefill_compute_share == pytest.approx(0.8)


def test_auto_deadband_prevents_churn():
    controller = _primed_auto_controller()
    controller.observe_demand(
        prefill_pressure=1.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=0.0,
    )
    controller.last_decode_completed_at = 1.0
    controller.observe_demand(
        prefill_pressure=1.05,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=1.0,
    )

    assert controller.effective_prefill_compute_share == pytest.approx(0.4)


def test_auto_reservation_uses_dispatch_share():
    controller = PrefillComputeShareController("auto", last_prefill_seconds=1.0)
    controller.effective_prefill_compute_share = 0.2
    controller.dispatch("prefill", contended=True)
    controller.effective_prefill_compute_share = 0.8

    controller.record("prefill", 1.0, contended=True, scheduled_tokens=1000, now=1.0)

    assert controller.prefill_virtual_runtime == pytest.approx(5.0)


def test_responsive_preset_moves_faster_than_smooth_preset():
    smooth = _primed_auto_controller("smooth")
    responsive = _primed_auto_controller("responsive")
    for controller in (smooth, responsive):
        controller.observe_demand(
            prefill_pressure=1.0,
            local_prefill_backlog_tokens=1000,
            decode_runnable=True,
            prefill_runnable=True,
            now=0.0,
        )
        controller.last_decode_completed_at = 1.0
        controller.observe_demand(
            prefill_pressure=4.0,
            local_prefill_backlog_tokens=1000,
            decode_runnable=True,
            prefill_runnable=True,
            now=1.0,
        )

    assert (
        responsive.effective_prefill_compute_share
        > smooth.effective_prefill_compute_share
        > 0.4
    )


@pytest.mark.parametrize(
    ("setting", "resolved", "max_slew"),
    [
        (None, 2.0, 0.05),
        ("smooth", 2.0, 0.05),
        ("responsive", 0.5, 0.2),
        (5.0, 5.0, 0.02),
        (0.2, 0.2, 0.5),
    ],
)
def test_auto_half_life_presets_and_numeric_values(setting, resolved, max_slew):
    controller = _primed_auto_controller(setting)

    assert controller.effective_prefill_compute_half_life == pytest.approx(resolved)
    assert (
        controller.AUTO_SLEW_HALF_LIFE_PRODUCT
        / controller.effective_prefill_compute_half_life
        == pytest.approx(max_slew)
    )


def test_live_numeric_reconfigure_preserves_dispatch_accounting():
    controller = PrefillComputeShareController(0.4, last_prefill_seconds=1.0)
    controller.dispatch("prefill", contended=True)
    controller.prefill_virtual_runtime += 50.0

    controller.reconfigure(0.9)

    assert controller.effective_prefill_compute_share == pytest.approx(0.9)
    assert controller.prefill_virtual_runtime == pytest.approx(2.5)
    assert list(controller.prefill_reservations) == [(2.5, 0.4)]

    controller.record("prefill", 1.0, contended=True)
    assert controller.prefill_virtual_runtime == pytest.approx(2.5)


def test_live_half_life_switch_preserves_effective_share_and_feedback():
    controller = _primed_auto_controller("smooth")
    controller.effective_prefill_compute_share = 0.6
    controller.last_adjustment_at = 10.0

    controller.reconfigure("auto", "responsive")

    assert controller.prefill_compute_share == "auto"
    assert controller.prefill_compute_half_life == "responsive"
    assert controller.effective_prefill_compute_half_life == pytest.approx(0.5)
    assert controller.effective_prefill_compute_share == pytest.approx(0.6)
    assert controller.last_decode_seconds == pytest.approx(0.1)
    assert controller.prefill_seconds_per_token == pytest.approx(0.001)
    assert controller.last_adjustment_at is None


def test_live_numeric_to_auto_switch_clamps_previous_share():
    controller = PrefillComputeShareController(0.9)

    controller.reconfigure("auto", "smooth")

    assert controller.effective_prefill_compute_share == pytest.approx(0.8)


@pytest.mark.parametrize("half_life", [0.0, -1.0, float("inf"), float("nan")])
def test_auto_rejects_invalid_half_life(half_life):
    with pytest.raises(ValueError, match="prefill_compute_half_life"):
        PrefillComputeShareController("auto", half_life)


def test_fixed_share_rejects_half_life():
    with pytest.raises(ValueError, match="requires prefill_compute_share='auto'"):
        PrefillComputeShareController(0.4, "responsive")


def test_numeric_share_never_adapts():
    controller = PrefillComputeShareController(
        0.65,
        last_decode_seconds=0.1,
        prefill_seconds_per_token=0.001,
        last_decode_completed_at=0.0,
    )
    controller.observe_demand(
        prefill_pressure=1000.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=0.0,
    )
    controller.observe_demand(
        prefill_pressure=1000.0,
        local_prefill_backlog_tokens=1000,
        decode_runnable=True,
        prefill_runnable=True,
        now=100.0,
    )

    assert controller.effective_prefill_compute_share == pytest.approx(0.65)
