# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

import pytest

from vllm.v1.core.sched.compute_fairness import PrefillComputeShareController


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


def test_neither_class_starves_at_asymmetric_share():
    controller = PrefillComputeShareController(0.9)
    totals = _run_controller(controller, decode_cost=0.1, prefill_cost=1.0, steps=1000)

    assert totals["decode"] > 0.0
    assert totals["prefill"] > 0.0
    assert totals["prefill"] / sum(totals.values()) == pytest.approx(0.9, abs=0.02)


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
