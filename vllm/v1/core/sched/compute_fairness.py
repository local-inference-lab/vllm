# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

ComputeServiceClass = Literal["decode", "prefill"]


@dataclass
class PrefillComputeShareController:
    """Allocate contended execution time with weighted virtual runtime."""

    prefill_compute_share: float
    decode_virtual_runtime: float = 0.0
    prefill_virtual_runtime: float = 0.0
    contention_active: bool = False
    last_decode_seconds: float | None = None
    last_prefill_seconds: float | None = None
    decode_reservations: deque[float] = field(default_factory=deque)
    prefill_reservations: deque[float] = field(default_factory=deque)

    def select(
        self, *, decode_runnable: bool, prefill_runnable: bool
    ) -> ComputeServiceClass | None:
        """Select the next runnable service class."""
        if not decode_runnable or not prefill_runnable:
            if not self.has_pending_reservations:
                self.reset()
            if decode_runnable:
                return "decode"
            if prefill_runnable:
                return "prefill"
            return None

        if not self.contention_active:
            self.decode_virtual_runtime = 0.0
            self.prefill_virtual_runtime = 0.0
            self.contention_active = True

        # Until a class has one measured completion, allow only one of its
        # quanta in flight when the other class can run. This prevents an async
        # batch queue from filling with duplicate, unpriced work at startup.
        decode_unmeasured = self.last_decode_seconds is None and bool(
            self.decode_reservations
        )
        prefill_unmeasured = self.last_prefill_seconds is None and bool(
            self.prefill_reservations
        )
        if decode_unmeasured != prefill_unmeasured:
            return "prefill" if decode_unmeasured else "decode"
        if decode_unmeasured and prefill_unmeasured:
            if len(self.decode_reservations) < len(self.prefill_reservations):
                return "decode"
            if len(self.prefill_reservations) < len(self.decode_reservations):
                return "prefill"

        if self.prefill_virtual_runtime < self.decode_virtual_runtime:
            return "prefill"
        return "decode"

    @property
    def has_pending_reservations(self) -> bool:
        return bool(self.decode_reservations or self.prefill_reservations)

    def dispatch(self, service_class: ComputeServiceClass, *, contended: bool) -> None:
        """Reserve estimated service before an async batch can be selected."""
        if not contended:
            return

        if service_class == "prefill":
            reservation = (
                0.0
                if self.last_prefill_seconds is None
                else self.last_prefill_seconds / self.prefill_compute_share
            )
            self.prefill_virtual_runtime += reservation
            self.prefill_reservations.append(reservation)
        else:
            reservation = (
                0.0
                if self.last_decode_seconds is None
                else self.last_decode_seconds / (1.0 - self.prefill_compute_share)
            )
            self.decode_virtual_runtime += reservation
            self.decode_reservations.append(reservation)

    def record(
        self,
        service_class: ComputeServiceClass,
        elapsed_seconds: float,
        *,
        contended: bool,
    ) -> None:
        """Charge a completed execution quantum to its actual service class."""
        if not contended:
            return

        if service_class == "prefill":
            reservation = (
                self.prefill_reservations.popleft()
                if self.prefill_reservations
                else 0.0
            )
            if elapsed_seconds <= 0.0:
                self.prefill_virtual_runtime -= reservation
                return
            charge = elapsed_seconds / self.prefill_compute_share
            self.prefill_virtual_runtime += charge - reservation
            self.last_prefill_seconds = elapsed_seconds
            if self.prefill_virtual_runtime - self.decode_virtual_runtime > charge:
                self.decode_virtual_runtime = self.prefill_virtual_runtime - charge
        else:
            reservation = (
                self.decode_reservations.popleft() if self.decode_reservations else 0.0
            )
            if elapsed_seconds <= 0.0:
                self.decode_virtual_runtime -= reservation
                return
            charge = elapsed_seconds / (1.0 - self.prefill_compute_share)
            self.decode_virtual_runtime += charge - reservation
            self.last_decode_seconds = elapsed_seconds
            if self.decode_virtual_runtime - self.prefill_virtual_runtime > charge:
                self.prefill_virtual_runtime = self.decode_virtual_runtime - charge

    def reset(self) -> None:
        """Discard credit when both classes are no longer runnable."""
        self.decode_virtual_runtime = 0.0
        self.prefill_virtual_runtime = 0.0
        self.contention_active = False
