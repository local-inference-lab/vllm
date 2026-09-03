# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Literal

ComputeServiceClass = Literal["decode", "prefill"]
PrefillComputeShare = float | Literal["auto"]


@dataclass
class PrefillComputeShareController:
    """Allocate contended execution time with weighted virtual runtime."""

    AUTO_INITIAL_SHARE: ClassVar[float] = 0.4
    AUTO_MIN_SHARE: ClassVar[float] = 0.2
    AUTO_MAX_SHARE: ClassVar[float] = 0.8
    AUTO_HALF_LIFE_SECONDS: ClassVar[float] = 2.0
    AUTO_MAX_SLEW_PER_SECOND: ClassVar[float] = 0.05
    AUTO_PRESSURE_DEADBAND: ClassVar[float] = 0.1
    PREFILL_COST_EWMA_ALPHA: ClassVar[float] = 0.2

    prefill_compute_share: PrefillComputeShare
    decode_virtual_runtime: float = 0.0
    prefill_virtual_runtime: float = 0.0
    contention_active: bool = False
    last_decode_seconds: float | None = None
    last_prefill_seconds: float | None = None
    decode_reservations: deque[tuple[float, float]] = field(default_factory=deque)
    prefill_reservations: deque[tuple[float, float]] = field(default_factory=deque)
    effective_prefill_compute_share: float = field(init=False)
    prefill_seconds_per_token: float | None = None
    decode_pressure: float = 1.0
    prefill_pressure: float = 1.0
    local_prefill_backlog_tokens: int = 0
    last_decode_completed_at: float | None = None
    last_auto_adjustment_at: float | None = None

    def __post_init__(self) -> None:
        self.effective_prefill_compute_share = (
            self.AUTO_INITIAL_SHARE
            if self.auto_enabled
            else float(self.prefill_compute_share)
        )

    @property
    def auto_enabled(self) -> bool:
        """Whether the controller should tune its effective share."""
        return self.prefill_compute_share == "auto"

    def observe_demand(
        self,
        *,
        prefill_pressure: float,
        local_prefill_backlog_tokens: int,
        decode_runnable: bool,
        prefill_runnable: bool,
        contention_started: bool = False,
        now: float | None = None,
    ) -> None:
        """Update the adaptive target from local demand and service feedback.

        Pressure is normalized so one means healthy service. The requested
        share moves toward the class experiencing more slowdown, with a
        deadband, a continuous-time low-pass filter, and a hard slew limit.
        """
        now = time.monotonic() if now is None else now
        if decode_runnable and (not prefill_runnable or contention_started):
            # Decode-only service is intentionally not timed by the fairness
            # path. Refresh its health reference so idle time or a prior
            # prefill-only phase is not mistaken for decode starvation when
            # contention begins.
            self.last_decode_completed_at = now
        self.prefill_pressure = max(prefill_pressure, 1.0)
        self.local_prefill_backlog_tokens = max(local_prefill_backlog_tokens, 0)
        self.decode_pressure = self._decode_pressure(now)

        previous_adjustment = self.last_auto_adjustment_at
        self.last_auto_adjustment_at = now
        if (
            not self.auto_enabled
            or previous_adjustment is None
            or not decode_runnable
            or not prefill_runnable
            or self.last_decode_seconds is None
            or self.prefill_seconds_per_token is None
        ):
            return

        elapsed = min(max(now - previous_adjustment, 0.0), 2.0)
        if elapsed == 0.0:
            return

        pressure_ratio = self.prefill_pressure / self.decode_pressure
        if abs(pressure_ratio - 1.0) <= self.AUTO_PRESSURE_DEADBAND:
            return

        desired_share = self.AUTO_INITIAL_SHARE + (
            (self.AUTO_MAX_SHARE - self.AUTO_INITIAL_SHARE)
            * math.tanh(math.log(pressure_ratio))
        )
        desired_share = min(
            max(desired_share, self.AUTO_MIN_SHARE), self.AUTO_MAX_SHARE
        )
        alpha = 1.0 - math.exp(-math.log(2.0) * elapsed / self.AUTO_HALF_LIFE_SECONDS)
        filtered_share = self.effective_prefill_compute_share + alpha * (
            desired_share - self.effective_prefill_compute_share
        )
        max_change = self.AUTO_MAX_SLEW_PER_SECOND * elapsed
        change = min(
            max(
                filtered_share - self.effective_prefill_compute_share,
                -max_change,
            ),
            max_change,
        )
        self.effective_prefill_compute_share = min(
            max(
                self.effective_prefill_compute_share + change,
                self.AUTO_MIN_SHARE,
            ),
            self.AUTO_MAX_SHARE,
        )

    def _decode_pressure(self, now: float) -> float:
        if self.last_decode_seconds is None or self.last_decode_completed_at is None:
            return 1.0
        elapsed = max(now - self.last_decode_completed_at, 0.0)
        return max(elapsed / self.last_decode_seconds, 1.0)

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

        share = self.effective_prefill_compute_share
        if service_class == "prefill":
            reservation = (
                0.0
                if self.last_prefill_seconds is None
                else self.last_prefill_seconds / share
            )
            self.prefill_virtual_runtime += reservation
            self.prefill_reservations.append((reservation, share))
        else:
            reservation = (
                0.0
                if self.last_decode_seconds is None
                else self.last_decode_seconds / (1.0 - share)
            )
            self.decode_virtual_runtime += reservation
            self.decode_reservations.append((reservation, share))

    def record(
        self,
        service_class: ComputeServiceClass,
        elapsed_seconds: float,
        *,
        contended: bool,
        scheduled_tokens: int = 0,
        now: float | None = None,
    ) -> None:
        """Charge a completed execution quantum to its actual service class."""
        if not contended:
            return

        completed_at = time.monotonic() if now is None else now
        if service_class == "prefill":
            reservation, dispatch_share = (
                self.prefill_reservations.popleft()
                if self.prefill_reservations
                else (0.0, self.effective_prefill_compute_share)
            )
            if elapsed_seconds <= 0.0:
                self.prefill_virtual_runtime -= reservation
                return
            charge = elapsed_seconds / dispatch_share
            self.prefill_virtual_runtime += charge - reservation
            self.last_prefill_seconds = elapsed_seconds
            if scheduled_tokens > 0:
                sample = elapsed_seconds / scheduled_tokens
                if self.prefill_seconds_per_token is None:
                    self.prefill_seconds_per_token = sample
                else:
                    alpha = self.PREFILL_COST_EWMA_ALPHA
                    self.prefill_seconds_per_token += alpha * (
                        sample - self.prefill_seconds_per_token
                    )
            if self.prefill_virtual_runtime - self.decode_virtual_runtime > charge:
                self.decode_virtual_runtime = self.prefill_virtual_runtime - charge
        else:
            reservation, dispatch_share = (
                self.decode_reservations.popleft()
                if self.decode_reservations
                else (0.0, self.effective_prefill_compute_share)
            )
            if elapsed_seconds <= 0.0:
                self.decode_virtual_runtime -= reservation
                return
            charge = elapsed_seconds / (1.0 - dispatch_share)
            self.decode_virtual_runtime += charge - reservation
            self.last_decode_seconds = elapsed_seconds
            self.last_decode_completed_at = completed_at
            if self.decode_virtual_runtime - self.prefill_virtual_runtime > charge:
                self.prefill_virtual_runtime = self.decode_virtual_runtime - charge

    def reset(self) -> None:
        """Discard credit when both classes are no longer runnable."""
        self.decode_virtual_runtime = 0.0
        self.prefill_virtual_runtime = 0.0
        self.contention_active = False
