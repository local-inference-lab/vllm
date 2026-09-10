# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class MicroSlicingSettings:
    """Runtime-tunable parameters for mixed decode/prefill micro-slicing."""

    max_num_prefill_tokens_per_step: int
    max_num_partial_prefills: int = 0
    decode_prefill_min_decode_steps: int = 0
    decode_prefill_max_wait_ms: int = 0


@dataclass
class MicroSlicingController:
    """Bound local prefill work while preserving decode progress.

    The controller owns policy state only. Request admission, KV allocation,
    and external-cache transfers remain scheduler responsibilities.
    """

    settings: MicroSlicingSettings
    quantum: int = 1
    steps_since_prefill: int = 0
    rotation: int = 0
    scheduled_prefill_tokens: int = 0
    decode_only_steps: int = 0
    fairness_bypasses: int = 0

    def __post_init__(self) -> None:
        self._validate(self.settings)

    def update(self, settings: MicroSlicingSettings) -> None:
        """Replace settings and discard policy history at an idle boundary."""
        self._validate(settings)
        self.settings = settings
        self.steps_since_prefill = 0
        self.rotation = 0

    def should_defer_prefill(
        self,
        *,
        has_eligible_decode: bool,
        has_pending_prefill: bool,
        oldest_waiter_age_ms: float,
    ) -> tuple[bool, bool]:
        """Return ``(defer, deadline_bypass_candidate)`` for this step."""
        min_decode_steps = self.settings.decode_prefill_min_decode_steps
        burst_would_defer = (
            min_decode_steps > 0
            and has_eligible_decode
            and has_pending_prefill
            and self.steps_since_prefill < min_decode_steps
        )
        deadline_bypass = (
            burst_would_defer
            and self.settings.decode_prefill_max_wait_ms > 0
            and oldest_waiter_age_ms >= self.settings.decode_prefill_max_wait_ms
        )
        return burst_would_defer and not deadline_bypass, deadline_bypass

    def select_running_limits(
        self, request_ids: list[str], available_tokens: int
    ) -> dict[str, int]:
        """Distribute the remaining mixed-step budget in rotating quanta."""
        num_quanta = available_tokens // self.quantum
        if num_quanta <= 0 or not request_ids:
            return {}

        start = self.rotation % len(request_ids)
        base_quanta, remainder = divmod(num_quanta, len(request_ids))
        limits = {
            request_id: base_quanta * self.quantum
            for request_id in request_ids
            if base_quanta > 0
        }
        for offset in range(remainder):
            request_id = request_ids[(start + offset) % len(request_ids)]
            limits[request_id] = limits.get(request_id, 0) + self.quantum

        # Rotate even when every request received the same number of quanta so
        # a later, smaller budget does not always begin with the same request.
        advance = remainder if remainder else 1
        self.rotation = (start + advance) % len(request_ids)
        return limits

    def observe_step(
        self,
        *,
        has_eligible_decode: bool,
        had_pending_prefill: bool,
        scheduled_decode_tokens: int,
        scheduled_prefill_tokens: int,
        deadline_bypass_candidate: bool,
    ) -> None:
        """Advance burst history and interval counters from actual work."""
        if scheduled_prefill_tokens > 0:
            self.steps_since_prefill = 0
            self.scheduled_prefill_tokens += scheduled_prefill_tokens
            if deadline_bypass_candidate:
                self.fairness_bypasses += 1
            return

        if has_eligible_decode and scheduled_decode_tokens > 0:
            # Decode service before a waiter arrives still satisfies the future
            # minimum-decode interval. Saturation keeps the counter bounded.
            min_decode_steps = self.settings.decode_prefill_min_decode_steps
            if min_decode_steps > 0:
                self.steps_since_prefill = min(
                    self.steps_since_prefill + 1, min_decode_steps
                )
            if had_pending_prefill:
                self.decode_only_steps += 1
            return

        if not has_eligible_decode:
            self.steps_since_prefill = 0

    def drain_stats(self, *, active_partial_prefills: int) -> dict[str, int]:
        stats = {
            "scheduled_prefill_tokens": self.scheduled_prefill_tokens,
            "active_partial_prefills": active_partial_prefills,
            "decode_only_steps": self.decode_only_steps,
            "fairness_bypasses": self.fairness_bypasses,
        }
        self.scheduled_prefill_tokens = 0
        self.decode_only_steps = 0
        self.fairness_bypasses = 0
        return stats

    def _validate(self, settings: MicroSlicingSettings) -> None:
        if settings.max_num_prefill_tokens_per_step <= 0:
            raise ValueError("max_num_prefill_tokens_per_step must be positive")
        if settings.max_num_prefill_tokens_per_step % self.quantum != 0:
            raise ValueError(
                "max_num_prefill_tokens_per_step must be a multiple of "
                f"the scheduler prefill quantum ({self.quantum})"
            )
        if settings.max_num_partial_prefills < 0:
            raise ValueError("max_num_partial_prefills cannot be negative")
        if settings.decode_prefill_min_decode_steps < 0:
            raise ValueError("decode_prefill_min_decode_steps cannot be negative")
        if settings.decode_prefill_max_wait_ms < 0:
            raise ValueError("decode_prefill_max_wait_ms cannot be negative")
        if (
            settings.decode_prefill_max_wait_ms > 0
            and settings.decode_prefill_min_decode_steps == 0
        ):
            raise ValueError(
                "decode_prefill_max_wait_ms requires decode_prefill_min_decode_steps"
            )
