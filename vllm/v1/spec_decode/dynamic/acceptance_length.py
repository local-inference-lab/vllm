# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class AcceptanceLengthUpdate:
    previous_num_spec_tokens: int
    num_spec_tokens: int
    mean_num_accepted_tokens: float
    mean_num_draft_tokens: float


class AcceptanceLengthController:
    """Adjust speculative depth from the observed accepted draft length."""

    def __init__(
        self,
        max_num_spec_tokens: int,
        observation_window: int,
        initial_num_spec_tokens: int | None = None,
    ) -> None:
        if max_num_spec_tokens <= 0:
            raise ValueError("max_num_spec_tokens must be greater than zero.")
        if observation_window <= 0:
            raise ValueError("observation_window must be greater than zero.")
        if initial_num_spec_tokens is None:
            initial_num_spec_tokens = max_num_spec_tokens
        if not 1 <= initial_num_spec_tokens <= max_num_spec_tokens:
            raise ValueError(
                "initial_num_spec_tokens must be between one and max_num_spec_tokens."
            )

        self.max_num_spec_tokens = max_num_spec_tokens
        self.observation_window = observation_window
        self.num_spec_tokens = initial_num_spec_tokens

        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0

    def observe_batch(
        self,
        *,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> AcceptanceLengthUpdate | None:
        """Observe one scheduler step and occasionally update the depth."""
        if num_drafts < 0 or num_draft_tokens < 0 or num_accepted_tokens < 0:
            raise ValueError("Speculative decoding counts must be non-negative.")
        if num_accepted_tokens > num_draft_tokens:
            raise ValueError("num_accepted_tokens must not exceed num_draft_tokens.")
        if num_drafts == 0:
            if num_draft_tokens or num_accepted_tokens:
                raise ValueError("Token counts require at least one draft.")
            return None

        self._num_observation_steps += 1
        self._num_drafts += num_drafts
        self._num_draft_tokens += num_draft_tokens
        self._num_accepted_tokens += num_accepted_tokens
        if self._num_observation_steps < self.observation_window:
            return None

        mean_num_accepted_tokens = self._num_accepted_tokens / self._num_drafts
        mean_num_draft_tokens = self._num_draft_tokens / self._num_drafts
        target_num_spec_tokens = min(
            self.max_num_spec_tokens,
            max(1, floor(mean_num_accepted_tokens + 1.5)),
        )

        previous_num_spec_tokens = self.num_spec_tokens
        if target_num_spec_tokens < self.num_spec_tokens:
            self.num_spec_tokens = target_num_spec_tokens
        elif target_num_spec_tokens > self.num_spec_tokens:
            self.num_spec_tokens += 1

        self._reset_window()
        return AcceptanceLengthUpdate(
            previous_num_spec_tokens=previous_num_spec_tokens,
            num_spec_tokens=self.num_spec_tokens,
            mean_num_accepted_tokens=mean_num_accepted_tokens,
            mean_num_draft_tokens=mean_num_draft_tokens,
        )

    def _reset_window(self) -> None:
        self._num_observation_steps = 0
        self._num_drafts = 0
        self._num_draft_tokens = 0
        self._num_accepted_tokens = 0


class BatchSizeAcceptanceLengthController:
    """Keep independent acceptance controllers for batch-size schedule bands."""

    def __init__(
        self,
        max_num_spec_tokens: int,
        observation_window: int,
        initial_num_spec_tokens: int | None = None,
        num_spec_tokens_by_batch_size: Sequence[int] | None = None,
    ) -> None:
        self.max_num_spec_tokens = max_num_spec_tokens
        self.observation_window = observation_window
        self.initial_num_spec_tokens = initial_num_spec_tokens
        self._global_controller: AcceptanceLengthController | None = None
        self._controllers_by_batch_size: (
            list[AcceptanceLengthController | None] | None
        ) = None

        if num_spec_tokens_by_batch_size is None:
            self._global_controller = self._make_controller(max_num_spec_tokens)
            return

        if len(num_spec_tokens_by_batch_size) < 2:
            raise ValueError(
                "num_spec_tokens_by_batch_size must contain a batch-size entry."
            )
        if num_spec_tokens_by_batch_size[0] != 0:
            raise ValueError("Batch-size lookup index zero must be unused.")

        controllers: list[AcceptanceLengthController | None] = [None]
        previous_cap: int | None = None
        controller: AcceptanceLengthController | None = None
        for cap in num_spec_tokens_by_batch_size[1:]:
            if not 0 <= cap <= max_num_spec_tokens:
                raise ValueError(
                    "Batch-size speculative-token caps must be between zero and "
                    "max_num_spec_tokens."
                )
            if cap != previous_cap:
                controller = self._make_controller(cap) if cap else None
                previous_cap = cap
            controllers.append(controller)
        self._controllers_by_batch_size = controllers

    def _make_controller(self, max_num_spec_tokens: int) -> AcceptanceLengthController:
        initial_num_spec_tokens = self.initial_num_spec_tokens
        if initial_num_spec_tokens is not None:
            initial_num_spec_tokens = min(initial_num_spec_tokens, max_num_spec_tokens)
        return AcceptanceLengthController(
            max_num_spec_tokens=max_num_spec_tokens,
            observation_window=self.observation_window,
            initial_num_spec_tokens=initial_num_spec_tokens,
        )

    def _controller_for_batch_size(
        self, batch_size: int
    ) -> AcceptanceLengthController | None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self._controllers_by_batch_size is None:
            return self._global_controller
        if batch_size >= len(self._controllers_by_batch_size):
            raise ValueError(
                f"batch_size {batch_size} exceeds the configured maximum "
                f"{len(self._controllers_by_batch_size) - 1}."
            )
        return self._controllers_by_batch_size[batch_size]

    def num_spec_tokens_for_batch_size(self, batch_size: int) -> int:
        controller = self._controller_for_batch_size(batch_size)
        return controller.num_spec_tokens if controller is not None else 0

    def observe_batch(
        self,
        *,
        batch_size: int,
        num_drafts: int,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> AcceptanceLengthUpdate | None:
        controller = self._controller_for_batch_size(batch_size)
        if controller is None:
            return None
        return controller.observe_batch(
            num_drafts=num_drafts,
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens,
        )
