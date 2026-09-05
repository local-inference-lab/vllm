# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import chain

from vllm.config.scheduler import (
    DecodeRefillTarget,
    MaxParallelPrefills,
    PrefillPolicy,
)
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request, RequestStatus

LocalPrefillPredicate = Callable[[Request], bool]
AUTO_MAX_PARALLEL_PREFILLS = 4


def resolve_max_parallel_prefills(
    configured: MaxParallelPrefills,
    *,
    max_num_seqs: int,
    max_num_scheduled_tokens: int,
    block_size: int,
) -> int:
    """Resolve a safe lane count from the active scheduler geometry."""
    requested = AUTO_MAX_PARALLEL_PREFILLS if configured == "auto" else configured
    budget_lanes = max(1, max_num_scheduled_tokens // block_size)
    return min(requested, max_num_seqs, budget_lanes)


def resolve_decode_refill_target(
    configured: DecodeRefillTarget,
    *,
    max_parallel_prefills: int,
) -> int:
    """Resolve the decode-aware refill threshold."""
    return max_parallel_prefills if configured == "auto" else configured


@dataclass
class PrefillInterleaveStep:
    """Mutable prefill-lane allocation for one scheduler step."""

    ordered_requests: list[Request]
    max_parallel_prefills: int
    request_lookup: dict[str, Request]
    is_local_prefill: LocalPrefillPredicate
    rank: dict[str, int] = field(init=False)
    selected_ids: set[str] = field(default_factory=set, init=False)
    unavailable_ids: set[str] = field(default_factory=set, init=False)
    _next_candidate: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.rank = {
            request.request_id: rank
            for rank, request in enumerate(self.ordered_requests)
        }
        self._replenish()

    def is_selected(self, request_id: str) -> bool:
        return request_id in self.selected_ids

    def release(self, request_id: str) -> None:
        """Return a lane that turned out not to require local compute."""
        if request_id in self.selected_ids:
            self.selected_ids.remove(request_id)
            self._replenish()

    def mark_unavailable(self, request_id: str) -> None:
        """Replace a lane whose request cannot run in this step."""
        self.unavailable_ids.add(request_id)
        self.release(request_id)

    def token_budget(
        self, total_token_budget: int, scheduled_prefill_ids: set[str]
    ) -> int:
        """Return a work-conserving share of the remaining token budget."""
        fanout = max(
            sum(
                request_id not in scheduled_prefill_ids
                and request_id not in self.unavailable_ids
                and (request := self.request_lookup.get(request_id)) is not None
                and self.is_local_prefill(request)
                for request_id in self.selected_ids
            ),
            1,
        )
        return (total_token_budget + fanout - 1) // fanout

    def select_waiting_request(
        self, queues: Iterable[RequestQueue]
    ) -> tuple[RequestQueue, Request] | None:
        """Select the highest-ranked waiting request with a prefill lane."""
        selected_requests: list[tuple[int, RequestQueue, Request]] = []
        for queue in queues:
            for request in queue:
                rank = self.rank.get(request.request_id)
                if (
                    rank is not None
                    and request.request_id in self.selected_ids
                    and request.request_id not in self.unavailable_ids
                ):
                    selected_requests.append((rank, queue, request))
        if not selected_requests:
            return None
        _, queue, request = min(selected_requests, key=lambda item: item[0])
        return queue, request

    def _replenish(self) -> None:
        while len(
            self.selected_ids
        ) < self.max_parallel_prefills and self._next_candidate < len(
            self.ordered_requests
        ):
            request = self.ordered_requests[self._next_candidate]
            self._next_candidate += 1
            if request.request_id in self.unavailable_ids:
                continue
            current_request = self.request_lookup.get(request.request_id)
            if current_request is None or not self.is_local_prefill(current_request):
                continue
            self.selected_ids.add(request.request_id)


class PrefillInterleaveController:
    """Rotate local prefill service fairly across scheduler steps."""

    def __init__(self) -> None:
        self.last_scheduled_step: dict[str, int] = {}

    def begin_step(
        self,
        *,
        running: Iterable[Request],
        waiting: Iterable[Request],
        skipped_waiting: Iterable[Request],
        request_lookup: dict[str, Request],
        is_local_prefill: LocalPrefillPredicate,
        max_parallel_prefills: int,
        policy: PrefillPolicy,
        num_runnable_decodes: int,
        decode_refill_target: int,
        current_step: int,
        respect_priority: bool,
    ) -> PrefillInterleaveStep | None:
        """Create fair step-local prefill lanes, or use legacy scheduling."""
        if max_parallel_prefills <= 1:
            return None

        requests: list[Request] = []
        seen_request_ids: set[str] = set()
        for request in chain(running, waiting, skipped_waiting):
            if request.request_id in seen_request_ids:
                continue
            if not is_local_prefill(request):
                continue
            if (
                request.status == RequestStatus.RUNNING
                and current_step < request.next_decode_eligible_step
            ):
                continue
            seen_request_ids.add(request.request_id)
            requests.append(request)

        requests.sort(
            key=lambda request: (
                request.priority if respect_priority else 0,
                self.last_scheduled_step.get(request.request_id, -1),
                request.arrival_time,
                request.request_id,
            )
        )
        if not requests:
            return None
        if policy == "decode-aware" and num_runnable_decodes < decode_refill_target:
            highest_priority = requests[0].priority if respect_priority else 0
            eligible = (
                request
                for request in requests
                if not respect_priority or request.priority == highest_priority
            )
            nearest_decode = min(
                eligible,
                key=lambda request: (
                    max(
                        request.num_tokens - 1 - request.num_computed_tokens,
                        0,
                    ),
                    request.arrival_time,
                    request.request_id,
                ),
            )
            requests.remove(nearest_decode)
            requests.insert(0, nearest_decode)
        return PrefillInterleaveStep(
            requests,
            max_parallel_prefills,
            request_lookup,
            is_local_prefill,
        )

    def record_scheduled(
        self,
        request_ids: Iterable[str],
        current_step: int,
    ) -> None:
        for request_id in request_ids:
            self.last_scheduled_step[request_id] = current_step

    def forget(self, request_id: str) -> None:
        self.last_scheduled_step.pop(request_id, None)
