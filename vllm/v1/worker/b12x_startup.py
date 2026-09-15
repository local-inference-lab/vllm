# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cooperative control for b12x preparation before ordinary model warmup.

One coordinator drives one worker's preparation batches through
complete-world rounds: each ``advance`` performs one bounded local step and
one all-gather on a store-backed control channel isolated from model
collectives. Ranks exchange collective readiness and the winners of their
disjoint candidate shards; tuning keys are scoped by the sharding rank set so
identical declarations on different pipeline stages never collide.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext


_CONTROL_GROUPS: dict[tuple[int, int], object] = {}


def _scoped_key(key: str, ranks: tuple[int, ...]) -> str:
    return f"{','.join(str(rank) for rank in ranks)}|{key}"


def _unscoped_key(key: str) -> str:
    return key.split("|", 1)[1]


class B12xPreparationCoordinator:
    """Advance one worker's preparation batches through world-coordinated rounds."""

    def __init__(
        self,
        session,
        batches,
        *,
        global_rank: int,
        world_group,
        process_local_only: bool = False,
    ) -> None:
        if type(global_rank) is not int or global_rank < 0:
            raise ValueError("global_rank must be a nonnegative integer")
        self.session = session
        self.global_rank = global_rank
        self.world_group = world_group
        self.process_local_only = process_local_only
        self.world_ranks = (
            (global_rank,) if process_local_only else _world_ranks(world_group)
        )
        if global_rank not in self.world_ranks:
            raise ValueError("global rank is not in preparation control domain")

        self._timing = None
        if os.environ.get("B12X_PREPARATION_TRACE_DIR"):
            from b12x.preparation._timing import PreparationTiming

            self._timing = PreparationTiming("coordinator", rank=global_rank)
        self._last_advance_end = None
        self._round = 0
        self._authorized_key: str | None = None
        self._authorized_tuning = None
        self._stop = False
        self._error: dict[str, object] | None = None
        self._last_progress = None
        self._cleanup_complete = False
        self._closed = False

        self._batches = [
            (tuple(requests), bool(autotune))
            for requests, autotune in batches
            if requests
        ]
        self._native = bool(self._batches)
        self._native_reason = "native" if self._native else "no_units"
        self._job = None
        self._local_done = not self._native
        self._global_done = process_local_only and self._local_done
        if self._native:
            if session is None:
                raise RuntimeError("native preparation requests require a session")
            self._begin_next_batch()

    def _begin_next_batch(self) -> None:
        requests, autotune = self._batches.pop(0)
        self._job = self.session.begin(requests, autotune=autotune)

    def status(self) -> dict[str, object]:
        """Return serializable state for the initial RPC response."""
        return self._outcome()

    def advance(self, *, cancel_tuning: bool = False) -> dict[str, object]:
        if self._timing is None:
            return self._advance(cancel_tuning=cancel_tuning)
        started = time.perf_counter()
        if self._last_advance_end is not None:
            self._timing.add("between_advances", started - self._last_advance_end)
        try:
            return self._advance(cancel_tuning=cancel_tuning)
        finally:
            self._timing.add("advance", time.perf_counter() - started)
            self._timing.record("progress", periodic=not self._global_done, round=self._round)
            self._last_advance_end = time.perf_counter()

    def _advance(self, *, cancel_tuning: bool = False) -> dict[str, object]:
        """Perform one bounded local step and one complete-world exchange."""
        if self._closed:
            return self._outcome()
        self._stop |= bool(cancel_tuning)
        try:
            with (self._timing.span("local") if self._timing else nullcontext()):
                self._advance_local()
        except BaseException as error:
            self._record_error(error)
            self._stop = True
            self._safe_close()

        if self.process_local_only:
            self._global_done = self._local_done or self._error is not None
            if self._global_done:
                self._safe_close()
                self._closed = True
            self._round += 1
            return self._outcome()

        with (self._timing.span("control_exchange") if self._timing else nullcontext()):
            gathered = _all_gather(self.world_group, self._payload())
        self._validate_domain(gathered)
        errors = [entry["error"] for entry in gathered if entry["error"]]
        if errors:
            self._error = min(errors, key=lambda item: int(item["rank"]))
            self._stop = True
            self._authorized_key = None
            self._safe_close()
        else:
            self._stop |= any(bool(entry["stop"]) for entry in gathered)
            authorization = _authorize_ready(gathered, self.world_ranks)
            self._authorized_key = (
                authorization[0]
                if authorization is not None
                and self.global_rank in authorization[1]
                else None
            )
            tuning = _authorize_tuning(gathered, self.world_ranks)
            if tuning is not None and self.global_rank in tuning[1]:
                from b12x.preparation import TuningRequirement

                key, ranks, assignment, latency_us, candidate_index = tuning
                self._authorized_tuning = TuningRequirement(
                    _unscoped_key(key), ranks, assignment, latency_us, candidate_index
                )
            else:
                self._authorized_tuning = None

        self._global_done = all(bool(entry["local_done"]) for entry in gathered)
        if self._error is not None:
            self._global_done = self._global_done and all(
                bool(entry["cleanup_complete"]) for entry in gathered
            )
        if self._global_done:
            self._safe_close()
            self._closed = True
        self._round += 1
        return self._outcome()

    def abort(self) -> dict[str, object]:
        """Stop optional work and close an active local preparation job."""
        self._stop = True
        self._authorized_key = None
        self._safe_close()
        self._local_done = True
        self._global_done = self.process_local_only
        if self._global_done:
            self._closed = True
        return self._outcome()

    def _advance_local(self) -> None:
        if self._local_done or self._error is not None:
            return
        job = self._job
        if job is None:
            raise RuntimeError("active preparation has no job")
        if self._stop:
            job.session.cancel_tuning()
        progress = job.advance(
            collective_key=self._authorized_key,
            tuning=self._authorized_tuning,
        )
        self._authorized_key = None
        self._authorized_tuning = None
        self._last_progress = progress
        if progress.pending_compilation:
            pool = job.session._pool
            if pool is not None:
                with (self._timing.span("compiler_wait") if self._timing else nullcontext()):
                    pool.wait_for_progress(timeout=0.05)
        if not progress.done:
            return

        job.result().close()
        self._job = None
        if self._batches:
            self._begin_next_batch()
        else:
            self._local_done = True

    def _ready(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        if self._last_progress is None:
            return ()
        return tuple(
            (item.key, item.ranks)
            for item in self._last_progress.ready_collectives
        )

    def _ready_tuning(self) -> tuple[tuple[object, ...], ...]:
        if self._last_progress is None:
            return ()
        return tuple(
            (
                _scoped_key(item.key, item.ranks),
                item.ranks,
                None if item.assignment is None else item.assignment.to_dict(),
                item.latency_us,
                item.candidate_index,
            )
            for item in getattr(self._last_progress, "ready_tuning", ())
        )

    def _payload(self) -> dict[str, object]:
        return {
            "round": self._round,
            "global_rank": self.global_rank,
            "world_ranks": self.world_ranks,
            "native": self._native,
            "native_reason": self._native_reason,
            "stop": self._stop,
            "ready": self._ready(),
            "tuning": self._ready_tuning(),
            "local_done": self._local_done,
            "error": self._error,
            "cleanup_complete": self._cleanup_complete,
        }

    def _validate_domain(self, gathered: list[dict[str, object]]) -> None:
        if len(gathered) != len(self.world_ranks):
            raise RuntimeError(
                "preparation control exchange did not include complete world domain"
            )
        ranks = tuple(sorted(int(entry["global_rank"]) for entry in gathered))
        if ranks != self.world_ranks or len(set(ranks)) != len(ranks):
            raise RuntimeError(
                "preparation control exchange has inconsistent global ranks"
            )
        for entry in gathered:
            if entry["world_ranks"] != self.world_ranks:
                raise RuntimeError(
                    "preparation control exchange has inconsistent world domain"
                )
            if entry["round"] != self._round:
                raise RuntimeError(
                    "preparation control exchange has inconsistent round"
                )
        native = {
            int(entry["global_rank"]): bool(entry.get("native"))
            for entry in gathered
        }
        if any(native.values()) and not all(native.values()):
            silent = sorted(rank for rank, value in native.items() if not value)
            reasons = sorted(
                {
                    str(entry.get("native_reason"))
                    for entry in gathered
                    if not bool(entry.get("native"))
                }
            )
            raise RuntimeError(
                "preparation world is asymmetrically native: ranks "
                f"{sorted(rank for rank, value in native.items() if value)} "
                f"declared native but ranks {silent} did not "
                f"(reason={reasons}). The native ranks would advance "
                "single-sided and never authorize collectives; refusing to "
                "prepare. Check the non-native ranks' unit collection: "
                "provider gates, b12x_native_supported, or module imports."
            )

    def _record_error(self, error: BaseException) -> None:
        self._error = {
            "rank": self.global_rank,
            "type": type(error).__name__,
            "message": str(error),
        }

    def _safe_close(self) -> None:
        if self._cleanup_complete:
            return
        primary = None
        if self._job is not None:
            try:
                self._job.close()
            except BaseException as error:
                primary = error
            self._job = None
        self._batches.clear()
        self._cleanup_complete = True
        self._local_done = True
        if primary is not None and self._error is None:
            self._record_error(primary)

    def _outcome(self) -> dict[str, object]:
        return {
            "native": self._native,
            "round": self._round,
            "global_rank": self.global_rank,
            "done": self._global_done,
            "progress": self._last_progress,
            "error": self._error,
            "cleanup_complete": self._cleanup_complete,
        }


def _world_ranks(world_group) -> tuple[int, ...]:
    if world_group is None:
        return (0,)
    ranks = getattr(world_group, "ranks", None)
    if ranks is not None:
        ranks = tuple(ranks)
        if (
            ranks
            and all(type(rank) is int and rank >= 0 for rank in ranks)
            and len(set(ranks)) == len(ranks)
        ):
            return tuple(sorted(ranks))
        raise ValueError(
            "preparation control domain ranks must be unique nonnegative integers"
        )
    size = getattr(world_group, "world_size", None)
    if type(size) is not int:
        cpu_group = getattr(world_group, "cpu_group", None)
        size = None if cpu_group is None else cpu_group.size()
    if type(size) is not int or size <= 0:
        raise ValueError("preparation control domain must expose global ranks")
    return tuple(range(size))


def _all_gather(
    world_group, payload: dict[str, object]
) -> list[dict[str, object]]:
    if world_group is None:
        return [payload]
    tcp_group = getattr(world_group, "tcp_store_group", None)
    if tcp_group is not None:
        return list(tcp_group.all_gather_obj(payload))

    control_group = _get_control_group(world_group)
    return list(control_group.all_gather_obj(payload))


def _get_control_group(world_group):
    """Return a store-backed channel isolated from model collectives."""
    cpu_group = getattr(world_group, "cpu_group", None)
    get_store = getattr(cpu_group, "get_group_store", None)
    rank = getattr(world_group, "rank_in_group", None)
    world_size = getattr(world_group, "world_size", None)
    if (
        get_store is None
        or type(rank) is not int
        or type(world_size) is not int
        or rank < 0
        or rank >= world_size
    ):
        raise RuntimeError(
            "preparation control requires the world CPU group's metadata store"
        )

    key = (id(cpu_group), rank)
    control_group = _CONTROL_GROUPS.get(key)
    if control_group is None:
        import torch.distributed as dist

        from vllm.distributed.utils import StatelessProcessGroup

        control_group = StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=dist.PrefixStore(
                "b12x_preparation_control_v1", get_store()
            ),
        )
        _CONTROL_GROUPS[key] = control_group
    elif control_group.world_size != world_size:
        raise RuntimeError(
            "preparation control world size changed for an active CPU group"
        )
    return control_group


def _authorize_ready(
    gathered: list[dict[str, object]], world_ranks: tuple[int, ...]
) -> tuple[str, tuple[int, ...]] | None:
    ready_by_key: dict[str, set[int]] = {}
    participants_by_key: dict[str, tuple[int, ...]] = {}
    for entry in gathered:
        rank = int(entry["global_rank"])
        for key, ranks in entry["ready"]:
            ranks = tuple(ranks)
            if ranks != tuple(sorted(set(ranks))) or not set(ranks) <= set(
                world_ranks
            ):
                raise RuntimeError(
                    "preparation collective has an invalid participant set"
                )
            previous = participants_by_key.setdefault(key, ranks)
            if previous != ranks:
                raise RuntimeError(
                    "preparation collective participants disagree for one key"
                )
            ready_by_key.setdefault(key, set()).add(rank)
    choices = [
        key
        for key, ranks in participants_by_key.items()
        if ready_by_key.get(key) == set(ranks)
    ]
    if not choices:
        return None
    key = min(choices)
    return key, participants_by_key[key]


def _authorize_tuning(
    gathered: list[dict[str, object]], world_ranks: tuple[int, ...]
) -> tuple[object, ...] | None:
    contributions: dict[str, list[tuple[float, int, int, object]]] = {}
    ready_by_key: dict[str, set[int]] = {}
    participants_by_key: dict[str, tuple[int, ...]] = {}
    for entry in gathered:
        rank = int(entry["global_rank"])
        for key, ranks, assignment, latency_us, candidate_index in entry.get(
            "tuning", ()
        ):
            ranks = tuple(ranks)
            if (
                ranks != tuple(sorted(set(ranks)))
                or not set(ranks) <= set(world_ranks)
                or rank not in ranks
            ):
                raise RuntimeError("preparation tuning has an invalid rank set")
            previous = participants_by_key.setdefault(key, ranks)
            if previous != ranks:
                raise RuntimeError(
                    "preparation tuning participants disagree for one key"
                )
            ready_by_key.setdefault(key, set()).add(rank)
            if assignment is not None:
                contributions.setdefault(key, []).append(
                    (float(latency_us), int(candidate_index), rank, assignment)
                )
    choices = [
        key
        for key, ranks in participants_by_key.items()
        if ready_by_key.get(key) == set(ranks) and contributions.get(key)
    ]
    if not choices:
        return None
    key = min(choices)
    candidates = contributions[key]
    indices = [candidate[1] for candidate in candidates]
    if len(indices) != len(set(indices)):
        raise RuntimeError("preparation tuning shards overlap")
    latency_us, candidate_index, _, assignment = min(candidates)
    return (
        key,
        participants_by_key[key],
        assignment,
        latency_us,
        candidate_index,
    )
