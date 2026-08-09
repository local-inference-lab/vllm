# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in runtime-path evidence for the fixed Fruit QSRT deployment."""

import contextlib
import json
import os
import stat
import tempfile
import threading
from pathlib import Path

import torch

_ENVIRONMENT_VARIABLE = "VLLM_KQUANT_RUNTIME_EVIDENCE"
_SCHEMA = "kquant_fruit_runtime_paths_v1"
_ORDINARY_LAYERS = range(3, 13)
_CAPTURE_SLOT_COUNT = 21


def _empty_evidence() -> dict[str, object]:
    layers: dict[str, object] = {}
    for layer in _ORDINARY_LAYERS:
        layers[str(layer)] = {
            "prefill": {"mode": "w4a16", "calls": 0},
            "decode": {
                "mode": "w4a8",
                "calls": 0,
                "part_count": 0,
                "capture_calls": 0,
                "replay_calls": 0,
            },
        }
    layers["13"] = {
        "mtp_decode": {
            "mode": "w4a8",
            "calls": 0,
            "part_count": 0,
            "capture_calls": 0,
            "replay_calls": 0,
        }
    }
    return {
        "schema": _SCHEMA,
        "version": 1,
        "layers": layers,
        "cudagraph": {
            "mode": "FULL_AND_PIECEWISE",
            "capture_count": 0,
            "replay_count": 0,
        },
        "speculative": {
            "method": "mtp",
            "num_speculative_tokens": 1,
            "draft_tokens": 0,
        },
    }


def _capture_slot(layer: int, mode: str, decode_only: bool) -> int | None:
    if 3 <= layer <= 12:
        if mode == "w4a16" and not decode_only:
            return layer - 3
        if mode == "w4a8" and decode_only:
            return 10 + layer - 3
    elif layer == 13 and mode == "w4a8" and decode_only:
        return 20
    return None


class RuntimeEvidence:
    """Fixed-shape, saturating evidence writer.

    Every counter records whether its path has been seen rather than an
    unbounded request total. Consequently the JSON can change at most once per
    evidence atom, regardless of server lifetime.
    """

    def __init__(self, path: str | os.PathLike[str] | None) -> None:
        self.path = Path(path) if path else None
        self.data = _empty_evidence() if self.path is not None else None
        self.lock = threading.Lock() if self.path is not None else None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def replay_saturated(self, observations: bytes | None) -> bool:
        """Return whether this replay can add runtime evidence."""
        if self.path is None:
            return True
        assert self.data is not None
        assert self.lock is not None
        with self.lock:
            cudagraph = self.data["cudagraph"]
            if cudagraph["replay_count"] == 0:
                return False
            if observations is None:
                return True
            layers = self.data["layers"]
            assert isinstance(layers, dict)
            for slot, part_count in enumerate(observations):
                if not part_count:
                    continue
                layer, mode, decode_only = _slot_observation(slot)
                if 3 <= layer <= 12 and mode == "w4a8" and decode_only:
                    observation = layers[str(layer)]["decode"]
                elif layer == 13 and mode == "w4a8" and decode_only:
                    observation = layers["13"]["mtp_decode"]
                else:
                    continue
                if observation["replay_calls"] == 0:
                    return False
            return True

    def _write(self) -> None:
        assert self.path is not None
        assert self.data is not None
        parent = self.path.parent
        parent_stat = parent.stat()
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise RuntimeError(f"runtime evidence parent is not a directory: {parent}")
        if self.path.is_symlink():
            raise RuntimeError(
                f"runtime evidence path must not be a symlink: {self.path}"
            )
        payload = (
            json.dumps(
                self.data,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{self.path.name}.",
        )
        try:
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                offset += os.write(descriptor, payload[offset:])
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary_name, self.path)
            directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)

    def _observe_layer(
        self,
        layer: int,
        mode: str,
        decode_only: bool,
        part_count: int,
        *,
        capture: bool = False,
        replay: bool = False,
    ) -> bool:
        assert self.data is not None
        layers = self.data["layers"]
        assert isinstance(layers, dict)
        changed = False
        if 3 <= layer <= 12 and mode == "w4a16" and not decode_only:
            observation = layers[str(layer)]["prefill"]
        elif 3 <= layer <= 12 and mode == "w4a8" and decode_only:
            observation = layers[str(layer)]["decode"]
        elif layer == 13 and mode == "w4a8" and decode_only:
            observation = layers["13"]["mtp_decode"]
        else:
            return False
        if observation["calls"] == 0:
            observation["calls"] = 1
            changed = True
        if mode == "w4a8" and observation["part_count"] == 0:
            observation["part_count"] = part_count
            changed = True
        if (
            capture
            and "capture_calls" in observation
            and observation["capture_calls"] == 0
        ):
            observation["capture_calls"] = 1
            changed = True
        if (
            replay
            and "replay_calls" in observation
            and observation["replay_calls"] == 0
        ):
            observation["replay_calls"] = 1
            changed = True
        if layer == 13 and replay:
            speculative = self.data["speculative"]
            if speculative["draft_tokens"] == 0:
                speculative["draft_tokens"] = 1
                changed = True
        return changed

    def observe_layer(
        self,
        layer: int,
        mode: str,
        decode_only: bool,
        part_count: int,
    ) -> None:
        if self.path is None:
            return
        assert self.lock is not None
        with self.lock:
            if self._observe_layer(layer, mode, decode_only, part_count):
                self._write()

    def observe_capture(self, observations: bytes) -> None:
        if self.path is None:
            return
        assert self.data is not None
        assert self.lock is not None
        with self.lock:
            cudagraph = self.data["cudagraph"]
            changed = False
            if cudagraph["capture_count"] == 0:
                cudagraph["capture_count"] = 1
                changed = True
            for slot, part_count in enumerate(observations):
                if part_count:
                    layer, mode, decode_only = _slot_observation(slot)
                    changed |= self._observe_layer(
                        layer,
                        mode,
                        decode_only,
                        part_count,
                        capture=True,
                    )
            if changed:
                self._write()

    def observe_replay(self, observations: bytes | None) -> None:
        if self.path is None:
            return
        assert self.data is not None
        assert self.lock is not None
        with self.lock:
            cudagraph = self.data["cudagraph"]
            changed = False
            if cudagraph["replay_count"] == 0:
                cudagraph["replay_count"] = 1
                changed = True
            if observations is not None:
                for slot, part_count in enumerate(observations):
                    if part_count:
                        layer, mode, decode_only = _slot_observation(slot)
                        changed |= self._observe_layer(
                            layer,
                            mode,
                            decode_only,
                            part_count,
                            replay=True,
                        )
            if changed:
                self._write()


def _slot_observation(slot: int) -> tuple[int, str, bool]:
    if slot < 10:
        return 3 + slot, "w4a16", False
    if slot < 20:
        return 3 + slot - 10, "w4a8", True
    if slot == 20:
        return 13, "w4a8", True
    raise ValueError(f"invalid runtime evidence capture slot: {slot}")


_runtime_evidence = RuntimeEvidence(os.getenv(_ENVIRONMENT_VARIABLE))
runtime_evidence_enabled = _runtime_evidence.enabled
_capture_observations = bytearray(_CAPTURE_SLOT_COUNT)
_capture_active = False


def record_layer_execution(
    layer: int | None,
    mode: str,
    decode_only: bool,
    part_count: int,
) -> None:
    """Record an eager call or mark a preallocated slot during capture."""
    if layer is None or not _runtime_evidence.enabled:
        return
    slot = _capture_slot(layer, mode, decode_only)
    if slot is None:
        return
    if _capture_active:
        _capture_observations[slot] = min(max(part_count, 1), 255)
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    _runtime_evidence.observe_layer(layer, mode, decode_only, part_count)


def begin_graph_capture() -> bool:
    """Prepare fixed storage before entering a CUDA capture context."""
    global _capture_active
    if not _runtime_evidence.enabled:
        return False
    for index in range(_CAPTURE_SLOT_COUNT):
        _capture_observations[index] = 0
    _capture_active = True
    return True


def finish_graph_capture(started: bool, succeeded: bool) -> bytes | None:
    """Publish a successful capture after leaving its CUDA context."""
    global _capture_active
    if not started:
        return None
    _capture_active = False
    if not succeeded:
        return None
    observations = bytes(_capture_observations)
    _runtime_evidence.observe_capture(observations)
    return observations


def record_graph_replay(observations: bytes | None) -> None:
    """Publish after a replay has completed on the host's current device."""
    if not _runtime_evidence.enabled or _runtime_evidence.replay_saturated(
        observations
    ):
        return
    torch.cuda.synchronize()
    _runtime_evidence.observe_replay(observations)
