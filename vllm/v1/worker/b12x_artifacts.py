# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded winner-artifact transfers over the preparation CPU store."""

from __future__ import annotations

import json
import struct
import threading
import time
from datetime import timedelta

from vllm.logger import init_logger

logger = init_logger(__name__)
_CHUNK_BYTES = 1024 * 1024
_RESPONSE_HEADER = struct.Struct("!Q?")


class WinnerArtifactExchange:
    def __init__(self, store, rank, ranks, *, cache=None, timeout=1.0):
        from b12x.preparation.artifacts import CuTeArtifactCache

        self.cache = CuTeArtifactCache() if cache is None else cache
        self.rank, self.ranks = rank, ranks
        # Each thread gets an independent TCP connection and timeout.
        self._client = store.clone()
        self._server = store.clone()
        self._client.set_timeout(timedelta(seconds=timeout))
        self._server.set_timeout(timedelta(seconds=timeout))
        self._timeout = timeout
        self._offered: frozenset[str] = frozenset()
        self._failed_peers: set[int] = set()
        self._sequence = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="b12x-artifacts", daemon=True
        )
        self.fetched = self.bytes_received = 0
        self._thread.start()

    def offer(self, keys):
        self._offered = self._offered.union(keys)

    @staticmethod
    def _slot(source, destination):
        return f"artifacts/{source}/{destination}"

    def _serve(self):
        try:
            while not self._stop.wait(0.005):
                for peer in self.ranks:
                    if peer == self.rank:
                        continue
                    slot = self._slot(self.rank, peer)
                    request = f"{slot}/request"
                    if not self._server.check([request]):
                        continue
                    sequence, key, suffix, offset = json.loads(
                        self._server.get(request)
                    )
                    self._server.delete_key(request)
                    data = None
                    if (
                        key in self._offered
                        and suffix in (".o", ".json")
                        and type(offset) is int
                        and offset >= 0
                        and self.cache.has(key)
                    ):
                        try:
                            with self.cache.path(key, suffix).open("rb") as stream:
                                stream.seek(offset)
                                data = stream.read(_CHUNK_BYTES)
                        except OSError:
                            pass
                    self._server.set(
                        f"{slot}/response",
                        _RESPONSE_HEADER.pack(sequence, data is not None)
                        + (data or b""),
                    )
        except Exception:
            if not self._stop.is_set():
                logger.warning("B12X artifact service stopped", exc_info=True)

    def _chunk(self, source, key, suffix, offset, deadline):
        slot = self._slot(source, self.rank)
        response = f"{slot}/response"
        self._sequence += 1
        self._client.set(
            f"{slot}/request",
            json.dumps((self._sequence, key, suffix, offset)),
        )
        while time.monotonic() < deadline and not self._stop.is_set():
            if self._client.check(["stop"]) or self._client.check(["failed"]):
                raise InterruptedError("preparation stopped")
            if self._client.check([response]):
                message = self._client.get(response)
                sequence, found = _RESPONSE_HEADER.unpack_from(message)
                self._client.delete_key(response)
                if sequence == self._sequence:
                    return message[_RESPONSE_HEADER.size :] if found else None
            self._stop.wait(0.005)
        raise TimeoutError("winner artifact transfer timed out")

    def fetch(self, source, keys):
        """Fetch missing winners; any unavailable artifact remains a local miss."""
        if source == self.rank or source in self._failed_peers:
            return
        for key in keys:
            if self.cache.has(key):
                continue
            try:
                # One deadline for the entire pair, including a slow peer that
                # keeps supplying partial chunks. A failed peer is not retried.
                deadline = time.monotonic() + self._timeout
                with self.cache.stage(key) as staged:
                    received = 0
                    for suffix in (".json", ".o"):
                        with (staged / (key + suffix)).open("wb") as stream:
                            while True:
                                data = self._chunk(
                                    source, key, suffix, stream.tell(), deadline
                                )
                                if data is None:
                                    raise FileNotFoundError(key)
                                if (
                                    not isinstance(data, bytes)
                                    or len(data) > _CHUNK_BYTES
                                ):
                                    raise ValueError("invalid artifact chunk")
                                stream.write(data)
                                received += len(data)
                                if len(data) < _CHUNK_BYTES:
                                    break
                    if self.cache.publish(key, staged):
                        self.fetched += 1
                        self.bytes_received += received
            except FileNotFoundError:
                continue
            except InterruptedError:
                return
            except Exception as error:
                self._failed_peers.add(source)
                logger.warning(
                    "B12X winner fetch from rank %d failed; compiling locally: %s",
                    source,
                    error,
                )
                return

    def close(self):
        self._stop.set()
        self._thread.join(timeout=self._timeout * 3 + 0.1)
        if self._thread.is_alive():
            logger.warning("B12X artifact service has not stopped")
            return
        try:
            for peer in self.ranks:
                if peer != self.rank:
                    slot = self._slot(self.rank, peer)
                    for suffix in ("request", "response"):
                        self._server.delete_key(f"{slot}/{suffix}")
        except Exception:
            logger.debug("B12X artifact slot cleanup failed", exc_info=True)
        if self.fetched:
            logger.info(
                "B12X fetched %d winner programs (%d bytes) on rank %d",
                self.fetched,
                self.bytes_received,
                self.rank,
            )
