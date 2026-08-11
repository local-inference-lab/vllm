# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FragmentResolver — Progressive Tensors fragment resolution (loader v2).

Resolves the payload of one routed expert at one bitrate K — the contiguous
per-expert byte range of a ``layer-{L:03d}.k{K}.safetensors`` segment file
(fq_repack layout, implementation/10 §1) — through the resolution order of
implementation/10 §2:

    local segment dirs  ->  manifest ``sources`` chain (HF ranged reads)
                        ->  (local encode from BF16: T3, not implemented)

Content addressing: every segment ships a per-expert sha256 map inside its
attestation line (``attestations/layer-{L:03d}.k{K}.jsonl``, fq-attestation/1
payload field ``expert_sha256``); the per-layer ``index-k{K}.json`` carries the
file-level sha256, ``body_offset`` and the per-expert ``[lo, hi)`` body
ranges. Every *fetched* fragment is verified against its expected sha before
use and cached content-addressed under ``VLLM_FQ_CACHE`` (default
``~/.cache/vllm/fq``); cache hits are re-hashed on read (they are small).
Local segment-dir reads are trusted by default (``VLLM_FQ_VERIFY=fetched``);
``VLLM_FQ_VERIFY=all`` extends verification to local reads, ``off`` disables
it (fetching without an attestation then becomes possible).

Operator knobs (this milestone):

* ``VLLM_FQ_SOURCES`` — ordered comma list of ``repo_id[@revision]`` source
  specs; ``VLLM_FQ_SOURCES_MODE`` = ``prepend`` (default) | ``replace`` |
  ``append`` relative to the manifest ``sources`` chain.  Local segment dirs
  always resolve first; each source keeps its own index / attestation fetch
  and cache.
* Trust filtering (10 §4): ``VLLM_FQ_TRUST_SIGNERS`` — comma list of hex
  ed25519 pubkeys (default: the manifest ``signer_pubkey``); when any signer
  is configured, a fragment is accepted from a source only if one of the
  source's attestation lines for that layer/K carries a valid signature by
  an allowed signer (countersignatures: ANY allowed line accepts) AND its
  predicate is in ``VLLM_FQ_TRUST_PREDICATES`` (default
  ``repack-of,encode-of,derived-from``).  With no signer configured the
  legacy sha-only behavior applies.  Integrity sha checks stay unconditional.
* ``VLLM_FQ_K_FALLBACK`` — comma-ordered substitute Ks tried per miss (e.g.
  ``3``: if K4 is unavailable/untrusted, load the K3 fragment instead).  The
  returned :class:`Fragment` records both ``k`` (actually loaded) and
  ``requested_k``; every substitution/miss is enqueued on the persisted
  lazy-encode queue (:mod:`.lazy_encode`).  Boot never blocks on encodes.
  Unset (or ``auto``) means :meth:`FragmentResolver.resolve_best` derives the
  ladder itself — the nearest *lower* Ks this deployment can actually supply,
  nearest first (``VLLM_FQ_K_FALLBACK_UP=1`` also allows upward substitution,
  off by default: a higher K costs memory and, on SM120, K5 does not serve as
  a mixed tier at all).  ``off``/``none`` disables substitution entirely.

Two resolution entry points, deliberately different about failure:

* :meth:`FragmentResolver.resolve` — strict. Uses only the *explicit*
  ``VLLM_FQ_K_FALLBACK`` ladder and raises
  :class:`FragmentUnavailableError` when nothing supplies the fragment. This
  is the auditing/tooling contract (and what the trust tests pin).
* :meth:`FragmentResolver.resolve_best` — **never raises**. Requested K, else
  the nearest available lower K (logged loudly, encode queued), else
  ``None`` so the caller can keep the incumbent tier. Every serving path
  (boot stream, live swap staging) goes through this one: a fragment that is
  missing at the required bitrate must never take an engine down.

Every :meth:`FragmentResolver.resolve` emits one structured decision line
(INFO on substitution/fallback, DEBUG on plain success, WARNING on failure)::

    FQ resolve L17/e204 K4: local(2 dirs) MISS; hf:repoA@ab12 REJECT
    predicate=repack-of not-trusted; hf:repoB@cd34 REJECT sha-mismatch;
    FALLBACK K3 hf:repoC@ee55 ACCEPT (encode queued #3)

and accumulates per-reason counters in ``resolver.stats``.

This module is deliberately stdlib-only at import time; torch is imported
lazily inside :meth:`FragmentResolver.expert_tensors` so the resolver core
stays testable on CPU without a built vllm.  Signature verification imports
PyNaCl or ``cryptography`` lazily, only when trust filtering is active.
"""
from __future__ import annotations

import base64
import contextlib
from collections import Counter
import hashlib
import json
import logging
import mmap
import os
import re
import struct
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

FQ_CACHE_ENV = "VLLM_FQ_CACHE"
FQ_VERIFY_ENV = "VLLM_FQ_VERIFY"
FQ_SOURCES_ENV = "VLLM_FQ_SOURCES"
FQ_SOURCES_MODE_ENV = "VLLM_FQ_SOURCES_MODE"
FQ_LOCAL_SEGMENTS_ENV = "VLLM_FQ_LOCAL_SEGMENTS"
FQ_TRUST_PREDICATES_ENV = "VLLM_FQ_TRUST_PREDICATES"
FQ_TRUST_SIGNERS_ENV = "VLLM_FQ_TRUST_SIGNERS"
FQ_K_FALLBACK_ENV = "VLLM_FQ_K_FALLBACK"
FQ_K_FALLBACK_UP_ENV = "VLLM_FQ_K_FALLBACK_UP"

DEFAULT_CACHE_DIR = "~/.cache/vllm/fq"
DEFAULT_TRUST_PREDICATES = ("repack-of", "encode-of", "derived-from")
SOURCES_MODES = ("prepend", "replace", "append")
# Ks to consider when nothing in the deployment advertises a K set at all.
DEFAULT_K_UNIVERSE = (2, 3, 4, 5, 6)
_INDEX_K_RE = re.compile(r"^index-k(\d+)\.json$")
_TRUE = ("1", "true", "yes", "on")

EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(\w+_proj)\.rank(\d+)\.(\w+)$"
)

_ST_TO_TORCH_NAME = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


class OfflineError(OSError):
    """Raised instead of a network call when HF_HUB_OFFLINE is set.

    Subclasses OSError so the existing transient-error handling treats it as
    a source MISS rather than a crash — offline is a configuration, not a
    fault."""


class FragmentUnavailableError(RuntimeError):
    """No local dir, cache entry, trusted source, or fallback K could supply
    the fragment; the miss is enqueued for lazy encode (:mod:`.lazy_encode`)
    before this is raised."""


class FragmentVerificationError(RuntimeError):
    """Fragment bytes did not match their expected sha256."""


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """Return (header_dict, body_offset) of a safetensors file."""
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen


def parse_segment_header_bytes(raw: bytes) -> tuple[dict, int]:
    """Parse header bytes fetched as ``file[0:body_offset)``."""
    hlen = struct.unpack("<Q", raw[:8])[0]
    if 8 + hlen != len(raw):
        raise ValueError(
            f"segment header length mismatch: 8+{hlen} != {len(raw)} bytes"
        )
    return json.loads(raw[8 : 8 + hlen]), 8 + hlen


def _attestation_expert_shas(text: str) -> dict[str, str]:
    """Extract the per-expert sha map from a fq-attestation/1 jsonl blob.

    Signature verification is a trust-policy concern (10 §4, VLLM_FQ_TRUST)
    and lands with the trust-list milestone; content addressing only needs
    the payload."""
    for _env, _raw, payload in _attestation_lines(text):
        shas = payload.get("expert_sha256")
        if isinstance(shas, dict):
            return shas
    return {}


def _attestation_lines(text: str):
    """Yield ``(envelope, raw_payload_bytes, payload)`` per parseable line."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
            raw = base64.b64decode(envelope["payload"])
            payload = json.loads(raw)
        except (ValueError, KeyError, TypeError):
            continue
        yield envelope, raw, payload


def _verify_ed25519(pubkey_hex: str, message: bytes, signature: bytes) -> bool:
    """Verify an ed25519 signature; PyNaCl if present, else cryptography."""
    try:
        key = bytes.fromhex(pubkey_hex)
    except ValueError:
        return False
    if len(key) != 32:
        return False
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError:
        pass
    else:
        try:
            VerifyKey(key).verify(message, signature)
            return True
        except BadSignatureError:
            return False
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise RuntimeError(
            "attestation trust filtering needs PyNaCl or cryptography for "
            "ed25519 verification"
        ) from exc
    try:
        Ed25519PublicKey.from_public_bytes(key).verify(signature, message)
        return True
    except (InvalidSignature, ValueError):
        return False


def _safe_source_key(name: str) -> str:
    """Filesystem-safe per-source cache subdir name."""
    return re.sub(r"[^A-Za-z0-9._@-]+", "_", name)


def _load_lazy_encode():
    """Import .lazy_encode (package or standalone-file fallback)."""
    try:
        from vllm.model_executor.layers.quantization.exl3_fungible import (
            lazy_encode,
        )

        return lazy_encode
    except ImportError:
        import importlib.util as _ilu
        import sys as _sys

        name = "fq_lazy_encode_standalone"
        if name in _sys.modules:
            return _sys.modules[name]
        spec = _ilu.spec_from_file_location(
            name, Path(__file__).resolve().parent / "lazy_encode.py"
        )
        mod = _ilu.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod


@dataclass(frozen=True)
class FragmentTensor:
    """One tensor inside a fragment payload (offsets relative to payload)."""

    name: str
    dtype: str  # safetensors dtype string
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass
class Fragment:
    """A resolved per-expert payload plus its tensor table.

    ``k`` is the ACTUALLY loaded bitrate; when the fallback ladder
    substituted a different K, ``requested_k`` records what the caller asked
    for and :attr:`substituted` is True — callers must thread ``k`` (not
    ``requested_k``) into tier maps / bits digests so metadata reflects
    reality."""

    layer: int
    expert: int
    k: int
    payload: Any  # bytes | memoryview (buffer protocol)
    tensors: list[FragmentTensor]
    origin: str  # "local" | "cache" | "fetched"
    sha256: str | None = None
    requested_k: int | None = None

    @property
    def substituted(self) -> bool:
        return self.requested_k is not None and self.requested_k != self.k


class SegmentHeaderMismatch(RuntimeError):
    """A segment header arrived truncated.

    Its own type because it is a TRANSPORT fault, not a bad artifact: the
    right response is to retry, not to reject the source. Raised before the
    bytes are parsed or cached, so a body cut mid-header (or an error page
    served with 200) cannot become a poisoned header cache entry.
    """


class _SegmentIndexEntry:
    """One layer's entry of an index-k{K}.json."""

    def __init__(self, entry: dict):
        self.file: str = entry["file"]
        self.sha256: str = entry["sha256"]
        self.body_offset: int = int(entry["body_offset"])
        self.experts: dict[str, list[int]] = entry["experts"]

    def expert_range(self, expert: int) -> tuple[int, int]:
        rng = self.experts.get(str(expert))
        if rng is None:
            raise KeyError(f"expert {expert} not in segment index for {self.file}")
        return int(rng[0]), int(rng[1])


def _expert_tables_from_header(
    header: dict, entry: _SegmentIndexEntry
) -> dict[int, list[FragmentTensor]]:
    """Group segment-header tensors per expert, offsets fragment-relative.

    One pass over the (large) header; validated against the index ranges."""
    tables: dict[int, list[FragmentTensor]] = {}
    for name, t in header.items():
        if name == "__metadata__":
            continue
        m = EXPERT_RE.match(name)
        if m is None:
            continue
        expert = int(m.group(2))
        lo, hi = entry.expert_range(expert)
        a, b = t["data_offsets"]
        if a < lo or b > hi:
            raise ValueError(
                f"segment tensor {name} at [{a},{b}) escapes indexed expert "
                f"range [{lo},{hi})"
            )
        tables.setdefault(expert, []).append(
            FragmentTensor(name, t["dtype"], tuple(t["shape"]), a - lo, b - lo)
        )
    for items in tables.values():
        items.sort(key=lambda ft: ft.start)
    return tables


class _LocalSegment:
    """One local layer-K segment file, mmapped, with its tensor tables."""

    def __init__(self, path: Path, entry: _SegmentIndexEntry):
        self.path = path
        self.entry = entry
        header, body_offset = read_safetensors_header(path)
        if body_offset != entry.body_offset:
            raise ValueError(
                f"{path}: body_offset {body_offset} != index {entry.body_offset}"
            )
        self.tables = _expert_tables_from_header(header, entry)
        self._f = open(path, "rb")
        # Never explicitly closed: fragments hand out zero-copy memoryviews
        # whose lifetime the GC ties to this mapping.
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

    def fragment(self, expert: int) -> tuple[memoryview, list[FragmentTensor]]:
        lo, hi = self.entry.expert_range(expert)
        view = memoryview(self._mm)[
            self.entry.body_offset + lo : self.entry.body_offset + hi
        ]
        return view, self.tables[expert]


class HfSource:
    """A Hugging Face repo source: whole-file JSON + ranged byte reads.

    Spec syntax: ``org/name``, ``hf:org/name`` or ``hf:org/name@revision``.
    Token from ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``."""

    def __init__(self, spec: str, *, timeout: float = 60.0):
        spec = spec.removeprefix("hf:")
        self.revision = "main"
        if "@" in spec:
            spec, self.revision = spec.rsplit("@", 1)
        self.repo_id = spec
        self.name = f"hf:{self.repo_id}@{self.revision}"
        self.timeout = timeout

    def _url(self, relpath: str) -> str:
        try:
            from huggingface_hub import hf_hub_url

            return hf_hub_url(self.repo_id, relpath, revision=self.revision)
        except ImportError:
            return (
                f"https://huggingface.co/{self.repo_id}/resolve/"
                f"{self.revision}/{relpath}"
            )

    @staticmethod
    def offline() -> bool:
        """True when the operator has asked for no network.

        Honours the ECOSYSTEM-STANDARD ``HF_HUB_OFFLINE`` (and vLLM's
        ``VLLM_NO_USAGE_STATS``-style truthiness) rather than inventing our
        own knob, because an operator who sets it expects it to bind
        everything that talks to the Hub. We do NOT go through
        ``huggingface_hub`` for payload reads — only ``hf_hub_url`` to build a
        URL, then raw urllib — so the library's own offline handling never
        applied to us. Without this check, ``HF_HUB_OFFLINE=1`` silently still
        hit the network, which is the worst kind of wrong for an air-gapped or
        reproducibility-audited run.

        ``FQ_ALLOW_NETWORK=1`` is the explicit override for the case where a
        caller wants Hub access despite the global flag.
        """
        if os.environ.get("FQ_ALLOW_NETWORK", "").strip().lower() in (
                "1", "true", "yes", "on"):
            return False
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            if os.environ.get(var, "").strip().lower() in (
                    "1", "true", "yes", "on"):
                return True
        return False

    def _open(self, relpath: str, headers: dict[str, str]):
        import urllib.request

        if self.offline():
            # Raise the same shape a 404 does so every caller already handles
            # it as "this source has nothing", falling through to local dirs,
            # the primed cache, and then the K ladder.
            raise OfflineError(
                f"{self.name}: HF_HUB_OFFLINE is set; refusing to fetch "
                f"{relpath}. Prime the cache, or set FQ_ALLOW_NETWORK=1.")
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if token:
            headers = {**headers, "Authorization": f"Bearer {token}"}
        req = urllib.request.Request(self._url(relpath), headers=headers)
        return urllib.request.urlopen(req, timeout=self.timeout)

    def read_json(self, relpath: str) -> dict | None:
        import urllib.error

        try:
            with self._open(relpath, {}) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    def read_text(self, relpath: str) -> str | None:
        import urllib.error

        try:
            with self._open(relpath, {}) as r:
                return r.read().decode()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise

    # A progressive boot issues one ranged read PER EXPERT — 19,200 of them for
    # GLM-5.2. At that volume a transient failure is not an edge case, it is a
    # certainty: one `IncompleteRead` on layer 3 expert 19 was enough to abort a
    # whole engine start. Retry the retryable, with backoff; leave 404 and
    # friends to the caller, which treats them as a genuine MISS.
    _RETRIES = 4
    _BACKOFF = 0.4

    # Whole-segment prefetch. A UNIFORM layer needs every expert out of the
    # same segment file, which as per-expert ranged reads is 256 HTTP requests
    # for one contiguous object -- 19,200 across GLM-5.2's 75 layers. Pulling
    # the object once and slicing locally is the same bytes in 1 request, and
    # it removes 255 independent chances to hit a transient failure.
    # Deliberately NOT automatic: a layer that needs a single K4 expert would
    # otherwise drag ~2.5 GB to use ~9 MB. The caller asks, because only the
    # caller knows the policy.
    def prefetch_whole(self, relpath: str, dest: "Path",
                       progress=None) -> "Path | None":
        """Fetch an entire segment once; return a readable path, or None.

        Prefers ``huggingface_hub.hf_hub_download`` when available: it brings
        connection reuse, resume, Xet dedup and — with ``hf_transfer``
        installed — parallel chunked transfer, none of which a single urllib
        stream gets. We only use ``hf_hub_url`` for URL construction on the
        RANGED path (arbitrary byte ranges are outside what hf_hub_download
        offers), but a whole-file pull is exactly its job.

        It also returns a path in the HF cache, which we use in place rather
        than copying: a 2.5 GB segment should not exist twice on disk.
        """
        import shutil
        import tempfile
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        if self.offline():
            return None
        if os.environ.get("FQ_PREFETCH_VIA_HF", "1") == "1":
            try:
                from huggingface_hub import hf_hub_download
                token = (os.environ.get("HF_TOKEN")
                         or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
                # local_dir puts the blob in OUR cache instead of the shared
                # HF cache. That costs cross-boot dedup but buys the thing we
                # actually need: the file is ours, so release_layer() may
                # unlink it. Left in the shared cache, a 75-layer boot pulls
                # the whole model in with nothing ever reclaiming it -- on a
                # box already at 86% that is a disk-exhaustion bug wearing an
                # optimisation's clothes. Opt back in with FQ_PREFETCH_HF_SHARED=1.
                _kw = {}
                if os.environ.get("FQ_PREFETCH_HF_SHARED", "0") != "1":
                    _kw["local_dir"] = str(dest.parent)
                path = hf_hub_download(
                    repo_id=self.repo_id, filename=relpath,
                    revision=self.revision, token=token, **_kw)
                got = Path(path)
                if got.exists() and got.stat().st_size > 0:
                    if progress is not None:
                        sz = got.stat().st_size
                        progress(sz, sz, float("inf"))
                    return got          # use the HF cache copy in place
            except Exception:  # noqa: BLE001 — fall back to the urllib path
                logger.debug("hf_hub_download unavailable for %s; "
                             "falling back to a plain stream", relpath,
                             exc_info=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with self._open(relpath, {}) as r:
                total = int(r.headers.get("Content-Length") or 0)
                fd, tmpname = tempfile.mkstemp(dir=str(dest.parent),
                                               suffix=".part")
                tmp = Path(tmpname)
                # Copy in chunks with periodic progress instead of one opaque
                # copyfileobj. A 2.5 GB segment is minutes of a silent log,
                # and an operator staring at a stalled-looking boot cannot
                # tell a slow download from a hung one.
                import time as _t
                t0 = _t.monotonic()
                done = 0
                nxt = 256 << 20          # report every 256 MiB
                with open(fd, "wb") as fh:
                    while True:
                        chunk = r.read(8 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        if progress is not None and done >= nxt:
                            el = max(_t.monotonic() - t0, 1e-9)
                            progress(done, total, done / el)
                            nxt += 256 << 20
                if progress is not None:
                    el = max(_t.monotonic() - t0, 1e-9)
                    progress(done, total or done, done / el)
            tmp.replace(dest)          # atomic: a torn file must never be read
            tmp = None
            return dest
        except Exception:  # noqa: BLE001 — prefetch is an optimisation
            return None
        finally:
            if tmp is not None and tmp.exists():
                tmp.unlink()

    def range_from_prefetched(self, cached: "Path", start: int,
                              end: int) -> bytes | None:
        """Slice a prefetched segment, or None if it cannot serve the range."""
        try:
            size = cached.stat().st_size
            if end > size:
                return None            # truncated/stale: fall back to HTTP
            with open(cached, "rb") as fh:
                fh.seek(start)
                data = fh.read(end - start)
            return data if len(data) == end - start else None
        except OSError:
            return None

    def read_range(self, relpath: str, start: int, end: int) -> bytes:
        import http.client
        import time
        import urllib.error

        want = end - start
        last: Exception | None = None
        for attempt in range(self._RETRIES):
            try:
                with self._open(
                    relpath, {"Range": f"bytes={start}-{end - 1}"}
                ) as r:
                    data = r.read()
                if len(data) != want:
                    # A short read is itself the transient failure mode here
                    # (truncated chunked transfer), so it retries rather than
                    # being reported as corruption.
                    raise IOError(
                        f"{self.name}/{relpath}: ranged read [{start},{end}) "
                        f"returned {len(data)} bytes")
                return data
            except OfflineError:
                raise                      # a configuration, not a blip
            except urllib.error.HTTPError:
                raise                      # 404/403: a real answer, not a blip
            except (http.client.IncompleteRead, http.client.RemoteDisconnected,
                    urllib.error.URLError, TimeoutError, ConnectionError,
                    IOError) as exc:
                last = exc
                if attempt + 1 < self._RETRIES:
                    time.sleep(self._BACKOFF * (2 ** attempt))
        raise IOError(
            f"{self.name}/{relpath}: ranged read [{start},{end}) failed after "
            f"{self._RETRIES} attempts: {type(last).__name__}: {last}")



class _DownloadMonitor:
    """Aggregate download progress for the boot log.

    hf_hub_download gives us ONE completion callback, not incremental ones, so
    the primary fetch path was a silent log -- exactly the "staring at a log
    that is not progressing" problem the per-file progress was meant to solve,
    just moved. Rather than reach into hub internals for per-file byte counts,
    watch the .incomplete files the client leaves in the cache and report the
    aggregate. For an operator that is the better line anyway: four ranks
    times several objects would interleave into noise, while one line answers
    "is it moving, how fast, how many in flight".
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, root: Path, every: float = 15.0):
        self.root = Path(root)
        self.every = every
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sizes: dict[str, int] = {}
        self._done_bytes = 0

    @classmethod
    def ensure(cls, root: Path) -> "_DownloadMonitor":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(root)
                cls._instance.start()
            return cls._instance

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="fq-dl-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _scan(self) -> tuple[int, int]:
        total = count = 0
        try:
            for f in self.root.rglob("*.incomplete"):
                try:
                    total += f.stat().st_size
                    count += 1
                except OSError:
                    continue
        except OSError:
            pass
        return total, count

    def _progress(self) -> tuple[int, int]:
        """(monotonic bytes downloaded, files in flight).

        Summing the in-flight files alone is NOT progress: a completed file
        leaves the .incomplete set, so the sum DROPS and the rate reads 0
        MiB/s next to gigabytes of real transfer -- observed live as
        "3 in flight, 1.8 GiB this boot, 0 MiB/s". Carry the bytes of files
        that vanished into a completed total so the series only ever rises.
        """
        seen: dict[str, int] = {}
        count = 0
        try:
            for f in self.root.rglob("*.incomplete"):
                try:
                    seen[str(f)] = f.stat().st_size
                    count += 1
                except OSError:
                    continue
        except OSError:
            pass
        for path, size in self._sizes.items():
            if path not in seen:
                self._done_bytes += size
        self._sizes = seen
        return self._done_bytes + sum(seen.values()), count

    def _run(self) -> None:
        prev, t_prev = self._progress()[0], time.monotonic()
        while not self._stop.wait(self.every):
            cur, n = self._progress()
            now = time.monotonic()
            delta, dt = cur - prev, max(now - t_prev, 1e-9)
            if n == 0 and delta <= 0:
                prev, t_prev = cur, now
                continue
            logger.info(
                "FQ downloads: %d in flight, %.1f GiB this boot, %.0f MiB/s",
                n, cur / (1 << 30), max(delta, 0) / dt / (1 << 20))
            prev, t_prev = cur, now


class FragmentResolver:
    """Resolve (layer, expert, K) fragments: local dirs -> sources -> error.

    The seam the M4 swap engine reuses: :meth:`resolve` for raw payloads,
    :meth:`expert_tensors` for materialized ``(name, tensor)`` pairs.
    """

    def __init__(
        self,
        manifest_dir: str | Path,
        *,
        local_dirs: list[str | Path] | None = None,
        sources: list[Any] | None = None,
        cache_dir: str | Path | None = None,
        verify: str | None = None,
        source_factory: Callable[[str], Any] = HfSource,
        environ: dict[str, str] = os.environ,  # noqa: B008
        encode_queue: Any = None,
    ):
        self._environ = environ
        self.manifest_dir = Path(manifest_dir)
        manifest_path = self.manifest_dir / "fq-manifest.json"
        self.manifest: dict = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        )

        dirs: list[Path] = [self.manifest_dir]
        for extra in (environ.get(FQ_LOCAL_SEGMENTS_ENV) or "").split(":"):
            if extra:
                dirs.append(Path(extra))
        if local_dirs:
            dirs.extend(Path(d) for d in local_dirs)
        self.local_dirs = dirs

        self.cache_dir = Path(
            cache_dir
            or environ.get(FQ_CACHE_ENV)
            or os.path.expanduser(DEFAULT_CACHE_DIR)
        )

        self.verify = (verify or environ.get(FQ_VERIFY_ENV) or "fetched").lower()
        if self.verify not in ("fetched", "all", "off"):
            raise ValueError(
                f"{FQ_VERIFY_ENV} must be fetched|all|off, got {self.verify!r}"
            )

        if sources is not None:
            self.sources = list(sources)
        else:
            manifest_specs = [
                s
                for s in self.manifest.get("sources", ())
                if isinstance(s, str) and not s.startswith("local:")
            ]
            env_raw = environ.get(FQ_SOURCES_ENV)
            env_specs = (
                [s.strip() for s in env_raw.split(",") if s.strip()]
                if env_raw is not None
                else None
            )
            mode = (environ.get(FQ_SOURCES_MODE_ENV) or "prepend").lower()
            if mode not in SOURCES_MODES:
                raise ValueError(
                    f"{FQ_SOURCES_MODE_ENV} must be one of {SOURCES_MODES}, "
                    f"got {mode!r}"
                )
            if env_specs is None:
                raw = manifest_specs
            elif mode == "replace":
                raw = env_specs
            elif mode == "append":
                raw = manifest_specs + env_specs
            else:  # prepend (default)
                raw = env_specs + manifest_specs
            seen_specs: set[str] = set()
            deduped: list[str] = []
            for spec in raw:
                key = spec.removeprefix("hf:")
                if key not in seen_specs:
                    seen_specs.add(key)
                    deduped.append(spec)
            self.sources = [source_factory(spec) for spec in deduped]

        # -- trust policy (10 §4): active only when trust anchors exist
        preds_raw = environ.get(FQ_TRUST_PREDICATES_ENV)
        self.trust_predicates: tuple[str, ...] = (
            tuple(p.strip() for p in preds_raw.split(",") if p.strip())
            if preds_raw is not None
            else DEFAULT_TRUST_PREDICATES
        )
        signers_raw = environ.get(FQ_TRUST_SIGNERS_ENV)
        if signers_raw is not None:
            signers = [s.strip().lower() for s in signers_raw.split(",") if s.strip()]
        else:
            manifest_key = self.manifest.get("signer_pubkey")
            signers = [str(manifest_key).lower()] if manifest_key else []
        self.trust_signers = frozenset(signers)
        self.trust_enabled = bool(self.trust_signers)

        # -- lazy-encode fallback ladder
        # mode: "explicit" (operator listed the Ks), "auto" (derive the
        # nearest-lower ladder from what this deployment can supply) or
        # "off" (no substitution at all).  ``k_fallback`` stays the EXPLICIT
        # list: strict resolve() only ever honours that, so tooling and the
        # trust tests keep their fail-closed contract.
        fallback_raw = environ.get(FQ_K_FALLBACK_ENV)
        token = (fallback_raw or "").strip().lower()
        self.k_fallback: tuple[int, ...] = ()
        if fallback_raw is None or token in ("", "auto"):
            self.k_fallback_mode = "auto"
        elif token in ("off", "none"):
            self.k_fallback_mode = "off"
        else:
            self.k_fallback_mode = "explicit"
            try:
                self.k_fallback = tuple(
                    int(part)
                    for part in fallback_raw.split(",") if part.strip()
                )
            except ValueError as exc:
                raise ValueError(
                    f"{FQ_K_FALLBACK_ENV} must be a comma list of Ks (or "
                    f"auto|off), got {fallback_raw!r}"
                ) from exc
        self.k_fallback_up = (
            (environ.get(FQ_K_FALLBACK_UP_ENV) or "0").strip().lower() in _TRUE
        )
        self._local_k_universe: tuple[int, ...] | None = None
        self._encode_queue = encode_queue
        self._encode_queue_ready = encode_queue is not None

        # Counter, not dict: `self.stats[k] += 1` on an
        # undeclared key raised KeyError from the SUCCESS path
        # of the prefetch fast-path, so every expert served
        # from a prefetched segment was reported as a source
        # rejection and fell down the K ladder. Telemetry must
        # never be able to fail the thing it measures.
        self.stats = Counter({
            "local": 0,
            "cache": 0,
            "fetched": 0,
            "verified": 0,
            "sha_mismatch": 0,
            "bytes_fetched": 0,
            # decision counters (this milestone)
            "source_miss": 0,
            "source_error": 0,
            "reject_predicate": 0,
            "reject_signer": 0,
            "reject_signature": 0,
            "reject_no_attestation": 0,
            "reject_no_expert_sha": 0,
            "reject_sha_mismatch": 0,
            "fallback_substituted": 0,
            "encode_queued": 0,
            "segments_prefetched": 0,
            "segments_released": 0,
            "segments_shared": 0,
            "bytes_from_prefetch": 0,
            "unavailable": 0,
            # hardening counters: a broken local dir / unreadable cache entry
            # / unwritable cache must degrade the attempt, never the engine
            "local_error": 0,
            "cache_error": 0,
            "cache_write_error": 0,
            "resolve_error": 0,
        })
        # (dir, layer, k) -> _LocalSegment | None
        self._local_segments: dict[tuple[Path, int, int], _LocalSegment | None] = {}
        # (layer, k) -> {expert_str: sha} from LOCAL dirs only
        self._local_sha_maps: dict[tuple[int, int], dict[str, str]] = {}
        # (source_name, layer, k) -> attestation text | None (None=definitive)
        # (layer, k) -> whole prefetched segment on disk, for uniform layers
        self._prefetched: dict[tuple[int, int], Path] = {}
        self._reject_traced: set = set()
        # Prefetch runs on a thread pool now; the memo dicts below
        # were written for a single thread.
        self._memo_lock = threading.RLock()
        self._att_texts: dict[tuple[str, int, int], str | None] = {}
        # (source_name, layer, k) -> (shas | None, reject reason | None)
        self._att_evals: dict[
            tuple[str, int, int], tuple[dict[str, str] | None, str | None]
        ] = {}
        # (source_name, k) -> index dict | None
        self._remote_indexes: dict[tuple[str, int], dict | None] = {}
        # file_sha -> (header, tables-by-expert)
        self._remote_headers: dict[str, tuple[dict, dict[int, list[FragmentTensor]]]] = {}

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _seg_name(layer: int, k: int) -> str:
        return f"layer-{layer:03d}.k{k}.safetensors"

    @staticmethod
    def _att_name(layer: int, k: int) -> str:
        return f"attestations/layer-{layer:03d}.k{k}.jsonl"

    def _local_segment(
        self, base: Path, layer: int, k: int
    ) -> _LocalSegment | None:
        key = (base, layer, k)
        if key in self._local_segments:
            return self._local_segments[key]
        seg: _LocalSegment | None = None
        try:
            index_path = base / f"index-k{k}.json"
            seg_path = base / self._seg_name(layer, k)
            if index_path.exists() and seg_path.exists():
                index = json.loads(index_path.read_text())
                entry = (index.get(str(layer))
                         if isinstance(index, dict) else None)
                if entry is not None:
                    seg = _LocalSegment(seg_path, _SegmentIndexEntry(entry))
        except Exception:  # noqa: BLE001 — a broken local dir is a MISS
            # Truncated index-k{K}.json, index/segment body-offset skew, an
            # unreadable segment: this dir cannot supply the fragment, but
            # the cache / sources / fallback ladder still can.  Memoized as
            # None so the warning fires once per (dir, layer, K).
            self.stats["local_error"] += 1
            logger.warning(
                "FQ local segment dir %s unusable for L%d K%d — treating as "
                "a miss", base, layer, k, exc_info=True,
            )
            seg = None
        self._local_segments[key] = seg
        return seg

    def _local_shas(self, layer: int, k: int) -> dict[str, str]:
        """Per-expert sha map from the LOCAL segment dirs' attestations."""
        key = (layer, k)
        shas = self._local_sha_maps.get(key)
        if shas is None:
            shas = {}
            for base in self.local_dirs:
                p = base / self._att_name(layer, k)
                try:
                    if p.exists():
                        shas = _attestation_expert_shas(p.read_text())
                        break
                except OSError:  # unreadable attestation: try the next dir
                    self.stats["local_error"] += 1
                    logger.warning(
                        "FQ attestation %s unreadable — skipping", p,
                        exc_info=True,
                    )
            self._local_sha_maps[key] = shas
        return shas

    # ------------------------------------------------- per-source trust

    def _source_att_text(
        self, source: Any, layer: int, k: int, *, fetch: bool = True
    ) -> str | None:
        """Attestation jsonl of one source (memo -> disk cache -> fetch)."""
        name = getattr(source, "name", str(source))
        key = (name, layer, k)
        if key in self._att_texts:
            return self._att_texts[key]
        path = self._cache_path(
            "attestations",
            f"{_safe_source_key(name)}/layer-{layer:03d}.k{k}.jsonl",
        )
        if path.exists():
            text = path.read_text()
            self._att_texts[key] = text
            return text
        if not fetch:
            return None  # unknown, not definitive: don't memoize
        text = source.read_text(self._att_name(layer, k))
        if text is not None:
            self._cache_store(path, text.encode())
        self._att_texts[key] = text
        return text

    def _evaluate_attestation(
        self, text: str
    ) -> tuple[dict[str, str] | None, str | None]:
        """(expert sha map, None) if some line is trusted, else (None, why).

        Trust disabled: any line with an ``expert_sha256`` payload counts
        (legacy sha-only behavior).  Trust enabled: a line counts only when
        its ed25519 signature verifies under an allowed signer AND its
        predicate is trusted; with multiple lines (countersignatures), ANY
        allowed line accepts the fragment."""
        if not self.trust_enabled:
            shas = _attestation_expert_shas(text)
            return (shas, None) if shas else (None, "no-attestation")
        reasons: list[str] = []
        for envelope, raw, payload in _attestation_lines(text):
            keyid = str(envelope.get("keyid", "")).lower()
            predicate = str(payload.get("predicate", "?"))
            if keyid not in self.trust_signers:
                reasons.append("signer not-trusted")
                continue
            try:
                signature = base64.b64decode(envelope.get("signature") or "")
            except (ValueError, TypeError):
                signature = b""
            if not _verify_ed25519(keyid, raw, signature):
                reasons.append("bad-signature")
                continue
            if predicate not in self.trust_predicates:
                reasons.append(f"predicate={predicate} not-trusted")
                continue
            shas = payload.get("expert_sha256")
            if isinstance(shas, dict) and shas:
                return shas, None
            reasons.append("no-expert-sha")
        for prefix in ("predicate=", "bad-signature", "signer not-trusted"):
            for reason in reasons:
                if reason.startswith(prefix):
                    return None, reason
        return None, (reasons[0] if reasons else "no-attestation")

    _REASON_COUNTERS = (
        ("predicate=", "reject_predicate"),
        ("signer not-trusted", "reject_signer"),
        ("bad-signature", "reject_signature"),
        ("no-attestation", "reject_no_attestation"),
        ("no-expert-sha", "reject_no_expert_sha"),
        ("sha-mismatch", "reject_sha_mismatch"),
    )

    def _count_reject(self, reason: str) -> None:
        for prefix, counter in self._REASON_COUNTERS:
            if reason.startswith(prefix):
                self.stats[counter] += 1
                return

    def _att_decision(
        self, source: Any, layer: int, k: int, *, fetch: bool = True
    ) -> tuple[dict[str, str] | None, str | None]:
        """Memoized trust decision for one source's (layer, k) attestation."""
        name = getattr(source, "name", str(source))
        key = (name, layer, k)
        if key in self._att_evals:
            return self._att_evals[key]
        text = self._source_att_text(source, layer, k, fetch=fetch)
        if text is None:
            decision: tuple[dict[str, str] | None, str | None] = (
                None,
                "no-attestation",
            )
            if fetch:  # definitive: the source has none
                self._att_evals[key] = decision
            return decision
        decision = self._evaluate_attestation(text)
        self._att_evals[key] = decision
        return decision

    def _cache_expected_sha(self, layer: int, k: int, expert: int) -> str | None:
        """Expected sha for the content-addressed cache step: local dirs
        first, then already-cached (disk) per-source attestations that pass
        the trust filter — never the network."""
        sha = self._local_shas(layer, k).get(str(expert))
        if sha:
            return sha
        for source in self.sources:
            try:
                shas, _reason = self._att_decision(
                    source, layer, k, fetch=False
                )
            except Exception:  # noqa: BLE001 — cache probing is best-effort
                continue
            if shas:
                sha = shas.get(str(expert))
                if sha:
                    return sha
        return None

    # -------------------------------------------------- lazy-encode queue

    def _get_encode_queue(self) -> Any:
        if not self._encode_queue_ready:
            self._encode_queue_ready = True
            try:
                self._encode_queue = _load_lazy_encode().EncodeQueue.from_env(
                    self._environ, cache_dir=self.cache_dir
                )
            except Exception:  # noqa: BLE001 — queue is telemetry, not boot
                logger.warning(
                    "FQ lazy-encode queue unavailable", exc_info=True
                )
                self._encode_queue = None
        return self._encode_queue

    def _enqueue_encode(
        self, layer: int, expert: int, k: int, reason: str
    ) -> tuple[int | None, bool]:
        queue = self._get_encode_queue()
        if queue is None:
            return None, False
        try:
            position, created = queue.enqueue(layer, expert, k, reason)
        except Exception:  # noqa: BLE001 — never block resolution on the queue
            logger.warning("FQ lazy-encode enqueue failed", exc_info=True)
            return None, False
        if created:
            self.stats["encode_queued"] += 1
        return position, created

    def prefetch_layer(self, layer: int, k: int) -> str | None:
        """Pull the whole K``k`` segment for ``layer`` from the first source
        that has it. Call this when the layer is UNIFORM at ``k`` -- it turns
        256 ranged reads into one object fetch. Returns a short description of
        what happened, for the boot log; never raises.
        """
        for source in self.sources:
            try:
                index = self._remote_index(source, k)
                if not isinstance(index, dict) or str(layer) not in index:
                    continue
                entry = _SegmentIndexEntry(index[str(layer)])
                dest = self._cache_path("segments", f"{entry.sha256}.seg")
                ready = self._segment_ready(dest, entry.file)
                if ready is not None:
                    self._prefetched[(layer, k)] = ready
                    self._claim_segment(ready)
                    return f"cached {entry.file}"
                got = getattr(source, "prefetch_whole", None)
                if got is None:
                    continue           # local dirs need no prefetch
                _DownloadMonitor.ensure(dest.parent)

                def _report(done, total, rate, _f=entry.file, _l=layer,
                            _k=k):
                    pct = (100.0 * done / total) if total else 0.0
                    logger.info(
                        "FQ fetch L%d K%d %s: %.1f/%.1f GiB (%.0f%%) at "
                        "%.0f MiB/s", _l, _k, _f,
                        done / (1 << 30), (total or done) / (1 << 30), pct,
                        rate / (1 << 20))

                # Only ONE rank downloads. The others block here, then find
                # the file already present on the re-check below.
                with self._segment_lock(dest):
                    again = self._segment_ready(dest, entry.file)
                    if again is not None:
                        self._prefetched[(layer, k)] = again
                        self._claim_segment(again)
                        self.stats["segments_shared"] += 1
                        return f"shared {entry.file} (fetched by another rank)"
                    path = got(entry.file, dest, _report)
                if path is not None:
                    self._prefetched[(layer, k)] = path
                    self._claim_segment(path)
                    self.stats["segments_prefetched"] += 1
                    return (f"prefetched {entry.file} "
                            f"({path.stat().st_size} B) from "
                            f"{getattr(source, 'name', source)}")
            except Exception:  # noqa: BLE001 — an optimisation must not fail a boot
                self.stats["source_error"] += 1
                continue
        return None

    # ----------------------------------------------------- cross-rank sharing
    # Every TP rank runs its OWN progressive_weights_iterator over the SAME
    # policy, so all of them want the same segment objects at the same time --
    # observed as four .part files racing for one file, i.e. 4x the bytes and
    # 4x the transient disk. With prefetch depth x width that becomes ~24
    # concurrent fetches of ~6 distinct files. One rank downloads; the rest
    # wait on the lock and then find it already there.
    @contextlib.contextmanager
    def _segment_lock(self, dest: Path, timeout: float = 1800.0):
        import fcntl
        lock = dest.with_name(dest.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock, "a+")
        deadline = time.monotonic() + timeout
        held = False
        try:
            while True:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        # Losing the lock must never fail a boot: fall through
                        # and fetch our own copy. The rename is atomic, so a
                        # duplicate download is wasteful, not incorrect.
                        logger.warning(
                            "FQ segment lock timeout on %s — fetching our own "
                            "copy", dest.name)
                        break
                    time.sleep(0.5)
            yield held
        finally:
            if held:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            fh.close()

    def _segment_ready(self, dest: Path, relpath: str) -> "Path | None":
        """An already-complete copy, under either name.

        The urllib path renames to ``dest``; hf_hub_download with local_dir
        writes to ``dest.parent/relpath``. Checking only ``dest`` would make
        every rank re-download a file the HF client had already placed.
        """
        for cand in (dest, dest.parent / relpath):
            try:
                if cand.exists() and cand.stat().st_size > 0:
                    return cand
            except OSError:
                continue
        return None

    def _users_dir(self, dest: Path) -> Path:
        return dest.with_name(dest.name + ".users")

    def _claim_segment(self, dest: Path) -> None:
        """Mark this process as a user of ``dest`` so another rank's
        release_layer() cannot unlink a segment we are still reading."""
        try:
            d = self._users_dir(dest)
            d.mkdir(parents=True, exist_ok=True)
            (d / str(os.getpid())).touch()
        except OSError:
            pass

    def _drop_segment_claim(self, dest: Path) -> bool:
        """Release our claim; return True if NO live user remains.

        Stale markers from killed ranks are pruned by checking /proc, so a
        preempted worker cannot pin a segment on disk forever.
        """
        d = self._users_dir(dest)
        try:
            (d / str(os.getpid())).unlink()
        except OSError:
            pass
        if not d.exists():
            # Never claimed (single-process use, or a caller that populated
            # _prefetched directly). No claim means no other holder -- NOT
            # "unknown, so refuse", which would make the segment unfreeable
            # and defeat the eviction entirely.
            return True
        try:
            for marker in d.iterdir():
                if marker.name.isdigit() and Path("/proc", marker.name).exists():
                    return False
                try:
                    marker.unlink()          # stale: owner is gone
                except OSError:
                    return False
            d.rmdir()
        except OSError:
            return False
        return True

    def release_layer(self, layer: int) -> int:
        """Drop the whole-segment objects prefetched for ``layer``.

        Prefetch had no eviction: ``_prefetched`` grew monotonically and
        nothing unlinked, so a full progressive boot left every segment it
        touched on disk -- hundreds of GB, unbounded, regardless of how far
        ahead we prefetch. Depth controls lookahead; only this controls
        FOOTPRINT. Called once a layer's tensors have streamed to the GPU, it
        makes resident bytes O(depth) instead of O(model).

        Only unlinks paths inside our own cache dir: a file the shared HF
        cache owns may be in use by another process, and deleting other
        people's cache entries is not ours to do. Set VLLM_FQ_PREFETCH_KEEP=1
        to retain everything (faster repeat boots, if disk is free).
        """
        if os.environ.get("VLLM_FQ_PREFETCH_KEEP", "0") == "1":
            return 0
        freed = 0
        try:
            root = self.cache_dir.resolve()
        except OSError:
            return 0
        for key in [k for k in self._prefetched if k[0] == layer]:
            path = self._prefetched.pop(key, None)
            if path is None:
                continue
            try:
                resolved = Path(path).resolve()
                if not resolved.is_relative_to(root):
                    continue           # shared HF cache: not ours to unlink
                if not self._drop_segment_claim(resolved):
                    continue           # another rank is still reading it
                size = resolved.stat().st_size
                resolved.unlink()
                for aux in (resolved.with_name(resolved.name + ".lock"),):
                    try:
                        aux.unlink()
                    except OSError:
                        pass
                freed += size
                self.stats["segments_released"] += 1
            except OSError:
                continue
        if freed:
            logger.info("FQ progressive L%d: released %.1f GiB of prefetched "
                        "segments", layer, freed / (1 << 30))
        return freed

    def _cache_path(self, kind: str, name: str) -> Path:
        return self.cache_dir / kind / name

    def _cache_fragment_path(self, sha: str) -> Path:
        return self.cache_dir / "fragments" / sha[:2] / sha

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def _cache_store(self, path: Path, data: bytes) -> bool:
        """Best-effort cache write: a full or read-only cache dir must never
        throw away a fragment that was already fetched and verified."""
        try:
            self._atomic_write(path, data)
            return True
        except OSError:
            self.stats["cache_write_error"] += 1
            logger.warning(
                "FQ cache write failed (%s) — serving uncached", path,
                exc_info=True,
            )
            return False

    def _check_sha(
        self, payload: Any, expected: str | None, what: str
    ) -> str:
        digest = hashlib.sha256(bytes(payload)).hexdigest()
        if expected is not None and digest != expected:
            self.stats["sha_mismatch"] += 1
            raise FragmentVerificationError(
                f"{what}: sha256 {digest[:16]}... != expected {expected[:16]}..."
            )
        self.stats["verified"] += 1
        return digest

    # ------------------------------------------------------------ resolve

    # -------------------------------------------------- fallback ladder

    def k_universe(self) -> tuple[int, ...]:
        """Bitrates this deployment could plausibly supply.

        Manifest ``k_values`` plus every ``index-k{K}.json`` in a local
        segment dir plus every K already consulted on a source; a deployment
        that advertises nothing falls back to :data:`DEFAULT_K_UNIVERSE`.
        Pure discovery — no network, no payload reads."""
        if self._local_k_universe is None:
            ks: set[int] = set()
            for value in self.manifest.get("k_values") or ():
                try:
                    ks.add(int(value))
                except (TypeError, ValueError):
                    continue
            for base in self.local_dirs:
                try:
                    names = [p.name for p in base.glob("index-k*.json")]
                except OSError:  # unreadable dir: it advertises nothing
                    continue
                for name in names:
                    m = _INDEX_K_RE.match(name)
                    if m is not None:
                        ks.add(int(m.group(1)))
            self._local_k_universe = tuple(sorted(ks))
        ks = set(self._local_k_universe)
        ks.update(k for _name, k in self._remote_indexes)
        return tuple(sorted(ks)) or DEFAULT_K_UNIVERSE

    def fallback_ladder(self, k: int) -> tuple[int, ...]:
        """Substitute Ks to try for a missed ``k``, best first.

        ``VLLM_FQ_K_FALLBACK`` verbatim when the operator listed one; empty
        when it is ``off``; otherwise the auto ladder: nearest LOWER K first
        (a lower bitrate always fits the memory the higher one needed), then
        — only with ``VLLM_FQ_K_FALLBACK_UP=1`` — the nearest higher."""
        if self.k_fallback_mode == "off":
            return ()
        if self.k_fallback_mode == "explicit":
            return tuple(x for x in self.k_fallback if x != k)
        universe = self.k_universe()
        ladder = tuple(sorted((x for x in universe if x < k), reverse=True))
        if self.k_fallback_up:
            ladder += tuple(sorted(x for x in universe if x > k))
        return ladder

    # ------------------------------------------------------------ resolve

    def resolve(self, layer: int, expert: int, k: int) -> Fragment:
        """Strict resolve: local dirs -> cache -> sources, then the EXPLICIT
        ``VLLM_FQ_K_FALLBACK`` ladder; emits one decision line per call and
        enqueues a lazy encode on every substitution/miss.

        Raises :class:`FragmentUnavailableError` (or the first verification
        error) when nothing supplies the fragment. Serving paths want
        :meth:`resolve_best` instead — this one is the strict, auditable
        contract for tooling."""
        fragment = self._resolve_ladder(
            layer, expert, k,
            tuple(x for x in self.k_fallback if x != k),
            strict=True,
        )
        assert fragment is not None  # strict=True never returns None
        return fragment

    def resolve_best(
        self, layer: int, expert: int, k: int, *,
        chain_out: list[str] | None = None,
    ) -> Fragment | None:
        """Best available fragment for ``(layer, expert)`` — **never raises**.

        Order: the requested ``k``; else the nearest available lower K of
        :meth:`fallback_ladder` (logged loudly, ``requested_k`` recorded, an
        encode queued); else ``None`` so the caller keeps the incumbent tier.
        Any unexpected failure inside the resolver is caught and reported as
        ``None`` too: a missing weight must never reach an engine loop.

        ``chain_out``, when given, receives the per-attempt decision
        segments so a caller can quote the actual reason ("REJECT
        no-attestation", "REJECT error:URLError", ...) in its own message."""
        try:
            return self._resolve_ladder(
                layer, expert, k, self.fallback_ladder(k), strict=False,
                chain_out=chain_out,
            )
        except (SystemExit, KeyboardInterrupt):
            # SHUTDOWN IS NOT A FRAGMENT FAILURE. vLLM's multiproc executor
            # raises SystemExit from its SIGTERM handler, and it lands wherever
            # the worker happens to be — typically mid-socket-read here, since
            # a progressive boot spends most of its time in one. Catching
            # BaseException swallowed it and logged "fragment unavailable,
            # keeping the incumbent tier", so a worker asked to stop instead
            # kept loading with silently degraded tiers. Let it through.
            raise
        except BaseException:  # noqa: BLE001 — the whole point of this seam
            self.stats["resolve_error"] += 1
            logger.exception(
                "FQ resolve_best L%d/e%d K%d crashed — reporting the "
                "fragment as unavailable and keeping the incumbent tier",
                layer, expert, k,
            )
            if chain_out is not None:
                chain_out.append("RESOLVER ERROR")
            return None

    def _resolve_ladder(
        self, layer: int, expert: int, k: int, ladder: tuple[int, ...], *,
        strict: bool, chain_out: list[str] | None = None,
    ) -> Fragment | None:
        chain: list[str] = chain_out if chain_out is not None else []
        errors: list[Exception] = []
        fragment = self._resolve_k(layer, expert, k, chain, errors)

        substituted = False
        if fragment is None:
            for fallback_k in ladder:
                if fallback_k == k:
                    continue
                mark = len(chain)
                fragment = self._resolve_k(
                    layer, expert, fallback_k, chain, errors
                )
                if len(chain) > mark:  # merge the marker into the attempt
                    chain[mark] = f"FALLBACK K{fallback_k} {chain[mark]}"
                else:
                    chain.append(f"FALLBACK K{fallback_k}")
                if fragment is not None:
                    substituted = True
                    break

        queue_note = ""
        if fragment is None or substituted:
            reason = (
                f"substituted-with-k{fragment.k}"
                if substituted
                else "unavailable"
            )
            position, created = self._enqueue_encode(layer, expert, k, reason)
            if position is not None:
                queue_note = (
                    f" (encode queued #{position})"
                    if created
                    else " (encode already queued)"
                )

        if fragment is not None:
            fragment.requested_k = k
            if substituted:
                self.stats["fallback_substituted"] += 1
                chain[-1] += queue_note
                self._log_chain(logging.INFO, layer, expert, k, chain)
                logger.warning(
                    "FQ DEGRADED L%d/e%d: K%d unavailable, serving K%d "
                    "instead (origin=%s)%s", layer, expert, k, fragment.k,
                    fragment.origin, queue_note,
                )
            else:
                self._log_chain(logging.DEBUG, layer, expert, k, chain)
            return fragment

        self.stats["unavailable"] += 1
        chain.append(f"UNAVAILABLE{queue_note}")
        self._log_chain(logging.WARNING, layer, expert, k, chain)
        if not strict:
            logger.error(
                "FQ UNAVAILABLE L%d/e%d K%d: no source and no fallback K "
                "could supply it%s — keeping the incumbent tier",
                layer, expert, k, queue_note,
            )
            return None
        if errors:
            raise errors[0]
        raise FragmentUnavailableError(
            f"layer {layer} expert {expert} k{k}: {'; '.join(chain)} "
            f"(local dirs: {[str(d) for d in self.local_dirs]}; extend "
            f"{FQ_SOURCES_ENV} / {FQ_K_FALLBACK_ENV} or drain the "
            "lazy-encode queue)"
        )

    # ------------------------------------------------------------- probe

    def probe(self, layer: int, expert: int, k: int, *,
              network: bool = True) -> bool:
        """Could ``(layer, expert, k)`` be supplied? No payload transfer.

        Local segment dirs and the content-addressed cache always count;
        source *indexes* count when ``network`` is on (one small JSON per
        (source, K), memoized). Used by the boot-time availability
        projection so the tier bitmap only ever declares Ks that exist."""
        for base in self.local_dirs:
            seg = self._local_segment(base, layer, k)
            if seg is not None and expert in seg.tables:
                return True
        try:
            expected = self._cache_expected_sha(layer, k, expert)
            if expected is not None and self._cache_fragment_path(
                    expected).exists():
                return True
        except Exception:  # noqa: BLE001 — probing is best-effort
            pass
        if not network:
            return False
        for source in self.sources:
            try:
                index = self._remote_index(source, k)
            except Exception:  # noqa: BLE001 — a down mirror proves nothing
                continue
            if not isinstance(index, dict):
                continue
            entry = index.get(str(layer))
            if isinstance(entry, dict) and str(expert) in (
                    entry.get("experts") or {}):
                return True
        return False

    def available_k(self, layer: int, expert: int, k: int, *,
                    network: bool = True) -> int | None:
        """The K :meth:`resolve_best` would actually serve, or ``None``."""
        if self.probe(layer, expert, k, network=network):
            return k
        for candidate in self.fallback_ladder(k):
            if candidate != k and self.probe(
                    layer, expert, candidate, network=network):
                return candidate
        return None

    @staticmethod
    def _log_chain(
        level: int, layer: int, expert: int, k: int, chain: list[str]
    ) -> None:
        logger.log(
            level,
            "FQ resolve L%d/e%d K%d: %s",
            layer,
            expert,
            k,
            "; ".join(chain),
        )

    def _resolve_k(
        self,
        layer: int,
        expert: int,
        k: int,
        chain: list[str],
        errors: list[Exception],
    ) -> Fragment | None:
        """One K's attempt chain: local dirs -> fragment cache -> sources.

        Appends one decision segment per attempt to ``chain``; verification
        failures are recorded in ``errors`` (re-raised by resolve() only if
        nothing else accepts) so one bad mirror cannot mask a good one."""
        # 1. local segment dirs
        for base in self.local_dirs:
            seg = self._local_segment(base, layer, k)
            if seg is None or expert not in seg.tables:
                continue
            payload, tensors = seg.fragment(expert)
            sha = None
            if self.verify == "all":
                try:
                    sha = self._check_sha(
                        payload,
                        self._local_shas(layer, k).get(str(expert)),
                        f"local {seg.path.name} expert {expert}",
                    )
                except FragmentVerificationError as exc:
                    chain.append("local REJECT sha-mismatch")
                    self._count_reject("sha-mismatch")
                    errors.append(exc)
                    return None
            self.stats["local"] += 1
            chain.append("local ACCEPT")
            return Fragment(layer, expert, k, payload, tensors, "local", sha)
        chain.append(f"local({len(self.local_dirs)} dirs) MISS")

        # 2. fragment cache (content-addressed by trusted expected sha)
        try:
            expected = self._cache_expected_sha(layer, k, expert)
        except Exception:  # noqa: BLE001 — a broken cache is never fatal
            self.stats["cache_error"] += 1
            logger.warning("FQ cache lookup failed for L%d/e%d K%d",
                           layer, expert, k, exc_info=True)
            expected = None
        if expected is not None:
            cached = self._cache_fragment_path(expected)
            try:
                payload = cached.read_bytes() if cached.exists() else None
            except OSError:  # evicted/unreadable under us: fall through
                self.stats["cache_error"] += 1
                logger.warning("FQ cache entry %s unreadable", cached,
                               exc_info=True)
                payload = None
            if payload is not None:
                try:
                    self._check_sha(
                        payload,
                        expected,
                        f"cache {cached.name} expert {expert}",
                    )
                except FragmentVerificationError as exc:
                    chain.append("cache REJECT sha-mismatch")
                    self._count_reject("sha-mismatch")
                    errors.append(exc)
                else:
                    tensors = self._tensor_table_for(layer, expert, k)
                    if tensors is not None:
                        self.stats["cache"] += 1
                        chain.append("cache ACCEPT")
                        return Fragment(
                            layer, expert, k, payload, tensors, "cache",
                            expected,
                        )

        # 3. sources chain
        for source in self.sources:
            fragment, segment, error = self._try_source(
                source, layer, expert, k
            )
            chain.append(segment)
            if error is not None:
                errors.append(error)
            if fragment is not None:
                return fragment
        return None

    def _remote_index(self, source: Any, k: int) -> dict | None:
        key = (source.name, k)
        if key in self._remote_indexes:
            return self._remote_indexes[key]
        index = source.read_json(f"index-k{k}.json")
        self._remote_indexes[key] = index
        return index

    def _remote_tables(
        self, source: Any, entry: _SegmentIndexEntry
    ) -> dict[int, list[FragmentTensor]]:
        cached = self._remote_headers.get(entry.sha256)
        if cached is not None:
            return cached[1]
        header_cache = self._cache_path("headers", f"{entry.sha256}.json")
        header: dict | None = None
        if header_cache.exists():
            try:
                header = json.loads(header_cache.read_text())
            except (ValueError, OSError):
                # A poisoned cache entry (torn write, truncated file) must
                # self-heal rather than pin this segment as unreadable
                # forever: drop it and go back to the source.
                self.stats["cache_error"] += 1
                logger.warning(
                    "FQ cached segment header %s is unreadable — discarding "
                    "and re-fetching", header_cache, exc_info=True,
                )
                header = None
                try:
                    header_cache.unlink()
                except OSError:
                    pass
        if header is None:
            raw = source.read_range(entry.file, 0, entry.body_offset)
            if len(raw) != entry.body_offset:
                # A short read here is not a fragment problem, it is a
                # TRANSPORT problem, and it must not be cached or parsed. A
                # truncated or substituted body (an error page served with
                # 200, a connection cut mid-header) parses into a header that
                # disagrees with the index, and the disagreement then surfaces
                # 19,200 times as a bare `KeyError` from expert_range with no
                # indication that the network was the cause.
                raise SegmentHeaderMismatch(
                    f"{entry.file}: header read returned {len(raw)} bytes, "
                    f"expected {entry.body_offset} — transport truncation, "
                    f"not a bad segment")
            header, _ = parse_segment_header_bytes(raw)
            self._cache_store(
                header_cache, json.dumps(header, separators=(",", ":")).encode()
            )
        tables = _expert_tables_from_header(header, entry)

        self._remote_headers[entry.sha256] = (header, tables)
        return tables

    def _tensor_table_for(
        self, layer: int, expert: int, k: int
    ) -> list[FragmentTensor] | None:
        """Tensor table for a cached payload, from any cached segment header.

        Best-effort by construction: a mirror that is down, an index that no
        longer parses or a header cache entry evicted between the ``exists``
        probe and the read must degrade to "no table" (the caller then walks
        the source chain), never propagate — this runs on the boot and live
        swap paths."""
        for source in self.sources:
            try:
                index = self._remote_index(source, k)
                if not isinstance(index, dict) or str(layer) not in index:
                    continue
                entry = _SegmentIndexEntry(index[str(layer)])
                header_cache = self._cache_path(
                    "headers", f"{entry.sha256}.json")
                if (entry.sha256 in self._remote_headers
                        or header_cache.exists()):
                    return self._remote_tables(source, entry).get(expert)
            except Exception:  # noqa: BLE001 — try the next mirror
                self.stats["source_error"] += 1
                logger.warning(
                    "FQ tensor-table lookup failed on %s for L%d K%d",
                    getattr(source, "name", source), layer, k, exc_info=True,
                )
                continue
        return None

    def _reject(self, name: str, exc: Exception, where: str) -> str:
        """A rejection string that can actually be diagnosed.

        `REJECT error:KeyError` names the exception TYPE and discards its
        argument -- so a boot that degraded 190 experts to K2 produced 270
        identical, contentless lines and the cause had to be reverse-
        engineered from the source. The key IS the diagnosis. Also emit one
        traceback per (source, site) so the first occurrence is debuggable
        without flooding a 75-layer boot with 19,200 stacks.
        """
        detail = str(exc).strip().replace("\n", " ")[:160]
        seen_key = (name, where, type(exc).__name__)
        if seen_key not in self._reject_traced:
            self._reject_traced.add(seen_key)
            logger.warning("FQ %s REJECT at %s: %s: %s", name, where,
                           type(exc).__name__, detail, exc_info=True)
        out = f"{name} REJECT error:{type(exc).__name__}"
        return f"{out}({detail})" if detail else out

    def _try_source(
        self, source: Any, layer: int, expert: int, k: int
    ) -> tuple[Fragment | None, str, Exception | None]:
        """One source's attempt: trust filter -> ranged fetch -> sha check.

        Returns ``(fragment | None, decision segment, remembered error)``;
        never raises — a rejected/broken source only fails itself. The outer
        guard makes that literal rather than aspirational."""
        name = getattr(source, "name", str(source))
        try:
            return self._try_source_inner(source, layer, expert, k, name)
        except Exception as exc:  # noqa: BLE001 — contract: never raises
            self.stats["source_error"] += 1
            logger.warning("FQ source %s raised for L%d/e%d K%d",
                           name, layer, expert, k, exc_info=True)
            return None, self._reject(name, exc, "try_source"), None

    def _try_source_inner(
        self, source: Any, layer: int, expert: int, k: int, name: str
    ) -> tuple[Fragment | None, str, Exception | None]:
        try:
            index = self._remote_index(source, k)
        except Exception as exc:  # noqa: BLE001 — mirror down, try the next
            self.stats["source_error"] += 1
            return None, self._reject(name, exc, "remote_index"), None
        if index is None or str(layer) not in index:
            self.stats["source_miss"] += 1
            return None, f"{name} MISS", None
        entry = _SegmentIndexEntry(index[str(layer)])

        expected: str | None = None
        if not self.trust_enabled:
            expected = self._local_shas(layer, k).get(str(expert))
        if self.trust_enabled or expected is None:
            try:
                shas, reason = self._att_decision(source, layer, k)
            except Exception as exc:  # noqa: BLE001
                self.stats["source_error"] += 1
                return None, self._reject(name, exc, "att_decision"), None
            if shas is None:
                if self.trust_enabled:
                    self._count_reject(reason or "no-attestation")
                    return None, f"{name} REJECT {reason}", None
            else:
                sha_from_source = shas.get(str(expert))
                if self.trust_enabled:
                    # verify against the TRUSTED attestation of this source
                    expected = sha_from_source
                elif expected is None:
                    expected = sha_from_source
            if self.trust_enabled and expected is None:
                self._count_reject("no-expert-sha")
                return None, f"{name} REJECT no-expert-sha", None
        if expected is None and self.verify != "off":
            self._count_reject("no-attestation")
            return (
                None,
                f"{name} REJECT no-attestation",
                FragmentVerificationError(
                    f"layer {layer} expert {expert} k{k}: source {name} has "
                    f"no attestation sha; refusing unverified fetch (set "
                    f"{FQ_VERIFY_ENV}=off to allow)"
                ),
            )

        try:
            lo, hi = entry.expert_range(expert)
            start = entry.body_offset + lo
            stop = entry.body_offset + hi
            payload = None
            cached = self._prefetched.get((layer, k))
            if cached is not None:
                slicer = getattr(source, "range_from_prefetched", None)
                if slicer is not None:
                    payload = slicer(cached, start, stop)
                    if payload is not None:
                        self.stats["bytes_from_prefetch"] += len(payload)
            if payload is None:
                payload = source.read_range(entry.file, start, stop)
        except Exception as exc:  # noqa: BLE001
            self.stats["source_error"] += 1
            return None, self._reject(name, exc, "expert_range/read_range"), None
        self.stats["bytes_fetched"] += len(payload)
        if self.verify == "off" and expected is None:
            sha = hashlib.sha256(payload).hexdigest()
        else:
            try:
                sha = self._check_sha(
                    payload,
                    expected,
                    f"fetched {entry.file} expert {expert} from {name}",
                )
            except FragmentVerificationError as exc:
                self._count_reject("sha-mismatch")
                return None, f"{name} REJECT sha-mismatch", exc

        self._cache_store(self._cache_fragment_path(sha), payload)
        try:
            tensors = self._remote_tables(source, entry).get(expert)
        except Exception as exc:  # noqa: BLE001
            self.stats["source_error"] += 1
            return None, self._reject(name, exc, "remote_tables"), None
        if tensors is None:
            self.stats["source_miss"] += 1
            return None, f"{name} MISS", None
        self.stats["fetched"] += 1
        return (
            Fragment(layer, expert, k, payload, tensors, "fetched", sha),
            f"{name} ACCEPT",
            None,
        )

    # ------------------------------------------------------- tensor view

    def expert_tensors(
        self,
        layer: int,
        expert: int,
        k: int,
        *,
        name_filter: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, Any]]:
        """Materialize the fragment as ``[(checkpoint_name, cpu_tensor)]``.

        Note: with ``VLLM_FQ_K_FALLBACK`` active the loaded K may differ
        from the requested one; callers that must surface the actual K
        should call :meth:`resolve` + :meth:`materialize` instead."""
        return self.materialize(
            self.resolve(layer, expert, k), name_filter=name_filter
        )

    def best_tensors(
        self,
        layer: int,
        expert: int,
        k: int,
        *,
        name_filter: Callable[[str], bool] | None = None,
    ) -> tuple[int, list[tuple[str, Any]]] | None:
        """``(actual_k, [(name, tensor)])`` via :meth:`resolve_best`, or None.

        The never-raising counterpart of :meth:`expert_tensors`: nothing in
        here reaches an engine loop, and the caller is told which K it
        actually got so tier metadata can record reality."""
        fragment = self.resolve_best(layer, expert, k)
        if fragment is None:
            return None
        try:
            return fragment.k, self.materialize(
                fragment, name_filter=name_filter)
        except Exception:  # noqa: BLE001 — malformed payload == unavailable
            self.stats["resolve_error"] += 1
            logger.exception(
                "FQ materialize L%d/e%d K%d failed — treating the fragment "
                "as unavailable", layer, expert, fragment.k,
            )
            return None

    def materialize(
        self,
        fragment: Fragment,
        *,
        name_filter: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, Any]]:
        """``[(checkpoint_name, cpu_tensor)]`` views of a resolved fragment.

        Tensors are zero-copy views over the fragment payload (mmap for
        local segments); downstream weight loaders copy them to device."""
        import torch

        mv = memoryview(fragment.payload)
        out: list[tuple[str, Any]] = []
        for ft in fragment.tensors:
            if name_filter is not None and not name_filter(ft.name):
                continue
            dtype = getattr(torch, _ST_TO_TORCH_NAME[ft.dtype])
            numel = 1
            for dim in ft.shape:
                numel *= dim
            nbytes = numel * dtype.itemsize
            if nbytes != ft.end - ft.start:
                raise ValueError(
                    f"{ft.name}: {ft.shape} {ft.dtype} needs {nbytes} bytes, "
                    f"fragment slot holds {ft.end - ft.start}"
                )
            if numel == 0:
                out.append((ft.name, torch.empty(ft.shape, dtype=dtype)))
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                tensor = torch.frombuffer(
                    mv, dtype=dtype, count=numel, offset=ft.start
                ).view(ft.shape)
            out.append((ft.name, tensor))
        return out


# ------------------------------------------------- boot-time projection


def project_bits_to_available(
    resolver: FragmentResolver,
    bits_by_layer: dict[int, Any],
    *,
    network: bool = True,
    log: Callable[[str], None] | None = None,
) -> tuple[
    dict[int, list[int]],
    list[tuple[int, int, int, int]],
    list[tuple[int, int, int]],
]:
    """Project a policy's per-expert Ks onto what can actually be supplied.

    A progressive boot is only crash-free if the tier bitmap the model was
    configured with agrees with the Ks the resolver ends up streaming: the
    bitmap sizes the slabs, so a fragment substituted at a *different* K
    would fail a shape check deep inside the weight loader. Running this
    projection BEFORE the bitmap is written closes that gap — the policy
    only ever asks for Ks that exist, so :meth:`FragmentResolver.resolve`
    cannot miss at boot and no substitution is needed at stream time.

    Returns ``(projected, substitutions, missing)`` where ``substitutions``
    is ``[(layer, expert, requested_k, available_k)]`` and ``missing`` is
    ``[(layer, expert, requested_k)]`` for experts no K can supply — those
    keep their requested K in ``projected`` (there is nothing better to say)
    and are the operator's cue to drain the lazy-encode queue.
    """
    projected: dict[int, list[int]] = {}
    substitutions: list[tuple[int, int, int, int]] = []
    missing: list[tuple[int, int, int]] = []
    for layer, bits in bits_by_layer.items():
        layer = int(layer)
        row = [int(b) for b in bits]
        for expert, k in enumerate(row):
            got = resolver.available_k(layer, expert, k, network=network)
            if got == k:
                continue
            if got is None:
                missing.append((layer, expert, k))
                continue
            row[expert] = got
            substitutions.append((layer, expert, k, got))
        projected[layer] = row
    if log is not None:
        for layer, expert, want, got in substitutions:
            log(f"FQ projection L{layer}/e{expert}: K{want} unavailable "
                f"-> K{got}")
        for layer, expert, want in missing:
            log(
                f"FQ projection L{layer}/e{expert}: K{want} unavailable at "
                "EVERY K — encode it (lazy_encode --drain) or add a source"
            )
        log(
            f"FQ projection: {len(substitutions)} expert(s) demoted, "
            f"{len(missing)} unsatisfiable"
        )
    return projected, substitutions, missing
