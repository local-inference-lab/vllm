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
import hashlib
import json
import logging
import mmap
import os
import re
import struct
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

DEFAULT_CACHE_DIR = "~/.cache/vllm/fq"
DEFAULT_TRUST_PREDICATES = ("repack-of", "encode-of", "derived-from")
SOURCES_MODES = ("prepend", "replace", "append")

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

    def _open(self, relpath: str, headers: dict[str, str]):
        import urllib.request

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

    def read_range(self, relpath: str, start: int, end: int) -> bytes:
        with self._open(
            relpath, {"Range": f"bytes={start}-{end - 1}"}
        ) as r:
            data = r.read()
        if len(data) != end - start:
            raise IOError(
                f"{self.name}/{relpath}: ranged read [{start},{end}) returned "
                f"{len(data)} bytes"
            )
        return data


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
        fallback_raw = environ.get(FQ_K_FALLBACK_ENV) or ""
        try:
            self.k_fallback: tuple[int, ...] = tuple(
                int(part) for part in fallback_raw.split(",") if part.strip()
            )
        except ValueError as exc:
            raise ValueError(
                f"{FQ_K_FALLBACK_ENV} must be a comma list of Ks, got "
                f"{fallback_raw!r}"
            ) from exc
        self._encode_queue = encode_queue
        self._encode_queue_ready = encode_queue is not None

        self.stats = {
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
            "unavailable": 0,
        }
        # (dir, layer, k) -> _LocalSegment | None
        self._local_segments: dict[tuple[Path, int, int], _LocalSegment | None] = {}
        # (layer, k) -> {expert_str: sha} from LOCAL dirs only
        self._local_sha_maps: dict[tuple[int, int], dict[str, str]] = {}
        # (source_name, layer, k) -> attestation text | None (None=definitive)
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
        index_path = base / f"index-k{k}.json"
        seg_path = base / self._seg_name(layer, k)
        if index_path.exists() and seg_path.exists():
            index = json.loads(index_path.read_text())
            entry = index.get(str(layer))
            if entry is not None:
                seg = _LocalSegment(seg_path, _SegmentIndexEntry(entry))
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
                if p.exists():
                    shas = _attestation_expert_shas(p.read_text())
                    break
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
            self._atomic_write(path, text.encode())
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

    def resolve(self, layer: int, expert: int, k: int) -> Fragment:
        """Resolve one fragment through local dirs -> cache -> sources, then
        the ``VLLM_FQ_K_FALLBACK`` ladder; emits one decision line per call
        and enqueues a lazy encode on every substitution/miss."""
        chain: list[str] = []
        errors: list[Exception] = []
        fragment = self._resolve_k(layer, expert, k, chain, errors)

        substituted = False
        if fragment is None:
            for fallback_k in self.k_fallback:
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
            else:
                self._log_chain(logging.DEBUG, layer, expert, k, chain)
            return fragment

        self.stats["unavailable"] += 1
        chain.append(f"UNAVAILABLE{queue_note}")
        self._log_chain(logging.WARNING, layer, expert, k, chain)
        if errors:
            raise errors[0]
        raise FragmentUnavailableError(
            f"layer {layer} expert {expert} k{k}: {'; '.join(chain)} "
            f"(local dirs: {[str(d) for d in self.local_dirs]}; extend "
            f"{FQ_SOURCES_ENV} / {FQ_K_FALLBACK_ENV} or drain the "
            "lazy-encode queue)"
        )

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
        expected = self._cache_expected_sha(layer, k, expert)
        if expected is not None:
            cached = self._cache_fragment_path(expected)
            if cached.exists():
                payload = cached.read_bytes()
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
        if header_cache.exists():
            header = json.loads(header_cache.read_text())
        else:
            raw = source.read_range(entry.file, 0, entry.body_offset)
            header, _ = parse_segment_header_bytes(raw)
            self._atomic_write(
                header_cache, json.dumps(header, separators=(",", ":")).encode()
            )
        tables = _expert_tables_from_header(header, entry)
        self._remote_headers[entry.sha256] = (header, tables)
        return tables

    def _tensor_table_for(
        self, layer: int, expert: int, k: int
    ) -> list[FragmentTensor] | None:
        """Tensor table for a cached payload, from any cached segment header."""
        for source in self.sources:
            try:
                index = self._remote_index(source, k)
            except Exception:  # noqa: BLE001
                continue
            if index is None or str(layer) not in index:
                continue
            entry = _SegmentIndexEntry(index[str(layer)])
            header_cache = self._cache_path("headers", f"{entry.sha256}.json")
            if entry.sha256 in self._remote_headers or header_cache.exists():
                return self._remote_tables(source, entry).get(expert)
        return None

    def _try_source(
        self, source: Any, layer: int, expert: int, k: int
    ) -> tuple[Fragment | None, str, Exception | None]:
        """One source's attempt: trust filter -> ranged fetch -> sha check.

        Returns ``(fragment | None, decision segment, remembered error)``;
        never raises — a rejected/broken source only fails itself."""
        name = getattr(source, "name", str(source))
        try:
            index = self._remote_index(source, k)
        except Exception as exc:  # noqa: BLE001 — mirror down, try the next
            self.stats["source_error"] += 1
            return None, f"{name} REJECT error:{type(exc).__name__}", None
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
                return None, f"{name} REJECT error:{type(exc).__name__}", None
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
            payload = source.read_range(
                entry.file, entry.body_offset + lo, entry.body_offset + hi
            )
        except Exception as exc:  # noqa: BLE001
            self.stats["source_error"] += 1
            return None, f"{name} REJECT error:{type(exc).__name__}", None
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

        self._atomic_write(self._cache_fragment_path(sha), payload)
        try:
            tensors = self._remote_tables(source, entry).get(expert)
        except Exception as exc:  # noqa: BLE001
            self.stats["source_error"] += 1
            return None, f"{name} REJECT error:{type(exc).__name__}", None
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
