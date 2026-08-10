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

This module is deliberately stdlib-only at import time; torch is imported
lazily inside :meth:`FragmentResolver.expert_tensors` so the resolver core
stays testable on CPU without a built vllm.
"""
from __future__ import annotations

import base64
import hashlib
import json
import mmap
import os
import re
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

FQ_CACHE_ENV = "VLLM_FQ_CACHE"
FQ_VERIFY_ENV = "VLLM_FQ_VERIFY"
FQ_SOURCES_ENV = "VLLM_FQ_SOURCES"
FQ_LOCAL_SEGMENTS_ENV = "VLLM_FQ_LOCAL_SEGMENTS"

DEFAULT_CACHE_DIR = "~/.cache/vllm/fq"

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
    """No local dir or source could supply the fragment (T3 encode is out
    of scope for loader v2)."""


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
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        envelope = json.loads(line)
        payload = json.loads(base64.b64decode(envelope["payload"]))
        shas = payload.get("expert_sha256")
        if isinstance(shas, dict):
            return shas
    return {}


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
    """A resolved per-expert payload plus its tensor table."""

    layer: int
    expert: int
    k: int
    payload: Any  # bytes | memoryview (buffer protocol)
    tensors: list[FragmentTensor]
    origin: str  # "local" | "cache" | "fetched"
    sha256: str | None = None


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
    ):
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
            specs = environ.get(FQ_SOURCES_ENV)
            if specs is not None:
                raw = [s for s in specs.split(",") if s]
            else:
                raw = [
                    s
                    for s in self.manifest.get("sources", ())
                    if isinstance(s, str) and not s.startswith("local:")
                ]
            self.sources = [source_factory(spec) for spec in raw]

        self.stats = {
            "local": 0,
            "cache": 0,
            "fetched": 0,
            "verified": 0,
            "sha_mismatch": 0,
            "bytes_fetched": 0,
        }
        # (dir, layer, k) -> _LocalSegment | None
        self._local_segments: dict[tuple[Path, int, int], _LocalSegment | None] = {}
        # (layer, k) -> {expert_str: sha}
        self._expert_shas: dict[tuple[int, int], dict[str, str]] = {}
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

    def _expected_sha(self, layer: int, k: int, expert: int) -> str | None:
        key = (layer, k)
        shas = self._expert_shas.get(key)
        if shas is None:
            shas = {}
            for base in self.local_dirs:
                p = base / self._att_name(layer, k)
                if p.exists():
                    shas = _attestation_expert_shas(p.read_text())
                    break
            else:
                cached = self._cache_path("attestations", f"layer-{layer:03d}.k{k}.jsonl")
                if cached.exists():
                    shas = _attestation_expert_shas(cached.read_text())
            self._expert_shas[key] = shas
        return shas.get(str(expert))

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
        # 1. local segment dirs
        for base in self.local_dirs:
            seg = self._local_segment(base, layer, k)
            if seg is None:
                continue
            payload, tensors = seg.fragment(expert)
            sha = None
            if self.verify == "all":
                sha = self._check_sha(
                    payload,
                    self._expected_sha(layer, k, expert),
                    f"local {seg.path.name} expert {expert}",
                )
            self.stats["local"] += 1
            return Fragment(layer, expert, k, payload, tensors, "local", sha)

        # 2. fragment cache (content-addressed by expected sha)
        expected = self._expected_sha(layer, k, expert)
        if expected is not None:
            cached = self._cache_fragment_path(expected)
            if cached.exists():
                payload = cached.read_bytes()
                self._check_sha(
                    payload, expected, f"cache {cached.name} expert {expert}"
                )
                tensors = self._tensor_table_for(layer, expert, k)
                if tensors is not None:
                    self.stats["cache"] += 1
                    return Fragment(
                        layer, expert, k, payload, tensors, "cache", expected
                    )

        # 3. sources chain
        errors: list[str] = []
        for source in self.sources:
            try:
                fragment = self._fetch(source, layer, expert, k)
            except FragmentVerificationError:
                raise
            except Exception as exc:  # noqa: BLE001 — try the next mirror
                errors.append(f"{getattr(source, 'name', source)}: {exc}")
                continue
            if fragment is not None:
                return fragment
            errors.append(f"{getattr(source, 'name', source)}: not found")

        raise FragmentUnavailableError(
            f"layer {layer} expert {expert} k{k}: no local segment dir has it "
            f"({[str(d) for d in self.local_dirs]}), sources exhausted "
            f"({errors or 'none configured'}); local encode from BF16 (T3) is "
            "not implemented in loader v2"
        )

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

    def _fetch(
        self, source: Any, layer: int, expert: int, k: int
    ) -> Fragment | None:
        index = self._remote_index(source, k)
        if index is None or str(layer) not in index:
            return None
        entry = _SegmentIndexEntry(index[str(layer)])

        expected = self._expected_sha(layer, k, expert)
        if expected is None:
            text = source.read_text(self._att_name(layer, k))
            if text is not None:
                self._atomic_write(
                    self._cache_path(
                        "attestations", f"layer-{layer:03d}.k{k}.jsonl"
                    ),
                    text.encode(),
                )
                self._expert_shas.pop((layer, k), None)
                expected = self._expected_sha(layer, k, expert)
        if expected is None and self.verify != "off":
            raise FragmentVerificationError(
                f"layer {layer} expert {expert} k{k}: source "
                f"{getattr(source, 'name', source)} has no attestation sha; "
                f"refusing unverified fetch (set {FQ_VERIFY_ENV}=off to allow)"
            )

        lo, hi = entry.expert_range(expert)
        payload = source.read_range(
            entry.file, entry.body_offset + lo, entry.body_offset + hi
        )
        self.stats["bytes_fetched"] += len(payload)
        if self.verify == "off" and expected is None:
            sha = hashlib.sha256(payload).hexdigest()
        else:
            sha = self._check_sha(
                payload,
                expected,
                f"fetched {entry.file} expert {expert} from "
                f"{getattr(source, 'name', source)}",
            )

        self._atomic_write(self._cache_fragment_path(sha), payload)
        tensors = self._remote_tables(source, entry).get(expert)
        if tensors is None:
            raise FragmentUnavailableError(
                f"segment {entry.file} header has no tensors for expert {expert}"
            )
        self.stats["fetched"] += 1
        return Fragment(layer, expert, k, payload, tensors, "fetched", sha)

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

        Tensors are zero-copy views over the fragment payload (mmap for
        local segments); downstream weight loaders copy them to device."""
        import torch

        fragment = self.resolve(layer, expert, k)
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
