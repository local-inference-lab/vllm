# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant atomic swap engine (M4) — 02-swap-engine.md row-write variant.

Moves experts between the K3 (tier0) and K4 (tier1) arms of a b12x
mixed-trellis MoE layer at runtime by rewriting slab rows, combined
rotation/suh/svh rows and the two routing maps in place — no reallocation,
no recompile, CUDA-graph-safe (maps are launch arguments read as data,
pre-M4 checks 1–2).

Design consequences bound by runs/pre-m4-checks/report.md:

1. Absence is expressed in ``global_to_combined`` only; this engine never
   creates holes (v1 swaps are total: occupancy == capacity) and validates
   rebuilt maps are a full permutation, fail-closed.
2. ``combine_trellis_rotations`` COPIES — rotation/suh/svh row writes target
   the COMBINED tensors at combined-slot indices (tier0 slots ``[0, t0)``,
   tier1 slots ``[t0, t0+t1)``). The per-tier source tensors are dead.
3. A tier swap changes BOTH experts' combined slots — both experts' slab
   rows and all four combined-table rows are rewritten at their NEW slots
   inside the quiesce window.
4. Descriptor words are exactly ``local`` or ``256 | local`` (validated on
   every rebuilt map before it is copied over the live one).
5. Stream discipline: every device write is issued on the caller's stream;
   nothing here creates device memory or reads device data inside the
   window (PERFORMANCE.md: pure copies from pre-allocated pinned staging).

Commit protocol (02 §Commit protocol, steps keyed for the T5 hook):
``slabs`` → ``rotations`` → ``maps`` → ``memo`` → ``persist``. Until the
map copy lands, the kernel routes every expert to its old intact encoding;
the overwritten rows are unreachable mid-window because the engine is
quiesced. Rollback is ``apply(plan.inverse())`` — sources are re-read from
the artifact pair (checkpoint-form tensors are freed post-prepare; the
artifact pair is the only source of truth for both directions).

Two properties T5 (torn-update fault injection) binds on top of that:

* **The flip is the only visibility point.** Host bookkeeping (the live
  tier orderings, the generation counter) commits with the map copy, in a
  ``finally``: an abort at ``memo``/``persist`` leaves device maps and
  in-memory orderings agreeing on the NEW membership, never a stale host
  view of freshly flipped maps.
* **Fail-atomic staging** (``stage(..., fail_atomic=True)``): the pre-swap
  encodings of both experts are staged alongside the new ones, so an abort
  *before* the flip restores the overwritten rows inside the same quiesce
  window. Without it a pre-flip abort leaves the layer genuinely torn (the
  displaced expert's slot holds its successor's bytes) — recoverable only by
  re-applying the plan or by reboot (slabs are a cache, never authoritative).

This module is import-light on purpose: no vllm imports, b12x is resolved
lazily through the ``build_maps_fn`` seam, segment files are parsed with
plain file IO. :class:`ResolverFragmentSource` adapts loader v2's
``fragments.FragmentResolver`` (multi-source, trust filtering, sha
verification, K-fallback ladder) to the same :class:`FragmentSource`
protocol by duck typing, so swaps stage from HF sources / trusted mirrors
with boot's verification and this module still imports standalone.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

K3, K4 = 3, 4
DESCRIPTOR_TIER_SHIFT = 8  # descriptor = (tier << 8) | local, single decode site

_PROJS = ("gate", "up", "down")
_COMPS = ("trellis", "suh", "svh", "mcg")
# safetensors dtype tag -> (torch dtype, itemsize)
_ST_DTYPES = {
    "I16": (torch.int16, 2),
    "F16": (torch.float16, 2),
    "I32": (torch.int32, 4),
}

COMMIT_STEPS = ("slabs", "rotations", "maps", "memo", "persist")


def tensor_name(layer: int, expert: int, proj: str, rank: int, comp: str) -> str:
    """Canonical fq-segment/1 tensor name (fq_repack's EXPERT_RE, inverted)."""
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}_proj.rank{rank}.{comp}"


def read_segment_header(path: Path) -> tuple[dict, int]:
    """Parse a safetensors header with plain file IO; return (header, body)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    return header, 8 + hlen


# --------------------------------------------------------------------- staging


class ExpertStage:
    """Pinned host staging for one expert's full payload at one bitrate.

    Holds exactly the rows the window writes: three native trellis slab rows,
    the two FC1 input rotations + FC2 output rotation (H-sized), and the
    pre-assembled combined intermediate-rotations row
    ``[gate_svh | up_svh | down_suh]`` (3I-sized, the prepare-time ABI order —
    exl3.py `_prepare_mixed_rank_sliced_weights`). ``mcg`` scalars are staged
    for validation only: the mixed kernel's codebook is layer-wide (r33 pins
    the MCG LUT ABI slots), so a fragment whose mcg differs is a corrupt or
    foreign encoding and staging refuses it.
    """

    def __init__(self, bits: int, hidden_size: int, intermediate_size: int,
                 *, pin_memory: bool):
        h16, i16 = hidden_size // 16, intermediate_size // 16
        words = 16 * int(bits)

        def buf(shape, dtype):
            return torch.empty(shape, dtype=dtype, pin_memory=pin_memory)

        self.bits = int(bits)
        self.trellis = {
            "gate": buf((h16, i16, words), torch.int16),
            "up": buf((h16, i16, words), torch.int16),
            "down": buf((i16, h16, words), torch.int16),
        }
        self.suh = {
            "gate": buf((hidden_size,), torch.float16),
            "up": buf((hidden_size,), torch.float16),
        }
        self.svh = {"down": buf((hidden_size,), torch.float16)}
        self.inter_rot = buf((3 * intermediate_size,), torch.float16)
        i = intermediate_size
        self.rot_slices = {
            ("gate", "svh"): self.inter_rot[0:i],
            ("up", "svh"): self.inter_rot[i:2 * i],
            ("down", "suh"): self.inter_rot[2 * i:3 * i],
        }
        self.mcg: dict[str, int] = {}

    def slab_rows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat sources for one expert's three slab rows (w13 gate, w13 up,
        w2 down) — commit-protocol step 1, in order."""
        return (self.trellis["gate"].view(-1), self.trellis["up"].view(-1),
                self.trellis["down"].view(-1))

    def rotation_rows(self) -> tuple[torch.Tensor, ...]:
        """Sources for the four COMBINED table rows of one expert, in commit
        order: ``rotations.intermediate``, ``gate_suh``, ``up_suh``,
        ``down_svh`` — step 2."""
        return (self.inter_rot, self.suh["gate"], self.suh["up"],
                self.svh["down"])

    def dest_tensor(self, proj: str, comp: str) -> torch.Tensor | None:
        """Host tensor a source must fill for (proj, comp); None for mcg."""
        if comp == "trellis":
            return self.trellis[proj]
        if comp == "suh" and proj in self.suh:
            return self.suh[proj]
        if comp == "svh" and proj in self.svh:
            return self.svh[proj]
        return self.rot_slices.get((proj, comp))

    @property
    def nbytes(self) -> int:
        tensors = [*self.trellis.values(), *self.suh.values(),
                   *self.svh.values(), self.inter_rot]
        return sum(t.numel() * t.element_size() for t in tensors)


class FragmentSource(Protocol):
    """Where expert payload bytes come from (NVMe segments now, HF ranges
    with loader v2). Implementations fill every ``dest.dest_tensor(proj,
    comp)`` buffer for the requested ``(layer, k, expert, rank)`` and set
    ``dest.mcg[proj]`` — host-side work only, always called pre-quiesce."""

    def read_expert(self, *, layer: int, k: int, expert: int, rank: int,
                    dest: ExpertStage) -> None:
        ...


class FragmentUnavailable(RuntimeError):
    """A source cannot supply this expert's payload at the requested K.

    Supply-side failure, not corruption: the fragment is missing from every
    configured source, its attestation is not trusted (10 §4), it failed sha
    verification, or the fallback ladder substituted a different K — which
    for a *promotion* means the K4 encode has not landed yet, so the
    promotion PENDS (07 §Pipeline 1: the swap list only includes promotions
    whose encodes have landed; pending promotions don't count against caps).

    :meth:`SwapEngine.stage` with ``on_unavailable="drop"`` turns this into a
    dropped pair instead of a failed interval. Structural failures
    (unregistered layer, bad residency, geometry mismatch, foreign mcg) are
    NOT droppable — they stay fatal, fail-closed.
    """


_UNAVAILABLE_ERROR_NAMES = frozenset((
    "FragmentUnavailable",        # this module
    "FragmentUnavailableError",   # fragments.py: no source could supply it
    "FragmentVerificationError",  # fragments.py: sha mismatch
))


def is_unavailable_error(exc: BaseException) -> bool:
    """Droppable supply-side failure (vs a fatal structural one)?

    Matched by class NAME over the exception's MRO: this module must import
    standalone (the GPU harness loads swap.py by path, where a relative
    import of fragments.py does not exist), and third-party sources can then
    signal "pend this promotion" without importing swap.py either.
    """
    return any(cls.__name__ in _UNAVAILABLE_ERROR_NAMES
               for cls in type(exc).__mro__)


class LocalSegmentSource:
    """fq-segment/1 reader over a local segment directory (fq_repack output:
    ``index-k{K}.json`` + ``layer-LLL.kK.safetensors``), plain file IO.

    Every tensor read is validated against the segment header (dtype/shape)
    and proven to fall inside the expert's ``index`` byte range — the same
    range a future HF source fetches with one ranged GET.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._indexes: dict[int, dict] = {}
        self._headers: dict[tuple[int, int], tuple[dict, int, Path]] = {}
        self._files: dict[Path, Any] = {}

    # ------------------------------------------------------------- internals

    def _index(self, k: int) -> dict:
        if k not in self._indexes:
            path = self.root / f"index-k{k}.json"
            self._indexes[k] = json.loads(path.read_text())
        return self._indexes[k]

    def _segment(self, layer: int, k: int) -> tuple[dict, int, Path, dict]:
        entry = self._index(k).get(str(layer))
        if entry is None:
            raise KeyError(f"layer {layer} not in index-k{k}.json")
        key = (layer, k)
        if key not in self._headers:
            path = self.root / entry["file"]
            header, body = read_segment_header(path)
            header.pop("__metadata__", None)
            if body != int(entry["body_offset"]):
                raise ValueError(
                    f"{path.name}: body offset {body} != indexed "
                    f"{entry['body_offset']} — index/segment mismatch")
            self._headers[key] = (header, body, path)
        header, body, path = self._headers[key]
        return header, body, path, entry["experts"]

    def _file(self, path: Path):
        f = self._files.get(path)
        if f is None:
            f = self._files[path] = open(path, "rb")
        return f

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()

    # -------------------------------------------------------------- protocol

    def read_expert(self, *, layer: int, k: int, expert: int, rank: int,
                    dest: ExpertStage) -> None:
        if dest.bits != k:
            raise ValueError(f"stage is K{dest.bits}, requested K{k}")
        header, body, path, experts = self._segment(layer, k)
        erange = experts.get(str(expert))
        if erange is None:
            raise KeyError(f"expert {expert} not indexed in {path.name}")
        e_start, e_end = int(erange[0]), int(erange[1])
        f = self._file(path)
        for proj in _PROJS:
            for comp in _COMPS:
                name = tensor_name(layer, expert, proj, rank, comp)
                info = header.get(name)
                if info is None:
                    raise KeyError(f"{path.name}: missing tensor {name}")
                a, b = (int(v) for v in info["data_offsets"])
                if not (e_start <= a and b <= e_end):
                    raise ValueError(
                        f"{path.name}: {name} range [{a},{b}) escapes the "
                        f"indexed expert range [{e_start},{e_end})")
                dtype, itemsize = _ST_DTYPES[info["dtype"]]
                if comp == "mcg":
                    f.seek(body + a)
                    (dest.mcg[proj],) = struct.unpack("<i", f.read(4))
                    continue
                t = dest.dest_tensor(proj, comp)
                assert t is not None
                if dtype != t.dtype or tuple(info["shape"]) != tuple(t.shape):
                    raise ValueError(
                        f"{path.name}: {name} is {info['dtype']}"
                        f"{info['shape']}, stage expects "
                        f"{t.dtype}{tuple(t.shape)}")
                nbytes = t.numel() * itemsize
                if b - a != nbytes:
                    raise ValueError(
                        f"{path.name}: {name} holds {b - a} bytes, "
                        f"expected {nbytes}")
                f.seek(body + a)
                view = t.numpy().view(np.uint8)
                got = f.readinto(view)
                if got != nbytes:
                    raise IOError(
                        f"{path.name}: short read on {name}: {got}/{nbytes}")


class ResolverFragmentSource:
    """:class:`FragmentSource` backed by loader v2's ``FragmentResolver``.

    Staging a swap then goes through boot's exact supply chain — local
    segment dirs → content-addressed cache → the manifest/``VLLM_FQ_SOURCES``
    chain of HF repos and trusted mirrors — with the same attestation trust
    filtering and per-fragment sha256 verification (implementation/10 §2/§4).
    The engine's per-tensor validation (dtype, shape, byte count, layer
    codebook) still runs on top of whatever the resolver hands back: one
    ranged GET per expert is sufficient exactly because every tensor of that
    expert is proven to live inside the indexed expert range.

    The resolver is duck-typed (``resolve`` + ``materialize``) so swap.py
    keeps its no-imports stance and tests can pass a fake.

    Substitution policy: a fragment the ``VLLM_FQ_K_FALLBACK`` ladder
    substituted at a different K cannot be written into a K``k`` slab row
    (different word count) *and* means the requested encode has not landed —
    :class:`FragmentUnavailable` is raised so the promotion pends (07 §1);
    the resolver has already queued the encode. ``allow_substituted=True``
    would only be meaningful with per-pair tier retargeting, which v1
    (same-cardinality K3/K4 swaps, D1) does not have.
    """

    def __init__(self, resolver: Any, *, allow_substituted: bool = False,
                 unavailable_errors: Sequence[type[BaseException]] = ()):
        self.resolver = resolver
        self.allow_substituted = bool(allow_substituted)
        self.unavailable_errors = tuple(unavailable_errors)
        self.reads = 0
        self.unavailable = 0

    def _unavailable(self, msg: str, cause: BaseException | None = None):
        self.unavailable += 1
        exc = FragmentUnavailable(msg)
        if cause is not None:
            exc.__cause__ = cause
        return exc

    def read_expert(self, *, layer: int, k: int, expert: int, rank: int,
                    dest: ExpertStage) -> None:
        if dest.bits != k:
            raise ValueError(f"stage is K{dest.bits}, requested K{k}")
        try:
            fragment = self.resolver.resolve(layer, expert, k)
        except Exception as exc:  # noqa: BLE001 — classified below
            if isinstance(exc, self.unavailable_errors) or is_unavailable_error(exc):
                raise self._unavailable(
                    f"L{layer}/e{expert} K{k}: resolver refused "
                    f"({type(exc).__name__}: {exc})", exc) from exc
            raise
        got_k = int(getattr(fragment, "k", k))
        if got_k != k and not self.allow_substituted:
            raise self._unavailable(
                f"L{layer}/e{expert} K{k}: resolver substituted K{got_k} "
                "(requested encode has not landed) — promotion pends")
        suffix = f".rank{rank}."
        pairs = self.resolver.materialize(
            fragment, name_filter=lambda n: suffix in n)
        tensors = dict(pairs)
        for proj in _PROJS:
            for comp in _COMPS:
                name = tensor_name(layer, expert, proj, rank, comp)
                src = tensors.get(name)
                if src is None:
                    raise KeyError(
                        f"fragment L{layer}/e{expert} K{got_k} "
                        f"({getattr(fragment, 'origin', '?')}) is missing "
                        f"{name} — {len(tensors)} rank{rank} tensors present")
                if comp == "mcg":
                    dest.mcg[proj] = int(src.reshape(-1)[0])
                    continue
                t = dest.dest_tensor(proj, comp)
                assert t is not None
                if src.dtype != t.dtype or tuple(src.shape) != tuple(t.shape):
                    raise ValueError(
                        f"fragment L{layer}/e{expert} K{got_k}: {name} is "
                        f"{src.dtype}{tuple(src.shape)}, stage expects "
                        f"{t.dtype}{tuple(t.shape)}")
                t.copy_(src)
        self.reads += 1


# ------------------------------------------------------------------ swap plan


class SwapPlan:
    """Ordered swap list ``[(layer, e_out, e_in), ...]`` between two
    same-cardinality memberships (policy.py conventions: ``e_out`` K4→K3,
    ``e_in`` K3→K4; deterministic order; ``inverse`` composes to identity)."""

    def __init__(self, swaps: Sequence[tuple[int, int, int]]):
        self.swaps = tuple(
            (int(l), int(e_out), int(e_in)) for l, e_out, e_in in swaps)
        seen: dict[int, set[int]] = {}
        for l, e_out, e_in in self.swaps:
            touched = seen.setdefault(l, set())
            for e in (e_out, e_in):
                if e in touched:
                    raise ValueError(
                        f"expert {e} appears twice in layer {l} — one swap "
                        "per expert per plan")
                touched.add(e)

    @classmethod
    def from_memberships(cls, old: Any, new: Any) -> "SwapPlan":
        """Diff two ``[L, E]`` membership arrays over {3, 4} into a plan.

        Same K4 cardinality per layer is required (D1 — project() first).
        Pairing is deterministic: leaving and entering experts are matched
        in ascending id order, layers ascending.
        """
        old = np.asarray(old)
        new = np.asarray(new)
        if old.shape != new.shape:
            raise ValueError(f"membership shapes differ: {old.shape} vs {new.shape}")
        swaps: list[tuple[int, int, int]] = []
        for l in range(old.shape[0]):
            outs = np.nonzero((old[l] == K4) & (new[l] == K3))[0]
            ins = np.nonzero((old[l] == K3) & (new[l] == K4))[0]
            if len(outs) != len(ins):
                raise ValueError(
                    f"layer {l}: cardinality changes ({len(outs)} out, "
                    f"{len(ins)} in) — same-cardinality plans only (D1)")
            swaps.extend((l, int(o), int(i)) for o, i in zip(outs, ins))
        return cls(swaps)

    @classmethod
    def from_policies(cls, old_doc: dict, new_doc: dict) -> "SwapPlan":
        """Diff two fq-policy/2 documents (store.py schema) into a plan."""
        old_bpe, new_bpe = old_doc["bits_per_expert"], new_doc["bits_per_expert"]
        if set(old_bpe) != set(new_bpe):
            raise ValueError("policies cover different layers")
        layers = sorted(old_bpe, key=int)
        old_m = np.array([old_bpe[l] for l in layers])
        new_m = np.array([new_bpe[l] for l in layers])
        base = cls.from_memberships(old_m, new_m)
        remap = {i: int(l) for i, l in enumerate(layers)}
        return cls([(remap[l], o, i) for l, o, i in base.swaps])

    def inverse(self) -> "SwapPlan":
        """Applying ``plan`` then ``plan.inverse()`` restores membership."""
        return SwapPlan(
            [(l, e_in, e_out) for (l, e_out, e_in) in reversed(self.swaps)])

    def layers(self) -> tuple[int, ...]:
        out: list[int] = []
        for l, _, _ in self.swaps:
            if l not in out:
                out.append(l)
        return tuple(out)

    def __len__(self) -> int:
        return len(self.swaps)

    def __iter__(self) -> Iterator[tuple[int, int, int]]:
        return iter(self.swaps)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SwapPlan) and self.swaps == other.swaps

    def __repr__(self) -> str:
        return f"SwapPlan({list(self.swaps)!r})"


# ---------------------------------------------------------------- layer state


@dataclass
class MixedLayerState:
    """The swap engine's view of one live mixed-trellis layer.

    ``tier0``/``tier1`` are prepared-tier objects whose slab tensors the
    kernel aliases (zero-copy, K6); ``rotations`` is the COMBINED table set
    (the only live copy — consequence 2); the two maps are the live launch
    arguments. ``tier0_globals``/``tier1_globals`` are the local→global slot
    orderings (b12x ``tier_ids``), mutated only by a committed apply().
    """

    tier0: Any
    tier1: Any
    rotations: Any
    global_to_combined: torch.Tensor
    descriptor_map: torch.Tensor
    tier0_globals: list[int]
    tier1_globals: list[int]
    mcg: int | None = None

    @classmethod
    def from_exl3_mixed_trellis(cls, mixed: Mapping[str, Any]) -> "MixedLayerState":
        """Adapt GG's ``layer.exl3_mixed_trellis`` dict (exl3.py)."""
        if tuple(mixed["tier_bits"]) != (K3, K4):
            raise ValueError(
                f"v1 swaps K3/K4 tiers only, got {tuple(mixed['tier_bits'])}")
        tier0, tier1 = mixed["tiers"]
        tier0_ids, tier1_ids = mixed["tier_ids"]
        return cls(
            tier0=tier0,
            tier1=tier1,
            rotations=mixed["rotations"],
            global_to_combined=mixed["global_to_combined"],
            descriptor_map=mixed["descriptor_map"],
            tier0_globals=list(tier0_ids),
            tier1_globals=list(tier1_ids),
        )

    @property
    def num_global_experts(self) -> int:
        return int(self.global_to_combined.numel())


@dataclass
class _StagedLayer:
    layer: int
    tier0_globals: list[int]  # post-swap orderings
    tier1_globals: list[int]
    g2c_host: torch.Tensor
    descriptor_host: torch.Tensor


@dataclass(frozen=True)
class DroppedPair:
    """A plan pair staging could not supply (``on_unavailable="drop"``).

    The pair is left entirely unapplied — both experts keep their current
    tier, so per-layer K4 cardinality is preserved (D1) and the promotion
    simply pends until its encode lands (07 §1)."""

    swap: tuple[int, int, int]
    expert: int
    k: int
    reason: str


@dataclass
class StagedBatch:
    """Everything apply() needs, fully resolved on the host pre-quiesce:
    three ordered (dst_view, src_pinned) op lists (commit-protocol steps
    1–3) plus the post-swap orderings to commit into the layer states.

    ``plan`` is the EFFECTIVE plan (``requested_plan`` minus ``dropped``);
    callers building the policy document to persist must derive it from
    ``plan``, never from what they asked for. ``undo_ops`` is the fail-atomic
    restore batch (pre-swap rows + maps) apply() replays inside the window if
    it aborts before the visibility flip."""

    plan: SwapPlan
    slab_ops: list[tuple[torch.Tensor, torch.Tensor]]
    rotation_ops: list[tuple[torch.Tensor, torch.Tensor]]
    map_ops: list[tuple[torch.Tensor, torch.Tensor]]
    staged_layers: list[_StagedLayer]
    bytes_h2d: int
    stage_seconds: float
    mcg: int | None
    requested_plan: SwapPlan | None = None
    dropped: tuple[DroppedPair, ...] = ()
    undo_ops: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    restored: bool = False

    @property
    def fail_atomic(self) -> bool:
        return self.undo_ops is not None


@dataclass
class ApplyReport:
    pairs: int
    layers: int
    bytes_h2d: int
    stage_seconds: float
    window_seconds: float
    generation: int
    mcg: int | None = None
    committed_policy_hash: str | None = None
    dropped: tuple[DroppedPair, ...] = field(default_factory=tuple)


# --------------------------------------------------------------- swap engine


def _default_build_maps(tier0_ids, tier1_ids, *, device):
    from b12x.moe._shared.kernels.w4a16.mixed_trellis import build_tiered_maps
    return build_tiered_maps(tier0_ids, tier1_ids, device=device)


class SwapEngine:
    """Applies swap plans to live mixed-trellis layers (02 commit protocol).

    All host work (fragment IO, slot resolution, map building, device-view
    resolution) happens in :meth:`stage`, callable from a background thread
    before the pause. :meth:`apply` runs the six-step commit inside the
    caller-provided quiesce window: pure ``copy_`` from pre-allocated pinned
    staging on the caller's stream — zero device allocation, zero host reads
    of device data.

    Not thread-safe; one staged batch is live at a time (staging slots are
    reused).
    """

    def __init__(
        self,
        layers: Mapping[int, MixedLayerState],
        source: FragmentSource,
        *,
        hidden_size: int,
        intermediate_size: int,
        tier_bits: tuple[int, int] = (K3, K4),
        rank: int = 0,
        max_pairs: int = 8,
        pin_memory: bool | None = None,
        build_maps_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
        expected_mcg: int | None = None,
    ):
        self.layers = dict(layers)
        self.source = source
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.tier_bits = (int(tier_bits[0]), int(tier_bits[1]))
        self.rank = int(rank)
        self.max_pairs = int(max_pairs)
        self.expected_mcg = expected_mcg
        self.generation = 0
        self._build_maps = build_maps_fn or _default_build_maps
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self._pin_memory = bool(pin_memory)

        for layer_id, state in self.layers.items():
            self._validate_layer(layer_id, state)

        # Pre-allocated pinned staging: one (K3, K4) stage pair per swap
        # pair, one host map pair per registered layer (map contents are
        # rebuilt per layer, not per pair).
        self._stages = [
            (ExpertStage(self.tier_bits[0], self.hidden_size,
                         self.intermediate_size, pin_memory=pin_memory),
             ExpertStage(self.tier_bits[1], self.hidden_size,
                         self.intermediate_size, pin_memory=pin_memory))
            for _ in range(self.max_pairs)
        ]
        self._map_stages = {
            layer_id: self._new_map_stage(state)
            for layer_id, state in self.layers.items()
        }
        # Fail-atomic staging (T5): the pre-swap encodings + pre-swap maps.
        # Allocated on first use — a plan staged without fail_atomic keeps
        # the 02 §Why-row-writes pinned budget (max_pairs x 7.875 MiB).
        self._undo_stages: list[tuple[ExpertStage, ExpertStage]] | None = None
        self._undo_map_stages: dict[
            int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _new_map_stage(self, state: MixedLayerState
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty(state.num_global_experts, dtype=torch.int32,
                        pin_memory=self._pin_memory),
            torch.empty(state.num_global_experts, dtype=torch.int32,
                        pin_memory=self._pin_memory),
        )

    def _undo_stage_pair(self, index: int) -> tuple[ExpertStage, ExpertStage]:
        """(K3, K4) pinned stages holding the PRE-swap payloads of pair
        ``index`` — the restore sources apply() replays on a pre-flip abort."""
        if self._undo_stages is None:
            self._undo_stages = [
                (ExpertStage(self.tier_bits[0], self.hidden_size,
                             self.intermediate_size,
                             pin_memory=self._pin_memory),
                 ExpertStage(self.tier_bits[1], self.hidden_size,
                             self.intermediate_size,
                             pin_memory=self._pin_memory))
                for _ in range(self.max_pairs)
            ]
        return self._undo_stages[index]

    def _undo_map_stage(self, layer_id: int
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        stage = self._undo_map_stages.get(layer_id)
        if stage is None:
            stage = self._undo_map_stages[layer_id] = self._new_map_stage(
                self.layers[layer_id])
        return stage

    # ---------------------------------------------------------- validation

    def _validate_layer(self, layer_id: int, state: MixedLayerState) -> None:
        h16 = self.hidden_size // 16
        i16 = self.intermediate_size // 16
        t0, t1 = len(state.tier0_globals), len(state.tier1_globals)
        combined = t0 + t1
        if sorted(state.tier0_globals + state.tier1_globals) != list(
                range(state.num_global_experts)):
            raise ValueError(
                f"layer {layer_id}: tier orderings are not a partition of "
                f"[0, {state.num_global_experts}) — v1 requires occupancy == "
                "capacity (holes go through global_to_combined, not here)")
        for tier_idx, (tier, bits, count) in enumerate(
                ((state.tier0, self.tier_bits[0], t0),
                 (state.tier1, self.tier_bits[1], t1))):
            if int(tier.num_experts) != count:
                raise ValueError(
                    f"layer {layer_id} tier{tier_idx}: prepared "
                    f"num_experts={int(tier.num_experts)} != "
                    f"{count} tier globals")
            w13_words = 2 * count * h16 * i16 * 16 * bits
            w2_words = count * i16 * h16 * 16 * bits
            if tier.w13.view(torch.int16).numel() != w13_words:
                raise ValueError(
                    f"layer {layer_id} tier{tier_idx}: w13 slab holds "
                    f"{tier.w13.view(torch.int16).numel()} i16 words, "
                    f"geometry says {w13_words}")
            if tier.w2.view(torch.int16).numel() != w2_words:
                raise ValueError(
                    f"layer {layer_id} tier{tier_idx}: w2 slab holds "
                    f"{tier.w2.view(torch.int16).numel()} i16 words, "
                    f"geometry says {w2_words}")
        rot = state.rotations
        expected = {
            "intermediate": (combined, 3 * self.intermediate_size),
            "gate_suh": (combined, self.hidden_size),
            "up_suh": (combined, self.hidden_size),
            "down_svh": (combined, self.hidden_size),
        }
        for name, shape in expected.items():
            t = getattr(rot, name)
            if tuple(t.shape) != shape:
                raise ValueError(
                    f"layer {layer_id}: combined rotations.{name} is "
                    f"{tuple(t.shape)}, expected {shape} — broadcast "
                    "suh/svh layouts are out of v1 swap scope "
                    "(pre-M4 consequence 3)")
        if int(state.descriptor_map.numel()) != combined:
            raise ValueError(
                f"layer {layer_id}: descriptor_map has "
                f"{int(state.descriptor_map.numel())} slots, expected {combined}")

    def _validate_host_maps(self, layer_id: int, t0n: int,
                            g2c: torch.Tensor, desc: torch.Tensor) -> None:
        """Fail closed on consequences 1 and 4 before touching live maps."""
        combined = int(desc.numel())
        expect = torch.cat((
            torch.arange(t0n, dtype=torch.int32),
            (1 << DESCRIPTOR_TIER_SHIFT)
            | torch.arange(combined - t0n, dtype=torch.int32),
        ))
        if not torch.equal(desc, expect):
            raise ValueError(
                f"layer {layer_id}: rebuilt descriptor_map violates "
                "local|256+local encoding")
        if not torch.equal(torch.sort(g2c).values,
                           torch.arange(combined, dtype=torch.int32)):
            raise ValueError(
                f"layer {layer_id}: rebuilt global_to_combined is not a "
                "permutation — a hole here silently blends garbage "
                "(pre-M4 consequence 1)")

    # -------------------------------------------------------------- staging

    def _stage_maps_for_layer(
        self, layer_id: int, tier0_globals: list[int],
        tier1_globals: list[int],
        into: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> _StagedLayer:
        g2c_cpu, desc_cpu = self._build_maps(
            tier0_globals, tier1_globals, device=torch.device("cpu"))
        self._validate_host_maps(layer_id, len(tier0_globals), g2c_cpu, desc_cpu)
        g2c_host, desc_host = into or self._map_stages[layer_id]
        g2c_host.copy_(g2c_cpu)
        desc_host.copy_(desc_cpu)
        return _StagedLayer(layer_id, list(tier0_globals), list(tier1_globals),
                            g2c_host, desc_host)

    @staticmethod
    def _slab_rows(tier, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        w13 = tier.w13.view(torch.int16).view(2, count, -1)
        w2 = tier.w2.view(torch.int16).view(count, -1)
        return w13, w2

    def stage(self, plan: SwapPlan, *, fail_atomic: bool = False,
              on_unavailable: str = "raise") -> StagedBatch:
        """Resolve a plan into device-view/pinned-source op lists (host-only,
        pre-quiesce: fragment IO, slot resolution, map rebuild).

        ``fail_atomic``: also stage both experts' PRE-swap encodings and the
        pre-swap maps, so :meth:`apply` can restore the layer inside the
        quiesce window if it aborts before the visibility flip (T5). Costs a
        second read of each expert and doubles pinned staging.

        ``on_unavailable="drop"``: a pair whose fragments no source can
        supply (missing / untrusted / unverified / substituted-K, i.e.
        :func:`is_unavailable_error`) is dropped from the batch instead of
        failing the interval — the promotion pends (07 §1). Both experts keep
        their tier, so cardinality is preserved. The surviving pairs are
        :attr:`StagedBatch.plan`; build the policy document to persist from
        that, not from the requested plan.
        """
        t_start = time.perf_counter()
        if on_unavailable not in ("raise", "drop"):
            raise ValueError(
                f"on_unavailable must be raise|drop, got {on_unavailable!r}")
        if len(plan) > self.max_pairs:
            raise ValueError(
                f"plan has {len(plan)} pairs, staging holds {self.max_pairs}")
        k3_bits, k4_bits = self.tier_bits
        slab_ops: list[tuple[torch.Tensor, torch.Tensor]] = []
        rotation_ops: list[tuple[torch.Tensor, torch.Tensor]] = []
        undo_ops: list[tuple[torch.Tensor, torch.Tensor]] | None = (
            [] if fail_atomic else None)
        dropped: list[DroppedPair] = []
        applied: list[tuple[int, int, int]] = []
        mcg_seen: int | None = self.expected_mcg
        # Scratch orderings so multi-pair layers resolve slots sequentially
        # without touching live state until commit; ``pre`` keeps the
        # pre-swap orderings for the fail-atomic map restore.
        orderings: dict[int, tuple[list[int], list[int]]] = {}
        pre_orderings: dict[int, tuple[list[int], list[int]]] = {}
        touched: list[int] = []
        pair_idx = 0

        for layer_id, e_out, e_in in plan:
            state = self.layers.get(layer_id)
            if state is None:
                raise KeyError(f"layer {layer_id} is not registered")
            if layer_id not in orderings:
                orderings[layer_id] = (list(state.tier0_globals),
                                       list(state.tier1_globals))
                pre_orderings[layer_id] = (list(state.tier0_globals),
                                           list(state.tier1_globals))
            t0_ids, t1_ids = orderings[layer_id]
            if e_out not in t1_ids or e_in not in t0_ids:
                raise ValueError(
                    f"invalid swap ({layer_id}, {e_out}, {e_in}): e_out must "
                    f"be resident K{k4_bits}, e_in resident K{k3_bits}")

            # Fragment IO first: a pair that cannot be supplied leaves no
            # trace (orderings are only mutated once every read landed).
            out_stage, in_stage = self._stages[pair_idx]
            reads = [(k3_bits, e_out, out_stage), (k4_bits, e_in, in_stage)]
            undo_k3 = undo_k4 = None
            if fail_atomic:
                undo_k3, undo_k4 = self._undo_stage_pair(pair_idx)
                # Pre-swap content of the two destination slots.
                reads += [(k4_bits, e_out, undo_k4), (k3_bits, e_in, undo_k3)]
            drop: DroppedPair | None = None
            for k, expert, dest in reads:
                try:
                    self.source.read_expert(layer=layer_id, k=k, expert=expert,
                                            rank=self.rank, dest=dest)
                except Exception as exc:  # noqa: BLE001 — classified here
                    if on_unavailable == "drop" and is_unavailable_error(exc):
                        drop = DroppedPair(
                            swap=(layer_id, e_out, e_in), expert=expert, k=k,
                            reason=f"{type(exc).__name__}: {exc}")
                        break
                    raise
            if drop is not None:
                logger.warning(
                    "FQ swap L%d e%d->K%d/e%d->K%d dropped: %s",
                    layer_id, e_out, k3_bits, e_in, k4_bits, drop.reason)
                dropped.append(drop)
                continue

            # e_in inherits e_out's tier1 slot and vice versa: every other
            # expert's slot (and slab row) is untouched.
            slot1 = t1_ids.index(e_out)
            slot0 = t0_ids.index(e_in)
            t1_ids[slot1] = e_in
            t0_ids[slot0] = e_out
            t0n = len(t0_ids)
            applied.append((layer_id, e_out, e_in))
            if layer_id not in touched:
                touched.append(layer_id)
            pair_idx += 1

            for stage, expert in ((out_stage, e_out), (in_stage, e_in),
                                  *(((undo_k4, e_out), (undo_k3, e_in))
                                    if fail_atomic else ())):
                for proj, value in stage.mcg.items():
                    if mcg_seen is None:
                        mcg_seen = value
                    elif value != mcg_seen:
                        raise ValueError(
                            f"layer {layer_id} expert {expert} {proj}: mcg "
                            f"{value} != layer codebook {mcg_seen} — foreign "
                            "encoding, refusing to stage")
                if state.mcg is not None and mcg_seen not in (None, state.mcg):
                    raise ValueError(
                        f"layer {layer_id}: staged mcg {mcg_seen} != layer "
                        f"mcg {state.mcg}")

            # Step-1 ops: slab rows for BOTH experts at their NEW slots.
            # The undo op for a destination is the SAME view fed the expert
            # that occupied it pre-swap.
            for tier, count, slot, stage, undo in (
                    (state.tier1, len(t1_ids), slot1, in_stage, undo_k4),
                    (state.tier0, t0n, slot0, out_stage, undo_k3)):
                w13, w2 = self._slab_rows(tier, count)
                dsts = (w13[0, slot], w13[1, slot], w2[slot])
                slab_ops.extend(zip(dsts, stage.slab_rows()))
                if undo_ops is not None:
                    undo_ops.extend(zip(dsts, undo.slab_rows()))

            # Step-2 ops: COMBINED rotation/suh/svh rows at NEW combined
            # slots (tier0 local -> slot, tier1 local -> t0 + slot).
            rot = state.rotations
            for combined_slot, stage, undo in ((t0n + slot1, in_stage, undo_k4),
                                               (slot0, out_stage, undo_k3)):
                dsts = (rot.intermediate[combined_slot],
                        rot.gate_suh[combined_slot],
                        rot.up_suh[combined_slot],
                        rot.down_svh[combined_slot])
                rotation_ops.extend(zip(dsts, stage.rotation_rows()))
                if undo_ops is not None:
                    undo_ops.extend(zip(dsts, undo.rotation_rows()))

        staged_layers = [
            self._stage_maps_for_layer(layer_id, *orderings[layer_id])
            for layer_id in touched
        ]
        map_ops: list[tuple[torch.Tensor, torch.Tensor]] = []
        for staged in staged_layers:
            state = self.layers[staged.layer]
            map_ops.append((state.global_to_combined, staged.g2c_host))
            map_ops.append((state.descriptor_map, staged.descriptor_host))
        if undo_ops is not None:
            for layer_id in touched:
                state = self.layers[layer_id]
                pre = self._stage_maps_for_layer(
                    layer_id, *pre_orderings[layer_id],
                    self._undo_map_stage(layer_id))
                undo_ops.append((state.global_to_combined, pre.g2c_host))
                undo_ops.append((state.descriptor_map, pre.descriptor_host))

        bytes_h2d = sum(
            src.numel() * src.element_size()
            for _, src in (*slab_ops, *rotation_ops, *map_ops))
        return StagedBatch(
            plan=SwapPlan(applied) if dropped else plan,
            slab_ops=slab_ops,
            rotation_ops=rotation_ops,
            map_ops=map_ops,
            staged_layers=staged_layers,
            bytes_h2d=bytes_h2d,
            stage_seconds=time.perf_counter() - t_start,
            mcg=mcg_seen,
            requested_plan=plan,
            dropped=tuple(dropped),
            undo_ops=undo_ops,
        )

    # ---------------------------------------------------------------- apply

    def apply(
        self,
        plan: SwapPlan | None = None,
        *,
        quiesce: AbstractContextManager,
        staged: StagedBatch | None = None,
        stream: torch.cuda.Stream | None = None,
        memo_hook: Callable[[], None] | None = None,
        policy_store: Any | None = None,
        policy_doc: dict | None = None,
        policy_num_experts: int | None = None,
        step_hook: Callable[[str], None] | None = None,
    ) -> ApplyReport:
        """Run the commit protocol for ``plan`` (or a pre-``stage``-d batch).

        ``quiesce`` is the caller's engine-drain context manager (pass
        ``contextlib.nullcontext()`` only when nothing can be replaying).
        Device work is ordered on ``stream`` (default: current stream) —
        the same stream replay uses, per consequence 5. ``memo_hook`` nulls
        the FusedMoEQuantConfig memo (step 4); ``policy_store``/
        ``policy_doc`` persist the committed policy (step 5). ``step_hook``
        is the T5 torn-update instrumentation seam (called after each of
        COMMIT_STEPS; raising aborts between steps).

        Abort semantics (T5), for an exception from any step or hook:

        * **before the flip** — if the batch was staged ``fail_atomic``, the
          pre-swap rows and maps are restored inside the same quiesce window
          (``staged.restored``) and the layer resumes fully-old; otherwise
          the layer is left torn and the caller must re-apply the plan (roll
          forward) or restart (slabs are a cache, rebuilt from artifacts).
        * **at or after the flip** — the swap is committed: live orderings
          and the generation counter advance with the maps (``finally``), so
          the layer resumes fully-new and rollback is the normal
          ``apply(plan.inverse())``. A failed ``memo_hook`` still leaves a
          stale FusedMoEQuantConfig memo — the caller must retry it.

        Rollback: ``apply(plan.inverse(), ...)`` — fragments are re-read
        from the artifact pair by :meth:`stage`.
        """
        if staged is None:
            if plan is None:
                raise ValueError("apply() needs a plan or a staged batch")
            staged = self.stage(plan)
        elif plan is not None and plan not in (staged.plan,
                                               staged.requested_plan):
            raise ValueError("staged batch does not match plan")

        stream_ctx: AbstractContextManager = (
            torch.cuda.stream(stream) if stream is not None else nullcontext())
        committed_hash: str | None = None
        flipped = False
        t_window = time.perf_counter()
        with quiesce:
            try:
                with stream_ctx:
                    for dst, src in staged.slab_ops:  # step 1
                        dst.copy_(src, non_blocking=True)
                    if step_hook is not None:
                        step_hook("slabs")
                    for dst, src in staged.rotation_ops:  # step 2
                        dst.copy_(src, non_blocking=True)
                    if step_hook is not None:
                        step_hook("rotations")
                    for dst, src in staged.map_ops:  # step 3 — visibility flip
                        dst.copy_(src, non_blocking=True)
                    flipped = True
                    if step_hook is not None:
                        step_hook("maps")
            except BaseException:
                if not flipped and staged.undo_ops is not None:
                    with stream_ctx:
                        for dst, src in staged.undo_ops:
                            dst.copy_(src, non_blocking=True)
                    staged.restored = True
                    logger.warning(
                        "FQ swap aborted before the visibility flip; %d "
                        "pre-swap rows/maps restored inside the quiesce "
                        "window (layer state is fully-old)",
                        len(staged.undo_ops))
                elif not flipped:
                    logger.error(
                        "FQ swap aborted before the visibility flip WITHOUT "
                        "fail-atomic staging — slab rows are torn; re-apply "
                        "the plan or restart (slabs are a cache)")
                raise
            finally:
                # The flip is the only visibility point: live orderings and
                # the generation counter commit WITH the maps, so an abort in
                # step 4/5 cannot leave a stale host view of flipped maps.
                if flipped:
                    self.generation += 1
                    for staged_layer in staged.staged_layers:
                        state = self.layers[staged_layer.layer]
                        state.tier0_globals[:] = staged_layer.tier0_globals
                        state.tier1_globals[:] = staged_layer.tier1_globals
            if memo_hook is not None:  # step 4
                memo_hook()
            if step_hook is not None:
                step_hook("memo")
            if policy_store is not None and policy_doc is not None:  # step 5
                committed_hash = policy_store.commit(
                    policy_doc, num_experts=policy_num_experts)
            if step_hook is not None:
                step_hook("persist")
        window_seconds = time.perf_counter() - t_window

        return ApplyReport(
            pairs=len(staged.plan),
            layers=len(staged.staged_layers),
            bytes_h2d=staged.bytes_h2d,
            stage_seconds=staged.stage_seconds,
            window_seconds=window_seconds,
            generation=self.generation,
            mcg=staged.mcg,
            committed_policy_hash=committed_hash,
            dropped=staged.dropped,
        )

    # ---------------------------------------------------------- map rebuild

    def rebuild_maps(
        self,
        layer_id: int,
        tier0_globals: Sequence[int],
        tier1_globals: Sequence[int],
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Rewrite one layer's map CONTENTS in place (T3 path; also the
        commit-step-3 mechanism apply() uses). Slab/rotation rows are not
        touched — callers own the membership/content consistency argument
        (inside apply() it is the commit ordering)."""
        state = self.layers[layer_id]
        staged = self._stage_maps_for_layer(
            layer_id, list(tier0_globals), list(tier1_globals))
        stream_ctx: AbstractContextManager = (
            torch.cuda.stream(stream) if stream is not None else nullcontext())
        with stream_ctx:
            state.global_to_combined.copy_(staged.g2c_host, non_blocking=True)
            state.descriptor_map.copy_(staged.descriptor_host, non_blocking=True)
        state.tier0_globals[:] = staged.tier0_globals
        state.tier1_globals[:] = staged.tier1_globals
