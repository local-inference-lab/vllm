# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy-encode queue + drain worker (Progressive Tensors T3 plumbing).

Boot must never block on a missing fragment: when the FragmentResolver
cannot resolve (or refuses to trust) a ``(layer, expert, K)`` fragment it
either substitutes a fallback K (``VLLM_FQ_K_FALLBACK``) or fails — in both
cases it *enqueues* the miss here so an out-of-band worker can encode the
missing fragment from BF16 + capture and publish it into a segment dir.

Queue: a persisted JSONL file (``VLLM_FQ_ENCODE_QUEUE``, default
``<VLLM_FQ_CACHE>/encode-queue.jsonl``) of
``{layer, expert, k, reason, requested_utc}`` entries, deduplicated by
``(layer, expert, k)``.  Appends are O(1) single-line writes; the resolver
never waits on encode work.

Worker CLI::

    python -m vllm.model_executor.layers.quantization.exl3_fungible.lazy_encode \
        --drain [--execute] [--queue PATH] [--bf16-dir DIR] [--capture-dir DIR] \
        [--encoder-cmd TEMPLATE] [--limit N]

``--drain`` alone runs in DRY-RUN mode: each entry is validated against the
available BF16 checkpoint (``VLLM_FQ_BF16_DIR`` — expert weights present in
``model.safetensors.index.json``) and Hessian capture
(``VLLM_FQ_CAPTURE_DIR`` — ``layer_{L:03d}/`` payload dir), and the encoder
command that WOULD run is printed.  ``--execute`` actually shells out to the
encoder driver per entry (GPU work) and removes completed entries from the
queue.

Encoder command template (``VLLM_FQ_ENCODER_CMD`` / ``--encoder-cmd``),
``str.format`` placeholders ``{layer} {layer03} {expert} {k} {bf16_dir}
{capture_dir}``.  The default targets the existing sha-pinned encoder driver
(tested-by-dryrun only — never executed by the resolver)::

    python .../tools/fruit-encoder/fruit_encode_driver.py \
        --encode --bits {k} --layers {layer} --src {bf16_dir} \
        --capture-dir {capture_dir} --workers 1 --gpus 0

The driver's unit of work is one layer (it encodes every expert of
``--layers {layer}`` — a superset of the queued expert; correct, amortized
across all queued misses of that layer).  The ``{expert}`` placeholder is
provided for future single-expert drivers.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

FQ_ENCODE_QUEUE_ENV = "VLLM_FQ_ENCODE_QUEUE"
FQ_BF16_DIR_ENV = "VLLM_FQ_BF16_DIR"
FQ_CAPTURE_DIR_ENV = "VLLM_FQ_CAPTURE_DIR"
FQ_ENCODER_CMD_ENV = "VLLM_FQ_ENCODER_CMD"

DEFAULT_QUEUE_NAME = "encode-queue.jsonl"

DEFAULT_ENCODER_DRIVER = (
    "/home/mbelleau/protensors-work/vllm-voipmonitor/research/fungible-quant/"
    "tools/fruit-encoder/fruit_encode_driver.py"
)
DEFAULT_ENCODER_CMD = (
    "python " + DEFAULT_ENCODER_DRIVER + " --encode --bits {k} "
    "--layers {layer} --src {bf16_dir} --capture-dir {capture_dir} "
    "--workers 1 --gpus 0"
)

_PROJS = ("gate_proj", "up_proj", "down_proj")


class EncodeQueue:
    """Persisted JSONL queue of lazy-encode requests, dedup by (L, E, K).

    Loading is total: the queue is telemetry written by a live serve with
    plain appends, so it can be found torn (a half-written last line after
    a kill -9), truncated, byte-garbage, or replaced by a directory. None
    of that may raise — an unreadable queue degrades to an empty one and
    the serve keeps running. ``corrupt_lines`` counts what was skipped so
    the drain worker can report it.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # insertion-ordered: position in this dict == queue position - 1
        self._entries: dict[tuple[int, int, int], dict] = {}
        self.corrupt_lines = 0
        for line in self._read_lines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                self.corrupt_lines += 1
                continue
            key = self._key(entry)
            if key is None:
                self.corrupt_lines += 1
                continue
            if key not in self._entries:
                self._entries[key] = entry
        if self.corrupt_lines:
            print(
                f"warning: encode queue {self.path}: skipped "
                f"{self.corrupt_lines} unparseable line(s)",
                file=sys.stderr,
            )

    def _read_lines(self) -> list[str]:
        """Queue file lines, or [] for anything unreadable."""
        try:
            if not self.path.is_file():
                return []
            # errors="replace": a torn append can leave a partial UTF-8
            # sequence; that line must be skipped, not kill the reader.
            return self.path.read_text(errors="replace").splitlines()
        except OSError:
            print(
                f"warning: encode queue {self.path} unreadable — "
                "continuing with an empty queue",
                file=sys.stderr,
            )
            return []

    @staticmethod
    def _key(entry: dict) -> tuple[int, int, int] | None:
        try:
            return (int(entry["layer"]), int(entry["expert"]), int(entry["k"]))
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def from_env(
        cls,
        environ: dict[str, str] | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> "EncodeQueue":
        environ = os.environ if environ is None else environ
        path = environ.get(FQ_ENCODE_QUEUE_ENV)
        if not path:
            base = (
                cache_dir
                or environ.get("VLLM_FQ_CACHE")
                or os.path.expanduser("~/.cache/vllm/fq")
            )
            path = Path(base) / DEFAULT_QUEUE_NAME
        return cls(path)

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> list[dict]:
        return [dict(e) for e in self._entries.values()]

    def position(self, layer: int, expert: int, k: int) -> int | None:
        """1-based queue position of an entry, or None."""
        key = (int(layer), int(expert), int(k))
        for i, existing in enumerate(self._entries, 1):
            if existing == key:
                return i
        return None

    def enqueue(
        self, layer: int, expert: int, k: int, reason: str
    ) -> tuple[int, bool]:
        """Append (dedup by key). Returns (1-based position, created)."""
        key = (int(layer), int(expert), int(k))
        if key in self._entries:
            return self.position(*key) or len(self._entries), False
        entry = {
            "layer": key[0],
            "expert": key[1],
            "k": key[2],
            "reason": str(reason),
            "requested_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
        self._entries[key] = entry
        return len(self._entries), True

    def rewrite(self, remaining: list[dict]) -> None:
        """Atomically replace the queue contents (drain bookkeeping)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(
            "".join(
                json.dumps(e, separators=(",", ":")) + "\n" for e in remaining
            )
        )
        os.replace(tmp, self.path)
        self._entries = {
            key: e
            for e in remaining
            if (key := self._key(e)) is not None
        }


# --------------------------------------------------------------- validation


def _bf16_check(
    bf16_dir: str | Path | None, layer: int, expert: int
) -> tuple[bool, str]:
    if not bf16_dir:
        return False, f"bf16-dir unset (set {FQ_BF16_DIR_ENV})"
    d = Path(bf16_dir)
    if not d.is_dir():
        return False, f"bf16-dir missing: {d}"
    index_path = d / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text()).get(
                "weight_map", {}
            )
        except ValueError:
            return False, f"bf16 index unreadable: {index_path}"
        missing = [
            proj
            for proj in _PROJS
            if f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
            not in weight_map
        ]
        if missing:
            return False, f"bf16 index lacks L{layer}/e{expert} {missing}"
        return True, "bf16=ok(index)"
    if any(d.glob("*.safetensors")):
        return True, "bf16=ok(dir)"
    return False, f"bf16-dir has no safetensors: {d}"


def _capture_check(
    capture_dir: str | Path | None, layer: int
) -> tuple[bool, str]:
    if not capture_dir:
        return False, f"capture-dir unset (set {FQ_CAPTURE_DIR_ENV})"
    d = Path(capture_dir)
    if not d.is_dir():
        return False, f"capture-dir missing: {d}"
    sub = d / f"layer_{layer:03d}"
    if sub.is_dir() and any(sub.iterdir()):
        return True, f"capture=ok({sub.name})"
    return False, f"capture missing layer_{layer:03d}"


class _LenientFields(dict):
    """``str.format_map`` mapping that leaves unknown fields untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def render_cmd(
    template: str,
    entry: dict,
    bf16_dir: str | Path | None,
    capture_dir: str | Path | None,
) -> str:
    """Render the encoder command for one queue entry.

    Lenient on purpose: an operator template naming a placeholder we do not
    provide renders it back verbatim instead of raising, so one typo in
    ``VLLM_FQ_ENCODER_CMD`` cannot abort a whole drain (the printed command
    still shows the mistake)."""
    layer = int(entry["layer"])
    fields = _LenientFields(
        layer=layer,
        layer03=f"{layer:03d}",
        expert=int(entry["expert"]),
        k=int(entry["k"]),
        bf16_dir=bf16_dir or "",
        capture_dir=capture_dir or "",
    )
    try:
        return template.format_map(fields)
    except (IndexError, ValueError) as exc:
        # positional {} / malformed format spec: unusable but not fatal
        return f"<unrenderable encoder template: {type(exc).__name__}: {exc}>"


# -------------------------------------------------------------------- drain


def drain(
    queue: EncodeQueue,
    *,
    dry_run: bool = True,
    bf16_dir: str | Path | None = None,
    capture_dir: str | Path | None = None,
    encoder_cmd: str | None = None,
    environ: dict[str, str] | None = None,
    limit: int | None = None,
    out: Callable[[str], None] = print,
    runner: Callable[..., Any] = subprocess.run,
) -> int:
    """Process the queue. DRY-RUN validates entries + prints the command;
    ``dry_run=False`` executes the encoder per entry and removes completed
    entries.  Returns 0 when every processed entry validated/succeeded."""
    environ = os.environ if environ is None else environ
    bf16_dir = bf16_dir or environ.get(FQ_BF16_DIR_ENV)
    capture_dir = capture_dir or environ.get(FQ_CAPTURE_DIR_ENV)
    template = (
        encoder_cmd or environ.get(FQ_ENCODER_CMD_ENV) or DEFAULT_ENCODER_CMD
    )

    entries = queue.entries()
    if limit is not None:
        entries = entries[: int(limit)]
    if not entries:
        out(f"encode queue empty: {queue.path}")
        return 0

    failed = 0
    done: list[dict] = []
    for i, entry in enumerate(entries, 1):
        try:
            tag = (
                f"#{i} L{entry['layer']}/e{entry['expert']} K{entry['k']} "
                f"reason={entry.get('reason', '?')}"
            )
            ok_bf16, msg_bf16 = _bf16_check(
                bf16_dir, int(entry["layer"]), int(entry["expert"])
            )
            ok_cap, msg_cap = _capture_check(capture_dir, int(entry["layer"]))
            cmd = render_cmd(template, entry, bf16_dir, capture_dir)
        except Exception as exc:  # noqa: BLE001 — one entry, not the drain
            failed += 1
            out(f"#{i} SKIPPED unusable entry {entry!r}: "
                f"{type(exc).__name__}: {exc}")
            continue
        if not (ok_bf16 and ok_cap):
            failed += 1
            out(f"{tag} BLOCKED {msg_bf16}; {msg_cap}")
            continue
        if dry_run:
            out(f"{tag} DRY-RUN OK {msg_bf16} {msg_cap} cmd: {cmd}")
            continue
        out(f"{tag} ENCODE {cmd}")
        try:
            result = runner(shlex.split(cmd))
        except Exception as exc:  # noqa: BLE001 — a dead encoder is a failure
            failed += 1
            out(f"{tag} FAILED {type(exc).__name__}: {exc}")
            continue
        rc = getattr(result, "returncode", result)
        if rc == 0:
            done.append(entry)
            out(f"{tag} DONE")
        else:
            failed += 1
            out(f"{tag} FAILED rc={rc}")

    if done:
        keep = [e for e in queue.entries() if e not in done]
        queue.rewrite(keep)
        out(f"queue: {len(done)} encoded, {len(keep)} remaining")
    return 1 if failed else 0


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Progressive Tensors lazy-encode queue worker"
    )
    p.add_argument(
        "--drain", action="store_true", help="process the queue (default: list)"
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="really run the encoder (default: DRY-RUN validation only)",
    )
    p.add_argument("--queue", default=None, help=f"path ({FQ_ENCODE_QUEUE_ENV})")
    p.add_argument("--bf16-dir", default=None, help=f"({FQ_BF16_DIR_ENV})")
    p.add_argument("--capture-dir", default=None, help=f"({FQ_CAPTURE_DIR_ENV})")
    p.add_argument(
        "--encoder-cmd", default=None, help=f"template ({FQ_ENCODER_CMD_ENV})"
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    queue = (
        EncodeQueue(args.queue) if args.queue else EncodeQueue.from_env()
    )
    if not args.drain:
        print(f"encode queue {queue.path}: {len(queue)} entries")
        for i, entry in enumerate(queue.entries(), 1):
            print(
                f"#{i} L{entry['layer']}/e{entry['expert']} K{entry['k']} "
                f"reason={entry.get('reason', '?')} "
                f"requested={entry.get('requested_utc', '?')}"
            )
        return 0
    return drain(
        queue,
        dry_run=not args.execute,
        bf16_dir=args.bf16_dir,
        capture_dir=args.capture_dir,
        encoder_cmd=args.encoder_cmd,
        limit=args.limit,
    )


if __name__ == "__main__":
    sys.exit(main())
