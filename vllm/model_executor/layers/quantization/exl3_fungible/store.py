# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fungible-quant policy store (M2) — 01-artifacts-policy-stats.md §1.2/§1.3.

Persists `fq-policy/2` documents under ``<cache_root>/fq/policy/<manifest>/``:
``current.json`` is the authoritative committed policy, written via
write-temp + atomic rename (the startup_plan pattern — a crash mid-write
can never tear it, T8); the previous 16 committed policies rotate through
``history/NNN.json`` for rollback/inspection. Decayed stats summaries dump
alongside under ``stats/`` keyed by the policy hash they informed.

Policies are keyed by logical expert id only — no rank/world_size/tp/device
fields (D4). ``n_k4_per_layer`` is capacity-fixed at startup (D1); a policy
whose counts differ is applied by projection (policy.project) only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

POLICY_SCHEMA = "fq-policy/2"
HISTORY_KEEP = 16


def policy_hash(doc: dict) -> str:
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_policy(doc: dict, *, num_experts: int | None = None) -> None:
    if doc.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"unsupported policy schema: {doc.get('schema')!r}")
    bpe = doc.get("bits_per_expert")
    if not isinstance(bpe, dict) or not bpe:
        raise ValueError("bits_per_expert missing/empty")
    budget = doc.get("budget", {})
    caps = budget.get("n_k4_per_layer", {})
    for layer, bits in bpe.items():
        if num_experts is not None and len(bits) != num_experts:
            raise ValueError(
                f"layer {layer}: {len(bits)} entries != {num_experts} experts")
        bad = [b for b in bits if b not in (3, 4)]
        if bad:
            raise ValueError(f"layer {layer}: non-{{3,4}} bits {bad[:4]}")
        if layer in caps and sum(b == 4 for b in bits) != caps[layer]:
            raise ValueError(
                f"layer {layer}: K4 occupancy != declared n_k4_per_layer "
                f"(v1 requires cap == n)")
    for banned in ("rank", "world_size", "tp", "device", "devices"):
        if banned in doc:
            raise ValueError(f"policy must be topology-neutral; found {banned!r}")


class PolicyStore:
    """Filesystem policy store bound to one artifact manifest."""

    def __init__(self, cache_root: str | Path, manifest_hash: str):
        self.root = Path(cache_root) / "fq" / "policy" / manifest_hash
        self.manifest_hash = manifest_hash
        (self.root / "history").mkdir(parents=True, exist_ok=True)
        (self.root / "stats").mkdir(exist_ok=True)

    # ------------------------------------------------------------------ io

    def _atomic_write(self, path: Path, doc: dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f, sort_keys=True, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def commit(self, doc: dict, *, num_experts: int | None = None) -> str:
        """Validate, rotate history, atomically publish as current.json."""
        validate_policy(doc, num_experts=num_experts)
        if doc.get("manifest") != self.manifest_hash:
            raise ValueError(
                f"policy manifest {doc.get('manifest')!r} != store manifest "
                f"{self.manifest_hash!r} (T8: hard refuse)")
        cur = self.root / "current.json"
        if cur.exists():
            gen = len(sorted((self.root / "history").glob("*.json")))
            os.replace(cur, self.root / "history" / f"{gen:04d}.json")
            hist = sorted((self.root / "history").glob("*.json"))
            for stale in hist[:-HISTORY_KEEP]:
                stale.unlink()
        self._atomic_write(cur, doc)
        return policy_hash(doc)

    def load_current(self, *, num_experts: int | None = None) -> dict | None:
        """Rehydrate the last committed policy (boot path, D8)."""
        cur = self.root / "current.json"
        if not cur.exists():
            return None
        doc = json.loads(cur.read_text())
        validate_policy(doc, num_experts=num_experts)
        if doc.get("manifest") != self.manifest_hash:
            raise ValueError("persisted policy manifest mismatch — refusing")
        return doc

    def history(self) -> list[Path]:
        return sorted((self.root / "history").glob("*.json"))

    def dump_stats(self, summary: dict, for_policy_hash: str) -> Path:
        p = self.root / "stats" / f"{for_policy_hash[:16]}.json"
        self._atomic_write(p, summary)
        return p
