# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Progressive Tensors boot (loader v2) — stream a mixed-K EXL3 checkpoint
directly from segments + a policy, with no materialized assembled checkpoint.

Pieces (torch + stdlib only; the vLLM loader class lives in
``progressive_loader.py``):

* :class:`ProgressiveSpec` — resolves manifest dir / policy / dense source
  from the environment (``VLLM_FQ_MANIFEST_DIR``, ``VLLM_FQ_POLICY`` or the
  M2 PolicyStore, ``VLLM_FQ_DENSE_SOURCE``) and validates them against each
  other.
* :func:`progressive_weights_iterator` — synthesizes the exact tensor
  stream ``fq_assemble`` would have materialized: dense/attention/shared
  (and bf16 MTP expert) tensors stream from the SOURCE checkpoint shards;
  routed-expert tensors resolve per-policy through
  :class:`~.fragments.FragmentResolver`. Name-set parity with the source
  shard header is enforced per layer.
* :func:`synthesize_hf_overrides` / :func:`write_tier_bitmap` — the mixed
  ``hybrid_tr3_tail`` metadata + ``quantization_config`` stub of the GG
  loader contract (runs/serve-baseline/fruit-mixed-report.md §2), emitted
  as an ``--hf-overrides`` JSON whose ``bits_per_expert`` is an
  absolute-path ``file.json:field`` reference to a tier bitmap written
  from the policy.  ``python -m ...exl3_fungible.progressive`` emits both.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from vllm.model_executor.layers.quantization.exl3_fungible.fragments import (
        _ST_TO_TORCH_NAME,
        EXPERT_RE,
        FQ_CACHE_ENV,
        DEFAULT_CACHE_DIR,
        FragmentResolver,
        read_safetensors_header,
    )
except ImportError:  # standalone execution (tests / CLI without built vllm)
    import importlib.util as _ilu
    import sys as _sys
    from pathlib import Path as _P

    _spec = _ilu.spec_from_file_location(
        "fq_fragments_standalone", _P(__file__).resolve().parent / "fragments.py"
    )
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    _ST_TO_TORCH_NAME = _mod._ST_TO_TORCH_NAME  # noqa: SLF001
    EXPERT_RE = _mod.EXPERT_RE
    FQ_CACHE_ENV = _mod.FQ_CACHE_ENV
    DEFAULT_CACHE_DIR = _mod.DEFAULT_CACHE_DIR
    FragmentResolver = _mod.FragmentResolver
    read_safetensors_header = _mod.read_safetensors_header

FQ_MANIFEST_DIR_ENV = "VLLM_FQ_MANIFEST_DIR"
FQ_POLICY_ENV = "VLLM_FQ_POLICY"
FQ_DENSE_SOURCE_ENV = "VLLM_FQ_DENSE_SOURCE"

QUANT_STUB_KEYS = ("codebook", "exllamav3_version")

_RANK_SEG_RE = re.compile(r"\.rank(\d+)\.")


def _policy_digest(doc: dict) -> str:
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _bits_digest(bits: Iterable[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(b)) for b in bits).encode()
    ).hexdigest()[:12]


@dataclass
class ProgressiveSpec:
    """Everything a progressive boot needs, resolved and cross-checked."""

    manifest_dir: Path
    manifest: dict
    policy: dict
    policy_path: Path | None
    dense_source: Path
    bits_by_layer: dict[int, tuple[int, ...]]
    k_values: tuple[int, ...]

    @property
    def policy_digest(self) -> str:
        return _policy_digest(self.policy)

    @classmethod
    def from_env(
        cls,
        model_path: str | Path | None = None,
        *,
        environ: dict[str, str] = os.environ,  # noqa: B008
        overrides: dict[str, Any] | None = None,
    ) -> "ProgressiveSpec":
        overrides = overrides or {}

        manifest_dir = overrides.get("manifest_dir") or environ.get(
            FQ_MANIFEST_DIR_ENV
        )
        if not manifest_dir:
            raise ValueError(
                "load_format=progressive requires the segment dir: set "
                f"{FQ_MANIFEST_DIR_ENV} (dir with fq-manifest.json + "
                "index-k*.json + layer-*.k*.safetensors) or pass "
                "model_loader_extra_config={'manifest_dir': ...}"
            )
        manifest_dir = Path(manifest_dir)
        manifest_path = manifest_dir / "fq-manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {}
        )

        policy_path: Path | None = None
        policy_ref = overrides.get("policy") or environ.get(FQ_POLICY_ENV)
        if policy_ref:
            policy_path = Path(policy_ref)
            policy = json.loads(policy_path.read_text())
        else:
            policy = cls._policy_from_store(manifest_path, environ)
            if policy is None:
                raise ValueError(
                    "load_format=progressive found no policy: set "
                    f"{FQ_POLICY_ENV} to a fq-policy JSON or commit one to "
                    "the PolicyStore for this manifest"
                )

        bpe = policy.get("bits_per_expert")
        if not isinstance(bpe, dict) or not bpe:
            raise ValueError("policy has no bits_per_expert map")
        bits_by_layer = {
            int(layer): tuple(int(b) for b in bits) for layer, bits in bpe.items()
        }
        k_values = tuple(
            sorted({b for bits in bits_by_layer.values() for b in bits})
        )

        num_experts = manifest.get("num_experts")
        if num_experts is not None:
            for layer, bits in bits_by_layer.items():
                if len(bits) != int(num_experts):
                    raise ValueError(
                        f"policy layer {layer} has {len(bits)} experts, "
                        f"manifest says {num_experts}"
                    )
        moe_layers = manifest.get("moe_layers")
        if isinstance(moe_layers, list) and len(moe_layers) == 2:
            first, last = int(moe_layers[0]), int(moe_layers[1])
            missing = sorted(
                set(range(first, last + 1)) - set(bits_by_layer)
            )
            if missing:
                raise ValueError(
                    f"policy is missing MoE layers {missing} "
                    f"(manifest moe_layers={moe_layers})"
                )

        dense_source = (
            overrides.get("dense_source")
            or environ.get(FQ_DENSE_SOURCE_ENV)
            or manifest.get("dense_source")
            or model_path
        )
        if not dense_source:
            raise ValueError(
                "load_format=progressive needs a dense source checkpoint: "
                f"set {FQ_DENSE_SOURCE_ENV}, add 'dense_source' to "
                "fq-manifest.json, or serve --model <dense checkpoint dir>"
            )
        dense_source = Path(dense_source)
        if not dense_source.is_dir():
            raise ValueError(
                f"progressive dense source is not a directory: {dense_source}"
            )

        return cls(
            manifest_dir=manifest_dir,
            manifest=manifest,
            policy=policy,
            policy_path=policy_path,
            dense_source=dense_source,
            bits_by_layer=bits_by_layer,
            k_values=k_values,
        )

    @staticmethod
    def _policy_from_store(
        manifest_path: Path, environ: dict[str, str]
    ) -> dict | None:
        """PolicyStore fallback: current.json keyed by the manifest hash."""
        if not manifest_path.exists():
            return None
        try:
            from vllm.model_executor.layers.quantization.exl3_fungible.store import (
                PolicyStore,
            )
        except ImportError:
            import importlib.util as _ilu
            import sys as _sys

            _spec = _ilu.spec_from_file_location(
                "fq_store_standalone",
                Path(__file__).resolve().parent / "store.py",
            )
            _mod = _ilu.module_from_spec(_spec)
            _sys.modules[_spec.name] = _mod
            _spec.loader.exec_module(_mod)
            PolicyStore = _mod.PolicyStore
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        cache_root = Path(
            environ.get(FQ_CACHE_ENV) or os.path.expanduser(DEFAULT_CACHE_DIR)
        )
        store = PolicyStore(cache_root, manifest_hash)
        return store.load_current()

    def make_resolver(self, **kwargs: Any) -> FragmentResolver:
        return FragmentResolver(self.manifest_dir, **kwargs)

    def shard_files(self) -> list[Path]:
        index_path = self.dense_source / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            names = sorted(set(index.get("weight_map", {}).values()))
            files = [self.dense_source / name for name in names]
            missing = [f.name for f in files if not f.exists()]
            if missing:
                raise FileNotFoundError(
                    f"dense source shards missing: {missing[:4]}"
                )
            return files
        files = sorted(self.dense_source.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(
                f"no safetensors shards in dense source {self.dense_source}"
            )
        return files


def _shard_tensor(mm, body_offset: int, entry: dict):
    """Zero-copy torch view of one tensor inside a source shard mmap."""
    import warnings

    import torch

    dtype = getattr(torch, _ST_TO_TORCH_NAME[entry["dtype"]])
    shape = tuple(entry["shape"])
    numel = 1
    for dim in shape:
        numel *= dim
    a, b = entry["data_offsets"]
    if numel == 0:
        return torch.empty(shape, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(
            memoryview(mm), dtype=dtype, count=numel, offset=body_offset + a
        ).view(shape)


def progressive_weights_iterator(
    spec: ProgressiveSpec,
    resolver: FragmentResolver,
    *,
    tp_rank: int | None = None,
    log: Callable[[str], None] | None = None,
    actual_bits_out: dict[int, list[int]] | None = None,
):
    """Yield ``(name, cpu_tensor)`` exactly as an assembled checkpoint would.

    Non-expert tensors stream from the dense-source shards; expert tensors
    of policy layers resolve through the FragmentResolver at the policy's
    per-expert K.  When ``tp_rank`` is given, only that rank's slice of the
    rank-sliced expert tensors is yielded (the model drops foreign ranks
    anyway; filtering here avoids materializing them at all).

    With ``VLLM_FQ_K_FALLBACK`` active the resolver may substitute a
    fallback K for a missing/untrusted fragment; the per-layer tier line and
    ``bits_digest`` are computed from the ACTUALLY loaded Ks (reality, not
    the requested policy), substitutions are called out on the line, and
    ``actual_bits_out`` (when given) receives the per-layer actually-loaded
    bits vectors — feed it to :func:`write_tier_bitmap` so the hybrid
    metadata records the loaded Ks.
    """
    import mmap as _mmap

    def _emit(msg: str) -> None:
        if log is not None:
            log(msg)

    def _rank_ok(name: str) -> bool:
        if tp_rank is None:
            return True
        m = _RANK_SEG_RE.search(name)
        return m is None or int(m.group(1)) == tp_rank

    t_start = time.perf_counter()
    dense_bytes = 0
    expert_tensor_count = 0

    for shard in spec.shard_files():
        header, body_offset = read_safetensors_header(shard)
        header.pop("__metadata__", None)
        ordered = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])

        expert_layers: dict[int, set[str]] = {}
        f = open(shard, "rb")
        # The mapping must outlive this generator: yielded tensors are
        # zero-copy views that the quant method only devices-copies in
        # process_weights_after_loading. GC owns the lifetime.
        mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)
        for name, entry in ordered:
            m = EXPERT_RE.match(name)
            if m is not None and int(m.group(1)) in spec.bits_by_layer:
                if _rank_ok(name):
                    expert_layers.setdefault(int(m.group(1)), set()).add(name)
                continue
            dense_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]
            yield name, _shard_tensor(mm, body_offset, entry)

        for layer in sorted(expert_layers):
            expected_names = expert_layers[layer]
            bits = spec.bits_by_layer[layer]
            actual_bits = [int(b) for b in bits]
            substitutions: list[tuple[int, int, int]] = []
            seen: set[str] = set()
            counts: dict[int, int] = {}
            for expert, k in enumerate(bits):
                fragment = resolver.resolve(layer, expert, int(k))
                actual_k = int(fragment.k)
                actual_bits[expert] = actual_k
                if actual_k != int(k):
                    substitutions.append((expert, int(k), actual_k))
                counts[actual_k] = counts.get(actual_k, 0) + 1
                for name, tensor in resolver.materialize(
                    fragment, name_filter=_rank_ok
                ):
                    seen.add(name)
                    expert_tensor_count += 1
                    yield name, tensor
            if actual_bits_out is not None:
                actual_bits_out[layer] = list(actual_bits)
            if seen != expected_names:
                missing = sorted(expected_names - seen)[:4]
                extra = sorted(seen - expected_names)[:4]
                raise ValueError(
                    f"progressive layer {layer}: fragment stream does not "
                    f"match the source shard tensor set "
                    f"(missing={missing}, unexpected={extra})"
                )
            sub_note = ""
            if substitutions:
                sub_note = " substituted=" + ",".join(
                    f"e{expert}:K{requested}->K{got}"
                    for expert, requested, got in substitutions
                )
            _emit(
                f"FQ progressive layer {layer}: tiers="
                f"{tuple(sorted(counts.items()))} "
                f"bits_digest={_bits_digest(actual_bits)} "
                f"tensors={len(seen)}{sub_note}"
            )

    stats = dict(resolver.stats)
    _emit(
        "FQ progressive stream complete: "
        f"policy={spec.policy_digest[:12]} k_values={spec.k_values} "
        f"dense_bytes={dense_bytes} expert_tensors={expert_tensor_count} "
        f"fragments(local={stats['local']}, cache={stats['cache']}, "
        f"fetched={stats['fetched']}, bytes_fetched={stats['bytes_fetched']}, "
        f"substituted={stats.get('fallback_substituted', 0)}, "
        f"encode_queued={stats.get('encode_queued', 0)}) "
        f"wall={time.perf_counter() - t_start:.2f}s"
    )


# --------------------------------------------------------------- metadata


def write_tier_bitmap(
    policy: dict,
    out_path: str | Path,
    *,
    actual_bits: dict[int, list[int]] | None = None,
) -> Path:
    """Write the per-layer expert bitrate map the exl3 loader dereferences.

    ``actual_bits`` (e.g. the ``actual_bits_out`` collected from
    :func:`progressive_weights_iterator`) overrides the policy's bits so the
    hybrid metadata records the ACTUALLY loaded Ks after fallback
    substitution."""
    out_path = Path(out_path)
    source = (
        actual_bits if actual_bits is not None else policy["bits_per_expert"]
    )
    bitmap = {
        str(layer): {"bits_per_expert": [int(b) for b in bits]}
        for layer, bits in source.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(bitmap, indent=1) + "\n")
    os.replace(tmp, out_path)
    return out_path


def synthesize_hf_overrides(
    dense_source: str | Path,
    policy: dict,
    bitmap_path: str | Path,
    *,
    extra: dict | None = None,
) -> dict:
    """Build the ``--hf-overrides`` dict for a progressive boot.

    Emits the full ``hybrid_tr3_tail`` (dict-valued hf_overrides replace the
    attribute wholesale) patched per the mixed GG loader contract, plus the
    ``quantization_config`` stub when the source config lacks one.  The
    ``bits_per_expert`` reference uses an absolute path, which
    ``get_hf_file_to_dict`` -> ``try_get_local_file`` resolves via
    ``Path(model) / file`` = the absolute path itself.
    """
    cfg = json.loads((Path(dense_source) / "config.json").read_text())
    tail = dict(cfg.get("hybrid_tr3_tail") or {})
    if tail.get("format") != "exl3-trellis":
        raise ValueError(
            f"{dense_source}/config.json has no exl3-trellis hybrid_tr3_tail; "
            "not a rank-sliced EXL3 dense source"
        )
    ks = sorted({int(b) for bits in policy["bits_per_expert"].values() for b in bits})
    if len(ks) == 1:
        tail["bits"] = float(ks[0])
        tail.pop("k_values", None)
        tail.pop("bits_per_expert", None)
    else:
        tail["bits"] = "mixed"
        tail["k_values"] = ks
        tail["bits_per_expert"] = (
            f"{Path(bitmap_path).resolve()}:bits_per_expert"
        )

    overrides: dict[str, Any] = {"hybrid_tr3_tail": tail}
    if "quantization_config" not in cfg:
        overrides["quantization_config"] = {
            "quant_method": "exl3",
            "bits": tail["bits"],
            "codebook": tail.get("codebook", "mcg"),
            "version": tail.get("exllamav3_version", "rank-sliced"),
        }
    if extra:
        overrides.update(extra)
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Emit the tier bitmap + --hf-overrides JSON for a progressive boot."""
    import argparse

    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--dense-source", default=None)
    p.add_argument(
        "--bitmap-out",
        default=None,
        help="tier bitmap path (default: <cache>/boot/<policy digest>.json)",
    )
    p.add_argument(
        "--extra-overrides",
        default=None,
        help="JSON dict merged into the emitted overrides",
    )
    args = p.parse_args(argv)

    overrides_in = {
        k: v
        for k, v in {
            "manifest_dir": args.manifest_dir,
            "policy": args.policy,
            "dense_source": args.dense_source,
        }.items()
        if v
    }
    spec = ProgressiveSpec.from_env(overrides=overrides_in)
    bitmap_path = args.bitmap_out
    if bitmap_path is None:
        cache_root = Path(
            os.environ.get(FQ_CACHE_ENV) or os.path.expanduser(DEFAULT_CACHE_DIR)
        )
        bitmap_path = cache_root / "boot" / f"{spec.policy_digest[:16]}.tier_bitmap.json"
    write_tier_bitmap(spec.policy, bitmap_path)
    extra = json.loads(args.extra_overrides) if args.extra_overrides else None
    overrides = synthesize_hf_overrides(
        spec.dense_source, spec.policy, bitmap_path, extra=extra
    )
    import sys

    print(json.dumps(overrides, separators=(",", ":")))
    print(f"tier bitmap: {bitmap_path}", file=sys.stderr)
    print(f"policy digest: {spec.policy_digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
