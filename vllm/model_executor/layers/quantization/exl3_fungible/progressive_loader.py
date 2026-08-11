# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--load-format progressive`` — boot mixed-K EXL3 from Progressive
Tensors segments + a policy, with no materialized assembled checkpoint.

The loader synthesizes the tensor stream an assembled checkpoint would
provide (see :mod:`.progressive`): dense/attention/shared tensors from the
dense-source checkpoint dir, routed experts resolved per-policy through the
:class:`~.fragments.FragmentResolver` (local segment dirs -> HF sources).
The mixed ``hybrid_tr3_tail`` / ``quantization_config`` metadata reaches
the model config through the ``--hf-overrides`` JSON that
``python -m ...exl3_fungible.progressive`` emits at serve time.

Configuration: ``VLLM_FQ_MANIFEST_DIR`` (+ ``VLLM_FQ_POLICY`` /
``VLLM_FQ_DENSE_SOURCE`` / ``VLLM_FQ_CACHE`` / ``VLLM_FQ_VERIFY`` /
``VLLM_FQ_SOURCES``), overridable per-serve via
``--model-loader-extra-config '{"manifest_dir": ..., "policy": ...,
"dense_source": ...}'``.
"""
from __future__ import annotations

import os
import time

from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.exl3_fungible.progressive import (
    ProgressiveSpec,
    progressive_weights_iterator,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

_EXTRA_CONFIG_KEYS = {"manifest_dir", "policy", "dense_source"}

# Set by the last progressive load_weights on this worker, consumed by
# record_weight_footprint once the weights are actually resident. Module-level
# rather than threaded through vLLM's worker API, which has no seam for it.
_LAST_LOAD: dict = {}


def _remember_load(loader, spec, tp_rank) -> None:
    _LAST_LOAD.update(loader=loader, spec=spec, tp_rank=tp_rank)


def record_weight_footprint(allocated_bytes: int) -> None:
    """Calibrate the dense term from a load that actually completed.

    ``allocated_bytes`` is the true per-rank weight footprint, measured after
    process_weights_after_loading. Subtracting the policy's expert bytes
    leaves the policy-INDEPENDENT dense term -- which is what the preflight
    cannot compute from headers, because non-expert tensors carry no .rankN.
    suffix and summing their file bytes charges one rank for all four.
    """
    loader = _LAST_LOAD.get("loader")
    if loader is None or loader._last_expert_bytes is None:
        return
    dense = allocated_bytes - loader._last_expert_bytes
    if dense <= 0:
        logger.warning(
            "FQ calibration skipped: footprint %.2f GiB is below the policy's "
            "own expert bytes %.2f GiB -- measured too early to be the real "
            "footprint", allocated_bytes / (1 << 30),
            loader._last_expert_bytes / (1 << 30))
        return
    loader._write_dense_calibration(
        _LAST_LOAD["spec"], _LAST_LOAD["tp_rank"], dense)


class ProgressiveModelLoader(BaseModelLoader):
    """Stream mixed-K EXL3 weights from Progressive Tensors segments."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        if not isinstance(extra, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for "
                "load_format=progressive"
            )
        unexpected = set(extra) - _EXTRA_CONFIG_KEYS
        if unexpected:
            raise ValueError(
                f"Unexpected extra config keys for load_format=progressive: "
                f"{unexpected} (allowed: {sorted(_EXTRA_CONFIG_KEYS)})"
            )
        self._extra = extra

    def _spec(self, model_config: ModelConfig) -> ProgressiveSpec:
        return ProgressiveSpec.from_env(
            model_config.model, overrides=self._extra
        )

    def download_model(self, model_config: ModelConfig) -> None:
        # Validates that manifest dir / policy / dense source resolve; all
        # fragment IO is lazy and rank-scoped, so there is nothing to
        # prefetch here.
        self._spec(model_config)

    @staticmethod
    def _tp_rank() -> int | None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            return get_tensor_model_parallel_rank()
        except (ImportError, AssertionError, RuntimeError, ValueError):
            return None

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        spec = self._spec(model_config)
        resolver = spec.make_resolver()
        tp_rank = self._tp_rank()
        logger.info(
            "Progressive boot: manifest_dir=%s policy=%s (digest %s) "
            "dense_source=%s k_values=%s tp_rank=%s",
            spec.manifest_dir,
            spec.policy_path or "<policy store>",
            spec.policy_digest[:12],
            spec.dense_source,
            spec.k_values,
            tp_rank,
        )
        # State the network posture BEFORE the nine-minute compile, not after
        # a failure. A run that silently fetched when the operator asked for
        # offline is not reproducible; a run that cannot fetch should say so
        # while there is still time to prime the cache.
        try:
            from vllm.model_executor.layers.quantization.exl3_fungible.\
                fragments import HfSource
            if HfSource.offline():
                logger.warning(
                    "Progressive boot: OFFLINE (HF_HUB_OFFLINE set) — Hub "
                    "sources disabled; only local segment dirs and the primed "
                    "cache will be used. Set FQ_ALLOW_NETWORK=1 to override.")
        except Exception:  # noqa: BLE001 — posture logging must not fail a boot
            pass
        self._preflight_memory(spec, resolver, tp_rank)
        start = time.perf_counter()
        model.load_weights(
            progressive_weights_iterator(
                spec,
                resolver,
                tp_rank=tp_rank,
                log=logger.info,
            )
        )
        logger.info_once(
            "Loading weights took %.2f seconds (progressive)",
            time.perf_counter() - start,
        )
        # Reclaim + calibration happen in gpu_worker AFTER
        # process_weights_after_loading -- see record_weight_footprint.
        _remember_load(self, spec, tp_rank)

    # ------------------------------------------------------------ memory

    def _preflight_memory(self, spec, resolver, tp_rank) -> None:
        """Reject an oversized policy before paying for the load.

        A mixed-K footprint is fixed by the tier bitmap, so this is arithmetic.
        We learned that the expensive way: a policy 5.72 GiB over budget was
        only caught after a 62-minute load, by the KV allocator reporting
        ``Available KV cache memory: -3.1 GiB``.
        """
        if os.environ.get("VLLM_FQ_BUDGET_PREFLIGHT", "1") == "0":
            return
        try:
            import torch

            from .memory_preflight import (_env_bytes, check_or_raise,
                                           project, project_expert_bytes,
                                           render)
            from .progressive import measure_footprint_inputs

            free, total = torch.cuda.mem_get_info()
            expert_bytes_by_k, dense_bytes = measure_footprint_inputs(
                spec, resolver, tp_rank=tp_rank)
            if not expert_bytes_by_k:
                logger.warning("FQ memory preflight skipped: no fragment "
                               "could be sized")
                return

            # vLLM constructs the model -- allocating every parameter at its
            # final sharded shape, which for a mixed-K checkpoint means the
            # tier bitmap has already been honoured -- BEFORE calling us to
            # fill those parameters in. So the true per-rank weight footprint
            # is not something to reconstruct from file headers; it is sitting
            # in the allocator right now, exact.
            #
            # Reconstructing it is in fact wrong: non-expert tensors carry no
            # .rankN. suffix in the source shards because vLLM shards them
            # internally at load, so summing their file bytes charges one rank
            # for all four (35.19 GiB instead of ~12.14 GiB on GLM-5.2 TP4 --
            # a 23 GiB error, enough to reject a policy that fits comfortably).
            measured = torch.cuda.memory_allocated()
            projected_experts, _ = project_expert_bytes(
                spec.bits_by_layer, expert_bytes_by_k)
            # Stash for the post-load calibration write.
            self._last_expert_bytes = projected_experts
            calibrated = self._read_dense_calibration(spec, tp_rank)
            if measured >= projected_experts:
                dense_bytes = measured - projected_experts
                source = "measured (allocator, post-construction)"
            elif calibrated is not None:
                # Measured on a PREVIOUS boot of this same model/TP/dense
                # source: total weight footprint minus that boot's expert
                # bytes. The dense term does not depend on the policy, so one
                # successful boot calibrates every later projection exactly.
                dense_bytes = calibrated
                source = "calibrated from a previous boot"
            else:
                # Parameters are not resident yet -- fall back to headers and
                # say so, because the dense term is then an upper bound that
                # over-charges a TP rank.
                source = ("header upper bound -- dense term counts all TP "
                          "ranks; treat a near-miss as inconclusive")
            util = float(os.environ.get("VLLM_FQ_BUDGET_UTIL", "0"))
            if not util:
                from vllm.config import get_current_vllm_config
                util = get_current_vllm_config().cache_config \
                    .gpu_memory_utilization
            proj = project(
                spec.bits_by_layer,
                expert_bytes_by_k,
                dense_bytes,
                device_total_bytes=total,
                gpu_memory_utilization=util,
                # Measured on this box: a flat-K3 TP4 boot leaves 5.27 GiB of
                # allocator + activation + graph residue. Overridable because
                # it is the one term that is a calibration, not a measurement
                # of THIS policy.
                runtime_overhead_bytes=_env_bytes(
                    "VLLM_FQ_BUDGET_OVERHEAD", int(5.5 * (1 << 30))),
                min_kv_bytes=_env_bytes(
                    "VLLM_FQ_BUDGET_MIN_KV", 4 * (1 << 30)),
            )
            logger.info("FQ memory preflight: weight source = %s", source)
            if not source.startswith("measured"):
                # Refusing a boot on a number we know is inflated would be a
                # worse failure than the one this check exists to prevent:
                # it would block policies that fit. Report, do not enforce.
                for line in render(proj, expert_bytes_by_k):
                    logger.info(line)
                if not proj["fits"]:
                    logger.warning(
                        "FQ memory preflight: projection is an UPPER BOUND "
                        "and does not fit -- proceeding anyway; the engine's "
                        "own KV sizing has the final say.")
                return
            check_or_raise(proj, expert_bytes_by_k, logger.info)
        except ImportError:
            logger.warning("FQ memory preflight unavailable", exc_info=True)

    _last_expert_bytes: int | None = None

    @staticmethod
    def _calibration_path(spec, tp_rank):
        """Where this (dense source, TP rank) records its dense footprint.

        Keyed by rank because TP sharding is not uniform across ranks for
        every tensor, and by dense source because that is what determines the
        non-expert bytes.
        """
        import hashlib
        from pathlib import Path

        key = hashlib.blake2b(
            f"{spec.dense_source}|tp{tp_rank}".encode(), digest_size=8
        ).hexdigest()
        root = os.environ.get("VLLM_FQ_CACHE") or os.path.expanduser(
            "~/.cache/fq")
        return Path(root) / "calibration" / f"dense-{key}.json"

    def _read_dense_calibration(self, spec, tp_rank) -> int | None:
        override = os.environ.get("VLLM_FQ_BUDGET_DENSE")
        if override:
            from .memory_preflight import _env_bytes
            return _env_bytes("VLLM_FQ_BUDGET_DENSE", 0)
        try:
            import json
            return int(json.loads(
                self._calibration_path(spec, tp_rank).read_text()
            )["dense_bytes"])
        except (OSError, ValueError, KeyError):
            return None

    def _write_dense_calibration(self, spec, tp_rank, dense_bytes: int) -> None:
        """Record the dense footprint a successful boot actually used.

        Written only from a boot that got all the way through load_weights, so
        the number reflects a configuration that really ran.
        """
        try:
            import json
            p = self._calibration_path(spec, tp_rank)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "dense_bytes": int(dense_bytes),
                "dense_source": str(spec.dense_source),
                "tp_rank": tp_rank,
            }))
            tmp.replace(p)  # atomic: a torn file would poison every later boot
            logger.info(
                "FQ memory preflight: calibrated dense footprint %.2f GiB "
                "for tp_rank=%s -- future boots will project exactly",
                dense_bytes / (1 << 30), tp_rank)
        except (OSError, ValueError):
            logger.warning("FQ dense calibration not written", exc_info=True)

    def _reclaim_allocator(self, spec=None, tp_rank=None) -> None:
        """Return the streaming residue to the driver before KV profiling.

        Progressive loading stages tens of thousands of small per-expert
        buffers. PyTorch's caching allocator keeps those blocks RESERVED after
        they are freed, but vLLM sizes the KV cache from what the DRIVER
        reports free -- so the residue is charged against KV even though
        nothing holds it. Measured on a TP4 GLM-5.2 boot: progressive left
        9.19 GiB of non-weight residue where the equivalent flat load left
        5.27 GiB, i.e. 3.92 GiB of KV cache silently consumed by an allocator
        bookkeeping artifact.
        """
        try:
            import gc

            import torch

            if not torch.cuda.is_available():
                return
            before_r = torch.cuda.memory_reserved()
            before_a = torch.cuda.memory_allocated()
            gc.collect()
            torch.cuda.empty_cache()
            after_r = torch.cuda.memory_reserved()
            logger.info(
                "FQ post-load reclaim: reserved %.2f -> %.2f GiB "
                "(freed %.2f GiB of allocator residue; allocated %.2f GiB "
                "is the real weight footprint)",
                before_r / (1 << 30), after_r / (1 << 30),
                (before_r - after_r) / (1 << 30), before_a / (1 << 30),
            )
            # allocated == the real per-rank weight footprint now that every
            # parameter is filled. Subtracting the policy's expert bytes
            # leaves the policy-independent dense term, which is exactly what
            # the next boot's projection is missing.
            if spec is not None and self._last_expert_bytes is not None:
                dense = before_a - self._last_expert_bytes
                if dense > 0:
                    self._write_dense_calibration(spec, tp_rank, dense)
        except ImportError:
            pass
