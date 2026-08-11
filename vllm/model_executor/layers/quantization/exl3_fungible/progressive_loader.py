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
        self._reclaim_allocator()

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

            from .memory_preflight import _env_bytes, check_or_raise, project
            from .progressive import measure_footprint_inputs

            free, total = torch.cuda.mem_get_info()
            expert_bytes_by_k, dense_bytes = measure_footprint_inputs(
                spec, resolver, tp_rank=tp_rank)
            if not expert_bytes_by_k:
                logger.warning("FQ memory preflight skipped: no fragment "
                               "could be sized")
                return
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
            check_or_raise(proj, expert_bytes_by_k, logger.info)
        except ImportError:
            logger.warning("FQ memory preflight unavailable", exc_info=True)

    @staticmethod
    def _reclaim_allocator() -> None:
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
        except ImportError:
            pass
