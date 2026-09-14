# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare b12x plans held by the loaded model at two lifecycle points.

The ``weights`` stage runs after model load and before memory profiling. It
selects, compiles, and primes every plan that depends only on weights or
communicators. The ``state`` stage runs after the KV and state pools exist and
prepares the plans that depend on them. Both stages collect one unit per
layer from ``layer.b12x_preparation_provider.get_b12x_preparation_units`` and
hand the units' requests to one ``PreparationSession`` per worker.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import TYPE_CHECKING

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    b12x_preparation_token_counts,
    b12x_unit_providers,
    has_b12x,
    scope_b12x_unit_calls,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)

if os.environ.get("B12X_HANG_DUMP"):
    # A stalled worker dumps every thread's stack on SIGUSR1; ptrace-based
    # tools cannot reach worker processes from an unrelated shell. When the
    # variable names a directory, each process writes <pid>.txt there, which
    # survives the workers' stderr redirection; otherwise stderr is used.
    import faulthandler
    import signal

    _dump_target = os.environ["B12X_HANG_DUMP"]
    if os.path.isdir(_dump_target):
        _dump_file = open(os.path.join(_dump_target, f"{os.getpid()}.txt"), "w")
        faulthandler.register(signal.SIGUSR1, file=_dump_file, all_threads=True, chain=True)
    else:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)


def b12x_native_supported(worker: "Worker") -> bool:
    return (
        has_b12x()
        and current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
    )


def _draft_lane(worker: "Worker") -> int:
    runner = worker.model_runner
    return int(getattr(runner, "_draft_workspace_lane", 0) or 0)


def _planned_decode_counts(
    worker: "Worker", *, capture_sizes: tuple[int, ...], speculative_tokens: int
) -> tuple[int, ...]:
    """Token counts of the decode graphs the runner will capture.

    Both stages must declare the same counts: a plan prepared in the weights
    stage serves every graph, and a count that first appears in the state
    stage would reach a weight-only family after its winners were chosen. The
    graph manager knows the exact set once the KV cache exists; before that
    the uniform-decode rounding is derived from the same configuration.
    """
    runner = worker.model_runner
    manager = getattr(runner, "cudagraph_manager", None)
    planned_counts = getattr(manager, "planned_token_counts", None)
    if callable(planned_counts):
        return tuple(int(count) for count in planned_counts() if int(count) > 0)
    query_len = int(getattr(runner, "decode_query_len", 0) or 0) or 1 + int(speculative_tokens)
    if query_len <= 1 or not capture_sizes:
        return ()
    compilation = worker.vllm_config.compilation_config
    limit = int(getattr(compilation, "max_cudagraph_capture_size", None) or max(capture_sizes))
    max_reqs = int(worker.scheduler_config.max_num_seqs)
    counts = set()
    for size in capture_sizes:
        rounded = -(-int(size) // query_len) * query_len
        if rounded // query_len <= max_reqs and rounded <= limit:
            counts.add(rounded)
    return tuple(sorted(counts))


def b12x_workload(worker: "Worker", *, stage: str, lane: int = 0) -> B12xWorkload:
    """Describe the serving shapes of one stage without constructing plans."""
    compilation = worker.vllm_config.compilation_config
    capture_sizes = tuple(compilation.cudagraph_capture_sizes or ())
    compile_sizes = tuple(
        size for size in (compilation.compile_sizes or ()) if isinstance(size, int)
    )
    endpoints = tuple(
        compile_range.end for compile_range in compilation.get_compile_ranges()
    )
    spec = worker.vllm_config.speculative_config
    speculative_tokens = 0 if spec is None else int(getattr(spec, "num_speculative_tokens", 0))
    max_tokens = int(worker.scheduler_config.max_num_batched_tokens)
    planned = _planned_decode_counts(
        worker, capture_sizes=capture_sizes, speculative_tokens=speculative_tokens,
    )
    token_counts = b12x_preparation_token_counts(
        max_tokens=max_tokens,
        cudagraph_capture_sizes=(*capture_sizes, *planned),
        compile_sizes=compile_sizes,
        compile_range_endpoints=endpoints,
        speculative_tokens=speculative_tokens,
    )
    if spec is not None and getattr(spec, "enable_adaptive_verification", False):
        from vllm.v1.worker.gpu.spec_decode.adaptive_verification import preparation_tail_sizes

        token_counts = tuple(sorted({
            *token_counts, *preparation_tail_sizes(capture_sizes, max_tokens),
        }))
    if getattr(worker, "use_v2_model_runner", False):
        from vllm.v1.worker.gpu.warmup import warmup_prefill_shape

        prompt_len, max_reqs = warmup_prefill_shape(
            max_num_seqs=int(worker.scheduler_config.max_num_seqs),
            max_num_batched_tokens=max_tokens,
            decode_query_len=int(worker.model_runner.decode_query_len),
        )
        token_counts = tuple(sorted({
            *token_counts, *(prompt_len * reqs for reqs in range(1, max_reqs + 1)),
        }))
    fixed = {int(count) for count in (*capture_sizes, *compile_sizes, *planned)}
    fixed_token_counts = tuple(
        count for count in token_counts if count in fixed and count < max_tokens
    )
    dtype = worker.model_config.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        dtype = torch.bfloat16
    return B12xWorkload(
        stage=stage,
        token_counts=token_counts,
        fixed_token_counts=fixed_token_counts,
        output_dtype=dtype,
        max_tokens=max_tokens,
        max_seqs=int(worker.scheduler_config.max_num_seqs),
        max_model_len=int(worker.model_config.max_model_len),
        speculative_tokens=speculative_tokens,
        lane=lane,
    )


def _module_workload(module: torch.nn.Module, workload: B12xWorkload) -> B12xWorkload:
    extra = tuple(int(count) for count in getattr(module, "b12x_eager_token_counts", ()))
    if not extra:
        return workload
    if getattr(module, "b12x_eager_only", False):
        return replace(
            workload,
            token_counts=tuple(sorted(set(extra))),
            fixed_token_counts=(),
            max_tokens=max(extra),
            eager_only=True,
        )
    counts = tuple(sorted(set(workload.token_counts) | set(extra)))
    return replace(
        workload,
        token_counts=counts,
        max_tokens=max(workload.max_tokens, counts[-1]),
    )


def _units_from_modules(
    model: torch.nn.Module, workload: B12xWorkload, *, seen: set[int] | None = None,
) -> Iterable[B12xPreparationUnit]:
    if seen is None:
        seen = set()
    for module in model.modules():
        if id(module) in seen:
            continue
        seen.add(id(module))
        if workload.stage == "weights":
            head = getattr(module, "lm_head", None)
            processor = getattr(module, "logits_processor", None)
            bind_head = getattr(processor, "prepare_b12x_vocab_projection", None)
            if head is not None and callable(bind_head):
                bind_head(head)
        provider = getattr(module, "b12x_preparation_provider", None)
        hook = getattr(provider, "get_b12x_preparation_units", None)
        if not callable(hook):
            continue
        scoped = _module_workload(module, workload)
        if scoped.eager_only and workload.stage != "weights":
            continue
        for unit in hook(module, scoped):
            if not isinstance(unit, B12xPreparationUnit):
                raise TypeError(
                    f"{type(provider).__qualname__} returned a non-unit preparation value"
                )
            if scoped.eager_only and unit.autotune:
                unit = replace(unit, autotune=False)
            yield unit


def mark_b12x_eager_shapes(worker: "Worker") -> None:
    """Stamp multimodal encoder and connector modules with their profile rows.

    Those rows come from the encoder budget, are executed eagerly, and are
    never captured, so their plans are prepared with default configurations.
    """
    model = worker.get_model()
    mm_registry = getattr(worker.model_runner, "mm_registry", None)
    visual = getattr(model, "visual", None)
    get_encoder_rows = getattr(model, "get_num_mm_encoder_tokens", None)
    if mm_registry is None or visual is None or not callable(get_encoder_rows):
        return
    from vllm.multimodal.encoder_budget import MultiModalBudget

    encoder_budget = MultiModalBudget(
        worker.vllm_config, mm_registry, enable_cache=False,
    ).get_encoder_budget()
    if encoder_budget <= 0:
        return
    encoder_rows = int(get_encoder_rows(encoder_budget))
    if encoder_rows <= 0:
        raise ValueError("multimodal encoder profiling rows must be positive")
    for module in visual.modules():
        module.b12x_eager_token_counts = (encoder_rows,)
        module.b12x_eager_only = True
    get_connector_rows = getattr(model, "get_num_mm_connector_tokens", None)
    get_mapping = getattr(model, "get_mm_mapping", None)
    if not callable(get_connector_rows) or not callable(get_mapping):
        return
    connector_rows = int(get_connector_rows(encoder_rows))
    if connector_rows <= 0:
        raise ValueError("multimodal connector profiling rows must be positive")
    for prefix in get_mapping().connector:
        connector = model.get_submodule(prefix.rstrip("."))
        for module in connector.modules():
            module.b12x_eager_token_counts = (connector_rows,)
            module.b12x_eager_only = True


def _draft_workload(worker: "Worker", workload: B12xWorkload, *, lane: int):
    """Include the parallel draft's query and sequential sampling row counts."""
    speculator = getattr(worker.model_runner, "speculator", None)
    query_len = int(getattr(speculator, "num_query_per_req", 0) or 0)
    if not query_len:
        return replace(workload, lane=lane)
    max_reqs = min(workload.max_seqs, workload.max_tokens // query_len)
    counts = set(workload.token_counts)
    draft_counts = set(range(1, max_reqs + 1))
    draft_counts.update(reqs * query_len for reqs in range(1, max_reqs + 1))
    manager = getattr(speculator, "query_cudagraph_manager", None)
    planned = getattr(manager, "planned_token_counts", None)
    if callable(planned):
        draft_counts.update(int(count) for count in planned() if int(count) > 0)
    counts.update(draft_counts)
    fixed = set(workload.fixed_token_counts)
    fixed.update(count for count in draft_counts if count < workload.max_tokens)
    return replace(
        workload, token_counts=tuple(sorted(counts)),
        fixed_token_counts=tuple(sorted(fixed)), lane=lane,
    )


def collect_b12x_units(worker: "Worker", workload: B12xWorkload) -> list[B12xPreparationUnit]:
    """Collect every unit of one stage from the model, the draft, and comms."""
    mark_b12x_eager_shapes(worker)
    units: list[B12xPreparationUnit] = []
    draft = worker.get_draft_model()
    if draft is not None:
        lane = _draft_lane(worker)
        workload = _draft_workload(worker, workload, lane=0)
    # Target and draft can share the same embedding and output-head modules.
    seen: set[int] = set()
    units.extend(_units_from_modules(worker.get_model(), workload, seen=seen))
    if draft is not None:
        draft_workload = replace(workload, lane=lane)
        from vllm.v1.worker.workspace import use_workspace_lane

        with use_workspace_lane(lane):
            draft_units = list(_units_from_modules(draft, draft_workload, seen=seen))
        if lane:
            draft_units = [scope_b12x_unit_calls(unit, lane) for unit in draft_units]
        units.extend(draft_units)
    for provider in b12x_unit_providers():
        hook = getattr(provider, "get_b12x_preparation_units", None)
        if callable(hook):
            units.extend(hook(provider, workload))
    # The state stage also carries the weights-stage units: their prepared
    # plans are skipped by the session, and any plan a layer declared for a
    # count that only the state stage exposes is prepared before capture.
    own = [unit for unit in units if unit.stage == workload.stage]
    if workload.stage == "state":
        units = own + [unit for unit in units if unit.stage != workload.stage]
    else:
        units = own
    _dump_declarations(own, workload)
    names = Counter(request.name for unit in units for request in unit.requests)
    duplicates = sorted(name for name, count in names.items() if count > 1)
    if duplicates:
        raise ValueError(f"conflicting b12x preparation request names: {duplicates}")
    return units


def _dump_declarations(units: Iterable[B12xPreparationUnit], workload: B12xWorkload) -> None:
    """Append every declaration of this stage to the corpus file, if requested.

    ``VLLM_B12X_DUMP_QUERIES`` names a JSON-lines file. Each line records one
    scalar declaration: component, encoded query, invocation, stage, and the
    request name. The preparation memory-envelope test replays those records
    on the host.
    """
    import json
    import os

    path = os.environ.get("VLLM_B12X_DUMP_QUERIES")
    if not path:
        return

    def scalar_plans(plan):
        variants = getattr(plan, "variants", None)
        return tuple(variants.values()) if variants else (plan,)

    def plain(value):
        """JSON form of declaration metadata: mappings, sequences, dtypes."""
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return plain(to_dict())
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [plain(item) for item in value]
        if isinstance(value, torch.dtype):
            return str(value).removeprefix("torch.")
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    try:
        with open(path, "a", encoding="utf-8") as handle:
            for unit in units:
                for request in unit.requests:
                    for plan in scalar_plans(request.plan):
                        contract = plan.contract
                        handle.write(json.dumps(plain({
                            "stage": workload.stage,
                            "unit": unit.name,
                            "request": request.name,
                            "autotune": unit.autotune,
                            "component": contract.component_id,
                            "query_schema": contract.query_schema_version,
                            "query": contract.encode_query(plan.query),
                            "invocation": plan.invocation,
                            "shared": bool(plan.shared),
                        }), sort_keys=True) + "\n")
    except Exception:  # A diagnostic dump never interrupts preparation.
        logger.exception("b12x declaration dump to %s failed", path)


def b12x_batches(units: Iterable[B12xPreparationUnit], *, autotune: bool = True):
    """Group requests: timed selection first, then default-only preparation.

    With autotune disabled each rank compiles and primes its own defaults
    or explicit pins in the calling process.
    """
    tuned, defaults = [], []
    for unit in units:
        (tuned if unit.autotune and autotune else defaults).extend(unit.requests)
    batches = []
    if tuned:
        batches.append((tuple(tuned), True))
    if defaults:
        batches.append((tuple(defaults), False))
    return batches


def get_b12x_session(worker: "Worker"):
    """Return the worker's preparation session, creating it on first use."""
    session = getattr(worker, "_b12x_session", None)
    if session is not None and session.state != "CLOSED":
        return session
    from b12x.preparation import PreparationSession

    config = worker.vllm_config
    spec = config.speculative_config
    namespace = {
        "model": str(config.model_config.model),
        "dtype": str(config.model_config.dtype),
        "kv_cache_dtype": str(config.cache_config.cache_dtype),
        "tensor_parallel": int(config.parallel_config.tensor_parallel_size),
        "pipeline_parallel": int(config.parallel_config.pipeline_parallel_size),
        "decode_context_parallel": int(
            getattr(config.parallel_config, "decode_context_parallel_size", 1) or 1
        ),
        "draft_model": None if spec is None else str(getattr(spec, "model", None)),
        "speculative_method": None if spec is None else str(getattr(spec, "method", None)),
    }
    session = PreparationSession(
        device=worker.device,
        autotune=bool(config.kernel_config.enable_b12x_autotune),
        namespace=namespace,
        # Compile workers are host processes that never touch CUDA, so the pool
        # is sized against host cores rather than against the device. The count
        # is per rank: a node runs this many compiler processes for each local
        # rank it hosts.
        compile_workers=16,
    )
    if session.autotune and os.environ.get("B12X_AUTOTUNE", "1") != "0":
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        ranks = tuple(sorted(int(rank) for rank in tp_group.ranks))
        session.configure_tuning_shard(int(worker.rank), ranks)
    worker._b12x_session = session
    return session


def begin_b12x_preparation(worker: "Worker", *, stage: str):
    """Start the world-coordinated preparation of one stage."""
    from vllm.distributed.parallel_state import get_world_group
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    batches = []
    if b12x_native_supported(worker):
        workload = b12x_workload(worker, stage=stage)
        batches = b12x_batches(
            collect_b12x_units(worker, workload),
            autotune=(bool(worker.vllm_config.kernel_config.enable_b12x_autotune)
                      and os.environ.get("B12X_AUTOTUNE", "1") != "0"),
        )
    session = get_b12x_session(worker) if batches else None
    return B12xPreparationCoordinator(
        session, batches, global_rank=int(worker.rank), world_group=get_world_group(),
    )


class B12xPreparedBatch:
    """Plans first prepared outside the executor rounds, released explicitly.

    Plans that were already prepared when the batch was built are not owned
    by it; releasing the batch leaves them installed.
    """

    def __init__(self, session, plans):
        self.session = session
        self.plans = tuple(plans)

    def release(self) -> None:
        plans, self.plans = self.plans, ()
        for plan in reversed(plans):
            self.session.release(plan)


def prepare_b12x_profile(worker: "Worker", *, stage: str) -> B12xPreparedBatch:
    """Prime profiling-pool plans with defaults in complete-world control rounds."""
    from vllm.distributed.parallel_state import get_world_group
    from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator

    requests = ()
    if b12x_native_supported(worker):
        workload = b12x_workload(worker, stage=stage)
        units = collect_b12x_units(worker, workload)
        requests = tuple(request for unit in units for request in unit.requests)
    session = get_b12x_session(worker) if requests else None
    fresh = tuple(request.plan for request in requests if request.plan.prepared is None)
    batch = B12xPreparedBatch(session, fresh)
    coordinator = B12xPreparationCoordinator(
        session, [(requests, False)] if requests else [],
        global_rank=int(worker.rank), world_group=get_world_group(),
    )
    outcome = coordinator.status()
    while not outcome["done"]:
        outcome = coordinator.advance()
    if outcome["error"] is not None:
        batch.release()
        raise RuntimeError(f"b12x profiling preparation failed: {outcome['error']}")
    return batch
