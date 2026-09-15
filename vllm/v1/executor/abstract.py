# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import replace
from functools import cached_property
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.tasks import SupportedTask
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase

logger = init_logger(__name__)

_R = TypeVar("_R")

FailureCallback = Callable[[], None]


def _aggregate_b12x_progress(outcomes, *, expected_ranks=None):
    ranked = sorted(
        (
            (int(outcome.get("global_rank", index)), outcome.get("progress"))
            for index, outcome in enumerate(outcomes)
            if outcome.get("progress") is not None
        ),
        key=lambda item: item[0],
    )
    if not ranked:
        return None
    progress = [item[1] for item in ranked]
    primary = next((item for item in progress if not item.done), progress[0])
    updates = {
        "measured_candidates": sum(item.measured_candidates for item in progress),
        "total_candidates": (
            sum(item.total_candidates for item in progress)
            if all(item.total_candidates is not None for item in progress)
            and (expected_ranks is None or {rank for rank, _ in ranked} == set(expected_ranks))
            else None
        ),
        "compilations": sum(item.compilations for item in progress),
        "active_compilations": sum(item.active_compilations for item in progress),
        "done": all(bool(outcome.get("done")) for outcome in outcomes),
        "elapsed_seconds": max(item.elapsed_seconds for item in progress),
    }
    return replace(primary, **updates)


class Executor(ABC):
    """Abstract base class for vLLM executors."

    An executor is responsible for executing the model on one device,
    or it can be a distributed executor that can execute the model on multiple devices.
    """

    uses_ray: bool = False  # whether the executor uses Ray for orchestration.
    supports_pp: bool = False  # whether the executor supports PP

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        executor_class: type[Executor]
        parallel_config = vllm_config.parallel_config
        distributed_executor_backend = parallel_config.distributed_executor_backend
        # distributed_executor_backend must be set in VllmConfig.__post_init__
        if isinstance(distributed_executor_backend, type):
            if not issubclass(distributed_executor_backend, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {distributed_executor_backend}."
                )
            executor_class = distributed_executor_backend
        elif distributed_executor_backend == "ray":
            if envs.VLLM_USE_RAY_V2_EXECUTOR_BACKEND:
                from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

                executor_class = RayExecutorV2
            else:
                from vllm.v1.executor.ray_executor import RayDistributedExecutor

                executor_class = RayDistributedExecutor
        elif distributed_executor_backend == "mp":
            from vllm.v1.executor.multiproc_executor import MultiprocExecutor

            executor_class = MultiprocExecutor
        elif distributed_executor_backend == "uni":
            from vllm.v1.executor.uniproc_executor import UniProcExecutor

            executor_class = UniProcExecutor
        elif distributed_executor_backend == "external_launcher":
            # TODO: make v1 scheduling deterministic
            # to support external launcher
            executor_class = ExecutorWithExternalLauncher
        elif isinstance(distributed_executor_backend, str):
            executor_class = resolve_obj_by_qualname(distributed_executor_backend)
            if not issubclass(executor_class, Executor):
                raise TypeError(
                    "distributed_executor_backend must be a subclass of "
                    f"Executor. Got {executor_class}."
                )
        else:
            raise ValueError(
                f"Unknown distributed executor backend: {distributed_executor_backend}"
            )
        return executor_class

    @instrument(span_name="Executor init")
    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self._b12x_autotuning_cancel = threading.Event()
        self._init_executor()
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
        self.kv_output_aggregator: KVOutputAggregator | None = None
        self.ec_output_aggregator: ECOutputAggregator | None = None

    @abstractmethod
    def _init_executor(self) -> None:
        raise NotImplementedError

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        """Initialize the KV caches on the underlying workers."""
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))

    def compile_or_warm_up_model(self) -> None:
        """Compile/warm up the model and capture cudagraphs on workers."""
        self._run_b12x_preparation(stage="state")
        compilation_times: list[CompilationTimes] = self.collective_rpc(
            "compile_or_warm_up_model"
        )
        if compilation_times:
            self.vllm_config.compilation_config.compilation_time = max(
                t.language_model for t in compilation_times
            )
            self.vllm_config.compilation_config.encoder_compilation_time = max(
                t.encoder for t in compilation_times
            )

    @contextmanager
    def b12x_warmup_control(self):
        """Keep startup cancellation active across both native preparation stages."""
        from vllm.platforms import current_platform
        from vllm.utils.b12x import has_b12x

        if not (has_b12x() and current_platform.is_cuda()
                and current_platform.is_device_capability_family(120)):
            yield
            return
        if (not self.vllm_config.kernel_config.enable_b12x_autotune
                or os.environ.get("B12X_AUTOTUNE", "1") == "0"):
            self.cancel_b12x_autotuning()
            yield
            return
        from ._b12x_terminal import EscapeKey

        with EscapeKey(self.cancel_b12x_autotuning) as keyboard:
            self._b12x_keyboard = keyboard
            try:
                yield
            finally:
                self._b12x_keyboard = None

    def cancel_b12x_autotuning(self) -> None:
        """Request optional native tuning cancellation at the next bounded round."""
        self._b12x_autotuning_cancel.set()

    def _run_b12x_preparation(self, *, stage: str) -> None:
        """Run ranks independently while a host thread reports progress and cancellation."""
        import pickle
        from datetime import timedelta
        from queue import Empty, SimpleQueue

        import torch.distributed as dist
        from vllm.utils.network_utils import get_ip

        display = output = None
        local_output = SimpleQueue()
        begun = False
        completed = False

        def drain_local_output():
            while True:
                try:
                    line = local_output.get_nowait()
                except Empty:
                    return
                display.write_output(line)

        try:
            begun = True
            outcomes = self.collective_rpc(
                "begin_b12x_preparation",
                kwargs={"stage": stage},
            )
            if any(item.get("native") for item in outcomes):
                from b12x.preparation import PreparationDisplay
                from vllm.utils.system_utils import undecorated_log_stream

                stream = undecorated_log_stream(sys.stderr)
                if stream.isatty():
                    from ._b12x_output import PreparationOutput

                    output = PreparationOutput(local_output.put).start()
                    stream = output.stream or stream
                phase_number = {"weights": 1, "state": 2}[stage]
                display = PreparationDisplay(
                    global_rank=0, stream=stream,
                    title=f"b12x / kernel autotuning (phase {phase_number}/2)",
                    cancel_available=bool(
                        getattr(self, "_b12x_keyboard", None)
                        and self._b12x_keyboard.active
                    ),
                )
                display.__enter__()
            if not all(bool(item.get("done")) for item in outcomes):
                address = get_ip()
                store = dist.TCPStore(
                    address, 0, is_master=True, wait_for_workers=False,
                    timeout=timedelta(seconds=30),
                )
                stopped = threading.Event()
                reporting_errors = []
                ranks = tuple(item["global_rank"] for item in outcomes)
                output_seen = dict.fromkeys(ranks, 0)

                def drain_output():
                    if display is None:
                        return
                    drain_local_output()
                    for rank in ranks:
                        count_key = f"output_count/{rank}"
                        if not store.check([count_key]):
                            continue
                        count = int(store.get(count_key))
                        while output_seen[rank] < count:
                            output_seen[rank] += 1
                            key = f"output/{rank}/{output_seen[rank]}"
                            display.write_output(store.get(key).decode("utf-8"))
                            store.delete_key(key)

                def report_progress():
                    try:
                        while not stopped.wait(0.1):
                            drain_output()
                            if self._b12x_autotuning_cancel.is_set():
                                store.set("cancel", b"1")
                                if display is not None:
                                    display.tuning_stopped()
                            if display is not None:
                                snapshots = [
                                    pickle.loads(store.get(f"progress/{rank}"))
                                    for rank in ranks if store.check([f"progress/{rank}"])
                                ]
                                progress = _aggregate_b12x_progress(snapshots, expected_ranks=ranks)
                                if progress is not None and not progress.done:
                                    display.update(progress)
                    except BaseException as error:
                        reporting_errors.append(error)

                reporter = threading.Thread(target=report_progress, name="b12x-progress", daemon=True)
                reporter.start()
                try:
                    if self._b12x_autotuning_cancel.is_set():
                        store.set("cancel", b"1")
                    outcomes = self.collective_rpc(
                        "run_b12x_preparation",
                        kwargs={
                            "control_address": (address, store.port),
                            "capture_output": output is not None,
                        },
                    )
                finally:
                    stopped.set()
                    reporter.join()
                    drain_output()
                if reporting_errors:
                    raise reporting_errors[0]
                if output is not None:
                    output.stop()
                    drain_local_output()
                if display is not None:
                    progress = _aggregate_b12x_progress(outcomes)
                    if progress is not None:
                        display.update(progress)
                errors = [item["error"] for item in outcomes if item.get("error")]
                if errors:
                    primary = min(errors, key=lambda item: int(item["rank"]))
                    raise RuntimeError(
                        f"b12x preparation failed on rank {primary['rank']}: "
                        f"{primary['type']}: {primary['message']}"
                    )
            completed = True
        except BaseException as error:
            if begun and not completed:
                try:
                    self.collective_rpc("abort_b12x_preparation")
                except BaseException as cleanup_error:
                    error.add_note(
                        f"b12x preparation abort failed: {cleanup_error!r}"
                    )
            raise
        finally:
            try:
                if output is not None:
                    output.stop()
                if display is not None:
                    drain_local_output()
            finally:
                try:
                    if display is not None:
                        display.close(failed=not completed)
                finally:
                    if output is not None:
                        output.close()

    def register_failure_callback(self, callback: FailureCallback):  # noqa: B027
        """
        Register a function to be called if the executor enters a permanent
        failed state.
        """
        pass

    def determine_available_memory(self) -> list[int]:  # in bytes
        self._run_b12x_preparation(stage="weights")
        return self.collective_rpc("determine_available_memory")

    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        return self.collective_rpc("get_kv_cache_spec")

    def get_supported_kv_cache_layouts(self) -> list[list[str]]:
        """Layouts each worker's backends support, most preferred first."""
        return self.collective_rpc("get_supported_kv_cache_layouts")

    def set_kv_cache_layout(self, layout_name: str) -> None:
        """Publish the resolved KV cache layout to the workers."""
        self.collective_rpc("set_kv_cache_layout", args=(layout_name,))

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[False] = False,
    ) -> list[_R]:
        """
        Execute an RPC call on all workers.

        Args:
            method: Name of the worker method to execute, or a callable that
                is serialized and sent to all workers to execute.

                If the method is a callable, it should accept an additional
                `self` argument, in addition to the arguments passed in `args`
                and `kwargs`. The `self` argument will be the worker object.
            timeout: Maximum time in seconds to wait for execution. Raises a
                [`TimeoutError`][] on timeout. `None` means wait indefinitely.
            args: Positional arguments to pass to the worker method.
            kwargs: Keyword arguments to pass to the worker method.
            non_block: If `True`, returns a list of Futures instead of waiting
                for the results.

        Returns:
            A list containing the results from each worker.

        Note:
            It is recommended to use this API to only pass control messages,
            and set up data-plane communication to pass data.
        """
        pass

    @overload
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: Literal[True] = True,
    ) -> Future[list[_R]]:
        pass

    @abstractmethod
    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block: bool = False
    ):
        raise NotImplementedError

    def get_kv_connector_handshake_metadata(
        self,
    ) -> list[dict[tuple[int, int], KVConnectorHandshakeMetadata]]:
        return self.collective_rpc("get_kv_connector_handshake_metadata")

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[False] = False
    ) -> ModelRunnerOutput | None:
        pass

    @overload
    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput | None]:
        pass

    def execute_model(
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[False] = False
    ) -> ModelRunnerOutput:
        pass

    @overload
    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: Literal[True] = True
    ) -> Future[ModelRunnerOutput]:
        pass

    def sample_tokens(
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | Future[ModelRunnerOutput]:
        output = self.collective_rpc(  # type: ignore[call-overload]
            "sample_tokens", args=(grammar_output,), non_block=non_block
        )
        return output[0]

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        output: list[DraftTokenIds] = self.collective_rpc("take_draft_token_ids")
        return output[0]

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        self.collective_rpc("profile", args=(is_start, profile_prefix))

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.collective_rpc(
            "save_sharded_state",
            kwargs=dict(path=path, pattern=pattern, max_size=max_size),
        )

    @abstractmethod
    def check_health(self) -> None:
        """Checks if the executor is healthy. If not, it should raise an
        exception."""
        raise NotImplementedError

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.collective_rpc("shutdown")

    def init_kv_output_aggregator(self, connector: "KVConnectorBase") -> None:
        """Init KVOutputAggregator"""
        self.kv_output_aggregator = KVOutputAggregator.from_connector(
            connector, self.parallel_config.world_size
        )

    def init_ec_output_aggregator(self) -> None:
        self.ec_output_aggregator = ECOutputAggregator()

    @cached_property  # Avoid unnecessary RPC calls
    def supported_tasks(self) -> tuple[SupportedTask, ...]:
        output: list[tuple[SupportedTask, ...]]
        output = self.collective_rpc("get_supported_tasks")
        return output[0]

    def supports_draft_weight_updates(self) -> bool:
        worker_support: list[bool] = self.collective_rpc(
            "supports_draft_weight_updates"
        )
        return all(worker_support)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        assert lora_request.lora_int_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("add_lora", args=(lora_request,)))

    def remove_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("remove_lora", args=(lora_id,)))

    def pin_lora(self, lora_id: int) -> bool:
        assert lora_id > 0, "lora_id must be greater than 0."
        return all(self.collective_rpc("pin_lora", args=(lora_id,)))

    def list_loras(self) -> set[int]:
        sets: list[set[int]] = self.collective_rpc("list_loras")
        for s in sets:
            assert s == sets[0], "All workers should have the same LORAs."
        return sets[0]

    def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache in each worker."""
        self.collective_rpc("reset_mm_cache")

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache in each worker to clear cached encoder outputs."""
        self.collective_rpc("reset_encoder_cache")

    def sleep(self, level: int = 1):
        if self.is_sleeping:
            logger.warning("Executor is already sleeping.")
            return
        time_before_sleep = time.perf_counter()
        self.collective_rpc("sleep", kwargs=dict(level=level))
        time_after_sleep = time.perf_counter()
        self.sleeping_tags = {"weights", "kv_cache"}
        self.is_sleeping = True
        logger.info(
            "It took %.6f seconds to fall asleep.", time_after_sleep - time_before_sleep
        )

    def wake_up(self, tags: list[str] | None = None):
        if not self.is_sleeping:
            logger.warning("Executor is not sleeping.")
            return
        if tags:
            for tag in tags:
                if tag not in self.sleeping_tags:
                    logger.warning(
                        "Tag %s is not in sleeping tags %s", tag, self.sleeping_tags
                    )
                    return
        time_before_wakeup = time.perf_counter()
        self.collective_rpc("wake_up", kwargs=dict(tags=tags))
        time_after_wakeup = time.perf_counter()
        logger.info(
            "It took %.6f seconds to wake up tags %s.",
            time_after_wakeup - time_before_wakeup,
            tags if tags is not None else self.sleeping_tags,
        )
        if tags:
            for tag in tags:
                self.sleeping_tags.remove(tag)
        else:
            self.sleeping_tags.clear()
        if not self.sleeping_tags:
            self.is_sleeping = False

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        raise NotImplementedError

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        """
        Whether the executor supports async scheduling.
        """
        return False


from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    ExecutorWithExternalLauncher as _ExecutorWithExternalLauncher,
)
from vllm.v1.executor.uniproc_executor import (  # noqa: E402
    UniProcExecutor as _UniProcExecutor,
)

# For backwards compatibility.
UniProcExecutor = _UniProcExecutor
ExecutorWithExternalLauncher = _ExecutorWithExternalLauncher
