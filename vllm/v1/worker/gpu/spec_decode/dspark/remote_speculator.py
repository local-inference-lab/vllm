# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier-side proxy for a dedicated RTX 3090 K3 draft process."""

from __future__ import annotations

import ctypes
import json
import math
import os
import queue  # kimi-k3-draft-async-ingest
import threading  # kimi-k3-draft-async-ingest
import time
from dataclasses import dataclass
from pathlib import Path as _Path
from typing import Any

import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.device_communicators.cuda_wrapper import find_loaded_library
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.speculator import (
    BaseSpeculator,
    CUDAGraphCapturePhase,
)
from vllm.v1.worker.gpu.spec_decode.utils import draft_gumbel_pos

logger = init_logger(__name__)

PROTOCOL_VERSION = 2


@dataclass
class _RetainedRequestPrefix:
    token_ids: torch.Tensor
    committed_end: int
    context_start: int
    serial: int


def _build_valid_context_plan(
    input_batch: InputBatch,
    rejected_counts: list[int],
) -> tuple[list[int], list[int]]:
    """Return valid row indices and per-request counts."""
    if len(rejected_counts) != input_batch.num_reqs:
        raise ValueError("Rejected-token count does not match the request batch")
    gather_indices: list[int] = []
    valid_counts: list[int] = []
    offset = 0
    for request_idx, (scheduled, rejected) in enumerate(
        zip(input_batch.num_scheduled_tokens.tolist(), rejected_counts)
    ):
        valid = int(scheduled) - int(rejected)
        if not 0 <= valid <= int(scheduled):
            raise ValueError(
                f"Invalid valid-context length for request {request_idx}: "
                f"scheduled={scheduled}, rejected={rejected}"
            )
        gather_indices.extend(range(offset, offset + valid))
        valid_counts.append(valid)
        offset += int(scheduled)
    return gather_indices, valid_counts


def _anchor_positions_from_context(
    context_counts: list[int], context_positions: torch.Tensor
) -> list[int]:
    """Return the position immediately following each request's context."""
    anchors: list[int] = []
    offset = 0
    for count in context_counts:
        if count <= 0:
            raise ValueError("Every remote draft request requires context rows")
        offset += count
        anchors.append(int(context_positions[offset - 1]) + 1)
    if offset != context_positions.numel():
        raise ValueError("Remote draft context counts do not match the position tensor")
    return anchors


def _contiguous_draft_output(
    draft_tokens: torch.Tensor,
    num_reqs: int,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """Return the active TP-broadcast region with a compact row stride."""
    return draft_tokens[:num_reqs, :num_speculative_tokens].contiguous()


LOGITS_CAPABILITY = "dflash_logits_bf16_v1"


TOPK_LOGITS_CAPABILITY = "dflash_logits_topk_bf16_v1"
# Logit assigned to vocabulary entries outside the draft's transmitted top-k.
# Finite (bf16 -10048) so the logsumexp blocks of the rejection sampler never
# see an all -inf block; exp(-1e4 - max) underflows to exactly zero, so the
# draft distribution is the top-k slice renormalized, i.e. q(x) = 0 outside it.
TOPK_LOGITS_FILL = -1.0e4


def _decode_topk_logits_frame(
    response: dict[str, Any],
    expected_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a top-k draft-logit frame: BF16 values then int32 indices,
    both shaped [requests, K, k]."""
    metadata = response.get("logits")
    frame = response.get("_logits_frame")
    if not isinstance(metadata, dict) or not isinstance(frame, bytes):
        raise ValueError("Remote DFlash response is missing its logits frame")
    if metadata.get("capability") != TOPK_LOGITS_CAPABILITY:
        raise ValueError(f"Unsupported logits capability: {metadata}")
    if metadata.get("dtype") != "bfloat16":
        raise ValueError(f"Unsupported logits dtype: {metadata}")
    if tuple(metadata.get("shape", ())) != expected_shape:
        raise ValueError(
            f"Remote DFlash top-k logits shape mismatch: expected={expected_shape}, "
            f"metadata={metadata}"
        )
    count = math.prod(expected_shape)
    values_nbytes = count * 2
    indices_nbytes = count * 4
    if (
        metadata.get("nbytes_values") != values_nbytes
        or metadata.get("nbytes_indices") != indices_nbytes
        or len(frame) != values_nbytes + indices_nbytes
    ):
        raise ValueError(
            "Remote DFlash top-k logits byte count mismatch: "
            f"expected={values_nbytes}+{indices_nbytes}, metadata={metadata}, "
            f"frame={len(frame)}"
        )
    buffer = bytearray(frame)
    values = (
        torch.frombuffer(buffer, dtype=torch.uint16, count=count)
        .view(torch.bfloat16)
        .reshape(expected_shape)
    )
    indices = torch.frombuffer(
        buffer, dtype=torch.int32, count=count, offset=values_nbytes
    ).reshape(expected_shape)
    return values, indices


def _decode_bfloat16_logits_frame(
    response: dict[str, Any],
    expected_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Decode and validate a BF16 draft-logit multipart frame."""
    metadata = response.get("logits")
    frame = response.get("_logits_frame")
    if not isinstance(metadata, dict) or not isinstance(frame, bytes):
        raise ValueError("Remote DFlash response is missing its logits frame")
    if metadata.get("capability") != LOGITS_CAPABILITY:
        raise ValueError(f"Unsupported logits capability: {metadata}")
    if metadata.get("dtype") != "bfloat16":
        raise ValueError(f"Unsupported logits dtype: {metadata}")
    if tuple(metadata.get("shape", ())) != expected_shape:
        raise ValueError(
            f"Remote DFlash logits shape mismatch: expected={expected_shape}, "
            f"metadata={metadata}"
        )
    expected_nbytes = math.prod(expected_shape) * 2
    if metadata.get("nbytes") != expected_nbytes or len(frame) != expected_nbytes:
        raise ValueError(
            "Remote DFlash logits byte count mismatch: "
            f"expected={expected_nbytes}, metadata={metadata}, frame={len(frame)}"
        )
    return (
        torch.frombuffer(bytearray(frame), dtype=torch.uint16)
        .view(torch.bfloat16)
        .reshape(expected_shape)
    )


# kimi-k3-dflash-feature-capture ------------------------------------------
_K3_CAPTURE_DIR = os.environ.get("VLLM_K3_DRAFT_CAPTURE_DIR", "").strip()
_K3_CAPTURE_STATE: dict[str, dict] = {}


def _k3_capture_safe_name(request_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in request_id)[:180]


def _k3_capture_new_state(
    directory: _Path, request_id: str, width: int
) -> dict[str, Any]:
    """Create empty binary and JSON capture files for one request."""
    name = _k3_capture_safe_name(request_id)
    state = {
        "request_id": request_id,
        "width": width,
        "bytes": 0,
        "record_count": 0,
        "bin": str(directory / (name + ".bin")),
        "index": str(directory / (name + ".json")),
    }
    metadata = {
        key: state[key] for key in ("request_id", "width", "bin", "index")
    }
    prefix = json.dumps(metadata, separators=(",", ":"))[:-1] + ',"records":['
    tail = '],"bytes":0}'
    _Path(state["bin"]).write_bytes(b"")
    _Path(state["index"]).write_text(prefix + tail)
    state["index_tail_bytes"] = len(tail.encode())
    return state


def _k3_capture_append_index(
    state: dict[str, Any], record: dict[str, Any], total_bytes: int
) -> None:
    """Append one record while keeping the compact index valid JSON."""
    separator = "," if state["record_count"] else ""
    tail = f'],"bytes":{total_bytes}}}'
    payload = (separator + json.dumps(record, separators=(",", ":")) + tail).encode()
    with _Path(state["index"]).open("r+b") as handle:
        handle.seek(-int(state["index_tail_bytes"]), os.SEEK_END)
        handle.write(payload)
        handle.truncate()
    state["bytes"] = total_bytes
    state["record_count"] += 1
    state["index_tail_bytes"] = len(tail.encode())


def _k3_capture_records(
    requests: list[dict],
    positions_frame: bytes,
    context_frame: bytes,
    width: int,
    anchor_positions: list[int],
) -> None:
    """Split one PROPOSE payload back into its per-request row ranges."""
    directory = _Path(_K3_CAPTURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    row_bytes = width * 2
    row_offset = 0
    for request, anchor_position in zip(requests, anchor_positions):
        count = int(request["context_count"])
        if count <= 0:
            continue
        request_id = str(request["request_id"])
        state = _K3_CAPTURE_STATE.get(request_id)
        if state is None or bool(request["reset"]):
            state = _k3_capture_new_state(directory, request_id, width)
            _K3_CAPTURE_STATE[request_id] = state
        positions_slice = positions_frame[row_offset * 8 : (row_offset + count) * 8]
        context_slice = context_frame[
            row_offset * row_bytes : (row_offset + count) * row_bytes
        ]
        with open(state["bin"], "ab") as handle:
            handle.write(positions_slice)
            handle.write(context_slice)
        record = {
            "offset": state["bytes"],
            "rows": count,
            "positions_bytes": len(positions_slice),
            "context_bytes": len(context_slice),
            "reset": bool(request["reset"]),
            "reset_position": int(request["reset_position"]),
            "anchor_token_id": int(request["anchor_token_id"]),
            "anchor_position": int(anchor_position),
        }
        total_bytes = state["bytes"] + len(positions_slice) + len(context_slice)
        _k3_capture_append_index(state, record, total_bytes)
        row_offset += count


class _StreamGate:
    """Parks a CUDA stream until a reply thread publishes.

    ``arm(stream)`` enqueues ``cudaLaunchHostFunc(pthread_barrier_wait)`` on
    the stream: work enqueued afterwards runs only after ``release()`` is
    called from another thread. The two-party barrier resets itself after
    every cycle, so one gate serves one proposal at a time; every ``arm``
    must be matched by exactly one ``release`` (a reply thread releases in
    its ``finally`` clause). ``pthread_barrier_wait`` cannot return EINTR,
    so a signal delivered to the driver thread cannot open the gate early.
    """

    _BARRIER_BYTES = 64

    def __init__(self) -> None:
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.pthread_barrier_init.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self._libc.pthread_barrier_wait.argtypes = [ctypes.c_void_p]
        self._barrier = ctypes.create_string_buffer(self._BARRIER_BYTES)
        if self._libc.pthread_barrier_init(self._barrier, None, 2) != 0:
            raise OSError(ctypes.get_errno(), "pthread_barrier_init failed")
        self._wait_fn = ctypes.cast(self._libc.pthread_barrier_wait, ctypes.c_void_p)
        cudart_path = find_loaded_library("libcudart")
        if cudart_path is None:
            raise RuntimeError("libcudart is not loaded; cannot arm a stream gate")
        self._cudart = ctypes.CDLL(cudart_path)
        self._cudart.cudaLaunchHostFunc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._cudart.cudaLaunchHostFunc.restype = ctypes.c_int

    def arm(self, stream: torch.cuda.Stream) -> None:
        err = self._cudart.cudaLaunchHostFunc(
            ctypes.c_void_p(stream.cuda_stream),
            self._wait_fn,
            ctypes.cast(self._barrier, ctypes.c_void_p),
        )
        if err != 0:
            raise RuntimeError(f"cudaLaunchHostFunc failed with error {err}")

    def release(self) -> None:
        self._libc.pthread_barrier_wait(self._barrier)


@dataclass
class _PendingProposal:
    """A proposal whose reply is consumed by the next step in stream order."""

    epoch: int
    num_reqs: int
    num_speculative_tokens: int
    active_indices: list[int]
    request_ids: list[str]
    idx_mapping: torch.Tensor
    temperature: torch.Tensor
    seeds: torch.Tensor
    thread: threading.Thread | None = None
    failed: bool = False
    timing_ms: dict[str, float] | None = None


class RemoteK3DSparkSpeculator(BaseSpeculator):
    """Forward target auxiliary states to a standalone draft server."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        address: str,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = str(self.speculative_config.method)
        if self.method not in ("dspark", "dflash"):
            raise ValueError(f"Unsupported remote K3 draft method: {self.method}")
        sample_method = self.speculative_config.draft_sample_method
        if sample_method not in ("greedy", "probabilistic"):
            raise ValueError(f"Unsupported remote draft sample method: {sample_method}")
        self._probabilistic = sample_method == "probabilistic"
        if self._probabilistic and self.method != "dflash":
            raise ValueError("Remote probabilistic sampling supports DFlash only")
        if self.speculative_config.rejection_sample_method != "block":
            raise ValueError(
                "Remote K3 draft currently requires block rejection sampling"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Remote K3 DSpark transport currently requires BF16")
        self.num_speculative_steps = int(self.speculative_config.num_speculative_tokens)
        self.max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        draft_hf_config = self.speculative_config.draft_model_config.hf_config
        aux_layers = get_eagle3_aux_layers_from_config(self.speculative_config)
        if not aux_layers:
            raise ValueError(
                f"Remote K3 {self.method} config does not declare auxiliary layers"
            )
        self.num_aux_layers = len(aux_layers)
        target_hidden_size = int(
            getattr(draft_hf_config, "target_hidden_size", None)
            or draft_hf_config.hidden_size
        )
        self.raw_context_width = int(target_hidden_size * self.num_aux_layers)
        self.vocab_size = int(draft_hf_config.vocab_size)
        self.use_fp64_gumbel = bool(vllm_config.model_config.use_fp64_gumbel)
        self.address = address
        self.timeout_ms = int(
            os.environ.get(
                "VLLM_K3_DRAFT_REMOTE_TIMEOUT_MS",
                os.environ.get("VLLM_K3_DSPARK_REMOTE_TIMEOUT_MS", "30000"),
            )
        )
        self.supports_mm_inputs = False
        self.draft_logits: torch.Tensor | None = None
        self._remote_logits: torch.Tensor | None = None
        self._remote_sample_positions: torch.Tensor | None = None
        # Top-k draft distribution transport (VLLM_K3_DRAFT_LOGITS_TOPK=k > 0):
        # the draft returns its k largest logits per position instead of the
        # full vocabulary; every rank scatters them over TOPK_LOGITS_FILL into
        # its dense logits buffer, so the draft is sampled from, and verified
        # against, the same truncated distribution. Rejection sampling stays
        # exact with respect to the target for any draft distribution; only
        # the acceptance rate depends on the mass outside the top-k.
        self._logits_topk = int(os.environ.get("VLLM_K3_DRAFT_LOGITS_TOPK", "0"))
        if self._logits_topk < 0:
            raise ValueError("VLLM_K3_DRAFT_LOGITS_TOPK must be >= 0")
        self._remote_topk_values: torch.Tensor | None = None
        self._remote_topk_indices: torch.Tensor | None = None
        if self._probabilistic:
            head_dtype = vllm_config.model_config.head_dtype
            if head_dtype != torch.bfloat16:
                raise ValueError(
                    "Remote probabilistic DFlash transport requires a BF16 head, "
                    f"got {head_dtype}"
                )
            shape = (
                self.max_num_reqs,
                self.num_speculative_steps,
                self.vocab_size,
            )
            self.draft_logits = torch.zeros(shape, dtype=head_dtype, device=device)
            self._remote_logits = torch.zeros(shape, dtype=head_dtype, device=device)
            self._remote_sample_positions = torch.full(
                shape[:2],
                -1,
                dtype=torch.int64,
                device=device,
            )
            if self._logits_topk > 0:
                if self._logits_topk > self.vocab_size:
                    raise ValueError(
                        "VLLM_K3_DRAFT_LOGITS_TOPK exceeds the draft vocabulary: "
                        f"{self._logits_topk} > {self.vocab_size}"
                    )
                topk_shape = (
                    self.max_num_reqs,
                    self.num_speculative_steps,
                    self._logits_topk,
                )
                self._remote_topk_values = torch.full(
                    topk_shape, TOPK_LOGITS_FILL, dtype=head_dtype, device=device
                )
                self._remote_topk_indices = torch.zeros(
                    topk_shape, dtype=torch.int64, device=device
                )
        self.draft_tokens = torch.full(
            (self.max_num_reqs, self.num_speculative_steps),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self._known_requests: set[str] = set()
        self._disabled_requests: set[str] = set()
        self._active_requests: set[str] = set()
        self._retained_prefixes: dict[str, _RetainedRequestPrefix] = {}
        self._retained_serial = 0
        self._remote_max_requests = self.max_num_reqs
        self._remote_block_size = 1
        self._remote_window_size = 0
        self._remote_prefix_cache_tokens = 0
        self._timing_log_interval = int(
            os.environ.get("VLLM_K3_DRAFT_TIMING_LOG_INTERVAL", "0")
        )
        if self._timing_log_interval < 0:
            raise ValueError("VLLM_K3_DRAFT_TIMING_LOG_INTERVAL must be >= 0")
        self._timing_count = 0
        self._timing_totals_ms: dict[str, float] = {}
        self._deferred_resolve = (
            os.environ.get("VLLM_K3_DRAFT_DEFERRED_RESOLVE", "1") == "1"
            and not _K3_CAPTURE_DIR
        )
        # Set by the model runner before propose(): False when the scheduler
        # needs the real draft token ids on the host after this step.
        self.deferred_resolve_allowed = True
        self._pending: _PendingProposal | None = None
        self._pending_failure: tuple[BaseException, list[str]] | None = None
        self._pending_epoch = 0
        self._last_resolved: _PendingProposal | None = None

        tp_group = get_tp_group()
        self._tp_group = tp_group
        self._tp_rank = int(tp_group.rank_in_group)
        self._cpu_token_broadcast = (
            os.environ.get("VLLM_K3_DRAFT_CPU_TOKEN_BROADCAST", "0") == "1"
        )
        self._cpu_draft_tokens = {
            depth: torch.empty(
                (self.max_num_reqs, depth),
                dtype=torch.int64,
                pin_memory=True,
            )
            for depth in range(1, self.num_speculative_steps + 1)
        }
        self._cpu_broadcast_count = 0
        self._cpu_broadcast_total_ms = 0.0
        if self._cpu_token_broadcast:
            logger.info_once(
                "Remote K3 draft tokens use the CPU process group before H2D"
            )
        self._zmq_context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        if self._tp_rank == 0:
            # P2P is unavailable on the target host. Keep the mandatory D2H
            # hop off pageable memory so it can be queued directly after the
            # target forward on the current stream.
            self._positions_staging = torch.empty(
                self.max_num_tokens,
                dtype=torch.int64,
                pin_memory=True,
            )
            self._context_staging = torch.empty(
                (self.max_num_tokens, self.raw_context_width),
                dtype=vllm_config.model_config.dtype,
                pin_memory=True,
            )
            self._rejected_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                pin_memory=True,
            )
            self._anchor_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int64,
                pin_memory=True,
            )
            # kimi-k3-draft-async-ingest: begin
            self._sampled_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                pin_memory=True,
            )
            self._async_depth = int(
                os.environ.get("VLLM_K3_DRAFT_ASYNC_PREFILL_INGEST", "2") or 0
            )
            self._async_ring: list[tuple[torch.Tensor, torch.Tensor]] = []
            self._async_slot_free: list[threading.Event] = []
            self._async_slot = 0
            self._async_queue: queue.Queue = queue.Queue()
            self._async_errors: list[tuple[BaseException, list[str]]] = []
            self._async_error_lock = threading.Lock()
            self._async_lock = threading.Lock()
            self._async_stats = {"deferred": 0, "deferred_ms": 0.0}
            for _ in range(self._async_depth):
                self._async_ring.append(
                    (
                        torch.empty(
                            self.max_num_tokens,
                            dtype=torch.int64,
                            pin_memory=True,
                        ),
                        torch.empty(
                            (self.max_num_tokens, self.raw_context_width),
                            dtype=vllm_config.model_config.dtype,
                            pin_memory=True,
                        ),
                    )
                )
                free = threading.Event()
                free.set()
                self._async_slot_free.append(free)
            if self._async_depth > 0:
                threading.Thread(
                    target=self._async_ingest_worker,
                    name="k3-draft-async-ingest",
                    daemon=True,
                ).start()
                logger.info(
                    "Remote K3 %s deferred prefill ingest enabled: ring depth %d",
                    self.method,
                    self._async_depth,
                )
            # kimi-k3-draft-async-ingest: end
            if self._probabilistic and self._logits_topk > 0:
                topk_shape = (
                    self.max_num_reqs,
                    self.num_speculative_steps,
                    self._logits_topk,
                )
                self._topk_values_staging = torch.empty(
                    topk_shape, dtype=torch.bfloat16, pin_memory=True
                )
                self._topk_indices_staging = torch.empty(
                    topk_shape, dtype=torch.int32, pin_memory=True
                )
            elif self._probabilistic:
                self._logits_staging = torch.empty(
                    (
                        self.max_num_reqs,
                        self.num_speculative_steps,
                        self.vocab_size,
                    ),
                    dtype=torch.bfloat16,
                    pin_memory=True,
                )
            # Deferred resolve (VLLM_K3_DRAFT_DEFERRED_RESOLVE=1): the RPC runs
            # on a reply thread while the runner prepares the next step; the
            # next step's stream is parked on the gate until the reply is
            # staged. Rank 0 stages every reply field through pinned memory
            # so the copies enqueued behind the gate never touch pageable
            # memory (a pageable H2D copy would synchronize the host with the
            # parked stream).
            self._tokens_staging = [
                torch.full(
                    (self.max_num_reqs, self.num_speculative_steps),
                    -1,
                    dtype=torch.int64,
                    pin_memory=True,
                )
                for _ in range(2)
            ]
            self._sample_positions_staging = [
                torch.full(
                    (self.max_num_reqs, self.num_speculative_steps),
                    -1,
                    dtype=torch.int64,
                    pin_memory=True,
                )
                for _ in range(2)
            ]
            self._staging_slot = 0
            self._gate = _StreamGate() if self._deferred_resolve else None
            self._zmq_context = zmq.Context()
            self._connect()
            response = self._rpc(
                [json.dumps({"protocol": PROTOCOL_VERSION, "op": "PING"}).encode()]
            )
            if response.get("op") != "PONG":
                raise RuntimeError(f"Unexpected K3 draft health response: {response}")
            if response.get("method") != self.method:
                raise RuntimeError(
                    "Remote K3 draft method mismatch: "
                    f"target={self.method}, server={response.get('method')}"
                )
            capabilities = response.get("capabilities", [])
            if self._probabilistic and LOGITS_CAPABILITY not in capabilities:
                raise RuntimeError(
                    "Remote DFlash server does not advertise probabilistic logits"
                )
            self._remote_max_requests = int(
                response.get("max_requests", self.max_num_reqs)
            )
            self._remote_block_size = int(response.get("block_size", 1))
            self._remote_window_size = int(response.get("window_size", 0))
            self._remote_prefix_cache_tokens = int(
                response.get("prefix_cache_tokens", 0)
            )
            if int(response.get("active_requests", 0)):
                # A restarted verifier cannot safely identify state left by an
                # older process, so establish a clean protocol epoch.
                self._rpc(
                    [json.dumps({"protocol": PROTOCOL_VERSION, "op": "CLEAR"}).encode()]
                )

        logger.info(
            "Remote K3 %s proxy initialized: address=%s, TP rank=%d, K=%d",
            self.method,
            address,
            self._tp_rank,
            self.num_speculative_steps,
        )

    def _connect(self) -> None:
        assert self._zmq_context is not None
        if self._socket is not None:
            self._socket.close()
        socket = self._zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.address)
        self._socket = socket

    # kimi-k3-draft-async-ingest: begin
    def _drain_async_ingest(self) -> None:
        pending = getattr(self, "_async_queue", None)
        if pending is not None:
            pending.join()

    def _apply_async_error(self) -> None:
        with self._async_error_lock:
            errors = self._async_errors
            self._async_errors = []
        if not errors:
            return
        for exc, request_ids in errors:
            self._disabled_requests.update(request_ids)
            logger.warning(
                "Remote K3 %s deferred prefill ingest failed (%s); drafting is "
                "disabled for %d request(s) until they leave the batch",
                self.method,
                exc,
                len(request_ids),
            )

    def _async_ingest_worker(self) -> None:
        while True:
            job = self._async_queue.get()
            slot, frames, request_ids, enqueued = job
            try:
                with self._async_lock:
                    try:
                        self._socket.send_multipart(frames, copy=False)
                        response_frames = self._socket.recv_multipart()
                    except Exception:
                        self._connect()
                        raise
                response = json.loads(response_frames[0])
                if not isinstance(response, dict) or not response.get("ok", False):
                    raise RuntimeError(f"K3 draft deferred ingest failed: {response}")
                self._async_stats["deferred_ms"] += (
                    time.perf_counter() - enqueued
                ) * 1000
            except Exception as exc:  # noqa: BLE001
                logger.exception("Remote K3 deferred prefill ingest raised")
                with self._async_error_lock:
                    self._async_errors.append((exc, list(request_ids)))
            finally:
                self._async_slot_free[slot].set()
                self._async_queue.task_done()
    # kimi-k3-draft-async-ingest: end

    def _rpc(self, frames: list[bytes]) -> dict[str, Any]:
        assert self._socket is not None
        self._drain_async_ingest()
        with self._async_lock:
            try:
                self._socket.send_multipart(frames, copy=False)
                response_frames = self._socket.recv_multipart()
            except Exception:
                self._connect()
                raise
        if not 1 <= len(response_frames) <= 2:
            raise RuntimeError(
                f"K3 draft RPC returned {len(response_frames)} response frames"
            )
        response = json.loads(response_frames[0])
        if len(response_frames) == 2:
            response["_logits_frame"] = response_frames[1]
        if not isinstance(response, dict) or not response.get("ok", False):
            raise RuntimeError(f"K3 DSpark RPC failed: {response}")
        if int(response.get("protocol", -1)) != PROTOCOL_VERSION:
            raise RuntimeError(f"K3 DSpark protocol mismatch: {response}")
        return response

    def init_cudagraph_manager(self, cudagraph_mode=None) -> None:
        """The standalone drafter owns its CUDA graph lifecycle."""

    def capture(self, *, capture_phase: CUDAGraphCapturePhase) -> None:
        """The verifier has no local draft graph to capture."""

    def _free_remote_requests(self, request_ids: set[str] | list[str]) -> None:
        remote_request_ids = sorted(set(request_ids) & self._known_requests)
        if not remote_request_ids:
            return
        self._rpc(
            [
                json.dumps(
                    {
                        "protocol": PROTOCOL_VERSION,
                        "op": "FREE",
                        "request_ids": remote_request_ids,
                    }
                ).encode()
            ]
        )
        self._known_requests.difference_update(remote_request_ids)
        for request_id in remote_request_ids:
            self._retained_prefixes.pop(request_id, None)
            _K3_CAPTURE_STATE.pop(request_id, None)

    def _ensure_remote_capacity(self, current_request_ids: set[str]) -> None:
        while len(self._known_requests) >= self._remote_max_requests:
            candidates = self._known_requests - current_request_ids
            if not candidates:
                raise RuntimeError(
                    "Remote DSpark request capacity is exhausted by active requests"
                )
            request_id = min(
                candidates,
                key=lambda req_id: (
                    self._retained_prefixes[req_id].serial
                    if req_id in self._retained_prefixes
                    else -1
                ),
            )
            self._free_remote_requests({request_id})

    @staticmethod
    def _token_prefix(
        input_batch: InputBatch,
        request_idx: int,
        prefix_end: int,
    ) -> torch.Tensor | None:
        token_table = input_batch.all_token_ids_cpu
        if token_table is None or prefix_end < 0:
            return None
        state_idx = int(input_batch.idx_mapping_np[request_idx])
        if state_idx < 0 or prefix_end > token_table.shape[1]:
            return None
        return token_table[state_idx, :prefix_end]

    def _can_restore_prefix(
        self,
        retained: _RetainedRequestPrefix,
        prefix_end: int,
    ) -> bool:
        if (
            prefix_end <= 0
            or retained.committed_end < prefix_end
            or self._remote_window_size <= 0
            or self._remote_prefix_cache_tokens < self._remote_window_size
        ):
            return False
        restore_start = max(0, prefix_end - self._remote_window_size)
        restore_start = (
            restore_start // self._remote_block_size * self._remote_block_size
        )
        retained_start = max(
            retained.context_start,
            retained.committed_end - self._remote_prefix_cache_tokens,
        )
        if restore_start >= retained_start:
            return True
        # reconnect-partial-window: the source begins after the ideal window
        # origin because it was itself cold-bootstrapped from a cache-restored
        # prefix.  Every conversation shorter than the window would otherwise
        # be refused forever.  The server clamps the restore origin up to what
        # its projected cache holds, which still hands the draft more history
        # than a fresh bootstrap at prefix_end, so accept any source retaining
        # a non-empty block-aligned range that ends at or after prefix_end.
        aligned_start = (
            (retained_start + self._remote_block_size - 1)
            // self._remote_block_size
            * self._remote_block_size
        )
        return aligned_start < prefix_end

    def _find_reconnect_source(
        self,
        token_prefix: torch.Tensor,
        prefix_end: int,
        current_request_ids: set[str],
    ) -> str | None:
        candidates: list[tuple[int, int, str]] = []
        rejected: list[str] = []
        for request_id in self._known_requests - current_request_ids:
            retained = self._retained_prefixes.get(request_id)
            if retained is None:
                rejected.append(f"{request_id}:no-retained")
                continue
            if not self._can_restore_prefix(retained, prefix_end):
                rejected.append(
                    f"{request_id}:gate(committed={retained.committed_end},"
                    f"ctx={retained.context_start},need={prefix_end})"
                )
                continue
            if torch.equal(retained.token_ids[:prefix_end], token_prefix):
                candidates.append(
                    (
                        retained.committed_end - prefix_end,
                        -retained.serial,
                        request_id,
                    )
                )
            else:
                kept = retained.token_ids
                limit = min(prefix_end, kept.shape[0], token_prefix.shape[0])
                neq = (kept[:limit] != token_prefix[:limit]).nonzero()
                where = int(neq[0].item()) if neq.numel() else -limit
                rejected.append(f"{request_id}:tok@{where}")
        if not candidates and rejected:
            logger.info(
                "Remote K3 %s reconnect diagnostics: prefix_end=%d rejected=%s",
                self.method,
                prefix_end,
                rejected[:6],
            )
        return min(candidates)[2] if candidates else None

    def _reconnect_request(
        self,
        source_request_id: str,
        request_id: str,
        prefix_end: int,
        token_prefix: torch.Tensor,
    ) -> bool:
        try:
            response = self._rpc(
                [
                    json.dumps(
                        {
                            "protocol": PROTOCOL_VERSION,
                            "op": "RECONNECT",
                            "source_request_id": source_request_id,
                            "request_id": request_id,
                            "prefix_end": prefix_end,
                        }
                    ).encode()
                ]
            )
        except Exception:
            logger.exception(
                "Remote K3 DSpark prefix reconnect failed: source=%s, "
                "request=%s, prefix_end=%d",
                source_request_id,
                request_id,
                prefix_end,
            )
            return False

        self._retained_prefixes.pop(source_request_id)
        self._known_requests.discard(source_request_id)
        self._known_requests.add(request_id)
        self._retained_serial += 1
        self._retained_prefixes[request_id] = _RetainedRequestPrefix(
            token_ids=token_prefix.clone(),
            committed_end=prefix_end,
            context_start=int(response.get("restored_start", 0)),
            serial=self._retained_serial,
        )
        logger.info(
            "Remote K3 DSpark prefix reconnected: source=%s, request=%s, "
            "prefix_end=%d, restored_start=%s, latency_ms=%.1f",
            source_request_id,
            request_id,
            prefix_end,
            response.get("restored_start"),
            float(response.get("latency_ms", 0.0)),
        )
        return True

    def _remember_prefix(
        self,
        input_batch: InputBatch,
        request_idx: int,
        request_id: str,
        committed_end: int,
        context_start: int | None = None,
    ) -> None:
        token_prefix = self._token_prefix(input_batch, request_idx, committed_end)
        if token_prefix is None:
            return
        previous = self._retained_prefixes.get(request_id)
        if context_start is None:
            context_start = previous.context_start if previous is not None else 0
        self._retained_serial += 1
        self._retained_prefixes[request_id] = _RetainedRequestPrefix(
            token_ids=token_prefix.clone(),
            committed_end=committed_end,
            context_start=context_start,
            serial=self._retained_serial,
        )

    # Deferred resolve -----------------------------------------------------

    def has_pending_proposal(self) -> bool:
        return self._pending is not None

    def _join_reply_thread(self) -> None:
        resolved = getattr(self, "_last_resolved", None)
        if resolved is None:
            return
        self._last_resolved = None
        if resolved.thread is not None:
            resolved.thread.join()
        if resolved.timing_ms:
            self._record_timing(resolved.timing_ms)

    def _apply_pending_failure(self) -> None:
        failure = self._pending_failure
        if failure is None:
            return
        self._pending_failure = None
        exc, request_ids = failure
        self._disabled_requests.update(request_ids)
        logger.warning(
            "Remote K3 %s deferred proposal failed (%s); drafting is disabled "
            "for %d request(s) until they leave the batch",
            self.method,
            exc,
            len(request_ids),
        )

    def _stage_tokens_from_response(
        self,
        response: dict[str, Any],
        active_indices: list[int],
        num_speculative_tokens: int,
        slot: int,
    ) -> None:
        tokens = response.get("tokens")
        expected_shape = (len(active_indices), num_speculative_tokens)
        if (
            not isinstance(tokens, list)
            or len(tokens) != expected_shape[0]
            or any(
                not isinstance(row, list) or len(row) != expected_shape[1]
                for row in tokens
            )
        ):
            raise ValueError(
                f"Remote DSpark token response has the wrong shape; "
                f"expected={expected_shape}, got={tokens!r}"
            )
        self._tokens_staging[slot][: len(active_indices), :num_speculative_tokens] = (
            torch.tensor(tokens, dtype=torch.int64)
        )

    def _stage_positions_from_response(
        self,
        response: dict[str, Any],
        active_indices: list[int],
        num_speculative_tokens: int,
        slot: int,
    ) -> None:
        sample_positions = response["logits"].get("sample_positions")
        expected_positions = (len(active_indices), num_speculative_tokens)
        if (
            not isinstance(sample_positions, list)
            or len(sample_positions) != expected_positions[0]
            or any(
                not isinstance(row, list) or len(row) != expected_positions[1]
                for row in sample_positions
            )
        ):
            raise ValueError(
                "Remote DFlash sample positions have the wrong shape: "
                f"expected={expected_positions}, got={sample_positions!r}"
            )
        self._sample_positions_staging[slot][
            : len(active_indices), :num_speculative_tokens
        ] = torch.tensor(sample_positions, dtype=torch.int64)

    def _stage_logits_from_response(
        self,
        response: dict[str, Any],
        active_indices: list[int],
        num_speculative_tokens: int,
        slot: int,
    ) -> None:
        rows = len(active_indices)
        if self._logits_topk > 0:
            values, indices = _decode_topk_logits_frame(
                response, (rows, num_speculative_tokens, self._logits_topk)
            )
            self._topk_values_staging[:rows, :num_speculative_tokens].copy_(values)
            self._topk_indices_staging[:rows, :num_speculative_tokens].copy_(indices)
        else:
            logits = _decode_bfloat16_logits_frame(
                response, (rows, num_speculative_tokens, self.vocab_size)
            )
            self._logits_staging[:rows, :num_speculative_tokens].copy_(logits)
        self._stage_positions_from_response(
            response, active_indices, num_speculative_tokens, slot
        )

    def _stage_no_draft(self, pending: _PendingProposal, slot: int) -> None:
        """Stage the reply of a failed proposal: no draft for any request."""
        rows = len(pending.active_indices)
        k = pending.num_speculative_tokens
        self._tokens_staging[slot][:rows, :k].fill_(-1)
        self._sample_positions_staging[slot][:rows, :k].fill_(-1)
        if self._probabilistic:
            if self._logits_topk > 0:
                self._topk_values_staging[:rows, :k].fill_(TOPK_LOGITS_FILL)
                self._topk_indices_staging[:rows, :k].zero_()
            else:
                self._logits_staging[:rows, :k].zero_()

    def _start_reply_thread(
        self,
        frames: list[Any],
        active_indices: list[int],
        request_ids: list[str],
        num_speculative_tokens: int,
        input_batch: InputBatch,
        temperature: torch.Tensor | None,
        seeds: torch.Tensor | None,
    ) -> None:
        assert temperature is not None and seeds is not None
        slot = self._staging_slot
        self._staging_slot = (slot + 1) % len(self._tokens_staging)
        self._pending_epoch += 1
        pending = _PendingProposal(
            epoch=self._pending_epoch,
            num_reqs=input_batch.num_reqs,
            num_speculative_tokens=num_speculative_tokens,
            active_indices=active_indices,
            request_ids=request_ids,
            idx_mapping=input_batch.idx_mapping[: input_batch.num_reqs],
            temperature=temperature,
            seeds=seeds,
        )
        pending.active_gpu = torch.tensor(  # type: ignore[attr-defined]
            active_indices, dtype=torch.int64, device=self.device
        )
        pending.slot = slot  # type: ignore[attr-defined]
        thread = threading.Thread(
            target=self._reply_worker,
            args=(frames, pending, slot),
            name="k3-draft-reply",
            daemon=True,
        )
        pending.thread = thread
        self._pending = pending
        thread.start()

    def _reply_worker(
        self, frames: list[Any], pending: _PendingProposal, slot: int
    ) -> None:
        assert self._gate is not None
        started = time.perf_counter()
        try:
            response = self._rpc(frames)
            rpc_done = time.perf_counter()
            self._stage_tokens_from_response(
                response, pending.active_indices, pending.num_speculative_tokens, slot
            )
            if self._probabilistic:
                self._stage_logits_from_response(
                    response,
                    pending.active_indices,
                    pending.num_speculative_tokens,
                    slot,
                )
            timing_ms = {
                "reply_rpc": (rpc_done - started) * 1000,
                "reply_stage": (time.perf_counter() - rpc_done) * 1000,
            }
            server_timing = response.get("timing_ms")
            if isinstance(server_timing, dict):
                for key, value in server_timing.items():
                    if isinstance(value, int | float):
                        timing_ms[f"server_{key}"] = float(value)
            pending.timing_ms = timing_ms
        except Exception as exc:  # noqa: BLE001
            logger.exception("Remote K3 deferred proposal raised")
            pending.failed = True
            self._pending_failure = (exc, list(pending.request_ids))
            self._stage_no_draft(pending, slot)
        finally:
            self._gate.release()

    def resolve_pending(self) -> torch.Tensor | None:
        """Enqueue the consumption of the pending proposal on the current stream.

        Rank 0 parks the stream on the gate, then copies the staged reply to
        the device; every rank broadcasts the frames, samples the draft and
        returns the ``[num_reqs, K]`` draft tokens. Nothing here waits on the
        host: the caller keeps enqueuing work that does not read the draft
        tokens, and the stream itself waits for the reply.

        Returns:
            The draft token tensor for the proposal, or ``None`` when no
            proposal is pending.
        """
        if self._pending is None:
            return None
        return self._resolve_pending_locked()

    def _resolve_pending_locked(self) -> torch.Tensor:
        pending = self._pending
        assert pending is not None
        self._pending = None
        num_reqs = pending.num_reqs
        active_k = pending.num_speculative_tokens
        if self._tp_rank == 0 and pending.thread is not None:
            assert self._gate is not None
            self._last_resolved = pending
            slot = pending.slot  # type: ignore[attr-defined]
            active_gpu = pending.active_gpu  # type: ignore[attr-defined]
            rows = len(pending.active_indices)
            if torch.cuda.is_current_stream_capturing():
                # A host function cannot be captured: open the gate on a side
                # stream and wait for the reply on the host instead
                # (proposals are never issued during capture, so this only
                # covers a stale proposal).
                side_stream = torch.cuda.Stream(self.device)
                self._gate.arm(side_stream)
                pending.thread.join()
                side_stream.synchronize()
            else:
                self._gate.arm(torch.cuda.current_stream(self.device))
            self.draft_tokens[:, :active_k].index_copy_(
                0,
                active_gpu,
                self._tokens_staging[slot][:rows, :active_k].to(
                    self.device, non_blocking=True
                ),
            )
            if self._probabilistic:
                assert self._remote_sample_positions is not None
                if self._logits_topk > 0:
                    assert self._remote_topk_values is not None
                    assert self._remote_topk_indices is not None
                    self._remote_topk_values[:, :active_k].index_copy_(
                        0,
                        active_gpu,
                        self._topk_values_staging[:rows, :active_k].to(
                            self.device, non_blocking=True
                        ),
                    )
                    self._remote_topk_indices[:, :active_k].index_copy_(
                        0,
                        active_gpu,
                        self._topk_indices_staging[:rows, :active_k]
                        .to(self.device, non_blocking=True)
                        .to(torch.int64),
                    )
                else:
                    assert self._remote_logits is not None
                    self._remote_logits[:, :active_k].index_copy_(
                        0,
                        active_gpu,
                        self._logits_staging[:rows, :active_k].to(
                            self.device, non_blocking=True
                        ),
                    )
                self._remote_sample_positions[:, :active_k].index_copy_(
                    0,
                    active_gpu,
                    self._sample_positions_staging[slot][:rows, :active_k].to(
                        self.device, non_blocking=True
                    ),
                )
        if self._probabilistic:
            self._broadcast_remote_logits(active_k)
            self._sample_remote_probabilistic_masked(
                pending.idx_mapping,
                num_reqs,
                pending.temperature,
                pending.seeds,
                active_k,
            )
        output = _contiguous_draft_output(self.draft_tokens, num_reqs, active_k)
        self._tp_group.broadcast(output, src=0)
        return output

    def _broadcast_remote_logits(self, active_k: int) -> None:
        assert self._remote_logits is not None
        assert self._remote_sample_positions is not None
        if self._logits_topk > 0:
            assert self._remote_topk_values is not None
            assert self._remote_topk_indices is not None
            self._tp_group.broadcast(self._remote_topk_values, src=0)
            self._tp_group.broadcast(self._remote_topk_indices, src=0)
            self._materialize_topk_logits(active_k)
        else:
            self._tp_group.broadcast(self._remote_logits, src=0)
        self._tp_group.broadcast(self._remote_sample_positions, src=0)

    def _sample_remote_probabilistic_masked(
        self,
        idx_mapping: torch.Tensor,
        num_reqs: int,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_speculative_tokens: int,
    ) -> None:
        """Sample every (request, step) slot without a host round trip.

        Slots whose sample position is negative carry no draft: they sample
        with request index -1, which keeps the Gumbel kernel from writing
        their logits cache row, and their token stays -1.
        """
        assert self.draft_logits is not None
        assert self._remote_logits is not None
        assert self._remote_sample_positions is not None
        positions = self._remote_sample_positions[:num_reqs, :num_speculative_tokens]
        valid = positions >= 0
        rows = torch.arange(num_reqs, device=self.device).repeat_interleave(
            num_speculative_tokens
        )
        steps = torch.arange(num_speculative_tokens, device=self.device).repeat(
            num_reqs
        )
        request_idx = torch.where(
            valid.reshape(-1), idx_mapping[:num_reqs].index_select(0, rows), -1
        )
        logits = self._remote_logits[:num_reqs, :num_speculative_tokens].reshape(
            num_reqs * num_speculative_tokens, self.vocab_size
        )
        sampled = self._sample_probabilistic_draft(
            logits=logits.contiguous(),
            positions=positions.reshape(-1) - 2,
            idx_mapping=request_idx,
            temperature=temperature,
            seeds=seeds,
            draft_step=steps,
            draft_logits=self.draft_logits,
        )
        tokens = self.draft_tokens[:num_reqs, :num_speculative_tokens]
        tokens.copy_(
            torch.where(valid, sampled.view(num_reqs, num_speculative_tokens), tokens)
        )

    def _copy_tokens_from_response(
        self,
        response: dict[str, Any],
        active_indices: list[int],
        num_speculative_tokens: int,
    ) -> None:
        tokens = response.get("tokens")
        expected_shape = (len(active_indices), num_speculative_tokens)
        if (
            not isinstance(tokens, list)
            or len(tokens) != expected_shape[0]
            or any(
                not isinstance(row, list) or len(row) != expected_shape[1]
                for row in tokens
            )
        ):
            raise ValueError(
                f"Remote DSpark token response has the wrong shape; "
                f"expected={expected_shape}, got={tokens!r}"
            )
        remote_tokens = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        active_gpu = torch.tensor(active_indices, dtype=torch.int64, device=self.device)
        # ``draft_tokens`` is allocated at the configured maximum depth, while
        # adaptive speculation and the per-batch schedule can request a
        # smaller depth for an individual step.  Copy into the matching width
        # instead of requiring every response to have the maximum width.
        self.draft_tokens[:, :num_speculative_tokens].index_copy_(
            0, active_gpu, remote_tokens
        )

    def _copy_logits_from_response(
        self,
        response: dict[str, Any],
        active_indices: list[int],
        num_speculative_tokens: int,
    ) -> None:
        assert self._remote_logits is not None
        assert self._remote_sample_positions is not None
        if self._logits_topk > 0:
            self._copy_topk_logits_from_response(
                response, active_indices, num_speculative_tokens
            )
            return
        expected_shape = (
            len(active_indices),
            num_speculative_tokens,
            self.vocab_size,
        )
        logits = _decode_bfloat16_logits_frame(response, expected_shape)
        sample_positions = response["logits"].get("sample_positions")
        expected_positions = expected_shape[:2]
        if (
            not isinstance(sample_positions, list)
            or len(sample_positions) != expected_positions[0]
            or any(
                not isinstance(row, list) or len(row) != expected_positions[1]
                for row in sample_positions
            )
        ):
            raise ValueError(
                "Remote DFlash sample positions have the wrong shape: "
                f"expected={expected_positions}, got={sample_positions!r}"
            )
        active_gpu = torch.tensor(active_indices, dtype=torch.int64, device=self.device)
        logits_staging = self._logits_staging[
            : len(active_indices), :num_speculative_tokens
        ]
        logits_staging.copy_(logits)
        self._remote_logits.index_copy_(
            0,
            active_gpu,
            logits_staging.to(self.device, non_blocking=True),
        )
        positions = torch.tensor(
            sample_positions,
            dtype=torch.int64,
            device=self.device,
        )
        self._remote_sample_positions.index_copy_(0, active_gpu, positions)

    def _copy_topk_logits_from_response(
        self,
        response: dict[str, Any],
        active_indices: list[int],
        num_speculative_tokens: int,
    ) -> None:
        """Stage the draft's top-k logits (rank 0) for the TP broadcast."""
        assert self._remote_topk_values is not None
        assert self._remote_topk_indices is not None
        assert self._remote_sample_positions is not None
        expected_shape = (
            len(active_indices),
            num_speculative_tokens,
            self._logits_topk,
        )
        values, indices = _decode_topk_logits_frame(response, expected_shape)
        sample_positions = response["logits"].get("sample_positions")
        expected_positions = expected_shape[:2]
        if (
            not isinstance(sample_positions, list)
            or len(sample_positions) != expected_positions[0]
            or any(
                not isinstance(row, list) or len(row) != expected_positions[1]
                for row in sample_positions
            )
        ):
            raise ValueError(
                "Remote DFlash sample positions have the wrong shape: "
                f"expected={expected_positions}, got={sample_positions!r}"
            )
        active_gpu = torch.tensor(active_indices, dtype=torch.int64, device=self.device)
        values_staging = self._topk_values_staging[
            : len(active_indices), :num_speculative_tokens
        ]
        indices_staging = self._topk_indices_staging[
            : len(active_indices), :num_speculative_tokens
        ]
        values_staging.copy_(values)
        indices_staging.copy_(indices)
        self._remote_topk_values[:, :num_speculative_tokens].index_copy_(
            0, active_gpu, values_staging.to(self.device, non_blocking=True)
        )
        self._remote_topk_indices[:, :num_speculative_tokens].index_copy_(
            0,
            active_gpu,
            indices_staging.to(self.device, non_blocking=True).to(torch.int64),
        )
        positions = torch.tensor(
            sample_positions,
            dtype=torch.int64,
            device=self.device,
        )
        self._remote_sample_positions.index_copy_(0, active_gpu, positions)

    def _materialize_topk_logits(self, num_speculative_tokens: int) -> None:
        """Expand the broadcast top-k slice into the dense logits buffer:
        TOPK_LOGITS_FILL everywhere, the draft's values at their indices."""
        assert self._remote_logits is not None
        assert self._remote_topk_values is not None
        assert self._remote_topk_indices is not None
        logits = self._remote_logits[:, :num_speculative_tokens]
        logits.fill_(TOPK_LOGITS_FILL)
        logits.scatter_(
            -1,
            self._remote_topk_indices[:, :num_speculative_tokens],
            self._remote_topk_values[:, :num_speculative_tokens],
        )

    def _sample_remote_probabilistic(
        self,
        input_batch: InputBatch,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_speculative_tokens: int,
    ) -> None:
        assert self.draft_logits is not None
        assert self._remote_logits is not None
        assert self._remote_sample_positions is not None
        if self._logits_topk > 0:
            logger.info_once(
                "Remote K3 DFlash probabilistic sampling is active with top-%d "
                "BF16 draft logits (entries outside the top-k carry logit %g)",
                self._logits_topk,
                TOPK_LOGITS_FILL,
            )
        else:
            logger.info_once(
                "Remote K3 DFlash probabilistic sampling is active with "
                "full-vocabulary BF16 logits"
            )
        num_reqs = input_batch.num_reqs
        positions = self._remote_sample_positions[:num_reqs, :num_speculative_tokens]
        request_rows, draft_steps = torch.where(positions >= 0)
        if request_rows.numel() == 0:
            return
        idx_mapping = input_batch.idx_mapping[:num_reqs].index_select(0, request_rows)
        logits = self._remote_logits[
            request_rows,
            draft_steps,
        ].contiguous()
        sampled = self._sample_probabilistic_draft(
            logits=logits,
            positions=positions[request_rows, draft_steps] - 2,
            idx_mapping=idx_mapping,
            temperature=temperature,
            seeds=seeds,
            draft_step=draft_steps,
            draft_logits=self.draft_logits,
        )
        self.draft_tokens[request_rows, draft_steps] = sampled

    def _sample_probabilistic_draft(
        self,
        logits: torch.Tensor,
        positions: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample remote logits from the standard disjoint draft stream."""
        return gumbel_sample(
            logits,
            idx_mapping,
            temperature,
            seeds,
            draft_gumbel_pos(positions),
            apply_temperature=True,
            logits_cache=draft_logits,
            logits_cache_col=draft_step,
            use_fp64=self.use_fp64_gumbel,
        )

    def _record_timing(self, timing_ms: dict[str, float]) -> None:
        if self._timing_log_interval <= 0:
            return
        self._timing_count += 1
        for key, value in timing_ms.items():
            self._timing_totals_ms[key] = self._timing_totals_ms.get(key, 0.0) + value
        if self._timing_count < self._timing_log_interval:
            return
        means = {
            key: value / self._timing_count
            for key, value in self._timing_totals_ms.items()
        }
        logger.info(
            "Remote K3 %s timing over %d proposals (ms): %s",
            self.method,
            self._timing_count,
            ", ".join(f"{key}={value:.3f}" for key, value in means.items()),
        )
        self._timing_count = 0
        self._timing_totals_ms.clear()

    def _rank0_propose(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        num_speculative_tokens: int,
        *,
        defer_resolve: bool = False,
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
    ) -> None:
        started = time.perf_counter()
        self._join_reply_thread()
        self._apply_pending_failure()
        # kimi-k3-metadata-d2h-detail: drain the stream at entry to measure
        # how much pending GPU/UVA work the runner queued before propose().
        # The measurement is the only reader; the copies below synchronize
        # before any host read.
        if self._timing_log_interval > 0:
            torch.cuda.current_stream(self.device).synchronize()
        self._detail_entry_drain = time.perf_counter() - started
        if aux_hidden_states is None or len(aux_hidden_states) != self.num_aux_layers:
            raise ValueError(
                f"Remote K3 {self.method} requires {self.num_aux_layers} configured "
                "target auxiliary hidden states"
            )

        num_reqs = input_batch.num_reqs
        current_request_ids = set(input_batch.req_ids)
        previous_active_requests = self._active_requests
        self._disabled_requests.intersection_update(current_request_ids)
        idx_mapping = input_batch.idx_mapping[:num_reqs].long()
        sampled_counts = num_sampled[:num_reqs]
        sampled_anchors = last_sampled[idx_mapping, 0]
        prefill_anchors = next_prefill_tokens[0, idx_mapping]
        anchor_tokens = torch.where(
            sampled_counts > 0,
            sampled_anchors,
            prefill_anchors,
        ).to(torch.int64)
        # Queue both small D2H copies and wait once.  The same stream owns the
        # preceding token-table update, so this synchronization also makes the
        # UVA-backed table safe for prefix matching below.
        _detail_prep_t0 = time.perf_counter()
        torch.cuda.current_stream(self.device).synchronize()
        self._detail_prep_drain = time.perf_counter() - _detail_prep_t0
        self._apply_async_error()  # kimi-k3-draft-async-ingest
        rejected_staging = self._rejected_staging[:num_reqs]
        anchor_staging = self._anchor_staging[:num_reqs]
        sampled_staging = self._sampled_staging[:num_reqs]  # kimi-k3-draft-async-ingest
        rejected_staging.copy_(num_rejected[:num_reqs], non_blocking=True)
        anchor_staging.copy_(anchor_tokens, non_blocking=True)
        sampled_staging.copy_(sampled_counts, non_blocking=True)
        _detail_copy_t0 = time.perf_counter()
        torch.cuda.current_stream(self.device).synchronize()
        self._detail_copy_sync = time.perf_counter() - _detail_copy_t0
        rejected_counts = rejected_staging.tolist()
        anchor_tokens_cpu = anchor_staging.tolist()
        sampled_counts_cpu = sampled_staging.tolist()  # kimi-k3-draft-async-ingest
        gather_indices, valid_counts = _build_valid_context_plan(
            input_batch, rejected_counts
        )
        metadata_ready = time.perf_counter()

        active_indices: list[int] = []
        requests: list[dict[str, Any]] = []
        request_context_starts: list[int | None] = []
        selected_gather_indices: list[int] = []
        gather_offset = 0
        for request_idx, request_id in enumerate(input_batch.req_ids):
            valid_count = valid_counts[request_idx]
            request_gather = gather_indices[gather_offset : gather_offset + valid_count]
            gather_offset += valid_count
            if request_id in self._disabled_requests or valid_count <= 0:
                continue
            first_position = int(input_batch.num_computed_tokens_np[request_idx])
            is_continuing = (
                request_id in previous_active_requests
                and request_id in self._known_requests
            )
            reset = False
            context_start: int | None = None
            if not is_continuing:
                if first_position == 0:
                    if request_id in self._known_requests:
                        self._free_remote_requests({request_id})
                    self._ensure_remote_capacity(current_request_ids)
                    reset = True
                    context_start = 0
                else:
                    token_prefix = self._token_prefix(
                        input_batch,
                        request_idx,
                        first_position,
                    )
                    source_request_id: str | None = None
                    if token_prefix is not None:
                        retained = self._retained_prefixes.get(request_id)
                        if (
                            request_id in self._known_requests
                            and retained is not None
                            and self._can_restore_prefix(retained, first_position)
                            and torch.equal(
                                retained.token_ids[:first_position], token_prefix
                            )
                        ):
                            source_request_id = request_id
                        else:
                            source_request_id = self._find_reconnect_source(
                                token_prefix,
                                first_position,
                                current_request_ids,
                            )
                    if source_request_id is None or token_prefix is None:
                        if request_id in self._known_requests:
                            self._free_remote_requests({request_id})
                        self._ensure_remote_capacity(current_request_ids)
                        reset = True
                        context_start = first_position
                        logger.warning(
                            "Remote K3 %s cold-bootstrapping cache-restored "
                            "request %s at position %d from %d fresh context "
                            "rows; target verification preserves correctness.",
                            self.method,
                            request_id,
                            first_position,
                            valid_count,
                        )
                    elif not self._reconnect_request(
                        source_request_id,
                        request_id,
                        first_position,
                        token_prefix,
                    ):
                        self._disabled_requests.add(request_id)
                        continue
            requests.append(
                {
                    "request_id": request_id,
                    "reset": reset,
                    "reset_position": first_position if reset else 0,
                    "context_count": valid_count,
                    "anchor_token_id": int(anchor_tokens_cpu[request_idx]),
                }
            )
            self._known_requests.add(request_id)
            active_indices.append(request_idx)
            request_context_starts.append(context_start)
            selected_gather_indices.extend(request_gather)

        self._active_requests = self._known_requests & current_request_ids
        if not active_indices:
            return
        requests_ready = time.perf_counter()

        indices_gpu = torch.tensor(
            selected_gather_indices, dtype=torch.int64, device=self.device
        )
        positions = input_batch.positions.index_select(0, indices_gpu)
        context = torch.cat(
            [hidden.index_select(0, indices_gpu) for hidden in aux_hidden_states],
            dim=-1,
        )
        num_context_rows = int(context.shape[0])
        if num_context_rows > self.max_num_tokens:
            raise ValueError(
                f"Remote DSpark context has {num_context_rows} rows, max is "
                f"{self.max_num_tokens}"
            )
        if context.shape[1] != self.raw_context_width:
            raise ValueError(
                f"Remote DSpark context width is {context.shape[1]}, expected "
                f"{self.raw_context_width}"
            )
        context_ready = time.perf_counter()
        # A step in which no active request sampled a token is a pure mid-prefill
        # step. Its proposal is never consumed, so only the context ingest matters
        # and it can complete off the critical path.
        deferred = self._async_depth > 0 and not _K3_CAPTURE_DIR and all(
            sampled_counts_cpu[request_idx] == 0 for request_idx in active_indices
        )
        if deferred:
            slot = self._async_slot
            self._async_slot = (slot + 1) % self._async_depth
            self._async_slot_free[slot].wait()
            self._async_slot_free[slot].clear()
            positions_full, context_full = self._async_ring[slot]
            positions_staging = positions_full[:num_context_rows]
            context_staging = context_full[:num_context_rows]
        else:
            slot = -1
            positions_staging = self._positions_staging[:num_context_rows]
            context_staging = self._context_staging[:num_context_rows]
        positions_staging.copy_(positions, non_blocking=True)
        context_staging.copy_(context, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        context_copied = time.perf_counter()
        anchor_positions = _anchor_positions_from_context(
            [int(request["context_count"]) for request in requests],
            positions_staging,
        )
        for request, anchor_position in zip(requests, anchor_positions):
            request["anchor_position"] = anchor_position
        if deferred:
            # The ring slot stays reserved until the ingest worker receives a reply.
            positions_frame = memoryview(positions_staging.numpy())
            context_frame = memoryview(context_staging.view(torch.uint16).numpy())
        else:
            # The synchronous RPC retains both staging buffers until its reply.
            positions_frame = memoryview(positions_staging.numpy())
            # NumPy has inconsistent bfloat16 support; preserve its exact bits as u16.
            context_frame = memoryview(context_staging.view(torch.uint16).numpy())
        # kimi-k3-dflash-feature-capture: persist the exact served feature
        # stream for offline draft training. Inert without the env var.
        if _K3_CAPTURE_DIR:
            _k3_capture_records(
                requests,
                positions_frame,
                context_frame,
                int(context_staging.shape[1]),
                anchor_positions,
            )
        serialized = time.perf_counter()
        header = {
            "protocol": PROTOCOL_VERSION,
            "op": "PROPOSE",
            "projected": False,
            "num_speculative_tokens": num_speculative_tokens,
            "return_logits": self._probabilistic,
            "logits_topk": self._logits_topk,
            "requests": requests,
        }
        frames = [json.dumps(header).encode(), positions_frame, context_frame]
        if deferred:  # kimi-k3-draft-async-ingest
            self._async_queue.put(
                (
                    slot,
                    frames,
                    [str(request["request_id"]) for request in requests],
                    time.perf_counter(),
                )
            )
            self._async_stats["deferred"] += 1
            response = {}
            rpc_done = output_copied = time.perf_counter()
        elif defer_resolve:
            self._start_reply_thread(
                frames,
                active_indices,
                [str(request["request_id"]) for request in requests],
                num_speculative_tokens,
                input_batch,
                temperature,
                seeds,
            )
            response = {}
            rpc_done = output_copied = time.perf_counter()
        else:
            response = self._rpc(frames)
            rpc_done = time.perf_counter()
            self._copy_tokens_from_response(
                response,
                active_indices,
                num_speculative_tokens,
            )
            if self._probabilistic:
                self._copy_logits_from_response(
                    response,
                    active_indices,
                    num_speculative_tokens,
                )
            output_copied = time.perf_counter()
        timing_ms = {
            "entry_drain": getattr(self, "_detail_entry_drain", 0.0) * 1000,
            "prep_drain": getattr(self, "_detail_prep_drain", 0.0) * 1000,
            "copy_sync": getattr(self, "_detail_copy_sync", 0.0) * 1000,
            "metadata_d2h": (metadata_ready - started) * 1000,
            "request_plan": (requests_ready - metadata_ready) * 1000,
            "context_gather": (context_ready - requests_ready) * 1000,
            "context_d2h": (context_copied - context_ready) * 1000,
            "serialize": (serialized - context_copied) * 1000,
            "rpc_roundtrip": (rpc_done - serialized) * 1000,
            "tokens_h2d": (output_copied - rpc_done) * 1000,
            "client_total": (output_copied - started) * 1000,
            "deferred": 1.0 if deferred else 0.0,  # kimi-k3-draft-async-ingest
        }
        server_timing = response.get("timing_ms")
        if isinstance(server_timing, dict):
            for key, value in server_timing.items():
                if isinstance(value, int | float):
                    timing_ms[f"server_{key}"] = float(value)
        self._record_timing(timing_ms)
        for request_idx, request, anchor_position, context_start in zip(
            active_indices,
            requests,
            anchor_positions,
            request_context_starts,
        ):
            self._remember_prefix(
                input_batch,
                request_idx,
                str(request["request_id"]),
                anchor_position,
                context_start,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_speculative_tokens: int | None = None,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            num_tokens_across_dp,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        active_k = (
            int(num_speculative_tokens)
            if num_speculative_tokens is not None
            else self.num_speculative_steps
        )
        # A scheduler can intentionally disable speculation for one step (for
        # example, a request with max_tokens=1). ModelRunner treats an empty
        # second dimension as a normal non-speculative step.
        if active_k == 0:
            return self.draft_tokens[: input_batch.num_reqs, :0].contiguous()
        if not 1 <= active_k <= self.num_speculative_steps:
            raise ValueError(
                f"Remote DSpark depth must be in [1, "
                f"{self.num_speculative_steps}], got {active_k}"
            )
        if self._pending is not None:
            # The runner did not consume the previous proposal (a step without
            # a forward). Consume it here so its reply thread is released and
            # the buffers are free for this proposal.
            self._resolve_pending_locked()
        output = self.draft_tokens[: input_batch.num_reqs, :active_k]
        output.fill_(-1)
        if self._probabilistic:
            assert self.draft_logits is not None
            assert self._remote_logits is not None
            assert self._remote_sample_positions is not None
            self.draft_logits.zero_()
            if self._logits_topk == 0:
                self._remote_logits.zero_()
            self._remote_sample_positions.fill_(-1)
        defer_resolve = (
            self._deferred_resolve
            and self.deferred_resolve_allowed
            and not (dummy_run or is_profile)
        )
        if self._tp_rank == 0 and not (dummy_run or is_profile):
            try:
                self._rank0_propose(
                    input_batch,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    active_k,
                    defer_resolve=defer_resolve,
                    temperature=temperature,
                    seeds=seeds,
                )
            except Exception:
                output.fill_(-1)
                # The verifier cannot know whether a timed-out request mutated
                # remote KV. Fail closed for those requests until they leave
                # the active batch; FREE remains safe even if the server never
                # created the state.
                self._disabled_requests.update(input_batch.req_ids)
                logger.exception(
                    "Remote K3 DSpark proposal failed; drafting is disabled for "
                    "this step"
                )
        if defer_resolve:
            if self._pending is None:
                # No reply thread was started (rank > 0, a pure ingest step,
                # or a failed plan): the broadcast and sampling still run at
                # resolve time so every rank executes the same collectives.
                self._pending = _PendingProposal(
                    epoch=self._pending_epoch,
                    num_reqs=input_batch.num_reqs,
                    num_speculative_tokens=active_k,
                    active_indices=[],
                    request_ids=[],
                    idx_mapping=input_batch.idx_mapping[: input_batch.num_reqs],
                    temperature=temperature,
                    seeds=seeds,
                )
            return output
        if self._probabilistic and not (dummy_run or is_profile):
            assert self._remote_logits is not None
            assert self._remote_sample_positions is not None
            if self._logits_topk > 0:
                assert self._remote_topk_values is not None
                assert self._remote_topk_indices is not None
                self._tp_group.broadcast(self._remote_topk_values, src=0)
                self._tp_group.broadcast(self._remote_topk_indices, src=0)
                self._materialize_topk_logits(active_k)
            else:
                self._tp_group.broadcast(self._remote_logits, src=0)
            self._tp_group.broadcast(self._remote_sample_positions, src=0)
            self._sample_remote_probabilistic(
                input_batch,
                temperature,
                seeds,
                active_k,
            )
        # Slicing the active depth from the max-width persistent buffer leaves
        # a larger row stride whenever adaptive K is below the configured
        # maximum. NCCL broadcast requires a contiguous tensor. Materialize
        # only the tiny [batch, K] result after rank 0 has populated it.
        output = _contiguous_draft_output(
            self.draft_tokens,
            input_batch.num_reqs,
            active_k,
        )
        use_cpu_broadcast = (
            self._cpu_token_broadcast
            and not dummy_run
            and not is_profile
            and not torch.cuda.is_current_stream_capturing()
        )
        if use_cpu_broadcast:
            broadcast_started = time.perf_counter()
            output_cpu = self._cpu_draft_tokens[active_k][: input_batch.num_reqs]
            if self._tp_rank == 0:
                output_cpu.copy_(output, non_blocking=False)
            torch.distributed.broadcast(
                output_cpu,
                src=self._tp_group.ranks[0],
                group=self._tp_group.cpu_group,
            )
            output.copy_(output_cpu, non_blocking=True)
            if self._tp_rank == 0 and self._timing_log_interval > 0:
                self._cpu_broadcast_count += 1
                self._cpu_broadcast_total_ms += (
                    time.perf_counter() - broadcast_started
                ) * 1000
                if self._cpu_broadcast_count >= self._timing_log_interval:
                    logger.info(
                        "Remote K3 CPU token broadcast over %d proposals (ms): %.3f",
                        self._cpu_broadcast_count,
                        self._cpu_broadcast_total_ms / self._cpu_broadcast_count,
                    )
                    self._cpu_broadcast_count = 0
                    self._cpu_broadcast_total_ms = 0.0
        else:
            self._tp_group.broadcast(output, src=0)
        return output
