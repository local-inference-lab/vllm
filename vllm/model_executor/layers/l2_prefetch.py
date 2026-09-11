# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Overlap GLM decode collectives and attention with weight loads into L2.

Small tensor-parallel decode steps leave high-bandwidth memory underused while
collectives wait on PCIe synchronization and while the attention kernels run.
This module uses those intervals to stream BF16 projection weights into L2 on
a side stream. The consumer does not wait for the prefetch: weights that have
arrived are served from L2 and remaining weights are fetched from memory
normally.

Two windows per decoder layer (``register_attention_layers`` attaches both
target lists; ``VLLM_L2_PREFETCH_WINDOWS`` selects at run time which of them
launch, so the traced graph is the same for every setting):

- ``moe``: during the previous layer's MoE or dense-MLP output all-reduce
  (fused into this layer's input RMSNorm), this layer's fused query/key/value
  input projection, query output projection and, where the sparse indexer
  runs, its query projection. Gated by ``VLLM_L2_PREFETCH_MAX_TOKENS`` and
  loaded by ``VLLM_L2_PREFETCH_CTAS`` programs (64). With the weight-first
  projection and the early all-reduce trigger, the projections that follow
  the all-reduce start sooner than they did without them, and the loads
  that are still running then overlap the sparse indexer and attention
  kernels. Measured at four padded tokens on the eight-rank GLM-5.3 launch,
  64 programs against 32 leave the indexer kernel at its unprefetched
  duration (6.7 us against 11.5 us), keep the sparse attention kernel
  within 0.7 us of it, and shorten the decode step by 0.16 ms; eight and
  sixteen padded tokens and 30k-token contexts are faster or unchanged.
- ``attn``: at the start of the attention block, ordered on the normalised
  block input, this layer's attention output projection (``o_proj``), which
  is consumed right after the attention kernels. Gated by
  ``VLLM_L2_PREFETCH_ATTN_MAX_TOKENS`` (the loads beside the attention kernels
  cost more than they save at larger row counts) and loaded by
  ``VLLM_L2_PREFETCH_ATTN_CTAS`` programs (32; 64 programs load faster but
  slow the small projection kernels they run beside, which costs more than
  the attention kernel gains). The router gate and shared-expert
  weights that follow ``o_proj`` are not in this list: the weight-first
  projection stages them from memory while the attention-output all-reduce
  waits.

The side stream is used only in full CUDA graphs whose padded token count is
within the window's gate. Every forward joins the side stream once before
returning, which closes the graph forks. Eager and piecewise-graph forwards
only warm the Triton kernel; creating an event for every eager layer would
retain driver allocations while the host runs ahead of the GPU.

``register_attention_layers`` must run during model construction. It attaches
fixed weight lists to each decoder layer and allocates a persistent sink
before graph tracing. Live request quantities never enter a compile or cache
key.
"""

from __future__ import annotations

import os

import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Each program of the prefetch kernel stores one word into the per-device
# sink, which holds this many words.
_SINK_WORDS = 1 << 16


def _program_count(name: str, default: str) -> int:
    """Program count of a prefetch window from the environment.

    Args:
        name: Environment variable holding the count.
        default: Value used when the variable is unset.

    Returns:
        The count, within ``1.._SINK_WORDS`` (the sink holds one word per
        program).

    Raises:
        ValueError: The value is not an integer in that range.
    """
    value = int(os.environ.get(name, default))
    if not 1 <= value <= _SINK_WORDS:
        raise ValueError(f"{name} must be in 1..{_SINK_WORDS}, got {value}")
    return value


ENABLED = os.environ.get("VLLM_L2_PREFETCH", "1") == "1"
MAX_TOKENS = int(os.environ.get("VLLM_L2_PREFETCH_MAX_TOKENS", "64"))
NUM_PROGRAMS = _program_count("VLLM_L2_PREFETCH_CTAS", "64")
WINDOWS = frozenset(
    window.strip()
    for window in os.environ.get("VLLM_L2_PREFETCH_WINDOWS", "moe,attn").split(",")
    if window.strip()
)
ATTN_MAX_TOKENS = int(os.environ.get("VLLM_L2_PREFETCH_ATTN_MAX_TOKENS", "8"))
ATTN_NUM_PROGRAMS = _program_count("VLLM_L2_PREFETCH_ATTN_CTAS", "32")
MOE_WINDOW = "moe"
ATTN_WINDOW = "attn"

_BLOCK = 2048
_UNROLL = 8
_streams: dict[int, torch.cuda.Stream] = {}
_sinks: dict[int, torch.Tensor] = {}
_pending_devices: set[int] = set()
_warmed_devices: set[int] = set()


@triton.jit
def _l2_prefetch_kernel(
    ptr,
    n,
    out_ptr,
    BLOCK: tl.constexpr,
    NPROG: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for start in range(pid * BLOCK * UNROLL, n, NPROG * BLOCK * UNROLL):
        for offset in tl.static_range(UNROLL):
            indices = start + offset * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(ptr + indices, mask=indices < n, other=0)
    tl.store(out_ptr + pid, tl.sum(acc))


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("L2 weight prefetch requires a CUDA device")
    return (
        device.index
        if device.index is not None
        else torch.accelerator.current_device_index()
    )


def _stream(device: torch.device) -> torch.cuda.Stream:
    device_index = _device_index(device)
    stream = _streams.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _streams[device_index] = stream
    return stream


def _sink(device: torch.device) -> torch.Tensor:
    device_index = _device_index(device)
    value = _sinks.get(device_index)
    if value is None:
        value = torch.zeros(_SINK_WORDS, dtype=torch.int64, device=device)
        _sinks[device_index] = value
    return value


def _launch(
    weight: torch.Tensor, sink: torch.Tensor, num_programs: int = NUM_PROGRAMS
) -> None:
    packed_weight = weight.view(-1).view(torch.int64)
    _l2_prefetch_kernel[(num_programs,)](
        packed_weight,
        packed_weight.numel(),
        sink,
        BLOCK=_BLOCK,
        NPROG=num_programs,
        UNROLL=_UNROLL,
        num_warps=16,
    )


def _window_max_tokens(window: str) -> int:
    return ATTN_MAX_TOKENS if window == ATTN_WINDOW else MAX_TOKENS


def _window_num_programs(window: str) -> int:
    return ATTN_NUM_PROGRAMS if window == ATTN_WINDOW else NUM_PROGRAMS


def _active(num_tokens: int, window: str = MOE_WINDOW) -> bool:
    if (
        not ENABLED
        or window not in WINDOWS
        or num_tokens > _window_max_tokens(window)
        or not is_forward_context_available()
    ):
        return False
    mode = get_forward_context().cudagraph_runtime_mode
    if mode == CUDAGraphMode.PIECEWISE:
        return False
    return mode == CUDAGraphMode.FULL or torch.cuda.is_current_stream_capturing()


def _warm_up(weights: list[torch.Tensor], sink: torch.Tensor) -> None:
    """Compile and load every program-count specialization outside any
    capture (a first launch inside a capture fails).

    Args:
        weights: Prefetch targets; the first one is read by the warm-up
            launches.
        sink: The device's sink tensor, which also selects the device.
    """
    device_index = _device_index(sink.device)
    if device_index in _warmed_devices or torch.cuda.is_current_stream_capturing():
        return
    for num_programs in sorted({NUM_PROGRAMS, ATTN_NUM_PROGRAMS}):
        _launch(weights[0], sink, num_programs)
    _warmed_devices.add(device_index)


def _prefetch_impl(
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    sink: torch.Tensor,
    window: str,
) -> None:
    if not _active(hidden_states.shape[0], window):
        _warm_up(weights, sink)
        return

    device_index = _device_index(hidden_states.device)
    main_stream = torch.cuda.current_stream(hidden_states.device)
    prefetch_stream = _stream(hidden_states.device)
    prefetch_stream.wait_stream(main_stream)
    num_programs = _window_num_programs(window)
    with torch.cuda.stream(prefetch_stream):
        for weight in weights:
            _launch(weight, sink, num_programs)
    _pending_devices.add(device_index)
    logger.info_once(
        "L2 weight prefetch window %s is active in full CUDA graphs "
        "through %d padded tokens (%d programs).",
        window,
        _window_max_tokens(window),
        num_programs,
    )


def _prefetch_fake(
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor],
    sink: torch.Tensor,
    window: str,
) -> None:
    return None


def _join_impl(sink: torch.Tensor) -> None:
    device_index = _device_index(sink.device)
    if device_index not in _pending_devices:
        return
    torch.cuda.current_stream(sink.device).wait_stream(_stream(sink.device))
    _pending_devices.remove(device_index)


def _join_fake(sink: torch.Tensor) -> None:
    return None


direct_register_custom_op(
    op_name="l2_weight_prefetch",
    op_func=_prefetch_impl,
    mutates_args=["sink"],
    fake_impl=_prefetch_fake,
)
direct_register_custom_op(
    op_name="l2_weight_prefetch_join",
    op_func=_join_impl,
    mutates_args=["sink"],
    fake_impl=_join_fake,
)


def issue(
    hidden_states: torch.Tensor,
    weights: list[torch.Tensor] | None,
    window: str = MOE_WINDOW,
) -> None:
    """Issue a prefetch of ``window`` ordered after ``hidden_states`` when
    targets exist.

    Args:
        hidden_states: The activation whose row count gates the window and
            whose stream orders the prefetch.
        weights: The window's prefetch targets, or ``None`` when the layer
            has none.
        window: ``MOE_WINDOW`` or ``ATTN_WINDOW``.
    """
    if weights and hidden_states.device.type == "cuda":
        torch.ops.vllm.l2_weight_prefetch(
            hidden_states,
            weights,
            _sink(hidden_states.device),
            window,
        )


def join(hidden_states: torch.Tensor) -> None:
    """Join an existing device prefetch stream before the model returns.

    Args:
        hidden_states: The model output; its device selects the stream.
    """
    if hidden_states.device.type != "cuda":
        return
    sink = _sinks.get(_device_index(hidden_states.device))
    if sink is not None:
        torch.ops.vllm.l2_weight_prefetch_join(sink)


def _bf16_weight(module: object, name: str) -> torch.Tensor | None:
    submodule = getattr(module, name, None)
    weight = getattr(submodule, "weight", None)
    if (
        isinstance(weight, torch.Tensor)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    ):
        return weight
    return None


def register_attention_layers(layers: torch.nn.ModuleList, start: int, end: int) -> int:
    """Attach fixed attention projection weights to decoder layers.

    ``moe`` window (``layer._l2_prefetch_weights``): each local layer after
    the first receives its own fused query/key/value input projection, query
    output projection, and sparse indexer query projection when that layer
    computes an index selection. The preceding layer leaves its MLP output
    unreduced, so these weights can load alongside the entry fused all-reduce
    and RMSNorm.

    ``attn`` window (``layer._l2_prefetch_attn_weights``): every local layer
    receives its own attention output projection, loaded at the start of the
    attention block so it is in L2 when the attention kernels finish.

    Both attributes are set on every layer (``None`` where there is nothing
    to load) whatever ``VLLM_L2_PREFETCH_WINDOWS`` selects.

    Args:
        layers: The model's decoder layers.
        start: Index of the first local layer.
        end: One past the index of the last local layer.

    Returns:
        The number of decoder transitions with a non-empty ``moe`` target
        list.
    """
    target_count = 0
    attn_count = 0
    for layer_index in range(start, end):
        layer = layers[layer_index]
        attention = getattr(layer, "self_attn", None)
        layer._l2_prefetch_weights = None
        layer._l2_prefetch_attn_weights = None
        if attention is None:
            continue

        o_proj = _bf16_weight(attention, "o_proj")
        if o_proj is not None:
            layer._l2_prefetch_attn_weights = [o_proj]
            attn_count += 1
            if o_proj.device.type == "cuda":
                _sink(o_proj.device)
        if layer_index == start:
            continue

        weights = [
            _bf16_weight(attention, "fused_qkv_a_proj"),
            _bf16_weight(attention, "q_b_proj"),
        ]
        indexer = getattr(attention, "indexer", None)
        if indexer is not None and not getattr(attention, "skip_topk", True):
            weights.append(_bf16_weight(indexer, "wq_b"))
        targets = [weight for weight in weights if weight is not None]
        if not targets:
            continue

        layer._l2_prefetch_weights = targets
        target_count += 1
        if targets[0].device.type == "cuda":
            _sink(targets[0].device)

    logger.info_once(
        "Registered %d decoder transitions for the L2 attention-weight prefetch "
        "and %d attention blocks for the output-projection prefetch.",
        target_count,
        attn_count,
    )
    return target_count
