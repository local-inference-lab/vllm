# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L2 weight prefetch for GLM-5.3 decode on SM120.

At decode batch sizes (M <= 256) the dense projections are pure device-memory (GDDR7) streams
(in_proj_qkvgfab is 50 MB per GPU: 28.8 us at M=8) while the all-reduces, mHC,
routing chain and small kernels leave device memory idle for roughly half of each layer.
This module issues ``cp.async.bulk.prefetch.L2`` with an ``evict_last`` cache
policy for the *upcoming* dense weights on a side stream inside those idle
windows, sized so the fills finish before the routed-expert stream starts.
cuBLAS then reads the weights from L2 (128 MB on RTX PRO 6000 Blackwell):

  in_proj 29.9 -> 17.1 us, o_proj 11.6 -> 5.5, MLA q_b 11.7 -> 8.3 (C1 trace)
  C1: 85.1 -> 91.1 verifier steps/s (+7%), Sieve coding peak 433 -> 462 tok/s

Numerics are untouched (cache hints only).  The side stream is forked per
window and rejoined once at the end of the model forward, which is valid in
FULL cudagraph captures and eager runs; inside a *breakable* (PIECEWISE)
capture the wrapper ends the capture segment at eager ops, so prefetch is
skipped there.

Windows per decoder layer (C1, M=8):
  A  inside attention after the first projection: this layer's o_proj
     (optionally + a head slice of the next layer's first projection)
  B  before the attention-output all-reduce: router weight + next layer's
     first projection
  C  before the MoE all-reduce: the remainder

Environment:
  VLLM_GLM53_L2_PREFETCH=0            disable (default: on for SM120)
  VLLM_GLM53_L2_PREFETCH_MAX_TOKENS   only prefetch for batches up to this (256)
  VLLM_GLM53_L2_PREFETCH_BUDGET_{A,B,C,A_MLA}_MB   per-window fill budgets
  VLLM_GLM53_L2_PREFETCH_A_NEXT_MB    next-layer head bytes carried in window A
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MIN_BYTES = 64 * 1024
_CHUNK_BYTES = 4096
_GRID = 16
_BLOCK = 128
_MAX_TOKENS = int(os.getenv("VLLM_GLM53_L2_PREFETCH_MAX_TOKENS", "256"))


def _mb(name: str, default: str) -> int:
    return int(float(os.getenv(name, default)) * 1e6)


BUDGET_A = _mb("VLLM_GLM53_L2_PREFETCH_BUDGET_A_MB", "20")
# Windows B/C fire *before* the all-reduces (hooks in RowParallelLinear and the
# MoE runner), which adds the reduction time to each idle window.
BUDGET_B = _mb("VLLM_GLM53_L2_PREFETCH_BUDGET_B_MB", "50")
BUDGET_C = _mb("VLLM_GLM53_L2_PREFETCH_BUDGET_C_MB", "15")
BUDGET_A_MLA = _mb("VLLM_GLM53_L2_PREFETCH_BUDGET_A_MLA_MB", "36")
# Measured neutral-to-negative on C1 (fills overlap the KDA core); off by default.
A_NEXT_BYTES = _mb("VLLM_GLM53_L2_PREFETCH_A_NEXT_MB", "0")

Segment = tuple[str, int, int]  # (name, ptr, bytes)


def _platform_enabled() -> bool:
    """Disable-only override: the feature never turns on without CUDA."""
    if not torch.cuda.is_available():
        return False
    raw = os.getenv("VLLM_GLM53_L2_PREFETCH")
    if raw is not None:
        return raw != "0"
    return torch.cuda.get_device_capability() == (12, 0)


ENABLED = _platform_enabled()


# ---------------------------------------------------------------------------
# CuTe DSL kernel: bulk L2 prefetch with an evict_last policy
# ---------------------------------------------------------------------------

try:  # CuTe DSL is optional; the feature silently disables without it.
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass._mlir import ir as _ir
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cutlass_dsl import dsl_user_op

    _CUTE_OK = True
except Exception:  # noqa: BLE001
    _CUTE_OK = False


if _CUTE_OK:

    @dsl_user_op
    def _createpolicy_evict_last(*, loc=None, ip=None):
        i64 = _ir.IntegerType.get_signless(64)
        res = _llvm.inline_asm(
            i64,
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
        return cutlass.Int64(res)

    @dsl_user_op
    def _bulk_prefetch_l2(addr, size, policy, *, loc=None, ip=None):
        i32 = _ir.IntegerType.get_signless(32)
        _llvm.inline_asm(
            i32,
            [
                addr.ir_value(loc=loc, ip=ip),
                size.ir_value(loc=loc, ip=ip),
                policy.ir_value(loc=loc, ip=ip),
            ],
            "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$0], $1, $2; mov.u32 $3, 0;",
            "l,r,l,=r",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )

    class L2PrefetchKernel:
        """grid x block threads walk the (ptr, bytes) segment table and issue
        one bulk L2 prefetch per 4 KB chunk (fire-and-forget)."""

        def __init__(self, chunk_bytes: int = _CHUNK_BYTES, grid: int = _GRID, block: int = _BLOCK):
            self.chunk_bytes = int(chunk_bytes)
            self.grid = int(grid)
            self.block = int(block)

        @cute.jit
        def __call__(self, gSegs: cute.Tensor, stream: CUstream) -> None:
            self.kernel(gSegs).launch(
                grid=[self.grid, 1, 1],
                block=[self.block, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, gSegs: cute.Tensor) -> None:
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            chunk: cutlass.Constexpr = self.chunk_bytes
            stride = cutlass.Int64(self.grid * self.block)
            tid = cutlass.Int64(bidx) * cutlass.Int64(self.block) + cutlass.Int64(tidx)
            nseg = cutlass.Int32(cute.size(gSegs, mode=[0]) // 2)
            policy = _createpolicy_evict_last()
            s = cutlass.Int32(0)
            while s < nseg:
                base = cutlass.Int64(gSegs[2 * s])
                nbytes = cutlass.Int64(gSegs[2 * s + 1])
                nchunks = (nbytes + cutlass.Int64(chunk - 1)) // cutlass.Int64(chunk)
                c = tid
                while c < nchunks:
                    off = c * cutlass.Int64(chunk)
                    rem = nbytes - off
                    size = cutlass.Int32(chunk)
                    if rem < cutlass.Int64(chunk):
                        size = cutlass.Int32(rem)
                    size = (size // cutlass.Int32(16)) * cutlass.Int32(16)
                    if size > cutlass.Int32(0):
                        _bulk_prefetch_l2(base + off, size, policy)
                    c = c + stride
                s = s + cutlass.Int32(1)


_compiled = None
_compile_failed = False


def _get_launcher():
    """Compile once (shape-dynamic segment table); returns a callable
    (segs_tensor, cuda_stream_handle) -> None, or None if unavailable."""
    global _compiled, _compile_failed
    if _compiled is not None:
        return _compiled
    if _compile_failed:
        return None
    if not _CUTE_OK:
        _compile_failed = True
        logger.warning("[l2_prefetch] disabled: CuTe DSL not available")
        return None
    try:
        from quack.compile_utils import make_fake_tensor

        n = cute.sym_int(divisibility=2)
        fake = make_fake_tensor(cutlass.Int64, (n,), divisibility=2)
        stream = CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(L2PrefetchKernel(), fake, stream, options="--enable-tvm-ffi")

        def launch(segs: torch.Tensor, stream_handle: int) -> None:
            compiled(segs, CUstream(stream_handle))

        _compiled = launch
        logger.info(
            "[l2_prefetch] CuTe kernel ready (grid=%d block=%d chunk=%d; budgets A/B/C/A_mla=%.0f/%.0f/%.0f/%.0f MB)",
            _GRID, _BLOCK, _CHUNK_BYTES, BUDGET_A / 1e6, BUDGET_B / 1e6, BUDGET_C / 1e6, BUDGET_A_MLA / 1e6,
        )
        return _compiled
    except Exception as exc:  # noqa: BLE001
        _compile_failed = True
        logger.warning("[l2_prefetch] disabled: kernel compile failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------


def segments_of(module: torch.nn.Module, prefix: str = "", skip: tuple[str, ...] = ()) -> list[Segment]:
    """Large, contiguous CUDA parameters/buffers under ``module``."""
    out: list[Segment] = []
    seen: set[int] = set()

    def add(name: str, t: torch.Tensor) -> None:
        if not isinstance(t, torch.Tensor) or not t.is_cuda or not t.is_contiguous():
            return
        nbytes = t.numel() * t.element_size()
        ptr = t.data_ptr()
        if nbytes < _MIN_BYTES or ptr % 16 != 0 or ptr in seen:
            return
        if any(s in name for s in skip):
            return
        seen.add(ptr)
        out.append((name, ptr, nbytes))

    for name, p in module.named_parameters():
        add(f"{prefix}{name}", p.data)
    for m_name, m in module.named_modules():
        for attr in ("W_UK_T", "W_UV", "W_UK", "W_K", "W_V"):
            t = getattr(m, attr, None)
            if isinstance(t, torch.Tensor):
                add(f"{prefix}{m_name}.{attr}", t)
    return out


def take_budget(segments: list[Segment], budget: int) -> tuple[list[Segment], list[Segment]]:
    """Split segments into (fits in budget, remainder); the straddling segment
    is sliced so no byte is prefetched twice."""
    taken: list[Segment] = []
    rest: list[Segment] = []
    used = 0
    for name, ptr, nbytes in segments:
        if used >= budget:
            rest.append((name, ptr, nbytes))
            continue
        room = budget - used
        if nbytes <= room:
            taken.append((name, ptr, nbytes))
            used += nbytes
        else:
            head = room - (room % _CHUNK_BYTES)
            if head > 0:
                taken.append((name + "[head]", ptr, head))
                used += head
            rest.append((name + "[tail]", ptr + head, nbytes - head))
    return taken, rest


class L2PrefetchPlan:
    """Device-resident [ptr, bytes] table for one prefetch launch."""

    __slots__ = ("segs", "nseg", "total_bytes", "names")

    def __init__(self, segments: list[Segment], device: torch.device):
        segments = [s for s in segments if s[2] >= 16]
        self.nseg = len(segments)
        self.total_bytes = sum(s[2] for s in segments)
        self.names = [f"{n}:{b / 1e6:.1f}MB" for n, _, b in segments]
        flat = [v for _, ptr, nbytes in segments for v in (ptr, nbytes)]
        self.segs = torch.tensor(flat, dtype=torch.int64, device=device) if flat else None

    def describe(self) -> str:
        return f"{self.nseg} segs {self.total_bytes / 1e6:.1f} MB: " + ", ".join(self.names[:8])


def make_plan(segments: list[Segment], budget: int, device: torch.device) -> tuple[L2PrefetchPlan | None, list[Segment]]:
    taken, rest = take_budget(segments, budget)
    plan = L2PrefetchPlan(taken, device)
    return (plan if plan.nseg else None), rest


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def _prefetch_allowed() -> bool:
    """False inside a breakable (PIECEWISE) capture, where eager breaks end
    the capture segment and an un-joined side stream would be illegal."""
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        cap = BreakableCUDAGraphCapture.current()
        if cap is not None and bool(getattr(cap, "_capturing", False)):
            from vllm.config import CUDAGraphMode
            from vllm.forward_context import get_forward_context, is_forward_context_available

            if is_forward_context_available():
                return get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
            return False
    except Exception:  # noqa: BLE001
        return not torch.cuda.is_current_stream_capturing()
    return True


class L2Prefetcher:
    """One side stream per device: issue() forks it off the current stream and
    launches the prefetch; join() (model end) rejoins."""

    _instances: dict[int, "L2Prefetcher"] = {}

    def __init__(self, device: torch.device):
        self.device = device
        self.side = torch.cuda.Stream(device=device)
        self.pending = False

    @classmethod
    def get(cls, device: torch.device | None = None) -> "L2Prefetcher":
        idx = device.index if device is not None and device.index is not None else torch.cuda.current_device()
        inst = cls._instances.get(idx)
        if inst is None:
            inst = cls(torch.device("cuda", idx))
            cls._instances[idx] = inst
        return inst

    def issue(self, plan: L2PrefetchPlan | None, num_tokens: int) -> None:
        if plan is None or plan.segs is None or num_tokens > _MAX_TOKENS:
            return
        launch = _get_launcher()
        if launch is None or not _prefetch_allowed():
            return
        main = torch.cuda.current_stream(self.device)
        self.side.wait_stream(main)
        launch(plan.segs, self.side.cuda_stream)
        self.pending = True

    def join(self) -> None:
        if not self.pending:
            return
        torch.cuda.current_stream(self.device).wait_stream(self.side)
        self.pending = False


def issue(plan: L2PrefetchPlan | None, num_tokens: int) -> None:
    if ENABLED and plan is not None and plan.segs is not None:
        L2Prefetcher.get(plan.segs.device).issue(plan, num_tokens)


def join_all() -> None:
    """Rejoin every pending prefetch side stream (no-op when nothing was issued)."""
    if not ENABLED:
        return
    for inst in list(L2Prefetcher._instances.values()):
        inst.join()


__all__ = [
    "ENABLED", "BUDGET_A", "BUDGET_B", "BUDGET_C", "BUDGET_A_MLA", "A_NEXT_BYTES",
    "L2PrefetchPlan", "L2Prefetcher", "segments_of", "take_budget", "make_plan", "issue", "join_all",
]
