# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared EXL3 prefill-reconstruct scratch arena.

CPU-only; no CUDA, b12x, or exllamav3_ext required. The arena replaces one
persistent fp16 buffer per (device, K, N-chunk) with one byte arena per
device sized to the largest live geometry. Safety rests on two invariants
these tests pin down:

  * the scratch is written and fully consumed inside one eager
    ``_reconstruct_hgemm_into`` call on one stream, so at most one geometry's
    scratch is live at a time and every view can start at arena offset 0;
  * a CUDA-graph capture bakes the arena address into replayed kernels, so a
    capture-recorded arena is frozen and never reallocated; geometries that
    no longer fit fall back to dedicated persistent buffers (the pre-arena
    behaviour).
"""

import torch

import vllm.model_executor.layers.quantization.exl3 as exl3_module

# Small synthetic geometries (K, N); all multiples of 16 as the trellis
# packing requires. The wide one exercises the chunk loop once
# _EXL3_RECONSTRUCT_SLICE_N is narrowed in the harness below.
GEOMETRIES = [
    (256, 512),
    (512, 256),
    (256, 1024),
    (1024, 256),
    (256, 2048),
]
NARROW_SLICE_N = 512


class _FakeExt:
    """Tags every scratch write and verifies the tag at consumption."""

    def __init__(self):
        self._tag = 0.0
        self._live = None
        self.consumed = 0
        self.views = []

    def had_r_128(self, x, out, su, sv, scale):
        if x.data_ptr() != out.data_ptr():
            out.copy_(x)

    def _write(self, view):
        self._tag += 1.0
        assert view.dtype == torch.float16
        assert view.storage_offset() == 0, "arena views must start at offset 0"
        row_bytes = ((view.shape[0] - 1) * view.stride(0) + view.shape[1]) * 2
        assert row_bytes <= view.untyped_storage().nbytes(), "view out of bounds"
        self.views.append(view)
        view.fill_(self._tag)
        self._live = (view, self._tag)

    def reconstruct(self, view, trellis, bits, mcg, mul1):
        self._write(view)

    def reconstruct_slice(self, view, trellis, bits, mcg, mul1, start):
        self._write(view)

    def hgemm(self, x_had, view, out):
        live, tag = self._live
        assert bool((live == tag).all()), (
            "scratch clobbered between reconstruct write and hgemm read"
        )
        self.consumed += 1
        out.fill_(0.0)


class _Harness:
    def __init__(self, capturing=False):
        self.ext = _FakeExt()
        self._capturing = capturing

    def set_capturing(self, value):
        self._capturing = value

    def __enter__(self):
        self._saved_capturing = torch.cuda.is_current_stream_capturing
        torch.cuda.is_current_stream_capturing = lambda: self._capturing
        self._saved_current_device = torch.cuda.current_device
        torch.cuda.current_device = lambda: 0
        self._saved_load_ext = exl3_module._load_exl3_ext
        exl3_module._load_exl3_ext = lambda: self.ext
        self._saved_slice_n = exl3_module._EXL3_RECONSTRUCT_SLICE_N
        exl3_module._EXL3_RECONSTRUCT_SLICE_N = NARROW_SLICE_N
        exl3_module._EXL3_RECONSTRUCT_ARENA.clear()
        exl3_module._EXL3_RECONSTRUCT_ARENA_FROZEN.clear()
        exl3_module._EXL3_RECONSTRUCT_SCRATCH.clear()
        return self

    def __exit__(self, *exc):
        torch.cuda.is_current_stream_capturing = self._saved_capturing
        torch.cuda.current_device = self._saved_current_device
        exl3_module._load_exl3_ext = self._saved_load_ext
        exl3_module._EXL3_RECONSTRUCT_SLICE_N = self._saved_slice_n
        exl3_module._EXL3_RECONSTRUCT_ARENA.clear()
        exl3_module._EXL3_RECONSTRUCT_ARENA_FROZEN.clear()
        exl3_module._EXL3_RECONSTRUCT_SCRATCH.clear()


def _run(k, n, rows=8):
    trellis = torch.zeros((k // 16, n // 16, 80), dtype=torch.int16)
    x = torch.zeros((rows, k), dtype=torch.float16)
    out = torch.empty((rows, n), dtype=torch.float16)
    suh = torch.zeros((k,), dtype=torch.float16)
    svh = torch.zeros((n,), dtype=torch.float16)
    exl3_module._reconstruct_hgemm_into(out, x, trellis, suh, svh, False, False)


def _expected_arena_bytes():
    return max(2 * k * min(n, NARROW_SLICE_N) for k, n in GEOMETRIES)


def test_arena_is_shared_and_sized_to_largest_geometry():
    with _Harness() as h:
        for k, n in GEOMETRIES:
            _run(k, n)
        assert list(exl3_module._EXL3_RECONSTRUCT_ARENA) == [0]
        arena = exl3_module._EXL3_RECONSTRUCT_ARENA[0]
        assert arena.numel() == _expected_arena_bytes()
        assert not exl3_module._EXL3_RECONSTRUCT_SCRATCH
        assert h.ext.consumed == sum(
            -(-n // NARROW_SLICE_N) for _, n in GEOMETRIES
        )


def test_arena_view_matches_torch_empty_layout():
    with _Harness():
        for k, n in GEOMETRIES:
            chunk = min(n, NARROW_SLICE_N)
            view = exl3_module._reconstruct_scratch(torch.device("cpu"), k, chunk)
            ref = torch.empty((k, chunk), dtype=torch.float16)
            assert view.shape == ref.shape
            assert view.stride() == ref.stride()
            assert view.dtype == ref.dtype
            assert view.is_contiguous()


def test_scratch_survives_until_consumption_across_interleaving():
    with _Harness() as h:
        # Repeated interleaved layers, including the chunked wide geometry:
        # any aliasing between a write and its consumption fails in hgemm.
        for _ in range(3):
            for k, n in GEOMETRIES:
                _run(k, n)
        assert h.ext.consumed == 3 * sum(
            -(-n // NARROW_SLICE_N) for _, n in GEOMETRIES
        )


def test_arena_never_grows_during_capture_and_freezes_afterwards():
    with _Harness() as h:
        _run(256, 512)  # eager growth to 256 KiB
        arena = exl3_module._EXL3_RECONSTRUCT_ARENA[0]
        h.set_capturing(True)
        # Same-size geometry under capture is served from the arena and
        # freezes it.
        view = exl3_module._reconstruct_scratch(torch.device("cpu"), 256, 512)
        assert view.data_ptr() == arena.data_ptr()
        assert 0 in exl3_module._EXL3_RECONSTRUCT_ARENA_FROZEN
        # A bigger geometry under capture gets a dedicated buffer; the arena
        # object is not replaced.
        big = exl3_module._reconstruct_scratch(torch.device("cpu"), 1024, 256)
        assert exl3_module._EXL3_RECONSTRUCT_ARENA[0] is arena
        assert (0, 1024, 256) in exl3_module._EXL3_RECONSTRUCT_SCRATCH
        # The dedicated buffer address is stable across calls (a replayed
        # graph's baked pointer stays valid).
        again = exl3_module._reconstruct_scratch(torch.device("cpu"), 1024, 256)
        assert big.data_ptr() == again.data_ptr()
        # After capture, the frozen arena still never reallocates.
        h.set_capturing(False)
        exl3_module._reconstruct_scratch(torch.device("cpu"), 256, 2048)
        assert exl3_module._EXL3_RECONSTRUCT_ARENA[0] is arena
        assert (0, 256, 2048) in exl3_module._EXL3_RECONSTRUCT_SCRATCH


def test_unfrozen_arena_grows_eagerly_and_reuses_bytes():
    with _Harness():
        small = exl3_module._reconstruct_scratch(torch.device("cpu"), 256, 512)
        assert exl3_module._EXL3_RECONSTRUCT_ARENA[0].numel() == 2 * 256 * 512
        big = exl3_module._reconstruct_scratch(torch.device("cpu"), 1024, 512)
        assert exl3_module._EXL3_RECONSTRUCT_ARENA[0].numel() == 2 * 1024 * 512
        # A smaller geometry after growth reuses the same backing at offset 0.
        small2 = exl3_module._reconstruct_scratch(torch.device("cpu"), 256, 512)
        assert small2.data_ptr() == exl3_module._EXL3_RECONSTRUCT_ARENA[0].data_ptr()
        assert small2.data_ptr() == big.data_ptr()
        assert not exl3_module._EXL3_RECONSTRUCT_SCRATCH
        del small
