# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV paged decode attention for the GLM-5.3 DFlash draft on SM120.

The DFlash2 draft's five layers attend over a 2048-token sliding window with
eight query tokens per request and GQA 4 (8 heads over 2 KV heads, head size
128). FlashAttention 2 serves that as one varlen call with a single KV split:
45 to 49 us per layer at concurrency 1. ``dflash_attn.cu`` splits the window
into 128-key chunks (one CTA per chunk, KV head and request; 32 rows = 8
queries x GQA 4; ``mma.sync`` m16n8k16 for QK^T and PV; fp32 partials) and a
second kernel combines the partials: 13 us per layer.

The library is loaded from ``VLLM_GLM53_DFLASH_ATTN_LIB`` when set, otherwise
``dflash_attn.cu`` is compiled once with the CUDA toolkit's ``nvcc`` into the
vLLM cache directory. Without a toolkit ``is_available()`` is False and the
FlashAttention path is used.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

CHUNK = 128  # keys per split
ROWS = 32  # 8 query tokens x GQA 4
HEAD_DIM = 128
GQA = 4
MAX_QUERY_LEN = 8

_SOURCE = Path(__file__).with_name("dflash_attn.cu")
_lib: ctypes.CDLL | None = None
_lib_error: str | None = None


def _nvcc() -> str | None:
    for candidate in (
        os.getenv("CUDA_NVCC"),
        shutil.which("nvcc"),
        os.path.join(os.getenv("CUDA_HOME", "/usr/local/cuda"), "bin", "nvcc"),
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _arch_flag(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"


def _build(device: torch.device) -> str:
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError("nvcc not found (set CUDA_NVCC or CUDA_HOME)")
    arch = _arch_flag(device)
    digest = hashlib.sha256(_SOURCE.read_bytes() + arch.encode()).hexdigest()[:16]
    out_dir = Path(envs.VLLM_CACHE_ROOT) / "dflash_attn" / digest
    out = out_dir / "libdflash_attn.so"
    if out.is_file():
        return str(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".so", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [
        nvcc,
        "-O3",
        "-std=c++17",
        "--shared",
        "-Xcompiler",
        "-fPIC",
        "-cudart",
        "static",
        arch,
        str(_SOURCE),
        "-o",
        tmp_path,
    ]
    logger.info("[dflash_attn] building %s", out)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp_path)
        raise RuntimeError(f"nvcc failed: {result.stderr.strip()[-2000:]}")
    os.replace(tmp_path, out)
    return str(out)


def _load(device: torch.device) -> ctypes.CDLL:
    global _lib, _lib_error
    if _lib is not None:
        return _lib
    if _lib_error is not None:
        raise RuntimeError(_lib_error)
    try:
        path = os.getenv("VLLM_GLM53_DFLASH_ATTN_LIB") or _build(device)
        lib = ctypes.CDLL(path)
        argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,  # q, q_stride_t
            ctypes.c_void_p,
            ctypes.c_void_p,  # k cache, v cache
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,  # strides
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,  # block size, hkv, H
            ctypes.c_void_p,
            ctypes.c_int,  # block table, its row stride
            ctypes.c_void_p,
            ctypes.c_void_p,  # seqused_k, cu_seqlens_q
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_int,  # scale, window, causal
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,  # partials o/m/l
            ctypes.c_int,
            ctypes.c_int,  # max_splits, batch
            ctypes.c_void_p,
            ctypes.c_void_p,  # out, stream
        ]
        for name in ("dflash_attn_launch", "dflash_attn_launch_mma"):
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
        _lib = lib
        logger.info("[dflash_attn] loaded %s", path)
        return lib
    except Exception as exc:  # noqa: BLE001
        _lib_error = f"split-KV draft attention unavailable: {exc}"
        logger.warning("[dflash_attn] %s", _lib_error)
        raise RuntimeError(_lib_error) from exc


def is_available(device: torch.device) -> bool:
    if not torch.cuda.is_available() or device.type != "cuda":
        return False
    try:
        _load(device)
    except RuntimeError:
        return False
    return True


class DFlashDecodeAttention:
    """Workspace holder and launcher; one per (device, KV heads, window)."""

    def __init__(
        self,
        device: torch.device,
        hkv: int,
        max_batch: int,
        window: int,
        mma: bool = True,
    ):
        if window <= 0:
            raise ValueError("window must be positive")
        self.mma = mma
        self.max_splits = (window + MAX_QUERY_LEN + CHUNK - 1) // CHUNK
        self.hkv, self.window, self.max_batch = int(hkv), int(window), int(max_batch)
        n = self.max_batch * self.hkv * self.max_splits
        self.part_o = torch.zeros(
            (n, ROWS, HEAD_DIM), dtype=torch.float32, device=device
        )
        self.part_m = torch.full((n, ROWS), -1e30, dtype=torch.float32, device=device)
        self.part_l = torch.zeros((n, ROWS), dtype=torch.float32, device=device)
        self._lib = _load(device)

    def __call__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        seqused_k: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        scale: float,
        out: torch.Tensor,
        num_reqs: int | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """q/out: [T, H, D] bf16 contiguous; caches: [num_blocks, block, hkv, D]
        bf16 views with a unit last stride (K and V identically strided);
        block_table [B, max_blocks] int32; seqused_k [B] int32; cu [B+1] int32."""
        batch = int(cu_seqlens_q.numel() - 1) if num_reqs is None else int(num_reqs)
        if batch > self.max_batch:
            raise ValueError(f"batch {batch} exceeds workspace {self.max_batch}")
        heads = q.shape[1]
        if heads != self.hkv * GQA or q.shape[2] != HEAD_DIM:
            raise ValueError("unsupported head geometry")
        if not (q.is_contiguous() and out.is_contiguous()):
            raise ValueError("q and out must be contiguous")
        if k_cache.dtype != torch.bfloat16 or k_cache.shape[2] != self.hkv:
            raise ValueError("unsupported KV cache")
        if k_cache.shape[3] != HEAD_DIM or k_cache.stride(3) != 1:
            raise ValueError("unsupported KV cache layout")
        if v_cache.stride() != k_cache.stride() or v_cache.shape != k_cache.shape:
            raise ValueError("K and V caches must share layout")
        s_blk, s_pos, s_h = (int(x) for x in k_cache.stride()[:3])
        fn = (
            self._lib.dflash_attn_launch_mma
            if self.mma
            else self._lib.dflash_attn_launch
        )
        rc = fn(
            q.data_ptr(),
            heads * HEAD_DIM,
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            s_blk,
            s_pos,
            s_h,
            int(k_cache.shape[1]),
            self.hkv,
            heads,
            block_table.data_ptr(),
            int(block_table.stride(0)),
            seqused_k.data_ptr(),
            cu_seqlens_q.data_ptr(),
            float(scale),
            self.window,
            1 if causal else 0,
            self.part_o.data_ptr(),
            self.part_m.data_ptr(),
            self.part_l.data_ptr(),
            self.max_splits,
            batch,
            out.data_ptr(),
            torch.cuda.current_stream(q.device).cuda_stream,
        )
        if rc != 0:
            raise RuntimeError(f"dflash_attn launch failed: {rc}")
        return out


__all__ = [
    "DFlashDecodeAttention",
    "is_available",
    "CHUNK",
    "ROWS",
    "HEAD_DIM",
    "GQA",
    "MAX_QUERY_LEN",
]
