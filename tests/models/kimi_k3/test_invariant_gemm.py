# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fixed-order small-N GEMM: row-count invariance, epilogues and gating.

Runs on a GPU (bf16 operands, the served dtype) or under the Triton
interpreter (``TRITON_INTERPRET=1``, fp16 operands: the interpreter has no
bf16 arithmetic). The invariance property is a property of the kernel's
structure, so the interpreter proves it for the kernel logic and the GPU run
proves it for the compiled kernel.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm.models.kimi_k3.nvidia.ops import invariant_gemm

INTERPRET = os.getenv("TRITON_INTERPRET") == "1"
CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not (CUDA or INTERPRET), reason="needs a GPU or TRITON_INTERPRET=1"
)
DEVICE = "cuda" if CUDA else "cpu"
DTYPE = torch.bfloat16 if CUDA else torch.float16
# (K, N) of the three served shapes the kernel replaces.
ROUTER = (7168, 104)
DOWN = (7168, 400)
UP = (3584, 796)
HIDDEN = 7168


def _operands(
    rows: int, k: int, n: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    a = (torch.randn(rows, k, generator=generator) * 0.05).to(DTYPE).to(DEVICE)
    # The served weights are [N, K] row-major; the GEMM consumes their
    # transpose, so B has stride (1, K).
    weight = (torch.randn(n, k, generator=generator) * 0.05).to(DTYPE).to(DEVICE)
    return a, weight.t()


def _reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.float() @ b.float()


@pytest.mark.parametrize("shape", [ROUTER, DOWN])
def test_mm_matches_fp32_reference(shape: tuple[int, int]) -> None:
    k, n = shape
    a, b = _operands(96, k, n, seed=1)
    out = invariant_gemm.mm(a, b)
    assert out.dtype == DTYPE and out.shape == (96, n)
    torch.testing.assert_close(out.float(), _reference(a, b), atol=4e-3, rtol=1e-2)


def test_fp32_output_is_the_unrounded_accumulator() -> None:
    k, n = ROUTER
    a, b = _operands(64, k, n, seed=2)
    out = invariant_gemm.mm(a, b, out_dtype=torch.float32)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _reference(a, b), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("shape", [ROUTER, DOWN])
def test_rows_do_not_depend_on_row_count_or_position(shape: tuple[int, int]) -> None:
    """A row inside a 96-row call, inside either 48-row half, and alone in a
    one-row call carries identical bits."""
    k, n = shape
    a, b = _operands(96, k, n, seed=3)
    whole = invariant_gemm.mm(a, b, out_dtype=torch.float32)
    halves = torch.cat(
        [
            invariant_gemm.mm(a[:48].contiguous(), b, out_dtype=torch.float32),
            invariant_gemm.mm(a[48:].contiguous(), b, out_dtype=torch.float32),
        ]
    )
    assert torch.equal(whole, halves)
    for row in (0, 31, 32, 47, 48, 95):
        alone = invariant_gemm.mm(a[row : row + 1], b, out_dtype=torch.float32)
        assert torch.equal(alone[0], whole[row]), f"row {row}"


def test_addmm_accumulates_into_a_strided_column_window() -> None:
    """The tier-2 tail: ``hidden[:, window] += latent @ up_proj_shard.T`` with
    the window at an offset inside a full-width output, rounded once."""
    k, n = UP
    rows = 64
    a, b = _operands(rows, k, n, seed=4)
    generator = torch.Generator().manual_seed(5)
    hidden = (torch.randn(rows, HIDDEN, generator=generator) * 0.5).to(DTYPE).to(DEVICE)
    window = hidden.narrow(1, n, n)
    before = window.clone()
    outside = torch.cat([hidden[:, :n], hidden[:, 2 * n :]], dim=1).clone()
    returned = invariant_gemm.addmm_(window, a, b)
    assert returned.data_ptr() == window.data_ptr()
    expected = (before.float() + _reference(a, b)).to(DTYPE)
    torch.testing.assert_close(window.float(), expected.float(), atol=8e-3, rtol=1e-2)
    assert torch.equal(torch.cat([hidden[:, :n], hidden[:, 2 * n :]], dim=1), outside)
    # Row-count invariance holds through the accumulate epilogue as well.
    hidden_half = torch.cat([before[:32], torch.zeros_like(before[:32])]).clone()
    invariant_gemm.addmm_(hidden_half[:32], a[:32].contiguous(), b)
    assert torch.equal(hidden_half[:32], window[:32])


def test_mm_into_a_caller_owned_strided_output() -> None:
    k, n = DOWN
    a, b = _operands(40, k, n, seed=6)
    backing = torch.zeros(40, 2 * n, dtype=DTYPE, device=DEVICE)
    out = backing.narrow(1, 0, n)
    assert invariant_gemm.mm(a, b, out=out).data_ptr() == out.data_ptr()
    assert torch.equal(out, invariant_gemm.mm(a, b))
    assert not backing[:, n:].any()


def test_operand_checks() -> None:
    k, n = ROUTER
    a, b = _operands(8, k, n, seed=7)
    with pytest.raises(ValueError):
        invariant_gemm.mm(a, b.t())
    with pytest.raises(ValueError):
        invariant_gemm.mm(a, b, out=torch.empty(8, n + 1, dtype=DTYPE, device=DEVICE))
    with pytest.raises(ValueError):
        invariant_gemm.mm(a, b, out_dtype=torch.float64)
    with pytest.raises(ValueError):
        invariant_gemm.mm(a.float(), b.float())


def test_gating_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = torch.empty(64, 8, dtype=torch.bfloat16)
    monkeypatch.delenv("VLLM_K3_INVARIANT_SMALL_N_GEMM", raising=False)
    invariant_gemm.settings.cache_clear()
    assert invariant_gemm.settings() == (False, 32)
    assert not invariant_gemm.applies_to(rows)
    monkeypatch.setenv("VLLM_K3_INVARIANT_SMALL_N_GEMM", "1")
    monkeypatch.setenv("VLLM_K3_INVARIANT_GEMM_MIN_ROWS", "48")
    invariant_gemm.settings.cache_clear()
    assert invariant_gemm.settings() == (True, 48)
    # CPU tensors never qualify; a CUDA bf16 2-D tensor of enough rows does.
    assert not invariant_gemm.applies_to(rows)
    if CUDA:
        on_gpu = rows.cuda()
        assert invariant_gemm.applies_to(on_gpu, on_gpu)
        assert not invariant_gemm.applies_to(on_gpu[:47], on_gpu)
        assert not invariant_gemm.applies_to(on_gpu, on_gpu.float())
        assert not invariant_gemm.applies_to(on_gpu.view(-1), on_gpu)
    invariant_gemm.settings.cache_clear()


def test_tile_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_K3_INVARIANT_GEMM_TILE", raising=False)
    invariant_gemm._tile_override.cache_clear()
    assert invariant_gemm.tiles_for(104) == (32, 128, 64, 4, 3)
    assert invariant_gemm.tiles_for(400) == (64, 128, 64, 4, 3)
    monkeypatch.setenv("VLLM_K3_INVARIANT_GEMM_TILE", "16,64,32,2,2")
    invariant_gemm._tile_override.cache_clear()
    assert invariant_gemm.tiles_for(400) == (16, 64, 32, 2, 2)
    monkeypatch.setenv("VLLM_K3_INVARIANT_GEMM_TILE", "16,64")
    invariant_gemm._tile_override.cache_clear()
    with pytest.raises(ValueError):
        invariant_gemm.tiles_for(400)
    invariant_gemm._tile_override.cache_clear()
