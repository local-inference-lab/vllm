# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.warmup import kernel_warmup


def _worker(model: torch.nn.Module, *, backend_name: str | None = None):
    attn_groups = []
    if backend_name is not None:
        backend = SimpleNamespace(get_name=lambda: backend_name)
        attn_groups = [[SimpleNamespace(backend=backend)]]
    return SimpleNamespace(
        get_model=lambda: model,
        model_runner=SimpleNamespace(attn_groups=attn_groups),
    )


def test_flashinfer_compute_detection_ignores_b12x_only_configuration() -> None:
    b12x_kernel_cls = type(
        "B12xKernel",
        (),
        {"__module__": "vllm.model_executor.kernels.linear.scaled_mm.b12x"},
    )

    class ModuleWithB12xKernel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = SimpleNamespace(kernel=b12x_kernel_cls())

    worker = _worker(ModuleWithB12xKernel(), backend_name="B12X")

    assert not kernel_warmup._uses_flashinfer_compute_kernels(worker)


def test_flashinfer_compute_detection_finds_attention_backend() -> None:
    worker = _worker(torch.nn.Linear(2, 2), backend_name="FLASHINFER")

    assert kernel_warmup._uses_flashinfer_compute_kernels(worker)


def test_flashinfer_compute_detection_finds_nested_model_kernel() -> None:
    flashinfer_kernel_cls = type(
        "FlashInferKernel",
        (),
        {"__module__": "vllm.model_executor.kernels.linear.scaled_mm.flashinfer"},
    )

    class ModuleWithFlashInferKernel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.quant_method = SimpleNamespace(
                scheme=SimpleNamespace(kernel=flashinfer_kernel_cls())
            )

    worker = _worker(ModuleWithFlashInferKernel())

    assert kernel_warmup._uses_flashinfer_compute_kernels(worker)
