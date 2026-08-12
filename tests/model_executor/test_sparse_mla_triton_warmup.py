# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup import sparse_mla_triton_warmup

# Mirrors the production runner interface: both the V1 (gpu_model_runner) and
# V2 (gpu/model_runner) runners expose ``vllm_config.parallel_config`` and
# ``dcp_rank``.  The V2-only aliases ``dcp_size``/``cp_interleave`` are
# deliberately *not* provided here so that a regression to ``getattr(runner,
# "dcp_size", 1)`` silently degrades to DCP1 and fails the DCP4 assertion.
_DCP_CASES = [(1, 1, 0), (4, 64, 2)]


@pytest.mark.parametrize(
    ("dcp_world_size", "cp_kv_cache_interleave_size", "dcp_rank"),
    _DCP_CASES,
)
def test_b12x_prefill_metadata_warmup_uses_runtime_dcp_shape(
    monkeypatch,
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int,
    dcp_rank: int,
) -> None:
    calls = []

    class B12xBackend:
        @staticmethod
        def get_name() -> str:
            return "B12X_MLA_SPARSE"

    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_world_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        device=torch.device("cpu"),
        dcp_rank=dcp_rank,
        vllm_config=SimpleNamespace(parallel_config=parallel_config),
        attn_groups=[[SimpleNamespace(backend=B12xBackend())]],
    )
    worker = SimpleNamespace(
        model_runner=runner,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=3072),
    )

    monkeypatch.setattr(
        sparse_mla_triton_warmup,
        "_warm_prefill_chunk_metadata_kernel",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    sparse_mla_triton_warmup.sparse_mla_triton_warmup_if_needed(worker)

    assert calls == [
        (
            (torch.device("cpu"),),
            {
                "compress_ratio": 1,
                "query_len": 8,
                "dcp_rank": dcp_rank,
                "dcp_world_size": dcp_world_size,
                "cp_kv_cache_interleave_size": cp_kv_cache_interleave_size,
            },
        )
    ]


def test_b12x_prewarm_extend_kernels_warms_ckv_plan_and_global_topk_remap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 96-line ``_prewarm_extend_kernels_once`` restructuring must warm both
    the full-CKV local-head extend plan *and* the global top-k → gathered-CKV
    remap kernel, using the runtime DCP specialization."""
    import vllm.v1.attention.backends.mla.b12x_mla_sparse as b12x

    # Isolate the module-level dedup set so prior tests do not short-circuit.
    monkeypatch.setattr(b12x, "_EXTEND_PREWARM_DONE", set())

    remap_calls: list[dict] = []
    mask_calls: list = []

    def fake_remap(*args, **kwargs) -> None:
        remap_calls.append(kwargs)

    def fake_mask(*args, **kwargs) -> None:
        mask_calls.append(args)

    monkeypatch.setattr(b12x, "_map_global_topk_to_gathered_ckv", fake_remap)
    monkeypatch.setattr(b12x, "_mask_page_table_after_nsa_len", fake_mask)

    class FakePlan:
        def __init__(self, name: str) -> None:
            self.name = name
            self.bind_calls: list[dict] = []

        def bind(self, **kwargs):
            self.bind_calls.append(kwargs)
            return SimpleNamespace(plan=self.name)

    ckv_plan = FakePlan("ckv")
    base_plan = FakePlan("base")

    forward_calls: list[dict] = []

    # The real method allocates tensors on ``self.device``.  On a CPU-only host
    # we cannot allocate on CUDA, so redirect every ``device=cuda`` allocation
    # to CPU.  The tensors are only ever handed to the mocked kernels/plans, so
    # their backing store is irrelevant.
    _real_zeros = torch.zeros
    _real_empty = torch.empty
    _real_full = torch.full
    _real_ones = torch.ones

    def _to_cpu(device):
        if device is not None and str(device).startswith("cuda"):
            return torch.device("cpu")
        return device

    def _patched(fn):
        def wrapper(*args, **kwargs):
            if "device" in kwargs:
                kwargs["device"] = _to_cpu(kwargs["device"])
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(torch, "zeros", _patched(_real_zeros))
    monkeypatch.setattr(torch, "empty", _patched(_real_empty))
    monkeypatch.setattr(torch, "full", _patched(_real_full))
    monkeypatch.setattr(torch, "ones", _patched(_real_ones))

    mock_self = SimpleNamespace(
        device=torch.device("cuda"),
        q_head_dim=576,
        kv_lora_rank=512,
        scale=1.0 / math.sqrt(576),
        _kernel_num_heads=8,
        _ckv_kernel_num_heads=8,
        topk_tokens=128,
        block_size=128,
        need_to_return_lse_for_decode=False,
        kv_cache_dtype="fp8_ds_mla",
        _kv_fp8_rope=True,
        _ckv_extend_plan=ckv_plan,
        _extend_plan=base_plan,
        dcp_world_size=4,
        cp_kv_cache_interleave_size=64,
        _kv_record_bytes=656,
        _scratch_nbytes=1024,
        _b12x_kernel_format_kwargs=lambda *a, **k: {},
        _sync_warmup=lambda *a, **k: None,
        _sparse_mla_extend_forward=lambda **k: forward_calls.append(k),
    )

    b12x.B12xMLASparseImpl._prewarm_extend_kernels_once(mock_self, 8)

    # Global top-k remap prewarmed with the runtime DCP specialization.
    assert len(remap_calls) == 1
    assert remap_calls[0]["dcp_size"] == 4
    assert remap_calls[0]["cp_kv_cache_interleave_size"] == 64
    # Page-table masking ran immediately after the remap.
    assert len(mask_calls) == 1
    # Full-CKV local-head extend plan was warmed.
    assert len(ckv_plan.bind_calls) >= 1
    # Base extend plan was also warmed.
    assert len(base_plan.bind_calls) >= 1
    # Both plans drove at least one forward launch.
    assert len(forward_calls) >= 2
