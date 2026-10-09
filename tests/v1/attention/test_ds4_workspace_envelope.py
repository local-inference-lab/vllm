# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.deepseek_v4.nvidia import b12x as adapter
from vllm.utils.b12x import B12xWorkload
from vllm.v1.worker import workspace as workspace_module
from vllm.v1.worker.b12x_startup import B12xPreparationCoordinator
from vllm.v1.worker.workspace import WorkspaceManager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="B12X device plan required")
@pytest.mark.parametrize("max_length", [786_688, 1_048_576])
@pytest.mark.parametrize("drafts", [0, 5])
@pytest.mark.parametrize("image_tokens", [0, 384])
def test_declared_workspace_covers_shorter_compressed_prefixes(
    monkeypatch, max_length, drafts, image_tokens
):
    """Reserve decode-prefix and capacity-prefill plans before workspace lock.

    Native plan scratch requirements are evaluated without compiling attention
    kernels. The job stub completes only the coordinator's control protocol;
    declaration, plan selection, and workspace reservation are real.
    """
    device = torch.device("cuda", torch.accelerator.current_device_index())
    spec = SimpleNamespace(use_dspark=lambda: True, num_speculative_tokens=drafts)
    layer = adapter.DeepseekV4B12xAttention.__new__(adapter.DeepseekV4B12xAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.self_attn"
    layer.indexer = None
    layer.compress_ratio = 128
    layer.max_model_len = max_length
    layer.max_num_batched_tokens = 4096
    layer.max_image_tokens = image_tokens
    layer.window_size = 128
    layer.padded_heads = 32
    layer._b12x_cache_page_views = {}
    layer._b12x_mla_plans = {}
    layer.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        speculative_config=spec if drafts else None,
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=4096),
    )
    page_bytes = adapter._DSV4_CACHE_BYTES_PER_TOKEN
    layer.swa_cache_layer = SimpleNamespace(
        block_size=256,
        kv_cache=torch.empty((2, 256 * page_bytes), dtype=torch.uint8, device=device),
    )
    layer.kv_cache = torch.empty((2, 2 * page_bytes), dtype=torch.uint8, device=device)
    workload = B12xWorkload(
        stage="state",
        token_counts=(1, 8, 48, 128, 256, 4096),
        fixed_token_counts=(1, 8, 48, 128, 256),
        output_dtype=torch.bfloat16,
        max_tokens=4096,
        max_seqs=8,
        max_model_len=max_length,
        speculative_tokens=drafts,
    )
    (unit,) = layer.get_b12x_preparation_units(layer, workload)
    queries = [request.plan.query for request in unit.requests]
    widths = {128, 256, 512, 1024, 2048, 4096}
    widths.add(6272 if max_length == 786_688 else 8192)
    assert {q.indexed_width for q in queries} == widths
    assert {q.query_rows for q in queries if q.mode == "decode"} == set(range(1, 257))
    assert {q.query_rows for q in queries if q.mode == "extend"} == {4096}

    job = Mock()
    job.advance.return_value = SimpleNamespace(
        done=True, pending_compilation=False, ready_collectives=()
    )
    session = Mock()
    session.begin.return_value = job
    workspace = WorkspaceManager(device, num_ubatches=2)
    coordinator = B12xPreparationCoordinator(
        session,
        [(unit.requests, False)],
        global_rank=0,
        world_group=None,
        process_local_only=True,
        workspace=workspace,
    )
    outcome = coordinator.advance()
    assert outcome["done"] and outcome["error"] is None, outcome
    workspace.lock()
    pointers = [buffer.data_ptr() for buffer in workspace._current_workspaces]
    sizes = [buffer.numel() for buffer in workspace._current_workspaces]
    assert sizes[0] == sizes[1] > 0

    for ubatch in range(2):
        monkeypatch.setattr(
            workspace_module, "dbo_current_ubatch_id", lambda ubatch=ubatch: ubatch
        )
        for rows in (1, 48, 128, 240, 256, 257, 1024, 4096):
            mode = "decode" if rows <= 256 else "extend"
            swa_widths = {q.swa_width for q in queries if q.mode == mode}
            for swa_width in swa_widths:
                for index_width in widths:
                    plan = layer._b12x_mla_plan(
                        mode,
                        rows,
                        torch.empty((rows, swa_width), device="meta"),
                        torch.empty((rows, index_width), device="meta"),
                    )
                    workspace.get_simultaneous(
                        *((spec.shape, spec.dtype) for spec in plan.scratch_specs())
                    )
                assert (
                    workspace._current_workspaces[ubatch].data_ptr() == pointers[ubatch]
                )
