# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        dcp_size=dcp_size,
        vllm_config=vllm_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


@pytest.mark.parametrize("dummy_run_fails", [False, True])
def test_glm_dcp_attention_profile_uses_single_request_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    dummy_run_fails: bool,
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(
        architecture="Glm5NextForConditionalGeneration"
    )
    runner.dcp_size = 4
    runner.cp_interleave = 4
    runner.max_num_tokens = 4096
    events: list[object] = []

    monkeypatch.setattr(
        model_runner_module,
        "_init_minimal_kv_cache_for_profiling",
        lambda _: events.append("init-kv"),
    )
    monkeypatch.setattr(
        model_runner_module,
        "_teardown_profiling_state",
        lambda _: events.append("cleanup"),
    )

    def dummy_run(*args, **kwargs):
        events.append(("dummy-run", args, kwargs))
        if dummy_run_fails:
            raise RuntimeError("expected DCP profile failure")
        return torch.empty(1), torch.empty(1)

    runner._dummy_run = dummy_run
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))

    if dummy_run_fails:
        with pytest.raises(RuntimeError, match="expected DCP profile failure"):
            runner.profile_glm_dcp_attention()
    else:
        runner.profile_glm_dcp_attention()

    assert events[0] == "init-kv"
    assert events[1] == (
        "dummy-run",
        (4096,),
        {
            "context_len": 16,
            "skip_eplb": True,
            "is_profile": True,
            "single_request_prefill": True,
            "profile_all_kv_cache_groups": True,
        },
    )
    assert events[-1] == "cleanup"
    if not dummy_run_fails:
        assert events[-2] == "sync"


@pytest.mark.parametrize(
    ("architecture", "dcp_size"),
    [("OtherArchitecture", 4), ("Glm5NextForConditionalGeneration", 1)],
)
def test_glm_dcp_attention_profile_skips_irrelevant_configurations(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    dcp_size: int,
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(architecture=architecture)
    runner.dcp_size = dcp_size
    initialized = False

    def record_initialization(_):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(
        model_runner_module,
        "_init_minimal_kv_cache_for_profiling",
        record_initialization,
    )

    runner.profile_glm_dcp_attention()

    assert not initialized


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV4ForCausalLM", "DeepseekV4ForConditionalGeneration"],
)
@pytest.mark.parametrize("dummy_run_fails", [False, True])
def test_deepseek_v4_attention_profile_uses_reachable_prefill_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    dummy_run_fails: bool,
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(architecture=architecture)
    runner.max_num_tokens = 4096
    events: list[object] = []

    monkeypatch.setattr(
        model_runner_module,
        "_init_minimal_kv_cache_for_profiling",
        lambda _: events.append("init-kv"),
    )
    monkeypatch.setattr(
        model_runner_module,
        "_teardown_profiling_state",
        lambda _: events.append("cleanup"),
    )

    def dummy_run(*args, **kwargs):
        events.append(("dummy-run", args, kwargs))
        if dummy_run_fails:
            raise RuntimeError("expected DeepSeek V4 profile failure")
        return torch.empty(1), torch.empty(1)

    runner._dummy_run = dummy_run
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))

    if dummy_run_fails:
        with pytest.raises(RuntimeError, match="expected DeepSeek V4 profile failure"):
            runner._profile_deepseek_v4_attention()
    else:
        runner._profile_deepseek_v4_attention()

    assert events[0] == "init-kv"
    assert events[1] == (
        "dummy-run",
        (4096,),
        {
            "skip_eplb": True,
            "is_profile": True,
            "single_request_prefill": True,
            "profile_all_kv_cache_groups": True,
        },
    )
    assert events[-1] == "cleanup"
    if not dummy_run_fails:
        assert events[-2] == "sync"


def test_deepseek_v4_attention_profile_skips_other_architectures(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(architecture="OtherArchitecture")
    initialized = False

    def record_initialization(_):
        nonlocal initialized
        initialized = True

    monkeypatch.setattr(
        model_runner_module,
        "_init_minimal_kv_cache_for_profiling",
        record_initialization,
    )

    runner._profile_deepseek_v4_attention()

    assert not initialized


def test_profile_run_profiles_attention_before_releasing_encoder_cache(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.supports_mm_inputs = False
    runner.max_num_tokens = 4096
    runner.is_last_pp_rank = False
    events: list[object] = []

    runner._dummy_run = lambda *args, **kwargs: (
        events.append(("dummy-run", args, kwargs)) or (None, None)
    )
    runner._profile_deepseek_v4_attention = lambda: events.append("profile-attention")
    runner.reset_encoder_cache = lambda: events.append("reset-encoder-cache")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))

    runner.profile_run()

    assert events == [
        (
            "dummy-run",
            (4096,),
            {"skip_attn": True, "is_profile": True},
        ),
        "profile-attention",
        "sync",
        "reset-encoder-cache",
    ]
