# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace
from weakref import ref

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("logits_only", [False, True])
def test_boundary_capture_uses_recovered_accepted_state(
    monkeypatch, recovery, logits_only
):
    """A stop inside a draft window must be applied before checkpoint export."""
    events = []
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.is_last_pp_rank = False
    runner.cache_config = SimpleNamespace(use_kda_recoverssm=recovery)
    runner.req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([19])),
        last_sampled_tokens=None,
        all_token_ids=SimpleNamespace(gpu=None),
        total_len=SimpleNamespace(gpu=None),
    )
    sampled = torch.tensor([4])

    def update(*args):
        # A restored prompt can terminate after sampling its first token.
        sampled.fill_(1 if logits_only else 2)
        events.append("trim")

    def commit(*args):
        assert not logits_only
        assert sampled.item() == 2
        events.append("commit")

    def capture(*args, accepted_state_committed=False):
        assert accepted_state_committed is recovery
        events.append("capture")

    monkeypatch.setattr(model_runner_module, "post_update", update)
    runner.model_state = SimpleNamespace(postprocess_state=commit)
    runner.boundary_checkpoint_state = SimpleNamespace(capture_mamba=capture)
    runner.postprocess_sampled(
        torch.tensor([0]),
        torch.tensor([[1, 2, 3, 4]]),
        sampled,
        torch.tensor([0]),
        boundary_capture=torch.empty(3, 1, 3),
        logits_only=logits_only,
    )
    if logits_only:
        assert events == ["trim", "capture"]
    else:
        assert events == (
            ["trim", "commit", "capture"] if recovery else ["trim", "capture", "commit"]
        )


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.jit_warmup_registry = JitWarmupRegistry(runner.vllm_config)
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.speculative_config = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


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
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        dcp_size=dcp_size,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
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


def _make_capture_runner(captured: bool) -> GPUModelRunner:
    """Minimal V2 runner for capture_model: fakes everything except the
    cudagraph_manager's needs_capture decision."""
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_state = SimpleNamespace(supports_mm_inputs=False)
    runner.cudagraph_manager = SimpleNamespace(
        needs_capture=lambda: captured,
        capture=lambda *args, **kwargs: None,
    )
    runner.lora_config = None
    runner.maybe_setup_dummy_loras = lambda _cfg: contextlib.nullcontext()
    runner.speculator = None
    runner.adaptive_verification = None
    runner.model = None
    runner.input_buffers = None
    runner.pcp_manager = None
    runner.intermediate_tensors = None
    runner.block_tables = None
    runner.attn_groups = None
    runner.kv_cache_config = None
    runner.use_aux_hidden_state_outputs = False
    runner.kv_connector = model_runner_module.NO_OP_KV_CONNECTOR
    return runner


def test_capture_model_locks_workspace_after_capture(monkeypatch):
    """A workspace resize after capture frees the buffer the captured graphs
    baked in, so capture_model must lock the workspace before returning
    (https://github.com/vllm-project/vllm/issues/55336)."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model()

    assert lock_calls == ["lock"]


def test_capture_model_skips_lock_when_nothing_captured(monkeypatch):
    """With no graphs to capture (e.g. enforce_eager) there is nothing baked
    into the workspace, so the early return must not lock it."""
    runner = _make_capture_runner(captured=False)
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    assert runner.capture_model() == 0
    assert lock_calls == []


def test_capture_model_profile_only_skips_lock(monkeypatch):
    """The memory-profiling capture pass runs before kernel warmup and the
    real capture; locking there would stop the warmup from growing the
    workspace to its scheduler-realistic size."""
    runner = _make_capture_runner(captured=True)
    monkeypatch.setattr(
        model_runner_module, "freeze_gc_for_cudagraph_capture", contextlib.nullcontext
    )
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda: (1 << 30, 1 << 30)
    )
    lock_calls = []
    monkeypatch.setattr(
        model_runner_module, "lock_workspace", lambda: lock_calls.append("lock")
    )

    runner.capture_model(profile_only=True)

    assert lock_calls == []


@pytest.mark.parametrize("cudagraph_metrics", [False, True])
def test_boundary_logits_only_dispatches_pending_cache_tasks(
    monkeypatch, cudagraph_metrics
):
    """A restored prompt can share a step with another request's cache store."""
    checkpoint = SimpleNamespace(num_tokens=4, auxiliary_block_ids=[3])
    hidden_states = torch.ones(1, 8)
    input_batch = SimpleNamespace(
        num_reqs=1,
        **{
            name: torch.empty(1, dtype=torch.int32)
            for name in (
                "positions",
                "input_ids",
                "seq_lens",
                "seq_lens_cpu_upper_bound",
            )
        },
    )
    dispatched_tasks = []
    dp_sync = object()
    cudagraph_stats = object()
    scheduler_output = SimpleNamespace(
        boundary_logits_only=True,
        scheduled_new_reqs=[
            SimpleNamespace(
                boundary_checkpoint=checkpoint, prefill_token_ids=[1, 2, 3, 4]
            )
        ],
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"restored-request": 1},
        finished_req_ids=set(),
        kv_connector_metadata=["store-another-request"],
        resolve_num_spec_tokens_to_schedule=lambda _: 0,
    )
    runner = SimpleNamespace(
        **{
            name: lambda *args: None
            for name in (
                "update_pp_decode_requests",
                "finish_requests",
                "free_states",
                "add_requests",
                "update_requests",
            )
        },
        block_tables=SimpleNamespace(apply_staged_writes=lambda: None),
        boundary_checkpoint_state=SimpleNamespace(
            get_hidden_states=lambda _: hidden_states
        ),
        kv_connector=SimpleNamespace(
            pre_forward=lambda output: dispatched_tasks.extend(
                output.kv_connector_metadata
            )
        ),
        speculator=None,
        model_state=SimpleNamespace(),
        lora_config=None,
        is_encoder_decoder=False,
        dp_size=1,
        dp_rank=0,
        pcp_manager=None,
        ubatch_runner=None,
        parallel_config=SimpleNamespace(),
        observability_config=SimpleNamespace(cudagraph_metrics=cudagraph_metrics),
        num_speculative_steps=0,
        decode_query_len=1,
        cudagraph_manager=None,
        gather_batch_req_state=lambda *args: (SimpleNamespace(num_tokens=1), 1),
        prepare_inputs=lambda *args: input_batch,
        prepare_attn=lambda *args: pytest.fail("logits-only must skip attention"),
    )
    monkeypatch.setattr(
        model_runner_module,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (SimpleNamespace(num_tokens=1), dp_sync),
    )
    monkeypatch.setattr(
        model_runner_module, "make_cudagraph_stats", lambda *args: cudagraph_stats
    )

    assert GPUModelRunner.execute_model(runner, scheduler_output) is None

    assert dispatched_tasks == ["store-another-request"]
    assert runner.execute_model_state.boundary_logits_only
    assert runner.execute_model_state.hidden_states is hidden_states
    assert runner.execute_model_state.dp_sync is dp_sync
    assert runner.execute_model_state.cudagraph_stats is (
        cudagraph_stats if cudagraph_metrics else None
    )
    assert input_batch.positions.item() == 3


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
    [
        "DeepseekV4ForCausalLM",
        "DeepseekV4ForConditionalGeneration",
        "DeepseekV41ForCausalLM",
    ],
)
@pytest.mark.parametrize(
    ("init_fails", "dummy_run_fails"),
    [(False, False), (False, True), (True, False)],
)
def test_deepseek_v4_attention_profile_uses_reachable_prefill_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    init_fails: bool,
    dummy_run_fails: bool,
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(architecture=architecture)
    runner.max_num_tokens = 4096
    events: list[object] = []

    def init_kv(_, *, num_blocks=None):
        events.append(("init-kv", num_blocks))
        if init_fails:
            raise RuntimeError("expected DeepSeek V4 KV initialization failure")

    monkeypatch.setattr(
        model_runner_module, "_init_minimal_kv_cache_for_profiling", init_kv
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

    prepare = lambda: events.append("prepare")

    if init_fails:
        with pytest.raises(
            RuntimeError, match="expected DeepSeek V4 KV initialization failure"
        ):
            runner._profile_deepseek_v4_attention(prepare)
    elif dummy_run_fails:
        with pytest.raises(RuntimeError, match="expected DeepSeek V4 profile failure"):
            runner._profile_deepseek_v4_attention(prepare)
    else:
        runner._profile_deepseek_v4_attention(prepare)

    assert events[0] == ("init-kv", 1)
    if init_fails:
        assert events == [("init-kv", 1), "cleanup"]
    else:
        assert events[1] == "prepare"
        assert events[2] == (
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
    if not init_fails and not dummy_run_fails:
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


def test_profile_run_releases_generic_outputs_before_deepseek_profile(
    monkeypatch, workspace_init
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.supports_mm_inputs = False
    runner.max_num_tokens = 4096
    runner.is_last_pp_rank = False
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    events: list[object] = []
    output_refs: list[ref] = []

    class ProfileOutput:
        pass

    def dummy_run(*args, **kwargs):
        events.append(("dummy-run", args, kwargs))
        outputs = (ProfileOutput(), ProfileOutput())
        output_refs.extend(ref(output) for output in outputs)
        return outputs

    def profile_attention(prepare_profile_state=None):
        assert all(output_ref() is None for output_ref in output_refs)
        events.append("profile-attention")

    runner._dummy_run = dummy_run
    runner._profile_deepseek_v4_attention = profile_attention
    runner.reset_encoder_cache = lambda: events.append("reset-encoder-cache")
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))

    runner.profile_run()

    assert events == [
        (
            "dummy-run",
            (4096,),
            {"skip_attn": True, "is_profile": True},
        ),
        "sync",
        "profile-attention",
        "reset-encoder-cache",
    ]


@pytest.mark.parametrize("num_speculative_steps", [0, 2])
def test_pipeline_drafts_do_not_depend_on_boundary_checkpoints(
    monkeypatch, num_speculative_steps
):
    from unittest.mock import Mock

    batch = SimpleNamespace(
        req_ids=["request"],
        num_reqs=1,
        idx_mapping=torch.tensor([0]),
        query_start_loc=torch.tensor([0, 1]),
        num_draft_tokens_per_req=None,
    )
    states = SimpleNamespace(
        draft_tokens=torch.tensor([[3, 4]]),
        all_token_ids=SimpleNamespace(gpu=None),
        num_computed_tokens=SimpleNamespace(gpu=None),
        prompt_len=SimpleNamespace(np=None),
    )
    pp = SimpleNamespace(broadcast=Mock(), broadcast_drafts=Mock())
    sampler_output = SimpleNamespace(sampled_token_ids=torch.tensor([[1]]))
    runner = SimpleNamespace(
        execute_model_state=model_runner_module.ExecuteModelState(
            input_batch=batch,
            attn_metadata=None,
            slot_mappings_by_layer=None,
            hidden_states=torch.ones(1, 8),
            aux_hidden_states=None,
            dp_sync=None,
            finished_req_ids=set(),
            ec_connector_output=None,
            routed_experts=None,
            cudagraph_stats=None,
            num_spec_tokens_to_schedule=0,
        ),
        is_last_pp_rank=True,
        pcp_manager=None,
        pp_handler=pp,
        sample=lambda *args: (sampler_output, torch.tensor([1]), torch.tensor([0])),
        prompt_logprobs_worker=SimpleNamespace(compute_prompt_logprobs=lambda *a: {}),
        model=SimpleNamespace(compute_logits=None),
        req_states=states,
        adaptive_verification=None,
        boundary_checkpoint_state=None,
        main_stream=None,
        output_copy_stream=None,
        check_ep_fault=False,
        speculator=None,
        num_speculative_steps=num_speculative_steps,
        device=torch.device("cpu"),
        postprocess_sampled=Mock(),
        block_tables=None,
        draft_tokens_handler=SimpleNamespace(set_draft_tokens=Mock()),
        kv_connector=SimpleNamespace(post_forward=lambda *a: None),
        eplb=SimpleNamespace(step=Mock()),
    )
    monkeypatch.setattr(model_runner_module, "AsyncOutput", lambda **kwargs: kwargs)

    GPUModelRunner.sample_tokens(runner, None)

    pp.broadcast.assert_called_once()
    if num_speculative_steps:
        pp.broadcast_drafts.assert_called_once_with(states.draft_tokens, batch)
        runner.draft_tokens_handler.set_draft_tokens.assert_called_once()
    else:
        pp.broadcast_drafts.assert_not_called()
        runner.draft_tokens_handler.set_draft_tokens.assert_not_called()
