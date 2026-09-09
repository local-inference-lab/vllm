# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for GLM mHC admission, reduction, and token ownership."""

import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch

import vllm.distributed as distributed
import vllm.envs as envs
import vllm.forward_context as forward_context
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.models.glm5next.nvidia import mhc_prefill_sharding as ownership
from vllm.models.glm5next.nvidia import model as glm
from vllm.utils import torch_utils


def configuration():
    parallel = NS(
        tensor_parallel_size=4,
        decode_context_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
        use_sequence_parallel_moe=False,
    )
    return NS(
        parallel_config=parallel,
        model_config=NS(dtype=torch.bfloat16),
        scheduler_config=NS(max_num_batched_tokens=8192),
        compilation_config=NS(cudagraph_capture_sizes=[4, 8, 16, 32, 64]),
    )


def metadata(**changes):
    fields = dict(
        num_prefills=1,
        num_prefill_tokens=8192,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
    )
    return NS(**(fields | changes))


@pytest.fixture
def admission(monkeypatch):
    comm = NS(
        available=True,
        disabled=False,
        world_size=4,
        rank=0,
        device=torch.device("cuda", 0),
    )
    group = NS(
        world_size=4,
        rank_in_group=0,
        cpu_group=object(),
        device_communicator=NS(pynccl_comm=comm),
    )
    votes = []

    def vote(result, value, **kwargs):
        votes.append(value)
        result[:] = [value] * 4

    monkeypatch.setattr(distributed, "get_tp_group", lambda: group)
    monkeypatch.setattr(torch.distributed, "all_gather_object", vote)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: NS(
            major=12, minor=1, multi_processor_count=48, name="NVIDIA GB10"
        ),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return NS(
        group=group,
        votes=votes,
        config=configuration(),
        model=NS(config=NS(hidden_size=4096)),
    )


def test_opt_in_defaults_off_without_device_access(monkeypatch, admission):
    monkeypatch.delenv("VLLM_GLM53_MHC_PREFILL_SHARD", raising=False)
    assert envs.environment_variables["VLLM_GLM53_MHC_PREFILL_SHARD"]() is False
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: pytest.fail("Disabled ownership must not inspect a CUDA device"),
    )
    ownership.configure(admission.model, admission.config, False)
    assert not admission.model._mhc_prefill_enabled
    assert ownership.maybe_create(admission.model, None, None) is None


@pytest.mark.parametrize(
    "sms,minor,name",
    [(47, 1, "NVIDIA GB10"), (48, 0, "NVIDIA GB10"), (48, 1, "NVIDIA RTX")],
)
def test_unsupported_hardware_rejects_all_ranks(
    monkeypatch, admission, sms, minor, name
):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: NS(major=12, minor=minor, multi_processor_count=sms, name=name),
    )
    with pytest.raises(RuntimeError, match="configuration admission"):
        ownership.configure(admission.model, admission.config, True)
    assert admission.votes[0][1] is not None


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 2),
        ("decode_context_parallel_size", 3),
        ("decode_context_parallel_size", 8),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("enable_expert_parallel", True),
        ("enable_eplb", True),
        ("use_sequence_parallel_moe", True),
    ],
)
def test_unsupported_parallel_config_is_rejected(admission, field, value):
    setattr(admission.config.parallel_config, field, value)
    with pytest.raises(RuntimeError, match="configuration admission"):
        ownership.configure(admission.model, admission.config, True)


@pytest.mark.parametrize("dcp", [1, 2, 4])
@pytest.mark.parametrize("rank", range(4))
def test_dcp_keeps_tp_communicator_and_quarter_token_ownership(
    monkeypatch, admission, dcp, rank
):
    """DCP subgroups must not replace TP rank ordering or its collective group."""
    from vllm.models.glm5next.nvidia.kda import Glm5NextLinearAttention

    admission.config.parallel_config.decode_context_parallel_size = dcp
    admission.group.rank_in_group = rank
    comm = admission.group.device_communicator.pynccl_comm
    comm.rank = rank
    monkeypatch.setattr(
        distributed,
        "get_dcp_group",
        lambda: pytest.fail("Token ownership must use the TP group"),
    )
    projection = linear.RowParallelLinear.__new__(linear.RowParallelLinear)
    torch.nn.Module.__init__(projection)
    projection.tp_size, projection.reduce_results, projection.bias = 4, True, None
    attn = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    torch.nn.Module.__init__(attn)
    attn.prefix, attn.o_proj = "gdn", projection
    admission.model.is_sequence_parallel = False
    admission.model._active_layers = [
        NS(
            mhc=True,
            is_mtp_layer=False,
            _b12x_mhc=object(),
            self_attn=attn,
            _mlp_is_moe=False,
            mlp=NS(down_proj=projection),
        )
    ]
    ownership.configure(admission.model, admission.config, True)
    assert ownership.validate_model(admission.model) == ("gdn",)
    context = NS(
        cudagraph_runtime_mode=NS(name="NONE"),
        ubatch_slices=None,
        attn_metadata={"gdn": metadata()},
    )
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    hidden = NS(
        shape=(8192, 4096), is_cuda=True, dtype=torch.bfloat16, device=comm.device
    )
    owner = ownership.maybe_create(admission.model, hidden, NS(shape=(8192,)))
    assert owner is not None and owner.comm is comm and owner.rank == rank
    values = torch.arange(8192).reshape(8192, 1)
    assert torch.equal(
        owner.local_view(values), values[rank * 2048 : (rank + 1) * 2048]
    )


def test_rank_flag_disagreement_rejects_disabled_rank(monkeypatch, admission):
    def vote(result, value, **kwargs):
        result[:] = [value, (True, None), value, value]

    monkeypatch.setattr(torch.distributed, "all_gather_object", vote)
    with pytest.raises(RuntimeError, match="All TP ranks"):
        ownership.configure(admission.model, admission.config, False)


@pytest.mark.parametrize(
    "change",
    [
        {"num_prefills": 0},
        {"num_prefill_tokens": 8191},
        {"num_decodes": 1},
        {"num_decode_tokens": 1},
        {"num_spec_decodes": 1},
        {"num_spec_decode_tokens": 4},
        {"num_prefills": torch.tensor(1)},
    ],
)
def test_only_pure_host_prefill_counts_admit_ownership(change):
    assert ownership.pure_prefill_metadata({"gdn": metadata()}, ("gdn",), 8192)
    assert not ownership.pure_prefill_metadata(
        {"gdn": metadata(**change)}, ("gdn",), 8192
    )
    assert not ownership.pure_prefill_metadata({}, ("gdn",), 8192)


def test_peer_ineligible_metadata_falls_back_before_partial_outputs(
    monkeypatch, admission
):
    ownership.configure(admission.model, admission.config, True)
    context = NS(
        cudagraph_runtime_mode=NS(name="NONE"),
        ubatch_slices=None,
        attn_metadata={"gdn": metadata()},
    )
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    monkeypatch.setattr(ownership, "validate_model", lambda model: ("gdn",))

    def vote(result, value, **kwargs):
        result[:] = [value, (False, None), value, value]

    monkeypatch.setattr(torch.distributed, "all_gather_object", vote)
    hidden = NS(
        shape=(8192, 4096),
        is_cuda=True,
        dtype=torch.bfloat16,
        device=torch.device("cuda", 0),
    )
    assert ownership.maybe_create(admission.model, hidden, NS(shape=(8192,))) is None
    context.cudagraph_runtime_mode.name = "FULL"
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda *a, **kw: pytest.fail("Graph decode must not enter ownership admission"),
    )
    assert ownership.maybe_create(admission.model, hidden, NS(shape=(8192,))) is None


def test_row_parallel_deferral_preserves_prefetch_hook_and_bias_tuple(monkeypatch):
    calls = []

    def reduce(value):
        calls.append("reduce")
        return value * 4

    monkeypatch.setattr(
        linear,
        "tensor_model_parallel_all_reduce",
        reduce,
    )
    obj = NS(
        input_is_parallel=True,
        tp_rank=0,
        tp_size=4,
        skip_bias_add=False,
        bias=None,
        quant_method=NS(apply=lambda self, x, bias: x * 2),
        reduce_results=True,
        return_bias=True,
        _l2_prefetch_pre_reduce_hook=lambda n: calls.append("prefetch"),
    )
    x = torch.arange(8).reshape(2, 4)
    baseline, _ = linear.RowParallelLinear.forward(obj, x)
    partial, bias = linear.RowParallelLinear.forward(obj, x, defer_tp_reduction=True)
    assert torch.equal(baseline, x * 8) and torch.equal(partial, x * 2)
    assert bias is None and obj.reduce_results
    assert calls == ["prefetch", "reduce", "prefetch"]


def test_moe_combines_shared_and_routed_before_deferred_reduction(monkeypatch):
    calls = []

    def reduce(value):
        calls.append("reduce")
        return value * 4

    monkeypatch.setattr(
        moe_runner,
        "tensor_model_parallel_all_reduce",
        reduce,
    )
    config = NS(
        tp_size=4,
        dp_size=1,
        ep_size=1,
        pcp_size=1,
        is_sequence_parallel=False,
        skip_final_all_reduce=False,
        hidden_dim_unpadded=0,
        moe_parallel_config=NS(use_all2all_kernels=False),
    )
    obj = NS(
        moe_config=config,
        _fused_output_is_reduced=False,
        routed_input_transform=None,
        routed_output_transform=None,
        router=NS(),
        _quant_method=NS(has_unpadded_output=False),
    )
    obj.apply_routed_input_transform = lambda x: (x, x)
    obj._maybe_pad_hidden_states = lambda shared, x: (x, None, 3)
    obj._forward_entry = lambda *args: (
        torch.full((2, 4), 2.0),
        torch.full((2, 4), 3.0),
    )
    obj._encode_layer_name = lambda: "test"
    obj._maybe_apply_routed_scale_to_output = lambda shared, routed: (shared, routed)
    obj.apply_routed_output_transform = lambda x: x
    obj._maybe_add_zero_expert_output = lambda x: x
    obj._l2_prefetch_pre_reduce_hook = lambda n: calls.append("prefetch")
    for name in (
        "_maybe_reduce_routed_output_before_transform",
        "_maybe_reduce_shared_expert_output",
        "_maybe_reduce_final_output",
    ):
        setattr(obj, name, MethodType(getattr(moe_runner.MoERunner, name), obj))
    x = torch.ones(2, 4)
    baseline = moe_runner.MoERunner.forward(obj, x, x)
    partial = moe_runner.MoERunner.forward(obj, x, x, defer_tp_reduction=True)
    assert baseline.shape == partial.shape == (2, 3)
    assert torch.all(baseline == 20) and torch.all(partial == 5)
    assert calls == ["prefetch", "reduce", "prefetch"]
    obj._fused_output_is_reduced = True
    obj._forward_entry = lambda *a: pytest.fail(
        "Unsupported MoE must fail before compute"
    )
    with pytest.raises(RuntimeError, match="unreduced"):
        moe_runner.MoERunner.forward(obj, x, x, defer_tp_reduction=True)


class CPUFabric:
    def __init__(self):
        self.barrier = Barrier(4, timeout=15)
        self.values = [None] * 4
        self.calls: list[list[str]] = [[] for _ in range(4)]

    def exchange(self, rank, kind, value):
        self.calls[rank].append(kind)
        self.values[rank] = value.clone()
        self.barrier.wait()
        result = (
            torch.cat(self.values) if kind == "ag" else torch.stack(self.values).sum(0)
        )
        if kind == "rs":
            result = result.chunk(4)[rank].clone()
        self.barrier.wait()
        return result


class CPUOwner:
    def __init__(self, fabric, rank):
        self.fabric, self.rank = fabric, rank
        self.mhc_calls = []

    def local_view(self, value):
        return value.chunk(4)[self.rank]

    def record_mhc(self, stage, output):
        self.mhc_calls.append((stage, output.shape[0]))

    def reduce_scatter(self, value):
        return self.fabric.exchange(self.rank, "rs", value)

    def all_gather(self, value):
        return self.fabric.exchange(self.rank, "ag", value)

    def finish(self, layers, auxiliary_gathers):
        calls = self.fabric.calls[self.rank]
        assert calls.count("rs") == 2 * layers
        assert calls.count("ag") == 2 * layers + auxiliary_gathers
        assert self.mhc_calls.count(("first_pre", 8)) == 1
        assert self.mhc_calls.count(("attention_post_pre", 2)) == layers - 1
        assert self.mhc_calls.count(("ffn_post_pre", 2)) == layers
        assert self.mhc_calls.count(("final_post", 2)) == 1


@pytest.mark.parametrize("dcp", [1, 2, 4])
def test_model_preserves_full_final_auxiliary_and_owner_residual_boundaries(
    monkeypatch, dcp
):
    """Completed DCP attention must feed one TP partial into row ownership."""
    monkeypatch.setattr(
        glm, "get_pp_group", lambda: NS(is_first_rank=True, is_last_rank=True)
    )
    monkeypatch.setattr(
        glm, "maybe_create_mhc_prefill_ownership", lambda model, *args: model.owner
    )
    monkeypatch.setattr(glm._l2pf, "ENABLED", False)
    monkeypatch.setattr(glm._l2pf, "join_all", lambda: None)

    def run(sharded):
        fabric = CPUFabric()

        class Layer(NS):
            __call__ = glm.Glm5NextDecoderLayer.forward

        def rank_forward(rank):
            layers = []
            for index in range(3):
                layer = Layer()
                layer.mhc, layer.is_mtp_layer, layer.is_sequence_parallel = (
                    True,
                    False,
                    False,
                )
                layer.layer_idx, layer.num_hidden_layers, layer.n = index, 3, 4
                layer._mlp_is_moe = index > 0

                def pre(x, *args, **kwargs):
                    n, h = x.shape
                    return (
                        x[:, None].expand(n, 4, h).clone(),
                        torch.ones(n, 4),
                        torch.ones(n, 4, 4),
                        x * 2,
                    )

                def post_pre(x, residual, post, comb, *args, **kwargs):
                    assert x.shape[0] == residual.shape[0] == (2 if sharded else 8)
                    updated = residual * 0.5 + x[:, None]
                    return updated, post + 0.25, comb + 0.5, updated.mean(1)

                layer._b12x_mhc = NS(run_pre=pre)
                layer.hc_attn_fn_broadcast = torch.ones(1)
                for kind in ("attn", "ffn"):
                    for part in ("fn", "scale", "base"):
                        setattr(layer, f"hc_{kind}_{part}", torch.ones(1))
                layer.input_layernorm = NS(weight=torch.ones(4), variance_epsilon=1e-5)
                layer.post_attention_layernorm = layer.input_layernorm
                layer.hc_fused_post_pre = post_pre
                layer.hc_post = lambda x, r, p, c: r + x[:, None]

                def attention(hidden_states, positions, *, defer_tp_reduction=False):
                    assert hidden_states.shape == (8, 4) and positions.shape == (8,)
                    context_partials = [
                        hidden_states * ((rank + 1) / (16 * dcp)) for _ in range(dcp)
                    ]
                    partial = torch.stack(context_partials).sum(0)
                    assert torch.equal(partial, hidden_states * ((rank + 1) / 16))
                    return (
                        partial
                        if defer_tp_reduction
                        else fabric.exchange(rank, "ar", partial)
                    )

                def mlp(
                    x, already_sequence_parallel=False, *, defer_tp_reduction=False
                ):
                    assert x.shape == (8, 4) and not already_sequence_parallel
                    partial = x * ((rank + 1) / 32)
                    return (
                        partial
                        if defer_tp_reduction
                        else fabric.exchange(rank, "ar", partial)
                    )

                layer.self_attn, layer.mlp = attention, mlp
                layers.append(layer)
            model = NS(
                owner=CPUOwner(fabric, rank) if sharded else None,
                is_sequence_parallel=False,
                start_layer=0,
                _active_layers=layers,
                aux_hidden_state_layers=[0, 1, 3],
                dflash_capture=False,
                norm=lambda x: x * 0.5,
            )
            model._prepare_aux_hidden_state = MethodType(
                glm.Glm5NextModel._prepare_aux_hidden_state, model
            )
            x = torch.arange(32, dtype=torch.float64).reshape(8, 4) / 32
            return glm.Glm5NextModel.forward(
                model, None, torch.arange(8), None, inputs_embeds=x
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(rank_forward, range(4))), fabric.calls

    baseline, baseline_calls = run(False)
    sharded, sharded_calls = run(True)
    for rank in range(4):
        assert baseline_calls[rank] == ["ar"] * 6
        assert sharded_calls[rank][0] == "rs"
        assert torch.equal(sharded[rank][0], baseline[rank][0])
        for actual, expected in zip(sharded[rank][1], baseline[rank][1]):
            assert actual.shape[0] == 8 and torch.equal(actual, expected)


def test_mtp_rejects_owner_before_attention_or_moe():
    with pytest.raises(RuntimeError, match="MTP"):
        glm.Glm5NextDecoderLayer.forward(
            NS(mhc=True, is_mtp_layer=True), None, None, mhc_prefill_ownership=object()
        )


def test_unavailable_cuda_is_reported_through_rank_admission(monkeypatch, admission):
    def unavailable():
        raise AssertionError("Torch was not compiled with CUDA")

    monkeypatch.setattr(torch.accelerator, "current_device_index", unavailable)
    with pytest.raises(RuntimeError, match="configuration admission"):
        ownership.configure(admission.model, admission.config, True)
    assert "not compiled" in admission.votes[0][1]


@pytest.mark.parametrize("change", ["batch", "capture", "dtype", "hidden"])
def test_unsupported_model_geometry_is_rejected(admission, change):
    if change == "batch":
        admission.config.scheduler_config.max_num_batched_tokens = 4096
    elif change == "capture":
        admission.config.compilation_config.cudagraph_capture_sizes = [8192]
    elif change == "dtype":
        admission.config.model_config.dtype = torch.float16
    else:
        admission.model.config.hidden_size = 2048
    with pytest.raises(RuntimeError, match="configuration admission"):
        ownership.configure(admission.model, admission.config, True)


def test_collective_outputs_have_owned_lifetimes_and_explicit_streams(monkeypatch):
    """Exercise the actual ownership helper with CPU allocation/stream doubles."""
    stream = object()
    calls = []
    monkeypatch.setattr(torch_utils, "current_stream", lambda: stream)

    class Tensor:
        is_cuda = True
        dtype = torch.bfloat16
        device = torch.device("cuda", 0)

        def __init__(self, shape):
            self.shape = shape
            self.streams = []

        def is_contiguous(self):
            return True

        def new_empty(self, shape):
            return Tensor(shape)

        def record_stream(self, value):
            self.streams.append(value)

    comm = NS(
        available=True,
        disabled=False,
        device=Tensor.device,
        reduce_scatter=lambda out, source, **kw: calls.append(("rs", out, source, kw)),
        all_gather=lambda out, source, **kw: calls.append(("ag", out, source, kw)),
    )
    owner = ownership.PrefillOwnership(comm, 0)
    partial = Tensor((8192, 4096))
    shard = owner.reduce_scatter(partial)
    full = owner.all_gather(shard)
    other_full = owner.all_gather(shard)
    assert shard.shape == (2048, 4096) and full.shape == (8192, 4096)
    assert full is not other_full
    assert calls[0] == ("rs", shard, partial, {"stream": stream})
    assert all(call[3] == {"stream": stream} for call in calls)
    assert partial.streams == full.streams == [stream]
    assert owner.rs_count == 1 and owner.ag_count == 2
    comm.disabled = True
    with pytest.raises(RuntimeError, match="unavailable"):
        owner.reduce_scatter(partial)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "prompt,expected_chunks",
    [
        (8192, [4096, 2048, 2048]),
        (16384, [8192, 4096, 2048, 2048]),
        (32768, [8192] * 3 + [4096, 2048, 2048]),
        (65536, [8192] * 7 + [4096, 2048, 2048]),
    ],
)
def test_unmodified_scheduler_produces_independent_mhc_chunks(
    monkeypatch, admission, prompt, expected_chunks
):
    """Stock one-checkpoint scheduling can admit mHC without coalescing."""
    from tests.v1.core.utils import create_requests
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
    )

    cache_config = KVCacheConfig(
        num_blocks=1000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=512, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            ),
            KVCacheGroupSpec(
                ["recurrent"],
                MambaSpec(
                    block_size=512,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=3,
                    num_prefill_checkpoint_blocks=1,
                ),
            ),
        ],
    )
    cache = KVCacheManager(
        cache_config,
        max_model_len=131072,
        scheduler_block_size=2048,
        hash_block_size=512,
        enable_caching=True,
        use_eagle=True,
    )
    (request,) = create_requests(1, num_tokens=prompt, block_size=512)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._kda_coalescing_enabled = False
    scheduler.cache_config = NS(block_size=512, prefix_cache_retention_interval=0)
    scheduler.kv_cache_manager = cache
    scheduler.max_num_scheduled_tokens = 8192
    scheduler.scheduler_config = NS(long_prefill_token_threshold=0)
    scheduler.mamba_has_prefill_checkpoint_blocks = True
    scheduler.mamba_partial_cache_hit = False
    scheduler.hash_block_size = 512
    scheduler.drop_last_prefix_cache_block = True
    scheduler.use_eagle = True
    ownership.configure(admission.model, admission.config, True)
    context = NS(
        cudagraph_runtime_mode=NS(name="NONE"), ubatch_slices=None, attn_metadata={}
    )
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    monkeypatch.setattr(ownership, "validate_model", lambda model: ("recurrent",))
    chunks, owned = [], []
    while request.num_computed_tokens < prompt:
        size = scheduler._mamba_block_aligned_split(
            request, min(8192, prompt - request.num_computed_tokens)
        )
        assert size > 0
        assert cache.allocate_slots(request, size, num_lookahead_tokens=3) is not None
        chunks.append(size)
        context.attn_metadata = {"recurrent": metadata(num_prefill_tokens=size)}
        hidden = NS(
            shape=(size, 4096),
            is_cuda=True,
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
        )
        owner = ownership.maybe_create(admission.model, hidden, NS(shape=(size,)))
        owned.append(owner is not None)
        request.num_computed_tokens += size
    assert chunks == expected_chunks
    assert owned == [size == 8192 for size in expected_chunks]


@pytest.mark.parametrize(
    "change,reason",
    [
        ("disabled", "disabled"),
        ("hidden_shape", "hidden_shape"),
        ("capture", "stream_capture"),
        ("no_context", "missing_forward_context"),
        ("graph", "execution_context"),
        ("ubatch", "execution_context"),
        ("positions", "local_ineligible"),
        ("metadata", "local_ineligible"),
        ("eligible", "admitted"),
    ],
)
def test_diagnostics_report_actual_admission_branches(
    monkeypatch, admission, change, reason
):
    monkeypatch.setattr(envs, "VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS", True)
    ownership.configure(admission.model, admission.config, True)
    context = NS(
        cudagraph_runtime_mode=NS(name="NONE"),
        ubatch_slices=None,
        attn_metadata={"gdn": metadata()},
        is_dummy_run=False,
    )
    monkeypatch.setattr(
        forward_context, "is_forward_context_available", lambda: change != "no_context"
    )
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    monkeypatch.setattr(ownership, "validate_model", lambda model: ("gdn",))
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: change == "capture"
    )
    hidden = NS(
        shape=(8192, 4096),
        is_cuda=True,
        dtype=torch.bfloat16,
        device=admission.group.device_communicator.pynccl_comm.device,
    )
    positions = NS(shape=(8192,))
    if change == "disabled":
        admission.model._mhc_prefill_enabled = False
    elif change == "hidden_shape":
        hidden.shape = (16, 4096)
    elif change == "graph":
        context.cudagraph_runtime_mode.name = "PIECEWISE"
    elif change == "ubatch":
        context.ubatch_slices = object()
    elif change == "positions":
        positions.shape = (16,)
    elif change == "metadata":
        context.attn_metadata = {"gdn": metadata(num_spec_decodes=1)}
    result = ownership.maybe_create(admission.model, hidden, positions)
    assert (result is not None) == (change == "eligible")
    diagnostic = admission.model._mhc_prefill_diagnostics
    kind = "no_context" if change == "no_context" else "request"
    assert diagnostic.decisions[f"{kind}:{reason}"] == 1


def test_suppressed_logging_retains_request_diagnostic_and_enqueue_witness(
    monkeypatch, caplog
):
    monkeypatch.setattr(ownership._LOG, "handlers", [caplog.handler])
    context = NS(is_dummy_run=True)
    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    diagnostic = ownership.PrefillDiagnostics(2)
    with monkeypatch.context() as suppressed:
        suppressed.setattr(ownership._LOG, "isEnabledFor", lambda level: False)
        for _ in range(12):
            diagnostic.decision("admitted")
        assert diagnostic.emitted == set()
    with caplog.at_level(logging.WARNING, logger=ownership._LOG.name):
        diagnostic.decision("admitted")
        context.is_dummy_run = False
        diagnostic.decision("admitted")
        owner = ownership.PrefillOwnership(
            None,
            2,
            rs_count=90,
            ag_count=90,
            diagnostics=diagnostic,
            diagnostic_kind="request",
            diagnostic_forward=14,
        )
        owner.record_mhc("first_pre", NS(shape=(8192, 4096)))
        for _ in range(44):
            owner.record_mhc("attention_post_pre", NS(shape=(2048, 4096)))
        for _ in range(45):
            owner.record_mhc("ffn_post_pre", NS(shape=(2048, 4096)))
        owner.record_mhc("final_post", NS(shape=(2048, 4, 4096)))
        owner.finish(45, 0)
    assert diagnostic.decisions == {"dummy:admitted": 13, "request:admitted": 1}
    assert '"kind": "request"' in caplog.text
    assert '"kind": "dummy"' in caplog.text
    assert "GLM_MHC_ENQUEUE" in caplog.text
    assert '"attention_post_pre": {"2048": 44}' in caplog.text
    assert '"gpu_completion_verified": false' in caplog.text


def test_metadata_diagnostic_never_reads_non_host_counts():
    class DeviceCount:
        def __repr__(self):
            pytest.fail("Diagnostics must not represent device-backed values")

        def __int__(self):
            pytest.fail("Diagnostics must not read device-backed values")

    result = ownership.metadata_diagnostics(
        {"gdn": metadata(num_prefills=DeviceCount())}, ("gdn", "absent")
    )
    assert result["metadata_counts"]["gdn"]["num_prefills"] == {
        "type": "DeviceCount",
        "value": None,
    }
    assert result["missing_names"] == ["absent"]


def test_forward_context_retains_host_dummy_marker():
    config = NS(
        compilation_config=NS(fast_moe_cold_start=False, static_forward_context={})
    )
    assert not forward_context.create_forward_context(None, config).is_dummy_run
    assert forward_context.create_forward_context(
        None, config, is_dummy_run=True
    ).is_dummy_run
