# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3's fused chunked-context prefill must match the generic MLA impl.

The layer owns its own context loop so it can fuse the per-chunk K/V pack and
skip re-quantizing an already-quantized query. That is only safe if it feeds the
prefill backend exactly what ``MLACommonBaseImpl._compute_prefill_context``
would, chunk for chunk.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBaseImpl,
    MLACommonPrefillMetadata,
    build_mla_chunked_context_metadata,
)
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Kimi-K3 fused MLA requires CUDA"
)

_KV_LORA_RANK = 512
_QK_NOPE = 128
_QK_ROPE = 64
_V_HEAD_DIM = 128
_ENTRY = _KV_LORA_RANK + _QK_ROPE
_NUM_HEADS = 2
_BLOCK_SIZE = 16
_WORKSPACE_TOKENS = 128
# Splits one long request across chunks, packs short ones together, and leaves
# the last request without any context.
_CONTEXT_LENS = [200, 48, 32, 0]
_QUERY_LENS = [8, 4, 6, 5]


def test_prepared_prefill_preserves_kimi_decode_preparation():
    """Kimi owns both backends; preparing only decode leaves prefill unusable."""
    layer = MultiHeadLatentAttention.__new__(MultiHeadLatentAttention)
    torch.nn.Module.__init__(layer)
    workload = object()
    calls = []

    def units(owner, request, kind):
        assert owner is layer and request is workload
        calls.append(kind)
        return (kind,)

    layer.impl = SimpleNamespace(
        b12x_preparation_provider=SimpleNamespace(
            get_b12x_preparation_units=lambda owner, request: units(
                owner, request, "decode"
            )
        )
    )
    layer.prefill_backend = SimpleNamespace(
        get_b12x_preparation_units=lambda owner, request: units(
            owner, request, "prefill"
        )
    )
    assert layer.get_b12x_preparation_units(layer, workload) == ("decode", "prefill")
    assert calls == ["decode", "prefill"]
    with pytest.raises(ValueError, match="owner mismatch"):
        layer.get_b12x_preparation_units(object(), workload)


class _RecordingPrefillBackend:
    """Records what each chunk is asked to attend over.

    ``honors_out`` mimics the two backend families: one that writes into a
    caller-provided ``out`` (trtllm_ragged, flashinfer, tokenspeed) and one that
    always returns its own buffer (flash_attn with a padded V, aiter).
    """

    def __init__(self, honors_out: bool = False) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.out_destinations: list[torch.Tensor | None] = []
        self._honors_out = honors_out

    @staticmethod
    def get_name() -> str:
        return "recording"

    def supports_out(self) -> bool:
        return self._honors_out

    def run_prefill_context_chunk(self, *, chunk, q, k, v, out=None):
        self.calls.append((q.float().clone(), k.float().clone(), v.float().clone()))
        self.out_destinations.append(out)
        assert out is None or self._honors_out
        # Fold K/V into the partial so any packing difference shows up in the
        # merged context output, not just in the recorded calls.
        digest = (k.float().mean() + v.float().mean()).item()
        num_q = q.shape[0]
        if out is None:
            out = torch.empty(
                (num_q, _NUM_HEADS, _V_HEAD_DIM), device=q.device, dtype=torch.bfloat16
            )
        else:
            assert out.shape == (num_q, _NUM_HEADS, _V_HEAD_DIM)
        out.fill_(digest)
        lse = torch.full(
            (_NUM_HEADS, num_q),
            1.0 + chunk.index,
            device=q.device,
            dtype=torch.float32,
        )
        return out, lse


class _KVBProj(torch.nn.Module):
    """Stand-in for the layer's ``kv_b_proj`` (returns an (out, bias) tuple).

    Enforces the same input contract as the real linear methods, because that is
    what decides whether the gathered latent needs a cast:

    * an fp8 weight consumes the fp8 latent directly and dequantizes internally,
      so it takes fp8 or bf16;
    * a bf16 weight -- what a stock K3 checkpoint carries -- is a plain
      ``F.linear`` and rejects anything but bf16, exactly as torch does.

    Either weight dtype produces a bf16 output.
    """

    def __init__(self, device: torch.device, weight_dtype: torch.dtype) -> None:
        super().__init__()
        weight = (
            torch.randn(
                _NUM_HEADS * (_QK_NOPE + _V_HEAD_DIM),
                _KV_LORA_RANK,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.05
        )
        self.register_buffer("weight", weight.to(weight_dtype))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self.weight.dtype == torch.bfloat16 and x.dtype != torch.bfloat16:
            raise RuntimeError(
                "a bfloat16 kv_b_proj cannot consume the gathered latent as "
                f"{x.dtype}; it must be cast first"
            )
        return torch.nn.functional.linear(
            x.to(torch.bfloat16), self.weight.to(torch.bfloat16)
        ), None


class _FusedLayer:
    """Only the attributes K3's context loop reads."""

    _compute_prefill_context = MultiHeadLatentAttention._compute_prefill_context
    _gather_context_latent = MultiHeadLatentAttention._gather_context_latent
    _attn_read_kv_cache = MultiHeadLatentAttention._attn_read_kv_cache
    _forward_prefill_fused = MultiHeadLatentAttention._forward_prefill_fused

    def __init__(self, kv_b_proj, kv_cache, kv_cache_dtype, k_scale) -> None:
        self.kv_b_proj = kv_b_proj
        self.kv_cache = kv_cache
        self.kv_cache_dtype = kv_cache_dtype
        self._k_scale = k_scale
        self.kv_lora_rank = _KV_LORA_RANK
        self.num_local_heads = _NUM_HEADS
        self.qk_nope_head_dim = _QK_NOPE
        self.v_head_dim = _V_HEAD_DIM


class _ReferenceImpl:
    """Only the attributes the generic context loop reads."""

    _compute_prefill_context = MLACommonBaseImpl._compute_prefill_context
    _concat_k_nope_k_pe = MLACommonBaseImpl._concat_k_nope_k_pe
    _use_flashinfer_concat_mla_k = False

    def __init__(self, kv_b_proj, kv_cache_dtype) -> None:
        self.kv_b_proj = kv_b_proj
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_lora_rank = _KV_LORA_RANK
        self.num_heads = _NUM_HEADS
        self.qk_nope_head_dim = _QK_NOPE
        self.qk_rope_head_dim = _QK_ROPE
        self.v_head_dim = _V_HEAD_DIM


def _build_prefill_metadata(
    device: torch.device,
    workspace_dtype: torch.dtype,
    q_data_type: torch.dtype,
    backend: _RecordingPrefillBackend,
    dcp_world_size: int = 1,
) -> MLACommonPrefillMetadata:
    query_start_loc_cpu = torch.zeros(len(_QUERY_LENS) + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(_QUERY_LENS, dtype=torch.int32).cumsum(0)
    workspace = torch.empty(
        (
            _WORKSPACE_TOKENS
            + (_WORKSPACE_TOKENS // dcp_world_size if dcp_world_size > 1 else 0),
            _ENTRY,
        ),
        dtype=workspace_dtype,
        device=device,
    )
    chunked_context = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(_CONTEXT_LENS, dtype=torch.int32),
        prefill_query_start_loc_cpu=query_start_loc_cpu,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=_WORKSPACE_TOKENS,
        block_size=_BLOCK_SIZE,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=dcp_world_size,
        dcp_local_block_size=1,
        dcp_virtual_block_size=dcp_world_size,
    )
    assert chunked_context is not None
    assert len(chunked_context.chunks) > 1, "the batch must exercise accumulation"

    max_blocks = (max(_CONTEXT_LENS) + max(_QUERY_LENS)) // _BLOCK_SIZE + 1
    num_blocks = max_blocks * len(_CONTEXT_LENS)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        len(_CONTEXT_LENS), max_blocks
    )
    return MLACommonPrefillMetadata(
        block_table=block_table,
        query_start_loc=query_start_loc_cpu.to(device),
        max_query_len=max(_QUERY_LENS),
        chunked_context=chunked_context,
        q_data_type=q_data_type,
        output_dtype=torch.bfloat16,
        prefill_backend=backend,
    ), num_blocks


@pytest.mark.parametrize("world", [2, 8])
@torch.inference_mode()
def test_dcp_fp8_transport_preserves_projected_context_and_accumulation(world):
    """Exercise raw cache extraction, upconversion and the actual context loop.

    Rank inputs are replicated by the transport test double. The native cache
    gather, KV projection, packed-context layout and partial merge are real.
    """
    import functools

    from vllm.v1.attention.ops.dcp_prefetch import DCPContextPrefetch

    class CopyCommunicator:
        disabled = False

        def all_gather(self, output, source, stream):
            for rank in range(world):
                output[rank * source.shape[0] : (rank + 1) * source.shape[0]].copy_(
                    source
                )

    torch.manual_seed(123)
    device = torch.device("cuda")
    comm = CopyCommunicator()
    manager = SimpleNamespace(
        _kv_gather=functools.partial(torch.distributed.all_gather_into_tensor),
        group=SimpleNamespace(
            world_size=world,
            device_communicator=SimpleNamespace(pynccl_comm=comm),
        ),
        kv_gather=lambda output, source: comm.all_gather(
            output, source, torch.cuda.current_stream()
        ),
    )
    projection = _KVBProj(device, torch.bfloat16)
    impl = _ReferenceImpl(projection, "fp8")
    q = torch.randn(
        (sum(_QUERY_LENS), _NUM_HEADS, _QK_NOPE + _QK_ROPE),
        device=device,
        dtype=torch.bfloat16,
    )
    scale = torch.tensor([0.1], device=device, dtype=torch.float32)
    outputs, calls = [], []
    cache = None
    for transport in (None, False, True):
        backend = _RecordingPrefillBackend()
        prefill, blocks = _build_prefill_metadata(
            device, torch.bfloat16, torch.bfloat16, backend, world
        )
        if cache is None:
            cache = torch.randn((blocks, _BLOCK_SIZE, _ENTRY), device=device).to(
                torch.float8_e4m3fn
            )
        context = prefill.chunked_context
        context.dcp_manager = manager
        if transport is not None:
            context.dcp_prefetch = DCPContextPrefetch(
                manager, context.workspace, fp8_transport=transport
            )
        output = MLACommonBaseImpl._context_parallel_compute_prefill_context(
            impl, q, cache, SimpleNamespace(prefill=prefill), scale, world
        )
        torch.accelerator.synchronize()
        outputs.append(tuple(t.clone() for t in output))
        calls.append(backend.calls)
    for output, captured in zip(outputs[1:], calls[1:]):
        for actual, expected in zip(output, outputs[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual_chunk, expected_chunk in zip(captured, calls[0], strict=True):
            for actual, expected in zip(actual_chunk, expected_chunk):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("honors_out", [False, True], ids=["copy_out", "writes_out"])
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "kv_b_proj_quantized", [True, False], ids=["fp8_kv_b_proj", "bf16_kv_b_proj"]
)
@torch.inference_mode()
def test_fused_context_matches_generic_impl(
    kv_b_proj_quantized: bool, kv_cache_dtype: str, honors_out: bool
) -> None:
    """Cache dtype and ``kv_b_proj`` dtype vary independently.

    A stock K3 checkpoint pairs a bf16 ``kv_b_proj`` with an fp8 cache, so the
    fused loop cannot assume the gathered latent is already in the dtype
    ``kv_b_proj`` accepts.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    fp8 = current_platform.fp8_dtype()
    quantized = kv_cache_dtype == "fp8"
    # An fp8 cache is read by an fp8 query, so the workspace keeps the fp8
    # layout; a bf16 cache dequantizes into a bf16 workspace.
    q_data_type = fp8 if quantized else torch.bfloat16
    workspace_dtype = q_data_type

    kv_b_proj = _KVBProj(
        device, weight_dtype=fp8 if kv_b_proj_quantized else torch.bfloat16
    )
    k_scale = torch.ones(1, dtype=torch.float32, device=device)

    backend_fused = _RecordingPrefillBackend(honors_out=honors_out)
    # The reference impl never passes `out`, so it always allocates its own.
    backend_ref = _RecordingPrefillBackend()
    prefill_fused, num_blocks = _build_prefill_metadata(
        device, workspace_dtype, q_data_type, backend_fused
    )
    prefill_ref, _ = _build_prefill_metadata(
        device, workspace_dtype, q_data_type, backend_ref
    )

    cache = torch.randn(
        (num_blocks, _BLOCK_SIZE, _ENTRY), device=device, dtype=torch.bfloat16
    )
    kv_cache = cache.to(fp8) if quantized else cache
    q = (
        torch.randn(
            (sum(_QUERY_LENS), _NUM_HEADS, _QK_NOPE + _QK_ROPE),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.2
    ).to(q_data_type)

    layer = _FusedLayer(kv_b_proj, kv_cache, kv_cache_dtype, k_scale)
    fused_out, fused_lse = layer._compute_prefill_context(
        q, SimpleNamespace(prefill=prefill_fused)
    )

    impl = _ReferenceImpl(kv_b_proj, kv_cache_dtype)
    ref_out, ref_lse = impl._compute_prefill_context(
        q, kv_cache, SimpleNamespace(prefill=prefill_ref), k_scale
    )

    assert len(backend_fused.calls) == len(backend_ref.calls)
    for chunk_idx, (fused_call, ref_call) in enumerate(
        zip(backend_fused.calls, backend_ref.calls, strict=True)
    ):
        for name, fused_t, ref_t in zip(
            ("q", "k", "v"), fused_call, ref_call, strict=True
        ):
            torch.testing.assert_close(
                fused_t,
                ref_t,
                atol=0,
                rtol=0,
                msg=lambda m, n=name, i=chunk_idx: f"chunk {i} {n} differs: {m}",
            )
    torch.testing.assert_close(fused_out, ref_out, atol=0, rtol=0)
    torch.testing.assert_close(fused_lse, ref_lse, atol=0, rtol=0)

    # Every chunk but the continuation should have been written in place, i.e.
    # straight into the returned accumulator, with no intermediate copy.
    wrote_in_place = [
        out is not None and out.data_ptr() == fused_out[chunk.token_slice].data_ptr()
        for out, chunk in zip(
            backend_fused.out_destinations,
            prefill_fused.chunked_context.chunks,
            strict=True,
        )
    ]
    continuations = [c.is_continuation for c in prefill_fused.chunked_context.chunks]
    assert any(continuations), "the batch must exercise a continuation chunk"
    if honors_out:
        assert wrote_in_place == [not c for c in continuations]
    else:
        assert not any(wrote_in_place)


@torch.inference_mode()
def test_fused_context_rejects_an_unquantized_query() -> None:
    """The fp8 query is produced by the new-token epilogue, not re-cast here."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    fp8 = current_platform.fp8_dtype()
    prefill, num_blocks = _build_prefill_metadata(
        device, fp8, fp8, _RecordingPrefillBackend()
    )
    layer = _FusedLayer(
        _KVBProj(device, weight_dtype=fp8),
        torch.zeros((num_blocks, _BLOCK_SIZE, _ENTRY), device=device, dtype=fp8),
        "fp8",
        torch.ones(1, dtype=torch.float32, device=device),
    )
    q = torch.zeros(
        (sum(_QUERY_LENS), _NUM_HEADS, _QK_NOPE + _QK_ROPE),
        device=device,
        dtype=torch.bfloat16,
    )
    with pytest.raises(AssertionError, match="new-token epilogue"):
        layer._compute_prefill_context(q, SimpleNamespace(prefill=prefill))


@torch.inference_mode()
def test_bf16_nope_prefill_writes_fp8_cache_without_quantizing_attention():
    """Cache storage precision must not force lower-precision prefill Q/K/V."""
    torch.manual_seed(411)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache = torch.zeros(
        (2, _BLOCK_SIZE, _ENTRY), dtype=torch.float8_e4m3fn, device=device
    )
    layer = _FusedLayer(
        _KVBProj(device, dtype),
        cache,
        "fp8",
        torch.tensor([0.75], dtype=torch.float32, device=device),
    )
    latent = torch.randn(4, _KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(4, 1, _QK_ROPE, dtype=dtype, device=device)
    q = torch.randn(4, _NUM_HEADS, _QK_NOPE + _QK_ROPE, dtype=dtype, device=device)
    slots = torch.tensor([3, 0, -1, 19], dtype=torch.int64, device=device)
    projected = layer.kv_b_proj(latent)[0].view(4, _NUM_HEADS, -1)
    expected_k, expected_v = projected.split([_QK_NOPE, _V_HEAD_DIM], dim=-1)
    expected_k = torch.cat([expected_k, k_pe.expand(-1, _NUM_HEADS, -1)], -1)
    calls = []

    def prefill(*, q, k, v, return_softmax_lse, out):
        calls.append((q, k, v))
        assert not return_softmax_lse
        out.copy_(v)
        return out

    metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            q_data_type=dtype,
            chunked_context=None,
            prefill_backend=SimpleNamespace(
                supports_out=lambda: True, run_prefill_new_tokens=prefill
            ),
        )
    )
    output = torch.empty(4, _NUM_HEADS * _V_HEAD_DIM, dtype=dtype, device=device)
    layer._forward_prefill_fused(q, latent, k_pe, None, None, slots, metadata, output)
    actual_q, actual_k, actual_v = calls[0]
    assert actual_q is q
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(output.view_as(expected_v), expected_v, rtol=0, atol=0)
    expected_cache = torch.zeros_like(cache)
    payload = torch.cat([latent, k_pe.flatten(1)], -1).float() / layer._k_scale
    for row, slot in enumerate([3, 0, -1, 19]):
        if slot >= 0:
            expected_cache.view(-1, _ENTRY)[slot] = payload[row].to(cache.dtype)
    torch.testing.assert_close(cache.float(), expected_cache.float(), rtol=0, atol=0)
