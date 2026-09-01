# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3's fused chunked-context prefill must match the generic MLA impl.

The layer owns its own context loop so it can fuse the per-chunk K/V pack and
skip re-quantizing an already-quantized query. That is only safe if it feeds the
prefill backend exactly what ``MLACommonBaseImpl._compute_prefill_context``
would, chunk for chunk.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBaseImpl,
    MLACommonPrefillMetadata,
    build_mla_chunked_context_metadata,
)
from vllm.models.kimi_k3.nvidia.mla import (
    KimiK3PrefillProjectionWorkspace,
    MultiHeadLatentAttention,
)
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
        from vllm.model_executor.layers.linear import UnquantizedLinearMethod

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
        self.quant_method = UnquantizedLinearMethod()
        self.bias = None
        self.gather_output = False
        self.forward_calls = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.forward_calls += 1
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

    def __init__(
        self,
        kv_b_proj,
        kv_cache,
        kv_cache_dtype,
        k_scale,
        prefill_projection_workspace=None,
    ) -> None:
        self.kv_b_proj = kv_b_proj
        self.kv_cache = kv_cache
        self.kv_cache_dtype = kv_cache_dtype
        self._k_scale = k_scale
        self.kv_lora_rank = _KV_LORA_RANK
        self.num_local_heads = _NUM_HEADS
        self.dcp_world_size = 1
        self.qk_nope_head_dim = _QK_NOPE
        self.v_head_dim = _V_HEAD_DIM
        self.prefill_projection_workspace = prefill_projection_workspace


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
) -> MLACommonPrefillMetadata:
    query_start_loc_cpu = torch.zeros(len(_QUERY_LENS) + 1, dtype=torch.int32)
    query_start_loc_cpu[1:] = torch.tensor(_QUERY_LENS, dtype=torch.int32).cumsum(0)
    workspace = torch.empty(
        (_WORKSPACE_TOKENS, _ENTRY), dtype=workspace_dtype, device=device
    )
    chunked_context = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(_CONTEXT_LENS, dtype=torch.int32),
        prefill_query_start_loc_cpu=query_start_loc_cpu,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=_WORKSPACE_TOKENS,
        block_size=_BLOCK_SIZE,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=1,
        dcp_local_block_size=1,
        dcp_virtual_block_size=1,
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

    projection_workspace = None
    if not kv_b_proj_quantized:
        projection_workspace = KimiK3PrefillProjectionWorkspace(
            num_ubatches=1, min_tokens=0
        )
        projection_workspace.reserve(
            max_tokens=_WORKSPACE_TOKENS,
            output_size=_NUM_HEADS * (_QK_NOPE + _V_HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
    layer = _FusedLayer(
        kv_b_proj,
        kv_cache,
        kv_cache_dtype,
        k_scale,
        prefill_projection_workspace=projection_workspace,
    )
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
    expected_projection_calls = len(prefill_ref.chunked_context.chunks)
    if kv_b_proj_quantized:
        expected_projection_calls *= 2
    assert kv_b_proj.forward_calls == expected_projection_calls

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


def test_prefill_projection_workspace_reuses_reserved_storage() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = KimiK3PrefillProjectionWorkspace(num_ubatches=1, min_tokens=4)
    workspace.reserve(8, 16, torch.bfloat16, device)

    assert workspace.get(3, 16, torch.bfloat16, device) is None
    short = workspace.get(4, 16, torch.bfloat16, device)
    full = workspace.get(8, 16, torch.bfloat16, device)
    assert short is not None and full is not None
    assert short.data_ptr() == full.data_ptr()
    assert workspace.nbytes == 8 * 16 * torch.bfloat16.itemsize

    with pytest.raises(ValueError, match="needs 9 rows"):
        workspace.get(9, 16, torch.bfloat16, device)


@torch.inference_mode()
def test_fused_context_consumes_direct_dcp_final_layout(monkeypatch) -> None:
    """The K3 loop must consume compact DCP planes without rank-major reorg."""
    device = torch.device("cuda")
    workspace = torch.empty((2, _ENTRY), device=device, dtype=torch.bfloat16)
    kv_c = torch.randn((3, 1, _KV_LORA_RANK), device=device, dtype=torch.bfloat16)
    k_pe = torch.randn((3, 1, _QK_ROPE), device=device, dtype=torch.bfloat16)
    dcp_kv_gather = MagicMock(use_direct_kv_gather=True)
    dcp_kv_gather.direct_kv_gather.return_value = (kv_c, k_pe)
    chunk = SimpleNamespace(
        request_slice=slice(0, 1),
        padded_local_cu_seq_lens=torch.tensor([0, 2], device=device),
        padded_local_token_to_seq=torch.zeros(2, dtype=torch.int32, device=device),
        final_layout_dst_rows=torch.tensor([0, 2], dtype=torch.int32, device=device),
        num_local_context_tokens=2,
        num_context_tokens=3,
        num_requests=1,
        starts=torch.tensor([0], dtype=torch.int32, device=device),
        index=5,
    )
    prefill = SimpleNamespace(
        block_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        chunked_context=SimpleNamespace(
            workspace=workspace,
            dcp_manager=dcp_kv_gather,
        ),
    )
    gather_cache = MagicMock()
    monkeypatch.setattr(
        "vllm.models.kimi_k3.nvidia.mla.ops.cp_gather_cache", gather_cache
    )
    layer = _FusedLayer(
        _KVBProj(device, weight_dtype=torch.bfloat16),
        torch.empty((1, _BLOCK_SIZE, _ENTRY), device=device, dtype=torch.bfloat16),
        "auto",
        torch.ones(1, dtype=torch.float32, device=device),
    )
    layer.dcp_world_size = 2

    actual_kv_c, actual_k_pe = layer._gather_context_latent(
        chunk,
        layer.kv_cache,
        prefill,
        fp8_prefill=False,
    )

    assert actual_kv_c is kv_c
    assert actual_k_pe is k_pe
    gather_cache.assert_called_once()
    dcp_kv_gather.direct_kv_gather.assert_called_once()
    local_rows, dst_rows, output_tokens, slot = (
        dcp_kv_gather.direct_kv_gather.call_args.args
    )
    assert local_rows.data_ptr() == workspace.data_ptr()
    assert dst_rows is chunk.final_layout_dst_rows
    assert output_tokens == 3
    assert slot == 1
