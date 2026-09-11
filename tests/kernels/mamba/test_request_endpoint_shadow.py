# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-endpoint shadow snapshot and materialization kernels.

``precopy_mamba_align_fused_kernel`` with ``HAS_SHADOW`` copies every
request's committed state into its shadow pages before the boundary
migration, including requests that stay in their running block: the conv
window unshifted and the temporal states of the columns ``src_col + t`` for
``t`` up to the accepted-token bias. ``materialize_mamba_endpoint_kernel``
then writes a pool block from either the shadow pages (conv shifted by the
endpoint bias, temporal slot at that bias) or the request's own state
blocks, with byte-identical results to the V1 copy specs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.mamba_utils import (
    _TEMPORAL_TILES,
    ENDPOINT_COPY_META_FIXED,
    MambaSpecDecodeGPUContext,
    materialize_mamba_endpoint_kernel,
    precopy_mamba_align_fused_kernel,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="the shadow kernels need CUDA/Triton"
)

NUM_LAYERS = 3
CONV_WIDTH = 4  # conv_kernel - 1 + num_spec
CONV_DIM = 96
SSM_SHAPE = (4, 16, 16)
MAX_COLS = 8
MAX_REQS = 6
TEMPORAL_SLOTS = 4  # 1 + num_speculative_blocks


def _build_state(num_blocks, device, conv_state_dim_first):
    convs, ssms = [], []
    for _ in range(NUM_LAYERS):
        conv_shape = (
            (num_blocks, CONV_DIM, CONV_WIDTH)
            if conv_state_dim_first
            else (num_blocks, CONV_WIDTH, CONV_DIM)
        )
        convs.append(torch.randn(*conv_shape, dtype=torch.bfloat16, device=device))
        ssms.append(
            torch.randn(num_blocks, *SSM_SHAPE, dtype=torch.float32, device=device)
        )
    return convs, ssms


def _build_meta(convs, ssms, device, conv_state_dim_first):
    n = NUM_LAYERS * 2
    base = torch.zeros(n, dtype=torch.int64, device=device)
    blk_stride = torch.zeros(n, dtype=torch.int64, device=device)
    elem = torch.zeros(n, dtype=torch.int32, device=device)
    inner = torch.zeros(n, dtype=torch.int64, device=device)
    width = torch.zeros(n, dtype=torch.int32, device=device)
    group = torch.zeros(n, dtype=torch.int32, device=device)
    drc = torch.zeros(n, dtype=torch.int32, device=device)
    drs = torch.zeros(n, dtype=torch.int64, device=device)
    i = 0
    for layer in range(NUM_LAYERS):
        conv, ssm = convs[layer], ssms[layer]
        base[i] = conv.data_ptr()
        blk_stride[i] = conv.stride(0) * conv.element_size()
        elem[i] = conv.element_size()
        if conv_state_dim_first:
            width[i] = conv.size(2)
            inner[i] = 1
            drc[i] = conv.size(1)
            drs[i] = conv.stride(1) * conv.element_size()
        else:
            width[i] = conv.size(1)
            inner[i] = conv.stride(1)
        i += 1
        base[i] = ssm.data_ptr()
        blk_stride[i] = ssm.stride(0) * ssm.element_size()
        elem[i] = ssm.element_size()
        width[i] = 0
        inner[i] = ssm[0].numel()
        i += 1
    return base, blk_stride, elem, inner, width, group, drc, drs


def _build_shadow(convs, ssms, device):
    """Per-state shadow pages with the state tensor's block stride: one conv
    page per request slot, ``TEMPORAL_SLOTS`` temporal pages per request slot;
    viewed with the state's own shape for comparison."""
    buffers, views, addrs, strides = [], [], [], []
    for layer in range(NUM_LAYERS):
        for state, pages in (
            (convs[layer], MAX_REQS),
            (ssms[layer], MAX_REQS * TEMPORAL_SLOTS),
        ):
            stride = state.stride(0) * state.element_size()
            buf = torch.zeros(pages * stride, dtype=torch.uint8, device=device)
            buffers.append(buf)
            addrs.append(buf.data_ptr())
            strides.append(stride)
            views.append(buf.view(state.dtype).view(pages, *state.shape[1:]))
    return (
        buffers,
        views,
        torch.tensor(addrs, dtype=torch.int64, device=device),
        torch.tensor(strides, dtype=torch.int64, device=device),
    )


def _normalized(conv, ssm, bt, req, src_col, bias, conv_dim_first):
    """The committed state of a request per the V1 copy specs: the window of
    ``src_col`` shifted by ``bias`` and the temporal state of column
    ``src_col + bias``."""
    sblk = int(bt[req, src_col])
    tblk = int(bt[req, src_col + bias])
    conv_out = torch.zeros_like(conv[sblk])
    if conv_dim_first:
        conv_out[:, : CONV_WIDTH - bias] = conv[sblk][:, bias:]
    else:
        conv_out[: CONV_WIDTH - bias] = conv[sblk][bias:]
    return conv_out, ssm[tblk].clone()


def _leading(conv, bias, conv_dim_first):
    """The window entries a reader with the given bias consumes."""
    if conv_dim_first:
        return conv[..., : CONV_WIDTH - bias]
    return conv[: CONV_WIDTH - bias]


@pytest.mark.parametrize("conv_state_dim_first", [False, True])
@pytest.mark.parametrize("temporal_tiles", [1, _TEMPORAL_TILES])
def test_snapshot_then_materialize_matches_the_copy_specs(
    conv_state_dim_first, temporal_tiles
):
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_reqs = 5
    num_blocks = MAX_REQS * MAX_COLS + 4
    # Distinct physical blocks per request slot; the kernel reads the block
    # table in batch order (V2 layout), so it gets the rows gathered by
    # idx_mapping while the reference indexes the per-slot table.
    bt = torch.empty(MAX_REQS, MAX_COLS, dtype=torch.int32, device=device)
    for r in range(MAX_REQS):
        bt[r] = torch.arange(
            1 + r * MAX_COLS, 1 + (r + 1) * MAX_COLS, dtype=torch.int32, device=device
        )
    # Batch rows map to request slots out of order to exercise idx_mapping.
    idx_mapping = torch.tensor([3, 0, 4, 1, 2], dtype=torch.int32, device=device)
    bt_batch = bt[idx_mapping.long()].contiguous()
    # Per slot: (src_col, dst_col, bias). Slot 0 fresh; slot 1 stays in its
    # running block (snapshot only); the others cross a boundary.
    src_col = torch.tensor([-1, 2, 1, 3, 1, 0], dtype=torch.int32, device=device)
    dst_col = torch.tensor([0, 2, 0, 4, 5, 0], dtype=torch.int32, device=device)
    bias = torch.tensor([0, 2, 1, 0, 3, 0], dtype=torch.int32, device=device)

    convs, ssms = _build_state(num_blocks, device, conv_state_dim_first)
    conv_pre = [c.clone() for c in convs]
    ssm_pre = [s.clone() for s in ssms]
    base, blk_stride, elem, inner, width, group, drc, drs = _build_meta(
        convs, ssms, device, conv_state_dim_first
    )
    _, shadow_views, shadow_addrs, shadow_strides = _build_shadow(convs, ssms, device)
    bt_ptrs = torch.tensor([bt_batch.data_ptr()], dtype=torch.int64, device=device)

    grid = (num_reqs, NUM_LAYERS * 2, temporal_tiles)
    precopy_mamba_align_fused_kernel[grid](
        dst_col,
        src_col,
        bias,
        bt_ptrs,
        bt_batch.stride(0),
        base,
        blk_stride,
        elem,
        inner,
        width,
        group,
        drc,
        drs,
        idx_mapping,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first,
        HAS_IDX_MAPPING=True,
        TEMPORAL_TILES=temporal_tiles,
        shadow_base_addrs_ptr=shadow_addrs,
        shadow_temporal_slots=TEMPORAL_SLOTS,
        HAS_SHADOW=True,
        shadow_page_strides_ptr=shadow_strides,
    )
    torch.accelerator.synchronize()

    bt_cpu = bt.cpu()
    scheduled = set(idx_mapping.tolist())
    for slot in range(MAX_REQS):
        sc, dc, tb = int(src_col[slot]), int(dst_col[slot]), int(bias[slot])
        for layer in range(NUM_LAYERS):
            conv_view = shadow_views[2 * layer]
            ssm_slots = shadow_views[2 * layer + 1][
                slot * TEMPORAL_SLOTS : (slot + 1) * TEMPORAL_SLOTS
            ]
            if slot not in scheduled or sc < 0:
                assert not conv_view[slot].any() and not ssm_slots.any()
                continue
            # The conv window is kept unshifted; temporal slots 0..bias hold
            # the states of the columns src_col..src_col + bias.
            torch.testing.assert_close(
                conv_view[slot], conv_pre[layer][int(bt_cpu[slot, sc])], rtol=0, atol=0
            )
            for t in range(TEMPORAL_SLOTS):
                if t <= tb:
                    expected = ssm_pre[layer][int(bt_cpu[slot, sc + t])]
                    torch.testing.assert_close(ssm_slots[t], expected, rtol=0, atol=0)
                else:
                    assert not ssm_slots[t].any()
            # Boundary migration is unchanged by the snapshot.
            if sc != dc:
                conv_ref, ssm_ref = _normalized(
                    conv_pre[layer],
                    ssm_pre[layer],
                    bt_cpu,
                    slot,
                    sc,
                    tb,
                    conv_state_dim_first,
                )
                dblk = int(bt_cpu[slot, dc])
                torch.testing.assert_close(
                    _leading(convs[layer][dblk], tb, conv_state_dim_first),
                    _leading(conv_ref, tb, conv_state_dim_first),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(ssms[layer][dblk], ssm_ref, rtol=0, atol=0)

    # Materialize: slot 1 from its shadow at its full bias (2); slot 4 from
    # its own blocks (bias 3); slot 4 again from its shadow at bias 1 (a stop
    # inside the accepted tokens keeps two rows less).
    dst_shadow, dst_slots, dst_trim = num_blocks - 2, num_blocks - 1, num_blocks - 3
    num_groups = 1
    width_meta = ENDPOINT_COPY_META_FIXED + 3 * num_groups
    meta = torch.full((3, width_meta), -1, dtype=torch.int32, device=device)
    meta[0, :3] = torch.tensor([1, 1, 2], dtype=torch.int32)
    meta[0, 4] = dst_shadow
    meta[1, :3] = torch.tensor([0, 0, 3], dtype=torch.int32)
    meta[1, 4] = dst_slots
    meta[1, 5] = int(bt_cpu[4, 1])
    meta[1, 6] = int(bt_cpu[4, 1 + 3])
    meta[2, :3] = torch.tensor([1, 4, 1], dtype=torch.int32)
    meta[2, 4] = dst_trim
    grid = (3, NUM_LAYERS * 2, temporal_tiles)
    materialize_mamba_endpoint_kernel[grid](
        meta,
        meta.stride(0),
        shadow_addrs,
        shadow_strides,
        TEMPORAL_SLOTS,
        base,
        blk_stride,
        elem,
        inner,
        width,
        group,
        drc,
        drs,
        3,
        NUM_GROUPS=num_groups,
        COPY_BLOCK_SIZE=1024,
        CONV_STATE_DIM_FIRST=conv_state_dim_first,
        TEMPORAL_TILES=temporal_tiles,
    )
    torch.accelerator.synchronize()
    for layer in range(NUM_LAYERS):
        for dst, req, sc, tb in (
            (dst_shadow, 1, 2, 2),
            (dst_slots, 4, 1, 3),
            (dst_trim, 4, 1, 1),
        ):
            conv_ref, ssm_ref = _normalized(
                conv_pre[layer],
                ssm_pre[layer],
                bt_cpu,
                req,
                sc,
                tb,
                conv_state_dim_first,
            )
            torch.testing.assert_close(
                _leading(convs[layer][dst], tb, conv_state_dim_first),
                _leading(conv_ref, tb, conv_state_dim_first),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(ssms[layer][dst], ssm_ref, rtol=0, atol=0)


def _mock_attention(conv_state, temporal_state):
    attention = MagicMock()
    attention.kv_cache = [conv_state, temporal_state]
    return attention


class _Buffer:
    def __init__(self, size, dtype, device):
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.gpu = torch.zeros(size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()


def test_context_allocates_shadow_pages_and_materializes_from_them():
    """End-to-end through ``MambaSpecDecodeGPUContext``: the precopy launch
    snapshots the committed state and ``run_endpoint_materialize`` writes it
    into a pool block at the endpoint bias."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        is_conv_state_dim_first,
    )

    device = torch.device("cuda")
    torch.manual_seed(1)
    dim_first = is_conv_state_dim_first()
    # Block ids 1..MAX_REQS * MAX_COLS belong to the block table; the last
    # blocks are the endpoint destinations.
    num_blocks = MAX_REQS * MAX_COLS + 4
    layer_names = ["l0", "l1"]
    convs, ssms = [], []
    for _ in layer_names:
        shape = (
            (num_blocks, CONV_DIM, CONV_WIDTH)
            if dim_first
            else (num_blocks, CONV_WIDTH, CONV_DIM)
        )
        convs.append(torch.randn(*shape, dtype=torch.bfloat16, device=device))
        ssms.append(
            torch.randn(num_blocks, *SSM_SHAPE, dtype=torch.float32, device=device)
        )
    forward_context = {
        name: _mock_attention(c, s) for name, c, s in zip(layer_names, convs, ssms)
    }
    spec = MambaSpec(
        block_size=16,
        shapes=((CONV_WIDTH, CONV_DIM), SSM_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),  # type: ignore[arg-type]
        mamba_cache_mode="align",
        num_speculative_blocks=TEMPORAL_SLOTS - 1,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
    )
    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=MAX_REQS,
        kv_cache_config=kv_cache_config,
        num_state_types=2,
        device=device,
        make_buffer=lambda n, dtype: _Buffer(n, dtype, device),  # type: ignore[arg-type,return-value]
    )
    bt = torch.arange(1, 1 + MAX_REQS * MAX_COLS, dtype=torch.int32, device=device)
    bt = bt.view(MAX_REQS, MAX_COLS)
    num_reqs = 2
    idx_mapping = torch.tensor([2, 5], dtype=torch.int32, device=device)
    # The registered block table is in batch order (V2 layout).
    bt_batch = torch.zeros(MAX_REQS, MAX_COLS, dtype=torch.int32, device=device)
    bt_batch[:num_reqs] = bt[idx_mapping.long()]
    ctx.initialize_from_forward_context(
        kv_cache_config,
        forward_context,
        (get_conv_copy_spec, get_temporal_copy_spec),
        [bt_batch],
    )
    assert not ctx.has_endpoint_shadow
    ctx.ensure_endpoint_shadow(MAX_REQS)
    assert ctx.has_endpoint_shadow and ctx.shadow_num_slots == MAX_REQS
    assert ctx.shadow_temporal_slots == TEMPORAL_SLOTS
    ctx.ensure_endpoint_shadow(MAX_REQS)  # idempotent

    conv_pre = [c.clone() for c in convs]
    ssm_pre = [s.clone() for s in ssms]
    state_idx = torch.zeros(MAX_REQS, dtype=torch.int32, device=device)
    src_col = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=device)
    token_bias = torch.zeros(MAX_REQS, dtype=torch.int32, device=device)
    # Slot 2 stays in column 1 with 2 accepted drafts; slot 5 crosses 0 -> 1.
    state_idx[2], src_col[2], token_bias[2] = 1, 1, 2
    state_idx[5], src_col[5], token_bias[5] = 1, 0, 1
    ctx.run_fused_precopy(
        num_reqs, state_idx, src_col, token_bias, idx_mapping, snapshot_to_shadow=True
    )

    # Slot 2 materialized at its full bias and one row earlier (a stop inside
    # the accepted tokens); slot 5 at its full bias.
    dst_a, dst_a_trim, dst_b = num_blocks - 1, num_blocks - 2, num_blocks - 3
    meta = torch.full(
        (3, ENDPOINT_COPY_META_FIXED + 3 * ctx.num_groups),
        -1,
        dtype=torch.int32,
        device=device,
    )
    meta[0, :3] = torch.tensor([1, 2, 2], dtype=torch.int32)
    meta[0, 4] = dst_a
    meta[1, :3] = torch.tensor([1, 2, 1], dtype=torch.int32)
    meta[1, 4] = dst_a_trim
    meta[2, :3] = torch.tensor([1, 5, 1], dtype=torch.int32)
    meta[2, 4] = dst_b
    ctx.run_endpoint_materialize(meta)
    torch.accelerator.synchronize()

    bt_cpu = bt.cpu()
    for layer in range(len(layer_names)):
        for dst, req, sc, tb in (
            (dst_a, 2, 1, 2),
            (dst_a_trim, 2, 1, 1),
            (dst_b, 5, 0, 1),
        ):
            conv_ref, ssm_ref = _normalized(
                conv_pre[layer], ssm_pre[layer], bt_cpu, req, sc, tb, dim_first
            )
            torch.testing.assert_close(
                _leading(convs[layer][dst], tb, dim_first),
                _leading(conv_ref, tb, dim_first),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(ssms[layer][dst], ssm_ref, rtol=0, atol=0)


def _two_group_context(device, num_blocks, layer_names, convs, ssms, bts):
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
    )

    spec = MambaSpec(
        block_size=16,
        shapes=((CONV_WIDTH, CONV_DIM), SSM_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),  # type: ignore[arg-type]
        mamba_cache_mode="align",
        num_speculative_blocks=TEMPORAL_SLOTS - 1,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names[:2], spec),
            KVCacheGroupSpec(layer_names[2:], spec),
        ],
    )
    forward_context = {
        name: _mock_attention(c, s) for name, c, s in zip(layer_names, convs, ssms)
    }
    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=MAX_REQS,
        kv_cache_config=kv_cache_config,
        num_state_types=2,
        device=device,
        make_buffer=lambda n, dtype: _Buffer(n, dtype, device),  # type: ignore[arg-type,return-value]
    )
    ctx.initialize_from_forward_context(
        kv_cache_config,
        forward_context,
        (get_conv_copy_spec, get_temporal_copy_spec),
        bts,
    )
    ctx.ensure_endpoint_shadow(MAX_REQS)
    return ctx, kv_cache_config, forward_context


def _meta_from_records(records, num_groups, device):
    """Kernel records: [from_shadow, req_idx, token_bias, reserved, dst[G],
    conv_src[G], temporal_src[G]]."""
    width = ENDPOINT_COPY_META_FIXED + 3 * num_groups
    meta = torch.full((len(records), width), -1, dtype=torch.int32)
    for row, rec in enumerate(records):
        meta[row, :3] = torch.tensor(
            [int(rec.from_shadow), rec.req_idx, rec.token_bias]
        )
        meta[row, 4 : 4 + num_groups] = torch.tensor(rec.dst_block_ids)
        if not rec.from_shadow:
            meta[row, 4 + num_groups : 4 + 2 * num_groups] = torch.tensor(
                rec.conv_src_block_ids
            )
            meta[row, 4 + 2 * num_groups : width] = torch.tensor(
                rec.temporal_src_block_ids
            )
    return meta.to(device)


def test_torch_reference_matches_kernel_and_verifies():
    """``materialize_endpoints_torch`` writes the same bytes as the kernel for
    shadow and block sources with one destination block per group, and
    ``verify_endpoints`` reports no mismatch for either result but flags a
    corrupted destination."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.worker.mamba_utils import (
        EndpointCopyRecord,
        endpoint_layer_states,
        materialize_endpoints_torch,
        verify_endpoints,
    )

    device = torch.device("cuda")
    torch.manual_seed(2)
    dim_first = is_conv_state_dim_first()
    # Block-table ids, four records x two groups of destinations, and the
    # torch-path destinations 16 above them.
    num_blocks = MAX_REQS * MAX_COLS + 32
    layer_names = ["l0", "l1", "l2"]
    convs, ssms = [], []
    for _ in layer_names:
        shape = (
            (num_blocks, CONV_DIM, CONV_WIDTH)
            if dim_first
            else (num_blocks, CONV_WIDTH, CONV_DIM)
        )
        convs.append(torch.randn(*shape, dtype=torch.bfloat16, device=device))
        ssms.append(
            torch.randn(num_blocks, *SSM_SHAPE, dtype=torch.float32, device=device)
        )
    num_reqs = 2
    idx_mapping = torch.tensor([2, 5], dtype=torch.int32, device=device)
    bts = []
    for group in range(2):
        bt = torch.zeros(MAX_REQS, MAX_COLS, dtype=torch.int32, device=device)
        bt[0] = torch.arange(1, 1 + MAX_COLS) + group * 20
        bt[1] = torch.arange(9, 9 + MAX_COLS) + group * 20
        bts.append(bt)
    ctx, kv_cache_config, forward_context = _two_group_context(
        device, num_blocks, layer_names, convs, ssms, bts
    )
    layer_states = endpoint_layer_states(
        kv_cache_config, forward_context, ctx.mamba_group_ids
    )
    assert [s[0].data_ptr() for s in layer_states] == [c.data_ptr() for c in convs]

    state_idx = torch.zeros(MAX_REQS, dtype=torch.int32, device=device)
    src_col = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=device)
    token_bias = torch.zeros(MAX_REQS, dtype=torch.int32, device=device)
    state_idx[2], src_col[2], token_bias[2] = 1, 1, 3
    state_idx[5], src_col[5], token_bias[5] = 0, 0, 1
    ctx.run_fused_precopy(
        num_reqs, state_idx, src_col, token_bias, idx_mapping, snapshot_to_shadow=True
    )
    bt_cpu = [bt.cpu() for bt in bts]
    first = MAX_REQS * MAX_COLS + 1
    dst = [(first + r, first + 8 + r) for r in range(4)]
    records = [
        EndpointCopyRecord(True, 2, 3, dst[0], (), ()),
        EndpointCopyRecord(True, 2, 1, dst[1], (), ()),
        EndpointCopyRecord(True, 5, 1, dst[2], (), ()),
        EndpointCopyRecord(
            False,
            0,
            1,
            dst[3],
            tuple(int(bt_cpu[g][1, 0]) for g in range(2)),
            tuple(int(bt_cpu[g][1, 1]) for g in range(2)),
        ),
    ]
    ctx.run_endpoint_materialize(_meta_from_records(records, ctx.num_groups, device))
    torch.accelerator.synchronize()
    assert verify_endpoints(ctx, layer_states, records, dim_first) == []

    torch_records = [
        rec._replace(dst_block_ids=tuple(d + 16 for d in rec.dst_block_ids))
        for rec in records
    ]
    materialize_endpoints_torch(ctx, layer_states, torch_records, dim_first)
    torch.accelerator.synchronize()
    assert verify_endpoints(ctx, layer_states, torch_records, dim_first) == []
    for layer in range(len(layer_names)):
        group = ctx.state_group_indices_cpu[2 * layer]
        for rec, trec in zip(records, torch_records):
            d, td = rec.dst_block_ids[group], trec.dst_block_ids[group]
            torch.testing.assert_close(
                _leading(convs[layer][d], rec.token_bias, dim_first),
                _leading(convs[layer][td], rec.token_bias, dim_first),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(ssms[layer][d], ssms[layer][td], rtol=0, atol=0)

    # A corrupted destination is reported with its layer and state kind.
    ssms[1][records[0].dst_block_ids[0]].add_(1.0)
    mismatches = verify_endpoints(ctx, layer_states, records, dim_first)
    assert [(m["record"], m["layer"], m["kind"]) for m in mismatches] == [
        (0, 1, "temporal")
    ]


def test_groups_sharing_raw_tensors_need_distinct_destinations():
    """Served layout: one raw tensor per layer index is shared by every
    cache group, so a block id names the same page in each group. Distinct
    per-group destinations materialize every layer; a shared destination
    (the layout of the failed candidate) makes the groups overwrite each
    other."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.worker.mamba_utils import (
        EndpointCopyRecord,
        endpoint_layer_states,
        materialize_endpoints_torch,
        verify_endpoints,
    )

    device = torch.device("cuda")
    torch.manual_seed(4)
    dim_first = is_conv_state_dim_first()
    num_blocks = MAX_REQS * MAX_COLS + 24
    shape = (
        (num_blocks, CONV_DIM, CONV_WIDTH)
        if dim_first
        else (num_blocks, CONV_WIDTH, CONV_DIM)
    )
    # Layer i of group 0 and layer i of group 1 view the same storage.
    shared_convs = [
        torch.randn(*shape, dtype=torch.bfloat16, device=device) for _ in range(2)
    ]
    shared_ssms = [
        torch.randn(num_blocks, *SSM_SHAPE, dtype=torch.float32, device=device)
        for _ in range(2)
    ]
    layer_names = ["g0l0", "g0l1", "g1l0", "g1l1"]
    convs = [shared_convs[0], shared_convs[1], shared_convs[0], shared_convs[1]]
    ssms = [shared_ssms[0], shared_ssms[1], shared_ssms[0], shared_ssms[1]]
    num_reqs = 1
    idx_mapping = torch.tensor([1], dtype=torch.int32, device=device)
    # Block ids are unique across groups, as with a shared block pool.
    bts = []
    for group in range(2):
        bt = torch.zeros(MAX_REQS, MAX_COLS, dtype=torch.int32, device=device)
        bt[0] = torch.arange(1, 1 + MAX_COLS) + group * MAX_COLS
        bts.append(bt)
    ctx, kv_cache_config, forward_context = _two_group_context(
        device, num_blocks, layer_names, convs, ssms, bts
    )
    layer_states = endpoint_layer_states(
        kv_cache_config, forward_context, ctx.mamba_group_ids
    )
    state_idx = torch.zeros(MAX_REQS, dtype=torch.int32, device=device)
    src_col = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=device)
    token_bias = torch.zeros(MAX_REQS, dtype=torch.int32, device=device)
    state_idx[1], src_col[1], token_bias[1] = 0, 0, 2
    ctx.run_fused_precopy(
        num_reqs, state_idx, src_col, token_bias, idx_mapping, snapshot_to_shadow=True
    )
    bt_cpu = [bt.cpu() for bt in bts]
    first = 2 * MAX_COLS + 1
    good = [
        EndpointCopyRecord(True, 1, 2, (first, first + 1), (), ()),
        EndpointCopyRecord(
            False,
            0,
            2,
            (first + 2, first + 3),
            tuple(int(bt_cpu[g][0, 0]) for g in range(2)),
            tuple(int(bt_cpu[g][0, 2]) for g in range(2)),
        ),
    ]
    ctx.run_endpoint_materialize(_meta_from_records(good, ctx.num_groups, device))
    torch.accelerator.synchronize()
    assert verify_endpoints(ctx, layer_states, good, dim_first) == []
    torch_good = [
        rec._replace(dst_block_ids=tuple(d + 4 for d in rec.dst_block_ids))
        for rec in good
    ]
    materialize_endpoints_torch(ctx, layer_states, torch_good, dim_first)
    torch.accelerator.synchronize()
    assert verify_endpoints(ctx, layer_states, torch_good, dim_first) == []

    shared = [
        rec._replace(dst_block_ids=(first + 8 + i,) * 2) for i, rec in enumerate(good)
    ]
    materialize_endpoints_torch(ctx, layer_states, shared, dim_first)
    torch.accelerator.synchronize()
    mismatches = verify_endpoints(ctx, layer_states, shared, dim_first)
    # Group 1 writes last on the sequential torch path: group 0's states
    # (two layers, two state types each) are overwritten for both records.
    assert len(mismatches) == 2 * 4
    assert {m["layer"] for m in mismatches} == {0, 1}


def test_served_page_geometry_kernel_torch_and_trace():
    """The served Kimi-K3 layout: conv ``(6, 4608)`` bf16 and temporal
    ``(12, 128, 128)`` fp32 views of every layer share 1,007,616-byte pages
    (temporal at byte offset 55,296), four recurrent groups whose layers
    share one raw tensor, four temporal slots. The kernel's shadow and
    block-source materializations, the torch reference path and the
    resolved-address trace must all agree."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        is_conv_state_dim_first,
    )
    from vllm.v1.worker.mamba_utils import (
        EndpointCopyRecord,
        endpoint_layer_states,
        materialize_endpoints_torch,
        verify_endpoints,
    )

    if is_conv_state_dim_first():
        pytest.skip("the served layout keeps the conv slide axis first")
    device = torch.device("cuda")
    torch.manual_seed(3)
    page, temporal_off = 1007616, 55296
    conv_shape, temporal_shape = (6, 4608), (12, 128, 128)
    num_groups, num_blocks, max_reqs, max_cols, slots = 4, 96, 4, 8, 4
    # One raw tensor shared by the single layer of every group (served
    # sharing rule: raw tensor i holds layer i of each group).
    raw = torch.randint(0, 256, (num_blocks * page,), dtype=torch.uint8, device=device)
    conv = raw.view(torch.bfloat16).as_strided(
        (num_blocks,) + conv_shape, (page // 2, 4608, 1)
    )
    temp = (
        raw[temporal_off:]
        .view(torch.float32)
        .as_strided((num_blocks,) + temporal_shape, (page // 4, 16384, 128, 1))
    )
    conv.copy_(torch.randn(conv.shape, dtype=torch.bfloat16, device=device))
    temp.copy_(torch.randn(temp.shape, dtype=torch.float32, device=device))
    assert temp.data_ptr() - conv.data_ptr() == temporal_off
    groups = [[f"g{g}"] for g in range(num_groups)]
    forward_context = {names[0]: _mock_attention(conv, temp) for names in groups}
    spec = MambaSpec(
        block_size=4608,
        shapes=(conv_shape, temporal_shape),
        dtypes=(torch.bfloat16, torch.float32),  # type: ignore[arg-type]
        mamba_cache_mode="align",
        num_speculative_blocks=slots - 1,
        page_size_padded=page,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(names, spec) for names in groups],
    )
    ctx = MambaSpecDecodeGPUContext.create(
        max_num_reqs=max_reqs,
        kv_cache_config=kv_cache_config,
        num_state_types=2,
        device=device,
        make_buffer=lambda n, dtype: _Buffer(n, dtype, device),  # type: ignore[arg-type,return-value]
    )
    # Block ids are unique across groups (nine per group).
    bts = []
    for g in range(num_groups):
        base = 1 + 9 * g
        bt = torch.zeros(max_reqs, max_cols, dtype=torch.int32, device=device)
        bt[0] = torch.tensor([base, base + 1, base + 2, base + 3, base + 4, 0, 0, 0])
        bt[1] = torch.tensor([base + 5, base + 6, base + 7, base + 8, 0, 0, 0, 0])
        bts.append(bt)
    first_free = 1 + 9 * num_groups
    ctx.initialize_from_forward_context(
        kv_cache_config,
        forward_context,
        (get_conv_copy_spec, get_temporal_copy_spec),
        bts,
    )
    ctx.ensure_endpoint_shadow(max_reqs)
    assert ctx.state_block_strides_cpu[:2] == [page, page]
    layer_states = endpoint_layer_states(
        kv_cache_config, forward_context, ctx.mamba_group_ids
    )

    num_reqs = 2
    idx_mapping = torch.tensor([1, 3], dtype=torch.int32, device=device)
    state_idx = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    src_col = torch.full((max_reqs,), -1, dtype=torch.int32, device=device)
    token_bias = torch.zeros(max_reqs, dtype=torch.int32, device=device)
    state_idx[1], src_col[1], token_bias[1] = 1, 1, 3
    state_idx[3], src_col[3], token_bias[3] = 0, 0, 1
    conv_pre = conv.clone()
    temp_pre = temp.clone()
    ctx.run_fused_precopy(
        num_reqs, state_idx, src_col, token_bias, idx_mapping, snapshot_to_shadow=True
    )
    bt_cpu = [bt.cpu() for bt in bts]
    # Record r of group g writes block first_free + 4 * g + r.
    dst = [tuple(first_free + 4 * g + r for g in range(num_groups)) for r in range(4)]
    records = [
        EndpointCopyRecord(True, 1, 3, dst[0], (), ()),
        EndpointCopyRecord(True, 1, 2, dst[1], (), ()),
        EndpointCopyRecord(True, 3, 1, dst[2], (), ()),
        EndpointCopyRecord(
            False,
            0,
            1,
            dst[3],
            tuple(int(bt_cpu[g][1, 0]) for g in range(num_groups)),
            tuple(int(bt_cpu[g][1, 1]) for g in range(num_groups)),
        ),
    ]
    total_states = ctx.num_layers * ctx.num_state_types
    trace = torch.zeros(
        (len(records), total_states, 4), dtype=torch.int64, device=device
    )
    ctx.run_endpoint_materialize(_meta_from_records(records, num_groups, device), trace)
    torch.accelerator.synchronize()
    assert trace.cpu().tolist() == [
        [list(t) for t in per_state]
        for per_state in ctx.expected_endpoint_addresses(records)
    ]
    assert verify_endpoints(ctx, layer_states, records, False) == []

    # Reference semantics straight from the pre-copy clones: the window of
    # the source column shifted by the bias and the temporal slot at the bias.
    for layer in range(num_groups):
        g = ctx.state_group_indices_cpu[2 * layer]
        for rec, row in zip(records, (0, 0, 1, 1)):
            col = 1 if row == 0 else 0
            sblk = int(bt_cpu[g][row, col])
            tblk = int(bt_cpu[g][row, col + rec.token_bias])
            n = 6 - rec.token_bias
            d = rec.dst_block_ids[g]
            assert torch.equal(conv[d][:n], conv_pre[sblk][rec.token_bias :])
            assert torch.equal(temp[d], temp_pre[tblk])
    for b in range(first_free):
        assert torch.equal(conv[b], conv_pre[b])
        assert torch.equal(temp[b], temp_pre[b])

    torch_records = [
        rec._replace(dst_block_ids=tuple(d + 4 * num_groups for d in rec.dst_block_ids))
        for rec in records
    ]
    materialize_endpoints_torch(ctx, layer_states, torch_records, False)
    torch.accelerator.synchronize()
    assert verify_endpoints(ctx, layer_states, torch_records, False) == []
    for layer in range(num_groups):
        g = ctx.state_group_indices_cpu[2 * layer]
        for rec, trec in zip(records, torch_records):
            n = 6 - rec.token_bias
            d, td = rec.dst_block_ids[g], trec.dst_block_ids[g]
            assert torch.equal(conv[d][:n], conv[td][:n])
            assert torch.equal(temp[d], temp[td])
