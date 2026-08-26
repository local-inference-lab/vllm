# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float8E4M3FN, Float16, Float32, Int32, Uint8, Uint16, Uint32
from cutlass.cute.nvgpu import warp
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import mma_sync, recast_val
from vllm.utils.torch_utils import current_stream


class KVarNK4V2MmaStage1:
    D = 128
    GROUP = 128
    HQ = 32
    HK = 8
    QPK = 4
    THREADS = 128
    RECORDS_PER_TILE = 1
    TILE_TOKENS = 128
    SMEM_BYTES = 40192
    K_BITS = 4
    V_BITS = 2
    K_PACKED_OFFSET = 0
    K_S_COL_OFFSET = 8192
    K_ZP_OFFSET = 8448
    K_S_ROW_OFFSET = 8704
    V_PACKED_OFFSET = 8960
    V_S_COL_OFFSET = 13056
    V_S_ROW_OFFSET = 13312
    V_ZP_OFFSET = 13568
    TILE_BYTES = 13824

    def __init__(self, tail_scaled: bool):
        self.tail_scaled = tail_scaled

    @cute.jit
    def _load_f16(self, kv: cute.Tensor, page: Int32, hk: Int32, offset: Int32):
        lo = Uint16(kv[page, hk, offset])
        hi = Uint16(kv[page, hk, offset + Int32(1)])
        return recast_val(lo | (hi << Uint16(8)), Float16).to(Float32)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        kv: cute.Tensor,
        tail_k: cute.Tensor,
        tail_v: cute.Tensor,
        tail_ks: cute.Tensor,
        tail_vs: cute.Tensor,
        block_table: cute.Tensor,
        seq_lens: cute.Tensor,
        block_to_slot: cute.Tensor,
        mid_o: cute.Tensor,
        mid_lse: cute.Tensor,
        num_splits: Int32,
        stream: CUstream,
    ):
        batch = q.shape[0]
        self.kernel(
            q,
            kv,
            tail_k,
            tail_v,
            tail_ks,
            tail_vs,
            block_table,
            seq_lens,
            block_to_slot,
            mid_o,
            mid_lse,
            num_splits,
        ).launch(
            grid=[num_splits, batch * self.HK, 1],
            block=[self.THREADS, 1, 1],
            smem=self.SMEM_BYTES,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        kv: cute.Tensor,
        tail_k: cute.Tensor,
        tail_v: cute.Tensor,
        tail_ks: cute.Tensor,
        tail_vs: cute.Tensor,
        block_table: cute.Tensor,
        seq_lens: cute.Tensor,
        block_to_slot: cute.Tensor,
        mid_o: cute.Tensor,
        mid_lse: cute.Tensor,
        num_splits: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, bhk, _ = cute.arch.block_idx()
        b = bhk // Int32(self.HK)
        hk = bhk - b * Int32(self.HK)
        warp_id = cute.arch.warp_idx()
        lane_id = cute.arch.lane_idx()
        head = warp_id
        num_records = (seq_lens[b] + Int32(self.GROUP - 1)) // Int32(self.GROUP)

        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(
            Float16,
            cute.make_layout((self.TILE_TOKENS, self.D), stride=(self.D, 1)),
            byte_alignment=16,
        )
        sQ = smem.allocate_tensor(
            Float16,
            cute.make_layout((self.RECORDS_PER_TILE * 8, self.D), stride=(self.D, 1)),
            byte_alignment=16,
        )
        sScore = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.QPK, self.TILE_TOKENS), stride=(self.TILE_TOKENS, 1)
            ),
            byte_alignment=16,
        )
        sP = smem.allocate_tensor(
            Float16,
            cute.make_layout((8, self.TILE_TOKENS), stride=(self.TILE_TOKENS, 1)),
            byte_alignment=16,
        )
        sZq = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.RECORDS_PER_TILE, self.QPK), stride=(self.QPK, 1)),
        )
        sStats = smem.allocate_tensor(
            Float32, cute.make_layout((2, self.QPK), stride=(self.QPK, 1))
        )
        sGlobal = smem.allocate_tensor(
            Float32, cute.make_layout((2, self.QPK), stride=(self.QPK, 1))
        )
        sMerge = smem.allocate_tensor(
            Float32, cute.make_layout((2, self.QPK), stride=(self.QPK, 1))
        )

        elems = 8
        mma_k = 16
        ldsm_atom = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(num_matrices=4), Float16)
        rAccum = cute.make_rmem_tensor((4, 2, 1), Float32)
        rAccum.fill(0.0)
        if tid < Int32(self.QPK):
            sGlobal[0, tid] = -Float32.inf
            sGlobal[1, tid] = Float32(0.0)
        cute.arch.sync_threads()

        num_tiles = (num_records + Int32(self.RECORDS_PER_TILE - 1)) // Int32(
            self.RECORDS_PER_TILE
        )
        for tile_iter in cutlass.range_constexpr(8):
            logical = split + Int32(tile_iter) * num_splits
            if logical < num_tiles:
                start = logical * Int32(self.TILE_TOKENS)
                valid = cutlass.min(Int32(self.TILE_TOKENS), seq_lens[b] - start)
                record_offset = tid // Int32(self.GROUP)
                token_local = tid - record_offset * Int32(self.GROUP)
                record_index = logical * Int32(self.RECORDS_PER_TILE) + record_offset
                safe_record = cutlass.min(record_index, num_records - Int32(1))
                page = block_table[b, safe_record]
                pool_slot = block_to_slot[page]

                if pool_slot >= Int32(0):
                    k_scale = (
                        tail_ks[pool_slot, token_local, hk].to(Float32)
                        if cutlass.const_expr(self.tail_scaled)
                        else Float32(1.0)
                    )
                    for d in cutlass.range_constexpr(self.D):
                        kval = (
                            tail_k[pool_slot, token_local, hk, d].to(Float32) * k_scale
                        )
                        sK[tid, d] = kval.to(Float16) if tid < valid else Float16(0.0)
                else:
                    for d in cutlass.range_constexpr(self.D):
                        bit_offset = token_local * Int32(self.K_BITS)
                        byte_offset = (
                            Int32(self.K_PACKED_OFFSET)
                            + d * Int32(self.GROUP * self.K_BITS // 8)
                            + bit_offset // Int32(8)
                        )
                        packed = Uint32(kv[page, hk, byte_offset]) | (
                            Uint32(kv[page, hk, byte_offset + Int32(1)]) << Uint32(8)
                        )
                        shift = Uint32(bit_offset & Int32(7))
                        code = Float32(
                            (packed >> shift) & Uint32((1 << self.K_BITS) - 1)
                        )
                        sK[tid, d] = code.to(Float16) if tid < valid else Float16(0.0)

                for record in cutlass.range_constexpr(self.RECORDS_PER_TILE):
                    q_record = logical * Int32(self.RECORDS_PER_TILE) + Int32(record)
                    q_safe_record = cutlass.min(q_record, num_records - Int32(1))
                    q_page = block_table[b, q_safe_record]
                    q_pool_slot = block_to_slot[q_page]
                    if q_pool_slot >= Int32(0):
                        for h in cutlass.range_constexpr(8):
                            sQ[record * 8 + h, tid] = (
                                (
                                    q[b, hk * self.QPK + h, tid].to(Float32)
                                    / math.sqrt(self.D)
                                ).to(Float16)
                                if h < self.QPK
                                else Float16(0.0)
                            )
                        if warp_id < Int32(self.QPK):
                            sZq[record, warp_id] = Float32(0.0)
                    else:
                        s_col = self._load_f16(
                            kv,
                            q_page,
                            hk,
                            Int32(self.K_S_COL_OFFSET) + tid * Int32(2),
                        )
                        for h in cutlass.range_constexpr(8):
                            sQ[record * 8 + h, tid] = (
                                (
                                    q[b, hk * self.QPK + h, tid].to(Float32)
                                    * s_col
                                    / math.sqrt(self.D)
                                ).to(Float16)
                                if h < self.QPK
                                else Float16(0.0)
                            )
                        if warp_id < Int32(self.QPK):
                            zpart = Float32(0.0)
                            for j in cutlass.range_constexpr(4):
                                d = lane_id + Int32(j * 32)
                                zpart += (
                                    q[
                                        b,
                                        hk * self.QPK + warp_id,
                                        d,
                                    ].to(Float32)
                                    * self._load_f16(
                                        kv,
                                        q_page,
                                        hk,
                                        Int32(self.K_ZP_OFFSET) + d * Int32(2),
                                    )
                                    / math.sqrt(self.D)
                                )
                            sZq[record, warp_id] = cute.arch.warp_reduction_sum(zpart)
                for h in cutlass.range_constexpr(self.QPK, 8):
                    sP[h, tid] = Float16(0.0)
                cute.arch.sync_threads()

                qk_warps = self.GROUP // 32
                qk_record = warp_id // Int32(qk_warps)
                sK_warp = cute.local_tile(sK, (32, self.D), (warp_id, 0))
                sQ_record = cute.local_tile(sQ, (8, self.D), (qk_record, 0))
                sK_ldsm = cute.zipped_divide(
                    sK_warp, (16, cute.make_layout((elems, 2)))
                )
                sQ_ldsm = cute.zipped_divide(
                    sQ_record, (8, cute.make_layout((elems, 4)))
                )
                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16)), None]
                sQ_ldsm = sQ_ldsm[(lane_id % 8, (None, lane_id // 8)), None]
                rQ = cute.make_rmem_tensor(((4, 2), 4, 1), Float16)
                rK = cute.make_rmem_tensor((8, 2, 8), Float16)
                rC = cute.make_rmem_tensor((4, 2, 1), Float32)
                cute.copy(
                    ldsm_atom,
                    sQ_ldsm[None, (0, None)],
                    rQ[None, None, 0],
                )
                rC.fill(0.0)
                for k in cutlass.range_constexpr(self.D // mma_k):
                    cute.copy(
                        ldsm_atom,
                        sK_ldsm[None, (None, k)],
                        rK[None, None, k],
                    )
                    for m in cutlass.range_constexpr(2):
                        rC[None, m, 0] = mma_sync(
                            rK[None, m, k],
                            rQ[(None, k % 2), k // 2, 0],
                            rC[None, m, 0],
                        )

                score_record = logical * Int32(self.RECORDS_PER_TILE) + qk_record
                score_safe_record = cutlass.min(score_record, num_records - Int32(1))
                score_page = block_table[b, score_safe_record]
                score_pool_slot = block_to_slot[score_page]
                for i in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(2):
                        token = warp_id * Int32(32) + Int32(i * 8) + lane_id // Int32(4)
                        token_in_record = token - qk_record * Int32(self.GROUP)
                        h = (lane_id % Int32(4)) * Int32(2) + Int32(j)
                        if h < Int32(self.QPK):
                            score = rC[i * 2 + j]
                            if score_pool_slot < Int32(0):
                                score = (score + sZq[qk_record, h]) * self._load_f16(
                                    kv,
                                    score_page,
                                    hk,
                                    Int32(self.K_S_ROW_OFFSET)
                                    + token_in_record * Int32(2),
                                )
                            sScore[h, token] = score if token < valid else -Float32.inf
                cute.arch.sync_threads()

                local_max = -Float32.inf
                for j in cutlass.range_constexpr(self.TILE_TOKENS // 32):
                    token = lane_id + Int32(j * 32)
                    local_max = cute.arch.fmax(local_max, sScore[head, token])
                maxv = cute.arch.warp_reduction_max(local_max)
                local_sum = Float32(0.0)
                for j in cutlass.range_constexpr(self.TILE_TOKENS // 32):
                    token = lane_id + Int32(j * 32)
                    p = (
                        cute.math.exp2(
                            (sScore[head, token] - maxv) * math.log2(math.e),
                            fastmath=True,
                        )
                        if token < valid
                        else Float32(0.0)
                    )
                    local_sum += p
                    sP[head, token] = p.to(Float16)
                denom = cute.arch.warp_reduction_sum(local_sum)
                if lane_id == Int32(0):
                    sStats[0, head] = maxv
                    sStats[1, head] = denom
                cute.arch.sync_threads()

                sV = cute.make_tensor(
                    sK.iterator,
                    cute.make_layout(
                        (self.D, self.TILE_TOKENS),
                        stride=(self.TILE_TOKENS, 1),
                    ),
                )
                if pool_slot >= Int32(0):
                    v_scale = (
                        tail_vs[pool_slot, token_local, hk].to(Float32)
                        if cutlass.const_expr(self.tail_scaled)
                        else Float32(1.0)
                    )
                    for d in cutlass.range_constexpr(self.D):
                        vval = (
                            tail_v[pool_slot, token_local, hk, d].to(Float32) * v_scale
                        )
                        sV[d, tid] = vval.to(Float16) if tid < valid else Float16(0.0)
                else:
                    s_row = self._load_f16(
                        kv,
                        page,
                        hk,
                        Int32(self.V_S_ROW_OFFSET) + token_local * Int32(2),
                    )
                    zp_v = self._load_f16(
                        kv,
                        page,
                        hk,
                        Int32(self.V_ZP_OFFSET) + token_local * Int32(2),
                    )
                    for d in cutlass.range_constexpr(self.D):
                        s_col = self._load_f16(
                            kv,
                            page,
                            hk,
                            Int32(self.V_S_COL_OFFSET) + d * Int32(2),
                        )
                        bit_offset = d * Int32(self.V_BITS)
                        byte_offset = (
                            Int32(self.V_PACKED_OFFSET)
                            + token_local * Int32(self.D * self.V_BITS // 8)
                            + bit_offset // Int32(8)
                        )
                        packed = Uint32(kv[page, hk, byte_offset]) | (
                            Uint32(kv[page, hk, byte_offset + Int32(1)]) << Uint32(8)
                        )
                        shift = Uint32(bit_offset & Int32(7))
                        code = Float32(
                            (packed >> shift) & Uint32((1 << self.V_BITS) - 1)
                        )
                        vval = (code * s_row + zp_v) * s_col
                        sV[d, tid] = vval.to(Float16) if tid < valid else Float16(0.0)
                cute.arch.sync_threads()

                sV_warp = cute.local_tile(sV, (32, self.TILE_TOKENS), (warp_id, 0))
                sV_ldsm = cute.zipped_divide(
                    sV_warp, (16, cute.make_layout((elems, 2)))
                )
                sP_ldsm = cute.zipped_divide(sP, (8, cute.make_layout((elems, 4))))
                sV_ldsm = sV_ldsm[(lane_id % 16, (None, lane_id // 16)), None]
                sP_ldsm = sP_ldsm[(lane_id % 8, (None, lane_id // 8)), None]
                rP = cute.make_rmem_tensor(((4, 2), self.TILE_TOKENS // 32, 1), Float16)
                rV = cute.make_rmem_tensor((8, 2, self.TILE_TOKENS // mma_k), Float16)
                rO = cute.make_rmem_tensor((4, 2, 1), Float32)
                cute.copy(
                    ldsm_atom,
                    sP_ldsm[None, (0, None)],
                    rP[None, None, 0],
                )
                rO.fill(0.0)
                for k in cutlass.range_constexpr(self.TILE_TOKENS // mma_k):
                    cute.copy(
                        ldsm_atom,
                        sV_ldsm[None, (None, k)],
                        rV[None, None, k],
                    )
                    for m in cutlass.range_constexpr(2):
                        rO[None, m, 0] = mma_sync(
                            rV[None, m, k],
                            rP[(None, k % 2), k // 2, 0],
                            rO[None, m, 0],
                        )

                if tid < Int32(self.QPK):
                    h = tid
                    merged_max = cute.arch.fmax(sGlobal[0, h], sStats[0, h])
                    alpha = cute.math.exp2(
                        (sGlobal[0, h] - merged_max) * math.log2(math.e),
                        fastmath=True,
                    )
                    beta = cute.math.exp2(
                        (sStats[0, h] - merged_max) * math.log2(math.e),
                        fastmath=True,
                    )
                    sMerge[0, h] = alpha
                    sMerge[1, h] = beta
                    sGlobal[0, h] = merged_max
                    sGlobal[1, h] = sGlobal[1, h] * alpha + sStats[1, h] * beta
                cute.arch.sync_threads()
                for i in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(2):
                        h = (lane_id % Int32(4)) * Int32(2) + Int32(j)
                        if h < Int32(self.QPK):
                            rAccum[i * 2 + j] = (
                                rAccum[i * 2 + j] * sMerge[0, h]
                                + rO[i * 2 + j] * sMerge[1, h]
                            )
                cute.arch.sync_threads()

        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(2):
                d = warp_id * Int32(32) + Int32(i * 8) + lane_id // Int32(4)
                h = (lane_id % Int32(4)) * Int32(2) + Int32(j)
                if h < Int32(self.QPK):
                    row = b * Int32(self.HQ) + hk * Int32(self.QPK) + h
                    mid_o[row, split, d] = (
                        rAccum[i * 2 + j] / sGlobal[1, h]
                        if sGlobal[1, h] > Float32(0.0)
                        else Float32(0.0)
                    )
        if tid < Int32(self.QPK):
            row = b * Int32(self.HQ) + hk * Int32(self.QPK) + tid
            mid_lse[row, split] = (
                sGlobal[0, tid] + cute.math.log(sGlobal[1, tid], fastmath=True)
                if sGlobal[1, tid] > Float32(0.0)
                else -Float32.inf
            )


class KVarNK4V4MmaStage1(KVarNK4V2MmaStage1):
    V_BITS = 4
    V_S_COL_OFFSET = 17152
    V_S_ROW_OFFSET = 17408
    V_ZP_OFFSET = 17664
    TILE_BYTES = 17920


class KVarNK5V5MmaStage1(KVarNK4V2MmaStage1):
    RECORDS_PER_TILE = 2
    SMEM_BYTES = 43008
    GROUP = 64
    K_BITS = 5
    V_BITS = 5
    K_PACKED_OFFSET = 0
    K_S_COL_OFFSET = 5120
    K_ZP_OFFSET = 5376
    K_S_ROW_OFFSET = 5632
    V_PACKED_OFFSET = 5760
    V_S_COL_OFFSET = 10880
    V_S_ROW_OFFSET = 11136
    V_ZP_OFFSET = 11264
    TILE_BYTES = 11392


def _stream() -> CUstream:
    return CUstream(current_stream().cuda_stream)


@cache
def compile_kernel(cache_dtype: str, tail_scaled: bool):
    B, P, S, NB, R, NS, SA, SH, L = (cute.sym_int() for _ in range(9))
    tail_type = Float8E4M3FN if tail_scaled else Float16
    if cache_dtype == "kvarn_k4v2_g128":
        kernel_cls = KVarNK4V2MmaStage1
    elif cache_dtype == "kvarn_k4v4_g128":
        kernel_cls = KVarNK4V4MmaStage1
    elif cache_dtype == "kvarn_k5v5_g64":
        kernel_cls = KVarNK5V5MmaStage1
    else:
        raise ValueError(f"Unsupported CuTeDSL KVarN format: {cache_dtype}")
    args = (
        make_fake_tensor(Float16, (B, 32, 128), divisibility=8),
        make_fake_tensor(
            Uint8,
            (P, 8, kernel_cls.TILE_BYTES),
            divisibility=16,
        ),
        make_fake_tensor(
            tail_type,
            (S, kernel_cls.GROUP, 8, 128),
            divisibility=8,
        ),
        make_fake_tensor(
            tail_type,
            (S, kernel_cls.GROUP, 8, 128),
            divisibility=8,
        ),
        make_fake_tensor(Float16, (S, SA, SH), divisibility=1),
        make_fake_tensor(Float16, (S, SA, SH), divisibility=1),
        make_fake_tensor(Int32, (B, NB), divisibility=4),
        make_fake_tensor(Int32, (B,), divisibility=4),
        make_fake_tensor(Int32, (L,), divisibility=4),
        make_fake_tensor(Float32, (R, NS, 128), divisibility=4),
        make_fake_tensor(Float32, (R, NS), divisibility=1),
        Int32(1),
        _stream(),
    )
    return cute.compile(
        kernel_cls(tail_scaled),
        *args,
        options="--enable-tvm-ffi",
    )


def run(
    q,
    kv,
    tail_k,
    tail_v,
    tail_ks,
    tail_vs,
    block_table,
    seq_lens,
    block_to_slot,
    mid_o,
    mid_lse,
    num_splits,
    cache_dtype,
):
    compile_kernel(cache_dtype, tail_k.dtype == torch.float8_e4m3fn)(
        q,
        kv,
        tail_k,
        tail_v,
        tail_ks,
        tail_vs,
        block_table,
        seq_lens,
        block_to_slot,
        mid_o,
        mid_lse,
        num_splits,
        _stream(),
    )
