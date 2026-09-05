// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Split-KV paged decode attention for the DFlash draft (bf16 KV, head_dim 128, GQA 4, <= 8 queries per request,
// causal + left sliding window). Kernel 1: one CTA per (128-key chunk, kv head, request) computes fp32 partials
// (O, m, l) for the 32 rows = 8 queries x 4 q-heads. Kernel 2 merges the chunks. Draft-only numerics.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#define D 128
#define ROWS 32
#define QMAX 8
#define GQA 4
#define CHUNK 128
#define NTHREADS 128
#define NEG (-1e30f)

struct Params {
    const __nv_bfloat16* q; int q_stride_t;          // [T, H, D], stride per token (elements)
    const __nv_bfloat16* kc; const __nv_bfloat16* vc; // strided caches: row(blk,pos,h) = base + blk*s_blk + pos*s_pos + h*s_h
    long long s_blk, s_pos, s_h;
    int bs; int hkv; int H;
    const int* block_table; int bt_stride;           // [B, max_blocks]
    const int* seqused; const int* cu_q;
    float scale; int window;                          // window W: key kp visible iff kp > qpos - W (W <= 0: no window)
    int causal;                                       // 1: kp <= qpos; 0: bidirectional inside the window (DFlash block attention)
    float* part_o; float* part_m; float* part_l;      // [B][HKV][MAXS][ROWS][D] / [B][HKV][MAXS][ROWS]
    int max_splits;
    __nv_bfloat16* out;                               // [T, H, D]
};

__device__ __forceinline__ float dot8(uint4 a, uint4 b) {
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { float2 fa = __bfloat1622float2(pa[i]); float2 fb = __bfloat1622float2(pb[i]); acc = fmaf(fa.x, fb.x, acc); acc = fmaf(fa.y, fb.y, acc); }
    return acc;
}

extern __shared__ __align__(16) unsigned char smem_raw[];

__global__ void __launch_bounds__(NTHREADS)
attn_partial_kernel(Params p) {
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);                 // [ROWS][D]   8 KB
    float* S = reinterpret_cast<float*>(smem_raw + ROWS * D * 2);                    // [ROWS][CHUNK] 16 KB
    __nv_bfloat16* Vs = reinterpret_cast<__nv_bfloat16*>(smem_raw + ROWS * D * 2 + ROWS * CHUNK * 4); // [CHUNK][D] 32 KB
    float* ms = reinterpret_cast<float*>(smem_raw + ROWS * D * 2 + ROWS * CHUNK * 4 + CHUNK * D * 2);  // [ROWS]
    float* ls = ms + ROWS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int s = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
    const int q0 = p.cu_q[b], qlen = p.cu_q[b + 1] - q0, kend = p.seqused[b];
    int kbeg = 0;
    if (p.window > 0) { kbeg = kend - qlen - p.window + 1; if (kbeg < 0) kbeg = 0; }
    const int kbase = kbeg + s * CHUNK;
    const size_t pidx = ((size_t)(b * p.hkv + g) * p.max_splits + s);
    float* po = p.part_o + pidx * ROWS * D;
    if (kbase >= kend || qlen <= 0) {
        if (tid < ROWS) { p.part_m[pidx * ROWS + tid] = NEG; p.part_l[pidx * ROWS + tid] = 0.f; }
        for (int i = tid; i < ROWS * D; i += NTHREADS) po[i] = 0.f;
        return;
    }
    // Q rows r = j*8 + i  (j = q-head within the group, i = query index)
    for (int idx = tid; idx < ROWS * (D / 8); idx += NTHREADS) {
        const int r = idx / (D / 8), c = idx - r * (D / 8), i = r & 7, j = r >> 3;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (i < qlen) v = *reinterpret_cast<const uint4*>(p.q + (size_t)(q0 + i) * p.q_stride_t + (size_t)(g * GQA + j) * D + c * 8);
        *reinterpret_cast<uint4*>(Qs + r * D + c * 8) = v;
    }
    // this thread's key
    const int kp = kbase + tid;
    const bool valid = kp < kend;
    uint4 kreg[D / 8];
    if (valid) {
        const int blk = p.block_table[(size_t)b * p.bt_stride + kp / p.bs], off = kp - (kp / p.bs) * p.bs;
        const long long rowoff = (long long)blk * p.s_blk + (long long)off * p.s_pos + (long long)g * p.s_h;
        const uint4* kr = reinterpret_cast<const uint4*>(p.kc + rowoff);
        const uint4* vr = reinterpret_cast<const uint4*>(p.vc + rowoff);
#pragma unroll
        for (int c = 0; c < D / 8; ++c) kreg[c] = __ldg(kr + c);
#pragma unroll
        for (int c = 0; c < D / 8; ++c) *reinterpret_cast<uint4*>(Vs + tid * D + c * 8) = __ldg(vr + c);
    } else {
#pragma unroll
        for (int c = 0; c < D / 8; ++c) { kreg[c] = make_uint4(0u, 0u, 0u, 0u); *reinterpret_cast<uint4*>(Vs + tid * D + c * 8) = make_uint4(0u, 0u, 0u, 0u); }
    }
    __syncthreads();
    // scores S[r][tid]
#pragma unroll 4
    for (int r = 0; r < ROWS; ++r) {
        const uint4* qr = reinterpret_cast<const uint4*>(Qs + r * D);
        float acc = 0.f;
#pragma unroll
        for (int c = 0; c < D / 8; ++c) acc += dot8(qr[c], kreg[c]);
        const int i = r & 7, qpos = kend - qlen + i;
        const bool ok = valid && (i < qlen) && (!p.causal || kp <= qpos) && (p.window <= 0 || kp > qpos - p.window);
        S[r * CHUNK + tid] = ok ? acc * p.scale : NEG;
    }
    __syncthreads();
    // row stats + P (in place): warp w owns rows w*8 .. w*8+7; lane holds S[r][lane*4 .. +3]
    for (int rr = 0; rr < 8; ++rr) {
        const int r = warp * 8 + rr;
        float* srow = S + r * CHUNK;
        float v0 = srow[lane * 4], v1 = srow[lane * 4 + 1], v2 = srow[lane * 4 + 2], v3 = srow[lane * 4 + 3];
        float m = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        float l = 0.f;
        if (m > -1e29f) {
            v0 = __expf(v0 - m); v1 = __expf(v1 - m); v2 = __expf(v2 - m); v3 = __expf(v3 - m);
            l = v0 + v1 + v2 + v3;
        } else { v0 = v1 = v2 = v3 = 0.f; m = NEG; }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) l += __shfl_xor_sync(0xffffffffu, l, off);
        srow[lane * 4] = v0; srow[lane * 4 + 1] = v1; srow[lane * 4 + 2] = v2; srow[lane * 4 + 3] = v3;
        if (lane == 0) { ms[r] = m; ls[r] = l; }
    }
    __syncthreads();
    // O partial: thread = dim
    const int d = tid;
    for (int r = 0; r < ROWS; ++r) {
        const float* prow = S + r * CHUNK;
        float acc = 0.f;
#pragma unroll 8
        for (int t = 0; t < CHUNK; ++t) acc = fmaf(prow[t], __bfloat162float(Vs[t * D + d]), acc);
        po[r * D + d] = acc;
    }
    if (tid < ROWS) { p.part_m[pidx * ROWS + tid] = ms[tid]; p.part_l[pidx * ROWS + tid] = ls[tid]; }
}

// grid: (ROWS, HKV, B), block: 128 threads (= dims)
__global__ void __launch_bounds__(NTHREADS)
attn_combine_kernel(Params p) {
    const int r = blockIdx.x, g = blockIdx.y, b = blockIdx.z, d = threadIdx.x;
    const int q0 = p.cu_q[b], qlen = p.cu_q[b + 1] - q0;
    const int i = r & 7, j = r >> 3;
    if (i >= qlen) return;
    const size_t base = (size_t)(b * p.hkv + g) * p.max_splits;
    float M = NEG;
    for (int s = 0; s < p.max_splits; ++s) M = fmaxf(M, p.part_m[(base + s) * ROWS + r]);
    float L = 0.f, o = 0.f;
    if (M > -1e29f) {
        for (int s = 0; s < p.max_splits; ++s) {
            const float m = p.part_m[(base + s) * ROWS + r];
            if (m > -1e29f) {
                const float w = __expf(m - M);
                L = fmaf(p.part_l[(base + s) * ROWS + r], w, L);
                o = fmaf(p.part_o[((base + s) * ROWS + r) * D + d], w, o);
            }
        }
    }
    const float val = (L > 0.f) ? o / L : 0.f;
    p.out[(size_t)(q0 + i) * p.q_stride_t + (size_t)(g * GQA + j) * D + d] = __float2bfloat16(val);
}

extern "C" int dflash_attn_launch(const void* q, int q_stride_t, const void* kc, const void* vc, long long s_blk, long long s_pos, long long s_h, int bs, int hkv, int H,
                                  const void* block_table, int bt_stride, const void* seqused, const void* cu_q,
                                  float scale, int window, int causal, void* part_o, void* part_m, void* part_l, int max_splits,
                                  int B, void* out, cudaStream_t stream) {
    Params p;
    p.q = (const __nv_bfloat16*)q; p.q_stride_t = q_stride_t; p.kc = (const __nv_bfloat16*)kc; p.vc = (const __nv_bfloat16*)vc;
    p.s_blk = s_blk; p.s_pos = s_pos; p.s_h = s_h;
    p.bs = bs; p.hkv = hkv; p.H = H; p.block_table = (const int*)block_table; p.bt_stride = bt_stride;
    p.seqused = (const int*)seqused; p.cu_q = (const int*)cu_q; p.scale = scale; p.window = window; p.causal = causal;
    p.part_o = (float*)part_o; p.part_m = (float*)part_m; p.part_l = (float*)part_l; p.max_splits = max_splits;
    p.out = (__nv_bfloat16*)out;
    const size_t smem = ROWS * D * 2 + ROWS * CHUNK * 4 + CHUNK * D * 2 + 2 * ROWS * 4;
    static bool attr_set = false;
    if (!attr_set) {
        cudaGetLastError();
        if (cudaFuncSetAttribute((const void*)attn_partial_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) return 100;
        attr_set = true;
    }
    attn_partial_kernel<<<dim3(max_splits, hkv, B), NTHREADS, smem, stream>>>(p);
    cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return 200 + (int)e;
    attn_combine_kernel<<<dim3(ROWS, hkv, B), NTHREADS, 0, stream>>>(p);
    e = cudaGetLastError(); return e == cudaSuccess ? 0 : 300 + (int)e;
}

// ======================= tensor-core partial kernel (mma.sync m16n8k16 bf16) =======================
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__global__ void __launch_bounds__(NTHREADS)
attn_partial_mma_kernel(Params p) {
    // padded row strides (elements) so that rows g=0..7 of a fragment load land in distinct banks
    constexpr int LD = D + 8;            // bf16 rows: 272 B = 68 words -> bank offset 4 per row
    constexpr int LDK = CHUNK + 8;       // Vt rows (keys): 272 B
    constexpr int LDO = D + 4;           // fp32 rows: 528 B = 132 words -> bank offset 4 per row
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);                                   // [32][LD]
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem_raw + ROWS * LD * 2);                   // [128][LD]
    __nv_bfloat16* Vt = reinterpret_cast<__nv_bfloat16*>(smem_raw + (ROWS + CHUNK) * LD * 2);         // [128 dims][LDK]
    float* Oacc = reinterpret_cast<float*>(smem_raw + (ROWS + CHUNK) * LD * 2 + D * LDK * 2);         // [32][LDO]
    float* wm = Oacc + ROWS * LDO;                                                                     // [4][32]
    float* wl = wm + 4 * ROWS;                                                                    // [4][32]
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
    const int s = blockIdx.x, gk = blockIdx.y, b = blockIdx.z;
    const int q0 = p.cu_q[b], qlen = p.cu_q[b + 1] - q0, kend = p.seqused[b];
    int kbeg = 0;
    if (p.window > 0) { kbeg = kend - qlen - p.window + 1; if (kbeg < 0) kbeg = 0; }
    const int kbase = kbeg + s * CHUNK;
    const size_t pidx = ((size_t)(b * p.hkv + gk) * p.max_splits + s);
    float* po = p.part_o + pidx * ROWS * D;
    if (kbase >= kend || qlen <= 0) {
        if (tid < ROWS) { p.part_m[pidx * ROWS + tid] = NEG; p.part_l[pidx * ROWS + tid] = 0.f; }
        for (int i = tid; i < ROWS * D; i += NTHREADS) po[i] = 0.f;
        return;
    }
    // stage Q rows (r = j*8 + i)
    for (int idx = tid; idx < ROWS * (D / 8); idx += NTHREADS) {
        const int r = idx / (D / 8), c = idx - r * (D / 8), i = r & 7, j = r >> 3;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (i < qlen) v = *reinterpret_cast<const uint4*>(p.q + (size_t)(q0 + i) * p.q_stride_t + (size_t)(gk * GQA + j) * D + c * 8);
        *reinterpret_cast<uint4*>(Qs + r * LD + c * 8) = v;
    }
    // stage K row (natural) and V row (transposed) for key t = tid
    {
        const int kp = kbase + tid;
        uint4 kv[D / 8], vv[D / 8];
        if (kp < kend) {
            const int blk = p.block_table[(size_t)b * p.bt_stride + kp / p.bs], off = kp - (kp / p.bs) * p.bs;
            const long long rowoff = (long long)blk * p.s_blk + (long long)off * p.s_pos + (long long)gk * p.s_h;
            const uint4* kr = reinterpret_cast<const uint4*>(p.kc + rowoff);
            const uint4* vr = reinterpret_cast<const uint4*>(p.vc + rowoff);
#pragma unroll
            for (int c = 0; c < D / 8; ++c) { kv[c] = __ldg(kr + c); vv[c] = __ldg(vr + c); }
        } else {
#pragma unroll
            for (int c = 0; c < D / 8; ++c) { kv[c] = make_uint4(0u, 0u, 0u, 0u); vv[c] = make_uint4(0u, 0u, 0u, 0u); }
        }
#pragma unroll
        for (int c = 0; c < D / 8; ++c) *reinterpret_cast<uint4*>(Ks + tid * LD + c * 8) = kv[c];
#pragma unroll
        for (int c = 0; c < D / 8; ++c) {
            const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&vv[c]);
#pragma unroll
            for (int k = 0; k < 8; ++k) Vt[(c * 8 + k) * LDK + tid] = e[k];
        }
    }
    __syncthreads();
    // ---- S = Q K^T for this warp's 32 keys: sfrag[mt][nt][4] ----
    const int kw0 = warp * 32;
    float sfrag[2][4][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) sfrag[mt][nt][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
        uint32_t a[2][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const int r0 = mt * 16 + g;
            a[mt][0] = *reinterpret_cast<const uint32_t*>(Qs + r0 * LD + ks * 16 + 2 * q);
            a[mt][1] = *reinterpret_cast<const uint32_t*>(Qs + (r0 + 8) * LD + ks * 16 + 2 * q);
            a[mt][2] = *reinterpret_cast<const uint32_t*>(Qs + r0 * LD + ks * 16 + 8 + 2 * q);
            a[mt][3] = *reinterpret_cast<const uint32_t*>(Qs + (r0 + 8) * LD + ks * 16 + 8 + 2 * q);
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int key = kw0 + nt * 8 + g;
            uint32_t bb[2];
            bb[0] = *reinterpret_cast<const uint32_t*>(Ks + key * LD + ks * 16 + 2 * q);
            bb[1] = *reinterpret_cast<const uint32_t*>(Ks + key * LD + ks * 16 + 8 + 2 * q);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) mma16816(sfrag[mt][nt], a[mt], bb);
        }
    }
    // ---- scale + mask; per-row max over this warp's keys ----
    float rmax[2][2];   // [mt][half]: rows mt*16+g, mt*16+g+8
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) { rmax[mt][0] = NEG; rmax[mt][1] = NEG; }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int R = mt * 16 + g + 8 * (e >> 1), kk = kw0 + nt * 8 + 2 * q + (e & 1);
                const int i = R & 7, qpos = kend - qlen + i, kp = kbase + kk;
                const bool ok = (kp < kend) && (i < qlen) && (!p.causal || kp <= qpos) && (p.window <= 0 || kp > qpos - p.window);
                const float v = ok ? sfrag[mt][nt][e] * p.scale : NEG;
                sfrag[mt][nt][e] = v;
                rmax[mt][e >> 1] = fmaxf(rmax[mt][e >> 1], v);
            }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = rmax[mt][h];
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            rmax[mt][h] = m;
            if (q == 0) wm[warp * ROWS + mt * 16 + g + 8 * h] = m;
        }
    __syncthreads();
    float mrow[2][2], lsum[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int R = mt * 16 + g + 8 * h;
            float m = wm[R]; m = fmaxf(m, wm[ROWS + R]); m = fmaxf(m, wm[2 * ROWS + R]); m = fmaxf(m, wm[3 * ROWS + R]);
            mrow[mt][h] = m; lsum[mt][h] = 0.f;
        }
    // p = exp(s - m) (0 where masked or the row is fully masked), partial row sums
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int h = e >> 1;
                const float m = mrow[mt][h];
                float pv = 0.f;
                if (m > -1e29f && sfrag[mt][nt][e] > -1e29f) pv = __expf(sfrag[mt][nt][e] - m);
                sfrag[mt][nt][e] = pv;
                lsum[mt][h] += pv;
            }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float l = lsum[mt][h];
            l += __shfl_xor_sync(0xffffffffu, l, 1);
            l += __shfl_xor_sync(0xffffffffu, l, 2);
            if (q == 0) wl[warp * ROWS + mt * 16 + g + 8 * h] = l;
        }
    // ---- O_w = P_w V_w : ofrag[mt][nt(16)][4] ----
    uint32_t pa[2][2][4];   // [mt][ks2] A fragments from P (bf16)
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int ks2 = 0; ks2 < 2; ++ks2) {
            pa[mt][ks2][0] = pack_bf16x2(sfrag[mt][2 * ks2][0], sfrag[mt][2 * ks2][1]);
            pa[mt][ks2][1] = pack_bf16x2(sfrag[mt][2 * ks2][2], sfrag[mt][2 * ks2][3]);
            pa[mt][ks2][2] = pack_bf16x2(sfrag[mt][2 * ks2 + 1][0], sfrag[mt][2 * ks2 + 1][1]);
            pa[mt][ks2][3] = pack_bf16x2(sfrag[mt][2 * ks2 + 1][2], sfrag[mt][2 * ks2 + 1][3]);
        }
    float ofrag[2][16][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 16; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) ofrag[mt][nt][e] = 0.f;
#pragma unroll
    for (int ks2 = 0; ks2 < 2; ++ks2) {
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const int dim = nt * 8 + g;
            uint32_t bb[2];
            bb[0] = *reinterpret_cast<const uint32_t*>(Vt + dim * LDK + kw0 + ks2 * 16 + 2 * q);
            bb[1] = *reinterpret_cast<const uint32_t*>(Vt + dim * LDK + kw0 + ks2 * 16 + 8 + 2 * q);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) mma16816(ofrag[mt][nt], pa[mt][ks2], bb);
        }
    }
    // ---- deterministic cross-warp sum of O into Oacc ----
#pragma unroll 1
    for (int w = 0; w < 4; ++w) {
        if (warp == w) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt)
#pragma unroll
                for (int nt = 0; nt < 16; ++nt)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int R = mt * 16 + g + 8 * (e >> 1), C = nt * 8 + 2 * q + (e & 1);
                        if (w == 0) Oacc[R * LDO + C] = ofrag[mt][nt][e];
                        else Oacc[R * LDO + C] += ofrag[mt][nt][e];
                    }
        }
        __syncthreads();
    }
    for (int i = tid; i < ROWS * D; i += NTHREADS) po[i] = Oacc[(i / D) * LDO + (i % D)];
    if (tid < ROWS) {
        float m = wm[tid]; m = fmaxf(m, wm[ROWS + tid]); m = fmaxf(m, wm[2 * ROWS + tid]); m = fmaxf(m, wm[3 * ROWS + tid]);
        const float l = wl[tid] + wl[ROWS + tid] + wl[2 * ROWS + tid] + wl[3 * ROWS + tid];
        p.part_m[pidx * ROWS + tid] = m; p.part_l[pidx * ROWS + tid] = (m > -1e29f) ? l : 0.f;
    }
}

extern "C" int dflash_attn_launch_mma(const void* q, int q_stride_t, const void* kc, const void* vc, long long s_blk, long long s_pos, long long s_h, int bs, int hkv, int H,
                                      const void* block_table, int bt_stride, const void* seqused, const void* cu_q,
                                      float scale, int window, int causal, void* part_o, void* part_m, void* part_l, int max_splits,
                                      int B, void* out, cudaStream_t stream) {
    Params p;
    p.q = (const __nv_bfloat16*)q; p.q_stride_t = q_stride_t; p.kc = (const __nv_bfloat16*)kc; p.vc = (const __nv_bfloat16*)vc;
    p.s_blk = s_blk; p.s_pos = s_pos; p.s_h = s_h;
    p.bs = bs; p.hkv = hkv; p.H = H; p.block_table = (const int*)block_table; p.bt_stride = bt_stride;
    p.seqused = (const int*)seqused; p.cu_q = (const int*)cu_q; p.scale = scale; p.window = window; p.causal = causal;
    p.part_o = (float*)part_o; p.part_m = (float*)part_m; p.part_l = (float*)part_l; p.max_splits = max_splits;
    p.out = (__nv_bfloat16*)out;
    const size_t smem = (size_t)(ROWS + CHUNK) * (D + 8) * 2 + (size_t)D * (CHUNK + 8) * 2 + (size_t)ROWS * (D + 4) * 4 + 2 * 4 * ROWS * 4;
    static bool attr_set = false;
    if (!attr_set) {
        cudaGetLastError();
        if (cudaFuncSetAttribute((const void*)attn_partial_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem) != cudaSuccess) return 100;
        attr_set = true;
    }
    attn_partial_mma_kernel<<<dim3(max_splits, hkv, B), NTHREADS, smem, stream>>>(p);
    cudaError_t e = cudaGetLastError(); if (e != cudaSuccess) return 200 + (int)e;
    attn_combine_kernel<<<dim3(ROWS, hkv, B), NTHREADS, 0, stream>>>(p);
    e = cudaGetLastError(); return e == cudaSuccess ? 0 : 300 + (int)e;
}
