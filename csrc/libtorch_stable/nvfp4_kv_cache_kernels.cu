// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// NVFP4 KV cache store kernel.
// Quantizes bf16 key/value to packed FP4 + FP8 block scales and writes them
// into the paged KV cache.
//
// Per page layout: [K_data | K_scale | V_data | V_scale]
// Both data and scale regions are contiguous per head, enabling direct
// TMA descriptor use.
//
// Reuses device functions from nvfp4_utils.cuh:
//   - cvt_warp_fp16_to_fp4()  for bf16 → fp4 quantization + block scale
//   - pack_fp4()              for packing float pairs to fp4
//   - reciprocal_approximate_ftz() for fast reciprocal

#define NVFP4_ENABLE_ELTS16 1
#include "libtorch_stable/quantization/fp4/nvfp4_utils.cuh"

#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/torch_utils.h"
#include "libtorch_stable/type_convert.cuh"

#include <cmath>
#include <string>

namespace vllm {

// The cache layout is identical; this only selects the store-time scale search.
enum class NVFP4KVScaleSearch {
  DEFAULT,
  FOUR_OVER_SIX,
};

// Compute swizzled scale offset for SM100 trtllm-gen MHA kernel.
// The swizzle pattern for HND layout is:
//   [T//4, 4, 4, S//4] → permute(0, 2, 3, 1) → reshape to [T, S]
// where T = block_size (page_size), S = scale_dim = head_size // 16.
//
// For a linear (t, s) position, the swizzled position is:
//   swizzled_t = (t / 4) * 4 + (s / (S / 4))
//   swizzled_s = (s % (S / 4)) * 4 + (t % 4)
__device__ __forceinline__ int swizzle_scale_offset(int t, int s,
                                                    int scale_dim) {
  int s_group = scale_dim / 4;
  int swizzled_t = (t / 4) * 4 + (s / s_group);
  int swizzled_s = (s % s_group) * 4 + (t % 4);
  return swizzled_t * scale_dim + swizzled_s;
}

__device__ __forceinline__ float round_to_nearest_e2m1(float x) {
  const float ax = fabsf(x);
  float q;
  // Match cvt.rn.satfinite.e2m1x2.f32, including ties-to-even boundaries.
  if (ax <= 0.25f) {
    q = 0.0f;
  } else if (ax < 0.75f) {
    q = 0.5f;
  } else if (ax <= 1.25f) {
    q = 1.0f;
  } else if (ax < 1.75f) {
    q = 1.5f;
  } else if (ax <= 2.5f) {
    q = 2.0f;
  } else if (ax < 3.5f) {
    q = 3.0f;
  } else if (ax <= 5.0f) {
    q = 4.0f;
  } else {
    q = 6.0f;
  }
  return copysignf(q, x);
}

__device__ __forceinline__ void nvfp4_candidate_scale(float SFScaleVal,
                                                      float vecMax,
                                                      float denominator,
                                                      uint8_t* fp8_sf,
                                                      float* outputScale) {
  float SFValue =
      SFScaleVal * (vecMax * reciprocal_approximate_ftz(denominator));
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  reinterpret_cast<__nv_fp8_e4m3&>(*fp8_sf) = tmp;
  SFValue = float(tmp);
  *outputScale = SFValue != 0.0f
                     ? reciprocal_approximate_ftz(
                           SFValue * reciprocal_approximate_ftz(SFScaleVal))
                     : 0.0f;
}

template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ float nvfp4_reconstruction_error(
    PackedVec<Type, CVT_FP4_PACK16>& vec, float outputScale) {
  float error = 0.0f;
  const float dequantScale =
      outputScale != 0.0f ? reciprocal_approximate_ftz(outputScale) : 0.0f;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    float2 vals = cast_to_float2(vec.elts[i]);
    if (outputScale == 0.0f) {
      error += vals.x * vals.x + vals.y * vals.y;
    } else {
      float qx = round_to_nearest_e2m1(vals.x * outputScale);
      float qy = round_to_nearest_e2m1(vals.y * outputScale);
      float dx = qx * dequantScale - vals.x;
      float dy = qy * dequantScale - vals.y;
      error += dx * dx + dy * dy;
    }
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    error += __shfl_xor_sync(0xffffffffu, error, 1);
  }
  return error;
}

template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ fp4_packed_t cvt_warp_fp16_to_fp4_4over6(
    PackedVec<Type, CVT_FP4_PACK16>& vec, float SFScaleVal, uint8_t* SFout) {
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }
  const float vecMax = float(__hmax(localMax.x, localMax.y));

  uint8_t sf6;
  uint8_t sf4;
  float outputScale6;
  float outputScale4;
  nvfp4_candidate_scale(SFScaleVal, vecMax, 6.0f, &sf6, &outputScale6);
  nvfp4_candidate_scale(SFScaleVal, vecMax, 4.0f, &sf4, &outputScale4);

  const float err6 =
      nvfp4_reconstruction_error<Type, CVT_FP4_NUM_THREADS_PER_SF>(
          vec, outputScale6);
  const float err4 =
      nvfp4_reconstruction_error<Type, CVT_FP4_NUM_THREADS_PER_SF>(
          vec, outputScale4);
  const bool use4 = err4 < err6;
  const float outputScale = use4 ? outputScale4 : outputScale6;

  if (SFout) *SFout = use4 ? sf4 : sf6;

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = cast_to_float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  return pack_fp4(fp2Vals);
}

// Kernel: quantize bf16 key/value to NVFP4 and store in paged KV cache.
//
// Takes separate data and scale cache pointers for K and V.
// Within each KV side, data and scale are separate contiguous regions.
//
// Threading: one CUDA block per token, threads process heads and
// groups of 16 elements within each head.
template <typename scalar_t, NVFP4KVScaleSearch SCALE_SEARCH>
__global__ void reshape_and_cache_nvfp4_kernel(
    const scalar_t* __restrict__ key,      // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,    // [num_tokens, num_heads, head_size]
    uint8_t* __restrict__ key_data_cache,  // data region for K
    uint8_t* __restrict__ value_data_cache,    // data region for V
    uint8_t* __restrict__ key_scale_cache,     // scale region for K
    uint8_t* __restrict__ value_scale_cache,   // scale region for V
    const int64_t* __restrict__ slot_mapping,  // [num_actual_tokens]
    const float* __restrict__ k_scale_ptr,     // pointer to checkpoint k_scale
    const float* __restrict__ v_scale_ptr,     // pointer to checkpoint v_scale
    const int64_t key_stride,                  // key.stride(0) in elements
    const int64_t value_stride,                // value.stride(0) in elements
    const int num_heads, const int head_size, const int block_size,
    const int64_t data_block_stride,         // data cache stride for dim 0
    const int64_t data_head_stride,          // data cache stride for heads
    const int64_t data_block_offset_stride,  // data cache stride for tokens
    const int64_t scale_block_stride,        // scale cache stride for dim 0
    const int64_t scale_head_stride,         // scale cache stride for heads
    const int64_t scale_block_offset_stride  // scale cache stride for tokens
) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static constexpr int ELTS = CVT_FP4_ELTS_PER_THREAD;  // 16 or 8
  static constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / ELTS;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) return;

  const int64_t block_idx = slot_idx / block_size;
  const int block_offset = static_cast<int>(slot_idx % block_size);

  const int scale_dim = head_size / 16;
  const int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;

  const int total_groups = num_heads * groups_per_head;
  const int tid = threadIdx.x;
  const int num_thread_groups = blockDim.x / THREADS_PER_SF;
  const int tg_id = tid / THREADS_PER_SF;
  const int tg_lane = tid % THREADS_PER_SF;

  // Process both K (kv=0) and V (kv=1)
#pragma unroll
  for (int kv = 0; kv < 2; kv++) {
    const scalar_t* __restrict__ src = (kv == 0) ? key : value;
    const float global_scale = 1.0f / ((kv == 0) ? *k_scale_ptr : *v_scale_ptr);
    const int64_t src_stride = (kv == 0) ? key_stride : value_stride;
    uint8_t* __restrict__ data_cache =
        (kv == 0) ? key_data_cache : value_data_cache;
    uint8_t* __restrict__ sc_cache =
        (kv == 0) ? key_scale_cache : value_scale_cache;

    // Source pointer for this token (use actual stride, not assumed contiguous)
    const CudaType* __restrict__ token_src =
        reinterpret_cast<const CudaType*>(src) + token_idx * src_stride;

    // Destination bases in data and scale caches for this token's block
    uint8_t* __restrict__ data_block =
        data_cache + block_idx * data_block_stride;
    uint8_t* __restrict__ scale_block =
        sc_cache + block_idx * scale_block_stride;

    for (int g = tg_id; g < total_groups; g += num_thread_groups) {
      const int head = g / groups_per_head;
      const int group_in_head = g % groups_per_head;

      // Load 16 (or 8) bf16 elements from source
      PVec in_vec;
      const CudaType* __restrict__ src_ptr =
          token_src + head * head_size + group_in_head * CVT_FP4_SF_VEC_SIZE +
          tg_lane * ELTS;

#pragma unroll
      for (int i = 0; i < ELTS / 2; i++) {
        in_vec.elts[i] = reinterpret_cast<
            const typename PackedTypeConverter<CudaType>::Type*>(src_ptr)[i];
      }

      // Quantize: produces packed fp4 and writes scale factor.
      uint8_t sf_val;
      uint8_t* sf_out_ptr = (tg_lane == 0) ? &sf_val : nullptr;

      fp4_packed_t packed;
      if constexpr (SCALE_SEARCH == NVFP4KVScaleSearch::FOUR_OVER_SIX) {
        packed = cvt_warp_fp16_to_fp4_4over6<CudaType, THREADS_PER_SF>(
            in_vec, global_scale, sf_out_ptr);
      } else {
        packed = cvt_warp_fp16_to_fp4<CudaType, THREADS_PER_SF>(
            in_vec, global_scale, sf_out_ptr);
      }

      // Write packed FP4 data to data cache
      uint8_t* __restrict__ data_dst = data_block + head * data_head_stride +
                                       block_offset * data_block_offset_stride;

#if CVT_FP4_PACK16
      {
        // 16 elements → 8 bytes (u32x2)
        int data_byte_offset = group_in_head * 8;
        reinterpret_cast<uint64_t*>(data_dst + data_byte_offset)[0] =
            (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
      }
#else
      {
        // 8 elements → 4 bytes (uint32_t)
        int data_byte_offset =
            group_in_head * CVT_FP4_SF_VEC_SIZE / 2 + tg_lane * ELTS / 2;
        reinterpret_cast<uint32_t*>(data_dst + data_byte_offset)[0] = packed;
      }
#endif

      // Write block scale to scale cache.
      // K (kv==0): linear layout (no swizzle).
      // V (kv==1): swizzled layout for SM100 trtllm-gen MHA kernel.
      if (sf_out_ptr != nullptr) {
        int scale_idx = group_in_head;
        uint8_t* __restrict__ scale_dst;
        if (kv == 0) {
          scale_dst = scale_block + head * scale_head_stride +
                      block_offset * scale_block_offset_stride + scale_idx;
        } else {
          int swizzled_offset =
              swizzle_scale_offset(block_offset, scale_idx, scale_dim);
          int swizzled_t = swizzled_offset / scale_dim;
          int swizzled_s = swizzled_offset % scale_dim;
          scale_dst = scale_block + head * scale_head_stride +
                      swizzled_t * scale_block_offset_stride + swizzled_s;
        }
        *scale_dst = sf_val;
      }
    }
  }
}

template <typename scalar_t>
__global__ void concat_and_cache_nvfp4_mla_kernel(
    const scalar_t* __restrict__ kv_c,         // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,         // [num_tokens, pe_dim]
    uint8_t* __restrict__ kv_cache,            // [num_blocks, block_size, 432]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride, const int entry_stride, const int kv_c_stride,
    const int k_pe_stride, const int kv_lora_rank, const int pe_dim,
    const int block_size) {
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static constexpr int kNopeBytes = 256;
  static constexpr int kScaleBytes = 32;
  static constexpr int kPadBytes = 16;
  static constexpr int kRopeOffset = kNopeBytes + kScaleBytes + kPadBytes;
  static constexpr int kFp4GroupSize = CVT_FP4_SF_VEC_SIZE;
  static constexpr int kEltsPerThread = CVT_FP4_ELTS_PER_THREAD;
  static constexpr int kThreadsPerScale = kFp4GroupSize / kEltsPerThread;

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  uint8_t* __restrict__ token_dst =
      kv_cache + block_idx * block_stride + block_offset * entry_stride;

  const CudaType* __restrict__ token_src =
      reinterpret_cast<const CudaType*>(kv_c) + token_idx * kv_c_stride;

  const int group_count = kv_lora_rank / kFp4GroupSize;
  const int thread_group_count = blockDim.x / kThreadsPerScale;
  const int thread_group = threadIdx.x / kThreadsPerScale;
  const int thread_group_lane = threadIdx.x % kThreadsPerScale;

  for (int group = thread_group; group < group_count;
       group += thread_group_count) {
    PVec in_vec;
    const CudaType* __restrict__ src =
        token_src + group * kFp4GroupSize + thread_group_lane * kEltsPerThread;

#pragma unroll
    for (int i = 0; i < kEltsPerThread / 2; ++i) {
      in_vec.elts[i] =
          reinterpret_cast<const typename PackedTypeConverter<CudaType>::Type*>(
              src)[i];
    }

    uint8_t scale_byte;
    uint8_t* scale_out = (thread_group_lane == 0) ? &scale_byte : nullptr;
    fp4_packed_t packed = cvt_warp_fp16_to_fp4<CudaType, kThreadsPerScale>(
        in_vec, 1.0f, scale_out);

#if CVT_FP4_PACK16
    uint8_t* data_dst = token_dst + group * 8;
    reinterpret_cast<uint64_t*>(data_dst)[0] =
        (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
#else
    uint8_t* data_dst = token_dst + group * 8 + thread_group_lane * 4;
    reinterpret_cast<uint32_t*>(data_dst)[0] = packed;
#endif

    if (scale_out != nullptr) {
      token_dst[kNopeBytes + group] = scale_byte;
    }
  }

  for (int i = threadIdx.x; i < kPadBytes; i += blockDim.x) {
    token_dst[kNopeBytes + kScaleBytes + i] = 0;
  }

  scalar_t* __restrict__ rope_dst =
      reinterpret_cast<scalar_t*>(token_dst + kRopeOffset);
  const scalar_t* __restrict__ rope_src = k_pe + token_idx * k_pe_stride;
  for (int i = threadIdx.x; i < pe_dim; i += blockDim.x) {
    rope_dst[i] = rope_src[i];
  }
}

template <typename scalar_t>
__global__ void fused_deepseek_v4_qnorm_rope_nvfp4_mla_kernel(
    scalar_t* __restrict__ q, const scalar_t* __restrict__ kv,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ position_ids,
    const float* __restrict__ cos_sin_cache, const float eps,
    const int num_tokens, const int num_insert_tokens, const int num_heads,
    const int block_size, const int64_t block_stride,
    const int64_t token_stride, const int64_t cos_sin_rows) {
  using Converter = vllm::_typeConvert<scalar_t>;
  using CudaType = typename CUDATypeConverter<scalar_t>::Type;
  using PVec = PackedVec<CudaType, CVT_FP4_PACK16>;

  static_assert(CVT_FP4_PACK16,
                "the 432-byte NVFP4 record requires CVT_FP4_PACK16");

  constexpr int kHeadDim = 512;
  constexpr int kNopeDim = 448;
  constexpr int kRopeDim = 64;
  constexpr int kElemsPerLane = 16;
  constexpr int kNopeBytes = 256;
  constexpr int kScaleBytes = 32;
  constexpr int kPadBytes = 16;
  constexpr int kRopeOffset = kNopeBytes + kScaleBytes + kPadBytes;

  const int warps_per_block = blockDim.x / 32;
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int global_warp = blockIdx.x * warps_per_block + warp_id;
  const int slots_per_token = num_heads + 1;
  const int token_idx = global_warp / slots_per_token;
  const int slot_idx = global_warp % slots_per_token;
  if (token_idx >= num_tokens) return;

  const bool is_kv = slot_idx == num_heads;
  if (is_kv && token_idx >= num_insert_tokens) return;

  const int dim_base = lane_id * kElemsPerLane;
  const scalar_t* src =
      is_kv ? kv + static_cast<int64_t>(token_idx) * kHeadDim + dim_base
            : q +
                  (static_cast<int64_t>(token_idx) * num_heads + slot_idx) *
                      kHeadDim +
                  dim_base;
  const uint4 v0 = *reinterpret_cast<const uint4*>(src);
  const uint4 v1 = *reinterpret_cast<const uint4*>(src + 8);
  const auto* p0 =
      reinterpret_cast<const typename Converter::packed_hip_type*>(&v0);
  const auto* p1 =
      reinterpret_cast<const typename Converter::packed_hip_type*>(&v1);
  float elements[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 values = Converter::convert(p0[i]);
    elements[2 * i] = values.x;
    elements[2 * i + 1] = values.y;
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 values = Converter::convert(p1[i]);
    elements[8 + 2 * i] = values.x;
    elements[8 + 2 * i + 1] = values.y;
  }

  if (!is_kv) {
    float sum_of_squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      sum_of_squares += elements[i] * elements[i];
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      sum_of_squares += __shfl_xor_sync(0xffffffffu, sum_of_squares, mask, 32);
    }
    const float rms_rcp = rsqrtf(sum_of_squares / kHeadDim + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      elements[i] *= rms_rcp;
    }
  }

  if (dim_base >= kNopeDim) {
    constexpr int kHalfRope = kRopeDim / 2;
    const int64_t position = position_ids[token_idx];
    if (position < 0 || position >= cos_sin_rows) {
      asm volatile("trap;");
    }
    const float* cos_ptr = cos_sin_cache + position * kRopeDim;
    const float* sin_ptr = cos_ptr + kHalfRope;
    const int half_base = (dim_base - kNopeDim) >> 1;
    const float4 c0 = *reinterpret_cast<const float4*>(cos_ptr + half_base);
    const float4 c1 = *reinterpret_cast<const float4*>(cos_ptr + half_base + 4);
    const float4 s0 = *reinterpret_cast<const float4*>(sin_ptr + half_base);
    const float4 s1 = *reinterpret_cast<const float4*>(sin_ptr + half_base + 4);
    const float cos_values[8] = {c0.x, c0.y, c0.z, c0.w,
                                 c1.x, c1.y, c1.z, c1.w};
    const float sin_values[8] = {s0.x, s0.y, s0.z, s0.w,
                                 s1.x, s1.y, s1.z, s1.w};
#pragma unroll
    for (int pair = 0; pair < kElemsPerLane / 2; ++pair) {
      const float even = elements[2 * pair];
      const float odd = elements[2 * pair + 1];
      elements[2 * pair] = even * cos_values[pair] - odd * sin_values[pair];
      elements[2 * pair + 1] = even * sin_values[pair] + odd * cos_values[pair];
    }
  }

  uint4 out0;
  uint4 out1;
  auto* out_pairs0 =
      reinterpret_cast<typename Converter::packed_hip_type*>(&out0);
  auto* out_pairs1 =
      reinterpret_cast<typename Converter::packed_hip_type*>(&out1);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    out_pairs0[i] =
        Converter::convert(make_float2(elements[2 * i], elements[2 * i + 1]));
    out_pairs1[i] = Converter::convert(
        make_float2(elements[8 + 2 * i], elements[8 + 2 * i + 1]));
  }

  if (!is_kv) {
    scalar_t* dst =
        q +
        (static_cast<int64_t>(token_idx) * num_heads + slot_idx) * kHeadDim +
        dim_base;
    *reinterpret_cast<uint4*>(dst) = out0;
    *reinterpret_cast<uint4*>(dst + 8) = out1;
    return;
  }

  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) return;
  const int64_t block_idx = slot / block_size;
  const int64_t block_offset = slot % block_size;
  uint8_t* token_dst =
      kv_cache + block_idx * block_stride + block_offset * token_stride;

  PVec quant_input;
  auto* quant_pairs =
      reinterpret_cast<typename Converter::packed_hip_type*>(&quant_input);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    quant_pairs[i] = out_pairs0[i];
    quant_pairs[4 + i] = out_pairs1[i];
  }
  uint8_t scale;
  const fp4_packed_t packed =
      cvt_warp_fp16_to_fp4<CudaType, 1>(quant_input, 1.0f, &scale);
#if CVT_FP4_PACK16
  reinterpret_cast<uint64_t*>(token_dst + lane_id * 8)[0] =
      (uint64_t(packed.hi) << 32) | uint64_t(packed.lo);
#else
  reinterpret_cast<uint32_t*>(token_dst + lane_id * 4)[0] = packed;
#endif
  token_dst[kNopeBytes + lane_id] = scale;
  if (lane_id < kPadBytes) {
    token_dst[kNopeBytes + kScaleBytes + lane_id] = 0;
  }
  if (dim_base >= kNopeDim) {
    uint8_t* rope_dst =
        token_dst + kRopeOffset + (dim_base - kNopeDim) * sizeof(scalar_t);
    *reinterpret_cast<uint4*>(rope_dst) = out0;
    *reinterpret_cast<uint4*>(rope_dst + sizeof(uint4)) = out1;
  }
}

}  // namespace vllm

// Non-template entry point callable from cache_kernels.cu.
// Receives key_cache/value_cache as kv_cache[:, 0] and kv_cache[:, 1].
// Each KV side contains both data and scale:
//   page = [K_data | K_scale | V_data | V_scale]
void reshape_and_cache_nvfp4_dispatch(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& key_cache, torch::stable::Tensor& value_cache,
    torch::stable::Tensor& slot_mapping, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale, const std::string& kv_cache_dtype) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int data_dim = head_size / 2;
  int scale_dim = head_size / 16;
  int full_dim = data_dim + scale_dim;

  // key_cache is kv_cache[:, 0] with shape
  // [num_blocks, block_size, num_heads, full_dim] in logical order.
  // Strides encode the physical layout (HND or NHD).
  STD_TORCH_CHECK(key_cache.dim() == 4, "key_cache must be 4D");
  STD_TORCH_CHECK(key_cache.size(3) == full_dim,
                  "key_cache last dim must be data_dim + scale_dim, got ",
                  key_cache.size(3), " expected ", full_dim);

  int block_size = key_cache.size(1);

  STD_TORCH_CHECK(head_size % 16 == 0,
                  "head_size must be divisible by 16 for NVFP4 KV cache");
  STD_TORCH_CHECK(block_size % 4 == 0,
                  "block_size must be divisible by 4 for NVFP4 KV cache "
                  "swizzle");

  // Detect physical layout from strides (based on full_dim).
  // HND: head stride > block_offset stride.
  bool is_hnd = key_cache.stride(2) > key_cache.stride(1);

  int64_t data_block_stride = key_cache.stride(0);  // page_bytes
  int64_t data_head_stride, data_block_offset_stride;
  if (is_hnd) {
    data_head_stride = (int64_t)block_size * data_dim;
    data_block_offset_stride = data_dim;
  } else {
    data_head_stride = data_dim;
    data_block_offset_stride = (int64_t)num_heads * data_dim;
  }

  // Page layout: [K_data | K_scale | V_data | V_scale]
  // Scale follows data within each KV side.
  int64_t data_per_kv = (int64_t)num_heads * block_size * data_dim;

  uint8_t* key_scale_ptr = key_cache.mutable_data_ptr<uint8_t>() + data_per_kv;
  uint8_t* value_scale_ptr =
      value_cache.mutable_data_ptr<uint8_t>() + data_per_kv;

  // Scale strides: same page stride, inner strides from layout.
  int64_t scale_block_stride = data_block_stride;
  int64_t scale_head_stride, scale_block_offset_stride;
  if (is_hnd) {
    scale_head_stride = (int64_t)block_size * scale_dim;
    scale_block_offset_stride = scale_dim;
  } else {
    scale_head_stride = scale_dim;
    scale_block_offset_stride = (int64_t)num_heads * scale_dim;
  }

  const float* k_scale_ptr = k_scale.const_data_ptr<float>();
  const float* v_scale_ptr = v_scale.const_data_ptr<float>();

  int groups_per_head = head_size / CVT_FP4_SF_VEC_SIZE;
  int total_groups = num_heads * groups_per_head;
  constexpr int THREADS_PER_SF = CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;
  int num_threads = std::min(total_groups * THREADS_PER_SF, 512);
  num_threads = ((num_threads + 31) / 32) * 32;

  dim3 grid(num_tokens);
  dim3 block(num_threads);

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      key.scalar_type(), "reshape_and_cache_nvfp4", [&] {
        if (kv_cache_dtype == "nvfp4") {
          vllm::reshape_and_cache_nvfp4_kernel<
              scalar_t, vllm::NVFP4KVScaleSearch::DEFAULT>
              <<<grid, block, 0, stream>>>(
                  key.const_data_ptr<scalar_t>(),
                  value.const_data_ptr<scalar_t>(),
                  key_cache.mutable_data_ptr<uint8_t>(),
                  value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                  value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                  k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                  num_heads, head_size, block_size, data_block_stride,
                  data_head_stride, data_block_offset_stride,
                  scale_block_stride, scale_head_stride,
                  scale_block_offset_stride);
        } else if (kv_cache_dtype == "nvfp4_4over6") {
          vllm::reshape_and_cache_nvfp4_kernel<
              scalar_t, vllm::NVFP4KVScaleSearch::FOUR_OVER_SIX>
              <<<grid, block, 0, stream>>>(
                  key.const_data_ptr<scalar_t>(),
                  value.const_data_ptr<scalar_t>(),
                  key_cache.mutable_data_ptr<uint8_t>(),
                  value_cache.mutable_data_ptr<uint8_t>(), key_scale_ptr,
                  value_scale_ptr, slot_mapping.const_data_ptr<int64_t>(),
                  k_scale_ptr, v_scale_ptr, key.stride(0), value.stride(0),
                  num_heads, head_size, block_size, data_block_stride,
                  data_head_stride, data_block_offset_stride,
                  scale_block_stride, scale_head_stride,
                  scale_block_offset_stride);
        } else {
          STD_TORCH_CHECK(false,
                          "Unsupported NVFP4 KV cache dtype: ", kv_cache_dtype);
        }
      });
}

void concat_and_cache_nvfp4_mla_dispatch(torch::stable::Tensor& kv_c,
                                         torch::stable::Tensor& k_pe,
                                         torch::stable::Tensor& kv_cache,
                                         torch::stable::Tensor& slot_mapping) {
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);
  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const torch::stable::accelerator::DeviceGuard device_guard(
      kv_c.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  dim3 grid(num_tokens);
  dim3 block(128);
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      kv_c.scalar_type(), "concat_and_cache_nvfp4_mla", [&] {
        vllm::concat_and_cache_nvfp4_mla_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<scalar_t*>(kv_c.data_ptr()),
                reinterpret_cast<scalar_t*>(k_pe.data_ptr()),
                reinterpret_cast<uint8_t*>(kv_cache.data_ptr()),
                slot_mapping.const_data_ptr<int64_t>(), block_stride,
                entry_stride, kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim,
                block_size);
      });
}

void fused_deepseek_v4_qnorm_rope_nvfp4_mla(
    torch::stable::Tensor& q, torch::stable::Tensor const& kv,
    torch::stable::Tensor& kv_cache, torch::stable::Tensor const& slot_mapping,
    torch::stable::Tensor const& position_ids,
    torch::stable::Tensor const& cos_sin_cache, double eps,
    int64_t cache_block_size) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(q.device().is_cuda() && q.is_contiguous(),
                  "q must be contiguous CUDA");
  STD_TORCH_CHECK(kv.device().is_cuda() && kv.is_contiguous(),
                  "kv must be contiguous CUDA");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16 &&
                      kv.scalar_type() == ScalarType::BFloat16,
                  "q and kv must be bfloat16");
  STD_TORCH_CHECK(q.dim() == 3 && q.size(2) == 512,
                  "q must have shape [N, H, 512]");
  STD_TORCH_CHECK(kv.dim() == 2 && kv.size(0) == q.size(0) && kv.size(1) == 512,
                  "kv must have shape [N, 512]");
  STD_TORCH_CHECK(kv_cache.device().is_cuda() &&
                      kv_cache.scalar_type() == ScalarType::Byte &&
                      kv_cache.dim() == 3 &&
                      kv_cache.size(1) == cache_block_size &&
                      kv_cache.size(2) == 432 && kv_cache.stride(2) == 1,
                  "kv_cache must be uint8 [num_blocks, block_size, 432]");
  STD_TORCH_CHECK(kv_cache.stride(1) == 432,
                  "kv_cache token stride must be 432");
  STD_TORCH_CHECK(slot_mapping.device().is_cuda() &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64 CUDA");
  STD_TORCH_CHECK(position_ids.device().is_cuda() &&
                      position_ids.scalar_type() == ScalarType::Long &&
                      position_ids.size(0) == q.size(0),
                  "position_ids must be int64 CUDA with N entries");
  STD_TORCH_CHECK(cos_sin_cache.device().is_cuda() &&
                      cos_sin_cache.scalar_type() == ScalarType::Float &&
                      cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == 64,
                  "cos_sin_cache must have shape [max_pos, 64] float32");
  STD_TORCH_CHECK(slot_mapping.size(0) <= q.size(0),
                  "slot_mapping cannot have more rows than q");
  STD_TORCH_CHECK(
      get_device_prop()->major >= 10,
      "direct DeepSeek-V4 NVFP4 writes require a supported CUDA device");

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_insert_tokens = static_cast<int>(slot_mapping.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  constexpr int kThreads = 256;
  constexpr int kWarps = kThreads / 32;
  const int total_warps = num_tokens * (num_heads + 1);
  const int grid = (total_warps + kWarps - 1) / kWarps;
  const torch::stable::accelerator::DeviceGuard device_guard(
      q.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(q.get_device_index());

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      q.scalar_type(), "fused_deepseek_v4_qnorm_rope_nvfp4_mla", [&] {
        vllm::fused_deepseek_v4_qnorm_rope_nvfp4_mla_kernel<scalar_t>
            <<<grid, kThreads, 0, stream>>>(
                reinterpret_cast<scalar_t*>(q.mutable_data_ptr()),
                reinterpret_cast<const scalar_t*>(kv.const_data_ptr()),
                reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
                slot_mapping.const_data_ptr<int64_t>(),
                position_ids.const_data_ptr<int64_t>(),
                cos_sin_cache.const_data_ptr<float>(), static_cast<float>(eps),
                num_tokens, num_insert_tokens, num_heads,
                static_cast<int>(cache_block_size), kv_cache.stride(0),
                kv_cache.stride(1), cos_sin_cache.size(0));
      });
}
