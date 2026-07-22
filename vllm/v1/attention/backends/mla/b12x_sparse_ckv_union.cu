// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>

namespace {

__device__ __forceinline__ uint32_t hash_token_index(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  value ^= value >> 16;
  return value;
}

__device__ __forceinline__ int32_t find_first_position(
    int32_t key, const int32_t* hash_keys, const int32_t* hash_first_positions,
    int64_t hash_capacity) {
  const uint32_t mask = static_cast<uint32_t>(hash_capacity - 1);
  uint32_t slot = hash_token_index(static_cast<uint32_t>(key)) & mask;
  for (int64_t probe = 0; probe < hash_capacity; ++probe) {
    const int32_t stored = hash_keys[slot];
    if (stored == key) {
      return hash_first_positions[slot];
    }
    if (stored == -1) {
      return -1;
    }
    slot = (slot + 1) & mask;
  }
  return -1;
}

__global__ void build_dense_union_remap_kernel(
    const int32_t* __restrict__ indices, int32_t* __restrict__ union_indices,
    int32_t* __restrict__ remap, int32_t* __restrict__ union_counts,
    int32_t* __restrict__ hash_keys, int32_t* __restrict__ hash_first_positions,
    int32_t* __restrict__ first_to_dense, int destinations, int num_requests,
    int max_requests, int64_t input_entries, int64_t capacity,
    int64_t hash_capacity) {
  const int group = static_cast<int>(blockIdx.x);
  const int destination = group / num_requests;
  const int request = group % num_requests;
  if (destination >= destinations) {
    return;
  }
  const int state_group = destination * max_requests + request;
  const int64_t input_base = static_cast<int64_t>(group) * input_entries;
  const int64_t state_base = static_cast<int64_t>(state_group) * capacity;
  const int64_t hash_base = static_cast<int64_t>(state_group) * hash_capacity;

  for (int64_t i = threadIdx.x; i < hash_capacity; i += blockDim.x) {
    hash_keys[hash_base + i] = -1;
    hash_first_positions[hash_base + i] = std::numeric_limits<int32_t>::max();
  }
  for (int64_t i = threadIdx.x; i < capacity; i += blockDim.x) {
    union_indices[state_base + i] = -1;
    first_to_dense[state_base + i] = -1;
  }
  for (int64_t i = threadIdx.x; i < input_entries; i += blockDim.x) {
    remap[state_base + i] = -1;
  }
  if (threadIdx.x == 0) {
    union_counts[state_group] = 0;
  }
  __syncthreads();

  const uint32_t mask = static_cast<uint32_t>(hash_capacity - 1);
  for (int64_t i = threadIdx.x; i < input_entries; i += blockDim.x) {
    const int32_t key = indices[input_base + i];
    if (key < 0) {
      continue;
    }
    uint32_t slot = hash_token_index(static_cast<uint32_t>(key)) & mask;
    for (int64_t probe = 0; probe < hash_capacity; ++probe) {
      const int32_t previous = atomicCAS(hash_keys + hash_base + slot, -1, key);
      if (previous == -1 || previous == key) {
        atomicMin(hash_first_positions + hash_base + slot,
                  static_cast<int32_t>(i));
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
  __syncthreads();

  __shared__ int32_t first_flags[256];
  __shared__ int32_t running_count;
  if (threadIdx.x == 0) {
    running_count = 0;
  }
  __syncthreads();

  for (int64_t tile = 0; tile < input_entries; tile += blockDim.x) {
    const int64_t i = tile + threadIdx.x;
    int32_t is_first = 0;
    int32_t key = -1;
    if (i < input_entries) {
      key = indices[input_base + i];
      if (key >= 0) {
        is_first = find_first_position(key, hash_keys + hash_base,
                                       hash_first_positions + hash_base,
                                       hash_capacity) == i;
      }
    }
    first_flags[threadIdx.x] = is_first;
    __syncthreads();
    if (threadIdx.x == 0) {
      int32_t prefix = running_count;
      const int64_t remaining = input_entries - tile;
      const int32_t valid =
          static_cast<int32_t>(remaining < static_cast<int64_t>(blockDim.x)
                                   ? remaining
                                   : static_cast<int64_t>(blockDim.x));
      for (int32_t lane = 0; lane < valid; ++lane) {
        const int32_t flag = first_flags[lane];
        first_flags[lane] = prefix;
        prefix += flag;
      }
      running_count = prefix;
    }
    __syncthreads();
    if (i < input_entries && is_first) {
      const int32_t dense_slot = first_flags[threadIdx.x];
      first_to_dense[state_base + i] = dense_slot;
      union_indices[state_base + dense_slot] = key;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    union_counts[state_group] = running_count;
  }
  __syncthreads();

  for (int64_t i = threadIdx.x; i < input_entries; i += blockDim.x) {
    const int32_t key = indices[input_base + i];
    if (key < 0) {
      continue;
    }
    const int32_t first =
        find_first_position(key, hash_keys + hash_base,
                            hash_first_positions + hash_base, hash_capacity);
    if (first >= 0) {
      remap[state_base + i] = first_to_dense[state_base + first];
    }
  }
}

void validate_cuda_int32_contiguous(const torch::Tensor& value,
                                    const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(value.scalar_type() == torch::kInt32, name, " must be int32");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void build_dense_union_remap(torch::Tensor indices, torch::Tensor union_indices,
                             torch::Tensor remap, torch::Tensor union_counts,
                             torch::Tensor hash_keys,
                             torch::Tensor hash_first_positions,
                             torch::Tensor first_to_dense,
                             int64_t num_requests) {
  validate_cuda_int32_contiguous(indices, "indices");
  validate_cuda_int32_contiguous(union_indices, "union_indices");
  validate_cuda_int32_contiguous(remap, "remap");
  validate_cuda_int32_contiguous(union_counts, "union_counts");
  validate_cuda_int32_contiguous(hash_keys, "hash_keys");
  validate_cuda_int32_contiguous(hash_first_positions, "hash_first_positions");
  validate_cuda_int32_contiguous(first_to_dense, "first_to_dense");
  TORCH_CHECK(indices.dim() == 4,
              "indices must be [destination, request, row, topk]");
  TORCH_CHECK(num_requests > 0 && num_requests == indices.size(1),
              "active request count must match indices");
  TORCH_CHECK(union_indices.dim() == 3,
              "union_indices must be [destination, request, capacity]");
  TORCH_CHECK(hash_keys.dim() == 3,
              "hash_keys must be [destination, request, hash capacity]");

  const auto device = indices.device();
  TORCH_CHECK(union_indices.device() == device,
              "union_indices device mismatch");
  TORCH_CHECK(remap.device() == device, "remap device mismatch");
  TORCH_CHECK(union_counts.device() == device, "union_counts device mismatch");
  TORCH_CHECK(hash_keys.device() == device, "hash_keys device mismatch");
  TORCH_CHECK(hash_first_positions.device() == device,
              "hash_first_positions device mismatch");
  TORCH_CHECK(first_to_dense.device() == device,
              "first_to_dense device mismatch");

  const int destinations = static_cast<int>(indices.size(0));
  const int max_requests = static_cast<int>(union_indices.size(1));
  const int64_t input_entries = indices.size(2) * indices.size(3);
  const int64_t capacity = union_indices.size(2);
  const int64_t hash_capacity = hash_keys.size(2);
  TORCH_CHECK(capacity >= input_entries, "union capacity is too small");
  TORCH_CHECK((hash_capacity & (hash_capacity - 1)) == 0,
              "hash capacity must be a power of two");
  TORCH_CHECK(hash_capacity >= 2 * input_entries,
              "hash capacity must be at least 2x input entries");
  TORCH_CHECK(union_indices.size(0) == destinations,
              "union destination mismatch");
  TORCH_CHECK(num_requests <= max_requests,
              "active requests exceed union state capacity");
  TORCH_CHECK(remap.dim() == 4 && remap.size(0) == destinations &&
                  remap.size(1) == max_requests &&
                  remap.size(2) >= indices.size(2) &&
                  remap.size(3) == indices.size(3) &&
                  remap.size(2) * remap.size(3) == capacity,
              "remap shape mismatch");
  TORCH_CHECK(union_counts.dim() == 2 && union_counts.size(0) == destinations &&
                  union_counts.size(1) == max_requests,
              "count shape mismatch");
  TORCH_CHECK(hash_keys.sizes() == hash_first_positions.sizes(),
              "hash shape mismatch");
  TORCH_CHECK(
      hash_keys.size(0) == destinations && hash_keys.size(1) == max_requests,
      "hash state shape mismatch");
  TORCH_CHECK(first_to_dense.sizes() == union_indices.sizes(),
              "dense-slot shape mismatch");

  constexpr int threads = 256;
  const int blocks = destinations * static_cast<int>(num_requests);
  const auto stream = c10::cuda::getCurrentCUDAStream(indices.get_device());
  build_dense_union_remap_kernel<<<blocks, threads, 0, stream>>>(
      indices.data_ptr<int32_t>(), union_indices.data_ptr<int32_t>(),
      remap.data_ptr<int32_t>(), union_counts.data_ptr<int32_t>(),
      hash_keys.data_ptr<int32_t>(), hash_first_positions.data_ptr<int32_t>(),
      first_to_dense.data_ptr<int32_t>(), destinations,
      static_cast<int>(num_requests), max_requests, input_entries, capacity,
      hash_capacity);
  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("build_dense_union_remap", &build_dense_union_remap);
}
