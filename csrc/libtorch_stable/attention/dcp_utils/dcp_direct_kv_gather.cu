// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Direct symmetric-memory DCP KV gather.

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#include "dcp_direct_common.cuh"

namespace {

using vllm::direct_dcp::check_cuda_launch;
using vllm::direct_dcp::get_peer_ptr;
using vllm::direct_dcp::increment_epoch_kernel;
using vllm::direct_dcp::multimem_store_16;
using vllm::direct_dcp::multimem_store_release_system;
using vllm::direct_dcp::store_release_system;
using vllm::direct_dcp::wait_for_epoch;

constexpr int kThreads = 256;
// KV chunks need many blocks in flight to saturate the fabric.
constexpr int64_t kMaxMulticastBlocks = 128;
constexpr int64_t kMaxPeerBlocks = 1024;

// Bound on the peer-path launch grid. The kernel is PCIe-bound, so a few
// dozen blocks per destination move the payload as fast as a thousand, and
// the remaining resident blocks only hold SM thread and register slots that
// the attention and projection kernels running concurrently on the compute
// stream (the pipelined chunked-context loop) need.
// VLLM_DCP_KV_GATHER_MAX_BLOCKS overrides it; clamped to [world_size,
// kMaxPeerBlocks] at launch.
int64_t max_peer_blocks() {
  static const int64_t cap = [] {
    const char* value = std::getenv("VLLM_DCP_KV_GATHER_MAX_BLOCKS");
    int64_t parsed = value ? std::atoll(value) : kMaxPeerBlocks;
    return parsed > 0 ? parsed : kMaxPeerBlocks;
  }();
  return cap;
}

// Multicast each rank's valid local rows directly into compact, request-major
// kv_c and k_pe planes. dst_rows maps every padded local input row to its final
// output row, or -1 for padding. Destination rows are disjoint across source
// ranks, so all ranks can publish concurrently without atomics.
__global__ void direct_dcp_kv_gather_multimem_kernel(
    const uint4* local_kv, const int32_t* dst_rows, uint4* mc_kv,
    uint32_t* mc_signal, const uint32_t* received_signal,
    const int64_t* epoch_ptr, uint32_t* completion, int64_t world_size,
    int64_t rank, int64_t num_tokens, int64_t items_per_row,
    int64_t kv_c_items_per_row, int64_t output_tokens,
    int64_t max_gathered_tokens, int64_t buffer_slot,
    int64_t slot_stride_items) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  mc_kv += buffer_slot * slot_stride_items;

  int64_t item_count = num_tokens * items_per_row;
  int64_t item_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t item =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       item < item_count; item += item_stride) {
    int64_t src_row = item / items_per_row;
    int32_t dst_row = dst_rows[src_row];
    if (dst_row < 0) {
      continue;
    }
    if (dst_row >= output_tokens) {
      printf(
          "direct DCP final-layout kv-gather destination out of bounds "
          "source=%lld dst=%d output_tokens=%lld\n",
          static_cast<long long>(rank), dst_row,
          static_cast<long long>(output_tokens));
      asm volatile("trap;");
    }
    int64_t row_item = item - src_row * items_per_row;
    int64_t dst_item;
    if (row_item < kv_c_items_per_row) {
      dst_item = static_cast<int64_t>(dst_row) * kv_c_items_per_row + row_item;
    } else {
      int64_t k_pe_items_per_row = items_per_row - kv_c_items_per_row;
      dst_item = max_gathered_tokens * kv_c_items_per_row +
                 static_cast<int64_t>(dst_row) * k_pe_items_per_row + row_item -
                 kv_c_items_per_row;
    }
    multimem_store_16(mc_kv + dst_item, local_kv[item]);
  }

  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }

  uint32_t completed = atomicAdd(completion + buffer_slot, 1u);
  if (completed + 1u != gridDim.x) {
    return;
  }
  atomicExch(completion + buffer_slot, 0u);

  multimem_store_release_system(mc_signal + buffer_slot * world_size + rank,
                                epoch);
  for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
    int64_t signal_item = buffer_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + signal_item, epoch)) {
      printf("direct DCP final-layout kv-gather timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

// PCIe/UVA fallback for systems without NVLS multicast. Each source rank
// publishes its valid rows directly into every peer's compact, request-major
// planes. This moves the same payload volume as an all-gather while avoiding
// the rank-major materialization and the subsequent reorganization pass.
//
// Destinations are assigned per thread block and rotated by the source rank
// (block b serves destination (b + rank) mod world_size). All ranks run the
// same schedule at the same time, so with a destination-major walk every
// source would push into one destination's inbound link while the other
// links idle; the rotation keeps every inbound link busy with a different
// source. The host launches a block count that is a multiple of world_size.
__global__ void direct_dcp_kv_gather_peer_kernel(
    const uint4* local_kv, const int32_t* dst_rows,
    const int64_t* peer_kv_ptrs, const int64_t* peer_signal_ptrs,
    const uint32_t* received_signal, const int64_t* epoch_ptr,
    uint32_t* completion, int64_t world_size, int64_t rank,
    int64_t num_tokens, int64_t items_per_row, int64_t kv_c_items_per_row,
    int64_t output_tokens, int64_t max_gathered_tokens,
    int64_t buffer_slot, int64_t slot_stride_items) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t source_items = num_tokens * items_per_row;
  int64_t destination_rank =
      (static_cast<int64_t>(blockIdx.x) + rank) % world_size;
  int64_t blocks_per_destination = static_cast<int64_t>(gridDim.x) / world_size;
  int64_t block_in_destination = static_cast<int64_t>(blockIdx.x) / world_size;
  int64_t item_stride = blocks_per_destination * blockDim.x;
  uint4* peer_kv = get_peer_ptr<uint4>(peer_kv_ptrs, destination_rank) +
                   buffer_slot * slot_stride_items;
  for (int64_t item = block_in_destination * blockDim.x + threadIdx.x;
       item < source_items; item += item_stride) {
    int64_t src_row = item / items_per_row;
    int32_t dst_row = dst_rows[src_row];
    if (dst_row < 0) {
      continue;
    }
    if (dst_row >= output_tokens) {
      printf(
          "direct DCP peer final-layout destination out of bounds "
          "source=%lld dst=%d output_tokens=%lld\n",
          static_cast<long long>(rank), dst_row,
          static_cast<long long>(output_tokens));
      asm volatile("trap;");
    }
    int64_t row_item = item - src_row * items_per_row;
    int64_t dst_item;
    if (row_item < kv_c_items_per_row) {
      dst_item = static_cast<int64_t>(dst_row) * kv_c_items_per_row + row_item;
    } else {
      int64_t k_pe_items_per_row = items_per_row - kv_c_items_per_row;
      dst_item = max_gathered_tokens * kv_c_items_per_row +
                 static_cast<int64_t>(dst_row) * k_pe_items_per_row +
                 row_item - kv_c_items_per_row;
    }
    peer_kv[dst_item] = local_kv[item];
  }

  // Every block publishes its system-scope payload before contributing to the
  // completion count. The final block releases one signal to each peer, then
  // waits until every source has published into this rank's local replica.
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  uint32_t completed = atomicAdd(completion + buffer_slot, 1u);
  if (completed + 1u != gridDim.x) {
    return;
  }
  atomicExch(completion + buffer_slot, 0u);

  int64_t signal_item = buffer_slot * world_size + rank;
  for (int64_t destination_rank = 0; destination_rank < world_size;
       ++destination_rank) {
    uint32_t* peer_signal =
        get_peer_ptr<uint32_t>(peer_signal_ptrs, destination_rank);
    store_release_system(peer_signal + signal_item, epoch);
  }
  for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
    int64_t source_signal_item = buffer_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + source_signal_item, epoch)) {
      printf("direct DCP peer final-layout timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

void direct_dcp_kv_gather(const torch::stable::Tensor& local_kv,
                          const torch::stable::Tensor& dst_rows,
                          const torch::stable::Tensor& peer_kv_ptrs,
                          const torch::stable::Tensor& peer_signal_ptrs,
                          torch::stable::Tensor& received_kv,
                          torch::stable::Tensor& received_signal,
                          torch::stable::Tensor& completion,
                          torch::stable::Tensor& epoch, int64_t output_tokens,
                          int64_t plane_split_dim, int64_t buffer_slot,
                          int64_t world_size, int64_t rank,
                          int64_t max_gathered_tokens, int64_t kv_mc_ptr,
                          int64_t signal_mc_ptr) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(local_kv.is_cuda(), "local kv must be a CUDA tensor");
  ScalarType dtype = local_kv.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16 ||
                      dtype == ScalarType::Float8_e4m3fn,
                  "direct DCP final-layout kv-gather only supports FP16, "
                  "BF16, and FP8");
  STD_TORCH_CHECK(local_kv.dim() == 2 && local_kv.is_contiguous(),
                  "local kv must be a contiguous [T,D] tensor");
  STD_TORCH_CHECK(dst_rows.is_cuda() && dst_rows.is_contiguous() &&
                      dst_rows.scalar_type() == ScalarType::Int &&
                      dst_rows.dim() == 1 &&
                      dst_rows.numel() == local_kv.size(0),
                  "final-layout destination rows must be CUDA int32 [T]");
  STD_TORCH_CHECK(
      peer_kv_ptrs.is_cuda() && peer_kv_ptrs.is_contiguous() &&
          peer_kv_ptrs.scalar_type() == ScalarType::Long &&
          peer_kv_ptrs.dim() == 1 && peer_kv_ptrs.numel() == world_size,
      "KV peer pointer table must be CUDA int64 [world_size]");
  STD_TORCH_CHECK(
      peer_signal_ptrs.is_cuda() && peer_signal_ptrs.is_contiguous() &&
          peer_signal_ptrs.scalar_type() == ScalarType::Long &&
          peer_signal_ptrs.dim() == 1 &&
          peer_signal_ptrs.numel() == world_size,
      "signal peer pointer table must be CUDA int64 [world_size]");
  STD_TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  STD_TORCH_CHECK(output_tokens > 0 && output_tokens <= max_gathered_tokens,
                  "final-layout output exceeds symmetric buffer capacity");

  int64_t num_tokens = local_kv.size(0);
  int64_t token_dim = local_kv.size(1);
  int64_t element_size = local_kv.element_size();
  STD_TORCH_CHECK(num_tokens > 0 && token_dim > 0,
                  "local kv dimensions must be positive");
  STD_TORCH_CHECK(plane_split_dim > 0 && plane_split_dim < token_dim,
                  "final-layout plane split must be within the token row");
  STD_TORCH_CHECK(num_tokens * world_size <= max_gathered_tokens,
                  "padded gathered kv exceeds symmetric buffer capacity");

  // The symmetric buffer holds num_slots independent [T, D] slots; the
  // caller rotates through them so a gather can publish into one slot while
  // consumers still read another.
  STD_TORCH_CHECK(received_kv.is_cuda() && received_kv.scalar_type() == dtype &&
                      received_kv.is_contiguous() && received_kv.dim() == 3 &&
                      received_kv.size(0) >= 2 &&
                      received_kv.size(1) == max_gathered_tokens &&
                      received_kv.size(2) == token_dim,
                  "received kv has the wrong symmetric buffer layout");
  int64_t num_slots = received_kv.size(0);
  STD_TORCH_CHECK(buffer_slot >= 0 && buffer_slot < num_slots,
                  "final-layout buffer slot is out of range");
  STD_TORCH_CHECK(
      received_signal.is_cuda() && received_signal.is_contiguous() &&
          received_signal.scalar_type() == ScalarType::Int &&
          received_signal.dim() == 2 && received_signal.size(0) == num_slots &&
          received_signal.size(1) == world_size,
      "received signal has the wrong symmetric buffer layout");
  STD_TORCH_CHECK(completion.is_cuda() && completion.is_contiguous() &&
                      completion.scalar_type() == ScalarType::Int &&
                      completion.numel() == num_slots,
                  "completion counter must hold one CUDA int32 per slot");
  STD_TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                      epoch.scalar_type() == ScalarType::Long &&
                      epoch.numel() == 1,
                  "epoch must be a one-element CUDA int64 tensor");

  int64_t device_index = local_kv.get_device_index();
  STD_TORCH_CHECK(dst_rows.get_device_index() == device_index &&
                      peer_kv_ptrs.get_device_index() == device_index &&
                      peer_signal_ptrs.get_device_index() == device_index &&
                      received_kv.get_device_index() == device_index &&
                      received_signal.get_device_index() == device_index &&
                      completion.get_device_index() == device_index &&
                      epoch.get_device_index() == device_index,
                  "direct DCP final-layout tensors must share a CUDA device");

  int64_t row_bytes = token_dim * element_size;
  int64_t kv_c_row_bytes = plane_split_dim * element_size;
  int64_t k_pe_row_bytes = row_bytes - kv_c_row_bytes;
  int64_t slot_stride_bytes = max_gathered_tokens * row_bytes;
  bool vectorized =
      reinterpret_cast<uintptr_t>(local_kv.data_ptr()) % alignof(uint4) == 0 &&
      reinterpret_cast<uintptr_t>(received_kv.data_ptr()) % alignof(uint4) ==
          0 &&
      row_bytes % sizeof(uint4) == 0 && kv_c_row_bytes % sizeof(uint4) == 0 &&
      k_pe_row_bytes % sizeof(uint4) == 0 &&
      slot_stride_bytes % sizeof(uint4) == 0;
  STD_TORCH_CHECK(vectorized,
                  "direct DCP final-layout kv-gather requires 16-byte-aligned "
                  "planes, rows, and pointers");
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  cudaStream_t stream = get_current_cuda_stream();
  increment_epoch_kernel<<<1, 1, 0, stream>>>(
      epoch.mutable_data_ptr<int64_t>());
  check_cuda_launch("direct DCP final-layout kv-gather");

  int64_t items_per_row = row_bytes / sizeof(uint4);
  int64_t kv_c_items_per_row = kv_c_row_bytes / sizeof(uint4);
  int64_t item_count = num_tokens * items_per_row;
  int64_t blocks;
  if (kv_mc_ptr != 0 && signal_mc_ptr != 0) {
    blocks = (item_count + kThreads - 1) / kThreads;
    blocks = blocks < kMaxMulticastBlocks ? blocks : kMaxMulticastBlocks;
    direct_dcp_kv_gather_multimem_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(local_kv.data_ptr()),
        dst_rows.const_data_ptr<int32_t>(),
        reinterpret_cast<uint4*>(static_cast<uintptr_t>(kv_mc_ptr)),
        reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(signal_mc_ptr)),
        reinterpret_cast<const uint32_t*>(
            received_signal.const_data_ptr<int32_t>()),
        epoch.const_data_ptr<int64_t>(),
        reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
        world_size, rank, num_tokens, items_per_row, kv_c_items_per_row,
        output_tokens, max_gathered_tokens, buffer_slot,
        slot_stride_bytes / sizeof(uint4));
  } else {
    int64_t peer_item_count = world_size * item_count;
    blocks = (peer_item_count + kThreads - 1) / kThreads;
    int64_t cap = max_peer_blocks();
    cap = cap < kMaxPeerBlocks ? cap : kMaxPeerBlocks;
    cap = cap > world_size ? cap : world_size;
    blocks = blocks < cap ? blocks : cap;
    // One block set per destination: round up to a multiple of world_size.
    blocks = ((blocks + world_size - 1) / world_size) * world_size;
    direct_dcp_kv_gather_peer_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(local_kv.data_ptr()),
        dst_rows.const_data_ptr<int32_t>(),
        peer_kv_ptrs.const_data_ptr<int64_t>(),
        peer_signal_ptrs.const_data_ptr<int64_t>(),
        reinterpret_cast<const uint32_t*>(
            received_signal.const_data_ptr<int32_t>()),
        epoch.const_data_ptr<int64_t>(),
        reinterpret_cast<uint32_t*>(completion.mutable_data_ptr<int32_t>()),
        world_size, rank, num_tokens, items_per_row, kv_c_items_per_row,
        output_tokens, max_gathered_tokens, buffer_slot,
        slot_stride_bytes / sizeof(uint4));
  }
  check_cuda_launch("direct DCP final-layout kv-gather");
}

// ---------------------------------------------------------------------------
// Copy-engine variant of the final-layout publisher.
//
// The peer kernel above pushes rows with SM stores over PCIe. Those
// stores occupy the SMs' memory pipelines for the whole PCIe-bound gather,
// which slows every kernel that runs concurrently on the compute stream
// (measured 2026-09-02: FlashAttention 1.13 -> 3.48 ms, kv_b_proj 0.29 ->
// 1.96 ms per window at 1,024 blocks; 1.20 / 0.33 ms at 16 blocks). This
// variant moves the payload with cudaMemcpyAsync on the copy engines, which
// do not touch the SMs, and only launches two single-block kernels: one that
// releases this rank's epoch into every peer's signal slot after the copies
// completed (stream order; PCIe keeps the flag behind the data on the same
// link), and one that waits until every source's signal shows the epoch.
//
// The payload is plane-separated on the source side (contiguous kv_c rows
// and contiguous k_pe rows), so each request's rows are two linear copies
// per destination. `runs` is a CPU int64 [n, 3] tensor of
// (source row, destination row, rows) per request; destination rows are the
// compact request-major layout of build_dcp_kv_final_layout_dst_rows.
__global__ void direct_dcp_kv_signal_kernel(const int64_t* peer_signal_ptrs,
                                            const int64_t* epoch_ptr,
                                            int64_t world_size, int64_t rank,
                                            int64_t buffer_slot) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t signal_item = buffer_slot * world_size + rank;
  for (int64_t destination_rank = threadIdx.x; destination_rank < world_size;
       destination_rank += blockDim.x) {
    uint32_t* peer_signal =
        get_peer_ptr<uint32_t>(peer_signal_ptrs, destination_rank);
    store_release_system(peer_signal + signal_item, epoch);
  }
}

__global__ void direct_dcp_kv_wait_kernel(const uint32_t* received_signal,
                                          const int64_t* epoch_ptr,
                                          int64_t world_size,
                                          int64_t buffer_slot) {
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  for (int64_t source_rank = threadIdx.x; source_rank < world_size;
       source_rank += blockDim.x) {
    int64_t source_signal_item = buffer_slot * world_size + source_rank;
    if (!wait_for_epoch(received_signal + source_signal_item, epoch)) {
      printf("direct DCP DMA final-layout timeout source=%lld epoch=%u\n",
             static_cast<long long>(source_rank), epoch);
      asm volatile("trap;");
    }
  }
}

void direct_dcp_kv_gather_dma(const torch::stable::Tensor& local_kv_c,
                              const torch::stable::Tensor& local_k_pe,
                              const torch::stable::Tensor& runs,
                              const torch::stable::Tensor& peer_kv_ptrs_host,
                              const torch::stable::Tensor& peer_signal_ptrs,
                              torch::stable::Tensor& received_kv,
                              torch::stable::Tensor& received_signal,
                              torch::stable::Tensor& epoch,
                              int64_t output_tokens, int64_t plane_split_dim,
                              int64_t buffer_slot, int64_t world_size,
                              int64_t rank, int64_t max_gathered_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(local_kv_c.is_cuda() && local_k_pe.is_cuda(),
                  "local planes must be CUDA tensors");
  ScalarType dtype = local_kv_c.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::Half || dtype == ScalarType::BFloat16 ||
                      dtype == ScalarType::Float8_e4m3fn,
                  "direct DCP DMA kv-gather only supports FP16, BF16, and FP8");
  STD_TORCH_CHECK(local_k_pe.scalar_type() == dtype,
                  "local planes must share a dtype");
  STD_TORCH_CHECK(local_kv_c.dim() == 2 && local_kv_c.is_contiguous() &&
                      local_k_pe.dim() == 2 && local_k_pe.is_contiguous() &&
                      local_kv_c.size(0) == local_k_pe.size(0),
                  "local planes must be contiguous [T, D] tensors of equal T");
  STD_TORCH_CHECK(runs.dim() == 2 && runs.size(1) == 3 && runs.is_contiguous() &&
                      runs.scalar_type() == ScalarType::Long && !runs.is_cuda(),
                  "runs must be a contiguous CPU int64 [n, 3] tensor");
  STD_TORCH_CHECK(
      peer_kv_ptrs_host.dim() == 1 && peer_kv_ptrs_host.is_contiguous() &&
          peer_kv_ptrs_host.scalar_type() == ScalarType::Long &&
          !peer_kv_ptrs_host.is_cuda() &&
          peer_kv_ptrs_host.numel() == world_size,
      "KV peer pointer table must be a CPU int64 [world_size] tensor");
  STD_TORCH_CHECK(
      peer_signal_ptrs.is_cuda() && peer_signal_ptrs.is_contiguous() &&
          peer_signal_ptrs.scalar_type() == ScalarType::Long &&
          peer_signal_ptrs.dim() == 1 && peer_signal_ptrs.numel() == world_size,
      "signal peer pointer table must be CUDA int64 [world_size]");
  STD_TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");
  STD_TORCH_CHECK(output_tokens > 0 && output_tokens <= max_gathered_tokens,
                  "final-layout output exceeds symmetric buffer capacity");

  int64_t num_tokens = local_kv_c.size(0);
  int64_t kv_c_dim = local_kv_c.size(1);
  int64_t k_pe_dim = local_k_pe.size(1);
  int64_t token_dim = kv_c_dim + k_pe_dim;
  int64_t element_size = local_kv_c.element_size();
  STD_TORCH_CHECK(kv_c_dim == plane_split_dim && k_pe_dim > 0,
                  "local planes must split the token row at plane_split_dim");
  STD_TORCH_CHECK(received_kv.is_cuda() && received_kv.scalar_type() == dtype &&
                      received_kv.is_contiguous() && received_kv.dim() == 3 &&
                      received_kv.size(0) >= 2 &&
                      received_kv.size(1) == max_gathered_tokens &&
                      received_kv.size(2) == token_dim,
                  "received kv has the wrong symmetric buffer layout");
  int64_t num_slots = received_kv.size(0);
  STD_TORCH_CHECK(buffer_slot >= 0 && buffer_slot < num_slots,
                  "final-layout buffer slot is out of range");
  STD_TORCH_CHECK(
      received_signal.is_cuda() && received_signal.is_contiguous() &&
          received_signal.scalar_type() == ScalarType::Int &&
          received_signal.dim() == 2 && received_signal.size(0) == num_slots &&
          received_signal.size(1) == world_size,
      "received signal has the wrong symmetric buffer layout");
  STD_TORCH_CHECK(epoch.is_cuda() && epoch.is_contiguous() &&
                      epoch.scalar_type() == ScalarType::Long &&
                      epoch.numel() == 1,
                  "epoch must be a one-element CUDA int64 tensor");
  int64_t device_index = local_kv_c.get_device_index();
  STD_TORCH_CHECK(local_k_pe.get_device_index() == device_index &&
                      peer_signal_ptrs.get_device_index() == device_index &&
                      received_kv.get_device_index() == device_index &&
                      received_signal.get_device_index() == device_index &&
                      epoch.get_device_index() == device_index,
                  "direct DCP final-layout tensors must share a CUDA device");

  const int64_t* run_data = runs.const_data_ptr<int64_t>();
  int64_t num_runs = runs.size(0);
  for (int64_t i = 0; i < num_runs; ++i) {
    int64_t src = run_data[3 * i], dst = run_data[3 * i + 1],
            rows = run_data[3 * i + 2];
    STD_TORCH_CHECK(rows >= 0 && src >= 0 && dst >= 0 &&
                        src + rows <= num_tokens && dst + rows <= output_tokens,
                    "final-layout run exceeds the local rows or the output");
  }
  const int64_t* peer_kv = peer_kv_ptrs_host.const_data_ptr<int64_t>();
  int64_t kv_c_row_bytes = kv_c_dim * element_size;
  int64_t k_pe_row_bytes = k_pe_dim * element_size;
  int64_t slot_stride_bytes = max_gathered_tokens * token_dim * element_size;
  int64_t k_pe_plane_offset = max_gathered_tokens * kv_c_row_bytes;
  STD_TORCH_CHECK(static_cast<uintptr_t>(peer_kv[rank]) ==
                      reinterpret_cast<uintptr_t>(received_kv.data_ptr()),
                  "the local entry of the peer pointer table must be received_kv");

  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  cudaStream_t stream = get_current_cuda_stream();
  increment_epoch_kernel<<<1, 1, 0, stream>>>(
      epoch.mutable_data_ptr<int64_t>());
  check_cuda_launch("direct DCP DMA final-layout kv-gather");

  const char* src_kv_c = static_cast<const char*>(local_kv_c.data_ptr());
  const char* src_k_pe = static_cast<const char*>(local_k_pe.data_ptr());
  // Destinations rotated by the source rank: all ranks issue their copies at
  // the same time, and with a common order every source would write into
  // the same destination's inbound link at once (measured 2026-09-03:
  // 113-156 us per 3 MB copy for the first destinations, 437-445 us for
  // the last ones). Starting each source one rank past itself keeps every
  // inbound link busy with a different source. The local copy goes first.
  for (int64_t step = 0; step < world_size; ++step) {
    int64_t destination_rank = (rank + step) % world_size;
    char* slot_base = reinterpret_cast<char*>(
                          static_cast<uintptr_t>(peer_kv[destination_rank])) +
                      buffer_slot * slot_stride_bytes;
    for (int64_t i = 0; i < num_runs; ++i) {
      int64_t src = run_data[3 * i], dst = run_data[3 * i + 1],
              rows = run_data[3 * i + 2];
      if (rows == 0) {
        continue;
      }
      cudaError_t err = cudaMemcpyAsync(
          slot_base + dst * kv_c_row_bytes, src_kv_c + src * kv_c_row_bytes,
          rows * kv_c_row_bytes, cudaMemcpyDefault, stream);
      STD_TORCH_CHECK(err == cudaSuccess,
                      std::string("direct DCP DMA kv_c copy failed: ") +
                          cudaGetErrorString(err));
      err = cudaMemcpyAsync(slot_base + k_pe_plane_offset + dst * k_pe_row_bytes,
                            src_k_pe + src * k_pe_row_bytes,
                            rows * k_pe_row_bytes, cudaMemcpyDefault, stream);
      STD_TORCH_CHECK(err == cudaSuccess,
                      std::string("direct DCP DMA k_pe copy failed: ") +
                          cudaGetErrorString(err));
    }
  }
  direct_dcp_kv_signal_kernel<<<1, 32, 0, stream>>>(
      peer_signal_ptrs.const_data_ptr<int64_t>(), epoch.const_data_ptr<int64_t>(),
      world_size, rank, buffer_slot);
  check_cuda_launch("direct DCP DMA final-layout signal");
  direct_dcp_kv_wait_kernel<<<1, 32, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(
          received_signal.const_data_ptr<int32_t>()),
      epoch.const_data_ptr<int64_t>(), world_size, buffer_slot);
  check_cuda_launch("direct DCP DMA final-layout wait");
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_kv_gather_ops) {
  direct_dcp_kv_gather_ops.def(
      "direct_dcp_kv_gather("
      "Tensor local_kv, Tensor dst_rows, Tensor peer_kv_ptrs, "
      "Tensor peer_signal_ptrs, Tensor! received_kv, "
      "Tensor! received_signal, Tensor! completion, Tensor! epoch, "
      "int output_tokens, int plane_split_dim, int buffer_slot, "
      "int world_size, int rank, int max_gathered_tokens, "
      "int kv_mc_ptr, int signal_mc_ptr) -> ()");
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, direct_dcp_kv_gather_dma_ops) {
  direct_dcp_kv_gather_dma_ops.def(
      "direct_dcp_kv_gather_dma("
      "Tensor local_kv_c, Tensor local_k_pe, Tensor runs, "
      "Tensor peer_kv_ptrs_host, Tensor peer_signal_ptrs, Tensor! received_kv, "
      "Tensor! received_signal, Tensor! epoch, "
      "int output_tokens, int plane_split_dim, int buffer_slot, "
      "int world_size, int rank, int max_gathered_tokens) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_kv_gather_ops) {
  direct_dcp_kv_gather_ops.impl("direct_dcp_kv_gather",
                                TORCH_BOX(&direct_dcp_kv_gather));
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, direct_dcp_kv_gather_dma_ops) {
  direct_dcp_kv_gather_dma_ops.impl("direct_dcp_kv_gather_dma",
                                    TORCH_BOX(&direct_dcp_kv_gather_dma));
}
