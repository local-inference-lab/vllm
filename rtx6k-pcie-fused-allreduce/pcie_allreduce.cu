// PCIe-only custom allreduce extension.
// System-scope barriers for cross-GPU visibility over PCIe switch fabric.
// Self-contained: no sgl-kernel headers needed.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw std::runtime_error(_message.str());                         \
    }                                                                   \
  } while (0)

namespace pcie_allreduce {

// ---- Data structures ----

constexpr int kMaxBlocks = 512;
using FlagType = uint32_t;

// 128B stride per rank to avoid PCIe false sharing (one GPU L2 cache line).
constexpr int kFlagStride = 32;  // 32 * sizeof(uint32_t) = 128 bytes

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][8];
  // 16 slots: 0..7 for start flags, 8..15 for end flags (streaming kernel).
  alignas(128) FlagType peer_counter[2][kMaxBlocks][16 * kFlagStride];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[8];
};

struct __align__(16) RankSignals {
  Signal* signals[8];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

// ---- Scalar ops ----

DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) { return __float2half(val); }

DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) { a = __hadd(a, b); return a; }
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

// ---- Flag operations (system-scope relaxed for PCIe) ----

static DINLINE void st_flag_relaxed(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_relaxed(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// ---- System-scope barrier for PCIe ----
// Uses __threadfence_system() + relaxed atomics (DeepEP pattern).
// The fence flushes all prior writes system-wide, then relaxed ops
// handle signaling without per-operation ordering overhead.
template <int ngpus, bool is_start>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    __threadfence_system();
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank * kFlagStride];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x * kFlagStride];
    st_flag_relaxed(peer_counter_ptr, val);
    while (ld_flag_relaxed(self_counter_ptr) != val);
  }
  __syncthreads();
}

// ---- Reduction ----

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

template <typename P, int ngpus, typename A>
DINLINE A packed_reduce_acc(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return tmp;
}

template <typename P, typename A>
DINLINE P packed_add_downcast(A acc, P addend) {
  packed_assign_add(acc, upcast(addend));
  return downcast<P>(acc);
}

template <typename A>
DINLINE float packed_sum_squares(A val) {
  float out = 0.0f;
#pragma unroll
  for (int i = 0; i < A::size; i++) {
    out += val.data[i] * val.data[i];
  }
  return out;
}

template <typename P, typename A>
DINLINE P packed_rms_downcast(A val, P weight, float scale) {
  A w = upcast(weight);
  A out;
#pragma unroll
  for (int i = 0; i < A::size; i++) {
    out.data[i] = val.data[i] * scale * w.data[i];
  }
  return downcast<P>(out);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return reinterpret_cast<P*>(reinterpret_cast<char*>(sg) + sizeof(Signal));
}

// ---- PCIe kernel ----

// 1-stage allreduce with staggered peer reads and start-barrier-only.
// Staggering spreads PCIe traffic across the switch fabric by having each
// GPU read peers in a different rotated order.
// The end barrier is unnecessary because the caller double-buffers: two IPC
// buffer sets alternate, so the start barrier of the NEXT allreduce guarantees
// all peers finished reading the current slot before it gets overwritten.
// During CUDA graph capture each allreduce gets its own unique buffer, so
// there is no reuse conflict and no end barrier is needed either.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_kernel(
    RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(rotated, idx);
  }
}

template <typename T, int ngpus, bool end_barrier>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_add_kernel(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ addend,
    T* __restrict__ result,
    int rank,
    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  const P* addend_p = reinterpret_cast<const P*>(addend);
  P* result_p = reinterpret_cast<P*>(result);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    result_p[idx] = packed_add_downcast<P, A>(packed_reduce_acc<P, ngpus, A>(rotated, idx), addend_p[idx]);
  }
  if constexpr (end_barrier) {
    multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_2stage_add_kernel(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ addend,
    T* __restrict__ result,
    int rank,
    int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = reinterpret_cast<const P*>(dp.ptrs[target]);
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  P* tmp_out = tmps[0];
  const P* addend_p = reinterpret_cast<const P*>(addend);
  P* result_p = reinterpret_cast<P*>(result);

  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);

  // Stage 1: each rank reduces its output shard into its own IPC scratch.
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }

  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);

  // Stage 2: gather all shards and fuse the local residual add epilog.
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        result_p[dst_idx] = packed_add_downcast<P, A>(
            upcast(tmps[i][idx]), addend_p[dst_idx]);
      }
    }
  }
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_2stage_add_rms_kernel(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ addend,
    const T* __restrict__ weight,
    T* __restrict__ rms_out,
    T* __restrict__ residual_out,
    int rank,
    int rows,
    int hidden,
    float eps) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  extern __shared__ float smem[];

  const int row = blockIdx.x;
  const int hidden_p = hidden / P::size;
  const int rows_per_rank = rows / ngpus;
  const int owner = row / rows_per_rank;
  const int owner_local_row = row - owner * rows_per_rank;
  auto dp = *_dp;

  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    ptrs[i] = reinterpret_cast<const P*>(dp.ptrs[i]);
    tmps[i] = get_tmp_buf<P>(sg.signals[i]);
  }

  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);

  // Stage 1: the owner rank for each row reduces that full row into its
  // per-rank scratch. This row partition matches common decode graph sizes
  // where rows is divisible by TP world size.
  if (rank == owner) {
    P* tmp_row = tmps[rank] + owner_local_row * hidden_p;
    const int row_offset = row * hidden_p;
    for (int col = threadIdx.x; col < hidden_p; col += blockDim.x) {
      tmp_row[col] = packed_reduce<P, ngpus, A>(ptrs, row_offset + col);
    }
  }

  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);

  const P* reduced_row = tmps[owner] + owner_local_row * hidden_p;
  const P* addend_p = reinterpret_cast<const P*>(addend);
  const P* weight_p = reinterpret_cast<const P*>(weight);
  P* residual_p = reinterpret_cast<P*>(residual_out);
  P* rms_p = reinterpret_cast<P*>(rms_out);
  const int row_offset = row * hidden_p;

  float local_sumsq = 0.0f;
  for (int col = threadIdx.x; col < hidden_p; col += blockDim.x) {
    A val = upcast(reduced_row[col]);
    packed_assign_add(val, upcast(addend_p[row_offset + col]));
    residual_p[row_offset + col] = downcast<P>(val);
    local_sumsq += packed_sum_squares(val);
  }

  smem[threadIdx.x] = local_sumsq;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float scale = rsqrtf(smem[0] / static_cast<float>(hidden) + eps);

  for (int col = threadIdx.x; col < hidden_p; col += blockDim.x) {
    A val = upcast(residual_p[row_offset + col]);
    rms_p[row_offset + col] = packed_rms_downcast<P, A>(
        val, weight_p[col], scale);
  }
}

template <typename T, int ngpus, int group_size>
__global__ void __launch_bounds__(512, 1) pcie_group_reduce_kernel(
    RankData* _input_dp,
    RankData* _scratch_dp,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto input_dp = *_input_dp;
  auto scratch_dp = *_scratch_dp;
  const int group_base = (rank / group_size) * group_size;
  const P* local[group_size];
#pragma unroll
  for (int i = 0; i < group_size; i++) {
    local[i] = reinterpret_cast<const P*>(input_dp.ptrs[group_base + i]);
  }
  P* scratch = reinterpret_cast<P*>(const_cast<void*>(scratch_dp.ptrs[rank]));

  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    scratch[idx] = packed_reduce<P, group_size, A>(local, idx);
  }
  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
}

template <typename T, int ngpus, int group_size>
__global__ void __launch_bounds__(512, 1) pcie_hier_finalize_add_kernel(
    RankData* _scratch_dp,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ addend,
    T* __restrict__ result,
    int rank,
    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto scratch_dp = *_scratch_dp;
  const int remote_rank = (rank + group_size) % ngpus;
  const P* local = reinterpret_cast<const P*>(scratch_dp.ptrs[rank]);
  const P* remote = reinterpret_cast<const P*>(scratch_dp.ptrs[remote_rank]);
  const P* addend_p = reinterpret_cast<const P*>(addend);
  P* result_p = reinterpret_cast<P*>(result);

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    A acc = upcast(local[idx]);
    packed_assign_add(acc, upcast(remote[idx]));
    result_p[idx] = packed_add_downcast<P, A>(acc, addend_p[idx]);
  }
  // Experimental no-end-barrier path. This is intentionally unsafe under
  // arbitrary stream skew, but useful to measure whether the hierarchical
  // topology actually has enough upside before spending time on a correct
  // double-buffered/no-barrier protocol.
}

// ---- IPC key ----

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;

// ---- PCIeAllreduce class ----

class PCIeAllreduce {
 public:
  int rank_;
  int world_size_;

  RankSignals sg_;
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  std::map<IPC_KEY, char*> ipc_handles_;

  // Double-buffer state (eliminates end barrier).
  bool dbuf_enabled_ = false;
  bool add_end_barrier_ = false;
  int dbuf_slot_ = 0;
  void* dbuf_raw_[2][8] = {};
  RankData* dbuf_rd_[2] = {};
  bool hier_enabled_ = false;
  int hier_slot_ = 0;
  void* hier_scratch_raw_[2][8] = {};
  RankData* hier_scratch_rd_[2] = {};

  PCIeAllreduce(
      Signal** signals, void* rank_data, size_t rank_data_sz, int rank, int world_size)
      : rank_(rank),
        world_size_(world_size),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    const char* add_end_barrier_env = std::getenv("VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER");
    add_end_barrier_ = add_end_barrier_env != nullptr && std::strcmp(add_end_barrier_env, "0") != 0;
    for (int i = 0; i < world_size_; i++) sg_.signals[i] = signals[i];
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] = ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(
          (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)ipc_handle), cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      if (cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error("Rank data buffer overflow");
  }

  void register_pcie_buffers(void** ptrs0, void** ptrs1) {
    check_rank_data_capacity(2);
    for (int s = 0; s < 2; s++) {
      void** ptrs = s == 0 ? ptrs0 : ptrs1;
      RankData data;
      for (int i = 0; i < world_size_; i++) {
        data.ptrs[i] = ptrs[i];
        dbuf_raw_[s][i] = ptrs[i];
      }
      dbuf_rd_[s] = d_rank_data_base_++;
      CHECK_CUDA_SUCCESS(cudaMemcpy(dbuf_rd_[s], &data, sizeof(RankData), cudaMemcpyHostToDevice));
    }
    buffers_[ptrs0[rank_]] = dbuf_rd_[0];
    buffers_[ptrs1[rank_]] = dbuf_rd_[1];
    dbuf_enabled_ = true;
    dbuf_slot_ = 0;
  }

  void register_hier_scratch(void** ptrs0, void** ptrs1) {
    check_rank_data_capacity(2);
    for (int s = 0; s < 2; s++) {
      void** ptrs = s == 0 ? ptrs0 : ptrs1;
      RankData data;
      for (int i = 0; i < world_size_; i++) {
        data.ptrs[i] = ptrs[i];
        hier_scratch_raw_[s][i] = ptrs[i];
      }
      hier_scratch_rd_[s] = d_rank_data_base_++;
      CHECK_CUDA_SUCCESS(cudaMemcpy(hier_scratch_rd_[s], &data, sizeof(RankData), cudaMemcpyHostToDevice));
    }
    hier_enabled_ = true;
    hier_slot_ = 0;
  }

  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) data.ptrs[i] = ptrs[i];
    auto d_data = d_rank_data_base_++;
    CHECK_CUDA_SUCCESS(cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  void register_graph_buffers(
      const std::vector<std::string>& handles, const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle = open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CHECK_CUDA_SUCCESS(
        cudaMemcpy(d_rank_data_base_, rank_data.data(), sizeof(RankData) * num_buffers, cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size, int threads = 512, int block_limit = 36) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error("allreduce requires input length to be multiple of " + std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    if (dbuf_enabled_ && status != cudaStreamCaptureStatusActive) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      auto input_size = size * sizeof(T);
      AT_CUDA_CHECK(cudaMemcpyAsync(dbuf_raw_[slot][rank_], input, input_size, cudaMemcpyDeviceToDevice, stream));
      ptrs = dbuf_rd_[slot];
    } else if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    blocks = std::max(blocks, 1);

#define KL(ngpus) pcie_allreduce_kernel<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);
    switch (world_size_) {
      case 2: KL(2); break;
      case 4: KL(4); break;
      case 6: KL(6); break;
      case 8: KL(8); break;
      default:
        throw std::runtime_error("only supports (2,4,6,8) gpus, got " + std::to_string(world_size_));
    }
#undef KL
  }

  template <typename T>
  void allreduce_add(
      cudaStream_t stream,
      T* input,
      const T* addend,
      T* output,
      int size,
      int threads = 512,
      int block_limit = 36) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error("allreduce_add requires input length to be multiple of " + std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    if (dbuf_enabled_ && status != cudaStreamCaptureStatusActive) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      auto input_size = size * sizeof(T);
      AT_CUDA_CHECK(cudaMemcpyAsync(dbuf_raw_[slot][rank_], input, input_size, cudaMemcpyDeviceToDevice, stream));
      ptrs = dbuf_rd_[slot];
    } else if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    blocks = std::max(blocks, 1);

#define KL_WITH_BARRIER(ngpus) \
  pcie_allreduce_add_kernel<T, ngpus, true><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, addend, output, rank_, size)
#define KL_NO_BARRIER(ngpus) \
  pcie_allreduce_add_kernel<T, ngpus, false><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, addend, output, rank_, size)
#define KL(ngpus)          \
  if (add_end_barrier_) {  \
    KL_WITH_BARRIER(ngpus); \
  } else {                 \
    KL_NO_BARRIER(ngpus);  \
  }
    switch (world_size_) {
      case 2: KL(2); break;
      case 4: KL(4); break;
      case 6: KL(6); break;
      case 8: KL(8); break;
      default:
        throw std::runtime_error("only supports (2,4,6,8) gpus, got " + std::to_string(world_size_));
    }
#undef KL
#undef KL_NO_BARRIER
#undef KL_WITH_BARRIER
  }

  template <typename T>
  void allreduce_hier_add(
      cudaStream_t stream,
      T* input,
      const T* addend,
      T* output,
      int size,
      int threads = 512,
      int block_limit = 36) {
    if (!hier_enabled_) {
      throw std::runtime_error("hierarchical scratch buffers are not registered");
    }
    if (world_size_ != 4 && world_size_ != 8) {
      throw std::runtime_error("hierarchical allreduce currently supports only 4 or 8 GPUs");
    }
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error("allreduce_hier_add requires input length to be multiple of " + std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks));

    RankData* input_ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    if (dbuf_enabled_ && status != cudaStreamCaptureStatusActive) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      auto input_size = size * sizeof(T);
      AT_CUDA_CHECK(cudaMemcpyAsync(dbuf_raw_[slot][rank_], input, input_size, cudaMemcpyDeviceToDevice, stream));
      input_ptrs = dbuf_rd_[slot];
    } else if (status == cudaStreamCaptureStatusActive) {
      input_ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
      input_ptrs = it->second;
    }

    int scratch_slot = hier_slot_ % 2;
    hier_slot_++;
    RankData* scratch_ptrs = hier_scratch_rd_[scratch_slot];

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    blocks = std::max(blocks, 1);

#define KL(ngpus, group_size)                                                                    \
  pcie_group_reduce_kernel<T, ngpus, group_size><<<blocks, threads, 0, stream>>>(                 \
      input_ptrs, scratch_ptrs, sg_, self_sg_, rank_, size);                                      \
  pcie_hier_finalize_add_kernel<T, ngpus, group_size><<<blocks, threads, 0, stream>>>(             \
      scratch_ptrs, sg_, self_sg_, addend, output, rank_, size);
    switch (world_size_) {
      case 4: KL(4, 2); break;
      case 8: KL(8, 4); break;
      default:
        throw std::runtime_error("hierarchical allreduce currently supports only 4 or 8 GPUs");
    }
#undef KL
  }

  template <typename T>
  void allreduce_2stage_add(
      cudaStream_t stream,
      T* input,
      const T* addend,
      T* output,
      int size,
      int threads = 512,
      int block_limit = 36) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error("allreduce_2stage_add requires input length to be multiple of " + std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks));

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    if (dbuf_enabled_ && status != cudaStreamCaptureStatusActive) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      auto input_size = size * sizeof(T);
      AT_CUDA_CHECK(cudaMemcpyAsync(dbuf_raw_[slot][rank_], input, input_size, cudaMemcpyDeviceToDevice, stream));
      ptrs = dbuf_rd_[slot];
    } else if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    blocks = std::max(blocks, 1);

#define KL(ngpus) pcie_allreduce_2stage_add_kernel<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, addend, output, rank_, size);
    switch (world_size_) {
      case 2: KL(2); break;
      case 4: KL(4); break;
      case 6: KL(6); break;
      case 8: KL(8); break;
      default:
        throw std::runtime_error("only supports (2,4,6,8) gpus, got " + std::to_string(world_size_));
    }
#undef KL
  }

  template <typename T>
  void allreduce_2stage_add_rms(
      cudaStream_t stream,
      T* input,
      const T* addend,
      const T* weight,
      T* rms_out,
      T* residual_out,
      int rows,
      int hidden,
      float eps,
      int threads = 256) {
    auto d = packed_t<T>::P::size;
    if (hidden % d != 0)
      throw std::runtime_error("allreduce_2stage_add_rms requires hidden to be multiple of " + std::to_string(d));
    if (rows <= 0 || rows > kMaxBlocks)
      throw std::runtime_error("allreduce_2stage_add_rms rows must be in [1, " + std::to_string(kMaxBlocks) + "]");
    if (rows % world_size_ != 0)
      throw std::runtime_error("allreduce_2stage_add_rms requires rows divisible by world size");

    RankData* ptrs;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    const int size = rows * hidden;
    if (dbuf_enabled_ && status != cudaStreamCaptureStatusActive) {
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      auto input_size = size * sizeof(T);
      AT_CUDA_CHECK(cudaMemcpyAsync(dbuf_raw_[slot][rank_], input, input_size, cudaMemcpyDeviceToDevice, stream));
      ptrs = dbuf_rd_[slot];
    } else if (status == cudaStreamCaptureStatusActive) {
      ptrs = d_rank_data_base_ + graph_unreg_buffers_.size();
      graph_unreg_buffers_.push_back(input);
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error("buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
      ptrs = it->second;
    }

    dim3 blocks(rows);
    size_t smem = threads * sizeof(float);
#define KL(ngpus)                                                                                 \
  pcie_allreduce_2stage_add_rms_kernel<T, ngpus><<<blocks, threads, smem, stream>>>(              \
      ptrs, sg_, self_sg_, addend, weight, rms_out, residual_out, rank_, rows, hidden, eps);
    switch (world_size_) {
      case 2: KL(2); break;
      case 4: KL(4); break;
      case 6: KL(6); break;
      case 8: KL(8); break;
      default:
        throw std::runtime_error("only supports (2,4,6,8) gpus, got " + std::to_string(world_size_));
    }
#undef KL
  }

  ~PCIeAllreduce() {
    for (auto [_, ptr] : ipc_handles_) CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(ptr));
  }
};

}  // namespace pcie_allreduce

// ---- Python bindings ----

using fptr_t = int64_t;

static fptr_t
init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank");

  pcie_allreduce::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++)
    ipc_ptrs[i] = reinterpret_cast<pcie_allreduce::Signal*>(fake_ipc_ptrs[i]);
  return (fptr_t) new pcie_allreduce::PCIeAllreduce(
      ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank, world_size);
}

static bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

static void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();

  // When double-buffer is active, allreduce handles the memcpy and slot
  // alternation internally — pass input directly.
  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream, (float*)reg_buffer, (float*)out.data_ptr(), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream, (half*)reg_buffer, (half*)out.data_ptr(), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(stream, (nv_bfloat16*)reg_buffer, (nv_bfloat16*)out.data_ptr(), out.numel());
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void all_reduce_add(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& addend,
    torch::Tensor& out,
    fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(addend.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK_EQ(addend.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(addend));
  auto input_size = inp.numel() * inp.element_size();

  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce_add<float>(
          stream, (float*)reg_buffer, (const float*)addend.data_ptr(), (float*)out.data_ptr(), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce_add<half>(
          stream, (half*)reg_buffer, (const half*)addend.data_ptr(), (half*)out.data_ptr(), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce_add<nv_bfloat16>(
          stream,
          (nv_bfloat16*)reg_buffer,
          (const nv_bfloat16*)addend.data_ptr(),
          (nv_bfloat16*)out.data_ptr(),
          out.numel());
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void all_reduce_hier_add(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& addend,
    torch::Tensor& out,
    fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(addend.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK_EQ(addend.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(addend));
  auto input_size = inp.numel() * inp.element_size();

  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce_hier_add<float>(
          stream, (float*)reg_buffer, (const float*)addend.data_ptr(), (float*)out.data_ptr(), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce_hier_add<half>(
          stream, (half*)reg_buffer, (const half*)addend.data_ptr(), (half*)out.data_ptr(), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce_hier_add<nv_bfloat16>(
          stream,
          (nv_bfloat16*)reg_buffer,
          (const nv_bfloat16*)addend.data_ptr(),
          (nv_bfloat16*)out.data_ptr(),
          out.numel());
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void all_reduce_2stage_add(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& addend,
    torch::Tensor& out,
    fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(addend.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK_EQ(addend.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(addend));
  auto input_size = inp.numel() * inp.element_size();

  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce_2stage_add<float>(
          stream, (float*)reg_buffer, (const float*)addend.data_ptr(), (float*)out.data_ptr(), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce_2stage_add<half>(
          stream, (half*)reg_buffer, (const half*)addend.data_ptr(), (half*)out.data_ptr(), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce_2stage_add<nv_bfloat16>(
          stream,
          (nv_bfloat16*)reg_buffer,
          (const nv_bfloat16*)addend.data_ptr(),
          (nv_bfloat16*)out.data_ptr(),
          out.numel());
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void all_reduce_2stage_add_rms(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& addend,
    torch::Tensor& weight,
    torch::Tensor& rms_out,
    torch::Tensor& residual_out,
    double eps,
    fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.dim(), 2);
  TORCH_CHECK_EQ(addend.dim(), 2);
  TORCH_CHECK_EQ(rms_out.dim(), 2);
  TORCH_CHECK_EQ(residual_out.dim(), 2);
  TORCH_CHECK_EQ(inp.scalar_type(), addend.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), weight.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), rms_out.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), residual_out.scalar_type());
  TORCH_CHECK_EQ(inp.sizes(), addend.sizes());
  TORCH_CHECK_EQ(inp.sizes(), rms_out.sizes());
  TORCH_CHECK_EQ(inp.sizes(), residual_out.sizes());
  TORCH_CHECK_EQ(weight.numel(), inp.size(1));
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(addend));
  TORCH_CHECK(_is_weak_contiguous(weight));
  TORCH_CHECK(_is_weak_contiguous(rms_out));
  TORCH_CHECK(_is_weak_contiguous(residual_out));

  auto input_size = inp.numel() * inp.element_size();
  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  const int rows = static_cast<int>(inp.size(0));
  const int hidden = static_cast<int>(inp.size(1));
  switch (inp.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce_2stage_add_rms<float>(
          stream,
          (float*)reg_buffer,
          (const float*)addend.data_ptr(),
          (const float*)weight.data_ptr(),
          (float*)rms_out.data_ptr(),
          (float*)residual_out.data_ptr(),
          rows,
          hidden,
          static_cast<float>(eps));
      break;
    case at::ScalarType::Half:
      fa->allreduce_2stage_add_rms<half>(
          stream,
          (half*)reg_buffer,
          (const half*)addend.data_ptr(),
          (const half*)weight.data_ptr(),
          (half*)rms_out.data_ptr(),
          (half*)residual_out.data_ptr(),
          rows,
          hidden,
          static_cast<float>(eps));
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce_2stage_add_rms<nv_bfloat16>(
          stream,
          (nv_bfloat16*)reg_buffer,
          (const nv_bfloat16*)addend.data_ptr(),
          (const nv_bfloat16*)weight.data_ptr(),
          (nv_bfloat16*)rms_out.data_ptr(),
          (nv_bfloat16*)residual_out.data_ptr(),
          rows,
          hidden,
          static_cast<float>(eps));
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void dispose(fptr_t _fa) {
  delete reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
}

static int64_t meta_size() {
  return sizeof(pcie_allreduce::Signal);
}

static void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == (size_t)fa->world_size_);
  void* ipc_ptrs[8];
  for (size_t i = 0; i < fake_ipc_ptrs.size(); i++)
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  fa->register_buffer(ipc_ptrs);
}

static void register_pcie_buffers(fptr_t _fa, const std::vector<fptr_t>& ptrs0, const std::vector<fptr_t>& ptrs1) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(ptrs0.size() == (size_t)fa->world_size_);
  TORCH_CHECK(ptrs1.size() == (size_t)fa->world_size_);
  void* p0[8], *p1[8];
  for (size_t i = 0; i < ptrs0.size(); i++) {
    p0[i] = reinterpret_cast<void*>(ptrs0[i]);
    p1[i] = reinterpret_cast<void*>(ptrs1[i]);
  }
  fa->register_pcie_buffers(p0, p1);
}

static void register_hier_scratch(fptr_t _fa, const std::vector<fptr_t>& ptrs0, const std::vector<fptr_t>& ptrs1) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(ptrs0.size() == (size_t)fa->world_size_);
  TORCH_CHECK(ptrs1.size() == (size_t)fa->world_size_);
  void* p0[8], *p1[8];
  for (size_t i = 0; i < ptrs0.size(); i++) {
    p0[i] = reinterpret_cast<void*>(ptrs0[i]);
    p1[i] = reinterpret_cast<void*>(ptrs1[i]);
  }
  fa->register_hier_scratch(p0, p1);
}

static std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

static void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (size_t i = 0; i < handles.size(); i++)
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  fa->register_graph_buffers(bytes, offsets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar, "init PCIe allreduce");
  m.def("all_reduce", &all_reduce, "PCIe allreduce");
  m.def("all_reduce_add", &all_reduce_add, "fused PCIe allreduce + local addend");
  m.def("all_reduce_2stage_add", &all_reduce_2stage_add, "2-stage fused PCIe allreduce + local addend");
  m.def("all_reduce_2stage_add_rms", &all_reduce_2stage_add_rms, "2-stage fused PCIe allreduce + addend + RMSNorm");
  m.def("all_reduce_hier_add", &all_reduce_hier_add, "hierarchical fused PCIe allreduce + local addend");
  m.def("dispose", &dispose, "dispose PCIe allreduce");
  m.def("meta_size", &meta_size, "signal metadata size");
  m.def("register_buffer", &register_buffer, "register IPC buffer");
  m.def("register_pcie_buffers", &register_pcie_buffers, "register double-buffered IPC buffers");
  m.def("register_hier_scratch", &register_hier_scratch, "register double-buffered hierarchical scratch buffers");
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "get graph buffer IPC meta");
  m.def("register_graph_buffers", &register_graph_buffers, "register graph buffers");
}
