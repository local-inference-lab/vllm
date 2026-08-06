#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <limits>
#include <sstream>

namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::ostringstream error;
  error << operation << " failed with cuBLAS status "
        << static_cast<int>(status);
  STD_TORCH_CHECK(false, error.str());
}

int checked_int(int64_t value, const char* name) {
  STD_TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(), name,
                  " is outside cuBLAS int range: ", value);
  return static_cast<int>(value);
}

void check_bf16_cuda_3d(torch::stable::Tensor const& tensor,
                        const char* name) {
  STD_TORCH_CHECK(tensor.device().is_cuda(), name, " must be on CUDA");
  STD_TORCH_CHECK(tensor.dim() == 3, name, " must be a 3D tensor");
  STD_TORCH_CHECK(
      tensor.scalar_type() == torch::headeronly::ScalarType::BFloat16, name,
      " must be BF16");
}

}  // namespace

void safe_mla_query_bmm(torch::stable::Tensor const& query,
                        torch::stable::Tensor const& weight,
                        torch::stable::Tensor& output) {
  check_bf16_cuda_3d(query, "query");
  check_bf16_cuda_3d(weight, "weight");
  check_bf16_cuda_3d(output, "output");
  STD_TORCH_CHECK(query.get_device_index() == weight.get_device_index() &&
                      query.get_device_index() == output.get_device_index(),
                  "query, weight, and output must be on the same CUDA device");

  const int64_t heads_i64 = query.size(0);
  const int64_t tokens_i64 = query.size(1);
  const int64_t q_dim_i64 = query.size(2);
  const int64_t latent_dim_i64 = weight.size(2);

  STD_TORCH_CHECK(weight.size(0) == heads_i64 && weight.size(1) == q_dim_i64,
                  "weight must have shape [heads, q_dim, latent_dim]");
  STD_TORCH_CHECK(output.size(0) == heads_i64 &&
                      output.size(1) == tokens_i64 &&
                      output.size(2) == latent_dim_i64,
                  "output must have shape [heads, tokens, latent_dim]");
  STD_TORCH_CHECK(query.stride(2) == 1,
                  "query q_dim must be contiguous for safe_mla_query_bmm");
  STD_TORCH_CHECK(weight.stride(2) == 1,
                  "weight latent_dim must be contiguous for safe_mla_query_bmm");
  STD_TORCH_CHECK(output.stride(2) == 1,
                  "output latent_dim must be contiguous for safe_mla_query_bmm");

  const int heads = checked_int(heads_i64, "heads");
  const int tokens = checked_int(tokens_i64, "tokens");
  const int q_dim = checked_int(q_dim_i64, "q_dim");
  const int latent_dim = checked_int(latent_dim_i64, "latent_dim");
  const int query_ld = checked_int(query.stride(1), "query.stride(1)");
  const int weight_ld = checked_int(weight.stride(1), "weight.stride(1)");
  const int output_ld = checked_int(output.stride(1), "output.stride(1)");

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  cublasHandle_t handle = get_current_cuda_blas_handle();
  check_cublas(cublasSetStream(handle, get_current_cuda_stream(
                                           query.get_device_index())),
               "cublasSetStream");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
      cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_N, CUBLAS_OP_N, latent_dim, tokens, q_dim, &alpha,
          weight.const_data_ptr(), CUDA_R_16BF, weight_ld,
          static_cast<long long>(weight.stride(0)), query.const_data_ptr(),
          CUDA_R_16BF, query_ld, static_cast<long long>(query.stride(0)),
          &beta, output.mutable_data_ptr(), CUDA_R_16BF, output_ld,
          static_cast<long long>(output.stride(0)), heads,
          // The explicit operand order and leading dimensions provide the
          // tight-query contract. Regular FP32 accumulation keeps tensor-core
          // kernels eligible; PEDANTIC forces a much slower fallback for
          // production prefill shapes without improving that contract.
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
      "cublasGemmStridedBatchedEx");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("safe_mla_query_bmm", TORCH_BOX(&safe_mla_query_bmm));
}
