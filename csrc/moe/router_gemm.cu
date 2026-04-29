// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// bf16 x bf16 -> fp32 router GEMM via cuBLAS.
// Uses CUBLAS_COMPUTE_32F so bf16 operands accumulate into fp32,
// matching TRT-LLM's cuBLAS fallback behaviour in dsv3RouterGemmOp.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <limits>
#include <cublas_v2.h>
#include <cublasLt.h>

// cuBLAS column-major math for row-major PyTorch tensors:
//   weight[N,K]_row  lda=K  -> cuBLAS sees (K,N) col-major; CUBLAS_OP_T ->
//   (N,K) input[M,K]_row   ldb=K  -> cuBLAS sees (K,M) col-major; CUBLAS_OP_N
//   -> (K,M) out[M,N]_row     ldc=N  -> cuBLAS sees (N,M) col-major (written as
//   output^T)
// cuBLAS: C(N,M) = weight(N,K) @ input(K,M)  =>  C^T = output[M,N]
// params: m=N, n=M, k=K, lda=K (weight), ldb=K (input), ldc=N (output)

torch::Tensor router_gemm_bf16_fp32(torch::Tensor const& input,
                                    torch::Tensor const& weight) {
  TORCH_CHECK(input.dtype() == torch::kBFloat16,
              "router_gemm_bf16_fp32: input must be bfloat16");
  TORCH_CHECK(weight.dtype() == torch::kBFloat16,
              "router_gemm_bf16_fp32: weight must be bfloat16");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "router_gemm_bf16_fp32: input and weight must be 2-D");
  TORCH_CHECK(input.size(1) == weight.size(1),
              "router_gemm_bf16_fp32: inner dimensions must match");

  int64_t const M = input.size(0);
  int64_t const N = weight.size(0);
  int64_t const K = input.size(1);

  auto out = torch::empty({M, N}, input.options().dtype(torch::kFloat32));

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(
      cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));

  float const alpha = 1.0f;
  float const beta = 0.0f;

  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
      static_cast<int>(M), static_cast<int>(K), &alpha, weight.data_ptr(),
      CUDA_R_16BF, static_cast<int>(K), input.data_ptr(), CUDA_R_16BF,
      static_cast<int>(K), &beta, out.data_ptr(), CUDA_R_32F,
      static_cast<int>(N), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

  return out;
}

namespace {

constexpr int64_t KIMI_K25_HIDDEN_DIM = 7168;
constexpr int64_t KIMI_K25_NUM_EXPERTS = 384;
constexpr size_t KIMI_K25_CUBLASLT_WORKSPACE_SIZE = 32 * 1024 * 1024;

int32_t next_power_of_2(int32_t value) {
  int32_t power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

std::array<int, 8> kimi_k25_nvjet_algo_attrs(int32_t num_tokens) {
  int32_t const mp2 = std::max(next_power_of_2(num_tokens), 8);
  if (num_tokens < 8) {
    return {66, 12, 35, 1, 0, 0, 0, 4};
  }
  if (mp2 <= 16) {
    // Selects nvjet_sm100_tss_32x64_64x16_4x1_v_bz_splitK_TNN.
    return {66, 12, 35, 2, 2, 0, 0, 4};
  }
  if (mp2 <= 512) {
    // Selects nvjet_sm100_tss_32x64_64x16_4x1_v_bz_splitK_TNN.
    return {66, 12, 35, 4, 2, 0, 0, 4};
  }
  // Large token counts use the non-split-K tactic from TRT-LLM's LUT shape.
  return {66, 13, 35, 1, 0, 0, 1, 3};
}

void set_cublaslt_algo_attr(cublasLtMatmulAlgo_t& algo,
                            std::array<int, 8> const& attrs) {
  auto const [algo_id, tile_id, stages_id, split_k, reduction, swizzle,
              custom_option, cluster_shape] = attrs;
  uint32_t const custom_option_u32 = static_cast<uint32_t>(custom_option);
  uint16_t const cluster_shape_u16 = static_cast<uint16_t>(cluster_shape);
  (void)algo_id;

  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(tile_id)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_id, sizeof(stages_id)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction,
      sizeof(reduction)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &custom_option_u32,
      sizeof(custom_option_u32)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoConfigSetAttribute(
      &algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &cluster_shape_u16,
      sizeof(cluster_shape_u16)));
}

}  // namespace

torch::Tensor kimi_k25_router_gemm_bf16_fp32_cublaslt(
    torch::Tensor const& input, torch::Tensor const& weight) {
  TORCH_CHECK(input.is_cuda(), "Kimi-K2.5 router input must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "Kimi-K2.5 router weight must be CUDA");
  TORCH_CHECK(input.dtype() == torch::kBFloat16,
              "Kimi-K2.5 router input must be bfloat16");
  TORCH_CHECK(weight.dtype() == torch::kBFloat16,
              "Kimi-K2.5 router weight must be bfloat16");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2,
              "Kimi-K2.5 router input and weight must be 2-D");
  TORCH_CHECK(input.size(1) == KIMI_K25_HIDDEN_DIM,
              "Kimi-K2.5 router input hidden dim must be 7168");
  TORCH_CHECK(weight.size(0) == KIMI_K25_NUM_EXPERTS &&
                  weight.size(1) == KIMI_K25_HIDDEN_DIM,
              "Kimi-K2.5 router weight must have shape [384, 7168]");
  TORCH_CHECK(input.stride(1) == 1 && weight.stride(1) == 1,
              "Kimi-K2.5 router input and weight must be row-major");

  c10::cuda::CUDAGuard device_guard(input.device());

  int64_t const M64 = input.size(0);
  int64_t const N64 = weight.size(0);
  int64_t const K64 = input.size(1);
  TORCH_CHECK(M64 > 0 && M64 <= std::numeric_limits<int>::max(),
              "Kimi-K2.5 router token count is out of cuBLASLt range");

  int const M = static_cast<int>(M64);
  int const N = static_cast<int>(N64);
  int const K = static_cast<int>(K64);

  auto out = torch::empty({M64, N64}, input.options().dtype(torch::kFloat32));
  auto workspace = torch::empty(
      {static_cast<int64_t>(KIMI_K25_CUBLASLT_WORKSPACE_SIZE)},
      input.options().dtype(torch::kUInt8));

  static thread_local cublasLtHandle_t handle = [] {
    cublasLtHandle_t handle;
    TORCH_CUDABLAS_CHECK(cublasLtCreate(&handle));
    return handle;
  }();

  cublasLtMatmulDesc_t operation_desc = nullptr;
  cublasLtMatrixLayout_t input_desc = nullptr;
  cublasLtMatrixLayout_t weight_desc = nullptr;
  cublasLtMatrixLayout_t output_desc = nullptr;

  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescCreate(
      &operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t const trans_input = CUBLAS_OP_T;
  cublasOperation_t const trans_weight = CUBLAS_OP_N;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_input,
      sizeof(trans_input)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_weight,
      sizeof(trans_weight)));

  // The row-major [M, K] input is interpreted as column-major [K, M], then
  // transposed by cuBLASLt. The row-major [N, K] gate weight is interpreted as
  // column-major [K, N]. The output descriptor is explicitly row-major [M, N],
  // avoiding the transposed-output layout that selects the TNT NVJET tactic.
  cublasLtOrder_t const output_order = CUBLASLT_ORDER_ROW;
  TORCH_CUDABLAS_CHECK(
      cublasLtMatrixLayoutCreate(&input_desc, CUDA_R_16BF, K, M, K));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatrixLayoutCreate(&weight_desc, CUDA_R_16BF, K, N, K));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatrixLayoutCreate(&output_desc, CUDA_R_32F, M, N, N));
  TORCH_CUDABLAS_CHECK(
      cublasLtMatrixLayoutSetAttribute(output_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &output_order, sizeof(output_order)));

  cublasLtMatmulAlgo_t algo;
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoInit(
      handle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
      CUDA_R_32F, CUDA_R_32F, 66, &algo));
  set_cublaslt_algo_attr(algo, kimi_k25_nvjet_algo_attrs(M));

  float const alpha = 1.0f;
  float const beta = 0.0f;
  TORCH_CUDABLAS_CHECK(cublasLtMatmul(
      handle, operation_desc, &alpha, input.data_ptr(), input_desc,
      weight.data_ptr(), weight_desc, &beta, out.data_ptr(), output_desc,
      out.data_ptr(), output_desc, &algo, workspace.data_ptr(),
      KIMI_K25_CUBLASLT_WORKSPACE_SIZE, at::cuda::getCurrentCUDAStream()));

  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(output_desc));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(weight_desc));
  TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutDestroy(input_desc));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescDestroy(operation_desc));

  return out;
}
