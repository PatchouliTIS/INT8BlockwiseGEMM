// Bare CUTLASS SM80 grouped INT8 GEMM for vLLM MoE bring-up.

#include <int8_grouped_gemm.h>
#include <int8_hybrid_common.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/kernel/gemm_grouped.h>
#include <cutlass/layout/matrix.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = int32_t;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, 16, ElementB, LayoutB,
    cutlass::ComplexTransform::kNone, 16, ElementOutput, LayoutC,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    4>::GemmKernel;

using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

static_assert(sizeof(cutlass::gemm::GemmCoord) == 3 * sizeof(int32_t));
static_assert(alignof(cutlass::gemm::GemmCoord) == alignof(int32_t));

void check_inputs(torch::Tensor const& a, torch::Tensor const& b,
                  torch::Tensor const& out, torch::Tensor const& expert_offsets,
                  torch::Tensor const& problem_sizes) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(),
              "a, b, and out must be CUDA tensors");
  TORCH_CHECK(expert_offsets.is_cuda() && problem_sizes.is_cuda(),
              "expert_offsets and problem_sizes must be CUDA tensors");
  TORCH_CHECK(b.device() == a.device() && out.device() == a.device() &&
                  expert_offsets.device() == a.device() &&
                  problem_sizes.device() == a.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(),
              "a, b, and out must be contiguous");
  TORCH_CHECK(expert_offsets.is_contiguous() && problem_sizes.is_contiguous(),
              "expert_offsets and problem_sizes must be contiguous");
  TORCH_CHECK(a.scalar_type() == torch::kInt8, "a must be int8");
  TORCH_CHECK(b.scalar_type() == torch::kInt8, "b must be int8");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(a.dim() == 2, "a must have shape [sum_M, K]");
  TORCH_CHECK(b.dim() == 3, "b must have shape [E, N, K]");
  TORCH_CHECK(out.dim() == 2, "out must have shape [sum_M, N]");
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
              "problem_sizes must have shape [E, 3]");
  TORCH_CHECK(expert_offsets.dim() == 1,
              "expert_offsets must have shape [E + 1]");

  int64_t experts = b.size(0);
  int64_t n = b.size(1);
  int64_t k = b.size(2);
  TORCH_CHECK(problem_sizes.size(0) == experts,
              "problem_sizes.shape[0] must equal b.shape[0]");
  TORCH_CHECK(expert_offsets.size(0) == experts + 1,
              "expert_offsets must have E + 1 entries");
  TORCH_CHECK(a.size(1) == k, "a.shape[1] must equal b.shape[2]");
  TORCH_CHECK(out.size(0) == a.size(0), "out.shape[0] must equal a.shape[0]");
  TORCH_CHECK(out.size(1) == n, "out.shape[1] must equal b.shape[1]");
  TORCH_CHECK(k % 32 == 0,
              "K must be divisible by 32 for SM80 INT8 tensor ops");
}

void copy_int32_tensor_to_host(torch::Tensor const& src,
                               std::vector<int32_t>& dst, cudaStream_t stream) {
  dst.resize(src.numel());
  C10_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data_ptr<int32_t>(),
                                 dst.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream));
}

}  // namespace

void cutlass_int8_grouped_mm_host(torch::Tensor a, torch::Tensor b,
                                  torch::Tensor out, double a_scale,
                                  double b_scale, torch::Tensor expert_offsets,
                                  torch::Tensor problem_sizes) {
  check_inputs(a, b, out, expert_offsets, problem_sizes);
  c10::cuda::CUDAGuard device_guard(a.device());

  auto stream = at::cuda::getCurrentCUDAStream();
  std::vector<int32_t> h_offsets;
  std::vector<int32_t> h_problem_sizes_raw;
  copy_int32_tensor_to_host(expert_offsets, h_offsets, stream);
  copy_int32_tensor_to_host(problem_sizes, h_problem_sizes_raw, stream);
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  int experts = static_cast<int>(b.size(0));
  int n = static_cast<int>(b.size(1));
  int k = static_cast<int>(b.size(2));
  int sum_m = static_cast<int>(a.size(0));

  TORCH_CHECK(h_offsets.front() == 0, "expert_offsets[0] must be 0");
  TORCH_CHECK(h_offsets.back() == sum_m,
              "expert_offsets[-1] must equal a.shape[0]");

  std::vector<cutlass::gemm::GemmCoord> h_problem_sizes(experts);
  std::vector<int64_t> h_lda(experts, k);
  std::vector<int64_t> h_ldb(experts, k);
  std::vector<int64_t> h_ldc(experts, n);
  std::vector<int64_t> h_ldd(experts, n);
  std::vector<ElementA*> h_ptr_a(experts);
  std::vector<ElementB*> h_ptr_b(experts);
  std::vector<ElementOutput*> h_ptr_c(experts);
  std::vector<ElementOutput*> h_ptr_d(experts);

  auto* a_ptr = static_cast<ElementA*>(a.data_ptr());
  auto* b_ptr = static_cast<ElementB*>(b.data_ptr());
  auto* out_ptr = reinterpret_cast<ElementOutput*>(out.data_ptr());

  for (int e = 0; e < experts; ++e) {
    int m_e = h_problem_sizes_raw[e * 3 + 0];
    int n_e = h_problem_sizes_raw[e * 3 + 1];
    int k_e = h_problem_sizes_raw[e * 3 + 2];
    TORCH_CHECK(m_e == h_offsets[e + 1] - h_offsets[e],
                "problem_sizes[e, 0] must match expert_offsets delta");
    TORCH_CHECK(n_e == n, "all problems must use N == b.shape[1]");
    TORCH_CHECK(k_e == k, "all problems must use K == b.shape[2]");
    h_problem_sizes[e] = cutlass::gemm::GemmCoord(m_e, n_e, k_e);
    h_ptr_a[e] = a_ptr + static_cast<int64_t>(h_offsets[e]) * k;
    h_ptr_b[e] = b_ptr + static_cast<int64_t>(e) * n * k;
    h_ptr_c[e] = out_ptr + static_cast<int64_t>(h_offsets[e]) * n;
    h_ptr_d[e] = out_ptr + static_cast<int64_t>(h_offsets[e]) * n;
  }

  auto ptr_options = torch::dtype(torch::kInt64).device(a.device());
  auto ptr_a_tensor = torch::empty({experts}, ptr_options);
  auto ptr_b_tensor = torch::empty({experts}, ptr_options);
  auto ptr_c_tensor = torch::empty({experts}, ptr_options);
  auto ptr_d_tensor = torch::empty({experts}, ptr_options);
  auto lda_tensor = torch::empty({experts}, ptr_options);
  auto ldb_tensor = torch::empty({experts}, ptr_options);
  auto ldc_tensor = torch::empty({experts}, ptr_options);
  auto ldd_tensor = torch::empty({experts}, ptr_options);

  C10_CUDA_CHECK(cudaMemcpyAsync(ptr_a_tensor.data_ptr<int64_t>(),
                                 h_ptr_a.data(), experts * sizeof(ElementA*),
                                 cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(ptr_b_tensor.data_ptr<int64_t>(),
                                 h_ptr_b.data(), experts * sizeof(ElementB*),
                                 cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      ptr_c_tensor.data_ptr<int64_t>(), h_ptr_c.data(),
      experts * sizeof(ElementOutput*), cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      ptr_d_tensor.data_ptr<int64_t>(), h_ptr_d.data(),
      experts * sizeof(ElementOutput*), cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(lda_tensor.data_ptr<int64_t>(), h_lda.data(),
                                 experts * sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(ldb_tensor.data_ptr<int64_t>(), h_ldb.data(),
                                 experts * sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(ldc_tensor.data_ptr<int64_t>(), h_ldc.data(),
                                 experts * sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(ldd_tensor.data_ptr<int64_t>(), h_ldd.data(),
                                 experts * sizeof(int64_t),
                                 cudaMemcpyHostToDevice, stream));

  int threadblock_count =
      GemmGrouped::sufficient(h_problem_sizes.data(), experts);
  TORCH_CHECK(threadblock_count > 0,
              "CUTLASS grouped GEMM has no runnable threadblocks");

  typename GemmGrouped::EpilogueOutputOp::Params epilogue(
      static_cast<float>(a_scale * b_scale), 0.0f);
  typename GemmGrouped::Arguments args(
      reinterpret_cast<cutlass::gemm::GemmCoord*>(
          problem_sizes.data_ptr<int32_t>()),
      experts, threadblock_count, epilogue,
      reinterpret_cast<ElementA**>(ptr_a_tensor.data_ptr<int64_t>()),
      reinterpret_cast<ElementB**>(ptr_b_tensor.data_ptr<int64_t>()),
      reinterpret_cast<ElementOutput**>(ptr_c_tensor.data_ptr<int64_t>()),
      reinterpret_cast<ElementOutput**>(ptr_d_tensor.data_ptr<int64_t>()),
      lda_tensor.data_ptr<int64_t>(), ldb_tensor.data_ptr<int64_t>(),
      ldc_tensor.data_ptr<int64_t>(), ldd_tensor.data_ptr<int64_t>(),
      h_problem_sizes.data());

  GemmGrouped gemm;
  CUTLASS_CHECK(gemm.can_implement(args));
  size_t workspace_size = GemmGrouped::get_workspace_size(args);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                torch::dtype(torch::kUInt8).device(a.device()));
  CUTLASS_CHECK(gemm.initialize(args, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
