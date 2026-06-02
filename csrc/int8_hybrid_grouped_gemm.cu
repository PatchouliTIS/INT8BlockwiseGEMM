// CUTLASS SM80 INT8 hybrid grouped GEMM kernels for vLLM MoE.

#include <int8_hybrid_grouped_gemm.h>
#include <int8_hybrid_common.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator.h>
#include <cutlass/epilogue/threadblock/shared_load_iterator_mixed.h>
#include <cutlass/epilogue/warp/fragment_iterator_tensor_op.h>
#include <cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_grouped_problem_visitor.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "custom_mma_multistage.h"

namespace {

using ElementA = int8_t;
using ElementB = int8_t;
using ElementAccum = int32_t;
using ElementOutput = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

static constexpr int kAlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int kAlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

static_assert(sizeof(cutlass::gemm::GemmCoord) == 3 * sizeof(int32_t));
static_assert(alignof(cutlass::gemm::GemmCoord) == alignof(int32_t));

template <bool kHasBias, typename TileShape_, typename WarpShape_,
          typename FP32Accum_>
__device__ __forceinline__ void apply_bias_to_accum(
    FP32Accum_& fp32_accum, const float* __restrict__ ptr_bias, int cta_n,
    int warp_idx, int lane_idx, int N) {
  if constexpr (!kHasBias) return;

  static constexpr int kInstN = 8;
  static constexpr int kWarpCountM = TileShape_::kM / WarpShape_::kM;
  static constexpr int kMmaOpsM = WarpShape_::kM / 16;

  int warp_mn = warp_idx % (kWarpCountM * (TileShape_::kN / WarpShape_::kN));
  int warp_n = warp_mn / kWarpCountM;
  int lane_col_base = (lane_idx % 4) * 2;
  int col_warp_base = cta_n * TileShape_::kN + warp_n * WarpShape_::kN;

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < FP32Accum_::kElements; ++i) {
    int elem_in_mma = i % 4;
    int mma_idx = i / 4;
    int n_op = mma_idx / kMmaOpsM;
    int col = col_warp_base + n_op * kInstN + lane_col_base + (elem_in_mma % 2);
    if (col < N) {
      fp32_accum[i] += ptr_bias[col];
    }
  }
}

template <typename TileShape_, typename WarpShape_, int kStages_>
struct HybridGroupedConfig {
  using TileShape = TileShape_;
  using WarpShape = WarpShape_;
  static constexpr int kStages = kStages_;

  using DefaultMma = cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccum, LayoutOutput, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, TileShape, WarpShape, InstructionShape, kStages,
      cutlass::arch::OpMultiplyAddSaturate, false,
      cutlass::gemm::SharedMemoryClearOption::kNone>;

  using IteratorA = typename DefaultMma::IteratorA;
  using IteratorB = typename DefaultMma::IteratorB;

  using Mma = custom_mma::MmaMultistage<
      typename DefaultMma::MmaCore::Shape, IteratorA,
      typename DefaultMma::MmaCore::SmemIteratorA,
      DefaultMma::MmaCore::kCacheOpA, IteratorB,
      typename DefaultMma::MmaCore::SmemIteratorB,
      DefaultMma::MmaCore::kCacheOpB, ElementAccum, LayoutOutput,
      typename DefaultMma::MmaCore::MmaPolicy, kStages>;
  using FragmentC = typename Mma::FragmentC;

  using WarpMmaOperator = typename DefaultMma::MmaCore::MmaPolicy::Operator;
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using OperatorShape = typename ArchMmaOperator::Shape;
  using OperatorFragmentC = typename ArchMmaOperator::FragmentC;

  static constexpr int kEpilogueElementsPerAccess =
      128 / cutlass::sizeof_bits<ElementOutput>::value;

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, kEpilogueElementsPerAccess, float, float>;

  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
          TileShape, WarpShape, 1, ElementOutput,
          kEpilogueElementsPerAccess>::Type;

  using OutputTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIterator<
          OutputTileThreadMap, ElementOutput>;

  using AccumulatorFragmentIterator =
      cutlass::epilogue::warp::FragmentIteratorTensorOp<
          WarpShape, OperatorShape, float,
          cutlass::Array<float, OperatorFragmentC::kElements>,
          cutlass::layout::RowMajor>;

  using WarpTileIterator = cutlass::epilogue::warp::TileIteratorTensorOpMixed<
      WarpShape, OperatorShape, float, 32, 16, 8, 8>;

  using SharedLoadIterator =
      cutlass::epilogue::threadblock::SharedLoadIteratorMixed<
          typename OutputTileThreadMap::CompactedThreadMap, float, 32, 16, 8,
          8>;

  using Padding = typename WarpTileIterator::Padding;
  static constexpr int kFragmentsPerIteration =
      WarpShape::kN / OperatorShape::kN;

  struct FakeWarpMma {
    using Shape = WarpShape;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementC = float;
    struct FakeOperator {
      using Shape = OperatorShape;
      using ElementC = float;
      using FragmentC = cutlass::Array<float, OperatorFragmentC::kElements>;
    };
    struct FakePolicy {
      using Operator = FakeOperator;
    };
    using Policy = FakePolicy;
  };

  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      TileShape, FakeWarpMma, 1, OutputTileIterator,
      AccumulatorFragmentIterator, WarpTileIterator, SharedLoadIterator,
      EpilogueOutputOp, Padding, kFragmentsPerIteration>;

  using FP32AccumulatorTile =
      typename AccumulatorFragmentIterator::AccumulatorTile;

  static constexpr int kThreadCount = Mma::WarpCount::kCount * 32;
  using ProblemVisitor = cutlass::gemm::kernel::GemmGroupedProblemVisitor<
      TileShape, cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly,
      kThreadCount, kThreadCount>;

  struct SharedStorage {
    union KernelStorage {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    } kernel;
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

  struct Params {
    typename ProblemVisitor::Params problem_visitor;
    typename IteratorA::Params params_A;
    typename IteratorB::Params params_B;
    typename OutputTileIterator::Params params_D;
    const ElementA* ptr_A_base;
    const ElementB* ptr_B_base;
    ElementOutput* ptr_D_base;
    const int32_t* ptr_Q_A_base;
    const int32_t* ptr_Q_B_base;
    const float* ptr_F_A_base;
    const float* ptr_F_B_base;
    const float* ptr_bias_base;
    const int32_t* expert_offsets;
    // P0-A: precomputed prefix sum of ceil(M_e / TileM) over experts so each
    // CTA can resolve its A-side scale-row base in O(1) instead of an O(E)
    // device-side scan. Optional: when nullptr the kernel falls back to the
    // legacy O(E) scan path (kept for backwards compatibility / safety).
    const int32_t* ptr_scale_offsets;
    int k_tiles_per_qb;
    int q_stride;
    int f_stride;
    int n_scale_blocks;
    int bias_stride;
  };
};

using DefaultConfig =
    HybridGroupedConfig<cutlass::gemm::GemmShape<128, 64, 128>,
                        cutlass::gemm::GemmShape<64, 32, 128>, 3>;
using SmallConfig =
    HybridGroupedConfig<cutlass::gemm::GemmShape<64, 64, 128>,
                        cutlass::gemm::GemmShape<32, 32, 128>, 3>;

template <typename Config, bool kHasBias, int kTilesPerQbCT = 0>
__global__ void blockwise_fused_gemm_kernel_hybrid_grouped(
    typename Config::Params params) {
  extern __shared__ char smem_buf[];
  typename Config::SharedStorage& shared_storage =
      *reinterpret_cast<typename Config::SharedStorage*>(smem_buf);

  int thread_idx = threadIdx.x;
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_idx = threadIdx.x % 32;

  typename Config::ProblemVisitor visitor(
      params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);

  while (visitor.next_tile()) {
    cutlass::gemm::GemmCoord problem = visitor.problem_size();
    int problem_idx = visitor.problem_index();
    int threadblock_idx = visitor.threadblock_idx();
    cutlass::gemm::GemmCoord grid_shape = visitor.grid_shape(problem);

    int cta_m = threadblock_idx / grid_shape.n();
    int cta_n = threadblock_idx % grid_shape.n();

    int M = problem.m();
    int N = problem.n();
    int K = problem.k();

    int expert_m_offset = params.expert_offsets[problem_idx];
    // P0-A: O(1) scale-row base lookup. Precomputed on host/Triton side as
    // cumsum_e( ceil(M_e / TileM) ). Falls back to the legacy O(E) scan when
    // ptr_scale_offsets is nullptr to keep API backwards compatible.
    int scale_block_offset;
    if (params.ptr_scale_offsets != nullptr) {
      scale_block_offset = __ldg(params.ptr_scale_offsets + problem_idx);
    } else {
      scale_block_offset = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < problem_idx; ++i) {
        cutlass::gemm::GemmCoord prev_problem =
            params.problem_visitor.problem_sizes[i];
        scale_block_offset += (prev_problem.m() + Config::TileShape::kM - 1) /
                              Config::TileShape::kM;
      }
    }

    const ElementA* ptr_A =
        params.ptr_A_base + static_cast<int64_t>(expert_m_offset) * K;
    const ElementB* ptr_B =
        params.ptr_B_base + static_cast<int64_t>(problem_idx) * N * K;
    ElementOutput* ptr_D =
        params.ptr_D_base + static_cast<int64_t>(expert_m_offset) * N;
    const int32_t* ptr_Q_A =
        params.ptr_Q_A_base +
        static_cast<int64_t>(scale_block_offset) * params.q_stride;
    const int32_t* ptr_Q_B =
        params.ptr_Q_B_base + static_cast<int64_t>(problem_idx) *
                                  params.n_scale_blocks * params.q_stride;
    const float* ptr_F_A =
        params.ptr_F_A_base +
        static_cast<int64_t>(scale_block_offset) * params.f_stride;
    const float* ptr_F_B =
        params.ptr_F_B_base + static_cast<int64_t>(problem_idx) *
                                  params.n_scale_blocks * params.f_stride;
    const float* ptr_bias = nullptr;
    if constexpr (kHasBias) {
      ptr_bias = params.ptr_bias_base +
                 static_cast<int64_t>(problem_idx) * params.bias_stride;
    }

    cutlass::MatrixCoord tb_offset_A{cta_m * Config::TileShape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{0, cta_n * Config::TileShape::kN};

    typename Config::IteratorA iterator_A(
        params.params_A, const_cast<ElementA*>(ptr_A),
        cutlass::MatrixCoord(M, K), thread_idx, tb_offset_A);

    typename Config::IteratorB iterator_B(
        params.params_B, const_cast<ElementB*>(ptr_B),
        cutlass::MatrixCoord(K, N), thread_idx, tb_offset_B);

    int gemm_k_iterations =
        (K + Config::TileShape::kK - 1) / Config::TileShape::kK;

    __syncthreads();

    typename Config::Mma mma(shared_storage.kernel.main_loop, thread_idx,
                             warp_idx, lane_idx);
    mma.prologue(iterator_A, iterator_B, gemm_k_iterations);
    mma.gmem_wait();

    typename Config::FragmentC int32_accum;
    int32_accum.clear();

    typename Config::FragmentC int32_weighted;
    int32_weighted.clear();

    typename Config::Mma::PipeState pipe_state;
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    mma.warp_tile_iterator_A_.set_kgroup_index(0);
    mma.warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++mma.warp_tile_iterator_A_;

    mma.warp_tile_iterator_B_.set_kgroup_index(0);
    mma.warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++mma.warp_tile_iterator_B_;

    mma.warp_mma_.transform(pipe_state.warp_transformed_frag_A_[0],
                            pipe_state.warp_transformed_frag_B_[0],
                            pipe_state.warp_loaded_frag_A_[0],
                            pipe_state.warp_loaded_frag_B_[0]);

    int total_k_tiles = (K + Config::TileShape::kK - 1) / Config::TileShape::kK;
    // C2: when kTilesPerQbCT > 0 the divisor is a non-type template parameter
    // and nvcc constant-folds the (kt+1) % k_tiles_per_qb / kt / k_tiles_per_qb
    // chain in the boundary check below, eliminating the MUFU.RCP +
    // IMAD.HI.U32 + IMAD.IADD sequence that NCU showed at
    // int8_hybrid_grouped_gemm.cu:402 as the top wait/mio_throttle hot-PC in
    // baseline. kTilesPerQbCT == 0 preserves the legacy runtime-divide path for
    // safety / unsupported sizes.
    int k_tiles_per_qb =
        (kTilesPerQbCT > 0) ? kTilesPerQbCT : params.k_tiles_per_qb;

    static constexpr int kWarpGemmIter = Config::Mma::Base::kWarpGemmIterations;

    for (int kt = 0; kt < total_k_tiles; ++kt) {
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIter; ++warp_mma_k) {
        mma.warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) %
                                                   kWarpGemmIter);
        mma.warp_tile_iterator_A_.load(
            pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
        ++mma.warp_tile_iterator_A_;

        mma.warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) %
                                                   kWarpGemmIter);
        mma.warp_tile_iterator_B_.load(
            pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
        ++mma.warp_tile_iterator_B_;

        if (warp_mma_k > 0) {
          mma.warp_mma_.transform(
              pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
              pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
              pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
              pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
        }

        mma.warp_mma_(
            int32_accum, pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
            pipe_state.warp_transformed_frag_B_[warp_mma_k % 2], int32_accum);

        if (warp_mma_k < kWarpGemmIter - 1) {
          int group_start_A =
              warp_mma_k * Config::Mma::Detail::kAccessesPerGroupA;
          int group_start_B =
              warp_mma_k * Config::Mma::Detail::kAccessesPerGroupB;
          mma.copy_tiles_and_advance(iterator_A, iterator_B, group_start_A,
                                     group_start_B);
        }

        if (warp_mma_k + 2 == kWarpGemmIter) {
          int group_start_A =
              (warp_mma_k + 1) * Config::Mma::Detail::kAccessesPerGroupA;
          int group_start_B =
              (warp_mma_k + 1) * Config::Mma::Detail::kAccessesPerGroupB;
          mma.copy_tiles_and_advance(iterator_A, iterator_B, group_start_A,
                                     group_start_B);

          cutlass::arch::cp_async_fence();
          mma.gmem_wait();
          mma.advance_smem_write_stage(iterator_A, iterator_B);
          mma.advance_smem_read_stage();

          --gemm_k_iterations;
          iterator_A.clear_mask(gemm_k_iterations == 0);
          iterator_B.clear_mask(gemm_k_iterations == 0);
        }

        if (warp_mma_k + 1 == kWarpGemmIter) {
          mma.warp_mma_.transform(
              pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
              pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
              pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
              pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
        }
      }

      // C2: compile-time-folded boundary check + qb_idx. See top-of-file
      // template parameter kTilesPerQbCT and run_003_C2/preflight/PREFLIGHT.md
      // for the SASS-level proof that the production CT==1 case lowers to a
      // no-op test (no MUFU.RCP / IMAD.HI / IMAD.IADD chain, ~73% fewer SASS
      // instructions in the relevant skeleton).
      bool qb_boundary;
      int qb_idx;
      if constexpr (kTilesPerQbCT == 1) {
        qb_boundary = true;
        qb_idx = kt;
      } else if constexpr (kTilesPerQbCT > 0) {
        // kTilesPerQbCT is a known power-of-two compile-time constant
        // (dispatcher only emits CT in {1, 2, 4}). nvcc lowers
        //   (kt + 1) % CT  -> (kt + 1) & (CT - 1)   [LOP3]
        //   kt / CT        -> kt >> log2(CT)        [SHR.U32]
        qb_boundary = (((kt + 1) & (kTilesPerQbCT - 1)) == 0) ||
                      (kt == total_k_tiles - 1);
        qb_idx = kt / kTilesPerQbCT;
      } else {
        // Legacy runtime path; preserved as a safety net for any future
        // quant_block_size / TileK ratio outside {1, 2, 4}. Same SASS as
        // baseline pre-C2.
        qb_boundary =
            ((kt + 1) % k_tiles_per_qb == 0) || (kt == total_k_tiles - 1);
        qb_idx = kt / k_tiles_per_qb;
      }
      if (qb_boundary) {
        int q_combined = __ldg(ptr_Q_A + cta_m * params.q_stride + qb_idx) *
                         __ldg(ptr_Q_B + cta_n * params.q_stride + qb_idx);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Config::FragmentC::kElements; ++i) {
          int32_weighted[i] += q_combined * int32_accum[i];
        }
        int32_accum.clear();
      }
    }

    float fa = __ldg(ptr_F_A + cta_m * params.f_stride);
    float fb = __ldg(ptr_F_B + cta_n * params.f_stride);
    float F_final = fa * fb;

    typename Config::FP32AccumulatorTile fp32_accum;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Config::FP32AccumulatorTile::kElements; ++i) {
      fp32_accum[i] = static_cast<float>(int32_weighted[i]) * F_final;
    }

    apply_bias_to_accum<kHasBias, typename Config::TileShape,
                        typename Config::WarpShape,
                        typename Config::FP32AccumulatorTile>(
        fp32_accum, ptr_bias, cta_n, warp_idx, lane_idx, N);

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    cutlass::MatrixCoord threadblock_offset{cta_m * Config::TileShape::kM,
                                            cta_n * Config::TileShape::kN};

    typename Config::OutputTileIterator iterator_D(
        params.params_D, ptr_D, cutlass::MatrixCoord(M, N), thread_idx,
        threadblock_offset);

    typename Config::OutputTileIterator iterator_C = iterator_D;
    typename Config::EpilogueOutputOp output_op({1.0f, 0.0f});
    typename Config::Epilogue epilogue(shared_storage.kernel.epilogue,
                                       thread_idx, warp_idx, lane_idx);
    epilogue(output_op, iterator_D, fp32_accum, iterator_C);

    visitor.advance(gridDim.x);
  }
}

void copy_int32_tensor_to_host(torch::Tensor const& src,
                               std::vector<int32_t>& dst, cudaStream_t stream) {
  dst.resize(src.numel());
  C10_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data_ptr<int32_t>(),
                                 dst.size() * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream));
}

int ceil_div_int(int x, int y) { return (x + y - 1) / y; }

void check_base_inputs(torch::Tensor const& a_q, torch::Tensor const& a_qscale,
                       torch::Tensor const& a_fscale, torch::Tensor const& b_q,
                       torch::Tensor const& b_qscale,
                       torch::Tensor const& b_fscale,
                       torch::Tensor const& expert_offsets,
                       torch::Tensor const& problem_sizes,
                       torch::Tensor const& out) {
  TORCH_CHECK(a_q.is_cuda() && a_qscale.is_cuda() && a_fscale.is_cuda() &&
                  b_q.is_cuda() && b_qscale.is_cuda() && b_fscale.is_cuda() &&
                  expert_offsets.is_cuda() && problem_sizes.is_cuda() &&
                  out.is_cuda(),
              "all grouped hybrid tensors must be CUDA tensors");
  TORCH_CHECK(a_q.device() == b_q.device() &&
                  a_qscale.device() == a_q.device() &&
                  a_fscale.device() == a_q.device() &&
                  b_qscale.device() == a_q.device() &&
                  b_fscale.device() == a_q.device() &&
                  expert_offsets.device() == a_q.device() &&
                  problem_sizes.device() == a_q.device() &&
                  out.device() == a_q.device(),
              "all grouped hybrid tensors must be on the same device");
  TORCH_CHECK(a_q.is_contiguous() && a_qscale.is_contiguous() &&
                  a_fscale.is_contiguous() && b_q.is_contiguous() &&
                  b_qscale.is_contiguous() && b_fscale.is_contiguous() &&
                  expert_offsets.is_contiguous() &&
                  problem_sizes.is_contiguous() && out.is_contiguous(),
              "all grouped hybrid tensors must be contiguous");
  TORCH_CHECK(a_q.scalar_type() == torch::kInt8, "a_q must be int8");
  TORCH_CHECK(b_q.scalar_type() == torch::kInt8, "b_q must be int8");
  TORCH_CHECK(a_qscale.scalar_type() == torch::kInt32,
              "a_qscale must be int32");
  TORCH_CHECK(b_qscale.scalar_type() == torch::kInt32,
              "b_qscale must be int32");
  TORCH_CHECK(a_fscale.scalar_type() == torch::kFloat32,
              "a_fscale must be float32");
  TORCH_CHECK(b_fscale.scalar_type() == torch::kFloat32,
              "b_fscale must be float32");
  TORCH_CHECK(expert_offsets.scalar_type() == torch::kInt32,
              "expert_offsets must be int32");
  TORCH_CHECK(problem_sizes.scalar_type() == torch::kInt32,
              "problem_sizes must be int32");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");

  TORCH_CHECK(a_q.dim() == 2, "a_q must have shape [sum_M, K]");
  TORCH_CHECK(a_qscale.dim() == 2,
              "a_qscale must have shape [sum_e ceil(M_e / TileM), K_blocks]");
  TORCH_CHECK(
      a_fscale.dim() == 2,
      "a_fscale must have shape [sum_e ceil(M_e / TileM), num_super_groups]");
  TORCH_CHECK(b_q.dim() == 3, "b_q must have shape [E, N, K]");
  TORCH_CHECK(b_qscale.dim() == 3,
              "b_qscale must have shape [E, ceil(N / TileN), K_blocks]");
  TORCH_CHECK(
      b_fscale.dim() == 3,
      "b_fscale must have shape [E, ceil(N / TileN), num_super_groups]");
  TORCH_CHECK(expert_offsets.dim() == 1,
              "expert_offsets must have shape [E + 1]");
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
              "problem_sizes must have shape [E, 3]");
  TORCH_CHECK(out.dim() == 2, "out must have shape [sum_M, N]");

  int64_t experts = b_q.size(0);
  int64_t n = b_q.size(1);
  int64_t k = b_q.size(2);
  TORCH_CHECK(problem_sizes.size(0) == experts,
              "problem_sizes.shape[0] must equal b_q.shape[0]");
  TORCH_CHECK(expert_offsets.size(0) == experts + 1,
              "expert_offsets must have E + 1 entries");
  TORCH_CHECK(a_q.size(1) == k, "a_q.shape[1] must equal b_q.shape[2]");
  TORCH_CHECK(out.size(0) == a_q.size(0),
              "out.shape[0] must equal a_q.shape[0]");
  TORCH_CHECK(out.size(1) == n, "out.shape[1] must equal b_q.shape[1]");
  TORCH_CHECK(k % 32 == 0,
              "K must be divisible by 32 for SM80 INT8 tensor ops");
}

// C2 dispatcher helper. Holds the cudaFuncSetAttribute / occupancy probe /
// <<<...>>>  launch path so that each (Config, kHasBias, kTilesPerQbCT)
// instantiation gets its own dynamic-smem opt-in and its own occupancy cache
// (function-local statics naturally key on the template specialization).
template <typename Config, bool kHasBias, int kTilesPerQbCT>
void dispatch_launch_kernel_with_qb_ct(typename Config::Params const& params,
                                       int threadblock_count, int dev_id,
                                       cudaStream_t stream) {
  dim3 block(Config::kThreadCount);
  int smem_size = static_cast<int>(sizeof(typename Config::SharedStorage));

  static bool attr_set = false;
  if (smem_size > 48 * 1024 && !attr_set) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        blockwise_fused_gemm_kernel_hybrid_grouped<Config, kHasBias,
                                                   kTilesPerQbCT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    attr_set = true;
  }

  static int cached_dev_id = -1;
  static int cached_resident_grid = 0;
  if (cached_dev_id != dev_id) {
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, dev_id));

    int max_active_blocks_per_sm = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_per_sm,
        blockwise_fused_gemm_kernel_hybrid_grouped<Config, kHasBias,
                                                   kTilesPerQbCT>,
        Config::kThreadCount, smem_size));
    TORCH_CHECK(max_active_blocks_per_sm > 0,
                "grouped hybrid kernel has zero active blocks per SM");

    cached_resident_grid = props.multiProcessorCount * max_active_blocks_per_sm;
    cached_dev_id = dev_id;
  }

  int persistent_grid = std::min(threadblock_count, cached_resident_grid);
  dim3 grid(persistent_grid, 1, 1);

  blockwise_fused_gemm_kernel_hybrid_grouped<Config, kHasBias, kTilesPerQbCT>
      <<<grid, block, smem_size, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Config, bool kHasBias>
void launch_grouped_hybrid(torch::Tensor a_q, torch::Tensor a_qscale,
                           torch::Tensor a_fscale, torch::Tensor b_q,
                           torch::Tensor b_qscale, torch::Tensor b_fscale,
                           torch::Tensor expert_offsets,
                           torch::Tensor problem_sizes, torch::Tensor out,
                           torch::Tensor bias,
                           std::optional<torch::Tensor> scale_offsets,
                           int64_t quant_block_size, int64_t super_group_size) {
  check_base_inputs(a_q, a_qscale, a_fscale, b_q, b_qscale, b_fscale,
                    expert_offsets, problem_sizes, out);
  if (scale_offsets.has_value()) {
    TORCH_CHECK(scale_offsets->is_cuda(), "scale_offsets must be CUDA");
    TORCH_CHECK(scale_offsets->device() == a_q.device(),
                "scale_offsets must share device with a_q");
    TORCH_CHECK(scale_offsets->is_contiguous(),
                "scale_offsets must be contiguous");
    TORCH_CHECK(scale_offsets->scalar_type() == torch::kInt32,
                "scale_offsets must be int32");
    TORCH_CHECK(scale_offsets->dim() == 1, "scale_offsets must have shape [E]");
    TORCH_CHECK(scale_offsets->size(0) == b_q.size(0),
                "scale_offsets length must equal num experts");
  }
  if constexpr (kHasBias) {
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.device() == a_q.device(),
                "bias must be on the same device");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK((bias.dim() == 1 && bias.size(0) == b_q.size(1)) ||
                    (bias.dim() == 2 && bias.size(0) == b_q.size(0) &&
                     bias.size(1) == b_q.size(1)),
                "bias must have shape [N] or [E, N]");
  }

  TORCH_CHECK(quant_block_size > 0, "quant_block_size must be positive");
  TORCH_CHECK(super_group_size > 0, "super_group_size must be positive");
  TORCH_CHECK(quant_block_size % Config::TileShape::kK == 0,
              "quant_block_size must be divisible by the selected TileK");

  c10::cuda::CUDAGuard device_guard(a_q.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  int experts = static_cast<int>(b_q.size(0));
  int n = static_cast<int>(b_q.size(1));
  int k = static_cast<int>(b_q.size(2));
  int sum_m = static_cast<int>(a_q.size(0));
  int num_quant_blocks = ceil_div_int(k, static_cast<int>(quant_block_size));
  int k_tiles_per_qb =
      static_cast<int>(quant_block_size) / Config::TileShape::kK;
  int num_super_groups =
      ceil_div_int(num_quant_blocks, static_cast<int>(super_group_size));
  int n_scale_blocks = ceil_div_int(n, Config::TileShape::kN);

  TORCH_CHECK(a_qscale.size(1) == num_quant_blocks,
              "a_qscale.shape[1] must equal ceil(K / quant_block_size)");
  TORCH_CHECK(a_fscale.size(1) == num_super_groups,
              "a_fscale.shape[1] must equal num_super_groups");
  TORCH_CHECK(b_qscale.size(0) == experts &&
                  b_qscale.size(1) == n_scale_blocks &&
                  b_qscale.size(2) == num_quant_blocks,
              "b_qscale must have shape [E, ceil(N / TileN), K_blocks] for the "
              "selected variant");
  TORCH_CHECK(b_fscale.size(0) == experts &&
                  b_fscale.size(1) == n_scale_blocks &&
                  b_fscale.size(2) == num_super_groups,
              "b_fscale must have shape [E, ceil(N / TileN), num_super_groups] "
              "for the selected variant");

  int64_t threadblock_count_64 =
      static_cast<int64_t>(a_qscale.size(0)) * n_scale_blocks;
  if (threadblock_count_64 == 0) {
    return;
  }
  TORCH_CHECK(threadblock_count_64 <= std::numeric_limits<int>::max(),
              "grouped hybrid problem has too many threadblocks");
  int threadblock_count = static_cast<int>(threadblock_count_64);

  typename Config::Params params;
  params.problem_visitor = typename Config::ProblemVisitor::Params(
      reinterpret_cast<cutlass::gemm::GemmCoord*>(
          problem_sizes.data_ptr<int32_t>()),
      experts, nullptr, threadblock_count);
  params.params_A =
      typename Config::IteratorA::Params(LayoutA::packed({sum_m, k}));
  params.params_B = typename Config::IteratorB::Params(LayoutB::packed({k, n}));
  params.params_D = typename Config::OutputTileIterator::Params(
      LayoutOutput::packed({sum_m, n}));
  params.ptr_A_base = static_cast<const ElementA*>(a_q.data_ptr());
  params.ptr_B_base = static_cast<const ElementB*>(b_q.data_ptr());
  params.ptr_D_base = reinterpret_cast<ElementOutput*>(out.data_ptr());
  params.ptr_Q_A_base = static_cast<const int32_t*>(a_qscale.data_ptr());
  params.ptr_Q_B_base = static_cast<const int32_t*>(b_qscale.data_ptr());
  params.ptr_F_A_base = static_cast<const float*>(a_fscale.data_ptr());
  params.ptr_F_B_base = static_cast<const float*>(b_fscale.data_ptr());
  params.ptr_bias_base =
      kHasBias ? static_cast<const float*>(bias.data_ptr()) : nullptr;
  params.expert_offsets =
      static_cast<const int32_t*>(expert_offsets.data_ptr());
  params.ptr_scale_offsets =
      scale_offsets.has_value()
          ? static_cast<const int32_t*>(scale_offsets->data_ptr())
          : nullptr;
  params.k_tiles_per_qb = k_tiles_per_qb;
  params.q_stride = num_quant_blocks;
  params.f_stride = num_super_groups;
  params.n_scale_blocks = n_scale_blocks;
  params.bias_stride = kHasBias ? (bias.dim() == 2 ? n : 0) : 0;

  // C2: dispatch into one of the compile-time-folded instantiations whenever
  // k_tiles_per_qb is in the supported set {1, 2, 4}. All shipping configs
  // today use quant_block_size == 128 == TileK, i.e. CT == 1, which makes
  // the (kt+1) % k_tiles_per_qb / kt / k_tiles_per_qb chain in the K-loop
  // collapse to a constant-true boundary test and an identity index. Other
  // ratios fall back to the generic CT == 0 runtime-divide path so the API
  // remains backwards compatible. See
  // profile/run_003_C2/preflight/PREFLIGHT.md.
  int dev_id = a_q.get_device();
  switch (k_tiles_per_qb) {
    case 1:
      dispatch_launch_kernel_with_qb_ct<Config, kHasBias, 1>(
          params, threadblock_count, dev_id, stream);
      break;
    case 2:
      dispatch_launch_kernel_with_qb_ct<Config, kHasBias, 2>(
          params, threadblock_count, dev_id, stream);
      break;
    case 4:
      dispatch_launch_kernel_with_qb_ct<Config, kHasBias, 4>(
          params, threadblock_count, dev_id, stream);
      break;
    default:
      // Legacy runtime-divide path. Active only if a future config sets a
      // quant_block_size such that quant_block_size / TileK is not in
      // {1, 2, 4} (e.g. 8, 16). Strictly bit-identical to pre-C2 baseline.
      dispatch_launch_kernel_with_qb_ct<Config, kHasBias, 0>(
          params, threadblock_count, dev_id, stream);
      break;
  }
}

}  // namespace

void cutlass_int8_blockwise_hybrid_grouped_host(
    torch::Tensor a_q, torch::Tensor a_qscale, torch::Tensor a_fscale,
    torch::Tensor b_q, torch::Tensor b_qscale, torch::Tensor b_fscale,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    torch::Tensor out, int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets) {
  launch_grouped_hybrid<DefaultConfig, false>(
      a_q, a_qscale, a_fscale, b_q, b_qscale, b_fscale, expert_offsets,
      problem_sizes, out, torch::Tensor(), scale_offsets, quant_block_size,
      super_group_size);
}

void cutlass_int8_blockwise_hybrid_grouped_bias_host(
    torch::Tensor a_q, torch::Tensor a_qscale, torch::Tensor a_fscale,
    torch::Tensor b_q, torch::Tensor b_qscale, torch::Tensor b_fscale,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    torch::Tensor out, torch::Tensor bias,
    int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets) {
  launch_grouped_hybrid<DefaultConfig, true>(
      a_q, a_qscale, a_fscale, b_q, b_qscale, b_fscale, expert_offsets,
      problem_sizes, out, bias, scale_offsets, quant_block_size,
      super_group_size);
}

void cutlass_int8_blockwise_hybrid_grouped_small_host(
    torch::Tensor a_q, torch::Tensor a_qscale, torch::Tensor a_fscale,
    torch::Tensor b_q, torch::Tensor b_qscale, torch::Tensor b_fscale,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    torch::Tensor out, int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets) {
  launch_grouped_hybrid<SmallConfig, false>(
      a_q, a_qscale, a_fscale, b_q, b_qscale, b_fscale, expert_offsets,
      problem_sizes, out, torch::Tensor(), scale_offsets, quant_block_size,
      super_group_size);
}

void cutlass_int8_blockwise_hybrid_grouped_small_bias_host(
    torch::Tensor a_q, torch::Tensor a_qscale, torch::Tensor a_fscale,
    torch::Tensor b_q, torch::Tensor b_qscale, torch::Tensor b_fscale,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    torch::Tensor out, torch::Tensor bias,
    int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets) {
  launch_grouped_hybrid<SmallConfig, true>(
      a_q, a_qscale, a_fscale, b_q, b_qscale, b_fscale, expert_offsets,
      problem_sizes, out, bias, scale_offsets, quant_block_size,
      super_group_size);
}
