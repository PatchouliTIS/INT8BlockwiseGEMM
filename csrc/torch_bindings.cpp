// SPDX-License-Identifier: Apache-2.0
// Unified torch bindings for INT8 Hybrid MoE standalone benchmark.
// Registers all ops under the "int8_hybrid_moe" namespace:
//   - CUTLASS INT8 hybrid grouped GEMM (from /deploy/INT8HybridMoE)
//   - MoE permute / unpermute (from vllm csrc/moe)
//   - silu_and_mul activation (from vllm csrc/activation_kernels.cu)

#include <torch/library.h>
#include <torch/types.h>

#include <optional>

// ---------------------------------------------------------------------------
// Forward declarations: CUTLASS INT8 hybrid grouped GEMM
// ---------------------------------------------------------------------------
void cutlass_int8_grouped_mm_host(torch::Tensor, torch::Tensor, torch::Tensor,
                                  double, double, torch::Tensor, torch::Tensor);

void cutlass_int8_blockwise_hybrid_grouped_host(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, std::optional<torch::Tensor>);

void cutlass_int8_blockwise_hybrid_grouped_bias_host(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, std::optional<torch::Tensor>);

void cutlass_int8_blockwise_hybrid_grouped_small_host(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, std::optional<torch::Tensor>);

void cutlass_int8_blockwise_hybrid_grouped_small_bias_host(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    int64_t, int64_t, std::optional<torch::Tensor>);

// ---------------------------------------------------------------------------
// Forward declarations: MoE permute / unpermute (from vllm)
// ---------------------------------------------------------------------------
void moe_permute(
    const torch::Tensor& input, const torch::Tensor& topk_ids,
    const torch::Tensor& token_expert_indices,
    const std::optional<torch::Tensor>& expert_map,
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    torch::Tensor& permuted_input, torch::Tensor& expert_first_token_offset,
    torch::Tensor& inv_permuted_idx, torch::Tensor& permuted_idx);

void moe_permute_metadata(
    const torch::Tensor& topk_ids, const torch::Tensor& token_expert_indices,
    const std::optional<torch::Tensor>& expert_map,
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    torch::Tensor& expert_first_token_offset,
    torch::Tensor& inv_permuted_idx, torch::Tensor& permuted_idx);

void moe_unpermute(
    const torch::Tensor& permuted_hidden_states,
    const torch::Tensor& topk_weights, const torch::Tensor& inv_permuted_idx,
    const std::optional<torch::Tensor>& expert_first_token_offset,
    int64_t topk, torch::Tensor& hidden_states);

bool moe_permute_unpermute_supported();

// ---------------------------------------------------------------------------
// Forward declarations: Activation kernels (from vllm)
// ---------------------------------------------------------------------------
void silu_and_mul(torch::Tensor& out, torch::Tensor& input);

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------
TORCH_LIBRARY(int8_hybrid_moe, m) {
  // CUTLASS INT8 grouped GEMM
  m.def(
      "cutlass_int8_grouped_mm(Tensor a, Tensor b, Tensor(a!) out, "
      "float a_scale, float b_scale, Tensor expert_offsets, "
      "Tensor problem_sizes) -> ()");

  m.def(
      "cutlass_int8_blockwise_hybrid_grouped(Tensor a_q, Tensor a_qscale, "
      "Tensor a_fscale, Tensor b_q, Tensor b_qscale, Tensor b_fscale, "
      "Tensor expert_offsets, Tensor problem_sizes, Tensor(a!) out, "
      "int quant_block_size, int super_group_size, "
      "Tensor? scale_offsets=None) -> ()");

  m.def(
      "cutlass_int8_blockwise_hybrid_grouped_bias(Tensor a_q, Tensor "
      "a_qscale, Tensor a_fscale, Tensor b_q, Tensor b_qscale, Tensor "
      "b_fscale, Tensor expert_offsets, Tensor problem_sizes, Tensor(a!) "
      "out, Tensor bias, int quant_block_size, "
      "int super_group_size, Tensor? scale_offsets=None) -> ()");

  m.def(
      "cutlass_int8_blockwise_hybrid_grouped_small(Tensor a_q, Tensor "
      "a_qscale, Tensor a_fscale, Tensor b_q, Tensor b_qscale, Tensor "
      "b_fscale, Tensor expert_offsets, Tensor problem_sizes, Tensor(a!) "
      "out, int quant_block_size, "
      "int super_group_size, Tensor? scale_offsets=None) -> ()");

  m.def(
      "cutlass_int8_blockwise_hybrid_grouped_small_bias(Tensor a_q, Tensor "
      "a_qscale, Tensor a_fscale, Tensor b_q, Tensor b_qscale, Tensor "
      "b_fscale, Tensor expert_offsets, Tensor problem_sizes, Tensor(a!) "
      "out, Tensor bias, int quant_block_size, "
      "int super_group_size, Tensor? scale_offsets=None) -> ()");

  // MoE permute / unpermute
  m.def(
      "moe_permute(Tensor input, Tensor topk_ids, "
      "Tensor token_expert_indices, Tensor? expert_map, int n_expert, "
      "int n_local_expert, int topk, Tensor! permuted_input, "
      "Tensor! expert_first_token_offset, Tensor! inv_permuted_idx, "
      "Tensor! permuted_idx) -> ()");

  m.def(
      "moe_permute_metadata(Tensor topk_ids, Tensor token_expert_indices, "
      "Tensor? expert_map, int n_expert, int n_local_expert, int topk, "
      "Tensor! expert_first_token_offset, Tensor! inv_permuted_idx, "
      "Tensor! permuted_idx) -> ()");

  m.def(
      "moe_unpermute(Tensor permuted_hidden_states, Tensor topk_weights, "
      "Tensor inv_permuted_idx, Tensor? expert_first_token_offset, "
      "int topk, Tensor! hidden_states) -> ()");

  m.def("moe_permute_unpermute_supported() -> bool");
  m.impl("moe_permute_unpermute_supported", &moe_permute_unpermute_supported);

  // Activation
  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
}

TORCH_LIBRARY_IMPL(int8_hybrid_moe, CUDA, m) {
  // CUTLASS INT8 grouped GEMM
  m.impl("cutlass_int8_grouped_mm", &cutlass_int8_grouped_mm_host);
  m.impl("cutlass_int8_blockwise_hybrid_grouped",
         &cutlass_int8_blockwise_hybrid_grouped_host);
  m.impl("cutlass_int8_blockwise_hybrid_grouped_bias",
         &cutlass_int8_blockwise_hybrid_grouped_bias_host);
  m.impl("cutlass_int8_blockwise_hybrid_grouped_small",
         &cutlass_int8_blockwise_hybrid_grouped_small_host);
  m.impl("cutlass_int8_blockwise_hybrid_grouped_small_bias",
         &cutlass_int8_blockwise_hybrid_grouped_small_bias_host);

  // MoE permute / unpermute
  m.impl("moe_permute", &moe_permute);
  m.impl("moe_permute_metadata", &moe_permute_metadata);
  m.impl("moe_unpermute", &moe_unpermute);

  // Activation
  m.impl("silu_and_mul", &silu_and_mul);
}
