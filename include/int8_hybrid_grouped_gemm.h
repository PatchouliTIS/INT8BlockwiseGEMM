#pragma once

#include <torch/types.h>

#include <optional>

void cutlass_int8_blockwise_hybrid_grouped_host(
    torch::Tensor a_q,       // [sum_M, K] INT8
    torch::Tensor a_qscale,  // [sum_e ceil(M_e / TileM), K_blocks] INT32
    torch::Tensor a_fscale,  // [sum_e ceil(M_e / TileM), num_super_groups] FP32
    torch::Tensor b_q,       // [E, N, K] INT8
    torch::Tensor b_qscale,  // [E, ceil(N / TileN), K_blocks] INT32
    torch::Tensor b_fscale,  // [E, ceil(N / TileN), num_super_groups] FP32
    torch::Tensor expert_offsets,  // [E + 1] INT32
    torch::Tensor problem_sizes,   // [E, 3] INT32 rows are (M_e, N, K)
    torch::Tensor out,             // [sum_M, N] BF16
    int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets  // [E] INT32 (optional, P0-A)
);

void cutlass_int8_blockwise_hybrid_grouped_bias_host(
    torch::Tensor a_q, torch::Tensor a_qscale, torch::Tensor a_fscale,
    torch::Tensor b_q, torch::Tensor b_qscale, torch::Tensor b_fscale,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    torch::Tensor out,
    torch::Tensor bias,  // [E, N] or [N] FP32
    int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets);

void cutlass_int8_blockwise_hybrid_grouped_small_host(
    torch::Tensor a_q, torch::Tensor a_qscale, torch::Tensor a_fscale,
    torch::Tensor b_q, torch::Tensor b_qscale, torch::Tensor b_fscale,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    torch::Tensor out, int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets);

void cutlass_int8_blockwise_hybrid_grouped_small_bias_host(
    torch::Tensor a_q, torch::Tensor a_qscale, torch::Tensor a_fscale,
    torch::Tensor b_q, torch::Tensor b_qscale, torch::Tensor b_fscale,
    torch::Tensor expert_offsets, torch::Tensor problem_sizes,
    torch::Tensor out,
    torch::Tensor bias,  // [E, N] or [N] FP32
    int64_t quant_block_size, int64_t super_group_size,
    std::optional<torch::Tensor> scale_offsets);
