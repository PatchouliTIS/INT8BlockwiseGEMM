#pragma once

#include <torch/types.h>

void cutlass_int8_grouped_mm_host(
    torch::Tensor a,  // [sum_M, K] int8, row-major
    torch::Tensor
        b,  // [E, N, K] int8, row-major storage as logical col-major B
    torch::Tensor out,               // [sum_M, N] bf16, row-major
    double a_scale, double b_scale,  // per-tensor scales
    torch::Tensor expert_offsets,    // [E + 1] int32, CUDA
    torch::Tensor problem_sizes);    // [E, 3] int32 CUDA, rows are (M, N, K)
