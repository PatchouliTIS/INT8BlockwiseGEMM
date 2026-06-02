# SPDX-License-Identifier: Apache-2.0
"""Thin wrappers around CUDA kernel ops from libint8_hybrid_moe_ops.so.

All ops are registered under the unified torch.ops.int8_hybrid_moe namespace.
"""

import torch


# ---------------------------------------------------------------------------
# CUTLASS INT8 hybrid grouped GEMM kernels
# ---------------------------------------------------------------------------

def cutlass_int8_blockwise_hybrid_grouped(
    A: torch.Tensor,
    Q_A: torch.Tensor,
    F_A: torch.Tensor,
    B: torch.Tensor,
    Q_B: torch.Tensor,
    F_B: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    out: torch.Tensor,
    quant_block_size: int,
    super_group_size: int,
    scale_offsets: torch.Tensor | None = None,
) -> None:
    """Grouped hybrid INT8 blockwise GEMM (CTA tile 128x64x128, no bias)."""
    torch.ops.int8_hybrid_moe.cutlass_int8_blockwise_hybrid_grouped(
        A, Q_A, F_A, B, Q_B, F_B, expert_offsets, problem_sizes, out,
        quant_block_size, super_group_size, scale_offsets
    )


def cutlass_int8_blockwise_hybrid_grouped_small(
    A: torch.Tensor,
    Q_A: torch.Tensor,
    F_A: torch.Tensor,
    B: torch.Tensor,
    Q_B: torch.Tensor,
    F_B: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    out: torch.Tensor,
    quant_block_size: int,
    super_group_size: int,
    scale_offsets: torch.Tensor | None = None,
) -> None:
    """Grouped hybrid INT8 blockwise GEMM (CTA tile 64x64x128, no bias)."""
    torch.ops.int8_hybrid_moe.cutlass_int8_blockwise_hybrid_grouped_small(
        A, Q_A, F_A, B, Q_B, F_B, expert_offsets, problem_sizes, out,
        quant_block_size, super_group_size, scale_offsets
    )


# ---------------------------------------------------------------------------
# Activation kernels
# ---------------------------------------------------------------------------

def silu_and_mul(output: torch.Tensor, input: torch.Tensor) -> None:
    """SiLU gated activation: output = silu(gate) * up."""
    torch.ops.int8_hybrid_moe.silu_and_mul(output, input)


# ---------------------------------------------------------------------------
# MoE permute / unpermute kernels
# ---------------------------------------------------------------------------

def moe_permute(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    expert_map: torch.Tensor | None,
    n_expert: int,
    n_local_expert: int,
    topk: int,
    permuted_hidden_states: torch.Tensor,
    expert_first_token_offset: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    permuted_idx: torch.Tensor,
) -> None:
    """Dispatch to the C++ moe_permute kernel."""
    torch.ops.int8_hybrid_moe.moe_permute(
        hidden_states,
        topk_ids,
        token_expert_indices,
        expert_map,
        n_expert,
        n_local_expert,
        topk,
        permuted_hidden_states,
        expert_first_token_offset,
        inv_permuted_idx,
        permuted_idx,
    )


def moe_permute_metadata(
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    expert_map: torch.Tensor | None,
    n_expert: int,
    n_local_expert: int,
    topk: int,
    expert_first_token_offset: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    permuted_idx: torch.Tensor,
) -> None:
    """Dispatch to the C++ moe_permute_metadata kernel."""
    torch.ops.int8_hybrid_moe.moe_permute_metadata(
        topk_ids,
        token_expert_indices,
        expert_map,
        n_expert,
        n_local_expert,
        topk,
        expert_first_token_offset,
        inv_permuted_idx,
        permuted_idx,
    )


def moe_unpermute(
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None,
    topk: int,
    out: torch.Tensor,
) -> None:
    """Dispatch to the C++ moe_unpermute kernel."""
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    torch.ops.int8_hybrid_moe.moe_unpermute(
        permuted_hidden_states,
        topk_weights,
        inv_permuted_idx,
        expert_first_token_offset,
        topk,
        out,
    )
