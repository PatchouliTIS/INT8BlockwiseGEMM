# SPDX-License-Identifier: Apache-2.0
"""CUTLASS SM80 INT8 hybrid grouped MoE runner (standalone, no vllm).

This module implements _run_cutlass_int8_hybrid_grouped_moe which is the
core compute path for the INT8 hybrid MoE benchmark.
"""

from __future__ import annotations

import os

import torch

from .activation import MoEActivation, apply_moe_activation
from .ops import (
    cutlass_int8_blockwise_hybrid_grouped,
    cutlass_int8_blockwise_hybrid_grouped_small,
)
from .permute import moe_permute, moe_permute_metadata, moe_unpermute
from .quant import (
    _resize_cache,
    fused_permute_quant_int8_hybrid_grouped,
    fused_silu_quant_int8_hybrid_grouped,
    moe_kernel_quantize_input_int8_hybrid_grouped,
)

# ---------------------------------------------------------------------------
# Variant selection (must stay in sync with CUTLASS templates)
# ---------------------------------------------------------------------------
_VARIANT_TILE_M = {"hybrid": 128, "hybrid_small": 64}
_VARIANT_TILE_N = {"hybrid": 64, "hybrid_small": 64}
_VARIANT_TILE_K = {"hybrid": 128, "hybrid_small": 128}

# Environment switches (mirror vllm behavior)
_USE_FUSED_SILU_QUANT_W2: bool = (
    os.environ.get("VLLM_INT8_HYBRID_FUSED_SILU_QUANT", "1").strip() == "1"
)
_USE_FUSED_PERMUTE_QUANT_W1: bool = (
    os.environ.get("VLLM_INT8_HYBRID_FUSED_PERMUTE_QUANT", "0").strip() == "1"
)


def pick_grouped_hybrid_variant(sum_M: int, num_experts: int) -> str:
    """Pick grouped hybrid variant by average M per expert."""
    if num_experts <= 0:
        return "hybrid"
    avg_M = sum_M / num_experts
    if avg_M <= 70:
        return "hybrid_small"
    return "hybrid"


def _dispatch_grouped_hybrid(variant: str):
    table = {
        "hybrid": cutlass_int8_blockwise_hybrid_grouped,
        "hybrid_small": cutlass_int8_blockwise_hybrid_grouped_small,
    }
    return table[variant]


def _variant_tile_m(variant: str) -> int:
    return _VARIANT_TILE_M[variant]


def _expert_offsets_i32(expert_first_token_offset: torch.Tensor) -> torch.Tensor:
    return expert_first_token_offset.to(torch.int32)


def _run_cutlass_int8_hybrid_grouped_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    w1_qscale: torch.Tensor,
    w1_fscale: torch.Tensor,
    w2_qscale: torch.Tensor,
    w2_fscale: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    quant_block_size: int,
    super_group_size: int,
    q_max: int,
) -> None:
    """Run the full INT8 hybrid grouped MoE forward pass.

    This is a standalone reimplementation of the vllm function with the same
    name, using only the loaded .so kernels and Triton quantization.
    """
    assert activation.is_gated, "Only gated activation is supported"
    assert hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert w1.dtype == torch.int8 and w2.dtype == torch.int8
    assert w1_qscale is not None and w1_fscale is not None
    assert w2_qscale is not None and w2_fscale is not None
    assert global_num_experts != -1

    M = hidden_states.size(0)
    topk = topk_ids.size(1)
    local_E = w1.size(0)
    _, K, N = w2.shape
    assert w1.size(2) == K
    assert w1.size(1) == 2 * N

    mm1_out = _resize_cache(workspace13, (M * topk, N * 2))
    act_out = _resize_cache(workspace2, (M * topk, N))
    mm2_out = _resize_cache(workspace2, (M * topk, K))

    problem_sizes1 = torch.empty((local_E, 3), dtype=torch.int32, device=hidden_states.device)
    problem_sizes2 = torch.empty((local_E, 3), dtype=torch.int32, device=hidden_states.device)

    num_expert = global_num_experts if expert_map is None else expert_map.size(0)
    if _USE_FUSED_PERMUTE_QUANT_W1:
        expert_first_token_offset, inv_perm, permuted_idx = moe_permute_metadata(
            hidden_states,
            topk_ids,
            num_expert,
            local_E,
            expert_map,
        )
        a1 = None
    else:
        a1_perm = _resize_cache(workspace2, (M * topk, K))
        a1, _, expert_first_token_offset, inv_perm, _ = moe_permute(
            hidden_states,
            None,
            topk_ids,
            num_expert,
            local_E,
            expert_map,
            permuted_hidden_states=a1_perm,
        )
    expert_offsets = _expert_offsets_i32(expert_first_token_offset)
    counts = expert_offsets[1:] - expert_offsets[:-1]
    problem_sizes1[:, 0] = counts
    problem_sizes1[:, 1] = 2 * N
    problem_sizes1[:, 2] = K
    problem_sizes2[:, 0] = counts
    problem_sizes2[:, 1] = K
    problem_sizes2[:, 2] = N
    sum_M = M * topk

    # ---- w13 GEMM ----
    variant1 = pick_grouped_hybrid_variant(sum_M, local_E)
    tile_m1 = _variant_tile_m(variant1)
    if _USE_FUSED_PERMUTE_QUANT_W1:
        a1q, a1_qscale, a1_fscale, a1_scale_offsets = (
            fused_permute_quant_int8_hybrid_grouped(
                hidden_states,
                permuted_idx,
                topk=topk,
                block_k=quant_block_size,
                q_max=q_max,
                super_group_size=super_group_size,
                expert_offsets=expert_offsets,
                tile_m=tile_m1,
            )
        )
    else:
        a1q, a1_qscale, a1_fscale, a1_scale_offsets = (
            moe_kernel_quantize_input_int8_hybrid_grouped(
                a1,
                block_k=quant_block_size,
                q_max=q_max,
                super_group_size=super_group_size,
                expert_offsets=expert_offsets,
                tile_m=tile_m1,
            )
        )
    _dispatch_grouped_hybrid(variant1)(
        a1q,
        a1_qscale,
        a1_fscale,
        w1,
        w1_qscale,
        w1_fscale,
        expert_offsets,
        problem_sizes1,
        mm1_out,
        quant_block_size,
        super_group_size,
        scale_offsets=a1_scale_offsets,
    )

    # ---- w2 GEMM ----
    variant2 = pick_grouped_hybrid_variant(sum_M, local_E)
    tile_m2 = _variant_tile_m(variant2)

    if _USE_FUSED_SILU_QUANT_W2 and activation == MoEActivation.SILU:
        a2q, a2_qscale, a2_fscale, a2_scale_offsets = (
            fused_silu_quant_int8_hybrid_grouped(
                mm1_out,
                block_k=quant_block_size,
                q_max=q_max,
                super_group_size=super_group_size,
                expert_offsets=expert_offsets,
                tile_m=tile_m2,
            )
        )
    else:
        apply_moe_activation(activation, act_out, mm1_out)
        a2q, a2_qscale, a2_fscale, a2_scale_offsets = (
            moe_kernel_quantize_input_int8_hybrid_grouped(
                act_out,
                block_k=quant_block_size,
                q_max=q_max,
                super_group_size=super_group_size,
                expert_offsets=expert_offsets,
                tile_m=tile_m2,
            )
        )
    _dispatch_grouped_hybrid(variant2)(
        a2q,
        a2_qscale,
        a2_fscale,
        w2,
        w2_qscale,
        w2_fscale,
        expert_offsets,
        problem_sizes2,
        mm2_out,
        quant_block_size,
        super_group_size,
        scale_offsets=a2_scale_offsets,
    )

    moe_unpermute(
        out=output,
        permuted_hidden_states=mm2_out,
        topk_weights=topk_weights,
        inv_permuted_idx=inv_perm,
        expert_first_token_offset=expert_first_token_offset,
    )
