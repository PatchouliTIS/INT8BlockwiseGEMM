# SPDX-License-Identifier: Apache-2.0
"""Triton quantization kernels for INT8 hybrid MoE (standalone, no vllm).

Contains:
- round_int8: Triton JIT helper for round-to-nearest-even INT8 cast
- quantize_scales_for_hybrid: Two-level (Q int32, F fp32) scale factorization
- moe_kernel_quantize_input_int8_hybrid_grouped: Grouped blockwise INT8 quant
- fused_permute_quant_int8_hybrid_grouped: Fused permute + grouped INT8 quant
- fused_silu_quant_int8_hybrid_grouped: Fused SiLU + grouped INT8 quant
"""

from __future__ import annotations

from math import prod

import torch
import triton
import triton.language as tl


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """Shrink the given tensor and apply the given view to it."""
    assert prod(v) <= x.numel(), f"{v} ({prod(v)}) <= {x.shape} ({x.numel()})"
    return x.flatten()[: prod(v)].view(*v)


# ---------------------------------------------------------------------------
# round_int8 Triton JIT helper
# ---------------------------------------------------------------------------

@triton.jit
def round_int8(x):
    """Round-to-nearest-even and cast to int8 (CUDA libdevice)."""
    return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _rint_f32(x):
    """Round-to-nearest-even for fp32 (CUDA libdevice)."""
    return tl.extra.cuda.libdevice.rint(x)


# ---------------------------------------------------------------------------
# quantize_scales_for_hybrid (Triton-accelerated)
# ---------------------------------------------------------------------------

@triton.jit
def _quantize_scales_for_hybrid_kernel(
    scale_ptr,
    Q_ptr,
    F_ptr,
    stride_s_row,
    stride_s_col,
    stride_q_row,
    stride_q_col,
    stride_f_row,
    stride_f_col,
    K_blocks,
    super_group_size,
    Q_max,
    eps,
    BLOCK_SG: tl.constexpr,
):
    """Triton kernel for two-level scale quantization.

    Grid: (num_rows, num_super_groups)
    """
    pid_row = tl.program_id(0)
    pid_sg = tl.program_id(1)

    sg_start = pid_sg * super_group_size
    offs_k = sg_start + tl.arange(0, BLOCK_SG)
    sg_end = sg_start + super_group_size
    sg_end = tl.minimum(sg_end, K_blocks)
    mask = offs_k < sg_end

    ptrs = scale_ptr + pid_row * stride_s_row + offs_k * stride_s_col
    vals = tl.load(ptrs, mask=mask, other=0.0)

    abs_vals = tl.abs(vals)
    S_max = tl.max(abs_vals, axis=0)
    S_max = tl.maximum(S_max, eps)

    F_g = S_max / Q_max
    q_vals = _rint_f32(vals / F_g).to(tl.int32)

    q_ptrs = Q_ptr + pid_row * stride_q_row + offs_k * stride_q_col
    tl.store(q_ptrs, q_vals, mask=mask)

    f_ptr = F_ptr + pid_row * stride_f_row + pid_sg * stride_f_col
    tl.store(f_ptr, F_g)


def quantize_scales_for_hybrid(
    scale: torch.Tensor,
    Q_max: int = 64,
    super_group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton-accelerated two-level scale quantization.

    Converts FP32 blockwise scales into (Q int32, F fp32) pairs.
    """
    assert scale.ndim == 2, "scale must be a 2D tensor [X_blocks, K_blocks]"
    assert scale.dtype == torch.float32, "scale must be float32"

    num_rows, K_blocks = scale.shape
    if super_group_size is None:
        super_group_size = K_blocks
    num_super_groups = (K_blocks + super_group_size - 1) // super_group_size

    Q = torch.empty(num_rows, K_blocks, dtype=torch.int32, device=scale.device)
    F = torch.empty(
        num_rows, num_super_groups, dtype=torch.float32, device=scale.device
    )

    BLOCK_SG = triton.next_power_of_2(super_group_size)
    grid = (num_rows, num_super_groups)

    _quantize_scales_for_hybrid_kernel[grid](
        scale,
        Q,
        F,
        scale.stride(0),
        scale.stride(1),
        Q.stride(0),
        Q.stride(1),
        F.stride(0),
        F.stride(1),
        K_blocks,
        super_group_size,
        Q_max,
        1e-12,
        BLOCK_SG=BLOCK_SG,
    )
    return Q, F


# ---------------------------------------------------------------------------
# Grouped blockwise INT8 quantization kernel
# ---------------------------------------------------------------------------

@triton.jit
def _grouped_blockwise_quant_int8_kernel(
    x_ptr,
    xq_ptr,
    xs_ptr,
    expert_offsets_ptr,
    scale_offsets_ptr,
    E: tl.constexpr,
    M,
    K: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_xqm,
    stride_xqk,
    stride_xsm,
    stride_xsk,
    eps,
    int8_min,
    int8_max,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)
    total_scale_rows = tl.load(scale_offsets_ptr + E)
    if pid_s >= total_scale_rows:
        return

    lo = 0
    hi = E
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        mid_offset = tl.load(scale_offsets_ptr + mid)
        if mid_offset <= pid_s:
            lo = mid
        else:
            hi = mid

    expert = lo
    local_m_block = pid_s - tl.load(scale_offsets_ptr + expert)
    expert_start = tl.load(expert_offsets_ptr + expert)
    expert_end = tl.load(expert_offsets_ptr + expert + 1)
    row_start = expert_start + local_m_block * BLOCK_M

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_m[:, None] < expert_end) & (offs_m[:, None] < M) & (offs_k[None, :] < K)

    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    scale = tl.maximum(absmax, eps) / int8_max
    q = round_int8(tl.clamp(x / scale, int8_min, int8_max))

    tl.store(
        xq_ptr + offs_m[:, None] * stride_xqm + offs_k[None, :] * stride_xqk,
        q,
        mask=mask,
    )
    tl.store(xs_ptr + pid_s * stride_xsm + pid_k * stride_xsk, scale)


def moe_kernel_quantize_input_int8_hybrid_grouped(
    A: torch.Tensor,
    block_k: int,
    q_max: int,
    super_group_size: int,
    expert_offsets: torch.Tensor,
    tile_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Graph-safe grouped activation quantization for CUTLASS hybrid MoE.

    Returns:
        qA: [M, K] int8 activations.
        Q_A: [upper_scale_rows, K_blocks] int32 quantized scales.
        F_A: [upper_scale_rows, num_super_groups] fp32 super-group factors.
        scale_offsets_i32: [E] int32 prefix-sum bases for CUTLASS.
    """
    assert A.ndim == 2 and A.is_contiguous()
    assert A.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert expert_offsets.ndim == 1 and expert_offsets.is_cuda
    assert block_k > 0 and tile_m > 0 and super_group_size > 0

    M, K = A.shape
    E = expert_offsets.numel() - 1
    k_blocks = cdiv(K, block_k)
    upper_scale_rows = cdiv(M, tile_m) + E

    counts = expert_offsets[1:] - expert_offsets[:-1]
    scale_counts = torch.div(counts + tile_m - 1, tile_m, rounding_mode="floor")
    scale_offsets = torch.empty_like(expert_offsets)
    scale_offsets[:1].zero_()
    scale_offsets[1:] = torch.cumsum(scale_counts, dim=0, dtype=expert_offsets.dtype)

    qA = torch.empty_like(A, dtype=torch.int8)
    scale = torch.empty(
        (upper_scale_rows, k_blocks), dtype=torch.float32, device=A.device
    )
    iinfo = torch.iinfo(torch.int8)
    num_warps = 4 if tile_m * block_k <= 4096 else 8
    _grouped_blockwise_quant_int8_kernel[(upper_scale_rows, k_blocks)](
        A,
        qA,
        scale,
        expert_offsets,
        scale_offsets,
        E,
        M,
        K,
        A.stride(0),
        A.stride(1),
        qA.stride(0),
        qA.stride(1),
        scale.stride(0),
        scale.stride(1),
        1e-10,
        float(iinfo.min),
        float(iinfo.max),
        BLOCK_M=tile_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    Q_A, F_A = quantize_scales_for_hybrid(
        scale, Q_max=q_max, super_group_size=super_group_size
    )
    scale_offsets_i32 = scale_offsets[:-1].to(torch.int32)
    return qA, Q_A, F_A, scale_offsets_i32


# ---------------------------------------------------------------------------
# Fused permute + grouped INT8 quantization kernel (C7-P0)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_permute_quant_int8_kernel(
    hidden_ptr,         # [num_tokens, K] bf16/fp16/fp32
    permuted_idx_ptr,   # [num_tokens * topk] int32
    xq_ptr,             # [num_tokens * topk, K] int8
    xs_ptr,             # [upper_scale_rows, K_blocks] fp32
    expert_offsets_ptr,
    scale_offsets_ptr,
    E: tl.constexpr,
    M_PERM,
    K: tl.constexpr,
    stride_hm,
    stride_hk,
    stride_xqm,
    stride_xqk,
    stride_xsm,
    stride_xsk,
    eps,
    int8_min,
    int8_max,
    TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)
    total_scale_rows = tl.load(scale_offsets_ptr + E)
    if pid_s >= total_scale_rows:
        return

    lo = 0
    hi = E
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        mid_offset = tl.load(scale_offsets_ptr + mid)
        if mid_offset <= pid_s:
            lo = mid
        else:
            hi = mid

    expert = lo
    local_m_block = pid_s - tl.load(scale_offsets_ptr + expert)
    expert_start = tl.load(expert_offsets_ptr + expert)
    expert_end = tl.load(expert_offsets_ptr + expert + 1)
    row_start = expert_start + local_m_block * BLOCK_M

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = (offs_m < expert_end) & (offs_m < M_PERM)
    col_mask = offs_k < K
    mask = row_mask[:, None] & col_mask[None, :]

    expanded_source_row = tl.load(
        permuted_idx_ptr + offs_m,
        mask=row_mask,
        other=0,
    )
    source_m = expanded_source_row // TOPK

    x = tl.load(
        hidden_ptr + source_m[:, None] * stride_hm + offs_k[None, :] * stride_hk,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absmax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    scale = tl.maximum(absmax, eps) / int8_max
    q = round_int8(tl.clamp(x / scale, int8_min, int8_max))

    tl.store(
        xq_ptr + offs_m[:, None] * stride_xqm + offs_k[None, :] * stride_xqk,
        q,
        mask=mask,
    )
    tl.store(xs_ptr + pid_s * stride_xsm + pid_k * stride_xsk, scale)


def fused_permute_quant_int8_hybrid_grouped(
    hidden_states: torch.Tensor,
    permuted_idx: torch.Tensor,
    topk: int,
    block_k: int,
    q_max: int,
    super_group_size: int,
    expert_offsets: torch.Tensor,
    tile_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused pre-mm1 permute + grouped INT8 hybrid quantization.

    Returns:
        qA: [M * topk, K] int8 activations in expert-contiguous order.
        Q_A: [upper_scale_rows, K_blocks] int32 quantized scales.
        F_A: [upper_scale_rows, num_super_groups] fp32 factors.
        scale_offsets_i32: [E] int32 base offsets for CUTLASS.
    """
    assert hidden_states.ndim == 2 and hidden_states.is_contiguous()
    assert hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert permuted_idx.ndim == 1 and permuted_idx.is_cuda
    assert permuted_idx.dtype == torch.int32
    assert expert_offsets.ndim == 1 and expert_offsets.is_cuda
    assert topk > 0 and block_k > 0 and tile_m > 0 and super_group_size > 0

    _, K = hidden_states.shape
    M_permuted = permuted_idx.numel()
    E = expert_offsets.numel() - 1
    k_blocks = cdiv(K, block_k)
    upper_scale_rows = cdiv(M_permuted, tile_m) + E

    counts = expert_offsets[1:] - expert_offsets[:-1]
    scale_counts = torch.div(counts + tile_m - 1, tile_m, rounding_mode="floor")
    scale_offsets = torch.empty_like(expert_offsets)
    scale_offsets[:1].zero_()
    scale_offsets[1:] = torch.cumsum(scale_counts, dim=0, dtype=expert_offsets.dtype)

    qA = torch.empty((M_permuted, K), dtype=torch.int8, device=hidden_states.device)
    scale = torch.empty(
        (upper_scale_rows, k_blocks), dtype=torch.float32, device=hidden_states.device
    )

    iinfo = torch.iinfo(torch.int8)
    num_warps = 4 if tile_m * block_k <= 4096 else 8
    _fused_permute_quant_int8_kernel[(upper_scale_rows, k_blocks)](
        hidden_states,
        permuted_idx,
        qA,
        scale,
        expert_offsets,
        scale_offsets,
        E,
        M_permuted,
        K,
        hidden_states.stride(0),
        hidden_states.stride(1),
        qA.stride(0),
        qA.stride(1),
        scale.stride(0),
        scale.stride(1),
        1e-10,
        float(iinfo.min),
        float(iinfo.max),
        TOPK=topk,
        BLOCK_M=tile_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    Q_A, F_A = quantize_scales_for_hybrid(
        scale, Q_max=q_max, super_group_size=super_group_size
    )
    scale_offsets_i32 = scale_offsets[:-1].to(torch.int32)
    return qA, Q_A, F_A, scale_offsets_i32


# ---------------------------------------------------------------------------
# Fused SiLU + grouped INT8 quantization kernel (P1-A)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_silu_quant_int8_hybrid_grouped_kernel(
    a_ptr,              # mm1_out [sum_M, 2*N] bf16/fp16/fp32 (gate||up)
    aq_ptr,             # output  [sum_M, N]   int8
    xs_ptr,             # output  [upper_scale_rows, K_blocks] fp32
    expert_offsets_ptr,
    scale_offsets_ptr,
    E: tl.constexpr,
    M,                  # sum_M
    K: tl.constexpr,    # = N (intermediate, output K-dim)
    stride_am, stride_an,
    stride_aqm, stride_aqn,
    stride_xsm, stride_xsk,
    eps,
    int8_min, int8_max,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused SiLU(gate)*up + per-(TileM, BLOCK_K) absmax INT8 quant."""
    pid_s = tl.program_id(0)
    pid_k = tl.program_id(1)
    total_scale_rows = tl.load(scale_offsets_ptr + E)
    if pid_s >= total_scale_rows:
        return

    lo = 0
    hi = E
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        mid_offset = tl.load(scale_offsets_ptr + mid)
        if mid_offset <= pid_s:
            lo = mid
        else:
            hi = mid

    expert = lo
    local_m_block = pid_s - tl.load(scale_offsets_ptr + expert)
    expert_start = tl.load(expert_offsets_ptr + expert)
    expert_end = tl.load(expert_offsets_ptr + expert + 1)
    row_start = expert_start + local_m_block * BLOCK_M

    offs_m = row_start + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    row_mask = (offs_m[:, None] < expert_end) & (offs_m[:, None] < M)
    col_mask = offs_k[None, :] < K
    mask = row_mask & col_mask

    # gate ptrs: a[row, k] (cols [0, N))
    # up ptrs:   a[row, k + N] (cols [N, 2N))
    gate_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_an
    up_ptrs = gate_ptrs + K * stride_an

    gate = tl.load(gate_ptrs, mask=mask, other=0.0)
    up = tl.load(up_ptrs, mask=mask, other=0.0)

    # Bit-exact SiLU: cast back to native dtype before multiply
    gate_f32 = gate.to(tl.float32)
    silu_g_native = (gate_f32 / (1.0 + tl.exp(-gate_f32))).to(gate.dtype)
    act_native = silu_g_native * up
    x = act_native.to(tl.float32)

    # Per-tile (BLOCK_M x BLOCK_K) single-scalar scale
    absmax = tl.max(tl.max(tl.abs(x), axis=1), axis=0)
    scale = tl.maximum(absmax, eps) / int8_max
    q = round_int8(tl.clamp(x / scale, int8_min, int8_max))

    tl.store(
        aq_ptr + offs_m[:, None] * stride_aqm + offs_k[None, :] * stride_aqn,
        q,
        mask=mask,
    )
    tl.store(xs_ptr + pid_s * stride_xsm + pid_k * stride_xsk, scale)


def fused_silu_quant_int8_hybrid_grouped(
    mm1_out: torch.Tensor,
    block_k: int,
    q_max: int,
    super_group_size: int,
    expert_offsets: torch.Tensor,
    tile_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SiLU(gate)*up + grouped INT8 hybrid activation quantization.

    Returns:
        aq: [sum_M, N] int8.
        Q_A: [upper_scale_rows, K_blocks] int32.
        F_A: [upper_scale_rows, num_super_groups] fp32.
        scale_offsets_i32: [E] int32 base offsets for CUTLASS.
    """
    assert mm1_out.ndim == 2 and mm1_out.is_contiguous()
    assert mm1_out.dtype in (torch.float32, torch.float16, torch.bfloat16)
    assert expert_offsets.ndim == 1 and expert_offsets.is_cuda
    assert block_k > 0 and tile_m > 0 and super_group_size > 0
    sum_M, two_N = mm1_out.shape
    assert two_N % 2 == 0, "w13 output must have an even last dim (2*N)"
    N = two_N // 2

    E = expert_offsets.numel() - 1
    k_blocks = cdiv(N, block_k)
    upper_scale_rows = cdiv(sum_M, tile_m) + E

    counts = expert_offsets[1:] - expert_offsets[:-1]
    scale_counts = torch.div(counts + tile_m - 1, tile_m, rounding_mode="floor")
    scale_offsets = torch.empty_like(expert_offsets)
    scale_offsets[:1].zero_()
    scale_offsets[1:] = torch.cumsum(scale_counts, dim=0, dtype=expert_offsets.dtype)

    aq = torch.empty((sum_M, N), dtype=torch.int8, device=mm1_out.device)
    scale = torch.empty(
        (upper_scale_rows, k_blocks), dtype=torch.float32, device=mm1_out.device
    )

    iinfo = torch.iinfo(torch.int8)
    num_warps = 4 if tile_m * block_k <= 4096 else 8

    _fused_silu_quant_int8_hybrid_grouped_kernel[(upper_scale_rows, k_blocks)](
        mm1_out,
        aq,
        scale,
        expert_offsets,
        scale_offsets,
        E,
        sum_M,
        N,
        mm1_out.stride(0), mm1_out.stride(1),
        aq.stride(0), aq.stride(1),
        scale.stride(0), scale.stride(1),
        1e-10,
        float(iinfo.min), float(iinfo.max),
        BLOCK_M=tile_m,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    Q_A, F_A = quantize_scales_for_hybrid(
        scale, Q_max=q_max, super_group_size=super_group_size
    )
    scale_offsets_i32 = scale_offsets[:-1].to(torch.int32)
    return aq, Q_A, F_A, scale_offsets_i32
