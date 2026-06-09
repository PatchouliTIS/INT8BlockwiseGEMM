#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone INT8-hybrid MoE benchmark.

Three backends are compared on the same problem shape:

1. ``unquantized``           -- vllm TritonExperts with no quantization (baseline).
2. ``cutlass_int8_hybrid``   -- Standalone CUTLASS SM80 INT8 hybrid grouped MoE
                                kernel (compiled from this project's .so).
3. ``triton_int8_blockwise`` -- vllm TritonExperts with blockwise INT8 W8A8.

The cutlass_int8_hybrid backend uses the locally compiled .so library and does
NOT depend on vllm. The other two backends import from vllm directly.

Typical usage::

    # Run all backends
    python bench_int8_hybrid_moe.py \\
        --num-tokens 1024 --hidden 4096 --intermediate 1408 \\
        --num-experts 64 --top-k 8

    # Run only the cutlass backend (no vllm dependency)
    python bench_int8_hybrid_moe.py --backend cutlass_int8_hybrid

    # nsys profile (only the first profiled iteration is captured)
    nsys profile -c cudaProfilerApi -t cuda,nvtx --force-overwrite=true \\
        -o moe_cutlass python bench_int8_hybrid_moe.py \\
        --backend cutlass_int8_hybrid --profile

    # ncu profile (one kernel launch only)
    ncu --set full --target-processes all \\
        --profile-from-start off \\
        -o moe_cutlass python bench_int8_hybrid_moe.py \\
        --backend cutlass_int8_hybrid --profile
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import torch

if not torch.cuda.is_available():
    print("CUDA is required for this benchmark.", file=sys.stderr)
    sys.exit(1)

# Import the standalone package (loads .so libraries at import time)
import int8_hybrid_moe  # noqa: F401
from int8_hybrid_moe.activation import MoEActivation
from int8_hybrid_moe.cutlass_runner import _run_cutlass_int8_hybrid_grouped_moe
from int8_hybrid_moe.quant import quantize_scales_for_hybrid


# ---------------------------------------------------------------------------
# Profiler helpers (nsys/ncu capture window)
# ---------------------------------------------------------------------------
def cuda_profiler_start() -> None:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop() -> None:
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
@dataclass
class ProblemShape:
    num_tokens: int
    hidden: int                     # K
    intermediate: int               # N (per expert, per gate or up)
    num_experts: int
    top_k: int
    dtype: torch.dtype = torch.bfloat16


def _build_routing(num_tokens: int, num_experts: int, top_k: int,
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate uniformly random (topk_weights, topk_ids)."""
    topk_ids = torch.empty(num_tokens, top_k, dtype=torch.int32, device=device)
    for i in range(num_tokens):
        perm = torch.randperm(num_experts, device=device)[:top_k]
        topk_ids[i] = perm.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device=device, dtype=torch.float32),
        dim=-1,
    )
    return topk_weights, topk_ids


def _quantize_blockwise_int8(
    w: torch.Tensor,
    block_n: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized [E, N, K] blockwise INT8 quantization."""
    import torch.nn.functional as F

    E, N, K = w.shape
    n_blocks = (N + block_n - 1) // block_n
    k_blocks = (K + block_k - 1) // block_k
    pad_n = n_blocks * block_n - N
    pad_k = k_blocks * block_k - K

    padded = F.pad(w.float(), (0, pad_k, 0, pad_n)).contiguous()
    view = padded.view(E, n_blocks, block_n, k_blocks, block_k)
    view = view.permute(0, 1, 3, 2, 4)

    scale = view.abs().amax(dim=(-1, -2)).clamp(min=1e-10) / 127.0
    q_view = (view / scale[..., None, None]).round().clamp(-128, 127)
    q = q_view.to(torch.int8).permute(0, 1, 3, 2, 4).contiguous()
    q = q.view(E, n_blocks * block_n, k_blocks * block_k)[:, :N, :K]
    return q.contiguous(), scale.contiguous()


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------
class BackendRunner:
    """Base class: captures pre-built tensors and exposes a no-arg run()."""

    name: str

    def run(self) -> None:
        raise NotImplementedError


class UnquantizedTritonRunner(BackendRunner):
    """Runs vllm's TritonExperts with the unquantized config (baseline)."""

    name = "unquantized"

    def __init__(self, shape: ProblemShape, device: torch.device):
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation as VllmMoEActivation,
        )
        from vllm.model_executor.layers.fused_moe.config import (
            FUSED_MOE_UNQUANTIZED_CONFIG,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

        self._vllm_activation = VllmMoEActivation.SILU

        moe_config = _make_vllm_moe_config(shape, device)
        self.experts = TritonExperts(
            moe_config=moe_config,
            quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        )

        E = shape.num_experts
        N = shape.intermediate
        K = shape.hidden
        M = shape.num_tokens
        topk = shape.top_k
        dtype = shape.dtype

        self.hidden = torch.randn(M, K, device=device, dtype=dtype)
        self.w13 = torch.randn(E, 2 * N, K, device=device, dtype=dtype) * 0.02
        self.w2 = torch.randn(E, K, N, device=device, dtype=dtype) * 0.02
        self.topk_weights, self.topk_ids = _build_routing(M, E, topk, device)

        ws1, ws2, out = self.experts.workspace_shapes(
            M=M, N=N, K=K, topk=topk,
            global_num_experts=E, local_num_experts=E,
            expert_tokens_meta=None,
            activation=self._vllm_activation,
        )
        self.workspace13 = torch.empty(ws1, device=device, dtype=dtype)
        self.workspace2 = torch.empty(ws2, device=device, dtype=dtype)
        self.output = torch.empty(out, device=device, dtype=dtype)

    def run(self) -> None:
        self.experts.apply(
            output=self.output,
            hidden_states=self.hidden,
            w1=self.w13,
            w2=self.w2,
            topk_weights=self.topk_weights,
            topk_ids=self.topk_ids,
            activation=self._vllm_activation,
            global_num_experts=self.w13.size(0),
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=self.workspace13,
            workspace2=self.workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


class CutlassInt8HybridRunner(BackendRunner):
    """Runs the standalone SM80 CUTLASS INT8 hybrid grouped MoE kernel."""

    name = "cutlass_int8_hybrid"

    def __init__(self, shape: ProblemShape, device: torch.device,
                 quant_block_size: int,
                 super_group_size: int,
                 q_max: int):
        E = shape.num_experts
        N = shape.intermediate
        K = shape.hidden
        M = shape.num_tokens
        topk = shape.top_k
        dtype = shape.dtype

        self.shape = shape
        self.quant_block_size = quant_block_size
        self.super_group_size = super_group_size
        self.q_max = q_max

        block_n = 64
        block_k = quant_block_size

        self.hidden = torch.randn(M, K, device=device, dtype=dtype)

        # Build BF16 weights, then blockwise-INT8 quantize.
        w13_bf16 = torch.randn(E, 2 * N, K, device=device, dtype=dtype) * 0.02
        w2_bf16 = torch.randn(E, K, N, device=device, dtype=dtype) * 0.02
        w13_q, w13_scale = _quantize_blockwise_int8(w13_bf16, block_n, block_k)
        w2_q, w2_scale = _quantize_blockwise_int8(w2_bf16, block_n, block_k)
        self.w13 = w13_q
        self.w2 = w2_q

        # Two-level (Q, F) decomposition along the K-axis.
        def _hybrid(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            E_, Nb, Kb = scale.shape
            num_sg = (Kb + super_group_size - 1) // super_group_size
            flat = scale.reshape(E_ * Nb, Kb).contiguous()
            Q, F = quantize_scales_for_hybrid(
                flat, Q_max=q_max, super_group_size=super_group_size)
            return (Q.reshape(E_, Nb, Kb).contiguous(),
                    F.reshape(E_, Nb, num_sg).contiguous())

        self.w13_qscale, self.w13_fscale = _hybrid(w13_scale)
        self.w2_qscale, self.w2_fscale = _hybrid(w2_scale)

        self.topk_weights, self.topk_ids = _build_routing(M, E, topk, device)

        # Workspaces
        ws1 = (M * topk, max(2 * N, K))
        ws2 = (M * topk, max(N, K))
        self.workspace13 = torch.empty(ws1, device=device, dtype=dtype)
        self.workspace2 = torch.empty(ws2, device=device, dtype=dtype)
        self.output = torch.empty(M, K, device=device, dtype=dtype)

    def run(self) -> None:
        _run_cutlass_int8_hybrid_grouped_moe(
            output=self.output,
            hidden_states=self.hidden,
            w1=self.w13,
            w2=self.w2,
            topk_ids=self.topk_ids,
            topk_weights=self.topk_weights,
            activation=MoEActivation.SILU,
            global_num_experts=self.w13.size(0),
            expert_map=None,
            w1_qscale=self.w13_qscale,
            w1_fscale=self.w13_fscale,
            w2_qscale=self.w2_qscale,
            w2_fscale=self.w2_fscale,
            workspace13=self.workspace13,
            workspace2=self.workspace2,
            quant_block_size=self.quant_block_size,
            super_group_size=self.super_group_size,
            q_max=self.q_max,
        )


class TritonInt8BlockwiseRunner(BackendRunner):
    """Runs vllm's TritonExperts on the blockwise INT8 W8A8 path."""

    name = "triton_int8_blockwise"

    def __init__(self, shape: ProblemShape, device: torch.device,
                 quant_block_size: int):
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation as VllmMoEActivation,
        )
        from vllm.model_executor.layers.fused_moe.config import (
            FusedMoEQuantConfig,
        )
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
        from vllm.model_executor.layers.fused_moe.utils import (
            moe_kernel_quantize_input,
        )

        self._vllm_activation = VllmMoEActivation.SILU

        E = shape.num_experts
        N = shape.intermediate
        K = shape.hidden
        M = shape.num_tokens
        topk = shape.top_k
        dtype = shape.dtype

        block_n = 64        # block_shape[0]
        block_k = quant_block_size
        block_shape = [block_n, block_k]

        # Pre-quantize the input activation (simulates the modular prepare stage)
        hidden = torch.randn(M, K, device=device, dtype=dtype)
        self.hidden, self.a1q_scale = moe_kernel_quantize_input(
            hidden,
            None,
            torch.int8,
            False,
            block_shape,
        )

        w13_bf16 = torch.randn(E, 2 * N, K, device=device, dtype=dtype) * 0.02
        w2_bf16 = torch.randn(E, K, N, device=device, dtype=dtype) * 0.02
        self.w13, w13_scale = _quantize_blockwise_int8(w13_bf16, block_n, block_k)
        self.w2, w2_scale = _quantize_blockwise_int8(w2_bf16, block_n, block_k)
        self.topk_weights, self.topk_ids = _build_routing(M, E, topk, device)

        moe_config = _make_vllm_moe_config(shape, device)

        quant_config = FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=w13_scale,
            w2_scale=w2_scale,
            per_act_token_quant=False,
            per_out_ch_quant=False,
            block_shape=block_shape,
        )

        self.experts = TritonExperts(
            moe_config=moe_config,
            quant_config=quant_config,
        )

        ws1, ws2, out = self.experts.workspace_shapes(
            M=M, N=N, K=K, topk=topk,
            global_num_experts=E, local_num_experts=E,
            expert_tokens_meta=None,
            activation=self._vllm_activation,
        )
        self.workspace13 = torch.empty(ws1, device=device, dtype=dtype)
        self.workspace2 = torch.empty(ws2, device=device, dtype=dtype)
        self.output = torch.empty(out, device=device, dtype=dtype)

    def run(self) -> None:
        self.experts.apply(
            output=self.output,
            hidden_states=self.hidden,
            w1=self.w13,
            w2=self.w2,
            topk_weights=self.topk_weights,
            topk_ids=self.topk_ids,
            activation=self._vllm_activation,
            global_num_experts=self.w13.size(0),
            expert_map=None,
            a1q_scale=self.a1q_scale,
            a2_scale=None,
            workspace13=self.workspace13,
            workspace2=self.workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )


# ---------------------------------------------------------------------------
# vllm FusedMoEConfig builder (single GPU, no real distributed groups)
# ---------------------------------------------------------------------------
def _make_vllm_moe_config(shape: ProblemShape, device: torch.device):
    """Build a vllm FusedMoEConfig for the given problem shape."""
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation as VllmMoEActivation,
    )

    parallel_cfg = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0, sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=shape.num_experts,
        experts_per_token=shape.top_k,
        hidden_dim=shape.hidden,
        intermediate_size_per_partition=shape.intermediate,
        num_local_experts=shape.num_experts,
        num_logical_experts=shape.num_experts,
        activation=VllmMoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel_cfg,
        in_dtype=shape.dtype,
        max_num_tokens=max(shape.num_tokens, 1),
        has_bias=False,
        is_act_and_mul=True,
        is_lora_enabled=False,
        disable_inplace=True,
    )


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------
def benchmark(runner: BackendRunner, warmup: int, iters: int,
              profile: bool) -> dict[str, float]:
    # Warmup
    for _ in range(warmup):
        runner.run()
    torch.cuda.synchronize()

    # Per-iteration events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        if profile and i == 0:
            cuda_profiler_start()

        starts[i].record()
        runner.run()
        ends[i].record()

        if profile and i == 0:
            cuda_profiler_stop()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    times_ms_sorted = sorted(times_ms)
    n = len(times_ms_sorted)
    return {
        "iters": float(iters),
        "min_ms": times_ms_sorted[0],
        "median_ms": times_ms_sorted[n // 2],
        "mean_ms": sum(times_ms_sorted) / n,
        "p90_ms": times_ms_sorted[min(n - 1, int(math.ceil(n * 0.9)) - 1)],
        "max_ms": times_ms_sorted[-1],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Default constants (same as Int8HybridOnlineMoEMethod in vllm)
DEFAULT_Q_MAX = 16
DEFAULT_SUPER_GROUP_SIZE = 256


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="INT8-hybrid MoE benchmark (cutlass_int8_hybrid is "
                    "standalone; unquantized/triton_int8_blockwise use vllm)")

    # Problem shape
    p.add_argument("--num-tokens", type=int, default=8192,
                   help="Number of input tokens (M).")
    p.add_argument("--hidden", type=int, default=2048,
                   help="Hidden size (K).")
    p.add_argument("--intermediate", type=int, default=512,
                   help="Per-expert intermediate size (N).")
    p.add_argument("--num-experts", type=int, default=256,
                   help="Number of experts.")
    p.add_argument("--top-k", type=int, default=8,
                   help="Top-k experts per token.")
    p.add_argument("--dtype", choices=["bfloat16", "float16"],
                   default="bfloat16",
                   help="Activation / weight dtype before quantization.")

    # INT8-hybrid-specific knobs
    p.add_argument("--quant-block-size", type=int, default=128,
                   help="Block size along K for blockwise INT8 quantization.")
    p.add_argument("--super-group-size", type=int,
                   default=DEFAULT_SUPER_GROUP_SIZE,
                   help="Number of K-blocks per (Q, F) super group.")
    p.add_argument("--q-max", type=int,
                   default=DEFAULT_Q_MAX,
                   help="Max abs value for INT32 Q scales.")

    # Bench loop
    p.add_argument("--backend",
                   choices=["unquantized",
                            "cutlass_int8_hybrid",
                            "triton_int8_blockwise",
                            "all"],
                   default="all",
                   help="Which backend(s) to benchmark.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--profile", action="store_true",
                   help="Wrap the first post-warmup iteration with "
                        "cudaProfilerStart/Stop.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device_index = 0
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    shape = ProblemShape(
        num_tokens=args.num_tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dtype=dtype,
    )

    print(f"[bench] device={torch.cuda.get_device_name(device)}  "
          f"shape=M={shape.num_tokens} K={shape.hidden} N={shape.intermediate} "
          f"E={shape.num_experts} top_k={shape.top_k} dtype={dtype}")
    print(f"[bench] quant_block_size={args.quant_block_size} "
          f"super_group_size={args.super_group_size} q_max={args.q_max}")
    print(f"[bench] fused_silu_quant_w2="
          f"{os.environ.get('VLLM_INT8_HYBRID_FUSED_SILU_QUANT', '1')} "
          f"fused_permute_quant_w1="
          f"{os.environ.get('VLLM_INT8_HYBRID_FUSED_PERMUTE_QUANT', '0')}")

    backends = (["unquantized", "cutlass_int8_hybrid",
                 "triton_int8_blockwise"]
                if args.backend == "all" else [args.backend])

    # When profiling, only allow a single backend per process so the nsys/ncu
    # range covers exactly one kernel launch.
    if args.profile and len(backends) > 1:
        raise SystemExit(
            "--profile requires a single --backend so the cudaProfilerApi "
            "range covers exactly one kernel launch.")

    for name in backends:
        if name == "unquantized":
            runner: BackendRunner = UnquantizedTritonRunner(shape, device)
        elif name == "cutlass_int8_hybrid":
            runner = CutlassInt8HybridRunner(
                shape, device,
                quant_block_size=args.quant_block_size,
                super_group_size=args.super_group_size,
                q_max=args.q_max,
            )
        elif name == "triton_int8_blockwise":
            runner = TritonInt8BlockwiseRunner(
                shape, device,
                quant_block_size=args.quant_block_size,
            )
        else:
            raise ValueError(f"unknown backend {name}")

        stats = benchmark(runner, warmup=args.warmup, iters=args.iters,
                          profile=args.profile)
        print(f"[bench] backend={runner.name:>22s} "
              f"min={stats['min_ms']:.3f} ms  "
              f"median={stats['median_ms']:.3f} ms  "
              f"mean={stats['mean_ms']:.3f} ms  "
              f"p90={stats['p90_ms']:.3f} ms  "
              f"max={stats['max_ms']:.3f} ms")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
