# INT8 Hybrid MoE Standalone Benchmark

Standalone project for benchmarking the **CUTLASS SM80 INT8 hybrid grouped MoE GEMM** kernel, fully independent of the vllm Python package.

All CUDA kernels are compiled from source into a single shared library (`libint8_hybrid_moe_ops.so`). The Python benchmark script uses only PyTorch + Triton + this library.

---

## Prerequisites

| Dependency | Version |
|---|---|
| CUDA Toolkit | 12.9 |
| PyTorch | 2.11.0+cu129 |
| Triton | 3.6.0 |
| CMake | 4.3.2 |
| CUTLASS | v4.4.2 (included as git submodule) |
| GPU | SM80 (A100 / A800 / etc.) |

---

## Build

```bash
cd INT8HybridMoE

# First time: initialize the CUTLASS submodule
git submodule update --init --recursive

# Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

On success, `build/libint8_hybrid_moe_ops.so` is produced.

### CMake Options

| Option | Default | Description |
|---|---|---|
| `INT8HYBRID_CUTLASS_DIR` | `include/cutlass` (git submodule) | Path to CUTLASS root (must contain `include/cutlass/cutlass.h`) |
| `INT8HYBRID_CUDA_ARCHS` | `80;89` | CUDA architectures to compile for |

> **Note:** CUTLASS is managed as a git submodule pinned to **v4.4.2** at `include/cutlass/`.
> If you need to use a different CUTLASS version, either update the submodule or pass
> `-DINT8HYBRID_CUTLASS_DIR=/path/to/cutlass` to cmake.

---

## Run Benchmark

```bash
cd /deploy/Int8MoE_Phase2/INT8HybridMoE

# Default shape (M=1024, K=4096, N=1408, E=64, top_k=8)
python3 bench_int8_hybrid_moe.py

# Custom shape
python3 bench_int8_hybrid_moe.py \
    --num-tokens 8192 --hidden 2048 --intermediate 512 \
    --num-experts 256 --top-k 8 --warmup 50 --iters 100

# Enable fused permute+quant path (C7-P0)
VLLM_INT8_HYBRID_FUSED_PERMUTE_QUANT=1 python3 bench_int8_hybrid_moe.py \
    --num-tokens 8192 --hidden 2048 --intermediate 512 \
    --num-experts 256 --top-k 8

# nsys profiling (captures exactly one kernel launch)
nsys profile -c cudaProfilerApi -t cuda,nvtx --force-overwrite=true \
    -o moe_cutlass python3 bench_int8_hybrid_moe.py --profile

# ncu profiling
ncu --set full --target-processes all --profile-from-start off \
    -o moe_cutlass python3 bench_int8_hybrid_moe.py --profile
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `INT8_HYBRID_MOE_LIB` | `build/libint8_hybrid_moe_ops.so` | Override path to the compiled `.so` |
| `VLLM_INT8_HYBRID_FUSED_SILU_QUANT` | `1` | Enable fused SiLU+quant between w13 and w2 |
| `VLLM_INT8_HYBRID_FUSED_PERMUTE_QUANT` | `0` | Enable fused permute+quant before w13 |

### Benchmark Arguments

| Argument | Default | Description |
|---|---|---|
| `--num-tokens` | 1024 | Number of input tokens (M) |
| `--hidden` | 4096 | Hidden size (K) |
| `--intermediate` | 1408 | Per-expert intermediate size (N) |
| `--num-experts` | 64 | Number of experts |
| `--top-k` | 8 | Top-k experts per token |
| `--dtype` | `bfloat16` | Activation dtype (`bfloat16` or `float16`) |
| `--quant-block-size` | 128 | Block size along K for INT8 quantization |
| `--super-group-size` | 256 | K-blocks per (Q, F) super group |
| `--q-max` | 16 | Max abs value for INT32 Q scales |
| `--warmup` | 10 | Warmup iterations |
| `--iters` | 50 | Timed iterations |
| `--profile` | off | Wrap first iteration with cudaProfilerStart/Stop |

---

## Project Structure

```
INT8HybridMoE/
├── CMakeLists.txt                          # Build system
├── bench_int8_hybrid_moe.py                # Standalone benchmark entry point
│
├── int8_hybrid_moe/                        # Python package (no vllm dependency)
│   ├── __init__.py                         #   Library loader
│   ├── ops.py                              #   Thin wrappers around torch.ops.*
│   ├── activation.py                       #   MoEActivation enum
│   ├── permute.py                          #   MoE permute/unpermute wrappers
│   ├── quant.py                            #   Triton quantization kernels
│   └── cutlass_runner.py                   #   E2E MoE forward pass orchestrator
│
├── csrc/                                   # CUDA/C++ kernel sources
│   ├── torch_bindings.cpp                  #   Unified op registration (int8_hybrid_moe namespace)
│   ├── int8_grouped_gemm.cu               #   CUTLASS INT8 per-tensor grouped GEMM
│   ├── int8_hybrid_grouped_gemm.cu        #   CUTLASS INT8 hybrid blockwise grouped GEMM
│   ├── moe_permute_unpermute_op.cu         #   MoE permute/unpermute host functions
│   ├── activation_kernels.cu              #   silu_and_mul activation kernel
│   └── permute_unpermute_kernels/          #   MoE permute/unpermute device kernels
│       ├── dispatch.h
│       ├── moe_permute_unpermute_kernel.h
│       ├── moe_permute_unpermute_kernel.cu
│       └── moe_permute_unpermute_kernel.inl
│
├── include/                                # Header files
│   ├── cutlass/                            #   CUTLASS v4.4.2 (git submodule)
│   ├── cuda_compat.h                       #   CUDA/ROCm compatibility macros
│   ├── cuda_vec_utils.cuh                  #   Vectorized load/store utilities
│   ├── dispatch_utils.h                    #   Type dispatch macros
│   ├── custom_mma_base.h                   #   Custom CUTLASS MMA base
│   ├── custom_mma_multistage.h            #   Custom CUTLASS MMA multistage
│   ├── int8_grouped_gemm.h                #   INT8 grouped GEMM declarations
│   ├── int8_hybrid_common.h               #   Hybrid GEMM common definitions
│   └── int8_hybrid_grouped_gemm.h         #   Hybrid grouped GEMM declarations
│
├── .gitmodules                             # Git submodule configuration
├── .gitignore                              # Build artifacts exclusion
│
└── build/                                  # Build output (generated)
    └── libint8_hybrid_moe_ops.so           #   Compiled shared library
```

---

## Source Provenance

| Component | Source |
|---|---|
| `int8_grouped_gemm.cu`, `int8_hybrid_grouped_gemm.cu` | `/deploy/INT8HybridMoE/src/` (standalone CUTLASS kernels) |
| `custom_mma_base.h`, `custom_mma_multistage.h`, `int8_hybrid_common.h`, `int8_hybrid_grouped_gemm.h`, `int8_grouped_gemm.h` | `/deploy/INT8HybridMoE/include/` |
| `moe_permute_unpermute_op.cu`, `permute_unpermute_kernels/*` | `vllm/csrc/moe/` (MoE token routing) |
| `activation_kernels.cu` | `vllm/csrc/activation_kernels.cu` (gated activations) |
| `cuda_compat.h`, `cuda_vec_utils.cuh`, `dispatch_utils.h` | `vllm/csrc/` (utility headers) |
| `torch_bindings.cpp` | Written for this project (unified op registration) |
| `int8_hybrid_moe/*.py` | Written for this project (minimal Python wrappers + Triton kernels) |

---

## Architecture Overview

```
bench_int8_hybrid_moe.py
    │
    ▼
int8_hybrid_moe.cutlass_runner._run_cutlass_int8_hybrid_grouped_moe()
    │
    ├─ permute.moe_permute()          ──► torch.ops.int8_hybrid_moe.moe_permute
    │                                       (CUB radix sort + expand rows)
    │
    ├─ quant.moe_kernel_quantize_input_int8_hybrid_grouped()
    │   or quant.fused_permute_quant_int8_hybrid_grouped()
    │                                       (Triton: blockwise INT8 + Q/F factorization)
    │
    ├─ ops.cutlass_int8_blockwise_hybrid_grouped()
    │                                       ──► torch.ops.int8_hybrid_moe.cutlass_int8_blockwise_hybrid_grouped
    │                                       (CUTLASS SM80 grouped GEMM, TileShape 128×64×128)
    │
    ├─ quant.fused_silu_quant_int8_hybrid_grouped()
    │   or activation.apply_moe_activation() + quant.moe_kernel_quantize_input_int8_hybrid_grouped()
    │                                       (Triton: fused SiLU + INT8 quant)
    │
    ├─ ops.cutlass_int8_blockwise_hybrid_grouped()   (w2 GEMM)
    │
    └─ permute.moe_unpermute()        ──► torch.ops.int8_hybrid_moe.moe_unpermute
                                            (weighted reduce + scatter)
```
