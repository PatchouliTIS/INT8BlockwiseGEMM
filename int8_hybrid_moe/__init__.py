"""INT8 Hybrid MoE standalone benchmark package.

Loads a single pre-built shared library (libint8_hybrid_moe_ops.so) that
contains all required CUDA kernels:
  - CUTLASS INT8 hybrid grouped GEMM
  - MoE permute / unpermute
  - silu_and_mul activation

All ops are registered under the torch.ops.int8_hybrid_moe namespace.
No vllm Python package dependency is required.
"""

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Library path resolution
# ---------------------------------------------------------------------------
_DEFAULT_LIB_PATH = str(
    Path(__file__).resolve().parent.parent / "build" / "libint8_hybrid_moe_ops.so"
)
_LIB_PATH = os.environ.get("INT8_HYBRID_MOE_LIB", _DEFAULT_LIB_PATH)


def _load_library():
    """Load the CUDA kernel library. Call once at import time."""
    if not os.path.isfile(_LIB_PATH):
        raise RuntimeError(
            f"Cannot find libint8_hybrid_moe_ops.so at {_LIB_PATH}.\n"
            f"Please build the project first:\n"
            f"  mkdir -p build && cd build && cmake .. && make -j$(nproc)\n"
            f"Or set INT8_HYBRID_MOE_LIB to the path of the compiled .so."
        )
    torch.ops.load_library(_LIB_PATH)


_load_library()
