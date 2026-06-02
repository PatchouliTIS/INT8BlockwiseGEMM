# SPDX-License-Identifier: Apache-2.0
"""Minimal MoE activation enum and apply function (standalone, no vllm)."""

from enum import Enum

import torch

from . import ops


class MoEActivation(Enum):
    """Activation functions for MoE layers."""

    # Gated activations (gate * activation(up)) expect input [..., 2*d] -> [..., d]
    SILU = "silu"
    GELU = "gelu"

    # Non-gated activations
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.value.endswith("_no_mul")


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """Apply MoE activation function using the loaded CUDA kernels."""
    assert input.dim() == 2
    assert output.dim() == 2

    if activation == MoEActivation.SILU:
        ops.silu_and_mul(output, input)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    return output
