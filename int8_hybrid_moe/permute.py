# SPDX-License-Identifier: Apache-2.0
"""MoE permute / unpermute wrappers (standalone, no vllm)."""

import torch

from . import ops


def moe_permute(
    hidden_states: torch.Tensor,
    a1q_scale: torch.Tensor | None,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int = -1,
    expert_map: torch.Tensor | None = None,
    permuted_hidden_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand and permute activation to gather tokens for each expert."""
    n_token, n_hidden = hidden_states.size()
    topk = topk_ids.size(1)
    permuted_row_size = n_token * topk
    if n_local_expert == -1:
        n_local_expert = n_expert
    if permuted_hidden_states is None:
        permuted_hidden_states = torch.empty(
            (permuted_row_size, n_hidden),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    token_expert_indices = torch.arange(
        0, n_token * topk, dtype=torch.int32, device=hidden_states.device
    ).reshape((n_token, topk))

    expert_first_token_offset = torch.empty(
        n_local_expert + 1, dtype=torch.int64, device=hidden_states.device
    )
    permuted_idx = torch.full(
        (permuted_row_size,),
        n_token * topk,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    inv_permuted_idx = torch.empty(
        (n_token, topk), dtype=torch.int32, device=hidden_states.device
    )
    topk_ids = topk_ids.to(torch.int32)
    ops.moe_permute(
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

    if a1q_scale is not None and a1q_scale.dim() > 1:
        a1q_scale = a1q_scale[permuted_idx.clamp(max=n_token * topk - 1) // topk]
    return (
        permuted_hidden_states,
        a1q_scale,
        expert_first_token_offset,
        inv_permuted_idx.flatten(),
        permuted_idx,
    )


def moe_permute_metadata(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    n_expert: int,
    n_local_expert: int = -1,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate MoE permutation metadata without copying activation rows."""
    n_token = hidden_states.size(0)
    topk = topk_ids.size(1)
    permuted_row_size = n_token * topk
    if n_local_expert == -1:
        n_local_expert = n_expert

    token_expert_indices = torch.arange(
        0, n_token * topk, dtype=torch.int32, device=hidden_states.device
    ).reshape((n_token, topk))

    expert_first_token_offset = torch.empty(
        n_local_expert + 1, dtype=torch.int64, device=hidden_states.device
    )
    permuted_idx = torch.full(
        (permuted_row_size,),
        n_token * topk,
        dtype=torch.int32,
        device=hidden_states.device,
    )
    inv_permuted_idx = torch.empty(
        (n_token, topk), dtype=torch.int32, device=hidden_states.device
    )
    topk_ids = topk_ids.to(torch.int32)
    ops.moe_permute_metadata(
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
    return expert_first_token_offset, inv_permuted_idx.flatten(), permuted_idx


def moe_unpermute(
    out: torch.Tensor,
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None = None,
) -> None:
    """Reduce and unpermute activation tensor."""
    topk = topk_weights.size(1)
    ops.moe_unpermute(
        permuted_hidden_states,
        topk_weights,
        inv_permuted_idx,
        expert_first_token_offset,
        topk,
        out,
    )
