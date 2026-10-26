"""Loss functions."""

from __future__ import annotations

import torch
from torch import nn


def masked_cross_entropy(
    logits: torch.Tensor,
    solution: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy on Sudoku cells, averaged over masked positions.

    Args:
        logits: (B, 81, 9)
        solution: (B, 81) digits 1-9
        mask: (B, 81) — 1 where loss applies (empty cells)

    """
    targets = solution.clamp(min=1, max=9) - 1
    per_cell = nn.functional.cross_entropy(
        logits.reshape(-1, 9),
        targets.reshape(-1),
        reduction='none',
    )
    mask_flat = mask.reshape(-1)
    if mask_flat.sum() == 0:
        return per_cell.mean()
    return (per_cell * mask_flat).sum() / mask_flat.sum()
