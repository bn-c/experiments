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
    per_sample = masked_cross_entropy_per_sample(logits, solution, mask)
    return per_sample.mean()


def masked_cross_entropy_per_sample(
    logits: torch.Tensor,
    solution: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-example mean CE over masked cells; shape ``(B,)``."""
    targets = solution.clamp(min=1, max=9) - 1
    per_cell = nn.functional.cross_entropy(
        logits.reshape(-1, 9),
        targets.reshape(-1),
        reduction='none',
    ).view(solution.shape)
    denom = mask.sum(dim=-1).clamp(min=1)
    return (per_cell * mask).sum(dim=-1) / denom


def sequence_is_correct(
    logits: torch.Tensor,
    solution: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Whether every masked cell is correct for each example.

    Matches upstream TRM ``seq_is_correct`` (all labeled tokens correct).
    """
    pred = logits.argmax(dim=-1) + 1
    correct = (pred == solution) | (mask <= 0)
    return correct.all(dim=-1)
