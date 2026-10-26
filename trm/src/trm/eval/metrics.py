"""Sudoku accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SudokuMetrics:
    """Aggregate accuracy over a batch or dataset."""

    n_puzzles: int
    cell_accuracy: float
    puzzle_accuracy: float


def compute_metrics(
    pred_digits: torch.Tensor,
    solution: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> SudokuMetrics:
    """
    Compute cell and puzzle accuracy.

    Args:
        pred_digits: (N, 81) predicted digits 1-9
        solution: (N, 81) ground truth
        mask: optional (N, 81) — if set, cell accuracy uses only masked cells

    """
    correct = pred_digits == solution
    if mask is not None:
        denom = mask.sum().clamp(min=1)
        cell_acc = float((correct & (mask > 0)).sum().item() / denom.item())
    else:
        cell_acc = float(correct.float().mean().item())
    puzzle_acc = float(correct.all(dim=1).float().mean().item())
    return SudokuMetrics(n_puzzles=pred_digits.shape[0], cell_accuracy=cell_acc, puzzle_accuracy=puzzle_acc)
