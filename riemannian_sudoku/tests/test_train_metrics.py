"""Unit metrics: full row/col/box match vs target."""

from __future__ import annotations

import torch

from riemannian_sudoku.train import _blank_and_exact, unit_group_accuracies


def test_unit_group_accuracies_perfect_and_partial() -> None:
    # One perfect board + one with only first row correct.
    sol = torch.arange(1, 10).repeat(9).view(1, 81).expand(2, 81).clone()
    pred = sol.clone()
    # Break board 1 everywhere except row 0.
    pred[1, 9:] = ((pred[1, 9:] % 9) + 1)

    units = unit_group_accuracies(pred, sol)
    # Mean over (board, unit): board0 9/9 rows, board1 1/9 → 10/18.
    assert abs(units['row_acc'] - 10 / 18) < 1e-6
    # Col/box: board0 all match; board1 none (row0 alone doesn't complete a col/box).
    assert abs(units['col_acc'] - 0.5) < 1e-6
    assert abs(units['box_acc'] - 0.5) < 1e-6


def test_blank_and_exact() -> None:
    sol = torch.ones(2, 81, dtype=torch.long)
    pred = sol.clone()
    pred[1, 0] = 2
    mask = torch.ones(2, 81)
    blank, exact = _blank_and_exact(pred, sol, mask)
    assert exact == 0.5
    assert abs(blank - (161 / 162)) < 1e-6
