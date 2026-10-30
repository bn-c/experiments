"""Unit metrics: full row/col/box match vs target; early-stop cliff predicate."""

from __future__ import annotations

import torch

from riemannian_sudoku.train import _blank_and_exact, _cosine_t_max, unit_group_accuracies


def test_cosine_t_max_legacy_and_horizon() -> None:
    # Legacy: cool over remaining --steps.
    assert _cosine_t_max(steps=50000, step=0, lr_cosine_steps=0) == 50000
    assert _cosine_t_max(steps=50000, step=10000, lr_cosine_steps=0) == 40000
    # Short horizon: cool like a 10k A/B, then hold floor.
    assert _cosine_t_max(steps=50000, step=0, lr_cosine_steps=10000) == 10000
    assert _cosine_t_max(steps=50000, step=2500, lr_cosine_steps=10000) == 7500
    assert _cosine_t_max(steps=50000, step=10000, lr_cosine_steps=10000) is None
    assert _cosine_t_max(steps=50000, step=20000, lr_cosine_steps=10000) is None


def test_early_stop_cliff_predicate() -> None:
    # Mirror train loop: drop >= threshold below best counts as a bad eval.
    best, drop = 0.478, 0.15
    assert (best - 0.153) >= drop
    assert not ((best - 0.400) >= drop)
    # Two consecutive bad evals → stop (patience=2).
    bad = 0
    for val in (0.153, 0.111):
        if (best - val) >= drop:
            bad += 1
        else:
            bad = 0
    assert bad == 2


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
