"""Sudoku automorphism augmentation (full orbit generators)."""

from __future__ import annotations

import numpy as np


def _as_grid(board: np.ndarray) -> np.ndarray:
    """``(81,)`` → ``(9, 9)`` int copy."""
    return board.reshape(9, 9).copy()


def _flat(grid: np.ndarray) -> np.ndarray:
    return grid.reshape(81)


def _apply_digit_perm(board: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return perm[board]


def _apply_row_order(board: np.ndarray, row_order: np.ndarray) -> np.ndarray:
    return _flat(_as_grid(board)[row_order])


def _apply_col_order(board: np.ndarray, col_order: np.ndarray) -> np.ndarray:
    return _flat(_as_grid(board)[:, col_order])


def apply_automorphism(
    puzzle: np.ndarray,
    solution: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a random Sudoku automorphism to a puzzle/solution pair.

    Same random generators for both boards: digit relabel, within-band/stack
    shuffles, band/stack permutations, optional transpose.
    """
    if rng is None:
        rng = np.random.default_rng()

    digit_perm = np.zeros(10, dtype=puzzle.dtype)
    digit_perm[1:] = rng.permutation(9) + 1
    puzzle = _apply_digit_perm(puzzle, digit_perm)
    solution = _apply_digit_perm(solution, digit_perm)

    row_order = np.concatenate([rng.permutation(3) + b * 3 for b in range(3)])
    puzzle = _apply_row_order(puzzle, row_order)
    solution = _apply_row_order(solution, row_order)

    col_order = np.concatenate([rng.permutation(3) + s * 3 for s in range(3)])
    puzzle = _apply_col_order(puzzle, col_order)
    solution = _apply_col_order(solution, col_order)

    band_order = rng.permutation(3)
    row_order = np.concatenate([np.arange(3) + b * 3 for b in band_order])
    puzzle = _apply_row_order(puzzle, row_order)
    solution = _apply_row_order(solution, row_order)

    stack_order = rng.permutation(3)
    col_order = np.concatenate([np.arange(3) + s * 3 for s in stack_order])
    puzzle = _apply_col_order(puzzle, col_order)
    solution = _apply_col_order(solution, col_order)

    if rng.random() < 0.5:
        puzzle = _flat(_as_grid(puzzle).T)
        solution = _flat(_as_grid(solution).T)

    return puzzle, solution
