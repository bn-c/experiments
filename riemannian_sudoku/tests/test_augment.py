"""Automorphism preserves solution validity relative to puzzle clues."""

from __future__ import annotations

import numpy as np

from riemannian_sudoku.augment import apply_automorphism


def _valid_relative(puzzle: np.ndarray, solution: np.ndarray) -> bool:
    clues = puzzle != 0
    return bool(np.all(puzzle[clues] == solution[clues]))


def test_automorphism_preserves_clue_agreement() -> None:
    rng = np.random.default_rng(0)
    # Minimal fake: full solution board + sparse clues copied from it.
    solution = np.tile(np.arange(1, 10), 9)
    puzzle = solution.copy()
    puzzle[rng.choice(81, size=40, replace=False)] = 0
    for _ in range(20):
        p2, s2 = apply_automorphism(puzzle, solution, rng)
        assert _valid_relative(p2, s2)
        assert p2.shape == (81,) and s2.shape == (81,)
        assert set(s2.tolist()) <= set(range(1, 10))
