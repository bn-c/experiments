"""PyTorch Dataset for Sudoku puzzles."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SudokuTorchDataset(Dataset):
    """
    Convert 81-char puzzle/solution strings to integer grids.

    Returns:
        puzzle: (81,) int64 — digits 0-9 (0 = empty)
        solution: (81,) int64 — digits 1-9
        mask: (81,) float — 1 on empty cells (loss applied), 0 on givens

    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.puzzles = self._parse_strings(df['puzzle'].values)
        self.solutions = self._parse_strings(df['solution'].values)

    @staticmethod
    def _parse_strings(strings: np.ndarray) -> np.ndarray:
        count = len(strings)
        result = np.zeros((count, 81), dtype=np.int64)
        for idx, string in enumerate(strings):
            digits = np.frombuffer(string.encode('ascii'), dtype=np.uint8) - ord('0')
            result[idx] = digits
        return result

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        puzzle = self.puzzles[index]
        solution = self.solutions[index]
        empty = puzzle == 0
        return {
            'puzzle': torch.from_numpy(puzzle),
            'solution': torch.from_numpy(solution),
            'mask': torch.from_numpy(empty.astype(np.float32)),
        }
