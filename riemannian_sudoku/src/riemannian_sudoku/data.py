"""Hugging Face ``sapientinc/sudoku-extreme`` loader (desk HF default cache)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset

from riemannian_sudoku.augment import apply_automorphism

logger = logging.getLogger(__name__)

HF_DATASET_ID = 'sapientinc/sudoku-extreme'
_SPLIT_FILES = {
    'train': 'train.csv',
    'test': 'test.csv',
}


def normalize_board(board: str) -> str:
    """Normalize an 81-char Sudoku string (``.`` blanks → ``0``)."""
    if len(board) != 81:
        msg = f'Expected 81-character board, got length {len(board)}'
        raise ValueError(msg)
    return board.replace('.', '0')


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if 'question' not in df.columns or 'answer' not in df.columns:
        msg = f'Expected question/answer columns, got {list(df.columns)}'
        raise ValueError(msg)
    out = pd.DataFrame(
        {
            'puzzle': df['question'].astype(str).map(normalize_board),
            'solution': df['answer'].astype(str).map(normalize_board),
        }
    )
    if 'source' in df.columns:
        out['source'] = df['source'].values
    if 'rating' in df.columns:
        out['rating'] = df['rating'].values
    return out


def default_hf_cache_dir() -> Path:
    """Desk Hugging Face default hub cache (``$HF_HOME/hub`` or ``~/.cache/huggingface/hub``)."""
    import os

    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        return Path(hf_home) / 'hub'
    hub = os.environ.get('HUGGINGFACE_HUB_CACHE')
    if hub:
        return Path(hub)
    return Path.home() / '.cache' / 'huggingface' / 'hub'


class SudokuDataset:
    """Load ``sapientinc/sudoku-extreme`` CSVs via the HF default cache (no package-local dir)."""

    def __init__(self, dataset_id: str = HF_DATASET_ID) -> None:
        self.dataset_id = dataset_id

    def _csv_path(self, split: str) -> Path:
        if split not in _SPLIT_FILES:
            msg = f'Unknown split {split!r}; expected one of {sorted(_SPLIT_FILES)}'
            raise ValueError(msg)
        # Omit cache_dir → huggingface_hub uses HF default.
        path = hf_hub_download(
            repo_id=self.dataset_id,
            filename=_SPLIT_FILES[split],
            repo_type='dataset',
        )
        return Path(path)

    def load_split(self, split: str = 'train', n_samples: int | None = None) -> pd.DataFrame:
        """Load one HF split as ``puzzle`` / ``solution`` columns."""
        logger.info('Loading %s split=%s (n_samples=%s)', self.dataset_id, split, n_samples)
        csv_path = self._csv_path(split)
        df = pd.read_csv(csv_path, dtype=str, nrows=n_samples)
        return _normalize_frame(df)


def _parse_strings(strings: np.ndarray) -> np.ndarray:
    count = len(strings)
    result = np.zeros((count, 81), dtype=np.int64)
    for idx, string in enumerate(strings):
        digits = np.frombuffer(string.encode('ascii'), dtype=np.uint8) - ord('0')
        result[idx] = digits
    return result


class SudokuTorchDataset(Dataset):
    """
    Convert puzzle/solution strings to integer grids.

    Returns ``puzzle`` (0-9), ``solution`` (1-9), ``mask`` (1 on blanks).
    Optional train-time automorphism when ``augment=True``.
    """

    def __init__(self, df: pd.DataFrame, *, augment: bool = False, seed: int = 0) -> None:
        self.puzzles = _parse_strings(df['puzzle'].values)
        self.solutions = _parse_strings(df['solution'].values)
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        puzzle = self.puzzles[index].copy()
        solution = self.solutions[index].copy()
        if self.augment:
            puzzle, solution = apply_automorphism(puzzle, solution, self._rng)
        empty = puzzle == 0
        return {
            'puzzle': torch.from_numpy(puzzle),
            'solution': torch.from_numpy(solution),
            'mask': torch.from_numpy(empty.astype(np.float32)),
        }


def split_train_val(
    df: pd.DataFrame,
    val_size: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last ``val_size`` rows after a seeded shuffle of the loaded head."""
    if val_size <= 0 or val_size >= len(df):
        msg = f'val_size must be in (0, {len(df)}), got {val_size}'
        raise ValueError(msg)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = shuffled.iloc[-val_size:].reset_index(drop=True)
    train_df = shuffled.iloc[:-val_size].reset_index(drop=True)
    return train_df, val_df
