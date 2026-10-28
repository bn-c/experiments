"""Dataset loader for Hugging Face ``sapientinc/sudoku-extreme``."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from ebm.utils.config import config

logger = logging.getLogger(__name__)

HF_DATASET_ID = 'sapientinc/sudoku-extreme'
_SPLIT_FILES = {
    'train': 'train.csv',
    'test': 'test.csv',
}


def normalize_board(board: str) -> str:
    """
    Normalize an 81-char Sudoku string to Enso's digit encoding.

    ``sudoku-extreme`` uses ``.`` for blanks; Enso expects ``0``.
    """
    if len(board) != 81:
        msg = f'Expected 81-character board, got length {len(board)}'
        raise ValueError(msg)
    return board.replace('.', '0')


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map HF columns onto Enso ``puzzle`` / ``solution`` and normalize blanks."""
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


class SudokuDataset:
    """
    Provides access to ``sapientinc/sudoku-extreme`` via Hugging Face Hub CSVs.

    Official splits: ``train`` (~3.8M) and ``test`` (~423k). Boards are normalized
    so blanks are ``0`` and columns match Enso's ``puzzle`` / ``solution`` schema.

    ``n_samples`` uses ``pandas.read_csv(..., nrows=...)`` so smoke runs only parse
    the head of the CSV after Hub download/cache.
    """

    def __init__(self, dataset_id: str = HF_DATASET_ID) -> None:
        """Initialize the dataset loader (HF default cache; no package-local dir)."""
        self.dataset_id = dataset_id
        # Kept for callers/tests that still read data_dir; downloads use HF default.
        self._data_dir = config.data_dir

    def _csv_path(self, split: str) -> Path:
        if split not in _SPLIT_FILES:
            msg = f'Unknown split {split!r}; expected one of {sorted(_SPLIT_FILES)}'
            raise ValueError(msg)
        # Omit cache_dir → huggingface_hub uses $HF_HOME / ~/.cache/huggingface.
        path = hf_hub_download(
            repo_id=self.dataset_id,
            filename=_SPLIT_FILES[split],
            repo_type='dataset',
        )
        return Path(path)

    def load_split(self, split: str = 'train', n_samples: int | None = None) -> pd.DataFrame:
        """
        Load one HF split as a DataFrame with Enso column names.

        Args:
            split: Hugging Face split name (``train`` or ``test``).
            n_samples: Optional cap on rows (CSV head only).

        """
        logger.info('Loading %s split=%s (n_samples=%s)', self.dataset_id, split, n_samples)
        csv_path = self._csv_path(split)
        # Keep columns as strings so digit boards never get numeric coercion.
        df = pd.read_csv(csv_path, dtype=str, nrows=n_samples)
        return _normalize_frame(df)

    def load_all(self) -> pd.DataFrame:
        """Load the full train split (legacy helper used by older call sites)."""
        return self.load_split('train')

    def load_head(self, k: int = 100) -> pd.DataFrame:
        """Load the first *k* rows of the train split."""
        return self.load_split('train', n_samples=k)

    def load_fraction(self, frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
        """
        Load a random fraction of the train split without reading every row into RAM.

        Args:
            frac: Fraction of the dataset to return (0.0, 1.0].
            seed: Random seed for reproducible row keeps.

        """
        import random

        if not 0.0 < frac <= 1.0:
            msg = f'frac must be in (0, 1], got {frac}'
            raise ValueError(msg)
        csv_path = self._csv_path('train')
        rng = random.Random(seed)
        df = pd.read_csv(
            csv_path,
            dtype=str,
            skiprows=lambda i: i > 0 and rng.random() > frac,
        )
        return _normalize_frame(df)
