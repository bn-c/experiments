"""Dataset loader for Hugging Face ``sapientinc/sudoku-extreme``."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

HF_DATASET_ID = 'sapientinc/sudoku-extreme'
_SPLIT_FILES = {
    'train': 'train.csv',
    'test': 'test.csv',
}


class DataSettings(BaseSettings):
    """Local cache paths."""

    data_dir: Path = Path('data')

    model_config = SettingsConfigDict(env_prefix='TRM_')


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


class SudokuDataset:
    """
    Access ``sapientinc/sudoku-extreme`` via Hugging Face Hub CSVs.

    Official splits: ``train`` (~3.8M) and ``test`` (~423k). The paper trains on
    1K puzzles with heavy rule-preserving augmentation; this staging loader exposes
    ``n_samples`` on the train head only (see README).
    """

    def __init__(self, dataset_id: str = HF_DATASET_ID, data_dir: Path | None = None) -> None:
        settings = DataSettings()
        self.dataset_id = dataset_id
        self._data_dir = data_dir or settings.data_dir
        self._cache_dir = self._data_dir / 'hf_cache'
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _csv_path(self, split: str) -> Path:
        if split not in _SPLIT_FILES:
            msg = f'Unknown split {split!r}; expected one of {sorted(_SPLIT_FILES)}'
            raise ValueError(msg)
        path = hf_hub_download(
            repo_id=self.dataset_id,
            filename=_SPLIT_FILES[split],
            repo_type='dataset',
            cache_dir=str(self._cache_dir),
        )
        return Path(path)

    def load_split(self, split: str = 'train', n_samples: int | None = None) -> pd.DataFrame:
        """Load one HF split as a DataFrame with ``puzzle`` / ``solution`` columns."""
        logger.info('Loading %s split=%s (n_samples=%s)', self.dataset_id, split, n_samples)
        csv_path = self._csv_path(split)
        df = pd.read_csv(csv_path, dtype=str, nrows=n_samples)
        return _normalize_frame(df)
