"""Deterministic train/val holdout from a loaded train DataFrame."""

from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_val_size(n: int, val_size: int, *, fraction: float = 0.05) -> int:
    """
    Return a validation count that leaves at least one training row.

    When ``val_size`` would consume the whole dataset (typical for small
    ``--train-size`` smoke runs against large default ``val_size``), scale the
    holdout to ``fraction`` of ``n``, at least 1 when possible.
    """
    min_n = 2
    if n < min_n:
        msg = f'Need at least {min_n} samples for train/val split, got {n}'
        raise ValueError(msg)
    if val_size < n:
        return val_size
    return max(1, min(n - 1, int(n * fraction)))


def split_train_val(df: pd.DataFrame, val_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and validation sets deterministically."""
    n = len(df)
    if val_size < 1 or val_size >= n:
        msg = f'val_size must be in [1, {n - 1}], got {val_size}'
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df
