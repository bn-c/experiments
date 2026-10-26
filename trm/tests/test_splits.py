"""Tests for deterministic train/val splitting."""

from __future__ import annotations

import pandas as pd

from trm.dataset.splits import resolve_val_size, split_train_val


def _make_df(n: int) -> pd.DataFrame:
    return pd.DataFrame({'puzzle': ['0' * 81] * n, 'solution': ['1' * 81] * n})


def test_resolve_val_size_scales_down_for_small_n() -> None:
    assert resolve_val_size(32, 500) == 1


def test_resolve_val_size_keeps_default_when_room() -> None:
    assert resolve_val_size(2000, 500) == 500


def test_split_train_val_sizes() -> None:
    df = _make_df(100)
    train, val = split_train_val(df, val_size=20, seed=0)
    assert len(train) == 80
    assert len(val) == 20


def test_split_train_val_deterministic() -> None:
    df = _make_df(50)
    train1, val1 = split_train_val(df, val_size=10, seed=42)
    train2, val2 = split_train_val(df, val_size=10, seed=42)
    assert train1['puzzle'].tolist() == train2['puzzle'].tolist()
    assert val1['puzzle'].tolist() == val2['puzzle'].tolist()
