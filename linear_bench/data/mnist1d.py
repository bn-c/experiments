"""MNIST-1D loader (Greydanus et al.) with offline synthetic stand-in.

Primary path: download+cache the official ``mnist1d_data.pkl``.
Fallback (``source: synthetic`` or download failure): a tiny template-based
1D-digit generator with the same shapes so smoke runs stay offline.
"""

from __future__ import annotations

import pickle
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Official pickle shipped by greydanus/mnist1d
_MNIST1D_URL = (
    "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
)
_DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "mnist1d_data.pkl"

# Sequence length of the official dataset
SEQ_LEN = 40
N_CLASSES = 10


class _ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _download_pickle(cache: Path) -> dict[str, Any]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        urllib.request.urlretrieve(_MNIST1D_URL, cache)  # nosec B310 — fixed HTTPS URL
    with cache.open("rb") as f:
        return pickle.load(f)


def _synthetic_mnist1d(
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Faithful tiny stand-in: class-conditional bumps + noise/scale/shift."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        y = rng.integers(0, N_CLASSES, size=n)
        x = np.zeros((n, SEQ_LEN), dtype=np.float32)
        for i, label in enumerate(y):
            # Digits as localized Gaussians at class-dependent centers + harmonics
            center = 0.1 + 0.08 * int(label)
            width = 0.04 + 0.01 * (int(label) % 3)
            bump = np.exp(-0.5 * ((t - center) / width) ** 2)
            harm = 0.3 * np.sin(2 * np.pi * (label + 1) * t)
            scale = float(rng.uniform(0.7, 1.3))
            shift = int(rng.integers(-3, 4))
            signal = scale * (bump + harm)
            signal = np.roll(signal, shift)
            x[i] = signal + rng.normal(0.0, 0.15, size=SEQ_LEN).astype(np.float32)
        return x, y.astype(np.int64)

    x_tr, y_tr = make(n_train)
    x_te, y_te = make(n_test)
    return x_tr, y_tr, x_te, y_te


def load_mnist1d(kwargs: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    seed = int(kwargs.get("seed", 0))
    source = str(kwargs.get("source", "official"))  # official | synthetic
    cache = Path(kwargs.get("cache", _DEFAULT_CACHE))
    # Optional subset for fast smoke (None = full)
    max_train = kwargs.get("max_train")
    max_test = kwargs.get("max_test")

    meta: dict[str, Any] = {
        "task": "classification",
        "in_dim": SEQ_LEN,
        "out_dim": N_CLASSES,
        "n_classes": N_CLASSES,
    }

    if source == "synthetic":
        n_train = int(kwargs.get("n_train", 400))
        n_test = int(kwargs.get("n_test", 100))
        x_tr, y_tr, x_te, y_te = _synthetic_mnist1d(n_train, n_test, seed)
        meta["source"] = "synthetic"
    else:
        try:
            data = _download_pickle(cache)
            x_tr = np.asarray(data["x"], dtype=np.float32)
            y_tr = np.asarray(data["y"], dtype=np.int64)
            x_te = np.asarray(data["x_test"], dtype=np.float32)
            y_te = np.asarray(data["y_test"], dtype=np.int64)
            meta["source"] = "official"
        except Exception as exc:  # noqa: BLE001 — offline / network guard
            n_train = int(kwargs.get("n_train", 400))
            n_test = int(kwargs.get("n_test", 100))
            x_tr, y_tr, x_te, y_te = _synthetic_mnist1d(n_train, n_test, seed)
            meta["source"] = f"synthetic_fallback({type(exc).__name__})"

    if max_train is not None:
        m = int(max_train)
        x_tr, y_tr = x_tr[:m], y_tr[:m]
    if max_test is not None:
        m = int(max_test)
        x_te, y_te = x_te[:m], y_te[:m]

    # Zero-mean / unit-ish scale on train stats
    mean = x_tr.mean()
    std = x_tr.std() + 1e-6
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    train_ds: Dataset = _ArrayDataset(x_tr, y_tr)
    test_ds: Dataset = _ArrayDataset(x_te, y_te)
    meta["n_train"] = len(train_ds)
    meta["n_test"] = len(test_ds)
    return train_ds, test_ds, meta
