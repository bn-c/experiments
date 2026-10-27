"""Synthetic linear-representation generator (LRB-inspired, not a paper clone).

**Stays in-repo on purpose.** There is no HuggingFace Hub dataset for this
controlled generator: knobs (``n``, ``n_p``, sparsity, noise, seed) define
the ground-truth ``A`` and sparse codes. Prefer HF for every *real* corpus;
keep this local for recoverable synthetic ground truth.

Observation model
-----------------
Latents ``z ∈ R^{n_p}`` are sparse (Bernoulli-Gaussian). Observations are
``x = A z + ε`` with ``A ∈ R^{n × n_p}``. When ``n_p > n`` features are in
superposition. The training target is ``z`` (MSE recovery) so a linear map
vs MLP can be compared on the same knobs: ``n``, ``n_p``, ``sparsity``,
``noise``, ``seed``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class _RecoveryDataset(Dataset):
    def __init__(self, x: np.ndarray, z: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.z = torch.as_tensor(z, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.z[idx]


def _sample_sparse_z(rng: np.random.Generator, n: int, n_p: int, sparsity: float) -> np.ndarray:
    """Bernoulli-Gaussian sparse codes; expected fraction ``sparsity`` active."""
    mask = rng.random((n, n_p)) < sparsity
    values = rng.normal(0.0, 1.0, size=(n, n_p))
    return (mask * values).astype(np.float32)


def load_synthetic_lrb(kwargs: dict[str, Any]) -> tuple[Dataset, Dataset, dict[str, Any]]:
    n = int(kwargs.get("n", 32))  # observation dim
    n_p = int(kwargs.get("n_p", 64))  # latent / feature count
    sparsity = float(kwargs.get("sparsity", 0.1))
    noise = float(kwargs.get("noise", 0.05))
    n_train = int(kwargs.get("n_train", 512))
    n_test = int(kwargs.get("n_test", 128))
    seed = int(kwargs.get("seed", 0))

    if not (0.0 < sparsity <= 1.0):
        raise ValueError("sparsity must be in (0, 1]")
    if n < 1 or n_p < 1:
        raise ValueError("n and n_p must be >= 1")

    rng = np.random.default_rng(seed)
    # Fixed dictionary for the whole split (like a ground-truth feature matrix)
    A = rng.normal(0.0, 1.0 / np.sqrt(n), size=(n, n_p)).astype(np.float32)

    def make(m: int, rng_local: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        z = _sample_sparse_z(rng_local, m, n_p, sparsity)
        x = z @ A.T
        if noise > 0:
            x = x + rng_local.normal(0.0, noise, size=x.shape).astype(np.float32)
        return x, z

    # Independent RNG streams for train/test after A is fixed
    x_tr, z_tr = make(n_train, rng)
    x_te, z_te = make(n_test, rng)

    meta: dict[str, Any] = {
        "task": "regression",
        "in_dim": n,
        "out_dim": n_p,
        "n": n,
        "n_p": n_p,
        "sparsity": sparsity,
        "noise": noise,
        "superposition": n_p > n,
        "n_train": n_train,
        "n_test": n_test,
        # Frobenius scale of A for optional diagnostics
        "A_fro": float(np.linalg.norm(A)),
        # Ground-truth dictionary for repr/feature_align (runtime only; stripped from JSON).
        "W_true": A,
        "A": A,
    }
    return _RecoveryDataset(x_tr, z_tr), _RecoveryDataset(x_te, z_te), meta
