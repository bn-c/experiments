"""Dataset registry for the linear_bench harness."""

from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset

from .mnist1d import load_mnist1d
from .synthetic_lrb import load_synthetic_lrb

Loader = Callable[[dict[str, Any]], tuple[Dataset, Dataset, dict[str, Any]]]

REGISTRY: dict[str, Loader] = {
    "mnist1d": load_mnist1d,
    "synthetic_lrb": load_synthetic_lrb,
}


def load_dataset(name: str, kwargs: dict[str, Any] | None = None) -> tuple[Dataset, Dataset, dict[str, Any]]:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown dataset {name!r}; known: {known}")
    return REGISTRY[name](kwargs or {})


__all__ = ["REGISTRY", "load_dataset", "load_mnist1d", "load_synthetic_lrb"]
