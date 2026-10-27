"""Dataset registry for the linear_bench harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch.utils.data import Dataset

from .hf_vision import (
    PRESETS as HF_PRESETS,
)
from .hf_vision import (
    load_cifar10,
    load_cifar100,
    load_fashion_mnist,
    load_hf_vision,
    load_mnist,
    load_stl10,
    load_svhn,
)
from .mnist1d import load_mnist1d
from .synthetic_lrb import load_synthetic_lrb

Loader = Callable[[dict[str, Any]], tuple[Dataset, Dataset, dict[str, Any]]]

REGISTRY: dict[str, Loader] = {
    "mnist1d": load_mnist1d,
    "synthetic_lrb": load_synthetic_lrb,
    # HuggingFace multi-domain vision (flattened for linear/MLP)
    "hf": load_hf_vision,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "mnist": load_mnist,
    "svhn": load_svhn,
    "stl10": load_stl10,
    "fashion_mnist": load_fashion_mnist,
}


def load_dataset(name: str, kwargs: dict[str, Any] | None = None) -> tuple[Dataset, Dataset, dict[str, Any]]:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown dataset {name!r}; known: {known}")
    return REGISTRY[name](kwargs or {})


__all__ = [
    "REGISTRY",
    "HF_PRESETS",
    "load_dataset",
    "load_mnist1d",
    "load_synthetic_lrb",
    "load_hf_vision",
    "load_cifar10",
    "load_cifar100",
    "load_mnist",
    "load_svhn",
    "load_stl10",
    "load_fashion_mnist",
]
