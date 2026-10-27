"""Pluggable model registry. Add a formulation by registering a builder."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .linear import build_linear
from .mlp import build_mlp

Builder = Callable[[int, int, dict[str, Any]], nn.Module]

REGISTRY: dict[str, Builder] = {
    "linear": build_linear,
    "mlp": build_mlp,
}


def register(name: str, builder: Builder) -> None:
    """Register a new formulation under ``name`` (overwrites if present)."""
    REGISTRY[name] = builder


def build_model(name: str, in_dim: int, out_dim: int, kwargs: dict[str, Any] | None = None) -> nn.Module:
    if name not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown model {name!r}; known: {known}")
    return REGISTRY[name](in_dim, out_dim, kwargs or {})


__all__ = ["REGISTRY", "register", "build_model", "build_linear", "build_mlp"]
