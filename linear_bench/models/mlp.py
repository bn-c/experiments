"""Small MLP with width / depth / activation knobs."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


def build_mlp(in_dim: int, out_dim: int, kwargs: dict[str, Any]) -> nn.Module:
    width = int(kwargs.get("width", 64))
    depth = int(kwargs.get("depth", 2))  # number of hidden layers
    act_name = str(kwargs.get("activation", "relu")).lower()
    bias = bool(kwargs.get("bias", True))
    if depth < 1:
        raise ValueError("mlp depth must be >= 1")
    if act_name not in _ACTIVATIONS:
        raise KeyError(f"unknown activation {act_name!r}; known: {sorted(_ACTIVATIONS)}")
    act_cls = _ACTIVATIONS[act_name]

    layers: list[nn.Module] = [nn.Linear(in_dim, width, bias=bias), act_cls()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width, bias=bias), act_cls()])
    layers.append(nn.Linear(width, out_dim, bias=bias))
    return nn.Sequential(*layers)
