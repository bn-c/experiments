"""Plain linear map: y = Wx + b."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_linear(in_dim: int, out_dim: int, kwargs: dict[str, Any]) -> nn.Module:
    bias = bool(kwargs.get("bias", True))
    return nn.Linear(in_dim, out_dim, bias=bias)
