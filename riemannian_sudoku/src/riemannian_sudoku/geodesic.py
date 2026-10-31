"""Locked geodesic step on ``S^{D-1}`` for bipartite Sn–Tm (hazard order)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

_DELTA_MAX = 0.5 * math.pi
_EPS_NORM = 1e-12


def geodesic_step(
    x: Tensor,
    u: Tensor,
    x0: Tensor,
    is_clue: Tensor,
    *,
    delta_max: float = _DELTA_MAX,
    return_v_c: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Sphere Exp with tanh soft-cap on **tangent** speed (never ambient before project).

    Strict order:
      (1) ``v_tan = u - (x·u) x``
      (2) ``||v|| = sqrt(||v||² + 1e-12)``
      (3) ``θ = Δmax tanh(||v|| / Δmax)``, ``Δmax = π/2``
      (4) ``v_c = v · θ / ||v||``
      (5) ``cosθ x + sinc(θ/π) v_c``
      (6) normalize
      (7) ``where(is_clue, x0, ·)``

    When ``return_v_c``, also returns post-cap tangent (zeroed on clues) for momentum.
    """
    # (1) project ambient u to T_x — no tanh on ambient.
    xu = (x * u).sum(dim=-1, keepdim=True)
    v = u - xu * x
    # (2)
    v_norm = (v.square().sum(dim=-1, keepdim=True) + _EPS_NORM).sqrt()
    # (3)
    theta = delta_max * torch.tanh(v_norm / delta_max)
    # (4)
    v_c = v * (theta / v_norm)
    # (5) torch.sinc is normalized: sinc(θ/π) = sin(θ)/θ
    y = torch.cos(theta) * x + torch.sinc(theta / math.pi) * v_c
    # (6)
    y = F.normalize(y, dim=-1, eps=1e-6)
    # (7) clue freeze to init poles
    clue_m = is_clue.unsqueeze(-1)
    y = torch.where(clue_m, x0, y)
    if return_v_c:
        return y, torch.where(clue_m, torch.zeros_like(v_c), v_c)
    return y


def free_sphere_step(
    x: Tensor,
    u: Tensor,
    *,
    delta_max: float = _DELTA_MAX,
    return_v_c: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Project-before-cap Exp on the free sphere (no clue freeze); optional post-cap ``v_c``."""
    xu = (x * u).sum(dim=-1, keepdim=True)
    v = u - xu * x
    v_norm = (v.square().sum(dim=-1, keepdim=True) + _EPS_NORM).sqrt()
    theta = delta_max * torch.tanh(v_norm / delta_max)
    v_c = v * (theta / v_norm)
    y = torch.cos(theta) * x + torch.sinc(theta / math.pi) * v_c
    y = F.normalize(y, dim=-1, eps=1e-6)
    if return_v_c:
        return y, v_c
    return y


def torus_step(y: Tensor, omega: Tensor) -> Tensor:
    """Advance house angles: ``y ← y + ω`` (unwrapped; VF learns scale)."""
    return y + omega
