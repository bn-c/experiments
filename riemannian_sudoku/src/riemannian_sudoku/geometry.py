"""Sphere Exp map and positive-orthant retraction for S⁸₊."""

from __future__ import annotations

import torch
from torch import Tensor

# Soft ceiling on geodesic angle per step (avoids one-step wraps).
_MAX_ANGLE = 0.5 * torch.pi


def project_tangent(s: Tensor, v: Tensor) -> Tensor:
    """Project ``v`` onto the tangent space at unit ``s`` (last dim)."""
    return v - (v * s).sum(dim=-1, keepdim=True) * s


def project_to_orthant_sphere(s: Tensor, eps: float = 1e-8) -> Tensor:
    """Retraction onto S^8_+: ``|s| / ||s||`` (not a geodesic reflection)."""
    s = s.abs()
    return s / s.norm(dim=-1, keepdim=True).clamp_min(eps)


def riemannian_exp_map(s: Tensor, v: Tensor, dt: float = 1.0, eps: float = 1e-8) -> Tensor:
    """
    Exponential map on the sphere: ``Exp_s(dt·v)`` for ``v ⊥ s``.

    Clips the angle ``||v|| dt`` to ``π/2`` so a single step cannot wrap.
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    theta = (v_norm * dt).clamp(max=_MAX_ANGLE)
    # Re-scale v_hat so Exp uses the clipped angle while keeping direction.
    v_hat = v / v_norm
    return torch.cos(theta) * s + torch.sin(theta) * v_hat


def riemannian_exp_map_orthant(s: Tensor, v: Tensor, dt: float = 1.0, eps: float = 1e-8) -> Tensor:
    """Exp map followed by abs-renorm onto the positive orthant sphere."""
    return project_to_orthant_sphere(riemannian_exp_map(s, v, dt=dt, eps=eps), eps=eps)


def riemannian_log_map(s: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Logarithmic map on the sphere: ``Log_s(y) ∈ T_s`` for unit ``s``, ``y``.

    ``Log_s(y) = (θ / sin θ) · Proj_s(y)`` with ``θ = arccos(⟨s, y⟩)``.
    Small-angle path returns the tangent projection (``θ/sinθ → 1``).
    """
    cos_theta = (s * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = cos_theta.acos()
    u = y - cos_theta * s  # = Proj_s(y) when ||s||=1
    # θ/sinθ → 1 as θ→0; avoid 0/0 with a series-friendly clamp on sin.
    sin_theta = theta.sin().clamp_min(eps)
    scale = torch.where(theta < 1e-4, torch.ones_like(theta), theta / sin_theta)
    return u * scale


def geodesic_distance(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Geodesic distance ``arccos(⟨x, y⟩)`` on the unit sphere (last dim)."""
    cos = (x * y).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return cos.acos()


def sudoku_group_indices() -> Tensor:
    """Return ``(27, 9)`` long indices for 9 rows + 9 cols + 9 boxes."""
    groups: list[list[int]] = []
    groups.extend([list(range(r * 9, r * 9 + 9)) for r in range(9)])
    groups.extend([list(range(c, 81, 9)) for c in range(9)])
    groups.extend(
        [
            [(br * 3 + dr) * 9 + bc * 3 + dc for dr in range(3) for dc in range(3)]
            for br in range(3)
            for bc in range(3)
        ]
    )
    return torch.tensor(groups, dtype=torch.long)


def unit_sum_penalty(probs: Tensor, group_indices: Tensor) -> Tensor:
    """
    MSE of per-digit group sums vs 1 (Enso-style constraint).

    Args:
        probs: ``(B, 81, 9)`` simplex probabilities ``p = s²``.
        group_indices: ``(27, 9)`` cell indices.

    Returns:
        Scalar mean penalty over batch.
    """
    # (B, 27, 9, 9) → sum over cells → (B, 27, 9)
    group_probs = probs[:, group_indices]
    digit_sums = group_probs.sum(dim=2)
    return ((digit_sums - 1.0) ** 2).mean()
