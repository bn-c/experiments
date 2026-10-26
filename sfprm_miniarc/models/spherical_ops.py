"""Spherical geometry helpers for nGPT-style dynamics on S^{d-1}."""

from __future__ import annotations

import torch


def project_to_sphere(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Projects latent vectors onto the unit hypersphere."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def slerp(
    u: torch.Tensor,
    v: torch.Tensor,
    lambda_param: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Spherical linear interpolation between u and v along the great-circle arc.

    u, v: [..., Dim] on S^{d-1}
    lambda_param: scalar or broadcastable tensor in (0, 1)
    """
    dot = torch.sum(u * v, dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    small_angle_mask = sin_omega < eps
    res_slerp = (torch.sin((1.0 - lambda_param) * omega) / (sin_omega + eps)) * u + (
        torch.sin(lambda_param * omega) / (sin_omega + eps)
    ) * v
    res_lerp = (1.0 - lambda_param) * u + lambda_param * v

    return project_to_sphere(torch.where(small_angle_mask, res_lerp, res_slerp))


def geodesic_damping(z_prev: torch.Tensor, z_prop: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Rotates z_prev toward proposal z_prop by fraction eta along S^{d-1}."""
    return slerp(z_prev, z_prop, eta)
