"""Product-of-circles torus geometry in R^d (d even).

Pairs use **RoPE half-split** layout ``(x[..., i], x[..., i+d/2])`` so standard
RoPE 2-planes stay on-manifold (no post-RoPE reproject). ``project_to_torus``
sets equal pair radius ``1/√n_pairs`` → ambient ‖z‖₂≈1. Interp is per-pair 2D
guarded Slerp on unit directions then re-scaled. Halt residual is mean-over-pairs
direction chordal then max over tokens (r ∈ [0, 2], τ=0.01).

Not paper-faithful SFPRM — torus A/B experiment only.
"""

from __future__ import annotations

import math

import torch


def _as_pairs(x: torch.Tensor) -> torch.Tensor:
    """Half-split pairs matching RoPE ``_rotate_half`` planes: [..., n_pairs, 2]."""
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"torus ambient dim must be even, got {d}")
    n = d // 2
    return torch.stack((x[..., :n], x[..., n:]), dim=-1)


def _from_pairs(pairs: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_as_pairs``: [..., n, 2] → [..., 2n] half-split layout."""
    return torch.cat((pairs[..., 0], pairs[..., 1]), dim=-1)


def _pair_radius(n_pairs: int) -> float:
    """Equal per-pair radius so ambient ‖z‖₂ = 1 when all pairs are live."""
    return float(n_pairs) ** -0.5


def project_to_torus(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Per-pair direction renorm with radius ``1/√n_pairs`` (unit ambient).

    Half-split pairs on last dim (RoPE-aligned). Weight rows / latents / Q/K
    share this retract so ‖W_row‖₂≈1 and ‖z‖₂≈1 like the sphere nGPT path.
    """
    pairs = _as_pairs(x)
    n_pairs = pairs.shape[-2]
    norms = pairs.norm(dim=-1, keepdim=True).clamp_min(eps)
    return _from_pairs(pairs / norms * _pair_radius(n_pairs))


def nlerp(u: torch.Tensor, v: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Normalized lerp, then retract to the product torus (unit ambient)."""
    if not torch.is_tensor(eta):
        eta = torch.as_tensor(eta, device=u.device, dtype=u.dtype)
    while eta.ndim < u.ndim:
        eta = eta.unsqueeze(-1)
    return project_to_torus((1.0 - eta) * u + eta * v)


def _perp_unit_2d(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Analytic unit normal to u in R^2: (-u_y, u_x)."""
    del eps
    e = torch.stack((-u[..., 1], u[..., 0]), dim=-1)
    flip = (e * v).sum(dim=-1, keepdim=True) < 0
    return torch.where(flip, -e, e)


def torus_interp(
    u: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    *,
    eps_omega: float = 1e-4,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Per-pair guarded circular interpolation on unit directions, then unit-ambient scale."""
    if not torch.is_tensor(eta):
        eta = torch.as_tensor(eta, device=u.device, dtype=u.dtype)
    eta_b = eta
    while eta_b.ndim < u.ndim:
        eta_b = eta_b.unsqueeze(-1)

    up = _as_pairs(u)
    vp = _as_pairs(v)
    n_pairs = up.shape[-2]
    radius = _pair_radius(n_pairs)
    up = up / up.norm(dim=-1, keepdim=True).clamp_min(eps)
    vp = vp / vp.norm(dim=-1, keepdim=True).clamp_min(eps)

    eta_p = eta_b.unsqueeze(-1)
    while eta_p.ndim < up.ndim:
        eta_p = eta_p.unsqueeze(-1)

    c = (up * vp).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(c)
    sin_omega = torch.sin(omega).clamp_min(eps)

    slerp = (torch.sin((1.0 - eta_p) * omega) / sin_omega) * up + (
        torch.sin(eta_p * omega) / sin_omega
    ) * vp
    slerp = slerp / slerp.norm(dim=-1, keepdim=True).clamp_min(eps)

    near = (1.0 - eta_p) * up + eta_p * vp
    near = near / near.norm(dim=-1, keepdim=True).clamp_min(eps)

    e = _perp_unit_2d(up, vp, eps=eps)
    arc = torch.cos(eta_p * math.pi) * up + torch.sin(eta_p * math.pi) * e
    arc = arc / arc.norm(dim=-1, keepdim=True).clamp_min(eps)

    out = torch.where(c > 1.0 - eps_omega, near, slerp)
    out = torch.where(c < -1.0 + eps_omega, arc, out)
    return _from_pairs(out * radius)


guarded_slerp = torus_interp
slerp = torus_interp
geodesic_damping = torus_interp


def torus_residual(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean-over-pairs direction chordal residual, then max over tokens → [B], r ∈ [0, 2].

    Uses unit-pair directions so ambient radius does not change the τ scale.
    """
    zp = _as_pairs(z)
    ztp = _as_pairs(z_tilde)
    eps = 1e-7
    zp = zp / zp.norm(dim=-1, keepdim=True).clamp_min(eps)
    ztp = ztp / ztp.norm(dim=-1, keepdim=True).clamp_min(eps)
    cos = (zp * ztp).sum(dim=-1).clamp(-1.0, 1.0)
    per_tok = (1.0 - cos).mean(dim=-1)
    if token_mask is None:
        return per_tok.amax(dim=-1)
    neg_inf = torch.finfo(per_tok.dtype).min
    masked = per_tok.masked_fill(~token_mask, neg_inf)
    out = masked.amax(dim=-1)
    has_valid = token_mask.any(dim=-1)
    return torch.where(has_valid, out, torch.zeros_like(out))


chordal_residual = torus_residual
