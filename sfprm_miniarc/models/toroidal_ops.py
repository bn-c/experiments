"""Product-of-circles torus geometry in R^d (d even).

Latents are d/2 interleaved (cos, sin) pairs. Projection renorms each pair to
the unit circle. Geodesic interp is per-pair 2D guarded Slerp. Halt residual is
mean-over-pairs chordal then max over tokens so r ∈ [0, 2] (same scale as
sphere chordal → keep τ=0.01 without retuning).

Not paper-faithful SFPRM — torus A/B experiment only.
"""

from __future__ import annotations

import torch


def _as_pairs(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"torus ambient dim must be even, got {d}")
    return x.reshape(*x.shape[:-1], d // 2, 2)


def project_to_torus(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Per-pair L2 renorm onto (S^1)^{d/2} (interleaved pairs on last dim)."""
    pairs = _as_pairs(x)
    norms = pairs.norm(dim=-1, keepdim=True).clamp_min(eps)
    return (pairs / norms).reshape(x.shape)


def nlerp(u: torch.Tensor, v: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Normalized lerp, then retract to the product torus."""
    if not torch.is_tensor(eta):
        eta = torch.as_tensor(eta, device=u.device, dtype=u.dtype)
    while eta.ndim < u.ndim:
        eta = eta.unsqueeze(-1)
    return project_to_torus((1.0 - eta) * u + eta * v)


def _perp_unit_2d(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit vector in R^2 span{u,v} orthogonal to u (pairs: [..., 2])."""
    proj = v - (u * v).sum(dim=-1, keepdim=True) * u
    norm = proj.norm(dim=-1, keepdim=True)
    ok = norm > eps
    fallback = torch.stack((-u[..., 1], u[..., 0]), dim=-1)
    fallback = fallback - (u * fallback).sum(dim=-1, keepdim=True) * u
    fallback = fallback / fallback.norm(dim=-1, keepdim=True).clamp_min(eps)
    return torch.where(ok, proj / norm.clamp_min(eps), fallback)


def torus_interp(
    u: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    *,
    eps_omega: float = 1e-4,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Per-pair guarded circular interpolation (2D Slerp on each circle).

    Same branch structure as sphere guarded Slerp, applied independently per pair.
    """
    if not torch.is_tensor(eta):
        eta = torch.as_tensor(eta, device=u.device, dtype=u.dtype)
    eta_b = eta
    while eta_b.ndim < u.ndim:
        eta_b = eta_b.unsqueeze(-1)

    up = _as_pairs(u)
    vp = _as_pairs(v)
    # Broadcast eta onto pair axis: [..., n_pairs, 1]
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

    near = ((1.0 - eta_p) * up + eta_p * vp)
    near = near / near.norm(dim=-1, keepdim=True).clamp_min(eps)

    e = _perp_unit_2d(up, vp, eps=eps)
    arc = torch.cos(eta_p * omega) * up + torch.sin(eta_p * omega) * e
    arc = arc / arc.norm(dim=-1, keepdim=True).clamp_min(eps)

    out = torch.where(c > 1.0 - eps_omega, near, slerp)
    out = torch.where(c < -1.0 + eps_omega, arc, out)
    return out.reshape(u.shape)


# Aliases matching sphere call-site names (torus fork only).
guarded_slerp = torus_interp
slerp = torus_interp
geodesic_damping = torus_interp


def torus_residual(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mean-over-pairs chordal residual, then max over tokens → [B], r ∈ [0, 2].

    r = max_t (2/d) Σ_i (1 − ⟨u_{t,i}, ũ_{t,i}⟩) = max_t mean_i (1 − cos θ_{t,i}).
    """
    zp = _as_pairs(z)
    ztp = _as_pairs(z_tilde)
    cos = (zp * ztp).sum(dim=-1).clamp(-1.0, 1.0)  # [B, T, n_pairs]
    per_tok = (1.0 - cos).mean(dim=-1)  # [B, T] ∈ [0, 2]
    if token_mask is None:
        return per_tok.amax(dim=-1)
    neg_inf = torch.finfo(per_tok.dtype).min
    masked = per_tok.masked_fill(~token_mask, neg_inf)
    out = masked.amax(dim=-1)
    has_valid = token_mask.any(dim=-1)
    return torch.where(has_valid, out, torch.zeros_like(out))


# Name used by the torus model fork (mirrors chordal_residual call sites).
chordal_residual = torus_residual
