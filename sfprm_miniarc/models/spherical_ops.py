"""Spherical geometry helpers for nGPT-style dynamics on S^{d-1}.

Guarded Slerp matches arXiv:2610.25646 Appendix A.2 eq. (12).
Chordal residual matches eq. (9): max_t (1 − ⟨z_t, z̃_t⟩).
"""

from __future__ import annotations

import torch


def project_to_sphere(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Projects latent vectors onto the unit hypersphere (token-wise on last dim)."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def nlerp(u: torch.Tensor, v: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Normalized linear interpolation on the sphere."""
    if not torch.is_tensor(eta):
        eta = torch.as_tensor(eta, device=u.device, dtype=u.dtype)
    while eta.ndim < u.ndim:
        eta = eta.unsqueeze(-1)
    return project_to_sphere((1.0 - eta) * u + eta * v)


def _perp_unit(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit vector in span{u,v} orthogonal to u; stable fallback if nearly collinear."""
    # P_u v = v - ⟨u,v⟩ u
    proj = v - (u * v).sum(dim=-1, keepdim=True) * u
    norm = proj.norm(dim=-1, keepdim=True)
    ok = norm > eps
    # Fixed fallback: flip first two coords of u (or pad) to get a ⊥ direction.
    fallback = torch.zeros_like(u)
    fallback[..., 0] = -u[..., 1]
    fallback[..., 1] = u[..., 0]
    # If d=1 somehow, use ones; project off u.
    if u.shape[-1] == 1:
        fallback = torch.ones_like(u)
    fallback = fallback - (u * fallback).sum(dim=-1, keepdim=True) * u
    fallback = project_to_sphere(fallback)
    return torch.where(ok, proj / norm.clamp_min(eps), fallback)


def guarded_slerp(
    u: torch.Tensor,
    v: torch.Tensor,
    eta: torch.Tensor,
    *,
    eps_omega: float = 1e-4,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Guarded spherical linear interpolation (paper eq. 12).

    Branches on cosine c = ⟨u, v⟩:
      c > 1 − ε_Ω  → nlerp (near-coincident)
      c < −1 + ε_Ω → explicit arc through e ⊥ u (near-antipodal)
      else         → exact Slerp
    """
    if not torch.is_tensor(eta):
        eta = torch.as_tensor(eta, device=u.device, dtype=u.dtype)
    eta_b = eta
    while eta_b.ndim < u.ndim:
        eta_b = eta_b.unsqueeze(-1)

    c = (u * v).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(c)
    sin_omega = torch.sin(omega).clamp_min(eps)

    # Exact Slerp branch
    slerp = (torch.sin((1.0 - eta_b) * omega) / sin_omega) * u + (
        torch.sin(eta_b * omega) / sin_omega
    ) * v
    slerp = project_to_sphere(slerp)

    # Near-coincident → nlerp
    near = nlerp(u, v, eta)

    # Near-antipodal → Arc_η(u,v) = cos(ηΩ) u + sin(ηΩ) e(u,v)
    e = _perp_unit(u, v, eps=eps)
    # Ω ≈ π when c ≈ -1; use acos(c) already
    arc = torch.cos(eta_b * omega) * u + torch.sin(eta_b * omega) * e
    arc = project_to_sphere(arc)

    out = torch.where(c > 1.0 - eps_omega, near, slerp)
    out = torch.where(c < -1.0 + eps_omega, arc, out)
    return out


# Back-compat alias used by older call sites / tests
def slerp(
    u: torch.Tensor,
    v: torch.Tensor,
    lambda_param: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Guarded Slerp (preferred); ``eps`` kept for signature compatibility."""
    del eps
    return guarded_slerp(u, v, lambda_param)


def geodesic_damping(z_prev: torch.Tensor, z_prop: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Rotates z_prev toward proposal z_prop by fraction eta along S^{d-1}."""
    return guarded_slerp(z_prev, z_prop, eta)


def chordal_residual(
    z: torch.Tensor,
    z_tilde: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Per-sample worst-token chordal residual (paper eq. 9).

    r_c = max_t (1 − ⟨z_t, z̃_t⟩). Returns shape [B].

    When ``token_mask`` is provided ([B, T] bool, True = valid), PAD / packing
    positions are excluded so halt is not driven by pad tokens.
    """
    # ⟨z, z̃⟩ token-wise → [B, T]
    cos = (z * z_tilde).sum(dim=-1).clamp(-1.0, 1.0)
    per_tok = 1.0 - cos
    if token_mask is None:
        return per_tok.amax(dim=-1)
    # Masked amax: invalid positions → -inf so they never win the max
    neg_inf = torch.finfo(per_tok.dtype).min
    masked = per_tok.masked_fill(~token_mask, neg_inf)
    out = masked.amax(dim=-1)
    # Degenerate all-pad rows → 0 (already halted / ignore)
    has_valid = token_mask.any(dim=-1)
    return torch.where(has_valid, out, torch.zeros_like(out))
