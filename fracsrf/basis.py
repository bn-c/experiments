"""
Separable Gaussian-derivative basis, shared by the Cartesian and manifold filters.

Extracted from `fracsrf/layers.py` so the 1-D basis math lives in exactly one place.
`FracSRFConv2d` evaluates it on an integer Cartesian grid; `fracsrf.manifold`
evaluates it at arbitrary -- possibly scattered -- coordinates on a FOVI sensor
manifold. Neither caller is known here.

Basis convention (Saldanha et al., BMVC 2021, Eqs. 3 and 8)::

    G_nu(x; sigma) = d^nu/dx^nu N(x; 0, sigma)

built from the Hermite recursion (Martens 1990), with non-integer orders obtained by
linear interpolation between the two nearest integer orders so that `nu` stays
differentiable. The amplitude `alpha` is deliberately *not* part of the basis: callers
fold it in, because it means different things (a per-kernel scalar in `layers.py`, a
per-(out, in) scalar pair in `manifold.py`).
"""

import math

import torch


def hermite_gaussian_basis(
    coords: torch.Tensor, sigma: torch.Tensor, max_order: int
) -> torch.Tensor:
    """
    Stacked ``G_0 .. G_max_order`` evaluated at `coords`, shape ``(max_order + 1, *coords.shape)``.

    Args:
        coords: Arbitrary sample positions. Nothing is assumed about their spacing, so
            this is equally valid for an integer grid and for scattered positions in a
            normalised receptive-field frame -- which is what makes the manifold
            mapping work without resampling.
        sigma: Gaussian scale. Differentiable, so gradients reach a learned sigma.
        max_order: Highest integer order required.

    Returns:
        Tensor of shape ``(max_order + 1, *coords.shape)``.
    """
    xi = coords / (sigma * math.sqrt(2.0))
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2)) / (sigma * math.sqrt(2.0 * math.pi))

    h = [torch.ones_like(xi), 2.0 * xi]
    for i in range(2, max_order + 1):
        h.append(2.0 * xi * h[-1] - 2.0 * (i - 1) * h[-2])

    scale = -1.0 / (sigma * math.sqrt(2.0))
    basis, factor = [], scale.new_ones(())
    for i in range(max_order + 1):
        basis.append(factor * h[i] * gauss)
        factor = factor * scale
    return torch.stack(basis, dim=0)


def order_coefficients(nu: torch.Tensor, max_order: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a (fractional) order into the two nearest integer orders and a mixing weight.

    `interpolate_order` evaluates ``(1 - frac) * G_lo + frac * G_hi``; this function
    exposes the same decomposition as data so callers that must *not* materialise a
    tensor of interpolated kernels (e.g. one weight per neighbourhood) can bucket by
    ``(lo, hi)`` and fold ``frac`` into a coefficient instead.

    Args:
        nu: Fractional orders of any shape.
        max_order: Highest available integer order.

    Returns:
        ``(lo, hi, frac)``, each with `nu`'s shape. ``hi == lo + 1`` except where `lo`
        is clipped to `max_order`.
    """
    nu = nu.clamp(0.0, max_order - 1e-4)
    lo = nu.floor().long()
    hi = (lo + 1).clamp(max=max_order)
    frac = nu - lo.to(nu.dtype)
    return lo, hi, frac


def interpolate_order(basis: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """
    Evaluate fractional-order derivatives from the stacked integer-order `basis`.

    Args:
        basis: ``(max_order + 1, *spatial)`` from `hermite_gaussian_basis`.
        nu: Fractional orders of shape ``(*channels)``.

    Returns:
        Tensor of shape ``(*channels, *spatial)``. `spatial` may have any rank, so this
        works for the 1-D Cartesian grid and for a set of scattered neighbourhood
        coordinates alike.
    """
    max_order = basis.shape[0] - 1
    lo, hi, frac = order_coefficients(nu, max_order)
    # One broadcast axis per spatial dimension of the basis.
    frac = frac.reshape(*frac.shape, *((1,) * (basis.ndim - 1)))
    return (1.0 - frac) * basis[lo] + frac * basis[hi]
