"""
FracSRF structured convolutional filter (Saldanha et al., BMVC 2021).

Each (in_channel, out_channel) kernel is approximated with a single weighted
Gaussian derivative of fractional order, rather than a full set of
independently-learned pixel weights:

    F(c, k, x, y; sigma) = alpha_{ck} * ( G_{nu_x_ck}(x, sigma) (x) G_{nu_y_ck}(y, sigma) )

where G_nu is the 1D Gaussian derivative of (possibly fractional) order nu,
computed via the Hermite-polynomial recursion (Martens 1990) and interpolated
linearly between the two nearest integer orders (Eq. 8 of the paper) so that
nu remains differentiable. The scale `sigma` is shared across all kernels of
a layer; `alpha`, `nu_x`, `nu_y` are learned per-kernel scalars, i.e. 3
parameters per kernel instead of k*k.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FracSRFConv2d(nn.Module):
    """Drop-in replacement for a k x k nn.Conv2d using FracSRF structured filters."""

    # Clip fractional orders to this range: bounds the Hermite recursion depth
    # (and thus kernel size / compute) and prevents the order drift toward
    # negative values noted in the paper's Discussion (Sec. 5).
    MAX_ORDER = 10

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        sigma_init: float = 1.0,
        order_init_range: Tuple[float, float] = (1.0, 3.0),
        learn_sigma: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # sigma > 0 is enforced by learning its log (avoids negative-sigma gradients).
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(sigma_init), dtype=torch.float32),
            requires_grad=learn_sigma,
        )

        lo, hi = order_init_range
        self.nu_x = nn.Parameter(torch.empty(out_channels, in_channels).uniform_(lo, hi))
        self.nu_y = nn.Parameter(torch.empty(out_channels, in_channels).uniform_(lo, hi))

        fan_in = in_channels
        self.alpha = nn.Parameter(
            torch.empty(out_channels, in_channels).normal_(0.0, 1.0 / math.sqrt(fan_in))
        )

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def clip_orders_(self) -> None:
        """In-place clamp of nu_x/nu_y into [0, MAX_ORDER]; call after optimizer.step()."""
        with torch.no_grad():
            self.nu_x.clamp_(0.0, self.MAX_ORDER)
            self.nu_y.clamp_(0.0, self.MAX_ORDER)

    def _hermite_gaussian_basis(
        self, coords: torch.Tensor, sigma: torch.Tensor, max_order: int
    ) -> torch.Tensor:
        """Stacked G_0..G_max_order (Eq. 3), shape (max_order + 1, K)."""
        xi = coords / (sigma * math.sqrt(2.0))
        gauss = torch.exp(-coords ** 2 / (2.0 * sigma ** 2)) / (sigma * math.sqrt(2.0 * math.pi))

        h = [torch.ones_like(xi), 2.0 * xi]
        for i in range(2, max_order + 1):
            h.append(2.0 * xi * h[-1] - 2.0 * (i - 1) * h[-2])

        scale = -1.0 / (sigma * math.sqrt(2.0))
        basis, factor = [], scale.new_ones(())
        for i in range(max_order + 1):
            basis.append(factor * h[i] * gauss)
            factor = factor * scale
        return torch.stack(basis, dim=0)

    @staticmethod
    def _interpolate_order(basis: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between the two nearest integer-order bases (Eq. 8)."""
        max_order = basis.shape[0] - 1
        nu = nu.clamp(0.0, max_order - 1e-4)
        lo_idx = nu.floor().long()
        hi_idx = (lo_idx + 1).clamp(max=max_order)
        frac = (nu - lo_idx.to(nu.dtype)).unsqueeze(-1)
        return (1.0 - frac) * basis[lo_idx] + frac * basis[hi_idx]

    def _build_kernel(self) -> Tuple[torch.Tensor, int]:
        sigma = self.sigma
        # Spatial extent ~2*sigma around the center, per the paper's setup.
        radius = max(1, int(math.ceil(2.0 * sigma.item())))
        max_order = min(
            self.MAX_ORDER,
            int(math.ceil(max(self.nu_x.max().item(), self.nu_y.max().item()))) + 1,
        )
        max_order = max(max_order, 1)

        coords = torch.arange(-radius, radius + 1, device=sigma.device, dtype=sigma.dtype)
        basis = self._hermite_gaussian_basis(coords, sigma, max_order)  # (max_order+1, K)

        gx = self._interpolate_order(basis, self.nu_x)  # (out, in, K)
        gy = self._interpolate_order(basis, self.nu_y)  # (out, in, K)

        kernel = gy.unsqueeze(-1) * gx.unsqueeze(-2)  # (out, in, K, K)
        kernel = kernel * self.alpha.view(self.out_channels, self.in_channels, 1, 1)
        return kernel, radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, radius = self._build_kernel()
        return F.conv2d(x, weight, bias=self.bias, stride=self.stride, padding=radius)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"stride={self.stride}, sigma={self.sigma.item():.3f}"
        )
