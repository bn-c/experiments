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

from .basis import hermite_gaussian_basis, interpolate_order


class FracSRFConv2d(nn.Module):
    """Drop-in replacement for a k x k nn.Conv2d using FracSRF structured filters."""

    # Clip fractional orders to this range: bounds the Hermite recursion depth
    # (and thus kernel size / compute) and prevents the order drift toward
    # negative values noted in the paper's Discussion (Sec. 5).
    MAX_ORDER = 10

    # Clip sigma to this range. The kernel footprint is ~(4*sigma+1)^2 and is built
    # from only 3 scalars per (out, in) pair, so an unclamped learned sigma can turn
    # a cheap structured layer into an enormous dense convolution (sigma=30 gives a
    # 123x123 kernel) without any parameter-count signal that this is happening.
    MIN_SIGMA = 0.25
    MAX_SIGMA = 8.0

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
            torch.tensor(math.log(min(max(sigma_init, self.MIN_SIGMA), self.MAX_SIGMA)), dtype=torch.float32),
            requires_grad=learn_sigma,
        )

        lo, hi = order_init_range
        self.nu_x = nn.Parameter(torch.empty(out_channels, in_channels).uniform_(lo, hi))
        self.nu_y = nn.Parameter(torch.empty(out_channels, in_channels).uniform_(lo, hi))

        self.alpha = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self._init_alpha()

    def _init_alpha(self) -> None:
        """
        Scale the per-kernel amplitude so a freshly built layer is variance-preserving
        like a Kaiming-initialised dense conv.

        alpha ~ N(0, 1/sqrt(fan_in)) alone is not enough here: the structured kernel is
        a Gaussian-derivative basis whose L2 norm is far below 1, so the same alpha
        yields activations roughly 7x smaller than a dense conv's. The basis energy is
        measured once at init (it does not depend on alpha) and folded into the scale.
        """
        with torch.no_grad():
            self.alpha.fill_(1.0)                                 # unit amplitude
            kernel, _ = self._build_kernel()                      # basis energy per kernel
            basis_energy = kernel.pow(2).sum(dim=(-1, -2)).clamp_min(1e-12)   # (out, in)
            # Kaiming-style fan-in gain, divided by the basis energy.
            std = torch.sqrt((2.0 / max(self.in_channels, 1)) / basis_energy)
            self.alpha.copy_(torch.randn_like(self.alpha) * std)

    @property
    def sigma(self) -> torch.Tensor:
        """Effective sigma, clipped to [MIN_SIGMA, MAX_SIGMA] (see the class docstring)."""
        return self.log_sigma.clamp(math.log(self.MIN_SIGMA), math.log(self.MAX_SIGMA)).exp()

    def clip_orders_(self) -> None:
        """In-place clamp of nu_x/nu_y into [0, MAX_ORDER]; call after optimizer.step()."""
        with torch.no_grad():
            self.nu_x.clamp_(0.0, self.MAX_ORDER)
            self.nu_y.clamp_(0.0, self.MAX_ORDER)

    def clip_sigma_(self) -> None:
        """
        In-place clamp of log_sigma into [log(MIN_SIGMA), log(MAX_SIGMA)].

        `sigma` already clips on read, so the forward pass is safe either way; clamping
        the parameter keeps a runaway sigma from sitting in the checkpoint forever and
        from soaking up the whole learning rate budget.
        """
        with torch.no_grad():
            self.log_sigma.clamp_(math.log(self.MIN_SIGMA), math.log(self.MAX_SIGMA))

    def clip_params_(self) -> None:
        """Convenience: clip orders and sigma. Call after each optimizer.step()."""
        self.clip_orders_()
        self.clip_sigma_()

    @staticmethod
    def _hermite_gaussian_basis(
        coords: torch.Tensor, sigma: torch.Tensor, max_order: int
    ) -> torch.Tensor:
        """Stacked G_0..G_max_order (Eq. 3), shape (max_order + 1, K). See `fracsrf.basis`."""
        return hermite_gaussian_basis(coords, sigma, max_order)

    @staticmethod
    def _interpolate_order(basis: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between the two nearest integer-order bases (Eq. 8)."""
        return interpolate_order(basis, nu)

    def _build_kernel(self) -> Tuple[torch.Tensor, int]:
        sigma = self.sigma
        # The spatial extent and the Hermite order are integers, so they are read back
        # from the parameters. Both are taken in ONE host sync (a stacked .tolist())
        # rather than two separate .item() calls: this runs once per layer per forward
        # pass, i.e. 31 layers x 2 syncs serialising every step on the GPU previously.
        # The order must be the max over BOTH axes: _interpolate_order clips nu to
        # max_order - 1e-4, so deriving it from nu_x alone would silently truncate nu_y.
        sigma_v, order_x, order_y = torch.stack([
            sigma.detach().clamp_max(float(self.MAX_SIGMA)),
            self.nu_x.detach().max().clamp_max(float(self.MAX_ORDER)),
            self.nu_y.detach().max().clamp_max(float(self.MAX_ORDER)),
        ]).tolist()

        # Spatial extent ~2*sigma around the center, per the paper's setup.
        radius = max(1, int(math.ceil(2.0 * sigma_v)))
        max_order = max(1, min(self.MAX_ORDER, int(math.ceil(max(order_x, order_y))) + 1))

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
