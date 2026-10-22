"""
BioFrontEnd module.

Implements the scale-invariant biologically-inspired front-end:
1. Light Adaptation (global mean subtraction & local gain control).
2. Subcortical Difference-of-Gaussians (Parvocellular & Magnocellular pathways).
3. Divisive Contrast Normalization.
4. VOneBlock Gabor Filter Bank (512 channels: 256 simple/complex quadrature pairs).
"""

from typing import Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_gabor_kernel(
    kernel_size: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0.0
) -> torch.Tensor:
    """Generates a 2D spatial Gabor filter kernel."""
    half_size = kernel_size // 2
    y, x = torch.meshgrid(
        torch.linspace(-half_size, half_size, kernel_size),
        torch.linspace(-half_size, half_size, kernel_size),
        indexing="ij"
    )

    # Coordinate rotation
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)

    # 2D Gaussian envelope
    envelope = torch.exp(-0.5 * (x_theta ** 2 + (gamma * y_theta) ** 2) / (sigma ** 2 + 1e-8))
    # Sinusoidal carrier
    carrier = torch.cos(2.0 * math.pi * x_theta / (lambd + 1e-8) + psi)

    kernel = envelope * carrier
    # Zero-mean normalization for even/odd balance
    kernel = kernel - kernel.mean()
    norm = torch.sqrt(torch.sum(kernel ** 2) + 1e-8)
    return kernel / norm


class SubcorticalDoG(nn.Module):
    """
    Subcortical Difference-of-Gaussians (DoG) receptive field filtering.
    Models retinal ganglion and LGN On/Off center-surround pathways:
    - Parvocellular (P) pathway: Fine spatial scale, sustained.
    - Magnocellular (M) pathway: Coarse spatial scale, high contrast sensitivity.
    """
    def __init__(
        self,
        in_channels: int = 3,
        r_rel: float = 0.015,
        ws_ratio: float = 0.8,
        kernel_size: int = 7
    ):
        super().__init__()
        self.in_channels = in_channels
        self.r_rel = float(max(0.005, min(0.030, r_rel)))
        self.ws_ratio = ws_ratio
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

        self.register_buffer("dog_weights", self._build_dog_filters())

    def _build_dog_filters(self) -> torch.Tensor:
        """Constructs fixed 6-channel DoG filters (3 Parvo + 3 Magno)."""
        K = self.kernel_size
        half = K // 2
        y, x = torch.meshgrid(
            torch.linspace(-half, half, K),
            torch.linspace(-half, half, K),
            indexing="ij"
        )
        r_sq = x ** 2 + y ** 2

        # P-pathway (finer scale)
        sigma_c_p = 1.0
        sigma_s_p = 3.0 * sigma_c_p
        g_c_p = torch.exp(-0.5 * r_sq / (sigma_c_p ** 2)) / (2.0 * math.pi * sigma_c_p ** 2)
        g_s_p = torch.exp(-0.5 * r_sq / (sigma_s_p ** 2)) / (2.0 * math.pi * sigma_s_p ** 2)
        dog_p = g_c_p - self.ws_ratio * g_s_p
        dog_p = (dog_p - dog_p.mean()) / (torch.sqrt(torch.sum(dog_p ** 2)) + 1e-8)

        # M-pathway (coarser scale)
        sigma_c_m = 2.0
        sigma_s_m = 3.0 * sigma_c_m
        g_c_m = torch.exp(-0.5 * r_sq / (sigma_c_m ** 2)) / (2.0 * math.pi * sigma_c_m ** 2)
        g_s_m = torch.exp(-0.5 * r_sq / (sigma_s_m ** 2)) / (2.0 * math.pi * sigma_s_m ** 2)
        dog_m = g_c_m - self.ws_ratio * g_s_m
        dog_m = (dog_m - dog_m.mean()) / (torch.sqrt(torch.sum(dog_m ** 2)) + 1e-8)

        # Build depthwise filters for 3 input channels -> 6 output channels
        # Out channels 0..2: P-pathway, Out channels 3..5: M-pathway
        weights = torch.zeros(6, self.in_channels, K, K)
        for c in range(self.in_channels):
            weights[c, c] = dog_p
            weights[c + 3, c] = dog_m
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.conv2d(x, self.dog_weights, padding=pad)


class DivisiveContrastNorm(nn.Module):
    """
    Local Divisive Contrast Normalization.
    Normalizes local neural activity by the pooled energy of neighboring responses.
    """
    def __init__(self, channels: int = 6, r_norm: float = 0.035, kernel_size: int = 9, sigma_0: float = 0.1):
        super().__init__()
        self.channels = channels
        self.r_norm = float(max(0.010, min(0.080, r_norm)))
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma_0 = sigma_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        # Average pooled local squared energy
        energy = F.avg_pool2d(x ** 2, kernel_size=self.kernel_size, stride=1, padding=pad)
        norm_factor = self.sigma_0 + torch.sqrt(energy + 1e-8)
        return x / norm_factor


class VOneBlock(nn.Module):
    """
    VOneBlock Gabor filter bank simulating primary visual cortex (V1).
    Organized into 256 quadrature pairs (512 channels total):
    - Simple cells: half-wave rectified cosine response.
    - Complex cells: quadrature energy sqrt(even^2 + odd^2).
    Channels are interleaved: [S_1, C_1, S_2, C_2, ..., S_256, C_256].
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_pairs: int = 256,
        kernel_size: int = 15,
        stride: int = 2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_pairs = num_pairs
        self.out_channels = num_pairs * 2  # 512 channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.stride = stride

        even_filters, odd_filters = self._build_gabor_bank()
        self.register_buffer("even_filters", even_filters)
        self.register_buffer("odd_filters", odd_filters)

    def _build_gabor_bank(self) -> Tuple[torch.Tensor, torch.Tensor]:
        K = self.kernel_size
        num_p = self.num_pairs
        even_weights = torch.zeros(num_p, self.in_channels, K, K)
        odd_weights = torch.zeros(num_p, self.in_channels, K, K)

        num_orientations = 8
        num_scales = num_p // num_orientations  # 32 frequency scales

        pair_idx = 0
        for s_idx in range(num_scales):
            lambd = 3.0 + s_idx * 0.4  # Spatial wavelength
            sigma = 0.5 * lambd
            for o_idx in range(num_orientations):
                theta = o_idx * (math.pi / num_orientations)
                even_k = generate_gabor_kernel(K, sigma=sigma, theta=theta, lambd=lambd, psi=0.0)
                odd_k = generate_gabor_kernel(K, sigma=sigma, theta=theta, lambd=lambd, psi=math.pi / 2.0)

                for c in range(self.in_channels):
                    even_weights[pair_idx, c] = even_k / math.sqrt(self.in_channels)
                    odd_weights[pair_idx, c] = odd_k / math.sqrt(self.in_channels)

                pair_idx += 1
                if pair_idx >= num_p:
                    break
            if pair_idx >= num_p:
                break

        return even_weights, odd_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        even_resp = F.conv2d(x, self.even_filters, stride=self.stride, padding=pad)
        odd_resp = F.conv2d(x, self.odd_filters, stride=self.stride, padding=pad)

        # Simple cells: ReLU rectified even response
        simple = F.relu(even_resp)
        # Complex cells: Quadrature energy sqrt(even^2 + odd^2)
        complex_cell = torch.sqrt(even_resp ** 2 + odd_resp ** 2 + 1e-8)

        # Interleave simple and complex channels: (B, 512, H_out, W_out)
        # Shape: simple (B, 256, H, W), complex (B, 256, H, W)
        B, num_p, H, W = simple.shape
        stacked = torch.stack([simple, complex_cell], dim=2)  # (B, 256, 2, H, W)
        interleaved = stacked.view(B, num_p * 2, H, W)        # (B, 512, H, W)
        return interleaved


class BioFrontEnd(nn.Module):
    """
    Unified Bio-Inspired Vision Front-End f_theta(I).
    Integrates light adaptation, subcortical DoG, divisive contrast norm, and VOneBlock.
    Outputs: (B, 512, H_out, W_out).
    """
    def __init__(
        self,
        r_rel: float = 0.015,
        r_norm: float = 0.035,
        dog_kernel_size: int = 7,
        vone_kernel_size: int = 15,
        vone_stride: int = 2
    ):
        super().__init__()
        self.subcortical = SubcorticalDoG(in_channels=3, r_rel=r_rel, kernel_size=dog_kernel_size)
        self.divisive_norm = DivisiveContrastNorm(channels=6, r_norm=r_norm)
        self.vone_block = VOneBlock(in_channels=6, num_pairs=256, kernel_size=vone_kernel_size, stride=vone_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Global Light Adaptation (mean subtraction & variance normalization)
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        x_adapt = (x - mean) / std

        # 2. Subcortical Center-Surround DoG (3 RGB -> 6 P/M channels)
        x_dog = self.subcortical(x_adapt)

        # 3. Divisive Contrast Normalization
        x_norm = self.divisive_norm(x_dog)

        # 4. VOneBlock Gabor Quadrature Filter Bank (6 -> 512 channels, stride 2)
        x_v1 = self.vone_block(x_norm)

        return x_v1
