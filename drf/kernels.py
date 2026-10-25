"""Analytic filter kernels and spatial operators used by the DRF front-end."""


import torch
import torch.nn.functional as F


def create_gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Creates a normalised 2D Gaussian kernel of shape (1, 1, K, K)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    half = kernel_size // 2
    coords = torch.linspace(-half, half, kernel_size, device=device, dtype=dtype or torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    r_sq = x ** 2 + y ** 2
    sigma_sq = max(float(sigma) ** 2, 1e-6)
    kernel = torch.exp(-0.5 * r_sq / sigma_sq)
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel.unsqueeze(0).unsqueeze(0)


def safe_spatial_shift(
    x: torch.Tensor,
    dx: int,
    dy: int,
    mode: str = "replicate"
) -> torch.Tensor:
    """
    Spatially shifts a tensor by (dx, dy) with replicate padding.
    Avoids circular wrap-around artifacts that torch.roll introduces at frame borders.

    Convention:
        dx > 0: content shifts right  (position x receives value from x - dx).
        dx < 0: content shifts left   (position x receives value from x + |dx|).
        dy > 0: content shifts down   (position y receives value from y - dy).
        dy < 0: content shifts up     (position y receives value from y + |dy|).
    """
    if dx == 0 and dy == 0:
        return x

    pad_l = max(0, dx)
    pad_r = max(0, -dx)
    pad_t = max(0, dy)
    pad_b = max(0, -dy)

    padded = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode=mode)
    h, w = x.shape[-2:]
    start_x = max(0, -dx)
    start_y = max(0, -dy)
    return padded[..., start_y:start_y + h, start_x:start_x + w]
