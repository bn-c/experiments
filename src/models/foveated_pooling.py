"""
Space-Variant Foveated Pooling (SVFP) module.

Implements biologically-inspired cortical magnification resampling via PyTorch grid_sample.
Preserves high foveal visual acuity in central vision while progressively compressing
peripheral visual fields with smooth non-linear magnification.
Supports dynamic gaze fixation centers (for saccadic jitter) and aspect-ratio-preserving
sampling grids (56x56 for 1:1, 42x75 for 16:9).
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpaceVariantFoveatedPool(nn.Module):
    """
    Space-Variant Foveated Pooling (SVFP).
    
    Transforms feature maps from arbitrary spatial resolutions (e.g. 112x112, 540x960)
    to locked grid budgets (e.g. 56x56 = 3,136 points or 42x75 = 3,150 points)
    using cortical magnification mapping.
    
    Args:
        out_size: Optional fixed (H_g, W_g) target grid size. If None, dynamically chooses
                  (56, 56) for square aspect ratios and (42, 75) for widescreen (~16:9).
        r_fovea: Relative foveal boundary radius relative to frame height (clamped [0.05, 0.30]).
        alpha_fovea: Magnification scale factor in central fovea (< 1.0 concentrates sampling density).
        gamma: Non-linear peripheral compression exponent (>= 1.0).
        align_corners: Grid sample align_corners flag.
    """
    def __init__(
        self,
        out_size: Optional[Tuple[int, int]] = None,
        r_fovea: float = 0.15,
        alpha_fovea: float = 0.5,
        gamma: float = 1.5,
        align_corners: bool = True
    ):
        super().__init__()
        self.out_size = out_size
        self.r_fovea = float(max(0.05, min(0.30, r_fovea)))
        self.alpha_fovea = float(max(0.1, min(1.0, alpha_fovea)))
        self.gamma = float(max(1.0, min(3.0, gamma)))
        self.align_corners = align_corners

    def determine_grid_size(self, in_h: int, in_w: int) -> Tuple[int, int]:
        """Selects target grid size preserving aspect ratio while locking pixel budget."""
        if self.out_size is not None:
            return self.out_size
        aspect_ratio = in_w / in_h
        if aspect_ratio > 1.4:
            # Widescreen input (e.g., 16:9 ~ 1.778): 42 x 75 = 3,150 points
            return (42, 75)
        else:
            # Square / standard input: 56 x 56 = 3,136 points
            return (56, 56)

    def compute_sampling_grid(
        self,
        batch_size: int,
        out_h: int,
        out_w: int,
        gaze_center: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Generates normalized sampling grid [-1, 1] x [-1, 1] with radial cortical magnification.
        """
        # Linear normalized meshgrid for target foveated sensor
        u_lin = torch.linspace(-1.0, 1.0, out_w, device=device, dtype=dtype)
        v_lin = torch.linspace(-1.0, 1.0, out_h, device=device, dtype=dtype)
        v_grid, u_grid = torch.meshgrid(v_lin, u_lin, indexing="ij")  # (out_h, out_w)

        # Expand for batch
        u_grid = u_grid.unsqueeze(0).expand(batch_size, -1, -1)  # (B, out_h, out_w)
        v_grid = v_grid.unsqueeze(0).expand(batch_size, -1, -1)

        # Gaze fixation center handling
        if gaze_center is None:
            x_f = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
            y_f = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)
        else:
            if gaze_center.dim() == 1:
                gaze_center = gaze_center.unsqueeze(0)
            x_f = gaze_center[:, 0].view(-1, 1, 1).to(device=device, dtype=dtype)
            y_f = gaze_center[:, 1].view(-1, 1, 1).to(device=device, dtype=dtype)

        # Aspect ratio correction for radial distance
        aspect = float(out_w) / float(out_h)
        u_diff = (u_grid - x_f) * aspect
        v_diff = v_grid - y_f

        r_g = torch.sqrt(u_diff ** 2 + v_diff ** 2 + 1e-8)
        phi = torch.atan2(v_diff, u_diff)

        # Maximum possible radius on sensor
        r_max = math.sqrt(aspect ** 2 + 1.0)
        r0 = self.r_fovea * r_max
        r0 = max(1e-4, min(r_max * 0.95, r0))

        # Compute smooth cortical magnification mapping
        # Fovea (r_g <= r0): r_s = r_g * alpha_fovea
        # Periphery (r_g > r0): r_s = r0 * alpha_fovea + beta * (r_g - r0)^gamma
        # Boundary condition: at r_g = r_max, r_s = r_max
        beta = (r_max - r0 * self.alpha_fovea) / ((r_max - r0) ** self.gamma + 1e-8)

        fovea_mask = (r_g <= r0).to(dtype=dtype)
        r_fovea_sampled = r_g * self.alpha_fovea
        r_periph_sampled = r0 * self.alpha_fovea + beta * torch.pow(torch.clamp(r_g - r0, min=0.0), self.gamma)

        r_s = fovea_mask * r_fovea_sampled + (1.0 - fovea_mask) * r_periph_sampled

        # Back-project to normalized coordinates [-1, 1]
        x_s = x_f + (r_s * torch.cos(phi)) / aspect
        y_s = y_f + (r_s * torch.sin(phi))

        x_s = torch.clamp(x_s, -1.0, 1.0)
        y_s = torch.clamp(y_s, -1.0, 1.0)

        grid = torch.stack([x_s, y_s], dim=-1)  # (B, out_h, out_w, 2)
        return grid

    def forward(
        self,
        x: torch.Tensor,
        gaze_center: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass resampling input feature map to foveated grid.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            gaze_center: Optional (B, 2) tensor containing normalized (x, y) fixation coordinates in [-1, 1].
            
        Returns:
            Foveated feature map of shape (B, C, out_h, out_w).
        """
        B, C, H, W = x.shape
        out_h, out_w = self.determine_grid_size(H, W)

        grid = self.compute_sampling_grid(
            batch_size=B,
            out_h=out_h,
            out_w=out_w,
            gaze_center=gaze_center,
            device=x.device,
            dtype=x.dtype
        )

        return F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=self.align_corners
        )
