"""
Saccadic Jittering module.

Provides saliency-guided fixation perturbation around object centroids.
Prevents evolutionary search from overfitting to static center-camera biases
without throwing salient objects into the heavily blurred peripheral visual field.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def estimate_image_centroid(images: torch.Tensor) -> torch.Tensor:
    """
    Computes coarse spatial saliency centroid for each image in batch via edge/gradient energy.
    
    Args:
        images: Input tensor of shape (B, 3, H, W) normalized to [0, 1] or standardized.
        
    Returns:
        centroids: (B, 2) tensor of normalized coordinates (x, y) in [-1, 1].
    """
    B, C, H, W = images.shape
    device = images.device

    # Convert to grayscale luminance
    luma = 0.2989 * images[:, 0:1] + 0.5870 * images[:, 1:2] + 0.1140 * images[:, 2:3]

    # Approximate spatial gradients (Sobel filters)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)

    gx = F.conv2d(luma, kx, padding=1)
    gy = F.conv2d(luma, ky, padding=1)
    energy = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)  # (B, 1, H, W)

    # Apply spatial softmax to obtain probability density over pixels
    flat_energy = energy.view(B, -1)
    weights = F.softmax(flat_energy / 2.0, dim=-1).view(B, 1, H, W)

    # Normalized coordinate grids [-1, 1]
    u = torch.linspace(-1.0, 1.0, W, device=device).view(1, 1, 1, W)
    v = torch.linspace(-1.0, 1.0, H, device=device).view(1, 1, H, 1)

    cx = torch.sum(weights * u, dim=(1, 2, 3))
    cy = torch.sum(weights * v, dim=(1, 2, 3))

    return torch.stack([cx, cy], dim=-1)  # (B, 2)


class SaccadicJitterSampler:
    """
    Generates ocular fixation gaze coordinates for mini-batches.
    
    Args:
        sigma: Standard deviation of Gaussian perturbation around object centroid (default: 0.08).
        clip_bound: Maximum allowable normalized coordinate offset (default: 0.70).
    """
    def __init__(self, sigma: float = 0.08, clip_bound: float = 0.70):
        self.sigma = sigma
        self.clip_bound = clip_bound

    def sample_gaze_centers(
        self,
        images: torch.Tensor,
        use_saliency: bool = True
    ) -> torch.Tensor:
        """
        Samples gaze coordinates (B, 2) for the given batch.
        
        Args:
            images: Tensor of shape (B, 3, H, W).
            use_saliency: If True, centers jitter around saliency centroid; else around center (0, 0).
            
        Returns:
            gaze_centers: (B, 2) tensor in [-clip_bound, clip_bound].
        """
        B = images.shape[0]
        device = images.device

        if use_saliency:
            base_centroids = estimate_image_centroid(images)
        else:
            base_centroids = torch.zeros(B, 2, device=device)

        noise = torch.randn(B, 2, device=device) * self.sigma
        gaze = base_centroids + noise
        gaze = torch.clamp(gaze, -self.clip_bound, self.clip_bound)
        return gaze
