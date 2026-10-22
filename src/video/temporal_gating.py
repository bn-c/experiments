"""
Feature-Space Temporal Gating and State Caching Module.

Computes normalized inter-frame distance on deep representations (ResNet Layer 2)
to provide robustness against vehicle ego-motion on driving datasets like CamVid.
When delta falls below threshold tau_reuse, bypasses ResNet Stages 3 & 4 and reuses
the cached Layer 4 representation for high-FPS video throughput.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn

from ..models.pipeline import BioInspiredVisionPipeline


class TemporalFeatureGater:
    """
    Manages inter-frame state caching and delta gating on sequential video frames.
    
    Args:
        state_reuse_threshold: Relative Frobenius feature difference below which to reuse state.
        eps: Numerical stability epsilon.
    """
    def __init__(self, state_reuse_threshold: float = 0.10, eps: float = 1e-6):
        self.state_reuse_threshold = state_reuse_threshold
        self.eps = eps
        self.cached_l2: Optional[torch.Tensor] = None
        self.cached_l4: Optional[torch.Tensor] = None
        self.total_frames: int = 0
        self.bypassed_frames: int = 0

    def reset(self):
        """Resets cached representations between independent video clips."""
        self.cached_l2 = None
        self.cached_l4 = None
        self.total_frames = 0
        self.bypassed_frames = 0

    def compute_delta(self, feat_l2: torch.Tensor) -> float:
        """
        Computes normalized Frobenius distance between current and cached Layer 2 features.
        
        Delta = ||F_t - F_{t-1}||_F / (||F_t||_F + eps)
        """
        if self.cached_l2 is None:
            return float("inf")
        diff_norm = torch.linalg.norm(feat_l2 - self.cached_l2).item()
        base_norm = torch.linalg.norm(feat_l2).item() + self.eps
        return diff_norm / base_norm

    def process_frame(
        self,
        pipeline: BioInspiredVisionPipeline,
        frame: torch.Tensor,
        gaze_center: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, bool, float]:
        """
        Processes a single video frame with opportunistic temporal state reuse.
        
        Args:
            pipeline: BioInspiredVisionPipeline instance.
            frame: Input frame tensor (1, 3, H, W).
            gaze_center: Optional gaze fixation center.
            
        Returns:
            logits: Output classification logits (1, num_classes).
            bypassed: True if Stages 3/4 were skipped, False if full forward pass executed.
            delta: Computed normalized feature delta.
        """
        self.total_frames += 1

        # Step 1: Compute up to Layer 2
        x_v1 = pipeline.frontend(frame)
        x_fov = pipeline.foveated_pool(x_v1, gaze_center=gaze_center)
        x_adapt = pipeline.adapter(x_fov)
        x_l1 = pipeline.layer1(x_adapt)
        x_l2 = pipeline.layer2(x_l1)

        delta = self.compute_delta(x_l2)

        # Step 2: Check gating condition
        if delta < self.state_reuse_threshold and self.cached_l4 is not None:
            # Bypass Stages 3 & 4: reuse cached Layer 4 features
            self.bypassed_frames += 1
            pooled = pipeline.avgpool(self.cached_l4)
            flat = torch.flatten(pooled, 1)
            logits = pipeline.fc(flat)
            bypassed = True
        else:
            # Full forward pass through Stages 3 & 4
            x_l3 = pipeline.layer3(x_l2)
            x_l4 = pipeline.layer4(x_l3)
            # Update cache
            self.cached_l2 = x_l2.detach().clone()
            self.cached_l4 = x_l4.detach().clone()

            pooled = pipeline.avgpool(x_l4)
            flat = torch.flatten(pooled, 1)
            logits = pipeline.fc(flat)
            bypassed = False

        return logits, bypassed, delta

    @property
    def bypass_rate(self) -> float:
        """Percentage of frames where Stages 3 & 4 were bypassed."""
        return (self.bypassed_frames / max(1, self.total_frames)) * 100.0
