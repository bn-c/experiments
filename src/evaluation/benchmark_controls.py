"""
Scientific Control Benchmarks Module.

Provides the honest Information-Matched Uniform Downsampling Baseline.
Ensures standard ResNet-18 takes a 56x56 uniformly downsampled input and feeds
Layer 1 with identical (B, 64, 56, 56) tensor dimensions (bypassing conv1 + maxpool),
eliminating the 2x2 Layer 4 collapse and establishing publication-grade fairness.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class UniformDownsampledBaseline(nn.Module):
    """
    Honest Information-Matched Control Baseline.
    
    Standard ResNet-18 taking a 56x56 uniformly downsampled image.
    Uses a standard 3x3 conv (stride 1) + BN + ReLU to map from 3 to 64 channels
    and bypasses 7x7 conv1 and maxpool, so Layer 1 directly receives (B, 64, 56, 56).
    Layer 4 maintains the full (B, 512, 7, 7) spatial resolution matching the bio-pipeline.
    
    Args:
        num_classes: Number of classification classes (default: 100).
        pretrained_backbone: Whether to initialize ResNet layers from pre-trained weights.
    """
    def __init__(self, num_classes: int = 100, pretrained_backbone: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        try:
            base = models.resnet18(weights=weights)
        except Exception:
            base = models.resnet18(weights=None)

        # Stride-1 initial adapter: (B, 3, 56, 56) -> (B, 64, 56, 56)
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer1 = base.layer1  # (B, 64, 56, 56)
        self.layer2 = base.layer2  # (B, 128, 28, 28)
        self.layer3 = base.layer3  # (B, 256, 14, 14)
        self.layer4 = base.layer4  # (B, 512, 7, 7)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Uniformly downsampled input tensor of shape (B, 3, 56, 56).
            
        Returns:
            logits: Output classification logits (B, num_classes).
        """
        # If input is full size, downsample uniformly to match active pixel budget (56x56)
        if x.shape[-2:] != (56, 56):
            x = nn.functional.interpolate(x, size=(56, 56), mode="bilinear", align_corners=True)

        out = self.entry(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        logits = self.fc(out)
        return logits
