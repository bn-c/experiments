"""
Grouped Spatial-Aware Adapter module.

Provides a parameter-efficient, bio-physically motivated bridge between
the 512-channel V1 quadrature representation and downstream ResNet Stage 1.
Pairs simple and complex cell channels locally via grouped 3x3 convolution (groups=256),
applies GroupNorm and non-linear activation, and projects to 64 channels.
"""

import torch
import torch.nn as nn


class GroupedAdapter(nn.Module):
    """
    Grouped Spatial-Aware Adapter.
    
    Channels in the input tensor are expected to be organized in quadrature pairs:
    [S_1, C_1, S_2, C_2, ..., S_256, C_256].
    
    A grouped convolution with groups = in_channels // 2 (256 groups) pairs each
    simple cell with its complex cell counterpart across a local 3x3 spatial receptive field.
    This is followed by GroupNorm(32, 512), GELU activation, and 1x1 pointwise compression
    to 64 channels, matching ResNet Layer 1 input expectations.
    
    Args:
        in_channels: Number of input V1 channels (default: 512).
        out_channels: Number of output channels for downstream backbone (default: 64).
        group_norm_groups: Number of groups for GroupNorm (default: 32).
    """
    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 64,
        group_norm_groups: int = 32
    ):
        super().__init__()
        assert in_channels % 2 == 0, f"in_channels ({in_channels}) must be divisible by 2 for quadrature pairing."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_quadrature_groups = in_channels // 2  # 256 for 512 channels

        # 3x3 grouped convolution: groups=256 ensures pairwise channel coupling
        self.grouped_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=self.num_quadrature_groups,
            bias=False
        )

        # Normalization and non-linear activation
        self.norm = nn.GroupNorm(num_groups=group_norm_groups, num_channels=in_channels)
        self.act = nn.GELU()

        # 1x1 Pointwise convolution for cross-channel mixing and compression to backbone dimension
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.out_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input feature tensor of shape (B, 512, H_g, W_g).
            
        Returns:
            Projected feature tensor of shape (B, 64, H_g, W_g).
        """
        out = self.grouped_conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.pointwise_conv(out)
        out = self.out_norm(out)
        return out
