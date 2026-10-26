"""Two-layer MLP-Mixer backbone (token + channel mixing)."""

from __future__ import annotations

import torch
from torch import nn


class MLPMixerBlock(nn.Module):
    """One MLP-Mixer block: mix across sequence, then across channels."""

    def __init__(self, seq_len: int, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm_tokens = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * expansion),
            nn.GELU(),
            nn.Linear(seq_len * expansion, seq_len),
        )
        self.norm_channels = nn.LayerNorm(dim)
        hidden = dim * expansion
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token mixing then channel mixing with residuals. Shape: (B, L, D)."""
        t = self.norm_tokens(x).transpose(1, 2)
        x = x + self.token_mlp(t).transpose(1, 2)
        return x + self.channel_mlp(self.norm_channels(x))


class MLPMixerBackbone(nn.Module):
    """Stack of ``n_layers`` MLP-Mixer blocks."""

    def __init__(self, seq_len: int, dim: int, n_layers: int = 2, expansion: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MLPMixerBlock(seq_len, dim, expansion) for _ in range(n_layers)],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
