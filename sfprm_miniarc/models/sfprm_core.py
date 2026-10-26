"""Weight-tied spherical fixed-point reasoner (SFPRM)."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.spherical_ops import geodesic_damping, project_to_sphere, slerp


class NormalizedTransformerBlock(nn.Module):
    """Single weight-tied block with QK-normalized self-attention and MLP."""

    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 1536):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, S, D = z.shape
        q = project_to_sphere(self.q_proj(z)).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = project_to_sphere(self.k_proj(z)).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scale = self.head_dim**0.5
        attn = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, S, D)
        z_mid = project_to_sphere(z + self.out_proj(attn_out))
        return project_to_sphere(z_mid + self.mlp(z_mid))


class SFPRM(nn.Module):
    """~6.8M-parameter weight-tied nGPT reasoner on the unit sphere."""

    def __init__(self, vocab_size: int = 11, d_model: int = 512, max_seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.core_block = NormalizedTransformerBlock(d_model=d_model, n_heads=8, d_ff=1536)
        self.raw_lambda = nn.Parameter(torch.tensor(0.0))
        self.raw_eta = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Linear(d_model, 10, bias=False)

    def get_input_embedding(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        pos = torch.arange(seq_len, device=x.device)
        return project_to_sphere(self.token_emb(x) + self.pos_emb(pos))

    def step(self, z: torch.Tensor, z_input: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.raw_lambda)
        eta = torch.sigmoid(self.raw_eta)
        z_injected = slerp(z, z_input, lam)
        z_prop = self.core_block(z_injected)
        return geodesic_damping(z, z_prop, eta)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
