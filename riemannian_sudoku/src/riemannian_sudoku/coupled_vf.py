"""Shared (weight-tied) bipartite vector fields: S→T (ω) and T→S (u)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.manifold_embed import D, M, N_DIGITS, PE_DIM, clifford_embed


def _mlp(in_dim: int, hidden: int, out_dim: int, n_mid: int) -> nn.Sequential:
    """``Linear→SiLU→(Linear→SiLU)×n_mid→Linear``."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(n_mid):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class CoupledVectorField(nn.Module):
    """
    Leapfrog pair: house ω from cell aggregates + Clifford ``y_emb``, then cell
    ambient u from FiLM(cell+PE, house embeds) → mid×2 → out.

    A2 P2 widths ~681k: ``H_Y=320`` (1 mid), ``H_X=432`` (FiLM + 2 mid);
    ``M=16``, ``in_y=178``.
    """

    def __init__(
        self,
        d: int = D,
        m: int = M,
        h_y: int = 320,
        h_x: int = 432,
        pe_dim: int = PE_DIM,
    ) -> None:
        super().__init__()
        self.d = d
        self.m = m
        self.pe_dim = pe_dim
        emb_m = 2 * m  # Clifford cat(cos,sin)
        # mean+std (2d) + mean q + mean q² (2·9) + y_emb (2m)
        in_y = 2 * d + 2 * N_DIGITS + emb_m
        self.vf_y = _mlp(in_y, h_y, m, n_mid=1)
        # FiLM first block: cell(+PE) + house shift/gate → H_X; then mid×2 + out.
        self.x_in_cell = nn.Linear(d + pe_dim, h_x)
        self.x_in_house_shift = nn.Linear(3 * emb_m, h_x)
        self.x_in_house_gate = nn.Linear(3 * emb_m, h_x)
        self.vf_x_body = nn.Sequential(
            nn.Linear(h_x, h_x),
            nn.SiLU(),
            nn.Linear(h_x, h_x),
            nn.SiLU(),
            nn.Linear(h_x, d),
        )

    def house_features(
        self,
        x: Tensor,
        poles: Tensor,
        house_cell_idx: Tensor,
    ) -> Tensor:
        """
        Per-house ``(B, 27, 2D+18)``: mean+std of cell ``x`` + claim means.

        ``q = relu(x_{:9} P_{:9}^T)`` (unnormalized; near-``e_8`` blanks stay ``q≈0``);
        both ``mean(q)`` and ``mean(q²)`` over the 9 cells (Option A mean-scale).
        ``std = (var + 1e-6).sqrt()``. Static gather via ``HOUSE_CELL_IDX``.
        """
        # x_cells: (B, 27, 9, D)
        x_h = x[:, house_cell_idx]
        mean = x_h.mean(dim=2)
        var = x_h.var(dim=2, unbiased=False)
        std = (var + 1e-6).sqrt()
        p9 = poles[:, :N_DIGITS]  # (9, 9)
        q = F.relu(x[..., :N_DIGITS] @ p9.T)  # (B, 81, 9)
        q_h = q[:, house_cell_idx]  # (B, 27, 9, 9)
        q_mean = q_h.mean(dim=2)
        q_sq_mean = (q_h * q_h).mean(dim=2)
        return torch.cat([mean, std, q_mean, q_sq_mean], dim=-1)

    def omega(
        self,
        x: Tensor,
        y: Tensor,
        poles: Tensor,
        house_cell_idx: Tensor,
    ) -> Tensor:
        """House angular velocity ``(B, 27, M)`` from ``[house_feats(x), y_emb]``."""
        feat = self.house_features(x, poles, house_cell_idx)
        y_emb = clifford_embed(y)
        return self.vf_y(torch.cat([feat, y_emb], dim=-1))

    def ambient_u(
        self,
        x: Tensor,
        y: Tensor,
        pe: Tensor,
        cell_house_idx: Tensor,
    ) -> Tensor:
        """
        Cell ambient velocity ``(B, 81, D)`` via FiLM(cell+PE, house embeds).

        Clifford house embeds gathered with static ``CELL_HOUSE_IDX[81, 3]``.
        """
        y_emb = clifford_embed(y)  # (B, 27, 2M)
        emb_m = 2 * self.m
        house_in = y_emb[:, cell_house_idx].reshape(x.shape[0], 81, 3 * emb_m)
        pe_b = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
        cell_in = torch.cat([x, pe_b], dim=-1)  # (B, 81, D+PE_DIM)
        base = F.silu(self.x_in_cell(cell_in) + self.x_in_house_shift(house_in))
        gate = 1.0 + torch.tanh(self.x_in_house_gate(house_in))
        return self.vf_x_body(base * gate)
