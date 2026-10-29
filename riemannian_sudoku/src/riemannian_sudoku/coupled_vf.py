"""Shared (weight-tied) bipartite vector fields: S→T (ω) and T→S (u)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.manifold_embed import D, M, N_DIGITS, PE_DIM


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
    Leapfrog pair: house ω from cell aggregates, then cell ambient u from
    ``[x, y_r, y_c, y_b, pe]``.

    Widths target ~0.66M: ``H_Y=320`` (1 mid), ``H_X=448`` (2 mid).
    """

    def __init__(
        self,
        d: int = D,
        m: int = M,
        h_y: int = 320,
        h_x: int = 448,
        pe_dim: int = PE_DIM,
    ) -> None:
        super().__init__()
        self.d = d
        self.m = m
        self.pe_dim = pe_dim
        # mean+std of cell feats (2d) + sum q + sum q² (2·9)
        in_y = 2 * d + 2 * N_DIGITS
        in_x = d + 3 * m + pe_dim
        # X needs 2 mid Linear(H,H) to land ~0.66M with Y's 1 mid.
        self.vf_y = _mlp(in_y, h_y, m, n_mid=1)
        self.vf_x = _mlp(in_x, h_x, d, n_mid=2)

    def house_features(
        self,
        x: Tensor,
        poles: Tensor,
        house_cell_idx: Tensor,
    ) -> Tensor:
        """
        Per-house ``(B, 27, 2D+18)``: mean+std of cell ``x`` + claim sums.

        ``q = relu(x_{:9} P_{:9}^T)``; both ``∑q`` and ``∑q²`` over the 9 cells.
        Static gather via ``HOUSE_CELL_IDX`` (no dynamic bool index).
        """
        # x_cells: (B, 27, 9, D)
        x_h = x[:, house_cell_idx]
        mean = x_h.mean(dim=2)
        std = x_h.std(dim=2, unbiased=False)
        p9 = poles[:, :N_DIGITS]  # (9, 9)
        q = F.relu(x[..., :N_DIGITS] @ p9.T)  # (B, 81, 9)
        q_h = q[:, house_cell_idx]  # (B, 27, 9, 9)
        sum_q = q_h.sum(dim=2)
        sum_q2 = q_h.square().sum(dim=2)
        return torch.cat([mean, std, sum_q, sum_q2], dim=-1)

    def omega(
        self,
        x: Tensor,
        poles: Tensor,
        house_cell_idx: Tensor,
    ) -> Tensor:
        """House angular velocity ``(B, 27, M)``."""
        feat = self.house_features(x, poles, house_cell_idx)
        return self.vf_y(feat)

    def ambient_u(
        self,
        x: Tensor,
        y: Tensor,
        pe: Tensor,
        cell_house_idx: Tensor,
    ) -> Tensor:
        """
        Cell ambient velocity ``(B, 81, D)`` from ``[x, y_r, y_c, y_b, pe]``.

        House angles gathered with static ``CELL_HOUSE_IDX[81, 3]``.
        """
        # y_cell: (B, 81, 3, M) → flat (B, 81, 3M)
        y_c = y[:, cell_house_idx].reshape(x.shape[0], 81, 3 * self.m)
        pe_b = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
        inp = torch.cat([x, y_c, pe_b], dim=-1)
        return self.vf_x(inp)
