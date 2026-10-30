"""Shared bipartite VFs on product ``S^8 × S^{54}`` ↔ house (+ optional global) ``T^{16}``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.manifold_embed import (
    COORD_DIM,
    DECISION_DIM,
    M,
    PE_DIM,
    clifford_embed,
)

# Phase-1 (no z): H_Y=320, H_X=428 → ~680.6k (+180 chart)
# Phase-2 (+z):  H_Y=304, H_Z=96, H_X=412 → ~681.9k (+180 chart)
H_Y_PHASE1 = 320
H_X_PHASE1 = 428
H_Y_PHASE2 = 304
H_X_PHASE2 = 412
H_Z = 96
CHART_DIM = 18  # learned S^8-only chart; mean+std → 36D (Gram drop-in)


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
    Leapfrog pair: house ω from product-cell aggregates + Clifford embeds, then
    FiLM ambient ``(u_x, u_c)`` on ``S^8 × S^{54}``.

    Poles never enter this module. Optional global torus ``z`` via ``use_global_torus``.
    """

    def __init__(
        self,
        m: int = M,
        h_y: int | None = None,
        h_x: int | None = None,
        h_z: int = H_Z,
        pe_dim: int = PE_DIM,
        use_global_torus: bool = True,
    ) -> None:
        super().__init__()
        self.m = m
        self.pe_dim = pe_dim
        self.use_global_torus = bool(use_global_torus)
        if h_y is None:
            h_y = H_Y_PHASE2 if self.use_global_torus else H_Y_PHASE1
        if h_x is None:
            h_x = H_X_PHASE2 if self.use_global_torus else H_X_PHASE1
        self.h_y = h_y
        self.h_x = h_x
        emb_m = 2 * m
        # μ/σ_x (18) + μ/σ_c (110) + chart μ/σ (36) + house_type (3) + y_emb (32) [+ z_emb 32]
        in_y = 2 * DECISION_DIM + 2 * COORD_DIM + 2 * CHART_DIM + 3 + emb_m
        if self.use_global_torus:
            in_y += emb_m
        self.vf_y = _mlp(in_y, h_y, m, n_mid=1)
        self.vf_z: nn.Module | None = None
        if self.use_global_torus:
            # [mean_h(y_emb), std_h(y_emb), z_emb] → 96 → H_Z → 16
            self.vf_z = _mlp(3 * emb_m, h_z, m, n_mid=0)

        # S^8-only decision chart (never reads c / scratch) — 9→18 → +180 params.
        self.w_x_chart = nn.Linear(DECISION_DIM, CHART_DIM)

        # cell_in = [x, c, pe, is_clue] = 9+55+6+1 = 71
        cell_in = DECISION_DIM + COORD_DIM + pe_dim + 1
        house_in = 3 * emb_m + (emb_m if self.use_global_torus else 0)
        self.x_in_cell = nn.Linear(cell_in, h_x)
        self.x_in_house_shift = nn.Linear(house_in, h_x)
        self.x_in_house_gate = nn.Linear(house_in, h_x)
        self.vf_x_body = nn.Sequential(
            nn.Linear(h_x, h_x),
            nn.SiLU(),
            nn.Linear(h_x, h_x),
            nn.SiLU(),
            nn.Linear(h_x, DECISION_DIM + COORD_DIM),
        )
        self.register_buffer(
            'house_type',
            F.one_hot(torch.arange(27) // 9, num_classes=3).to(torch.float32),
            persistent=False,
        )

    def house_features(
        self,
        x: Tensor,
        c: Tensor,
        house_cell_idx: Tensor,
    ) -> Tensor:
        """
        Per-house feats: μ/σ of ``x`` and ``c`` + 36D learned S^8 chart + type.

        Chart ``φ = SiLU(W_chart x)`` is strictly S^8-only (never reads ``c``).
        """
        x_h = x[:, house_cell_idx]  # (B, 27, 9, 9)
        c_h = c[:, house_cell_idx]  # (B, 27, 9, 55)
        mu_x = x_h.mean(dim=2)
        std_x = (x_h.var(dim=2, unbiased=False) + 1e-6).sqrt()
        mu_c = c_h.mean(dim=2)
        std_c = (c_h.var(dim=2, unbiased=False) + 1e-6).sqrt()
        # Strictly S^8: chart once per cell (81×), gather houses, pool μ/σ → 36D.
        phi_x = F.silu(self.w_x_chart(x))  # (B, 81, 18)
        phi_h = phi_x[:, house_cell_idx]  # (B, 27, 9, 18)
        mu_phi = phi_h.mean(dim=2)
        std_phi = (phi_h.var(dim=2, unbiased=False) + 1e-6).sqrt()
        ht = self.house_type.to(dtype=mu_x.dtype).unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat([mu_x, std_x, mu_c, std_c, mu_phi, std_phi, ht], dim=-1)

    def omega(
        self,
        x: Tensor,
        c: Tensor,
        y: Tensor,
        z: Tensor | None,
        house_cell_idx: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """House ``ω_y`` and optional global ``ω_z``."""
        feat = self.house_features(x, c, house_cell_idx)
        y_emb = clifford_embed(y)
        if self.use_global_torus:
            if z is None:
                raise ValueError('use_global_torus=True requires z')
            z_emb = clifford_embed(z)  # (B, 1, 32)
            z_b = z_emb.expand(-1, 27, -1)
            omega_y = self.vf_y(torch.cat([feat, y_emb, z_b], dim=-1))
            mean_y = y_emb.mean(dim=1)
            std_y = (y_emb.var(dim=1, unbiased=False) + 1e-6).sqrt()
            assert self.vf_z is not None
            omega_z = self.vf_z(torch.cat([mean_y, std_y, z_emb.squeeze(1)], dim=-1)).unsqueeze(1)
            return omega_y, omega_z
        omega_y = self.vf_y(torch.cat([feat, y_emb], dim=-1))
        return omega_y, None

    def ambient_u(
        self,
        x: Tensor,
        c: Tensor,
        y: Tensor,
        z: Tensor | None,
        pe: Tensor,
        is_clue: Tensor,
        cell_house_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Cell ambient velocities ``(u_x, u_c)`` via FiLM(cell+coord+PE+clue, houses)."""
        y_emb = clifford_embed(y)  # (B, 27, 32)
        emb_m = 2 * self.m
        house_parts = [y_emb[:, cell_house_idx].reshape(x.shape[0], 81, 3 * emb_m)]
        if self.use_global_torus:
            if z is None:
                raise ValueError('use_global_torus=True requires z')
            z_emb = clifford_embed(z).expand(-1, 81, -1)
            house_parts.append(z_emb)
        house_in = torch.cat(house_parts, dim=-1)
        pe_b = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
        clue_f = is_clue.to(dtype=x.dtype).unsqueeze(-1)
        cell_in = torch.cat([x, c, pe_b, clue_f], dim=-1)
        base = F.silu(self.x_in_cell(cell_in) + self.x_in_house_shift(house_in))
        gate = 1.0 + torch.tanh(self.x_in_house_gate(house_in))
        u = self.vf_x_body(base * gate)
        return u[..., :DECISION_DIM], u[..., DECISION_DIM:]
