"""Shared (weight-tied) bipartite vector fields: S→T (ω) and T→S (u)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.manifold_embed import (
    D,
    M,
    N_DIGITS,
    PE_DIM,
    clifford_embed,
)

# Soft-cap amplitude for Kuramoto ω (Option A: R-normalized; effective max≈0.83).
_OMEGA_CAP = 1.25


def _mlp(in_dim: int, hidden: int, out_dim: int, n_mid: int) -> nn.Sequential:
    """``Linear→SiLU→(Linear→SiLU)×n_mid→Linear``."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(n_mid):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class BipartiteRouter:
    """
    Asymmetric hard incidence for S→T / T→S (no trainable params).

    ``A_s2t`` ∈ R^{H×N} rows sum to 1; ``A_t2s`` ∈ R^{N×K_inc×H} one-hot.
    Population variance via ``relu(A @ x² − μ²)`` (matches ``unbiased=False`` /9);
    no Bessel ``1−∑A²``, no dense ``(B,H,N,D)`` broadcast.
    """

    def __init__(self, a_s2t: Tensor, a_t2s: Tensor) -> None:
        self.a_s2t = a_s2t
        self.a_t2s = a_t2s

    @staticmethod
    def house_features_from_a(
        x: Tensor,
        poles: Tensor,
        a_s2t: Tensor,
        d_sem: int = N_DIGITS,
    ) -> Tensor:
        """
        Per-house ``(B, H, 2D+2·d_sem)`` via hard ``A_s2t`` matmuls only.

        Population second moment matches tip ``var(..., unbiased=False)`` (/C0):
        for row-stochastic hard-uniform ``A`` (entries ``1/C0``), reduce as
        ``sum(S x)/C0`` with binary ``S=A/max(A)`` so fp32 matches gather
        ``mean`` order (no Bessel ``1−∑A²``, no dense ``(B,H,N,D)`` broadcast).
        """
        # Hard-uniform: recover binary S and C0 so sum-then-divide matches gather.
        row_max = a_s2t.max(dim=-1).values.clamp_min(1e-12)  # (H,) = 1/C0
        s = a_s2t / row_max.unsqueeze(-1)  # (H, N) ∈ {0,1}
        c0 = s.sum(dim=-1).clamp_min(1.0).view(1, -1, 1)  # (1, H, 1)
        sx = torch.einsum('hn,bnd->bhd', s, x)
        mu = sx / c0
        var = F.relu(torch.einsum('hn,bnd->bhd', s, x.square()) / c0 - mu.square())
        std = (var + 1e-6).sqrt()
        p_sem = poles[:, :d_sem]
        q = F.relu(x[..., :d_sem] @ p_sem.T)  # (B, N, d_sem)
        q_mean = torch.einsum('hn,bnd->bhd', s, q) / c0
        q_sq_mean = torch.einsum('hn,bnd->bhd', s, q.square()) / c0
        return torch.cat([mu, std, q_mean, q_sq_mean], dim=-1)

    @staticmethod
    def house_concat_from_a(y_emb: Tensor, a_t2s: Tensor) -> Tensor:
        """T→S concat ``(B, N, K_inc·2m)`` via ``A_t2s``; matches gather+reshape."""
        gathered = torch.einsum('nkh,bhe->bnke', a_t2s, y_emb)
        b, n, k_inc, e = gathered.shape
        return gathered.reshape(b, n, k_inc * e)

    def house_features(self, x: Tensor, poles: Tensor, d_sem: int = N_DIGITS) -> Tensor:
        return self.house_features_from_a(x, poles, self.a_s2t, d_sem=d_sem)

    def house_concat(self, y_emb: Tensor) -> Tensor:
        return self.house_concat_from_a(y_emb, self.a_t2s)


class CoupledVectorField(nn.Module):
    """
    Leapfrog pair: Kuramoto house ω from cell aggregates + Clifford ``y_emb``,
    then cell ambient u via FiLM → ungated semantic ``:d_sem`` + gated scratch.

    Bulletproof attractor ~687k: ``H_Y=320`` (1 mid), ``H_X=432`` (FiLM + 2 mid);
    ``M=16``, ``in_y=178``, ``vf_y`` out ``2M``, ``vf_x_body`` out ``D+1``.

    Dual API: gather indices (reference) or hard ``A_s2t``/``A_t2s`` (production);
    both share the same MLP weights.
    """

    def __init__(
        self,
        d: int = D,
        m: int = M,
        h_y: int = 320,
        h_x: int = 432,
        pe_dim: int = PE_DIM,
        d_sem: int = N_DIGITS,
        k_inc: int = 3,
    ) -> None:
        super().__init__()
        self.d = d
        self.m = m
        self.pe_dim = pe_dim
        self.d_sem = d_sem
        self.k_inc = k_inc
        emb_m = 2 * m  # Clifford cat(cos,sin)
        # mean+std (2d) + mean q + mean q² (2·d_sem) + y_emb (2m)
        in_y = 2 * d + 2 * d_sem + emb_m
        # Kuramoto targets (c_cos, c_sin); no α-bias hack.
        self.vf_y = _mlp(in_y, h_y, 2 * m, n_mid=1)
        # FiLM first block: cell(+PE) + house shift/gate → H_X; then mid×2 + out.
        self.x_in_cell = nn.Linear(d + pe_dim, h_x)
        self.x_in_house_shift = nn.Linear(k_inc * emb_m, h_x)
        self.x_in_house_gate = nn.Linear(k_inc * emb_m, h_x)
        # Out D+1: signed ambient u (D) + gate logit (1).
        self.vf_x_body = _mlp(h_x, h_x, d + 1, n_mid=1)

    def house_features(
        self,
        x: Tensor,
        poles: Tensor,
        house_cell_idx: Tensor,
    ) -> Tensor:
        """
        Gather reference: per-house ``(B, 27, 2D+2·d_sem)`` mean+std + claim means.

        ``q = relu(x_{:d_sem} P_{:d_sem}^T)``; ``std = (var + 1e-6).sqrt()`` with
        ``var(..., unbiased=False)`` (/C0). Kept for parity tests.
        """
        # x_cells: (B, H, C0, D)
        x_h = x[:, house_cell_idx]
        mean = x_h.mean(dim=2)
        var = x_h.var(dim=2, unbiased=False)
        std = (var + 1e-6).sqrt()
        p_sem = poles[:, : self.d_sem]
        q = F.relu(x[..., : self.d_sem] @ p_sem.T)
        q_h = q[:, house_cell_idx]
        q_mean = q_h.mean(dim=2)
        q_sq_mean = (q_h * q_h).mean(dim=2)
        return torch.cat([mean, std, q_mean, q_sq_mean], dim=-1)

    def house_features_hard(
        self,
        x: Tensor,
        poles: Tensor,
        a_s2t: Tensor,
    ) -> Tensor:
        """Hard-A S→T features via ``BipartiteRouter`` (population /C0 variance)."""
        return BipartiteRouter.house_features_from_a(
            x, poles, a_s2t, d_sem=self.d_sem
        )

    def omega(
        self,
        x: Tensor,
        y: Tensor,
        poles: Tensor,
        house_cell_idx: Tensor,
    ) -> Tensor:
        """Gather-path Kuramoto ω ``(B, H, M)``."""
        feat = self.house_features(x, poles, house_cell_idx)
        return self._omega_from_feat(feat, y)

    def omega_hard(
        self,
        x: Tensor,
        y: Tensor,
        poles: Tensor,
        a_s2t: Tensor,
    ) -> Tensor:
        """Hard-A Kuramoto ω sharing ``vf_y`` weights."""
        feat = self.house_features_hard(x, poles, a_s2t)
        return self._omega_from_feat(feat, y)

    def _omega_from_feat(self, feat: Tensor, y: Tensor) -> Tensor:
        """
        Kuramoto Option A from features + Clifford ``y_emb``.

        MLP emits ``(c_cos, c_sin)``; ``ω_raw = c_sin·cos(y) - c_cos·sin(y)``,
        ``c_norm = √(c_cos²+c_sin²+1)``, then
        ``ω = A·tanh(ω_raw/(A·c_norm))`` with ``A=1.25``.
        """
        y_emb = clifford_embed(y)
        raw = self.vf_y(torch.cat([feat, y_emb], dim=-1))
        c_cos, c_sin = raw.split(self.m, dim=-1)
        c_norm = (c_cos.square() + c_sin.square()).add(1.0).sqrt()
        omega_raw = c_sin * torch.cos(y) - c_cos * torch.sin(y)
        return _OMEGA_CAP * torch.tanh(omega_raw / (_OMEGA_CAP * c_norm))

    def ambient_u(
        self,
        x: Tensor,
        y: Tensor,
        pe: Tensor,
        cell_house_idx: Tensor,
        poles: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Gather-path cell ambient ``(B, N, D)`` + gate."""
        del poles  # house ETF poles unused; ambient is signed coord head
        y_emb = clifford_embed(y)  # (B, H, 2m)
        emb_m = 2 * self.m
        house_in = y_emb[:, cell_house_idx].reshape(x.shape[0], x.shape[1], self.k_inc * emb_m)
        return self._ambient_from_house_in(x, pe, house_in)

    def ambient_u_hard(
        self,
        x: Tensor,
        y: Tensor,
        pe: Tensor,
        a_t2s: Tensor,
        poles: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Hard-A T→S ambient sharing FiLM / ``vf_x_body`` weights."""
        del poles
        y_emb = clifford_embed(y)
        house_in = BipartiteRouter.house_concat_from_a(y_emb, a_t2s)
        return self._ambient_from_house_in(x, pe, house_in)

    def _ambient_from_house_in(
        self,
        x: Tensor,
        pe: Tensor,
        house_in: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        FiLM → ungated semantic ``:d_sem`` + gated scratch ``d_sem:``.

        Head ``D+1``: ``u_raw`` (D) + gate logit. Returns ``(u, gate)``.
        """
        pe_b = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
        cell_in = torch.cat([x, pe_b], dim=-1)  # (B, N, D+PE_DIM)
        base = F.silu(self.x_in_cell(cell_in) + self.x_in_house_shift(house_in))
        film_gate = 1.0 + torch.tanh(self.x_in_house_gate(house_in))
        raw = self.vf_x_body(base * film_gate)  # (B, N, D+1)

        u_raw, g_raw = raw[..., : self.d], raw[..., self.d]
        gate = torch.sigmoid(g_raw + 2.0)  # ~0.88 open at init
        u_semantic = u_raw[..., : self.d_sem]
        u_scratch = gate.unsqueeze(-1) * u_raw[..., self.d_sem :]
        u = torch.cat([u_semantic, u_scratch], dim=-1)
        return u, gate
