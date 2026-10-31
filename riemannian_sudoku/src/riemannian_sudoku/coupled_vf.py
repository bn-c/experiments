"""Shared (weight-tied) bipartite vector fields: S→T (ω) and T→S (u)."""

from __future__ import annotations

import math

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
_SINKHORN_ITERS = 8
_SIGMA_INIT = 0.5
_HARD_ENTRY = 1.0 / 9.0


def _mlp(in_dim: int, hidden: int, out_dim: int, n_mid: int) -> nn.Sequential:
    """``Linear→SiLU→(Linear→SiLU)×n_mid→Linear``."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(n_mid):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _sinkhorn_rect(logits: Tensor, tau: Tensor, n_iters: int = _SINKHORN_ITERS) -> Tensor:
    """
    Rectangular Sinkhorn on ``(K, 9, 81)`` logits → log-domain ``Z``.

    Alternates col-norm (mass 1) and row-norm (mass 9). Returns ``Z`` after a
    final col-norm so ``exp(Z)`` columns sum to 1.
    """
    z = logits / tau.clamp_min(1e-6)
    log9 = math.log(9.0)
    for _ in range(n_iters):
        z = z - torch.logsumexp(z, dim=1, keepdim=True)  # cols → 1
        z = z - torch.logsumexp(z, dim=2, keepdim=True) + log9  # rows → 9
    z = z - torch.logsumexp(z, dim=1, keepdim=True)  # end on cols → 1
    return z


class BipartiteRouter(nn.Module):
    """
    Hard or soft-Sinkhorn bipartite incidence (Phase-1 / Phase-2).

    Hard: static buffers ``A_s2t`` (27×81 row-stochastic), ``A_t2s`` (81×3×27).
    Soft: trainable Φ ``(3,9,81)`` + optional pinned families → Sinkhorn →
    ``A_s2t`` rows sum 1 (never rows-sum-9 into house features) and block
    ``A_t2s``. Population variance: hard binary-S gather parity; soft weighted.
    """

    def __init__(
        self,
        a_s2t: Tensor,
        a_t2s: Tensor,
        *,
        routing_mode: str = 'hard',
        sigma_init: float = _SIGMA_INIT,
        sinkhorn_iters: int = _SINKHORN_ITERS,
    ) -> None:
        super().__init__()
        if routing_mode not in ('hard', 'soft_sinkhorn'):
            raise ValueError(f'routing_mode={routing_mode!r}')
        self.routing_mode = routing_mode
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.register_buffer('a_s2t_hard', a_s2t)
        self.register_buffer('a_t2s_hard', a_t2s)
        # Compat aliases for Phase-1 call sites / tests.
        self.a_s2t = a_s2t
        self.a_t2s = a_t2s
        if routing_mode == 'soft_sinkhorn':
            self.logits = nn.Parameter(torch.randn(3, 9, 81) * float(sigma_init))
            self.register_buffer('pinned_logits', torch.zeros(3, 9, 81))
            self.register_buffer('pinned_mask', torch.zeros(3, dtype=torch.bool))
        else:
            self.register_parameter('logits', None)
            self.register_buffer('pinned_logits', torch.zeros(3, 9, 81))
            self.register_buffer('pinned_mask', torch.zeros(3, dtype=torch.bool))

    def effective_logits(self) -> Tensor:
        """Blend pinned families over learnable Φ (soft only)."""
        assert self.logits is not None
        mask = self.pinned_mask.view(3, 1, 1)
        return torch.where(mask, self.pinned_logits, self.logits)

    def get_incidence(self, tau: Tensor | float = 0.5) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return ``(A_s2t, A_t2s, P_t2s)``.

        Soft: ``P_t2s=exp(Z)`` cols sum 1; ``A_s2t=exp(Z)/9`` rows sum 1 stacked
        over families → ``(27,81)``; ``A_t2s`` block-embeds ``P`` into ``(81,3,27)``.
        Hard: Phase-1 buffers; ``P_t2s`` is the GT one-hot ``(3,9,81)``.
        """
        if self.routing_mode == 'hard':
            p = torch.zeros(3, 9, 81, device=self.a_s2t_hard.device, dtype=self.a_s2t_hard.dtype)
            # Recover one-hot P from hard A_t2s for diags.
            for k in range(3):
                # houses [9k:9k+9] ← local 0..8
                block = self.a_t2s_hard[:, k, 9 * k : 9 * k + 9]  # (81, 9)
                p[k] = block.T
            return self.a_s2t_hard, self.a_t2s_hard, p

        tau_t = tau if isinstance(tau, Tensor) else torch.tensor(float(tau), device=self.logits.device)
        tau_t = tau_t.to(device=self.logits.device, dtype=self.logits.dtype).clamp_min(1e-3)
        # Per-family unit-std so AdamW ‖Φ‖ shrink cannot blur Sinkhorn at small τ.
        eff = self.effective_logits()
        phi_mean = eff.mean(dim=(1, 2), keepdim=True)
        phi_std = eff.std(dim=(1, 2), keepdim=True, unbiased=False).clamp(min=1e-4)
        phi_norm = (eff - phi_mean) / phi_std
        z = _sinkhorn_rect(phi_norm, tau_t, n_iters=self.sinkhorn_iters)
        p = z.exp()  # (3,9,81) cols sum 1
        # A_s2t: row-stochastic — NEVER feed rows-sum-9 into house_features.
        a_s2t = (p / 9.0).reshape(27, 81)
        a_t2s = torch.zeros(81, 3, 27, device=p.device, dtype=p.dtype)
        for k in range(3):
            a_t2s[:, k, 9 * k : 9 * k + 9] = p[k].T  # (81, 9)
        return a_s2t, a_t2s, p

    def pin_families(
        self,
        family_logits: Tensor,
        families: Tensor | list[int] | None = None,
    ) -> None:
        """Pin selected families to ``family_logits`` (no N(0,σ) on pinned)."""
        if families is None:
            families = [0, 1, 2]
        if not isinstance(families, Tensor):
            families = torch.tensor(list(families), dtype=torch.long)
        self.pinned_logits.copy_(family_logits.to(self.pinned_logits.dtype))
        self.pinned_mask.zero_()
        self.pinned_mask[families.long()] = True

    def clear_pins(self) -> None:
        self.pinned_mask.zero_()

    @staticmethod
    def house_features_from_a(
        x: Tensor,
        poles: Tensor,
        a_s2t: Tensor,
        d_sem: int = N_DIGITS,
        *,
        soft: bool = False,
    ) -> Tensor:
        """
        Per-house ``(B, H, 2D+2·d_sem)`` via ``A_s2t`` matmuls.

        Hard-uniform (default): binary ``S=A/max`` + ``/C0`` for gather fp32
        parity. Soft row-stochastic (``soft=True``): weighted mean/var via ``A``
        directly — ``A`` must rows-sum-1 (never rows-sum-9).
        """
        p_sem = poles[:, :d_sem]
        q = F.relu(x[..., :d_sem] @ p_sem.T)  # (B, N, d_sem)
        if soft:
            mu = torch.einsum('hn,bnd->bhd', a_s2t, x)
            var = F.relu(torch.einsum('hn,bnd->bhd', a_s2t, x.square()) - mu.square())
            q_mean = torch.einsum('hn,bnd->bhd', a_s2t, q)
            q_sq_mean = torch.einsum('hn,bnd->bhd', a_s2t, q.square())
        else:
            row_max = a_s2t.max(dim=-1).values.clamp_min(1e-12)
            s = a_s2t / row_max.unsqueeze(-1)
            c0 = s.sum(dim=-1).clamp_min(1.0).view(1, -1, 1)
            mu = torch.einsum('hn,bnd->bhd', s, x) / c0
            var = F.relu(torch.einsum('hn,bnd->bhd', s, x.square()) / c0 - mu.square())
            q_mean = torch.einsum('hn,bnd->bhd', s, q) / c0
            q_sq_mean = torch.einsum('hn,bnd->bhd', s, q.square()) / c0
        std = (var + 1e-6).sqrt()
        return torch.cat([mu, std, q_mean, q_sq_mean], dim=-1)

    @staticmethod
    def house_concat_from_a(y_emb: Tensor, a_t2s: Tensor) -> Tensor:
        """T→S concat ``(B, N, K_inc·2m)`` via ``A_t2s``; matches gather+reshape."""
        gathered = torch.einsum('nkh,bhe->bnke', a_t2s, y_emb)
        b, n, k_inc, e = gathered.shape
        return gathered.reshape(b, n, k_inc * e)

    def house_features(self, x: Tensor, poles: Tensor, d_sem: int = N_DIGITS) -> Tensor:
        return self.house_features_from_a(x, poles, self.a_s2t_hard, d_sem=d_sem)

    def house_concat(self, y_emb: Tensor) -> Tensor:
        return self.house_concat_from_a(y_emb, self.a_t2s_hard)


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
        *,
        soft: bool = False,
    ) -> Tensor:
        """S→T features via ``BipartiteRouter`` (hard gather-parity or soft weighted)."""
        return BipartiteRouter.house_features_from_a(
            x, poles, a_s2t, d_sem=self.d_sem, soft=soft
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
        *,
        soft: bool = False,
    ) -> Tensor:
        """Hard/soft-A Kuramoto ω sharing ``vf_y`` weights."""
        feat = self.house_features_hard(x, poles, a_s2t, soft=soft)
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
