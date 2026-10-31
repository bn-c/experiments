"""Phase-3 Multi-Manifold banks, shared-Φ router, and static T↔S attractor."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.geodesic import free_sphere_step, geodesic_step, torus_step
from riemannian_sudoku.manifold_embed import (
    BLANK_COORD,
    D,
    M,
    N_DIGITS,
    PE_ALPHA,
    PE_DIM,
    SCRATCH_START,
    SudokuProblemAdapter,
    build_pe,
    cell_house_idx,
    clifford_embed,
    gt_peer_mask,
    house_cell_idx,
    simplex_etf_poles,
)
from riemannian_sudoku.solver import (
    CE_GAMMA,
    K_DEFAULT,
    LAMBDA_CLASH_DEFAULT,
    LAMBDA_ENT_DEFAULT,
    LAMBDA_ORTHO_DEFAULT,
    LOGIT_SCALE_INIT,
    LOGIT_SCALE_MAX,
    ORTHO_MARGIN,
    PURITY_TOPK,
    TAU_INIT,
    column_entropy,
    family_gram_sims,
    ortho_hinge,
    sudoku_graph_purity,
)

# Soft-cap Kuramoto Option A (same lock as Phase-2).
_OMEGA_CAP = 1.25
_SINKHORN_ITERS = 8
_D_MSG = 64
_F_HEADS = 3
_N_PER = 9
_GAMMA_MAX = 1.0
_GAMMA_ANNEAL_STEPS = 1000
_BETA_INIT = 2.0
# v3.1 staggered depths — break T1/T2 init symmetry.
_H_INIT = {'S0': 0.0, 'T1': 0.95, 'T2': 1.65, 'S1': 1.35}
_EDGE_NAMES = ('S0_T1', 'S0_T2', 'S1_T1', 'S1_T2')
_LAMBDA_BETA_L1 = 1e-3
_LAMBDA_BETA_L0 = 1e-3
_LAMBDA_COMP = 0.05
_LAMBDA_REPEL = 0.05
_BETA_MOM = 0.4
_CE_RHO = 0.5
_READ_BIAS = -0.2


def _mlp(in_dim: int, hidden: int, out_dim: int, n_mid: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
    for _ in range(n_mid):
        layers.append(nn.Linear(hidden, hidden))
        layers.append(nn.SiLU())
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _phi_norm(phi: Tensor) -> Tensor:
    """Per-head unit-std (unbiased=False); τ clamp applied by caller."""
    mean = phi.mean(dim=(1, 2), keepdim=True)
    std = phi.std(dim=(1, 2), keepdim=True, unbiased=False).clamp(min=1e-4)
    return (phi - mean) / std


def _sinkhorn_dustbin(
    phi: Tensor,
    dust: Tensor,
    tau: Tensor,
    n_iters: int = _SINKHORN_ITERS,
) -> tuple[Tensor, Tensor]:
    """
    Unbalanced Sinkhorn with dustbin column on the sphere (source) axis.

    ``phi`` (F, H, N), ``dust`` (F, H). Returns ``(P_real, P_dust)`` after a
    final row-norm so each house row of ``cat(P_real, P_dust)`` sums to 1.
    """
    tau_t = tau.clamp_min(1e-3)
    z = torch.cat([phi, dust.unsqueeze(-1)], dim=-1) / tau_t
    n_cols = z.shape[-1]
    log_row = 0.0  # row mass 1 (incl. dustbin)
    for _ in range(n_iters):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True) + log_row  # rows → 1
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)  # cols → soft
    z = z - torch.logsumexp(z, dim=-1, keepdim=True) + log_row
    del n_cols
    p = z.exp()
    return p[..., :-1], p[..., -1]


def _row_stoch_dustbin(p_real: Tensor, p_dust: Tensor) -> Tensor:
    """``A = P_real / (∑_real P + P_dust + ε)`` — dustbin stays in the denom."""
    denom = p_real.sum(dim=-1, keepdim=True) + p_dust.unsqueeze(-1) + 1e-6
    return p_real / denom


def _row_stoch_transpose(p_real: Tensor) -> Tensor:
    """Cell←house: ``A[n,h] = P[h,n] / (∑_h P[h,n] + ε)``."""
    # p_real: (F, H, N) → A_t2s (F, N, H)
    pt = p_real.transpose(-1, -2)
    return pt / (pt.sum(dim=-1, keepdim=True) + 1e-6)


def alpha_from_depth(
    h_i: Tensor,
    h_j: Tensor,
    beta: Tensor,
    gamma: Tensor,
) -> Tensor:
    """``α = σ(β) · exp(−γ(|h_i−h_j|−1)²)``."""
    gate = torch.sigmoid(beta)
    depth = (-gamma * (torch.abs(h_i - h_j) - 1.0).square()).exp()
    return gate * depth


class SphereBank(nn.Module):
    """
    Pole-free sphere bank (v3.1): LN-free SiLU Read; FiLM Write on ``[μ,σ]``.

    ETF / blank NP stay in ``SudokuProblemAdapter`` only — never enter Read.
    """

    def __init__(
        self,
        n: int,
        d: int,
        d_msg: int = _D_MSG,
        pe_dim: int = 0,
        *,
        ungated_sem: bool = False,
        d_sem: int = N_DIGITS,
        h_x: int = 256,
        read_bias: float = _READ_BIAS,
        beta_mom: float = _BETA_MOM,
    ) -> None:
        super().__init__()
        self.n = int(n)
        self.d = int(d)
        self.d_msg = int(d_msg)
        self.pe_dim = int(pe_dim)
        self.ungated_sem = bool(ungated_sem)
        self.d_sem = int(d_sem)
        self.beta_mom = float(beta_mom)
        # LN-free SiLU(Wx+b); b≈−0.2 keeps North-Pole blanks quiet.
        self.read_proj = nn.Linear(d, d_msg)
        nn.init.constant_(self.read_proj.bias, float(read_bias))
        # FiLM: cell(+PE) + cat[μ,σ] → ambient u
        self.x_in_cell = nn.Linear(d + pe_dim, h_x)
        self.x_in_shift = nn.Linear(2 * d_msg, h_x)
        self.x_in_gate = nn.Linear(2 * d_msg, h_x)
        self.vf_body = _mlp(h_x, h_x, d + 1, n_mid=1)

    def read(self, x: Tensor) -> Tensor:
        """``m_S = SiLU(W_read x + b)`` — no LayerNorm, no poles."""
        return F.silu(self.read_proj(x))

    def write(
        self,
        x: Tensor,
        house_in: Tensor,
        pe: Tensor | None,
        x0: Tensor | None,
        is_clue: Tensor | None,
        v_prev: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        FiLM Write on ``house_in=cat[μ,σ]`` → ambient u → geodesic Exp.

        Geodesic momentum: ``u_eff = u + β v_prev``; returns post-cap ``v_c``
        (zero on clues). ``v_prev`` is a local integrate tensor, not a buffer.
        """
        if self.pe_dim > 0:
            assert pe is not None
            pe_b = pe.unsqueeze(0).expand(x.shape[0], -1, -1)
            cell_in = torch.cat([x, pe_b], dim=-1)
        else:
            cell_in = x
        base = F.silu(self.x_in_cell(cell_in) + self.x_in_shift(house_in))
        film = 1.0 + torch.tanh(self.x_in_gate(house_in))
        raw = self.vf_body(base * film)
        u_raw, g_raw = raw[..., : self.d], raw[..., self.d]
        gate = torch.sigmoid(g_raw + 2.0)
        if self.ungated_sem:
            u_sem = u_raw[..., : self.d_sem]
            u_scr = gate.unsqueeze(-1) * u_raw[..., self.d_sem :]
            u = torch.cat([u_sem, u_scr], dim=-1)
        else:
            u = gate.unsqueeze(-1) * u_raw
        if v_prev is not None:
            u = u + self.beta_mom * v_prev
        if x0 is not None and is_clue is not None:
            x_new, v_c = geodesic_step(x, u, x0, is_clue, return_v_c=True)
        else:
            x_new, v_c = free_sphere_step(x, u, return_v_c=True)
        return x_new, gate, v_c


class TorusBank(nn.Module):
    """
    Torus bank with ``F`` sub-family heads (``N = F · N_per``).

    **Shared** Clifford→``W_read`` across families (permutation-invariant σ).
    Write = Kuramoto Option A on S→T ``[μ,σ]``. VFs never see raw ``y``.
    """

    def __init__(
        self,
        n_per: int = _N_PER,
        m: int = M,
        f_heads: int = _F_HEADS,
        d_msg: int = _D_MSG,
        h_y: int = 256,
    ) -> None:
        super().__init__()
        self.n_per = int(n_per)
        self.m = int(m)
        self.f_heads = int(f_heads)
        self.n = self.f_heads * self.n_per
        self.d_msg = int(d_msg)
        emb_m = 2 * m
        # One shared Linear across all F heads — no per-family read / randperm.
        self.read_proj = nn.Linear(emb_m, d_msg)
        self.vf_y = _mlp(2 * d_msg + emb_m, h_y, 2 * m, n_mid=1)

    def read(self, y: Tensor) -> Tensor:
        return self.read_proj(clifford_embed(y))

    def write(self, y: Tensor, packet_mu_sigma: Tensor) -> tuple[Tensor, Tensor]:
        """Kuramoto Option A from ``[μ,σ]`` packet + Clifford ``y_emb``."""
        y_emb = clifford_embed(y)
        raw = self.vf_y(torch.cat([packet_mu_sigma, y_emb], dim=-1))
        c_cos, c_sin = raw.split(self.m, dim=-1)
        c_norm = (c_cos.square() + c_sin.square()).add(1.0).sqrt()
        omega_raw = c_sin * torch.cos(y) - c_cos * torch.sin(y)
        omega = _OMEGA_CAP * torch.tanh(omega_raw / (_OMEGA_CAP * c_norm))
        return torus_step(y, omega), omega


class MultiBankRouter(nn.Module):
    """
    Shared-Φ bipartite router on exactly 4 edges: S0↔T1, S0↔T2, S1↔T1, S1↔T2.

    ``Φ_{S,T} ∈ R^{F×N_per×N_S}``; phi_norm; unbalanced Sinkhorn + dustbin-in-denom;
    depth α with γ anneal; bidirectional ``(μ,σ)`` with active-mass T→S (v3.1).
    """

    def __init__(
        self,
        n_s0: int = 81,
        n_s1: int = 9,
        f_heads: int = _F_HEADS,
        n_per: int = _N_PER,
        sigma_init: float = 0.5,
        sinkhorn_iters: int = _SINKHORN_ITERS,
    ) -> None:
        super().__init__()
        self.f_heads = int(f_heads)
        self.n_per = int(n_per)
        self.n_s0 = int(n_s0)
        self.n_s1 = int(n_s1)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.edge_names = _EDGE_NAMES
        # Shared undirected Φ per unordered pair {S,T}.
        self.phi_S0_T1 = nn.Parameter(torch.randn(f_heads, n_per, n_s0) * sigma_init)
        self.phi_S0_T2 = nn.Parameter(torch.randn(f_heads, n_per, n_s0) * sigma_init)
        self.phi_S1_T1 = nn.Parameter(torch.randn(f_heads, n_per, n_s1) * sigma_init)
        self.phi_S1_T2 = nn.Parameter(torch.randn(f_heads, n_per, n_s1) * sigma_init)
        # Dustbin logits (F, H) — init slightly negative so real mass wins early.
        self.dust_S0_T1 = nn.Parameter(torch.full((f_heads, n_per), -1.0))
        self.dust_S0_T2 = nn.Parameter(torch.full((f_heads, n_per), -1.0))
        self.dust_S1_T1 = nn.Parameter(torch.full((f_heads, n_per), -1.0))
        self.dust_S1_T2 = nn.Parameter(torch.full((f_heads, n_per), -1.0))
        # Depth hierarchy: h_S0 pinned at 0; others learnable.
        self.h_T1 = nn.Parameter(torch.tensor(_H_INIT['T1']))
        self.h_T2 = nn.Parameter(torch.tensor(_H_INIT['T2']))
        self.h_S1 = nn.Parameter(torch.tensor(_H_INIT['S1']))
        self.register_buffer('h_S0', torch.tensor(_H_INIT['S0']), persistent=True)
        # β_ij init +2 (open gates); L0/L1 prune pressure in loss.
        self.beta_S0_T1 = nn.Parameter(torch.tensor(_BETA_INIT))
        self.beta_S0_T2 = nn.Parameter(torch.tensor(_BETA_INIT))
        self.beta_S1_T1 = nn.Parameter(torch.tensor(_BETA_INIT))
        self.beta_S1_T2 = nn.Parameter(torch.tensor(_BETA_INIT))
        # Optional GT pin masks for S0–T* edges (Phase-2 gate compat).
        self.register_buffer('pinned_S0_T1', torch.zeros(f_heads, n_per, n_s0))
        self.register_buffer('pin_mask_S0_T1', torch.zeros(f_heads, dtype=torch.bool))
        self.register_buffer('pinned_S0_T2', torch.zeros(f_heads, n_per, n_s0))
        self.register_buffer('pin_mask_S0_T2', torch.zeros(f_heads, dtype=torch.bool))

    def _eff_phi(self, phi: Tensor, pinned: Tensor, mask: Tensor) -> Tensor:
        if mask.any():
            return torch.where(mask.view(-1, 1, 1), pinned, phi)
        return phi

    def pin_s0_edge(
        self,
        edge: str,
        family_logits: Tensor,
        families: list[int] | None = None,
    ) -> None:
        """Pin selected F-heads on an S0–T edge to GT Φ."""
        if families is None:
            families = [0, 1, 2]
        fam = torch.tensor(families, dtype=torch.long)
        if edge == 'S0_T1':
            self.pinned_S0_T1.copy_(family_logits.to(self.pinned_S0_T1.dtype))
            self.pin_mask_S0_T1.zero_()
            self.pin_mask_S0_T1[fam] = True
        elif edge == 'S0_T2':
            self.pinned_S0_T2.copy_(family_logits.to(self.pinned_S0_T2.dtype))
            self.pin_mask_S0_T2.zero_()
            self.pin_mask_S0_T2[fam] = True
        else:
            raise ValueError(edge)

    def clear_pins(self) -> None:
        self.pin_mask_S0_T1.zero_()
        self.pin_mask_S0_T2.zero_()

    def alphas(self, gamma: Tensor) -> dict[str, Tensor]:
        return {
            'S0_T1': alpha_from_depth(self.h_S0, self.h_T1, self.beta_S0_T1, gamma),
            'S0_T2': alpha_from_depth(self.h_S0, self.h_T2, self.beta_S0_T2, gamma),
            'S1_T1': alpha_from_depth(self.h_S1, self.h_T1, self.beta_S1_T1, gamma),
            'S1_T2': alpha_from_depth(self.h_S1, self.h_T2, self.beta_S1_T2, gamma),
        }

    def get_incidence(self, tau: Tensor) -> dict[str, dict[str, Tensor]]:
        """Per-edge ``{A_s2t, A_t2s, P, P_dust}`` with dustbin-in-denom A."""
        specs = (
            ('S0_T1', self._eff_phi(self.phi_S0_T1, self.pinned_S0_T1, self.pin_mask_S0_T1), self.dust_S0_T1),
            ('S0_T2', self._eff_phi(self.phi_S0_T2, self.pinned_S0_T2, self.pin_mask_S0_T2), self.dust_S0_T2),
            ('S1_T1', self.phi_S1_T1, self.dust_S1_T1),
            ('S1_T2', self.phi_S1_T2, self.dust_S1_T2),
        )
        out: dict[str, dict[str, Tensor]] = {}
        for name, phi, dust in specs:
            pn = _phi_norm(phi)
            p_real, p_dust = _sinkhorn_dustbin(pn, dust, tau, n_iters=self.sinkhorn_iters)
            out[name] = {
                'P': p_real,
                'P_dust': p_dust,
                'A_s2t': _row_stoch_dustbin(p_real, p_dust),  # (F,H,N_S)
                'A_t2s': _row_stoch_transpose(p_real),  # (F,N_S,H)
            }
        return out

    def packet_s2t(
        self,
        a_s2t: Tensor,
        msg_s: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        """
        S→T: per-head [μ,σ], unit-sum over F into flat ``(B, F·H, 2·d_msg)``, ×α.

        ``a_s2t`` (F,H,N), ``msg_s`` (B,N,d_msg).
        """
        mu = torch.einsum('fhn,bnd->bfhd', a_s2t, msg_s)
        m2 = torch.einsum('fhn,bnd->bfhd', a_s2t, msg_s.square())
        sigma = (F.relu(m2 - mu.square()) + 1e-6).sqrt()
        pkt = torch.cat([mu, sigma], dim=-1)  # (B,F,H,2d)
        b, f, h, d2 = pkt.shape
        flat = pkt.reshape(b, f * h, d2)
        return flat * alpha

    def packet_t2s(
        self,
        p_real: Tensor,
        msg_t: Tensor,
        alpha: Tensor,
    ) -> Tensor:
        """
        T→S active-mass ``(μ,σ)`` → ``(B, N_S, 2·d_msg)``.

        ``w_{f,n} = α · N_per · Σ_h P[f,h,n]`` (GT ≈ α; dustbin ⇒ w→0).
        Per-head msgs via dustbin-aware ``A_t2s``; then
        ``μ = 3/max(W,3) · Σ_f w_f m_f``, ``σ`` = active-mass-weighted std across heads.
        """
        b, n_t, d = msg_t.shape
        f, h, n_s = p_real.shape
        if n_t != f * h:
            raise ValueError(f'msg_t N={n_t} != F·H={f * h}')
        msg_heads = msg_t.reshape(b, f, h, d)
        a_t2s = _row_stoch_transpose(p_real)  # (F, N, H)
        mu_f = torch.einsum('fnh,bfhd->bnfd', a_t2s, msg_heads)  # (B,N,F,d)
        # Active mass per head/cell; ×N_per so GT house mass 1/9 → w≈α.
        w = alpha * float(self.n_per) * p_real.sum(dim=1)  # (F, N)
        w_bnf = w.transpose(0, 1).unsqueeze(0).unsqueeze(-1)  # (1,N,F,1)
        weighted = (mu_f * w_bnf).sum(dim=2)  # (B,N,d)
        w_sum = w.sum(dim=0) + 1e-6  # (N,)
        scale = 3.0 / w_sum.clamp(min=3.0)  # (N,)
        mu = weighted * scale.view(1, n_s, 1)
        mean = weighted / w_sum.view(1, n_s, 1)
        diff = mu_f - mean.unsqueeze(2)
        var = (w_bnf * diff.square()).sum(dim=2) / w_sum.view(1, n_s, 1)
        sigma = (F.relu(var) + 1e-6).sqrt()
        return torch.cat([mu, sigma], dim=-1)


class MultiBankAttractorCore:
    """Static leapfrog: ``step_all_T()`` then ``step_all_S()``; no ``argsort(h)``."""

    def __init__(
        self,
        s0: SphereBank,
        s1: SphereBank,
        t1: TorusBank,
        t2: TorusBank,
        router: MultiBankRouter,
        num_steps: int = K_DEFAULT,
    ) -> None:
        self.s0 = s0
        self.s1 = s1
        self.t1 = t1
        self.t2 = t2
        self.router = router
        self.num_steps = int(num_steps)

    def integrate(
        self,
        x0: Tensor,
        s1_0: Tensor,
        y1_0: Tensor,
        y2_0: Tensor,
        is_clue: Tensor,
        pe: Tensor,
        incidence: dict[str, dict[str, Tensor]],
        alphas: dict[str, Tensor],
    ) -> dict[str, Tensor | list[Tensor]]:
        x, s1, y1, y2 = x0, s1_0, y1_0, y2_0
        # Local momentum state (compile-safe loop tensors — not nn.Buffer).
        v_s0 = torch.zeros_like(x0)
        v_s1 = torch.zeros_like(s1_0)
        omegas1: list[Tensor] = []
        omegas2: list[Tensor] = []
        states: list[Tensor] = []
        blank_f = (~is_clue).to(dtype=x.dtype)
        blank_denom = blank_f.sum().clamp_min(1.0)
        v_tan_sqs: list[Tensor] = []

        a = incidence
        al = alphas
        for _ in range(self.num_steps):
            # --- step_all_T: spheres → tori ---
            m_s0 = self.s0.read(x)
            m_s1 = self.s1.read(s1)
            pkt_t1 = (
                self.router.packet_s2t(a['S0_T1']['A_s2t'], m_s0, al['S0_T1'])
                + self.router.packet_s2t(a['S1_T1']['A_s2t'], m_s1, al['S1_T1'])
            )
            pkt_t2 = (
                self.router.packet_s2t(a['S0_T2']['A_s2t'], m_s0, al['S0_T2'])
                + self.router.packet_s2t(a['S1_T2']['A_s2t'], m_s1, al['S1_T2'])
            )
            y1, w1 = self.t1.write(y1, pkt_t1)
            y2, w2 = self.t2.write(y2, pkt_t2)
            omegas1.append(w1)
            omegas2.append(w2)

            # --- step_all_S: tori → spheres (active-mass μ,σ) ---
            m_t1 = self.t1.read(y1)
            m_t2 = self.t2.read(y2)
            pkt_s0 = (
                self.router.packet_t2s(a['S0_T1']['P'], m_t1, al['S0_T1'])
                + self.router.packet_t2s(a['S0_T2']['P'], m_t2, al['S0_T2'])
            )
            pkt_s1 = (
                self.router.packet_t2s(a['S1_T1']['P'], m_t1, al['S1_T1'])
                + self.router.packet_t2s(a['S1_T2']['P'], m_t2, al['S1_T2'])
            )
            x, _gate, v_s0 = self.s0.write(x, pkt_s0, pe, x0, is_clue, v_prev=v_s0)
            s1, _, v_s1 = self.s1.write(s1, pkt_s1, None, None, None, v_prev=v_s1)
            v_sq = v_s0.square().sum(dim=-1)
            v_tan_sqs.append((v_sq * blank_f).sum() / blank_denom)
            states.append(x)

        return {
            'x': x,
            's1': s1,
            'y1': y1,
            'y2': y2,
            'omegas1': omegas1,
            'omegas2': omegas2,
            'states': states,
            'v_tan_sqs': v_tan_sqs,
        }


def universal_clash_loss(x: Tensor, alpha_s0_t1: Tensor, alpha_s0_t2: Tensor) -> Tensor:
    """
    Board-mean-centered, detached universal clash × α² (v3.1 landmine 1).

    ``x̃ = normalize(x − mean_board(x)).detach()``; ``C_ij = relu(x̃_i·x̃_j)²``
    (diag 0). Trains Φ/α only — not cell answers. Strips NP + shared scratch PE
    without ETF poles inside the bank.
    """
    x_bar = x.mean(dim=1, keepdim=True)
    x_t = F.normalize(x - x_bar, dim=-1, eps=1e-6).detach()
    grams = torch.einsum('bid,bjd->bij', x_t, x_t)
    grams = grams - torch.diag_embed(torch.diagonal(grams, dim1=-2, dim2=-1))
    raw = F.relu(grams).square().mean()
    a2 = alpha_s0_t1.square() + alpha_s0_t2.square()
    return raw * a2


def competition_loss(
    alphas: dict[str, Tensor],
    curr_gamma: Tensor,
    lam: float = _LAMBDA_COMP,
) -> Tensor:
    """``L_comp = λ · γ · (α_S0T1 α_S0T2 + α_S1T1 α_S1T2)``."""
    return lam * curr_gamma * (
        alphas['S0_T1'] * alphas['S0_T2'] + alphas['S1_T1'] * alphas['S1_T2']
    )


def h_repel_loss(h_t1: Tensor, h_t2: Tensor, lam: float = _LAMBDA_REPEL) -> Tensor:
    """``L_h-repel = λ · exp(−(h_T1 − h_T2)²)``."""
    return lam * (-(h_t1 - h_t2).square()).exp()


def pooled_top3_purity(p_t1: Tensor, p_t2: Tensor, peer_mask: Tensor) -> Tensor:
    """Purity of the 3 highest per-head-purity families across T1∪T2."""
    p_all = torch.cat([p_t1, p_t2], dim=0)  # (6, H, N)
    scores = []
    for k in range(p_all.shape[0]):
        scores.append(sudoku_graph_purity(p_all[k : k + 1], peer_mask))
    stacked = torch.stack(scores)
    top_idx = stacked.topk(min(3, stacked.numel()), largest=True).indices
    return sudoku_graph_purity(p_all[top_idx], peer_mask)


class MultiBankSolver(nn.Module):
    """
    Phase-3 façade: S0/S1 spheres + T1/T2 F=3 tori, MultiBankRouter, static leapfrog.

    Compile-safe 0-D buffers: ``curr_tau``, ``curr_gamma``, ``curr_ent_weight``.
    """

    def __init__(
        self,
        num_steps: int = K_DEFAULT,
        d_s0: int = D,
        d_s1: int = 32,
        m: int = M,
        d_msg: int = _D_MSG,
        pe_alpha: float = PE_ALPHA,
        pe_mode: str = 't2_grid',
        sigma_init: float = 0.5,
        ortho_weight: float = LAMBDA_ORTHO_DEFAULT,
        ent_weight: float = LAMBDA_ENT_DEFAULT,
        clash_weight: float = LAMBDA_CLASH_DEFAULT,
        beta_l1_weight: float = _LAMBDA_BETA_L1,
        beta_l0_weight: float = _LAMBDA_BETA_L0,
        comp_weight: float = _LAMBDA_COMP,
        repel_weight: float = _LAMBDA_REPEL,
        ortho_margin: float = ORTHO_MARGIN,
        ce_gamma: float = CE_GAMMA,
        ce_rho: float = _CE_RHO,
        supervise_all_steps: bool = True,
        omega_weight: float = 0.0,
        gamma_max: float = _GAMMA_MAX,
    ) -> None:
        super().__init__()
        self.mixer = 'multibank_attractor'
        self.use_gauge = False
        self.num_steps = int(num_steps)
        self.d = d_s0
        self.m = m
        self.pe_alpha = pe_alpha
        self.pe_mode = pe_mode
        self.routing_mode = 'soft_sinkhorn'
        self.ortho_weight = float(ortho_weight)
        self.ent_weight = float(ent_weight)
        self.clash_weight = float(clash_weight)
        self.beta_l1_weight = float(beta_l1_weight)
        self.beta_l0_weight = float(beta_l0_weight)
        self.comp_weight = float(comp_weight)
        self.repel_weight = float(repel_weight)
        self.ortho_margin = float(ortho_margin)
        self.ce_gamma = float(ce_gamma)
        self.ce_rho = float(ce_rho)
        self.supervise_all_steps = bool(supervise_all_steps)
        self.omega_weight = float(omega_weight)
        self.gamma_max = float(gamma_max)
        # train.py summary shims
        self.mlp_activation = 'silu'
        self.mlp_layers = 1
        self.mlp_width = 'default'
        self.kinetic_weight = 0.0
        self.gauge_kinetic_weight = 0.0
        self.constraint_weight = 0.0
        self.d_model = d_msg
        self.dt = 1.0

        self.s0 = SphereBank(81, d_s0, d_msg=d_msg, pe_dim=PE_DIM, ungated_sem=True, h_x=432)
        self.s1 = SphereBank(9, d_s1, d_msg=d_msg, pe_dim=0, ungated_sem=False, h_x=128)
        self.t1 = TorusBank(n_per=_N_PER, m=m, f_heads=_F_HEADS, d_msg=d_msg, h_y=320)
        self.t2 = TorusBank(n_per=_N_PER, m=m, f_heads=_F_HEADS, d_msg=d_msg, h_y=320)
        self.router = MultiBankRouter(n_s0=81, n_s1=9, sigma_init=sigma_init)
        self.core = MultiBankAttractorCore(
            self.s0, self.s1, self.t1, self.t2, self.router, num_steps=num_steps
        )
        self.logit_scale = nn.Parameter(torch.tensor(float(LOGIT_SCALE_INIT)))

        self.register_buffer('poles', simplex_etf_poles(N_DIGITS, d_s0), persistent=True)
        self.register_buffer('pe', build_pe(pe_mode, PE_DIM), persistent=True)
        self.register_buffer('house_cell_idx', house_cell_idx(), persistent=True)
        self.register_buffer('cell_house_idx', cell_house_idx(), persistent=True)
        self.register_buffer('gt_peer_mask', gt_peer_mask(), persistent=True)
        self.register_buffer('curr_tau', torch.tensor(float(TAU_INIT)), persistent=True)
        self.register_buffer('curr_gamma', torch.tensor(0.0), persistent=True)
        self.register_buffer('curr_ent_weight', torch.tensor(0.0), persistent=True)
        # S1 north-pole template (d_s1); expanded per batch in forward.
        self.register_buffer('s1_pole', _s1_north(d_s1), persistent=True)
        self.adapter = SudokuProblemAdapter(
            self.poles,
            self.pe,
            pe_alpha=pe_alpha,
            house_cells=self.house_cell_idx,
            cell_houses=self.cell_house_idx,
        )

    def vf_param_count(self) -> int:
        """Trainable params excluding the MultiBankRouter block."""
        n = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('router.'):
                continue
            n += p.numel()
        return n

    def router_param_count(self) -> int:
        return sum(p.numel() for p in self.router.parameters() if p.requires_grad)

    def _logits(self, x: Tensor) -> Tensor:
        scale = self.logit_scale.clamp(max=LOGIT_SCALE_MAX)
        return scale * (x[..., :N_DIGITS] @ self.poles[:, :N_DIGITS].T)

    def integrate(self, puzzle: Tensor) -> dict[str, Tensor | list[Tensor]]:
        x0, is_clue, pe, _a, _b, poles, _y_unused = self.adapter.encode(
            puzzle, poles=self.poles, pe=self.pe
        )
        b = puzzle.shape[0]
        device, dtype = x0.device, x0.dtype
        s1_0 = self.s1_pole.to(device=device, dtype=dtype).expand(b, 9, -1).contiguous().clone()
        y1_0 = torch.full((b, self.t1.n, self.m), math.pi / 4, device=device, dtype=dtype)
        y2_0 = torch.full((b, self.t2.n, self.m), math.pi / 4, device=device, dtype=dtype)
        incidence = self.router.get_incidence(self.curr_tau)
        alphas = self.router.alphas(self.curr_gamma)
        out = self.core.integrate(
            x0, s1_0, y1_0, y2_0, is_clue, pe, incidence, alphas
        )
        out['incidence'] = incidence
        out['alphas'] = alphas
        out['is_clue'] = is_clue
        return out

    @staticmethod
    def _blank_ce(logits: Tensor, solution: Tensor, mask: Tensor, label_smoothing: float) -> Tensor:
        flat_logits = logits.reshape(-1, N_DIGITS)
        flat_target = (solution - 1).clamp(0, N_DIGITS - 1).reshape(-1)
        flat_mask = mask.reshape(-1)
        per = F.cross_entropy(
            flat_logits,
            flat_target,
            reduction='none',
            label_smoothing=label_smoothing,
        )
        return (per * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)

    def forward(
        self,
        puzzle: Tensor,
        solution: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        pack = self.integrate(puzzle)
        x = pack['x']
        assert isinstance(x, Tensor)
        logits = self._logits(x)
        pred = logits.argmax(dim=-1) + 1
        omegas1 = pack['omegas1']
        assert isinstance(omegas1, list)
        omega_k = omegas1[-1]
        omega1 = omegas1[0]
        blank_np = x[..., BLANK_COORD].square().mean()
        p9 = self.poles[:, :N_DIGITS]
        q = F.relu(x[..., :N_DIGITS] @ p9.T)
        claim_energy = q.square().mean()
        scratch_energy = x[..., SCRATCH_START:].square().mean()
        alphas = pack['alphas']
        assert isinstance(alphas, dict)
        incidence = pack['incidence']
        assert isinstance(incidence, dict)

        out: dict[str, Tensor] = {
            's': x,
            'x': x,
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'pred': pred,
            'velocities': omega_k,
            'omega_sq_1': omega1.square().mean().detach(),
            'omega_sq_k': omega_k.square().mean().detach(),
            'scratch_energy': scratch_energy.detach(),
            'blank_np': blank_np.detach(),
            'claim_energy': claim_energy.detach(),
            'logit_scale': self.logit_scale.detach().clamp(max=LOGIT_SCALE_MAX),
            'routing_tau': self.curr_tau.detach(),
            'curr_gamma': self.curr_gamma.detach(),
            'alpha_S0_T1': alphas['S0_T1'].detach(),
            'alpha_S0_T2': alphas['S0_T2'].detach(),
            'alpha_S1_T1': alphas['S1_T1'].detach(),
            'alpha_S1_T2': alphas['S1_T2'].detach(),
            'h_T1': self.router.h_T1.detach(),
            'h_T2': self.router.h_T2.detach(),
            'h_S1': self.router.h_S1.detach(),
            'h_S0': self.router.h_S0.detach(),
        }

        # Intra-bank ortho only (v3.1: drop inter_bank_ortho).
        p_t1 = incidence['S0_T1']['P']
        p_t2 = incidence['S0_T2']['P']
        ortho_intra = ortho_hinge(p_t1, margin=self.ortho_margin) + ortho_hinge(
            p_t2, margin=self.ortho_margin
        )
        # α-gate intra ortho lightly (keep pressure on active banks).
        ortho_gate = (alphas['S0_T1'] + alphas['S0_T2']).clamp_min(1e-6)
        ortho_term = ortho_intra * (0.5 * ortho_gate)
        clash = universal_clash_loss(x, alphas['S0_T1'], alphas['S0_T2'])
        l_comp = competition_loss(alphas, self.curr_gamma, lam=self.comp_weight)
        l_repel = h_repel_loss(self.router.h_T1, self.router.h_T2, lam=self.repel_weight)
        ent = column_entropy(p_t1) + column_entropy(p_t2)
        sig_betas = torch.stack(
            [
                torch.sigmoid(self.router.beta_S0_T1),
                torch.sigmoid(self.router.beta_S0_T2),
                torch.sigmoid(self.router.beta_S1_T1),
                torch.sigmoid(self.router.beta_S1_T2),
            ]
        )
        beta_l1 = sig_betas.sum()
        beta_l0 = (sig_betas * (1.0 - sig_betas)).sum()

        with torch.no_grad():
            pur1 = sudoku_graph_purity(p_t1, self.gt_peer_mask)
            pur2 = sudoku_graph_purity(p_t2, self.gt_peer_mask)
            gram1, _ = family_gram_sims(p_t1)
            gram2, _ = family_gram_sims(p_t2)
            out['sudoku_graph_purity'] = torch.maximum(pur1, pur2)
            out['sudoku_graph_purity_T1'] = pur1
            out['sudoku_graph_purity_T2'] = pur2
            out['pooled_top3_purity'] = pooled_top3_purity(p_t1, p_t2, self.gt_peer_mask)
            out['family_gram_sim'] = 0.5 * (gram1 + gram2)
            out['routing_entropy'] = (0.5 * (column_entropy(p_t1) + column_entropy(p_t2))).detach()
            out['ortho_loss'] = ortho_term.detach()
            out['routing_clash'] = clash.detach()
            out['ent_loss'] = ent.detach()
            out['beta_l1'] = beta_l1.detach()
            out['beta_l0'] = beta_l0.detach()
            out['l_comp'] = l_comp.detach()
            out['l_h_repel'] = l_repel.detach()

        if solution is None:
            return out

        if mask is None:
            mask = (puzzle == 0).float()
        label_smoothing = max(0.0, 1.0 - self.ce_gamma)
        states = pack['states']
        assert isinstance(states, list)
        # v3.1: exponentially weighted blank CE over K steps, ρ=0.5.
        ces = torch.stack(
            [self._blank_ce(self._logits(st), solution, mask, label_smoothing) for st in states]
        )
        k = ces.shape[0]
        t = torch.arange(k, device=ces.device, dtype=ces.dtype)
        w = self.ce_rho ** (k - 1 - t)
        task_loss = (ces * w).sum() / w.sum()

        omega_kin = omega_k.square().mean()
        loss = (
            task_loss
            + self.omega_weight * omega_kin
            + self.ortho_weight * ortho_term
            + self.curr_ent_weight * ent
            + self.clash_weight * clash
            + self.beta_l1_weight * beta_l1
            + self.beta_l0_weight * beta_l0
            + l_comp
            + l_repel
        )
        out.update(
            {
                'loss': loss,
                'task_loss': task_loss.detach(),
                'kinetic_loss': omega_kin.detach(),
                'gauge_kinetic_loss': omega_kin.new_zeros(()),
                'constraint_loss': omega_kin.new_zeros(()),
            }
        )
        return out


def _s1_north(d: int) -> Tensor:
    """Unit e_0 in R^d for S1 blank init."""
    x = torch.zeros(d)
    x[0] = 1.0
    return x


def gamma_anneal(step: int, gamma_max: float = _GAMMA_MAX, n: int = _GAMMA_ANNEAL_STEPS) -> float:
    """``γ: 0 → γ_max`` over first ``n`` steps (compile-safe fill from train)."""
    return float(gamma_max) * min(1.0, float(step) / float(max(n, 1)))
