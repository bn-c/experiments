"""Bipartite ``S^{63} ↔ T^{16}`` Sudoku solver (leapfrog Y→X, K steps)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.coupled_vf import BipartiteRouter, CoupledVectorField
from riemannian_sudoku.geodesic import geodesic_step, torus_step
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
    gt_peer_mask,
    hard_incidence_from_indices,
    house_cell_idx,
    simplex_etf_poles,
)

K_DEFAULT = 8
LOGIT_SCALE_INIT = 4.0
LOGIT_SCALE_MAX = 10.0
CE_GAMMA = 0.9  # → label_smoothing = 1 - γ
LAMBDA_OMEGA_DEFAULT = 0.0
LAMBDA_ORTHO_DEFAULT = 0.1
LAMBDA_ENT_DEFAULT = 0.005
LAMBDA_CLASH_DEFAULT = 1.0
ORTHO_MARGIN = 0.25
TAU_INIT = 0.5
TAU_MIN = 0.02
PURITY_TOPK = 20


class AttractorCore:
    """
    Riemannian factor-graph leapfrog: Kuramoto ω then geodesic Exp, K steps.

    Incidence is static for all K (hard buffers or soft Sinkhorn once per
    forward). Clue freeze is hard ``where(is_clue, x0, ·)``. Not an
    ``nn.Module`` — holds a ref to the shared ``CoupledVectorField``.
    """

    def __init__(self, vf: CoupledVectorField, num_steps: int = K_DEFAULT) -> None:
        self.vf = vf
        self.num_steps = int(num_steps)

    def integrate(
        self,
        x0: Tensor,
        y0: Tensor,
        is_clue: Tensor,
        pe: Tensor,
        a_s2t: Tensor,
        a_t2s: Tensor,
        poles: Tensor,
        *,
        soft: bool = False,
        permute_families: bool = False,
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Leapfrog ``K`` steps. Returns ``(x_K, omegas, states, v_tan_sqs, scratch_gates)``."""
        x, y = x0, y0
        omegas: list[Tensor] = []
        states: list[Tensor] = []
        v_tan_sqs: list[Tensor] = []
        scratch_gates: list[Tensor] = []
        blank_f = (~is_clue).to(dtype=x.dtype)
        blank_denom = blank_f.sum().clamp_min(1.0)
        # FiLM family-slot shuffle (training+soft only): stop pinned sharp rows/cols
        # from muting the learnable box slot permanently.
        a_t2s_u = a_t2s
        if permute_families and a_t2s.shape[1] == 3:
            perm = torch.randperm(3, device=a_t2s.device)
            a_t2s_u = a_t2s[:, perm, :]
        for _ in range(self.num_steps):
            omega = self.vf.omega_hard(x, y, poles, a_s2t, soft=soft)
            omegas.append(omega)
            y = torus_step(y, omega)
            u, scratch_gate = self.vf.ambient_u_hard(x, y, pe, a_t2s_u, poles)
            scratch_gates.append(scratch_gate)
            v_tan = u - (x * u).sum(dim=-1, keepdim=True) * x
            v_sq = v_tan.square().sum(dim=-1)
            v_tan_sqs.append((v_sq * blank_f).sum() / blank_denom)
            x = geodesic_step(x, u, x0, is_clue)
            states.append(x)
        return x, omegas, states, v_tan_sqs, scratch_gates

    def integrate_gather(
        self,
        x0: Tensor,
        y0: Tensor,
        is_clue: Tensor,
        pe: Tensor,
        house_cells: Tensor,
        cell_houses: Tensor,
        poles: Tensor,
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Gather reference leapfrog (parity tests); same ``vf`` weights."""
        x, y = x0, y0
        omegas: list[Tensor] = []
        states: list[Tensor] = []
        v_tan_sqs: list[Tensor] = []
        scratch_gates: list[Tensor] = []
        blank_f = (~is_clue).to(dtype=x.dtype)
        blank_denom = blank_f.sum().clamp_min(1.0)
        for _ in range(self.num_steps):
            omega = self.vf.omega(x, y, poles, house_cells)
            omegas.append(omega)
            y = torus_step(y, omega)
            u, scratch_gate = self.vf.ambient_u(x, y, pe, cell_houses, poles)
            scratch_gates.append(scratch_gate)
            v_tan = u - (x * u).sum(dim=-1, keepdim=True) * x
            v_sq = v_tan.square().sum(dim=-1)
            v_tan_sqs.append((v_sq * blank_f).sum() / blank_denom)
            x = geodesic_step(x, u, x0, is_clue)
            states.append(x)
        return x, omegas, states, v_tan_sqs, scratch_gates


def routing_tau(step: int, t_max: int, tau_init: float = TAU_INIT, tau_min: float = TAU_MIN) -> float:
    """``τ(t) = max(τ_min, τ_init · (τ_min/τ_init)^(t/(0.8 T_max)))``."""
    t_max = max(int(t_max), 1)
    ratio = float(step) / (0.8 * float(t_max))
    return max(tau_min, tau_init * (tau_min / tau_init) ** ratio)


def family_gram_sims(p_t2s: Tensor) -> tuple[Tensor, Tensor]:
    """
    Off-diag cell-cell Gram cosines between families.

    ``G_k = P[k]^T P[k]`` with diag zeroed; pairwise cosine of flattened
    off-diag. GT: row/col≈0, row/box≈0.25, col/box≈0.25 → mean ≈0.167.
    Returns ``(mean_sim, pairwise (3,))``.
    """
    grams: list[Tensor] = []
    for k in range(p_t2s.shape[0]):
        g = p_t2s[k].T @ p_t2s[k]  # (81, 81)
        g = g - torch.diag_embed(torch.diagonal(g))
        grams.append(g.reshape(-1))
    sims: list[Tensor] = []
    for i in range(3):
        for j in range(i + 1, 3):
            ni = grams[i].norm().clamp_min(1e-8)
            nj = grams[j].norm().clamp_min(1e-8)
            sims.append((grams[i] @ grams[j]) / (ni * nj))
    stacked = torch.stack(sims)
    return stacked.mean(), stacked


def ortho_hinge(p_t2s: Tensor, margin: float = ORTHO_MARGIN) -> Tensor:
    """``Σ relu(sim_ij − m)²`` on off-diag family Gram cosines only."""
    _mean, sims = family_gram_sims(p_t2s)
    return F.relu(sims - margin).square().sum()


def claim_clash_loss(p_t2s: Tensor, logits: Tensor) -> Tensor:
    """
    Soft house clash: off-diag cell Gram × detached digit co-claim.

    ``gram_off[k] = P[k]^T P[k]`` with diag zeroed; ``claim_co`` from
    ``softmax(logits.detach())``. GT houses + valid one-hot board → ≈0.
    """
    grams = []
    for k in range(p_t2s.shape[0]):
        g = p_t2s[k].T @ p_t2s[k]  # (81, 81)
        g = g - torch.diag_embed(torch.diagonal(g))
        grams.append(g)
    gram_off = torch.stack(grams, dim=0)  # (3, 81, 81)
    p_det = F.softmax(logits.detach(), dim=-1)  # (B, 81, 9)
    b = p_det.shape[0]
    claim_co = torch.einsum('bid,bjd->ij', p_det, p_det) / float(b)
    return (gram_off * claim_co).sum() / (81.0 * 8.0)


def column_entropy(p_t2s: Tensor) -> Tensor:
    """Mean column entropy of ``P_t2s (3,9,81)`` (nats)."""
    p = p_t2s.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=1).mean()


def sudoku_graph_purity(p_t2s: Tensor, peer_mask: Tensor, topk: int = PURITY_TOPK) -> Tensor:
    """Fraction of top-k soft peers that are true Sudoku row/col/box peers."""
    aff = torch.zeros(81, 81, device=p_t2s.device, dtype=p_t2s.dtype)
    for k in range(p_t2s.shape[0]):
        aff = aff + p_t2s[k].T @ p_t2s[k]
    aff = aff - torch.diag_embed(torch.diagonal(aff))
    idx = aff.topk(topk, dim=-1).indices  # (81, topk)
    rows = torch.arange(81, device=p_t2s.device).unsqueeze(1).expand_as(idx)
    return peer_mask[rows, idx].to(dtype=p_t2s.dtype).mean()


class BipartiteSnTmSolver(nn.Module):
    """
    Thin Sudoku façade: ``SudokuProblemAdapter`` + ``AttractorCore``.

    Phase-1 hard-A default (``routing_mode=hard``, ``pe_mode=t3_full``).
    Phase-2 soft Sinkhorn adds Φ=2187; VF block stays **687 010**.
    Compile-safe τ / entropy via 0-D buffers ``curr_tau`` / ``curr_ent_weight``.
    """

    def __init__(
        self,
        num_steps: int = K_DEFAULT,
        d: int = D,
        m: int = M,
        h_y: int = 320,
        h_x: int = 432,
        pe_alpha: float = PE_ALPHA,
        omega_weight: float = LAMBDA_OMEGA_DEFAULT,
        ce_gamma: float = CE_GAMMA,
        supervise_all_steps: bool = False,
        *,
        pe_mode: str = 't3_full',
        routing_mode: str = 'hard',
        sigma_init: float = 0.5,
        ortho_weight: float = LAMBDA_ORTHO_DEFAULT,
        ent_weight: float = LAMBDA_ENT_DEFAULT,
        clash_weight: float = LAMBDA_CLASH_DEFAULT,
        ortho_margin: float = ORTHO_MARGIN,
    ) -> None:
        super().__init__()
        self.mixer = 'bipartite_sn_tm'
        self.use_gauge = False
        self.num_steps = int(num_steps)
        self.d = d
        self.m = m
        self.pe_alpha = pe_alpha
        self.pe_mode = pe_mode
        self.routing_mode = routing_mode
        self.omega_weight = float(omega_weight)
        self.ortho_weight = float(ortho_weight)
        self.ent_weight = float(ent_weight)
        self.clash_weight = float(clash_weight)
        self.ortho_margin = float(ortho_margin)
        self.ce_gamma = float(ce_gamma)
        self.supervise_all_steps = bool(supervise_all_steps)
        # Compatibility shims for train.py summary fields.
        self.mlp_activation = 'silu'
        self.mlp_layers = 1
        self.mlp_width = 'default'
        self.kinetic_weight = 0.0
        self.gauge_kinetic_weight = 0.0
        self.constraint_weight = 0.0
        self.d_model = h_x
        self.dt = 1.0

        self.vf = CoupledVectorField(d=d, m=m, h_y=h_y, h_x=h_x, pe_dim=PE_DIM)
        self.core = AttractorCore(self.vf, num_steps=num_steps)
        self.logit_scale = nn.Parameter(torch.tensor(float(LOGIT_SCALE_INIT)))

        self.register_buffer('poles', simplex_etf_poles(N_DIGITS, d), persistent=True)
        self.register_buffer('pe', build_pe(pe_mode, PE_DIM), persistent=True)
        self.register_buffer('house_cell_idx', house_cell_idx(), persistent=True)
        self.register_buffer('cell_house_idx', cell_house_idx(), persistent=True)
        a_s2t, a_t2s = hard_incidence_from_indices(
            self.house_cell_idx, self.cell_house_idx
        )
        self.register_buffer('a_s2t', a_s2t, persistent=True)
        self.register_buffer('a_t2s', a_t2s, persistent=True)
        self.router = BipartiteRouter(
            a_s2t, a_t2s, routing_mode=routing_mode, sigma_init=sigma_init
        )
        self.register_buffer('gt_peer_mask', gt_peer_mask(), persistent=True)
        # Compile-safe schedule knobs (filled from train.py via .fill_()).
        self.register_buffer('curr_tau', torch.tensor(float(TAU_INIT)), persistent=True)
        self.register_buffer('curr_ent_weight', torch.tensor(0.0), persistent=True)
        self.adapter = SudokuProblemAdapter(
            self.poles,
            self.pe,
            pe_alpha=pe_alpha,
            house_cells=self.house_cell_idx,
            cell_houses=self.cell_house_idx,
        )

    def vf_param_count(self) -> int:
        """Trainable params excluding soft router Φ (lock 687 010)."""
        n = 0
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('router.logits'):
                continue
            n += p.numel()
        return n

    def _logits(self, x: Tensor) -> Tensor:
        scale = self.logit_scale.clamp(max=LOGIT_SCALE_MAX)
        return scale * (x[..., :N_DIGITS] @ self.poles[:, :N_DIGITS].T)

    def integrate(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Leapfrog ``K`` steps via hard-A or soft-Sinkhorn incidence.

        Returns final ``x``, per-step ω list, per-step post-update ``x`` states.
        Stashes blank-only ``v_tan_sq``, ``scratch_gate``, and routing tensors.
        """
        soft = self.routing_mode == 'soft_sinkhorn'
        if soft:
            a_s2t, a_t2s, p_t2s = self.router.get_incidence(self.curr_tau)
        else:
            a_s2t, a_t2s = self.a_s2t, self.a_t2s
            p_t2s = None
        x0, is_clue, pe, _a_s2t, _a_t2s, poles, y0 = self.adapter.encode(
            puzzle,
            poles=self.poles,
            pe=self.pe,
            a_s2t=a_s2t,
            a_t2s=a_t2s,
        )
        x, omegas, states, v_tan_sqs, scratch_gates = self.core.integrate(
            x0,
            y0,
            is_clue,
            pe,
            a_s2t,
            a_t2s,
            poles,
            soft=soft,
            permute_families=(self.training and soft),
        )
        self._last_omegas = omegas
        self._last_is_clue = is_clue
        self._last_v_tan_sqs = v_tan_sqs
        self._last_scratch_gates = scratch_gates
        self._last_p_t2s = p_t2s
        self._last_a_s2t = a_s2t
        return x, omegas, states

    def integrate_gather(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Gather reference integrate (parity tests); shares ``vf`` weights."""
        x0, is_clue, pe, _a_s2t, _a_t2s, poles, y0 = self.adapter.encode(
            puzzle,
            poles=self.poles,
            pe=self.pe,
            a_s2t=self.a_s2t,
            a_t2s=self.a_t2s,
        )
        x, omegas, states, v_tan_sqs, scratch_gates = self.core.integrate_gather(
            x0,
            y0,
            is_clue,
            pe,
            self.house_cell_idx,
            self.cell_house_idx,
            poles,
        )
        self._last_omegas = omegas
        self._last_is_clue = is_clue
        self._last_v_tan_sqs = v_tan_sqs
        self._last_scratch_gates = scratch_gates
        self._last_p_t2s = None
        return x, omegas, states

    @staticmethod
    def _blank_ce(logits: Tensor, solution: Tensor, mask: Tensor, label_smoothing: float) -> Tensor:
        """Mean blank-cell CE; ``solution`` digits 1..9 → class 0..8.

        Masked mean (no ``.item()`` / bool branch) so ``torch.compile`` stays
        Inductor-friendly.
        """
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
        x, omegas, states = self.integrate(puzzle)
        logits = self._logits(x)
        pred = logits.argmax(dim=-1) + 1

        omega1 = omegas[0]
        omega_k = omegas[-1]
        scratch_energy = x[..., SCRATCH_START:].square().mean()
        blank_np = x[..., BLANK_COORD].square().mean()
        p9 = self.poles[:, :N_DIGITS]
        q = F.relu(x[..., :N_DIGITS] @ p9.T)
        claim_energy = q.square().mean()
        v_tan_sqs = getattr(self, '_last_v_tan_sqs', None)
        scratch_gates = getattr(self, '_last_scratch_gates', None)
        p_t2s = getattr(self, '_last_p_t2s', None)

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
        }
        if v_tan_sqs:
            out['v_tan_sq_1'] = v_tan_sqs[0].detach()
            out['v_tan_sq_k'] = v_tan_sqs[-1].detach()
        if scratch_gates:
            out['scratch_gate_mean'] = scratch_gates[-1].mean().detach()

        ortho_loss = omega_k.new_zeros(())
        ent_loss = omega_k.new_zeros(())
        clash_loss = omega_k.new_zeros(())
        if p_t2s is not None:
            with torch.no_grad():
                gram_mean, _ = family_gram_sims(p_t2s)
                ent = column_entropy(p_t2s)
                purity = sudoku_graph_purity(p_t2s, self.gt_peer_mask)
                load = p_t2s.sum(dim=-1)  # (3,9) house mass
                out['family_gram_sim'] = gram_mean.detach()
                out['routing_entropy'] = ent.detach()
                out['sudoku_graph_purity'] = purity.detach()
                out['house_load_std'] = load.std().detach()
            ortho_loss = ortho_hinge(p_t2s, margin=self.ortho_margin)
            ent_loss = column_entropy(p_t2s)
            clash_loss = claim_clash_loss(p_t2s, logits)
            out['ortho_loss'] = ortho_loss.detach()
            out['ent_loss'] = ent_loss.detach()
            out['routing_clash'] = clash_loss.detach()

        if solution is None:
            return out

        if mask is None:
            mask = (puzzle == 0).float()
        label_smoothing = max(0.0, 1.0 - self.ce_gamma)
        if self.supervise_all_steps:
            ces = torch.stack(
                [self._blank_ce(self._logits(st), solution, mask, label_smoothing) for st in states]
            )
            k = ces.shape[0]
            w = torch.arange(1, k + 1, device=ces.device, dtype=ces.dtype)
            task_loss = (ces * w).sum() / w.sum()
        else:
            task_loss = self._blank_ce(logits, solution, mask, label_smoothing)

        omega_kin = omega_k.square().mean()
        # curr_ent_weight is 0-D buffer (compile-safe); λ_ortho / λ_clash on when soft.
        loss = (
            task_loss
            + self.omega_weight * omega_kin
            + self.ortho_weight * ortho_loss
            + self.curr_ent_weight * ent_loss
            + self.clash_weight * clash_loss
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
