"""Bipartite ``S^{63} ↔ T^{16}`` Sudoku solver (leapfrog Y→X, K steps)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.coupled_vf import CoupledVectorField
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
    cell_house_idx,
    hard_incidence_from_indices,
    house_cell_idx,
    simplex_etf_poles,
    torus_t3_pe,
)

K_DEFAULT = 8
LOGIT_SCALE_INIT = 4.0
LOGIT_SCALE_MAX = 10.0
CE_GAMMA = 0.9  # → label_smoothing = 1 - γ
LAMBDA_OMEGA_DEFAULT = 0.0


class AttractorCore:
    """
    Riemannian factor-graph leapfrog: Kuramoto ω then geodesic Exp, K steps.

    Incidence is hard ``A_s2t`` / ``A_t2s`` (static for all K). Clue freeze is
    hard ``where(is_clue, x0, ·)``. Not an ``nn.Module`` — holds a ref to the
    shared ``CoupledVectorField`` so the Sudoku façade keeps the param lock.
    Gather reference path stays on ``CoupledVectorField``.
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
    ) -> tuple[Tensor, list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """
        Leapfrog ``K`` steps on hard incidence.

        Returns ``(x_K, omegas, states, v_tan_sqs, scratch_gates)``.
        """
        x, y = x0, y0
        omegas: list[Tensor] = []
        states: list[Tensor] = []
        v_tan_sqs: list[Tensor] = []
        scratch_gates: list[Tensor] = []
        blank_f = (~is_clue).to(dtype=x.dtype)
        blank_denom = blank_f.sum().clamp_min(1.0)
        for _ in range(self.num_steps):
            omega = self.vf.omega_hard(x, y, poles, a_s2t)
            omegas.append(omega)
            y = torus_step(y, omega)
            u, scratch_gate = self.vf.ambient_u_hard(x, y, pe, a_t2s, poles)
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


class BipartiteSnTmSolver(nn.Module):
    """
    Thin Sudoku façade: ``SudokuProblemAdapter`` + ``AttractorCore`` (hard-A).

    Readout: unnormalized ``logits = logit_scale · (x[..., :9] @ P[:, :9].T)``
    with ``logit_scale`` init 4, clamp ≤10. Train CE with γ=0.9; ``λ_ω=0`` (log only).
    Param lock **687 010**.
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
    ) -> None:
        super().__init__()
        self.mixer = 'bipartite_sn_tm'
        self.use_gauge = False
        self.num_steps = int(num_steps)
        self.d = d
        self.m = m
        self.pe_alpha = pe_alpha
        self.omega_weight = float(omega_weight)
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
        self.register_buffer('pe', torus_t3_pe(PE_DIM), persistent=True)
        self.register_buffer('house_cell_idx', house_cell_idx(), persistent=True)
        self.register_buffer('cell_house_idx', cell_house_idx(), persistent=True)
        a_s2t, a_t2s = hard_incidence_from_indices(
            self.house_cell_idx, self.cell_house_idx
        )
        self.register_buffer('a_s2t', a_s2t, persistent=True)
        self.register_buffer('a_t2s', a_t2s, persistent=True)
        self.adapter = SudokuProblemAdapter(
            self.poles,
            self.pe,
            pe_alpha=pe_alpha,
            house_cells=self.house_cell_idx,
            cell_houses=self.cell_house_idx,
        )

    def _logits(self, x: Tensor) -> Tensor:
        scale = self.logit_scale.clamp(max=LOGIT_SCALE_MAX)
        return scale * (x[..., :N_DIGITS] @ self.poles[:, :N_DIGITS].T)

    def integrate(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Leapfrog ``K`` steps via hard-A ``AttractorCore``.

        Returns final ``x``, per-step ω list, per-step post-update ``x`` states.
        Stashes blank-only ``v_tan_sq`` and ``scratch_gate`` diags on ``self``.
        """
        x0, is_clue, pe, a_s2t, a_t2s, poles, y0 = self.adapter.encode(puzzle)
        # Prefer registered buffers (same values; device-safe under DDP/compile).
        x, omegas, states, v_tan_sqs, scratch_gates = self.core.integrate(
            x0,
            y0,
            is_clue,
            self.pe,
            self.a_s2t,
            self.a_t2s,
            self.poles,
        )
        del pe, a_s2t, a_t2s, poles  # adapter copies; buffers used above
        self._last_omegas = omegas
        self._last_is_clue = is_clue
        self._last_v_tan_sqs = v_tan_sqs
        self._last_scratch_gates = scratch_gates
        return x, omegas, states

    def integrate_gather(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Gather reference integrate (parity tests); shares ``vf`` weights."""
        x0, is_clue, pe, _a_s2t, _a_t2s, poles, y0 = self.adapter.encode(puzzle)
        x, omegas, states, v_tan_sqs, scratch_gates = self.core.integrate_gather(
            x0,
            y0,
            is_clue,
            self.pe,
            self.house_cell_idx,
            self.cell_house_idx,
            self.poles,
        )
        del pe, poles
        self._last_omegas = omegas
        self._last_is_clue = is_clue
        self._last_v_tan_sqs = v_tan_sqs
        self._last_scratch_gates = scratch_gates
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
        # Watch-only: unnormalized claims (matches house_features + unnorm _logits).
        p9 = self.poles[:, :N_DIGITS]
        q = F.relu(x[..., :N_DIGITS] @ p9.T)
        claim_energy = q.square().mean()
        v_tan_sqs = getattr(self, '_last_v_tan_sqs', None)
        scratch_gates = getattr(self, '_last_scratch_gates', None)

        out: dict[str, Tensor] = {
            's': x,  # train/eval expect 's'; bipartite state is x on S^{63}
            'x': x,
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'pred': pred,
            'velocities': omega_k,  # placeholder for kinetic log path
            'omega_sq_1': omega1.square().mean().detach(),
            'omega_sq_k': omega_k.square().mean().detach(),
            'scratch_energy': scratch_energy.detach(),
            'blank_np': blank_np.detach(),
            'claim_energy': claim_energy.detach(),
            'logit_scale': self.logit_scale.detach().clamp(max=LOGIT_SCALE_MAX),
        }
        if v_tan_sqs:
            out['v_tan_sq_1'] = v_tan_sqs[0].detach()
            out['v_tan_sq_k'] = v_tan_sqs[-1].detach()
        if scratch_gates:
            out['scratch_gate_mean'] = scratch_gates[-1].mean().detach()
        if solution is None:
            return out

        if mask is None:
            mask = (puzzle == 0).float()
        label_smoothing = max(0.0, 1.0 - self.ce_gamma)
        if self.supervise_all_steps:
            ces = torch.stack(
                [self._blank_ce(self._logits(st), solution, mask, label_smoothing) for st in states]
            )
            # Linear ramp: w_k = k / Σj (favor later ODE steps over uniform mean).
            k = ces.shape[0]
            w = torch.arange(1, k + 1, device=ces.device, dtype=ces.dtype)
            task_loss = (ces * w).sum() / w.sum()
        else:
            task_loss = self._blank_ce(logits, solution, mask, label_smoothing)

        omega_kin = omega_k.square().mean()
        loss = task_loss + self.omega_weight * omega_kin
        out.update(
            {
                'loss': loss,
                'task_loss': task_loss.detach(),
                'kinetic_loss': omega_kin.detach(),  # ||ω_K||² (λ_ω may be 0)
                'gauge_kinetic_loss': omega_kin.new_zeros(()),
                'constraint_loss': omega_kin.new_zeros(()),
            }
        )
        return out
