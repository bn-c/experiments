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
    cell_house_idx,
    house_cell_idx,
    init_bipartite_state,
    simplex_etf_poles,
    torus_t3_pe,
)

K_DEFAULT = 8
LOGIT_SCALE_INIT = 12.0
LOGIT_SCALE_MAX = 50.0
CE_GAMMA = 0.9  # → label_smoothing = 1 - γ
LAMBDA_OMEGA_DEFAULT = 0.0


class BipartiteSnTmSolver(nn.Module):
    """
    Shared VF, leapfrog ``Y→X``, ``K`` geodesic / torus steps.

    Readout: ``normalize(x[..., :9])``; ``logits = logit_scale · (x_dec @ P[:, :9].T)``
    with ``logit_scale`` init 12, clamp ≤50. Train CE with γ=0.9; ``λ_ω=0`` (log only).
    """

    def __init__(
        self,
        num_steps: int = K_DEFAULT,
        d: int = D,
        m: int = M,
        h_y: int = 320,
        h_x: int = 448,
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
        self.logit_scale = nn.Parameter(torch.tensor(float(LOGIT_SCALE_INIT)))

        self.register_buffer('poles', simplex_etf_poles(N_DIGITS, d), persistent=True)
        self.register_buffer('pe', torus_t3_pe(PE_DIM), persistent=True)
        self.register_buffer('house_cell_idx', house_cell_idx(), persistent=True)
        self.register_buffer('cell_house_idx', cell_house_idx(), persistent=True)

    def _logits(self, x: Tensor) -> Tensor:
        x_dec = F.normalize(x[..., :N_DIGITS], dim=-1, eps=1e-6)
        scale = self.logit_scale.clamp(max=LOGIT_SCALE_MAX)
        p9 = self.poles[:, :N_DIGITS]
        return scale * (x_dec @ p9.T)

    def integrate(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Leapfrog ``K`` steps: ω on houses, then geodesic on cells.

        Returns final ``x``, per-step ω list, per-step post-update ``x`` states.
        """
        x, y, is_clue = init_bipartite_state(
            puzzle, self.poles, self.pe, pe_alpha=self.pe_alpha
        )
        x0 = x
        omegas: list[Tensor] = []
        states: list[Tensor] = []
        for _ in range(self.num_steps):
            omega = self.vf.omega(x, y, self.poles, self.house_cell_idx)
            omegas.append(omega)
            y = torus_step(y, omega)
            u = self.vf.ambient_u(x, y, self.pe, self.cell_house_idx)
            x = geodesic_step(x, u, x0, is_clue)
            states.append(x)
        self._last_omegas = omegas
        self._last_is_clue = is_clue
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
        # Watch-only: cell-level claim energy (mirrors scratch_energy; house_features math unchanged).
        p9 = self.poles[:, :N_DIGITS]
        q = F.relu(x[..., :N_DIGITS] @ p9.T)
        claim_energy = q.square().mean()

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
        if solution is None:
            return out

        if mask is None:
            mask = (puzzle == 0).float()
        label_smoothing = max(0.0, 1.0 - self.ce_gamma)
        if self.supervise_all_steps:
            task_loss = torch.stack(
                [self._blank_ce(self._logits(st), solution, mask, label_smoothing) for st in states]
            ).mean()
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
