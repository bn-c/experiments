"""Bipartite product ``S^8 × S^{54} ↔ T^{16}`` Sudoku solver (leapfrog Y→X, K steps)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.coupled_vf import CoupledVectorField
from riemannian_sudoku.geodesic import geodesic_step, torus_step
from riemannian_sudoku.manifold_embed import (
    BLANK_COORD,
    COORD_REST,
    DECISION_DIM,
    M,
    N_DIGITS,
    PE_ALPHA,
    PE_DIM,
    cell_house_idx,
    house_cell_idx,
    init_bipartite_state,
    simplex_etf_poles,
    torus_t3_pe,
)

K_DEFAULT = 8
LOGIT_SCALE_INIT = 4.0
LOGIT_SCALE_MAX = 50.0
CE_GAMMA = 0.9  # → label_smoothing = 1 - γ
LAMBDA_OMEGA_DEFAULT = 0.0


class BipartiteSnTmSolver(nn.Module):
    """
    Shared VF, leapfrog ``Y→X``, ``K`` geodesic / torus steps on product cells.

    Readout: ``logits = logit_scale · (x @ P^T)`` with unit ``x ∈ S^8`` (no slice-norm).
    Optional global board torus ``z ∈ T^{16}`` via ``use_global_torus``.
    """

    def __init__(
        self,
        num_steps: int = K_DEFAULT,
        m: int = M,
        h_y: int | None = None,
        h_x: int | None = None,
        pe_alpha: float = PE_ALPHA,
        omega_weight: float = LAMBDA_OMEGA_DEFAULT,
        ce_gamma: float = CE_GAMMA,
        supervise_all_steps: bool = False,
        use_global_torus: bool = True,
    ) -> None:
        super().__init__()
        self.mixer = 'bipartite_sn_tm'
        self.use_gauge = False
        self.num_steps = int(num_steps)
        self.m = m
        self.pe_alpha = pe_alpha
        self.omega_weight = float(omega_weight)
        self.ce_gamma = float(ce_gamma)
        self.supervise_all_steps = bool(supervise_all_steps)
        self.use_global_torus = bool(use_global_torus)
        # Compatibility shims for train.py summary fields.
        self.mlp_activation = 'silu'
        self.mlp_layers = 1
        self.mlp_width = 'default'
        self.kinetic_weight = 0.0
        self.gauge_kinetic_weight = 0.0
        self.constraint_weight = 0.0
        self.dt = 1.0

        self.vf = CoupledVectorField(
            m=m,
            h_y=h_y,
            h_x=h_x,
            pe_dim=PE_DIM,
            use_global_torus=self.use_global_torus,
        )
        self.d_model = self.vf.h_x
        self.logit_scale = nn.Parameter(torch.tensor(float(LOGIT_SCALE_INIT)))

        self.register_buffer('poles', simplex_etf_poles(N_DIGITS, DECISION_DIM), persistent=True)
        self.register_buffer('pe', torus_t3_pe(PE_DIM), persistent=True)
        self.register_buffer('house_cell_idx', house_cell_idx(), persistent=True)
        self.register_buffer('cell_house_idx', cell_house_idx(), persistent=True)

    def _logits(self, x: Tensor) -> Tensor:
        # x is already unit-norm on S^8 — no slice-normalization.
        scale = self.logit_scale.clamp(max=LOGIT_SCALE_MAX)
        return scale * (x @ self.poles.T)

    def integrate(
        self, puzzle: Tensor
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor], list[Tensor | None]]:
        """
        Leapfrog ``K`` steps: ω on houses (+ global), then geodesic on ``x`` and ``c``.

        Returns final ``x``, ``c``, per-step ``ω_y``, post-update ``x`` states, per-step ``ω_z``.
        """
        x, c, y, z, is_clue = init_bipartite_state(
            puzzle,
            self.poles,
            self.pe,
            pe_alpha=self.pe_alpha,
            use_global_torus=self.use_global_torus,
        )
        x0 = x
        c0 = c  # clue anchors on S^{54}; geodesic_step returns new tensors (no clone)
        omegas: list[Tensor] = []
        omega_zs: list[Tensor | None] = []
        states: list[Tensor] = []
        for _ in range(self.num_steps):
            omega_y, omega_z = self.vf.omega(x, c, y, z, self.house_cell_idx)
            omegas.append(omega_y)
            omega_zs.append(omega_z)
            y = torus_step(y, omega_y)
            if z is not None and omega_z is not None:
                z = torus_step(z, omega_z)
            u_x, u_c = self.vf.ambient_u(
                x, c, y, z, self.pe, is_clue, self.cell_house_idx
            )
            x = geodesic_step(x, u_x, x0, is_clue)
            c = geodesic_step(c, u_c, c0, is_clue)
            states.append(x)
        self._last_omegas = omegas
        self._last_omega_zs = omega_zs
        self._last_is_clue = is_clue
        self._last_c = c
        return x, c, omegas, states, omega_zs

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
        x, c, omegas, states, omega_zs = self.integrate(puzzle)
        logits = self._logits(x)
        pred = logits.argmax(dim=-1) + 1

        omega1 = omegas[0]
        omega_k = omegas[-1]
        # Rest-axis displacement on S^{54} (not ‖c‖² which is constant 1/55).
        scratch_energy = (1.0 - c[..., COORD_REST]).mean()
        blank_np = x[..., BLANK_COORD].square().mean()
        q = F.relu(x @ self.poles.T)
        claim_energy = q.square().mean()

        out: dict[str, Tensor] = {
            's': torch.cat([x, c], dim=-1),  # (B,81,64) compat
            'x': x,
            'c': c,
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
        }
        if self.use_global_torus and omega_zs[0] is not None and omega_zs[-1] is not None:
            out['omega_z_sq_1'] = omega_zs[0].square().mean().detach()
            out['omega_z_sq_k'] = omega_zs[-1].square().mean().detach()

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
        loss = task_loss + self.omega_weight * omega_kin
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
