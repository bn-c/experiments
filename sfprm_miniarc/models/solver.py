"""Patience-based per-sample η schedule (arXiv:2610.25646 Algorithm 1 / Table 3).

η starts at 1, decays by γ when residual stalls for P iterations, floors at η_min.
Halt when residual < τ, η < η_min, or iteration budget exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.sfprm_core import SFPRM


@dataclass
class SolverConfig:
    patience: int = 5
    gamma: float = 0.9
    eta_min: float = 0.001
    tau: float = 0.01
    max_iter: int = 8  # train default; eval uses 200


@dataclass
class SolverState:
    """Per-sample solver bookkeeping (detached from autograd)."""

    eta: torch.Tensor  # [B]
    patience_left: torch.Tensor  # [B] int
    best_residual: torch.Tensor  # [B]
    steps_used: torch.Tensor  # [B] int
    halted: torch.Tensor  # [B] bool

    @classmethod
    def init(cls, batch_size: int, device: torch.device, cfg: SolverConfig) -> SolverState:
        return cls(
            eta=torch.ones(batch_size, device=device),
            patience_left=torch.full((batch_size,), cfg.patience, device=device, dtype=torch.long),
            best_residual=torch.full((batch_size,), float("inf"), device=device),
            steps_used=torch.zeros(batch_size, device=device, dtype=torch.long),
            halted=torch.zeros(batch_size, device=device, dtype=torch.bool),
        )

    def clone_detached(self) -> SolverState:
        return SolverState(
            eta=self.eta.detach().clone(),
            patience_left=self.patience_left.detach().clone(),
            best_residual=self.best_residual.detach().clone(),
            steps_used=self.steps_used.detach().clone(),
            halted=self.halted.detach().clone(),
        )


@torch.no_grad()
def update_schedule(
    state: SolverState,
    residual: torch.Tensor,
    cfg: SolverConfig,
) -> SolverState:
    """
    Update η / patience from a *detached* residual (does not enter the gradient).

    Improves when residual < best_residual; otherwise burn patience and decay η.
    """
    residual = residual.detach()
    improved = residual < state.best_residual
    state.best_residual = torch.where(improved, residual, state.best_residual)
    state.patience_left = torch.where(
        improved,
        torch.full_like(state.patience_left, cfg.patience),
        state.patience_left - 1,
    )
    decay = (~improved) & (state.patience_left <= 0) & (~state.halted)
    state.eta = torch.where(decay, state.eta * cfg.gamma, state.eta)
    state.patience_left = torch.where(decay, torch.full_like(state.patience_left, cfg.patience), state.patience_left)

    state.steps_used = state.steps_used + (~state.halted).long()
    hit_tau = residual < cfg.tau
    hit_floor = state.eta < cfg.eta_min
    hit_cap = state.steps_used >= cfg.max_iter
    state.halted = state.halted | hit_tau | hit_floor | hit_cap
    return state


def solver_step(
    model: SFPRM,
    z: torch.Tensor,
    z_input: torch.Tensor,
    state: SolverState,
    cfg: SolverConfig,
    *,
    variational_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SolverState]:
    """
    One geodesic damped step with schedule update on detached residual.

    Returns (z_next, residual_detached_for_logging, updated_state).
    Halted samples keep their z (η effectively unused).
    """
    z_next, _z_tilde, residual = model.step(
        z,
        z_input,
        state.eta,
        variational_mask=variational_mask,
        pad_mask=pad_mask,
    )
    # Keep halted samples frozen
    z_out = torch.where(state.halted[:, None, None], z, z_next)
    state = update_schedule(state, residual, cfg)
    return z_out, residual.detach(), state


def run_segment(
    model: SFPRM,
    z: torch.Tensor,
    z_input: torch.Tensor,
    state: SolverState,
    cfg: SolverConfig,
    k: int,
    *,
    variational_mask: torch.Tensor | None = None,
    pad_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, SolverState, torch.Tensor]:
    """
    Run up to ``k`` solver steps (BPTT window). Returns (z, state, last_residual).

    Samples a locked variational mask once (if training) when none is passed, and
    reuses it for every iter in the segment.
    """
    if variational_mask is None and model.training:
        variational_mask = model.sample_variational_mask(z.shape[0], z.device, z.dtype)
    residual = state.best_residual
    for _ in range(k):
        if bool(state.halted.all()):
            break
        z, residual, state = solver_step(
            model,
            z,
            z_input,
            state,
            cfg,
            variational_mask=variational_mask,
            pad_mask=pad_mask,
        )
    return z, state, residual
