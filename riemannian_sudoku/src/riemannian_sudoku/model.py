"""Riemannian Sudoku solver: T³ PE + axial attention + S⁸₊ Exp dynamics."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from riemannian_sudoku.axial import SudokuAxialAttention
from riemannian_sudoku.geometry import (
    project_tangent,
    riemannian_exp_map_orthant,
    sudoku_group_indices,
    unit_sum_penalty,
)


def _torus_t3_pe() -> Tensor:
    """Fixed Clifford T^3 in R^6: row/col/box circles, shape ``(81, 6)``."""
    rows = torch.arange(9).unsqueeze(1).expand(9, 9).reshape(81)
    cols = torch.arange(9).unsqueeze(0).expand(9, 9).reshape(81)
    boxes = (rows // 3) * 3 + (cols // 3)
    pe = torch.zeros(81, 6)
    for i, idx in enumerate((rows, cols, boxes)):
        theta = 2.0 * math.pi * idx.float() / 9.0
        pe[:, 2 * i] = torch.cos(theta)
        pe[:, 2 * i + 1] = torch.sin(theta)
    return pe


def init_sphere_state(puzzle: Tensor) -> tuple[Tensor, Tensor]:
    """
    Build ``s₀ ∈ S⁸₊`` and blank mask from digit grids.

    Args:
        puzzle: ``(B, 81)`` int digits ``0..9`` (0 = blank).

    Returns:
        ``s`` ``(B, 81, 9)`` on the positive orthant sphere;
        ``blank_mask`` ``(B, 81)`` float — 1 on blanks (velocity free), 0 on clues.
    """
    b = puzzle.shape[0]
    device = puzzle.device
    s = torch.full((b, 81, 9), 1.0 / 3.0, device=device, dtype=torch.float32)
    blank = puzzle == 0
    clues = ~blank
    if clues.any():
        digit_idx = (puzzle[clues] - 1).clamp(0, 8)
        s_clue = torch.zeros(int(clues.sum().item()), 9, device=device, dtype=torch.float32)
        s_clue.scatter_(1, digit_idx.unsqueeze(1), 1.0)
        s = s.clone()
        s[clues] = s_clue
    return s, blank.float()


class RiemannianSudokuSolver(nn.Module):
    """
        Weight-tied axial ODE on S^8_+.

    Each step: concat ``[s, T3]`` → Linear → axial block → velocity head →
    tangent project → clue freeze → Exp + abs-renorm.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_steps: int = 8,
        dt: float = 0.5,
        kinetic_weight: float = 0.05,
        constraint_weight: float = 0.1,
        supervise_all_steps: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps
        self.dt = dt
        self.kinetic_weight = kinetic_weight
        self.constraint_weight = constraint_weight
        self.supervise_all_steps = supervise_all_steps

        self.proj = nn.Linear(9 + 6, d_model)
        self.axial = SudokuAxialAttention(d_model, n_heads)
        self.velocity = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 9),
        )
        self.register_buffer('t3_pe', _torus_t3_pe(), persistent=True)
        self.register_buffer('group_indices', sudoku_group_indices(), persistent=True)

    def _velocity_at(self, s: Tensor) -> Tensor:
        pe = self.t3_pe.unsqueeze(0).expand(s.shape[0], -1, -1)
        h = self.proj(torch.cat([s, pe], dim=-1))
        h = self.axial(h)
        return self.velocity(h)

    def integrate(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Run ``num_steps`` of dynamics.

        Returns final ``s``, per-step tangent velocities (after clue mask), and
        post-step states (for optional multi-step task supervision).
        """
        s, blank_mask = init_sphere_state(puzzle)
        velocities: list[Tensor] = []
        states: list[Tensor] = []
        for _ in range(self.num_steps):
            v = self._velocity_at(s)
            v = project_tangent(s, v)
            v = v * blank_mask.unsqueeze(-1)
            velocities.append(v)
            s = riemannian_exp_map_orthant(s, v, dt=self.dt)
            # Re-clamp clues exactly (numerical drift guard).
            clues = blank_mask == 0
            if clues.any():
                digit_idx = (puzzle[clues] - 1).clamp(0, 8)
                s_clue = torch.zeros(int(clues.sum().item()), 9, device=s.device, dtype=s.dtype)
                s_clue.scatter_(1, digit_idx.unsqueeze(1), 1.0)
                s = s.clone()
                s[clues] = s_clue
            states.append(s)
        return s, velocities, states

    @staticmethod
    def _blank_nll(probs: Tensor, solution: Tensor, mask: Tensor) -> Tensor:
        """Mean blank-cell NLL on ``p = s²`` (clues excluded via mask)."""
        log_p = probs.clamp_min(1e-8).log()
        gather_idx = (solution - 1).clamp(0, 8).unsqueeze(-1)
        nll_cell = -log_p.gather(-1, gather_idx).squeeze(-1)
        blank_denom = mask.sum().clamp_min(1.0)
        return (nll_cell * mask).sum() / blank_denom

    def forward(
        self,
        puzzle: Tensor,
        solution: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Integrate and optionally compute training losses.

        Args:
            puzzle: ``(B, 81)`` digits 0-9.
            solution: ``(B, 81)`` digits 1-9 (required for loss).
            mask: ``(B, 81)`` blank mask; defaults to ``puzzle == 0``.
        """
        s, velocities, states = self.integrate(puzzle)
        probs = s.square()
        pred = probs.argmax(dim=-1) + 1  # digits 1-9

        out: dict[str, Tensor] = {
            's': s,
            'probs': probs,
            'pred': pred,
            'velocities': velocities[-1],
        }
        if solution is None:
            return out

        if mask is None:
            mask = (puzzle == 0).float()

        # Blank-cell NLL: final state only, or mean over ODE steps.
        if self.supervise_all_steps:
            task_loss = torch.stack([self._blank_nll(st.square(), solution, mask) for st in states]).mean()
        else:
            task_loss = self._blank_nll(probs, solution, mask)

        kinetic = velocities[-1].square().mean()
        constraint = unit_sum_penalty(probs, self.group_indices)
        loss = task_loss + self.kinetic_weight * kinetic + self.constraint_weight * constraint

        out.update(
            {
                'loss': loss,
                'task_loss': task_loss.detach(),
                'kinetic_loss': kinetic.detach(),
                'constraint_loss': constraint.detach(),
            }
        )
        return out
