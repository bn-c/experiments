"""Riemannian Sudoku solver: Euclidean axial or cross-coupled + live T² gauge."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from riemannian_sudoku.axial import SudokuAxialAttention
from riemannian_sudoku.geometry import (
    project_tangent,
    riemannian_exp_map,
    riemannian_exp_map_orthant,
    sudoku_group_indices,
    unit_sum_penalty,
)

# Ambient dims: flatten(s) + flatten(g) = 81*9 + 18*2.
_CROSS_IN = 81 * 9 + 18 * 2  # 765
_CROSS_S = 81 * 9  # 729
_CROSS_G = 18 * 2  # 36
_CROSS_H = 512


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
    # Vectorized clue one-hot (no .item() / data-dependent shapes — Inductor-friendly).
    digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, 8)
    onehot = torch.zeros(b, 81, 9, device=device, dtype=torch.float32)
    onehot.scatter_(-1, digit_idx.unsqueeze(-1), 1.0)
    s = torch.where(clues.unsqueeze(-1), onehot, s)
    return s, blank.float()


def init_t2_gauge(batch: int, device: torch.device | None = None, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Live ``T²`` gauge: 9 row + 9 col unit circles in ``R²``.

    Init from discrete angles ``θ = 2π · idx / 9`` (same grid for rows and cols).
    Shape ``(B, 18, 2)``.
    """
    idx = torch.arange(9, device=device, dtype=dtype)
    theta = 2.0 * math.pi * idx / 9.0
    circle = torch.stack([theta.cos(), theta.sin()], dim=-1)  # (9, 2)
    g0 = torch.cat([circle, circle], dim=0)  # (18, 2)
    return g0.unsqueeze(0).expand(batch, -1, -1).contiguous()


class RiemannianSudokuSolver(nn.Module):
    """
    Weight-tied ODE on S^8_+ with selectable mixer.

    ``euclid_axial`` (default): concat ``[s, T3]`` → Linear → axial → velocity →
    tangent project → clue freeze → Exp + abs-renorm.

    ``cross_coupled``: concat flatten(s)+flatten(g) → MLP → per-manifold tangent
    project → parallel Exp (cell orthant + circle Exp on each S¹ gauge factor).
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
        mixer: str = 'euclid_axial',
    ) -> None:
        super().__init__()
        if mixer not in ('euclid_axial', 'cross_coupled'):
            raise ValueError(f'unknown mixer {mixer!r}; expected euclid_axial|cross_coupled')
        self.mixer = mixer
        self.d_model = d_model
        self.num_steps = num_steps
        self.dt = dt
        self.kinetic_weight = kinetic_weight
        self.constraint_weight = constraint_weight
        self.supervise_all_steps = supervise_all_steps

        self.register_buffer('group_indices', sudoku_group_indices(), persistent=True)

        if mixer == 'euclid_axial':
            self.proj = nn.Linear(9 + 6, d_model)
            self.axial = SudokuAxialAttention(d_model, n_heads)
            self.velocity = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 9),
            )
            self.register_buffer('t3_pe', _torus_t3_pe(), persistent=True)
            self.cross_mlp = None
        else:
            self.proj = None
            self.axial = None
            self.velocity = None
            self.cross_mlp = nn.Sequential(
                nn.Linear(_CROSS_IN, _CROSS_H),
                nn.GELU(),
                nn.Linear(_CROSS_H, _CROSS_H),
                nn.GELU(),
                nn.Linear(_CROSS_H, _CROSS_S + _CROSS_G),
            )

    def _velocity_at(self, s: Tensor) -> Tensor:
        assert self.proj is not None and self.axial is not None and self.velocity is not None
        pe = self.t3_pe.unsqueeze(0).expand(s.shape[0], -1, -1)
        h = self.proj(torch.cat([s, pe], dim=-1))
        h = self.axial(h)
        return self.velocity(h)

    def _cross_velocities(self, s: Tensor, g: Tensor) -> tuple[Tensor, Tensor]:
        """Shared MLP → ambient (v_s, v_g); caller projects to each tangent."""
        assert self.cross_mlp is not None
        b = s.shape[0]
        x = torch.cat([s.reshape(b, _CROSS_S), g.reshape(b, _CROSS_G)], dim=-1)
        out = self.cross_mlp(x)
        v_s = out[:, :_CROSS_S].reshape(b, 81, 9)
        v_g = out[:, _CROSS_S:].reshape(b, 18, 2)
        return v_s, v_g

    def integrate(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Run ``num_steps`` of dynamics.

        Returns final ``s``, per-step tangent velocities (after clue mask), and
        post-step states (for optional multi-step task supervision).
        """
        if self.mixer == 'cross_coupled':
            s, velocities, states, _g = self._integrate_cross_coupled(puzzle)
            return s, velocities, states
        return self._integrate_euclid_axial(puzzle)

    def _integrate_euclid_axial(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        s, blank_mask = init_sphere_state(puzzle)
        velocities: list[Tensor] = []
        states: list[Tensor] = []
        for _ in range(self.num_steps):
            v = self._velocity_at(s)
            v = project_tangent(s, v)
            v = v * blank_mask.unsqueeze(-1)
            velocities.append(v)
            s = riemannian_exp_map_orthant(s, v, dt=self.dt)
            # Re-clamp clues exactly (numerical drift guard; vectorized for compile).
            clues = blank_mask == 0
            digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, 8)
            s_clue = torch.zeros_like(s)
            s_clue.scatter_(-1, digit_idx.unsqueeze(-1), 1.0)
            s = torch.where(clues.unsqueeze(-1), s_clue, s)
            states.append(s)
        return s, velocities, states

    def _integrate_cross_coupled(
        self, puzzle: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor]:
        """Parallel Exp on cell spheres + live T² gauge circles."""
        s, blank_mask = init_sphere_state(puzzle)
        g = init_t2_gauge(s.shape[0], device=s.device, dtype=s.dtype)
        velocities: list[Tensor] = []
        gauge_velocities: list[Tensor] = []
        states: list[Tensor] = []
        for _ in range(self.num_steps):
            v_s, v_g = self._cross_velocities(s, g)
            v_s = project_tangent(s, v_s)
            v_g = project_tangent(g, v_g)
            v_s = v_s * blank_mask.unsqueeze(-1)  # gauge always free
            velocities.append(v_s)
            gauge_velocities.append(v_g)
            s = riemannian_exp_map_orthant(s, v_s, dt=self.dt)
            g = riemannian_exp_map(g, v_g, dt=self.dt)  # S¹ Exp; no orthant
            clues = blank_mask == 0
            digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, 8)
            s_clue = torch.zeros_like(s)
            s_clue.scatter_(-1, digit_idx.unsqueeze(-1), 1.0)
            s = torch.where(clues.unsqueeze(-1), s_clue, s)
            states.append(s)
        # Stash last gauge velocity for tests / diagnostics (not used by loss).
        self._last_gauge = g
        self._last_gauge_velocities = gauge_velocities
        return s, velocities, states, g

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
        if self.mixer == 'cross_coupled':
            s, velocities, states, g = self._integrate_cross_coupled(puzzle)
        else:
            s, velocities, states = self._integrate_euclid_axial(puzzle)
            g = None

        probs = s.square()
        pred = probs.argmax(dim=-1) + 1  # digits 1-9

        out: dict[str, Tensor] = {
            's': s,
            'probs': probs,
            'pred': pred,
            'velocities': velocities[-1],
        }
        if g is not None:
            out['g'] = g
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
