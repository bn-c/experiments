"""Riemannian Sudoku solver: Euclidean axial, cross-coupled, or geometric attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from riemannian_sudoku.axial import SudokuAxialAttention
from riemannian_sudoku.geometry import (
    project_tangent,
    project_to_orthant_sphere,
    riemannian_exp_map,
    riemannian_exp_map_orthant,
    riemannian_log_map,
    sudoku_group_indices,
    unit_sum_penalty,
)

# Ambient dims: flatten(s) + flatten(g) = 81*9 + 18*2.
_CROSS_IN = 81 * 9 + 18 * 2  # 765
_CROSS_S = 81 * 9  # 729
_CROSS_G = 18 * 2  # 36
_CROSS_H = 512
_MLP_ACTIVATIONS = ('gelu', 'silu', 'swiglu')
_MIXERS = ('euclid_axial', 'cross_coupled', 'geo_karcher', 'geo_log')
_GEO_MIXERS = ('geo_karcher', 'geo_log')
_GEO_KARCHER_STEPS = 2
_SOFTPLUS_INV_1 = math.log(math.e - 1.0)  # softplus⁻¹(1)


class _SwiGLU(nn.Module):
    """``W3(silu(W_gate(x)) * W_up(x))``; hidden ≈ ``dim/3`` so Linear+this+Linear ≈ 3-layer GELU width."""

    def __init__(self, dim: int, hidden: int | None = None) -> None:
        super().__init__()
        # Match Linear(dim,dim) params: dim²+dim ≈ 3·dim·h + 2h + dim → h ≈ dim²/(3dim+2).
        h = hidden if hidden is not None else max(1, (dim * dim) // (3 * dim + 2))
        self.w12 = nn.Linear(dim, 2 * h)
        self.w3 = nn.Linear(h, dim)

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


def _cross_mlp(in_dim: int, out_dim: int, hidden: int, activation: str) -> nn.Module:
    """Build cross_coupled ambient MLP; ``activation`` ∈ gelu|silu|swiglu."""
    if activation not in _MLP_ACTIVATIONS:
        raise ValueError(f'unknown mlp_activation {activation!r}; expected gelu|silu|swiglu')
    if activation == 'swiglu':
        # Linear → SwiGLU (fair mid width) → Linear; silu gate lives inside SwiGLU.
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            _SwiGLU(hidden),
            nn.Linear(hidden, out_dim),
        )
    act: nn.Module = nn.GELU() if activation == 'gelu' else nn.SiLU()
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        act,
        nn.Linear(hidden, hidden),
        type(act)(),  # fresh instance (parameter-free; keeps Sequential tidy)
        nn.Linear(hidden, out_dim),
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


def _softplus_inv_init(target: float) -> float:
    """Scalar ``x`` such that ``softplus(x) ≈ target`` (target > 0)."""
    # softplus(x) = log(1+e^x) → x = log(e^t - 1)
    return math.log(math.exp(target) - 1.0)


def _unit_neighbor_mask() -> Tensor:
    """``(81, 81)`` bool: cells share a row, column, or box (includes self)."""
    rows = torch.arange(9).unsqueeze(1).expand(9, 9).reshape(81)
    cols = torch.arange(9).unsqueeze(0).expand(9, 9).reshape(81)
    boxes = (rows // 3) * 3 + (cols // 3)
    return (
        (rows[:, None] == rows[None, :])
        | (cols[:, None] == cols[None, :])
        | (boxes[:, None] == boxes[None, :])
    )


class GeometricCrossAttention(nn.Module):
    """
    Cells-only geodesic affinities + 2-step weighted Karcher mean on ``S⁸₊``.

    Affinities ``α_ij ∝ exp(-d²/(2σ²))`` with learnable ``σ = softplus(·)``,
    masked to row∪col∪box neighbors so position+clues break blank symmetry.
    Clues stay visible as keys; callers freeze clue *queries* via ``blank_mask``.
    """

    def __init__(self, karcher_steps: int = _GEO_KARCHER_STEPS, init_sigma: float = 1.0) -> None:
        super().__init__()
        self.karcher_steps = int(karcher_steps)
        self.raw_sigma = nn.Parameter(torch.tensor(_softplus_inv_init(init_sigma)))
        # Inner Karcher GD step size η = softplus(·), init 1.
        self.raw_eta = nn.Parameter(torch.tensor(_SOFTPLUS_INV_1))
        self.register_buffer('neighbor_mask', _unit_neighbor_mask(), persistent=True)

    @property
    def sigma(self) -> Tensor:
        return F.softplus(self.raw_sigma).clamp_min(1e-4)

    @property
    def eta(self) -> Tensor:
        return F.softplus(self.raw_eta).clamp_min(1e-4)

    def affinities(self, s: Tensor, eps: float = 1e-6) -> Tensor:
        """Geodesic RBF affinities ``(B, 81, 81)`` over unit neighbors; rows sum to 1."""
        cos = torch.einsum('bid,bjd->bij', s, s).clamp(-1.0 + eps, 1.0 - eps)
        d2 = cos.acos().square()
        logits = -d2 / (2.0 * self.sigma.square())
        logits = logits.masked_fill(~self.neighbor_mask.unsqueeze(0), float('-inf'))
        return torch.softmax(logits, dim=-1)

    def forward(self, s: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(m, alpha)`` — Karcher means and affinities for all cells."""
        alpha = self.affinities(s)
        m = s
        eta = self.eta
        eps = 1e-8
        for _ in range(self.karcher_steps):
            cos = torch.einsum('bqd,bkd->bqk', m, s).clamp(-1.0 + eps, 1.0 - eps)
            theta = cos.acos()
            u = s.unsqueeze(1) - cos.unsqueeze(-1) * m.unsqueeze(2)
            sin_t = theta.sin().clamp_min(eps)
            scale = torch.where(theta < 1e-4, torch.ones_like(theta), theta / sin_t)
            log = u * scale.unsqueeze(-1)
            v = (alpha.unsqueeze(-1) * log).sum(dim=2) * eta
            v = project_tangent(m, v)
            m = project_to_orthant_sphere(riemannian_exp_map(m, v, dt=1.0))
        return m, alpha


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

    ``use_gauge=False`` (``--no-gauge``): cells-only ablation — flatten(s) → MLP →
    cell tangent only (729→MLP→729); no live T² state / dynamics / λ_g.

    ``geo_karcher`` / ``geo_log``: cells-only geodesic affinities + 2-step Karcher;
    (1) state ← mean, or (2) ``v = Log_x(m)`` then Exp + orthant + kinetic.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        num_steps: int = 8,
        dt: float = 0.5,
        kinetic_weight: float = 0.05,
        gauge_kinetic_weight: float = 0.0,
        constraint_weight: float = 0.1,
        supervise_all_steps: bool = False,
        mixer: str = 'euclid_axial',
        use_gauge: bool = True,
        mlp_activation: str = 'gelu',
    ) -> None:
        super().__init__()
        if mixer not in _MIXERS:
            raise ValueError(f'unknown mixer {mixer!r}; expected {"|".join(_MIXERS)}')
        if mlp_activation not in _MLP_ACTIVATIONS:
            raise ValueError(f'unknown mlp_activation {mlp_activation!r}; expected gelu|silu|swiglu')
        self.mixer = mixer
        # Gauge only exists on the cross_coupled path; axial / geo ignore the flag.
        self.use_gauge = bool(use_gauge) and mixer == 'cross_coupled'
        # Axial / geo stay free of MLP activation; flag only selects cross ambient MLP.
        self.mlp_activation = mlp_activation if mixer == 'cross_coupled' else 'gelu'
        self.d_model = d_model
        self.num_steps = num_steps
        self.dt = dt
        self.kinetic_weight = kinetic_weight
        self.gauge_kinetic_weight = gauge_kinetic_weight
        self.constraint_weight = constraint_weight
        self.supervise_all_steps = supervise_all_steps

        self.register_buffer('group_indices', sudoku_group_indices(), persistent=True)

        self.proj = None
        self.axial = None
        self.velocity = None
        self.cross_mlp = None
        self.geo_attn = None

        if mixer == 'euclid_axial':
            self.proj = nn.Linear(9 + 6, d_model)
            self.axial = SudokuAxialAttention(d_model, n_heads)
            self.velocity = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 9),
            )
            self.register_buffer('t3_pe', _torus_t3_pe(), persistent=True)
        elif mixer == 'cross_coupled':
            # With gauge: 765→512→…→765; --no-gauge: 729→512→…→729 (cells only).
            # gelu/silu: 512→512 mid Linear; swiglu: mid SwiGLU with h≈170 (fair params).
            in_dim = _CROSS_IN if self.use_gauge else _CROSS_S
            out_dim = (_CROSS_S + _CROSS_G) if self.use_gauge else _CROSS_S
            self.cross_mlp = _cross_mlp(in_dim, out_dim, _CROSS_H, self.mlp_activation)
        else:
            # geo_karcher / geo_log: cells-only geometric attention (no gauge, no MLP v).
            self.geo_attn = GeometricCrossAttention(karcher_steps=_GEO_KARCHER_STEPS)

    def _velocity_at(self, s: Tensor) -> Tensor:
        assert self.proj is not None and self.axial is not None and self.velocity is not None
        pe = self.t3_pe.unsqueeze(0).expand(s.shape[0], -1, -1)
        h = self.proj(torch.cat([s, pe], dim=-1))
        h = self.axial(h)
        return self.velocity(h)

    def _cross_velocities(self, s: Tensor, g: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Shared MLP → ambient (v_s[, v_g]); caller projects to each tangent."""
        assert self.cross_mlp is not None
        b = s.shape[0]
        if not self.use_gauge:
            v_s = self.cross_mlp(s.reshape(b, _CROSS_S)).reshape(b, 81, 9)
            return v_s, None
        assert g is not None
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
        if self.mixer in _GEO_MIXERS:
            return self._integrate_geo(puzzle)
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
    ) -> tuple[Tensor, list[Tensor], list[Tensor], Tensor | None]:
        """Parallel Exp on cell spheres; live T² gauge when ``use_gauge`` else cells-only."""
        s, blank_mask = init_sphere_state(puzzle)
        g: Tensor | None = (
            init_t2_gauge(s.shape[0], device=s.device, dtype=s.dtype) if self.use_gauge else None
        )
        velocities: list[Tensor] = []
        gauge_velocities: list[Tensor] = []
        states: list[Tensor] = []
        for _ in range(self.num_steps):
            v_s, v_g = self._cross_velocities(s, g)
            v_s = project_tangent(s, v_s)
            v_s = v_s * blank_mask.unsqueeze(-1)  # gauge always free when present
            velocities.append(v_s)
            s = riemannian_exp_map_orthant(s, v_s, dt=self.dt)
            if g is not None and v_g is not None:
                v_g = project_tangent(g, v_g)
                gauge_velocities.append(v_g)
                g = riemannian_exp_map(g, v_g, dt=self.dt)  # S¹ Exp; no orthant
            clues = blank_mask == 0
            digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, 8)
            s_clue = torch.zeros_like(s)
            s_clue.scatter_(-1, digit_idx.unsqueeze(-1), 1.0)
            s = torch.where(clues.unsqueeze(-1), s_clue, s)
            states.append(s)
        # Stash for diagnostics + optional gauge kinetic (λ_g via forward).
        self._last_gauge = g
        self._last_gauge_velocities = gauge_velocities if self.use_gauge else None
        return s, velocities, states, g

    def _integrate_geo(self, puzzle: Tensor) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Cells-only geometric cross-attention ODE.

        ``geo_karcher``: blank cells ← Karcher mean (orthant); clues frozen.
        ``geo_log``: ``v = Log_s(m)``, clue-mask, Exp + orthant (existing kinetic).
        """
        assert self.geo_attn is not None
        s, blank_mask = init_sphere_state(puzzle)
        velocities: list[Tensor] = []
        states: list[Tensor] = []
        for _ in range(self.num_steps):
            m, _alpha = self.geo_attn(s)
            if self.mixer == 'geo_karcher':
                # State ← mean on blanks; Log displacement for kinetic logging only.
                m = project_to_orthant_sphere(m)
                v = riemannian_log_map(s, m)
                v = v * blank_mask.unsqueeze(-1)
                velocities.append(v)
                s = torch.where(blank_mask.unsqueeze(-1) > 0, m, s)
            else:
                # geo_log: free tangent v = Log → Exp + orthant (dt as usual).
                v = riemannian_log_map(s, m)
                v = project_tangent(s, v)
                v = v * blank_mask.unsqueeze(-1)
                velocities.append(v)
                s = riemannian_exp_map_orthant(s, v, dt=self.dt)
            clues = blank_mask == 0
            digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, 8)
            s_clue = torch.zeros_like(s)
            s_clue.scatter_(-1, digit_idx.unsqueeze(-1), 1.0)
            s = torch.where(clues.unsqueeze(-1), s_clue, s)
            states.append(s)
        self._last_gauge = None
        self._last_gauge_velocities = None
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
        if self.mixer == 'cross_coupled':
            s, velocities, states, g = self._integrate_cross_coupled(puzzle)
            gauge_velocities = self._last_gauge_velocities
        elif self.mixer in _GEO_MIXERS:
            s, velocities, states = self._integrate_geo(puzzle)
            g = None
            gauge_velocities = None
        else:
            s, velocities, states = self._integrate_euclid_axial(puzzle)
            g = None
            gauge_velocities = None

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

        # Cell kinetic (always); gauge kinetic optional (λ_g=0 default — axial unchanged).
        kinetic = velocities[-1].square().mean()
        if gauge_velocities is not None:
            gauge_kinetic = gauge_velocities[-1].square().mean()
        else:
            gauge_kinetic = kinetic.new_zeros(())
        constraint = unit_sum_penalty(probs, self.group_indices)
        loss = (
            task_loss
            + self.kinetic_weight * kinetic
            + self.gauge_kinetic_weight * gauge_kinetic
            + self.constraint_weight * constraint
        )

        out.update(
            {
                'loss': loss,
                'task_loss': task_loss.detach(),
                'kinetic_loss': kinetic.detach(),
                'gauge_kinetic_loss': gauge_kinetic.detach(),
                'constraint_loss': constraint.detach(),
            }
        )
        return out
