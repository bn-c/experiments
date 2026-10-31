"""Bipartite S^{63} cell + T^{16} house embeddings (Householder ETF, Clifford, PE)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

D = 64
M = 16
N_DIGITS = 9
BLANK_COORD = 8  # North-pole blank index in coords 0..8
SCRATCH_START = 9
PE_ALPHA = 0.25
PE_DIM = 6  # Clifford T^3 (row/col/box circles)


def simplex_etf_poles(n: int = N_DIGITS, d: int = D) -> Tensor:
    """
    ``n`` digit poles as a simplex ETF in ``R^d``, Householder-rotated so
    ``poles[:, blank]==0`` (orthogonal to North-pole ``e_8``).

    Build centered simplex rows, L2-normalize, apply Householder ``H`` mapping
    unit all-ones ``1/√n`` → ``e_{blank}``, then embed into ``R^d``. Shape ``(n, d)``.
    """
    eye = torch.eye(n)
    centered = eye - torch.ones(n, n) / float(n)
    poles9 = F.normalize(centered, dim=-1, eps=1e-6)

    # H: n_hat = 1/√n ↦ e_blank (plan: 1/3 for n=9).
    n_hat = torch.ones(n) / math.sqrt(float(n))
    e_blank = torch.zeros(n)
    e_blank[BLANK_COORD] = 1.0
    v = n_hat - e_blank
    H = torch.eye(n) - 2.0 * torch.outer(v, v) / (v @ v)
    poles9 = (H @ poles9.T).T

    p = torch.zeros(n, d)
    p[:, :n] = poles9
    if p[:, BLANK_COORD].abs().max() >= 1e-6:
        raise RuntimeError('Householder ETF: poles[:, blank] not ~0')
    return p


def clifford_embed(y: Tensor) -> Tensor:
    """
    Clifford torus lift: unwrapped angles ``(..., m)`` → unit embed ``(..., 2m)``.

    ``cat([cos(y), sin(y)], -1) * m^{-1/2}`` with ``m = y.shape[-1]``;
    ``‖y_emb‖₂ ≡ 1``.
    """
    m = y.shape[-1]
    return torch.cat([torch.cos(y), torch.sin(y)], dim=-1) * (m**-0.5)


def hard_incidence_from_indices(
    house_cells: Tensor | None = None,
    cell_houses: Tensor | None = None,
    *,
    n_cells: int = 81,
    n_houses: int = 27,
) -> tuple[Tensor, Tensor]:
    """
    Build hard bipartite incidence ``(A_s2t, A_t2s)`` from index buffers.

    ``A_s2t`` ∈ R^{H×N} (default 27×81): row-stochastic (entries ``1/9``) for
    S→T ``house_features``. ``A_t2s`` ∈ R^{N×K_inc×H} (81×3×27) one-hot for
    T→S concat; houses are blocks ``[0:9, 9:18, 18:27]`` (row/col/box).
    ``A_t2s`` is the block-structured transpose of hard incidence (Phase 2
    multi-head routing slots onto the same ``K_inc`` axis).
    """
    if house_cells is None:
        house_cells = house_cell_idx()
    if cell_houses is None:
        cell_houses = cell_house_idx()
    h, c0 = house_cells.shape
    n, k_inc = cell_houses.shape
    if h != n_houses or n != n_cells:
        raise ValueError(
            f'index shapes ({h},{c0})/({n},{k_inc}) vs n_houses={n_houses} n_cells={n_cells}'
        )
    # S→T: each house row averages its C0 cells.
    a_s2t = torch.zeros(n_houses, n_cells, dtype=torch.float32)
    house_rows = torch.arange(n_houses).unsqueeze(1).expand(-1, c0)
    a_s2t[house_rows, house_cells.long()] = 1.0 / float(c0)
    # T→S: one-hot over houses for each (cell, incidence-slot).
    a_t2s = torch.zeros(n_cells, k_inc, n_houses, dtype=torch.float32)
    cell_rows = torch.arange(n_cells).unsqueeze(1).expand(-1, k_inc)
    slot = torch.arange(k_inc).unsqueeze(0).expand(n_cells, -1)
    a_t2s[cell_rows, slot, cell_houses.long()] = 1.0
    return a_s2t, a_t2s


class SudokuProblemAdapter:
    """
    Sudoku → attractor pack: ``(x0, is_clue, pe, A_s2t, A_t2s, poles, y0)``.

    Identical to ``init_bipartite_state`` + static hard incidence from indices.
    Hard clue clamp only; no learned encoder / soft α.

    Stored ``_poles`` / ``_pe`` / ``_a_*`` are CPU fallbacks for unit tests only.
    Production ``encode`` must pass the solver's **live** registered buffers
    (``poles=self.poles``, …) — plain refs go stale after ``module.to(cuda)``.
    """

    def __init__(
        self,
        poles: Tensor,
        pe: Tensor,
        *,
        pe_alpha: float = PE_ALPHA,
        house_cells: Tensor | None = None,
        cell_houses: Tensor | None = None,
    ) -> None:
        self.pe_alpha = pe_alpha
        # Fallbacks for standalone tests; do not treat as device-tracking buffers.
        self._poles = poles
        self._pe = pe
        a_s2t, a_t2s = hard_incidence_from_indices(house_cells, cell_houses)
        self._a_s2t = a_s2t
        self._a_t2s = a_t2s

    # Compat aliases (tests / older call sites).
    @property
    def poles(self) -> Tensor:
        return self._poles

    @property
    def pe(self) -> Tensor:
        return self._pe

    @property
    def a_s2t(self) -> Tensor:
        return self._a_s2t

    @property
    def a_t2s(self) -> Tensor:
        return self._a_t2s

    def encode(
        self,
        puzzle: Tensor,
        *,
        poles: Tensor | None = None,
        pe: Tensor | None = None,
        a_s2t: Tensor | None = None,
        a_t2s: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return pack on ``puzzle.device`` using live buffers when provided."""
        poles = self._poles if poles is None else poles
        pe = self._pe if pe is None else pe
        a_s2t = self._a_s2t if a_s2t is None else a_s2t
        a_t2s = self._a_t2s if a_t2s is None else a_t2s
        # Batch device wins — never trust a possibly-stale poles.device snapshot.
        device = puzzle.device
        dtype = poles.dtype
        poles = poles.to(device=device, dtype=dtype)
        pe = pe.to(device=device, dtype=dtype)
        a_s2t = a_s2t.to(device=device, dtype=dtype)
        a_t2s = a_t2s.to(device=device, dtype=dtype)
        x0, y0, is_clue = init_bipartite_state(
            puzzle, poles, pe, pe_alpha=self.pe_alpha
        )
        return x0, is_clue, pe, a_s2t, a_t2s, poles, y0


def torus_t3_pe(pe_dim: int = PE_DIM) -> Tensor:
    """Fixed Clifford T³ PE in ``R^{pe_dim}`` (default 6), shape ``(81, pe_dim)``."""
    rows = torch.arange(9).unsqueeze(1).expand(9, 9).reshape(81)
    cols = torch.arange(9).unsqueeze(0).expand(9, 9).reshape(81)
    boxes = (rows // 3) * 3 + (cols // 3)
    pe = torch.zeros(81, pe_dim)
    for i, idx in enumerate((rows, cols, boxes)):
        if 2 * i + 1 >= pe_dim:
            break
        theta = 2.0 * math.pi * idx.float() / 9.0
        pe[:, 2 * i] = torch.cos(theta)
        pe[:, 2 * i + 1] = torch.sin(theta)
    return pe


def house_cell_idx() -> Tensor:
    """Static ``HOUSE_CELL_IDX[27, 9]``: cells in each row/col/box house."""
    groups: list[list[int]] = []
    groups.extend([list(range(r * 9, r * 9 + 9)) for r in range(9)])
    groups.extend([list(range(c, 81, 9)) for c in range(9)])
    groups.extend(
        [
            [(br * 3 + dr) * 9 + bc * 3 + dc for dr in range(3) for dc in range(3)]
            for br in range(3)
            for bc in range(3)
        ]
    )
    return torch.tensor(groups, dtype=torch.long)


def cell_house_idx() -> Tensor:
    """Static ``CELL_HOUSE_IDX[81, 3]``: row/col/box house ids for each cell."""
    rows = torch.arange(9).unsqueeze(1).expand(9, 9).reshape(81)
    cols = torch.arange(9).unsqueeze(0).expand(9, 9).reshape(81)
    boxes = (rows // 3) * 3 + (cols // 3)
    return torch.stack([rows, 9 + cols, 18 + boxes], dim=-1)


def north_pole(d: int = D, blank_coord: int = BLANK_COORD) -> Tensor:
    """Unit North-pole blank vector ``e_{blank}`` in ``R^d``."""
    x = torch.zeros(d)
    x[blank_coord] = 1.0
    return x


def init_bipartite_state(
    puzzle: Tensor,
    poles: Tensor,
    pe: Tensor,
    *,
    pe_alpha: float = PE_ALPHA,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Lift puzzle digits onto ``S^{D-1}`` and init house angles at ``π/4`` on ``T^M``.

    Clues ← ETF pole for digit; blanks ← North Pole; then ``α·PE`` into scratch
    for **every** cell and L2-normalize. Returns ``(x, y, is_clue)`` with shapes
    ``(B,81,D)``, ``(B,27,M)``, ``(B,81)`` bool. ``y`` is a plain tensor (not
    ``nn.Parameter``). After renorm, clues keep ``‖x[:9]‖≈0.970`` with PE mass
    in ``9:15``.

    Device: ``puzzle.device`` is canonical (batch / train loader). ``poles`` and
    ``pe`` are moved onto it so ``is_clue`` and ``pole_x`` cannot diverge under
    ``torch.compile`` + CUDA — callers must pass **live** module buffers, not
    stale CPU snapshots held outside ``nn.Module._buffers``.
    """
    device = puzzle.device
    dtype = poles.dtype
    poles = poles.to(device=device)
    pe = pe.to(device=device, dtype=dtype)
    b = puzzle.shape[0]
    d = poles.shape[-1]
    x = north_pole(d).to(device=device, dtype=dtype).expand(b, 81, d).contiguous().clone()
    is_clue = puzzle > 0
    digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, N_DIGITS - 1)
    pole_x = poles[digit_idx]  # (B, 81, D) — same device as puzzle
    x = torch.where(is_clue.unsqueeze(-1), pole_x, x)

    # PE into scratch for all cells, then renorm (clue :9 stays ≈ ETF after renorm).
    pe_b = pe.unsqueeze(0).expand(b, -1, -1)
    scratch = x[..., SCRATCH_START:].clone()
    n_pe = min(pe_b.shape[-1], scratch.shape[-1])
    scratch[..., :n_pe] = scratch[..., :n_pe] + pe_alpha * pe_b[..., :n_pe]
    x = torch.cat([x[..., :SCRATCH_START], scratch], dim=-1)
    x = F.normalize(x, dim=-1, eps=1e-6)

    # π/4 unlocks Clifford Jacobian at k=1 (cos=sin); not a Parameter.
    y = torch.full((b, 27, M), math.pi / 4, device=device, dtype=dtype)
    return x, y, is_clue
