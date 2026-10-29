"""Bipartite S^{63} cell + T^{32} house embeddings (ETF poles, PE, static gathers)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

D = 64
M = 32
N_DIGITS = 9
BLANK_COORD = 8  # North-pole blank index in coords 0..8
SCRATCH_START = 9
PE_ALPHA = 0.25
PE_DIM = 6  # Clifford T^3 (row/col/box circles)


def simplex_etf_poles(n: int = N_DIGITS, d: int = D) -> Tensor:
    """
    ``n`` digit poles as a simplex ETF, living in coords ``0..n-1`` of ``R^d``.

    Rows of ``(I - 11ᵀ/n)`` are centered then L2-normalized; scratch dims are 0.
    Shape ``(n, d)``.
    """
    eye = torch.eye(n)
    centered = eye - torch.ones(n, n) / float(n)
    poles = F.normalize(centered, dim=-1, eps=1e-6)
    p = torch.zeros(n, d)
    p[:, :n] = poles
    return p


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
    Lift puzzle digits onto ``S^{D-1}`` and zero-init house angles on ``T^M``.

    Clues ← ETF pole for digit; blanks ← North Pole + ``α·PE`` into scratch, then
    normalize. Returns ``(x, y, is_clue)`` with shapes ``(B,81,D)``, ``(B,27,M)``,
    ``(B,81)`` bool.
    """
    b = puzzle.shape[0]
    device = puzzle.device
    dtype = poles.dtype
    d = poles.shape[-1]
    x = north_pole(d).to(device=device, dtype=dtype).expand(b, 81, d).contiguous().clone()
    is_clue = puzzle > 0
    digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, N_DIGITS - 1)
    pole_x = poles[digit_idx]  # (B, 81, D)
    x = torch.where(is_clue.unsqueeze(-1), pole_x, x)

    # PE into scratch for blanks only, then renorm all (clues already unit poles).
    pe_b = pe.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
    scratch = x[..., SCRATCH_START:].clone()
    n_pe = min(pe_b.shape[-1], scratch.shape[-1])
    scratch_pe = scratch.clone()
    scratch_pe[..., :n_pe] = scratch[..., :n_pe] + pe_alpha * pe_b[..., :n_pe]
    scratch = torch.where(is_clue.unsqueeze(-1), scratch, scratch_pe)
    x = torch.cat([x[..., :SCRATCH_START], scratch], dim=-1)
    x = F.normalize(x, dim=-1, eps=1e-6)

    y = torch.zeros(b, 27, M, device=device, dtype=dtype)
    return x, y, is_clue
