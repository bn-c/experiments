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
    Clifford torus lift: unwrapped angles ``(..., M)`` → unit embed ``(..., 2M)``.

    ``cat([cos(y), sin(y)], -1) * M^{-1/2}``; ``‖y_emb‖₂ ≡ 1``.
    """
    return torch.cat([torch.cos(y), torch.sin(y)], dim=-1) * (M**-0.5)


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

    # PE into scratch for all cells, then renorm (clue :9 stays ≈ ETF after renorm).
    pe_b = pe.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
    scratch = x[..., SCRATCH_START:].clone()
    n_pe = min(pe_b.shape[-1], scratch.shape[-1])
    scratch[..., :n_pe] = scratch[..., :n_pe] + pe_alpha * pe_b[..., :n_pe]
    x = torch.cat([x[..., :SCRATCH_START], scratch], dim=-1)
    x = F.normalize(x, dim=-1, eps=1e-6)

    # π/4 unlocks Clifford Jacobian at k=1 (cos=sin); not a Parameter.
    y = torch.full((b, 27, M), math.pi / 4, device=device, dtype=dtype)
    return x, y, is_clue
