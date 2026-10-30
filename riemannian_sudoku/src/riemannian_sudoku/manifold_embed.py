"""Bipartite product cell ``S^8 × S^{54}`` + house ``T^{16}`` embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

DECISION_DIM = 9  # S^8 ⊂ R^9
COORD_DIM = 55  # S^{54} ⊂ R^{55}
D = DECISION_DIM + COORD_DIM  # 64 ambient (compat / concat)
M = 16
N_DIGITS = 9
BLANK_COORD = 8  # North-pole blank index on S^8
COORD_REST = 6  # rest axis on S^{54} (e_6)
SCRATCH_START = DECISION_DIM  # legacy alias for coord slice start
PE_ALPHA = 0.25
PE_DIM = 6  # Clifford T^3 (row/col/box circles)


def simplex_etf_poles(n: int = N_DIGITS, d: int = DECISION_DIM) -> Tensor:
    """
    ``n`` digit poles as a simplex ETF in ``R^d`` (default ``d=9`` on ``S^8``),
    Householder-rotated so ``poles[:, blank]==0`` (orthogonal to North-pole ``e_8``).

    Shape ``(n, d)`` with ``‖p_d‖₂=1``, ``⟨p_d,p_{d'}⟩=-1/8``, ``p_{d,8}=0``.
    """
    eye = torch.eye(n)
    centered = eye - torch.ones(n, n) / float(n)
    poles9 = F.normalize(centered, dim=-1, eps=1e-6)

    n_hat = torch.ones(n) / math.sqrt(float(n))
    e_blank = torch.zeros(n)
    e_blank[BLANK_COORD] = 1.0
    v = n_hat - e_blank
    H = torch.eye(n) - 2.0 * torch.outer(v, v) / (v @ v)
    poles9 = (H @ poles9.T).T

    if d == n:
        p = poles9
    else:
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


def house_pair_triu_idx() -> tuple[Tensor, Tensor]:
    """Upper-triangle off-diagonal indices for 9-cell house Gram (36 pairs)."""
    return torch.triu_indices(N_DIGITS, N_DIGITS, offset=1)


def north_pole(d: int = DECISION_DIM, blank_coord: int = BLANK_COORD) -> Tensor:
    """Unit North-pole blank vector ``e_{blank}`` in ``R^d`` (default on ``S^8``)."""
    x = torch.zeros(d)
    x[blank_coord] = 1.0
    return x


def coord_rest(d: int = COORD_DIM, rest: int = COORD_REST) -> Tensor:
    """Unit rest-axis vector ``e_6`` on the coordination sphere."""
    c = torch.zeros(d)
    c[rest] = 1.0
    return c


def init_bipartite_state(
    puzzle: Tensor,
    poles: Tensor,
    pe: Tensor,
    *,
    pe_alpha: float = PE_ALPHA,
    use_global_torus: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
    """
    Lift puzzle onto product ``S^8 × S^{54}`` and init house angles at ``π/4``.

    Returns ``(x, c, y, z, is_clue)`` with shapes ``(B,81,9)``, ``(B,81,55)``,
    ``(B,27,M)``, ``(B,1,M)|None``, ``(B,81)`` bool.
    """
    b = puzzle.shape[0]
    device = puzzle.device
    dtype = poles.dtype

    x = north_pole(DECISION_DIM).to(device=device, dtype=dtype).expand(b, 81, DECISION_DIM).contiguous().clone()
    is_clue = puzzle > 0
    digit_idx = (puzzle.clamp(min=1) - 1).clamp(0, N_DIGITS - 1)
    pole_x = poles[digit_idx]  # (B, 81, 9) when poles are (9, 9)
    if pole_x.shape[-1] > DECISION_DIM:
        pole_x = pole_x[..., :DECISION_DIM]
    x = torch.where(is_clue.unsqueeze(-1), pole_x, x)

    # Coordination: e_6 + α·PE on coords 0..5 for all 81 cells, then L2-normalize.
    pe_b = pe.to(device=device, dtype=dtype).unsqueeze(0).expand(b, -1, -1)
    c = coord_rest(COORD_DIM).to(device=device, dtype=dtype).expand(b, 81, COORD_DIM).contiguous().clone()
    n_pe = min(pe_b.shape[-1], COORD_REST)
    c = c.clone()
    c[..., :n_pe] = c[..., :n_pe] + pe_alpha * pe_b[..., :n_pe]
    c = F.normalize(c, dim=-1, eps=1e-6)

    y = torch.full((b, 27, M), math.pi / 4, device=device, dtype=dtype)
    z: Tensor | None = None
    if use_global_torus:
        z = torch.full((b, 1, M), math.pi / 4, device=device, dtype=dtype)
    return x, c, y, z, is_clue
