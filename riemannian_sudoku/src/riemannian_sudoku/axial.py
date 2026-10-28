"""Factorized row / column / box attention for 81-cell Sudoku grids."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _attend_rows(attn: nn.MultiheadAttention, x: Tensor) -> Tensor:
    """Self-attend within each of 9 rows. ``x`` is ``(B, 81, D)``."""
    b, _, d = x.shape
    h = x.view(b, 9, 9, d).reshape(b * 9, 9, d)
    out, _ = attn(h, h, h, need_weights=False)
    return out.view(b, 9, 9, d).reshape(b, 81, d)


def _attend_cols(attn: nn.MultiheadAttention, x: Tensor) -> Tensor:
    """Self-attend within each of 9 columns."""
    b, _, d = x.shape
    # (B, row, col, D) → (B, col, row, D) so seq dim is the 9 rows of one column.
    h = x.view(b, 9, 9, d).transpose(1, 2).reshape(b * 9, 9, d)
    out, _ = attn(h, h, h, need_weights=False)
    return out.view(b, 9, 9, d).transpose(1, 2).reshape(b, 81, d)


def _attend_boxes(attn: nn.MultiheadAttention, x: Tensor) -> Tensor:
    """Self-attend within each of 9 boxes (3×3 cells)."""
    b, _, d = x.shape
    # (B, br, dr, bc, dc, D) → (B, br, bc, dr, dc, D) → (B*9, 9, D)
    h = x.view(b, 3, 3, 3, 3, d).permute(0, 1, 3, 2, 4, 5).reshape(b * 9, 9, d)
    out, _ = attn(h, h, h, need_weights=False)
    return out.view(b, 3, 3, 3, 3, d).permute(0, 1, 3, 2, 4, 5).reshape(b, 81, d)


def box_cell_indices() -> Tensor:
    """``(9, 9)`` flat cell indices for each box — for reshape invariant tests."""
    boxes = [
        [(br * 3 + dr) * 9 + bc * 3 + dc for dr in range(3) for dc in range(3)]
        for br in range(3)
        for bc in range(3)
    ]
    return torch.tensor(boxes, dtype=torch.long)


class SudokuAxialAttention(nn.Module):
    """
    One weight-tied axial block: row MHA → col MHA → box MHA → MLP.

    Pre-norm LayerNorm before each residual branch (4 norms total).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4) -> None:
        super().__init__()
        self.ln_row = nn.LayerNorm(d_model)
        self.ln_col = nn.LayerNorm(d_model)
        self.ln_box = nn.LayerNorm(d_model)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.row_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.box_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """``x``: ``(B, 81, D)`` → same shape."""
        x = x + _attend_rows(self.row_attn, self.ln_row(x))
        x = x + _attend_cols(self.col_attn, self.ln_col(x))
        x = x + _attend_boxes(self.box_attn, self.ln_box(x))
        return x + self.mlp(self.ln_mlp(x))
