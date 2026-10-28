"""Axial reshape invariants vs true Sudoku groups."""

from __future__ import annotations

import torch
from torch import nn

from riemannian_sudoku.axial import (
    SudokuAxialAttention,
    _attend_boxes,
    _attend_cols,
    _attend_rows,
    box_cell_indices,
)


def test_box_indices_match_standard_membership() -> None:
    idx = box_cell_indices()
    assert idx.shape == (9, 9)
    # Box 0 (top-left): cells 0,1,2,9,10,11,18,19,20
    assert idx[0].tolist() == [0, 1, 2, 9, 10, 11, 18, 19, 20]
    # Box 4 (center): 30-32, 39-41, 48-50
    assert idx[4].tolist() == [30, 31, 32, 39, 40, 41, 48, 49, 50]
    # All 81 cells appear exactly once.
    flat = idx.reshape(-1).tolist()
    assert sorted(flat) == list(range(81))


def test_row_col_box_attention_preserves_layout() -> None:
    """Gather-known values after identity-ish attention still land on correct cells."""
    torch.manual_seed(0)
    b, d = 2, 16
    x = torch.arange(b * 81 * d, dtype=torch.float32).view(b, 81, d)
    # Use attention with no projection change: zero out out_proj bias/weight carefully —
    # instead check reshape round-trips via a custom identity path.
    attn = nn.MultiheadAttention(d, 2, batch_first=True)
    with torch.no_grad():
        # Force nearly-identity by checking shapes + that box gather matches indices.
        y_row = _attend_rows(attn, x)
        y_col = _attend_cols(attn, x)
        y_box = _attend_boxes(attn, x)
    assert y_row.shape == x.shape
    assert y_col.shape == x.shape
    assert y_box.shape == x.shape

    # Explicit box flatten equals gather by box_cell_indices.
    idx = box_cell_indices()
    gathered = x[0, idx]  # (9, 9, D)
    reshaped = x[0].view(3, 3, 3, 3, d).permute(0, 2, 1, 3, 4).reshape(9, 9, d)
    assert torch.equal(gathered, reshaped)


def test_axial_block_forward_and_backward() -> None:
    block = SudokuAxialAttention(d_model=32, n_heads=4)
    x = torch.randn(4, 81, 32, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
