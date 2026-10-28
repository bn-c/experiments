"""Geometry smoke: Exp map, orthant, unit sphere."""

from __future__ import annotations

import torch

from riemannian_sudoku.geometry import (
    project_tangent,
    project_to_orthant_sphere,
    riemannian_exp_map_orthant,
    sudoku_group_indices,
    unit_sum_penalty,
)


def test_exp_map_orthant_stays_on_positive_sphere() -> None:
    s = project_to_orthant_sphere(torch.randn(4, 81, 9))
    v = project_tangent(s, torch.randn_like(s))
    s2 = riemannian_exp_map_orthant(s, v, dt=0.5)
    norms = s2.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert (s2 >= 0).all()


def test_clue_freeze_zero_velocity_is_identity() -> None:
    s = torch.zeros(2, 81, 9)
    s[..., 0] = 1.0  # e_0 clues everywhere
    v = torch.zeros_like(s)
    s2 = riemannian_exp_map_orthant(s, v, dt=1.0)
    assert torch.allclose(s2, s, atol=1e-6)


def test_unit_sum_penalty_finite() -> None:
    probs = torch.softmax(torch.randn(3, 81, 9), dim=-1)
    loss = unit_sum_penalty(probs, sudoku_group_indices())
    assert torch.isfinite(loss)
    assert loss.ndim == 0
