"""Geometric cross-attention: Log/Exp, affinities, Karcher, clue freeze, backward."""

from __future__ import annotations

import torch

from riemannian_sudoku.geometry import (
    project_tangent,
    project_to_orthant_sphere,
    riemannian_exp_map,
    riemannian_log_map,
)
from riemannian_sudoku.model import GeometricCrossAttention, RiemannianSudokuSolver


def _geo_model(mixer: str = 'geo_log', **kw) -> RiemannianSudokuSolver:
    defaults = dict(num_steps=3, dt=0.5, mixer=mixer, kinetic_weight=0.5, constraint_weight=0.0)
    defaults.update(kw)
    return RiemannianSudokuSolver(**defaults)


def test_log_exp_roundtrip_small() -> None:
    torch.manual_seed(0)
    s = project_to_orthant_sphere(torch.rand(4, 81, 9) + 0.1)
    v = project_tangent(s, torch.randn_like(s) * 0.05)
    y = riemannian_exp_map(s, v, dt=1.0)
    v_back = riemannian_log_map(s, y)
    assert torch.allclose(v_back, v, atol=1e-4)


def test_exp_log_roundtrip_orthant() -> None:
    torch.manual_seed(1)
    s = project_to_orthant_sphere(torch.rand(2, 81, 9) + 0.05)
    y = project_to_orthant_sphere(torch.rand(2, 81, 9) + 0.05)
    # Stay near s so we avoid cut locus / orthant fold issues.
    y = project_to_orthant_sphere(0.9 * s + 0.1 * y)
    v = riemannian_log_map(s, y)
    assert (v * s).sum(dim=-1).abs().max() < 1e-5
    y2 = project_to_orthant_sphere(riemannian_exp_map(s, v, dt=1.0))
    assert torch.allclose(y2, y, atol=2e-4)


def test_affinities_sum_to_one_over_neighbors() -> None:
    torch.manual_seed(2)
    attn = GeometricCrossAttention()
    s = project_to_orthant_sphere(torch.rand(3, 81, 9) + 0.1)
    alpha = attn.affinities(s)
    assert alpha.shape == (3, 81, 81)
    row_sum = alpha.sum(dim=-1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5)
    assert (alpha >= 0).all()
    # Non-neighbors get zero mass (row 0 cell 0 vs cell 80 = different row/col/box).
    assert torch.allclose(alpha[:, 0, 80], torch.zeros(3), atol=1e-6)


def test_geo_log_sphere_norms_and_clue_freeze() -> None:
    torch.manual_seed(3)
    model = _geo_model('geo_log', num_steps=4)
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 7
    puzzle[:, 1] = 0
    out = model(puzzle)
    s = out['s']
    assert (s >= 0).all()
    assert torch.allclose(s.norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    expected = torch.zeros(9)
    expected[6] = 1.0
    assert torch.allclose(s[:, 0], expected.expand_as(s[:, 0]), atol=1e-5)
    assert 'g' not in out


def test_geo_karcher_sphere_norms_and_clue_freeze() -> None:
    torch.manual_seed(4)
    model = _geo_model('geo_karcher', num_steps=4, kinetic_weight=0.0)
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 3
    out = model(puzzle)
    s = out['s']
    assert (s >= 0).all()
    assert torch.allclose(s.norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    expected = torch.zeros(9)
    expected[2] = 1.0
    assert torch.allclose(s[:, 0], expected.expand_as(s[:, 0]), atol=1e-5)


def test_geo_log_forward_backward() -> None:
    torch.manual_seed(5)
    model = _geo_model('geo_log', num_steps=2, supervise_all_steps=True)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert out['gauge_kinetic_loss'].item() == 0.0
    out['loss'].backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(grads) > 0.0
    assert model.geo_attn is not None
    assert model.geo_attn.raw_sigma.grad is not None
    assert model.geo_attn.raw_sigma.grad.abs().sum() > 0


def test_geo_karcher_forward_backward() -> None:
    torch.manual_seed(6)
    model = _geo_model('geo_karcher', num_steps=2, kinetic_weight=0.0, supervise_all_steps=True)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    out['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_geo_mixers_tiny_param_count() -> None:
    """Geo arms are σ+η only — orders of magnitude below cross MLP."""
    k = RiemannianSudokuSolver(mixer='geo_karcher')
    g = RiemannianSudokuSolver(mixer='geo_log')
    n_k = sum(p.numel() for p in k.parameters() if p.requires_grad)
    n_g = sum(p.numel() for p in g.parameters() if p.requires_grad)
    assert n_k == n_g == 2
    assert k.cross_mlp is None and g.cross_mlp is None
    assert k.use_gauge is False and g.use_gauge is False


def test_resolve_mixer_geo() -> None:
    from riemannian_sudoku.train import _resolve_mixer

    assert _resolve_mixer(None, {'mixer': 'geo_log'}) == 'geo_log'
    assert _resolve_mixer('geo_karcher', {'mixer': 'geo_log'}) == 'geo_karcher'
