"""Cross-coupled mixer: norms, orthant, clue freeze, tangent, gauge, backward."""

from __future__ import annotations

import math

import torch

from riemannian_sudoku.geometry import project_tangent
from riemannian_sudoku.model import (
    RiemannianSudokuSolver,
    init_sphere_state,
    init_t2_gauge,
)


def _cc_model(**kw) -> RiemannianSudokuSolver:
    defaults = dict(num_steps=3, dt=0.3, mixer='cross_coupled')
    defaults.update(kw)
    return RiemannianSudokuSolver(**defaults)


def test_init_t2_gauge_unit_and_angles() -> None:
    g = init_t2_gauge(2)
    assert g.shape == (2, 18, 2)
    assert torch.allclose(g.norm(dim=-1), torch.ones(2, 18), atol=1e-5)
    # Row 0 / col 0 at θ=0 → (1, 0); idx=3 → θ=2π/3.
    assert torch.allclose(g[0, 0], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(g[0, 9], torch.tensor([1.0, 0.0]), atol=1e-5)
    th = 2.0 * math.pi * 3 / 9
    assert torch.allclose(g[0, 3], torch.tensor([math.cos(th), math.sin(th)]), atol=1e-5)


def test_cross_coupled_sphere_and_gauge_norms() -> None:
    torch.manual_seed(0)
    model = _cc_model()
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 4
    out = model(puzzle)
    s, g = out['s'], out['g']
    assert (s >= 0).all()
    assert torch.allclose(s.norm(dim=-1), torch.ones(2, 81), atol=1e-4)
    assert torch.allclose(g.norm(dim=-1), torch.ones(2, 18), atol=1e-4)


def test_cross_coupled_clue_freeze() -> None:
    torch.manual_seed(1)
    model = _cc_model(num_steps=4, dt=0.5)
    puzzle = torch.randint(0, 10, (2, 81))
    puzzle[:, 0] = 3
    puzzle[:, 1] = 0
    s, _, _, g = model._integrate_cross_coupled(puzzle)
    expected = torch.zeros(9)
    expected[2] = 1.0
    assert torch.allclose(s[:, 0], expected.expand_as(s[:, 0]), atol=1e-5)
    assert (s >= 0).all()
    assert torch.allclose(g.norm(dim=-1), torch.ones_like(g.norm(dim=-1)), atol=1e-4)


def test_cross_coupled_tangent_orthogonality() -> None:
    torch.manual_seed(2)
    model = _cc_model(num_steps=1)
    puzzle = torch.randint(0, 10, (3, 81))
    puzzle[:, 0] = 1  # clue
    s0, blank = init_sphere_state(puzzle)
    g0 = init_t2_gauge(3)
    v_s, v_g = model._cross_velocities(s0, g0)
    v_s = project_tangent(s0, v_s)
    v_g = project_tangent(g0, v_g)
    # Own-tangent: ⟨v, state⟩ ≈ 0 before clue mask.
    assert (v_s * s0).sum(dim=-1).abs().max() < 1e-5
    assert (v_g * g0).sum(dim=-1).abs().max() < 1e-5
    v_s_masked = v_s * blank.unsqueeze(-1)
    assert torch.allclose(v_s_masked[:, 0], torch.zeros(3, 9), atol=1e-6)


def test_cross_coupled_forward_backward() -> None:
    torch.manual_seed(0)
    model = _cc_model(num_steps=2, kinetic_weight=0.05, constraint_weight=0.0, supervise_all_steps=True)
    puzzle = torch.randint(0, 10, (4, 81))
    solution = torch.randint(1, 10, (4, 81))
    mask = (puzzle == 0).float()
    out = model(puzzle, solution, mask)
    assert torch.isfinite(out['loss'])
    assert 'g' in out
    out['loss'].backward()
    grads = [p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None]
    assert sum(grads) > 0.0
    # Gauge should have moved from init (untrained net, almost surely).
    g0 = init_t2_gauge(4)
    assert not torch.allclose(out['g'], g0, atol=1e-4)


def test_euclid_axial_still_default() -> None:
    model = RiemannianSudokuSolver(d_model=32, n_heads=4)
    assert model.mixer == 'euclid_axial'
    assert model.cross_mlp is None
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Tiny d_model; just ensure axial path builds.
    assert n > 0


def test_unknown_mixer_raises() -> None:
    try:
        RiemannianSudokuSolver(mixer='nope')
    except ValueError as e:
        assert 'nope' in str(e)
    else:
        raise AssertionError('expected ValueError')
